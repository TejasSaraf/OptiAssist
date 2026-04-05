"""
Modal.com deployment for MedGemma 4B QLoRA fine-tuning.

Fine-tunes google/medgemma-4b-it on retinal fundus data using 4-bit NF4
quantization (QLoRA) on a cloud H100 80 GB GPU.

Dataset: Stratified subset — 500 train (100 per DR grade) + 600 val (120
per DR grade) to ensure balanced class representation and prevent overfitting.

The existing training JSONL uses the OpenAI multi-turn chat format:

    {"messages": [
        {"role": "system",  "content": "..."},
        {"role": "user",    "content": [{"type": "image", "image": "..."}, {"type": "text", "text": "..."}]},
        {"role": "assistant","content": [{"type": "text", "text": "..."}]}
    ]}

Setup (one-time):
    1. pip install modal
    2. modal setup                          # authenticate via browser
    3. modal secret create hf-secret HF_TOKEN=hf_your_token_here

Upload data & run:
    4. modal volume create OpusAI-data   # (skip if already exists)
    5. modal volume put OpusAI-data data/finetune/medgemma/ /data/finetune/medgemma/
    6. modal volume put OpusAI-data data/Imagenes/ /data/Imagenes/
    7. modal volume put OpusAI-data data/dr_unified_v2/ /data/dr_unified_v2/
       # (JSONL refs dr_unified_v2_sampled/ → remapped to dr_unified_v2/ at runtime)
    8. modal run backend/scripts/modal_train_medgemma.py

Download results:
    9. mkdir -p checkpoints/medgemma
       modal volume get OpusAI-checkpoints medgemma/final ./checkpoints/medgemma/
"""

import modal


app = modal.App("OpusAI-medgemma-train")

model_cache = modal.Volume.from_name(
    "OpusAI-model-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("OpusAI-data", create_if_missing=True)
checkpoints_vol = modal.Volume.from_name(
    "OpusAI-checkpoints", create_if_missing=True)


train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "transformers>=4.52.0",
        "peft>=0.12.0",
        "accelerate>=0.34.0",
        "Pillow>=10.0.0",
        "bitsandbytes>=0.43.0",
        "tensorboard>=2.17.0",
        "python-dotenv>=1.0.0",
        "huggingface_hub>=0.23.0",
    )
    .env({
        "HF_HOME": "/model_cache",
        "TRANSFORMERS_VERBOSITY": "info",
    })
)


GPU_TYPE = "H100"       # H100 80 GB
TIMEOUT_HOURS = 1.5     # 30 min – 1 hr expected, 1.5 hr safety margin


@app.function(
    image=train_image,
    gpu=GPU_TYPE,
    memory=32768,
    volumes={
        "/model_cache": model_cache,
        "/data": data_vol,
        "/checkpoints": checkpoints_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=int(TIMEOUT_HOURS * 60 * 60),
)
def train(
    num_epochs: int = 1,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 5e-5,
    max_seq_len: int = 1024,
    lora_r: int = 8,
    lora_alpha: int = 16,
    train_samples: int = 500,
    val_samples: int = 600,
):
    import os
    import json
    import logging
    import random
    import re
    from collections import defaultdict
    from pathlib import Path

    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import (
        AutoProcessor,
        Gemma3ForConditionalGeneration,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("medgemma")

    DATA_ROOT = Path("/data")
    OUTPUT_DIR = "/checkpoints/medgemma"

    hf_token = os.environ["HF_TOKEN"]
    model_id = "google/medgemma-4b-it"

    train_jsonl = DATA_ROOT / "finetune/medgemma/train.jsonl"
    val_jsonl = DATA_ROOT / "finetune/medgemma/val.jsonl"

    # ── Stratified sampling helper ────────────────────────────────────

    def _extract_grade(answer_text: str) -> int:
        """Extract DR grade (0-4) from the assistant answer text."""
        m = re.search(r'\*\*DR Grade:\*\*\s*(\d)', answer_text)
        if m:
            return int(m.group(1))
        return -1

    def _stratified_sample(rows: list[dict], n_total: int, seed: int = 42) -> list[dict]:
        """
        Stratified sample: take equal counts per DR grade.

        Groups rows by DR grade (0–4), then takes n_per_class = n_total // 5
        from each grade.  If a grade has fewer samples than needed, take all
        of them and redistribute the remainder to other grades.
        """
        rng = random.Random(seed)
        buckets = defaultdict(list)

        for row in rows:
            answer = row["messages"][2]["content"]
            if isinstance(answer, list):
                answer = next(c["text"] for c in answer if c["type"] == "text")
            grade = _extract_grade(answer)
            buckets[grade].append(row)

        # Only use known grades (0–4)
        known_grades = sorted(g for g in buckets if 0 <= g <= 4)
        n_classes = len(known_grades)
        n_per_class = n_total // n_classes

        sampled = []
        overflow = 0  # samples we couldn't take from small classes

        # First pass: take up to n_per_class from each grade
        leftover_pools = {}
        for g in known_grades:
            pool = buckets[g]
            rng.shuffle(pool)
            take = min(len(pool), n_per_class)
            sampled.extend(pool[:take])
            if take < n_per_class:
                overflow += n_per_class - take
            else:
                leftover_pools[g] = pool[take:]

        # Second pass: distribute overflow from larger classes
        if overflow > 0 and leftover_pools:
            extra_pool = []
            for g in known_grades:
                if g in leftover_pools:
                    extra_pool.extend(leftover_pools[g])
            rng.shuffle(extra_pool)
            sampled.extend(extra_pool[:overflow])

        rng.shuffle(sampled)
        return sampled

    # ── Load & subsample ──────────────────────────────────────────────

    log.info("Loading + stratified-sampling datasets …")

    with open(train_jsonl) as f:
        all_train = [json.loads(line) for line in f]
    with open(val_jsonl) as f:
        all_val = [json.loads(line) for line in f]

    train_rows = _stratified_sample(all_train, train_samples, seed=42)
    val_rows = _stratified_sample(all_val, val_samples, seed=123)

    # Log class distribution
    for label, rows in [("Train", train_rows), ("Val", val_rows)]:
        grade_counts = defaultdict(int)
        for row in rows:
            answer = row["messages"][2]["content"]
            if isinstance(answer, list):
                answer = next(c["text"] for c in answer if c["type"] == "text")
            grade_counts[_extract_grade(answer)] += 1
        dist = ", ".join(f"G{g}={grade_counts[g]}" for g in sorted(grade_counts))
        log.info("%s: %d samples — %s", label, len(rows), dist)

    # ── Dataset class ─────────────────────────────────────────────────

    class MedGemmaDataset(Dataset):
        """
        Wraps pre-loaded raw JSONL rows.  Returns PIL image + parsed fields.
        """

        def __init__(self, rows, data_root):
            self.data_root = Path(data_root)
            self.samples = []

            for row in rows:
                msgs = row["messages"]
                user_content = msgs[1]["content"]
                image_rel = next(
                    c["image"] for c in user_content if c["type"] == "image"
                )
                rel = image_rel.replace("data/", "", 1)
                rel = rel.replace("dr_unified_v2_sampled/", "dr_unified_v2/")
                img_path = str(self.data_root / rel)

                assistant_content = msgs[2]["content"]
                if isinstance(assistant_content, list):
                    answer = next(
                        c["text"] for c in assistant_content
                        if c["type"] == "text"
                    )
                else:
                    answer = str(assistant_content)

                system_content = msgs[0]["content"]
                if isinstance(system_content, list):
                    system_text = next(
                        c["text"] for c in system_content
                        if c["type"] == "text"
                    )
                else:
                    system_text = str(system_content)

                user_text = next(
                    c["text"] for c in user_content if c["type"] == "text"
                )

                self.samples.append({
                    "image_path": img_path,
                    "system": system_text,
                    "question": user_text,
                    "answer": answer,
                })

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            image = Image.open(s["image_path"]).convert("RGB")
            _MAX_SIDE = 512
            if max(image.size) > _MAX_SIDE:
                image.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
            return {
                "image": image,
                "system": s["system"],
                "question": s["question"],
                "answer": s["answer"],
            }

    train_ds = MedGemmaDataset(train_rows, DATA_ROOT)
    val_ds = MedGemmaDataset(val_rows, DATA_ROOT)
    log.info("Train dataset: %d samples | Val dataset: %d samples",
             len(train_ds), len(val_ds))

    # ── Model loading with 4-bit quantization ─────────────────────────

    log.info(
        "Downloading + loading %s (first run downloads ~8GB, cached after) …",
        model_id,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log.info("[1/5] Loading processor …")
    processor = AutoProcessor.from_pretrained(
        model_id, token=hf_token, use_fast=True,
    )

    log.info("[2/5] Downloading model weights (4-bit NF4 quantized) …")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # ── Freeze vision encoder ─────────────────────────────────────────

    log.info("[3/5] Freezing vision encoder …")
    frozen_count = 0
    for name, param in model.named_parameters():
        if "vision" in name.lower() or "image_encoder" in name.lower():
            param.requires_grad = False
            frozen_count += 1
    log.info("Froze %d vision parameters", frozen_count)

    projector_count = 0
    for name, param in model.named_parameters():
        if "multi_modal_projector" in name:
            param.requires_grad = False
            projector_count += 1
    if projector_count:
        log.info("Froze %d multi-modal projector parameters", projector_count)

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
    )

    # ── LoRA adapters (conservative for small dataset → prevent overfitting) ──

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,       # higher dropout for small dataset
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    log.info("[4/5] Applying LoRA adapters (r=%d, alpha=%d, dropout=0.1) …",
             lora_r, lora_alpha)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log.info("[5/5] Setup complete. Preparing trainer …")

    # ── Collate function ──────────────────────────────────────────────

    def collate_fn(examples):
        """
        Tokenize chat messages and create labels with prompt masking.

        Gemma 3 (transformers >=4.52) requires ``token_type_ids`` during
        training to build the correct causal mask for image vs text tokens.
        The processor returns them; we carry them through and pad them.
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []
        batch_labels = []

        for ex in examples:
            prompt_messages = [
                {"role": "system", "content": [{"type": "text", "text": ex["system"]}]},
                {"role": "user", "content": [
                    {"type": "image", "image": ex["image"]},
                    {"type": "text", "text": ex["question"]},
                ]},
            ]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": [{"type": "text", "text": ex["answer"]}]},
            ]

            # Tokenize prompt only (for masking)
            prompt_inputs = processor(
                text=processor.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True,
                ),
                images=[ex["image"]],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_seq_len,
            )
            prompt_len = prompt_inputs["input_ids"].shape[1]

            # Tokenize full conversation
            full_inputs = processor(
                text=processor.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False,
                ),
                images=[ex["image"]],
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_seq_len,
            )

            input_ids = full_inputs["input_ids"].squeeze(0)
            attention_mask = full_inputs["attention_mask"].squeeze(0)

            # token_type_ids: processor may or may not return them
            if "token_type_ids" in full_inputs:
                token_type_ids = full_inputs["token_type_ids"].squeeze(0)
            else:
                token_type_ids = torch.zeros_like(input_ids)

            # Labels: mask the prompt tokens with -100
            labels = input_ids.clone()
            labels[:prompt_len] = -100

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(labels)

        # Pad to same length
        max_len = max(ids.shape[0] for ids in batch_input_ids)
        pad_id = processor.tokenizer.pad_token_id or 0

        padded_input_ids = []
        padded_attention = []
        padded_token_type = []
        padded_labels = []

        for ids, mask, ttids, labs in zip(
            batch_input_ids, batch_attention_mask,
            batch_token_type_ids, batch_labels,
        ):
            pad_len = max_len - ids.shape[0]
            padded_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_id, dtype=ids.dtype)])
            )
            padded_attention.append(
                torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            )
            padded_token_type.append(
                torch.cat([ttids, torch.zeros(pad_len, dtype=ttids.dtype)])
            )
            padded_labels.append(
                torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention),
            "token_type_ids": torch.stack(padded_token_type),
            "labels": torch.stack(padded_labels),
        }

    # ── Training arguments (tuned for 1 epoch, 500 samples, <1 hr) ───

    total_steps = max(1, len(train_ds) // (batch_size * grad_accum))
    eval_steps = max(1, total_steps // 3)   # evaluate ~3 times during the epoch

    log.info(
        "Total steps: %d | Eval every %d steps | 1 epoch",
        total_steps, eval_steps,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.05,          # higher weight decay for small dataset
        max_grad_norm=1.0,
        warmup_ratio=0.1,           # warm up 10% of steps
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="epoch",      # save once at end
        save_total_limit=1,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to=["tensorboard"],
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        label_smoothing_factor=0.1,  # smoothing to reduce overfitting
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
    )

    # ── Train ─────────────────────────────────────────────────────────

    log.info("Starting MedGemma QLoRA training — 1 epoch, %d train, %d val …",
             len(train_ds), len(val_ds))
    result = trainer.train()
    log.info("Training finished in %.1f min",
             result.metrics.get("train_runtime", 0) / 60)

    metrics = trainer.evaluate()
    log.info("Final eval: %s", metrics)

    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    checkpoints_vol.commit()
    log.info("LoRA adapters saved and committed to volume → %s", final_dir)
    return {
        "train_loss": result.metrics.get("train_loss"),
        "train_runtime_min": round(result.metrics.get("train_runtime", 0) / 60, 1),
        "eval_metrics": metrics,
    }


@app.local_entrypoint()
def main(
    num_epochs: int = 1,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 5e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    train_samples: int = 500,
    val_samples: int = 600,
):
    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  MedGemma 4B QLoRA Training on Modal ({GPU_TYPE} 80GB)  ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║  Model:     google/medgemma-4b-it (4-bit NF4)      ║")
    print(f"║  Epochs:    {num_epochs}                                     ║")
    print(f"║  Train:     {train_samples} samples (100/grade, balanced)     ║")
    print(f"║  Val:       {val_samples} samples (120/grade, balanced)      ║")
    print(f"║  Batch:     {batch_size} × {grad_accum} = {batch_size * grad_accum} effective                  ║")
    print(f"║  LR:        {learning_rate}                                ║")
    print(f"║  LoRA:      r={lora_r}, alpha={lora_alpha}                         ║")
    print(f"║  Overfitting guard: dropout=0.1, wd=0.05, ls=0.1  ║")
    print(f"║  Expected:  30–60 min                              ║")
    print(f"╚══════════════════════════════════════════════════════╝")
    print()

    result = train.remote(
        num_epochs=num_epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        train_samples=train_samples,
        val_samples=val_samples,
    )

    print(f"\n✅ Training complete!")
    print(f"  Train loss:    {result['train_loss']:.4f}")
    print(f"  Runtime:       {result['train_runtime_min']} min")
    if result["eval_metrics"]:
        print(f"  Eval loss:     {result['eval_metrics'].get('eval_loss', 'N/A')}")
    print(f"\nDownload adapters:")
    print(f"  mkdir -p checkpoints/medgemma && \\")
    print(f"  modal volume get OpusAI-checkpoints medgemma/final ./checkpoints/medgemma/")
