"""
Modal.com deployment for PaliGemma QLoRA fine-tuning.

Runs train_paligemma.py on a cloud GPU with persistent storage for
model weights, datasets, and checkpoints.

Setup (one-time):
    1. pip install modal
    2. modal setup                          # authenticate via browser
    3. modal secret create hf-secret HF_TOKEN=hf_your_token_here

Upload data & run:
    4. modal volume create OpusAI-data
    5. modal volume put OpusAI-data data/finetune/ /data/finetune/
    6. modal volume put OpusAI-data data/Imagenes/ /data/Imagenes/
    7. modal volume put OpusAI-data data/dr_unified_v2/ /data/dr_unified_v2/
    8. modal run backend/scripts/modal_train_paligemma.py

Download results (remote paths are relative to the volume root = container /checkpoints):
    9. mkdir -p checkpoints/paligemma
       modal volume get OpusAI-checkpoints paligemma/final ./checkpoints/paligemma/
       # → ./checkpoints/paligemma/final/adapter_model.safetensors, ...
       # Re-download: add --force, or rm -rf checkpoints/paligemma/final first.
"""

import modal


app = modal.App("OpusAI-paligemma-train")

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
        "transformers>=4.44.0",
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


GPU_TYPE = "H100"
TIMEOUT_HOURS = 3


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
    timeout=TIMEOUT_HOURS * 60 * 60,
)
def train(
    num_epochs: int = 8,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    early_stopping_patience: int = 2,
):
    import os
    import json
    import logging
    from pathlib import Path

    import torch
    from PIL import Image
    from torch.utils.data import Dataset
    from transformers import (
        PaliGemmaProcessor,
        PaliGemmaForConditionalGeneration,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("paligemma")

    DATA_ROOT = Path("/data")
    OUTPUT_DIR = "/checkpoints/paligemma"

    hf_token = os.environ["HF_TOKEN"]
    model_id = "google/paligemma-3b-pt-224"
    max_length = 512

    train_jsonl = DATA_ROOT / "finetune/paligemma/train_quality.jsonl"
    val_jsonl = DATA_ROOT / "finetune/paligemma/val_quality.jsonl"

    class PaliGemmaDataset(Dataset):
        def __init__(self, jsonl_path, data_root):
            self.data_root = Path(data_root)
            self.samples = []
            with open(jsonl_path) as f:
                for line in f:
                    row = json.loads(line)
                    user_content = row["messages"][0]["content"]
                    image_rel = next(c["image"]
                                     for c in user_content if c["type"] == "image")
                    question = next(c["text"]
                                    for c in user_content if c["type"] == "text")
                    answer = row["messages"][1]["content"][0]["text"]

                    rel = image_rel.replace("data/", "", 1)
                    rel = rel.replace(
                        "dr_unified_v2_sampled/", "dr_unified_v2/")
                    img_path = str(self.data_root / rel)
                    self.samples.append({
                        "image_path": img_path,
                        "question": question,
                        "answer": answer,
                    })
            unique = len(set(s["image_path"] for s in self.samples))
            log.info("Loaded %s → %d samples (%d images)",
                     Path(jsonl_path).name, len(self.samples), unique)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            image = Image.open(s["image_path"]).convert("RGB")
            return {"image": image, "question": s["question"], "answer": s["answer"]}

    log.info("Loading datasets …")
    train_ds = PaliGemmaDataset(train_jsonl, DATA_ROOT)
    val_ds = PaliGemmaDataset(
        val_jsonl, DATA_ROOT) if val_jsonl.exists() else None

    log.info(
        "Downloading + loading %s (first run downloads ~6GB, cached after) …", model_id)
    import sys
    sys.stdout.flush()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    log.info("[1/5] Loading processor …")
    processor = PaliGemmaProcessor.from_pretrained(model_id, token=hf_token)

    log.info("[2/5] Downloading model weights (cached after first run) …")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    log.info("[3/5] Model loaded. Freezing vision + projector …")

    for name, module in [
        ("vision_tower", getattr(model, "vision_tower", None)),
        ("vision_model", getattr(model, "vision_model", None)),
    ]:
        if module is None and hasattr(model, "model"):
            module = getattr(model.model, name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = False

    projector = getattr(model, "multi_modal_projector", None)
    if projector is None and hasattr(model, "model"):
        projector = getattr(model.model, "multi_modal_projector", None)
    if projector is not None:
        for p in projector.parameters():
            p.requires_grad = False

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    log.info("[4/5] Applying LoRA adapters …")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    log.info("[5/5] Setup complete. Preparing trainer …")

    def collate_fn(examples):
        images = [ex["image"] for ex in examples]
        questions = [ex["question"] for ex in examples]
        answers = [ex["answer"] for ex in examples]
        return processor(
            images=images, text=questions, suffix=answers,
            return_tensors="pt", padding="longest",
            truncation="only_second", max_length=max_length,
            tokenize_newline_separately=False,
        )

    steps_per_epoch = max(1, len(train_ds) // (batch_size * grad_accum))
    eval_steps = max(1, steps_per_epoch // 2)

    log.info("Steps/epoch: %d | Eval every %d steps | Patience: %d",
             steps_per_epoch, eval_steps, early_stopping_patience)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=50,
        logging_steps=25,
        eval_strategy="steps" if val_ds else "no",
        eval_steps=eval_steps if val_ds else None,
        save_strategy="steps" if val_ds else "epoch",
        save_steps=eval_steps if val_ds else None,
        save_total_limit=3,
        load_best_model_at_end=bool(val_ds),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to=["tensorboard"],
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    callbacks = []
    if val_ds:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
        ))

    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=collate_fn, callbacks=callbacks,
    )

    log.info("Starting QLoRA training — max %d epochs …", num_epochs)
    result = trainer.train()

    stopped_epoch = result.metrics.get("epoch", num_epochs)
    if stopped_epoch < num_epochs:
        log.info("Early stopping at epoch %.1f", stopped_epoch)
    else:
        log.info("Completed all %d epochs", num_epochs)

    if val_ds:
        metrics = trainer.evaluate()
        log.info("Final eval: %s", metrics)
        log.info("Best checkpoint: %s", trainer.state.best_model_checkpoint)

    final_dir = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)

    checkpoints_vol.commit()
    log.info("LoRA adapters saved and committed to volume → %s", final_dir)
    return {"stopped_epoch": stopped_epoch, "eval_metrics": metrics if val_ds else {}}


@app.local_entrypoint()
def main(
    num_epochs: int = 8,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    patience: int = 2,
):
    print(f"Launching PaliGemma QLoRA training on Modal ({GPU_TYPE})")
    print(f"  Epochs: {num_epochs} (early stopping patience={patience})")
    print(f"  Effective batch size: {batch_size * grad_accum}")
    print(f"  Learning rate: {learning_rate}")

    result = train.remote(
        num_epochs=num_epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        early_stopping_patience=patience,
    )

    print(f"\nTraining complete!")
    print(f"  Stopped at epoch: {result['stopped_epoch']}")
    if result["eval_metrics"]:
        print(
            f"  Final eval loss: {result['eval_metrics'].get('eval_loss', 'N/A')}")
    print(f"\nDownload adapters:")
    print(f"  mkdir -p checkpoints/paligemma && modal volume get OpusAI-checkpoints paligemma/final ./checkpoints/paligemma/")
    print(f"  (re-download: add --force if ./checkpoints/paligemma/final already exists)")