"""
PaliGemma QLoRA fine-tuning for Diabetic Retinopathy VQA.

Model: google/paligemma-3b-pt-224
    SigLIP ViT-So400m/14 (224px) → 256 visual tokens
    + Multi-Modal Projector (frozen)
    + Gemma 2B Language Model (4-bit base + LoRA adapters)

This is NOT image classification — the model generates free-form clinical
text conditioned on the fundus image.  The quality of generated text
depends entirely on the quality of training answers.

Dataset: 6,000 QA pairs from 1,000 fundus images — 6 QA types per image:
    1. Classification          (grade identification)
    2. Lesion identification   (what's present)
    3. Clinical reasoning      (why this grade)
    4. Clinical action         (what to do)
    5. Confidence assessment   (uncertainty calibration)
    6. Differential diagnosis  (what's absent, what else to consider)

Types 5-6 mitigate the "label → text shortcut" failure mode by forcing
the model to reason about confidence, absence, and alternatives rather
than just reciting findings for a known grade.

VRAM estimate (3B, 224px, QLoRA):
    Base model (4-bit):  ~1.5 GB
    LoRA adapters:       ~30 MB
    Activations + optim: ~2 GB
    Total:               ~4 GB

Training time: ~2 hrs max on H100 (8 epochs, effective batch size 16)
    Early stopping monitors eval_loss every half-epoch and halts
    when no improvement for 2 consecutive evaluations (patience=2).

Usage:
    python backend/scripts/train_paligemma.py
"""

import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / "backend" / ".env")


class PaliGemmaDataset(Dataset):
    """
    Loads the multimodal messages JSONL.  Returns raw (PIL image, question,
    answer) tuples — tokenisation is deferred to the collate function so
    the DataLoader can pad and batch efficiently.
    """

    def __init__(self, jsonl_path: str | Path, project_root: Path):
        self.project_root = Path(project_root)
        self.samples: list[dict] = []

        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                user_content = row["messages"][0]["content"]

                image_rel = next(c["image"]
                                 for c in user_content if c["type"] == "image")
                question = next(c["text"]
                                for c in user_content if c["type"] == "text")
                answer = row["messages"][1]["content"][0]["text"]

                self.samples.append({
                    "image_path": str(self.project_root / image_rel),
                    "question": question,
                    "answer": answer,
                })

        unique_images = len(set(s["image_path"] for s in self.samples))
        log.info("Loaded %s  →  %s samples  (%s unique images)",
                 Path(jsonl_path).name, f"{len(self.samples):,}", f"{unique_images:,}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        return {"image": image, "question": s["question"], "answer": s["answer"]}


def main():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError(
            "HF_TOKEN not set.  PaliGemma is a gated model — "
            "1) accept the license at https://huggingface.co/google/paligemma-3b-pt-224  "
            "2) create a token at https://huggingface.co/settings/tokens  "
            "3) add it to backend/.env"
        )

    model_id = os.environ.get("HF_MODEL_ID", "google/paligemma-3b-pt-224")
    max_length = int(os.environ.get("MAX_LENGTH", "512"))
    num_epochs = int(os.environ.get("NUM_EPOCHS", "8"))
    early_stopping_patience = int(
        os.environ.get("EARLY_STOPPING_PATIENCE", "2"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "4"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-4"))

    train_jsonl = PROJECT_ROOT / os.environ.get(
        "TRAIN_JSONL", "data/finetune/paligemma/train_quality.jsonl")
    val_jsonl = PROJECT_ROOT / os.environ.get(
        "VAL_JSONL", "data/finetune/paligemma/val_quality.jsonl")
    output_dir = str(PROJECT_ROOT / os.environ.get(
        "OUTPUT_DIR", "checkpoints/paligemma"))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log.info("Model: %s", model_id)
    log.info("Train: %s", train_jsonl)
    log.info("Output: %s", output_dir)

    train_ds = PaliGemmaDataset(train_jsonl, PROJECT_ROOT)
    val_ds = PaliGemmaDataset(
        val_jsonl, PROJECT_ROOT) if val_jsonl.exists() else None

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    processor = PaliGemmaProcessor.from_pretrained(
        model_id, token=hf_token, use_fast=True,
    )

    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    vision_module = getattr(model, "vision_tower", None)
    if vision_module is None:
        vision_module = getattr(model, "vision_model", None)
    if vision_module is None and hasattr(model, "model"):
        vision_module = getattr(model.model, "vision_tower", None) or getattr(
            model.model, "vision_model", None)
    if vision_module is None:
        raise AttributeError("Could not find PaliGemma vision module.")
    for p in vision_module.parameters():
        p.requires_grad = False

    projector_module = getattr(model, "multi_modal_projector", None)
    if projector_module is None and hasattr(model, "model"):
        projector_module = getattr(model.model, "multi_modal_projector", None)
    if projector_module is not None:
        for p in projector_module.parameters():
            p.requires_grad = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def collate_fn(examples):
        """
        The processor's `suffix` parameter creates labels where prompt + image
        tokens are masked to -100 so loss is only computed on answer tokens.
        """
        images = [ex["image"] for ex in examples]
        questions = [ex["question"] for ex in examples]
        answers = [ex["answer"] for ex in examples]

        inputs = processor(
            images=images,
            text=questions,
            suffix=answers,
            return_tensors="pt",
            padding="longest",
            truncation="only_second",
            max_length=max_length,
            tokenize_newline_separately=False,
        )

        return inputs

    steps_per_epoch = max(1, len(train_ds) // (batch_size * grad_accum))
    eval_steps = max(1, steps_per_epoch // 2)

    log.info("Steps/epoch: %d  |  Eval every: %d steps (2× per epoch)  |  "
             "Early stopping patience: %d evals", steps_per_epoch, eval_steps,
             early_stopping_patience)

    args = TrainingArguments(
        output_dir=output_dir,
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
        bf16=(device == "cuda"),
        fp16=False,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    callbacks = []
    if val_ds:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience,
        ))
        log.info("Early stopping ENABLED — will stop if eval_loss does not "
                 "improve for %d consecutive evaluations", early_stopping_patience)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        callbacks=callbacks,
    )

    log.info("Starting QLoRA training — max %d epochs …", num_epochs)
    result = trainer.train()

    stopped_epoch = result.metrics.get("epoch", num_epochs)
    if stopped_epoch < num_epochs:
        log.info("Early stopping triggered at epoch %.1f (max was %d)",
                 stopped_epoch, num_epochs)
    else:
        log.info("Completed all %d epochs", num_epochs)

    if val_ds:
        metrics = trainer.evaluate()
        log.info("Final validation metrics: %s", metrics)
        log.info("Best checkpoint: %s", trainer.state.best_model_checkpoint)

    final_dir = os.path.join(output_dir, "final")
    trainer.save_model(final_dir)
    processor.save_pretrained(final_dir)
    log.info("LoRA adapters + processor saved → %s", final_dir)


if __name__ == "__main__":
    main()