"""
merge_adapter.py

Merges the fine-tuned LoRA adapter into the base PaLiGemma model to produce
a single standalone model directory ready for local inference.

Steps:
    1. Load the base model (google/paligemma-3b-pt-224) in fp16
    2. Load the LoRA adapter from checkpoints/paligemma/final/
    3. Merge adapter weights into the base model via merge_and_unload()
    4. Save the merged model + processor to backend/models/paligemma-finetuned/

Usage:
    python backend/scripts/merge_adapter.py

After merging, set USE_PALIGEMMA = True in backend/agents/diagnosis.py
to use the fine-tuned model for inference.
"""

import os
import sys
import logging
from pathlib import Path

import torch
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("merge_adapter")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / "backend" / ".env")

ADAPTER_PATH = PROJECT_ROOT / "checkpoints" / "paligemma" / "final"
OUTPUT_PATH = PROJECT_ROOT / "backend" / "models" / "paligemma-finetuned"
BASE_MODEL_ID = os.environ.get("HF_MODEL_ID", "google/paligemma-3b-pt-224")
HF_TOKEN = os.environ.get("HF_TOKEN")


if not ADAPTER_PATH.exists():
    log.error("Adapter not found at %s", ADAPTER_PATH)
    sys.exit(1)

if not (ADAPTER_PATH / "adapter_model.safetensors").exists():
    log.error("adapter_model.safetensors not found in %s", ADAPTER_PATH)
    sys.exit(1)

log.info("Base model:   %s", BASE_MODEL_ID)
log.info("Adapter path: %s", ADAPTER_PATH)
log.info("Output path:  %s", OUTPUT_PATH)


def main():
    from peft import PeftModel
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    log.info("Loading base model in fp16 — this downloads ~6 GB on first run...")
    base_model = PaliGemmaForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    log.info("Base model loaded.")

    log.info("Loading LoRA adapter from %s", ADAPTER_PATH)
    model = PeftModel.from_pretrained(
        base_model,
        str(ADAPTER_PATH),
        torch_dtype=torch.float16,
    )
    log.info("Adapter loaded — trainable parameters attached.")

    log.info("Merging adapter weights into base model...")
    model = model.merge_and_unload()
    log.info("Merge complete — adapter layers dissolved into base weights.")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    log.info("Saving merged model to %s", OUTPUT_PATH)
    model.save_pretrained(str(OUTPUT_PATH), safe_serialization=True)
    log.info("Model saved.")

    log.info("Saving processor...")
    processor = PaliGemmaProcessor.from_pretrained(
        BASE_MODEL_ID, token=HF_TOKEN, use_fast=True,
    )
    processor.save_pretrained(str(OUTPUT_PATH))
    log.info("Processor saved.")

    total_size_mb = sum(
        f.stat().st_size for f in OUTPUT_PATH.rglob("*") if f.is_file()
    ) / (1024 * 1024)

    log.info("=" * 60)
    log.info("  Merge complete!")
    log.info("    Output: %s", OUTPUT_PATH)
    log.info("    Total size: %.0f MB", total_size_mb)
    log.info("")
    log.info("Next steps:")
    log.info("  1. Set USE_PALIGEMMA = True in backend/agents/diagnosis.py")
    log.info("  2. Run: python backend/scripts/test_paligemma.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()