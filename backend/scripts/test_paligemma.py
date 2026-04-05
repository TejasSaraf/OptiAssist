"""
test_paligemma.py

Quick smoke test for the merged fine-tuned PaLiGemma model.
Loads the model from backend/models/paligemma-finetuned/ and runs
inference on a user-supplied retinal image.

Usage:
    python backend/scripts/test_paligemma.py --image /path/to/retinal.jpg
    python backend/scripts/test_paligemma.py --image /path/to/retinal.jpg --prompt "Describe the optic disc."
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_paligemma")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "paligemma-finetuned"


def main():
    parser = argparse.ArgumentParser(
        description="Test fine-tuned PaLiGemma model")
    parser.add_argument(
        "--image", "-i", required=True, help="Path to a retinal fundus image"
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="Analyze this retinal fundus image. What condition is present, what is the severity, and what findings do you see?",
        help="Prompt / clinical question to ask the model",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max new tokens to generate"
    )
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        log.error("Merged model not found at %s", MODEL_PATH)
        log.error("Run `python backend/scripts/merge_adapter.py` first.")
        sys.exit(1)

    image_path = Path(args.image)
    if not image_path.exists():
        log.error("Image not found: %s", image_path)
        sys.exit(1)

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    log.info("Device: %s  |  dtype: %s", device, dtype)

    log.info("Loading model from %s", MODEL_PATH)
    t0 = time.time()

    processor = PaliGemmaProcessor.from_pretrained(
        str(MODEL_PATH), use_fast=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=dtype,
        device_map="auto" if device != "mps" else None,
    )
    if device == "mps":
        model = model.to(device)

    log.info("Model loaded in %.1fs", time.time() - t0)

    image = Image.open(image_path).convert("RGB")
    log.info("Image: %s  (%dx%d)", image_path.name, image.width, image.height)

    inputs = processor(
        text=args.prompt,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    log.info("Running inference (max_new_tokens=%d)...", args.max_tokens)
    t1 = time.time()

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )

    elapsed = time.time() - t1

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    result = processor.decode(generated_ids, skip_special_tokens=True)

    log.info("Inference completed in %.2fs", elapsed)

    print("\n" + "=" * 60)
    print("PROMPT:", args.prompt)
    print("=" * 60)
    print("\nMODEL OUTPUT:")
    print(result)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()