"""
eval_paligemma.py

Evaluates the fine-tuned PaLiGemma model on the held-out test set.

Asks the model to grade each retinal fundus image for diabetic retinopathy,
then compares the predicted grade against the ground-truth label from
data/splits/test.csv.

Metrics computed:
    • Per-image grade extraction + comparison
    • Overall accuracy (exact grade match)
    • ±1 accuracy (prediction within 1 grade of truth)
    • Per-class accuracy breakdown
    • Confusion matrix
    • Cohen's Kappa (inter-rater agreement)

Usage:
    python backend/scripts/eval_paligemma.py
    python backend/scripts/eval_paligemma.py --limit 20        # test on first 20 images
    python backend/scripts/eval_paligemma.py --output results   # save results to file
"""

import argparse
import csv
import json
import logging
import re
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
log = logging.getLogger("eval_paligemma")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "paligemma-finetuned"
TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"

# The grading prompt — matches the training QA type 1 (Classification)
GRADE_PROMPT = "How severe is the diabetic retinopathy in this image?"

# Grade label mapping
GRADE_LABELS = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR / Severe DR",
    4: "Proliferative DR (PDR)",
}


def extract_grade(text: str) -> int | None:
    """
    Extract the DR grade (0–4) from the model's free-text response.

    Tries multiple extraction strategies:
      1. Explicit "Grade X" pattern
      2. Keywords like "Proliferative", "Severe", "Moderate", "Mild", "No DR"
      3. Standalone digit 0–4
    """
    text_lower = text.lower().strip()

    match = re.search(r"grade\s*(\d)", text_lower)
    if match:
        grade = int(match.group(1))
        if 0 <= grade <= 4:
            return grade

    if "proliferative" in text_lower and "non" not in text_lower.split("proliferative")[0][-5:]:
        return 4
    if "pdr" in text_lower:
        return 4
    if "severe" in text_lower:
        if "npdr" in text_lower or "non-proliferative" in text_lower or "non proliferative" in text_lower:
            return 3
        return 3
    if "moderate" in text_lower:
        return 2
    if "mild" in text_lower:
        return 1
    if "no dr" in text_lower or "no diabetic" in text_lower or "normal" in text_lower or "grade 0" in text_lower:
        return 0
    if "no apparent" in text_lower:
        return 0

    digit_match = re.search(r"\b([0-4])\b", text)
    if digit_match:
        return int(digit_match.group(1))

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned PaLiGemma")
    parser.add_argument("--limit", "-n", type=int, default=None,
                        help="Limit evaluation to first N images")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save per-image results to this JSON file")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        log.error("Model not found at %s. Run merge_adapter.py first.", MODEL_PATH)
        sys.exit(1)
    if not TEST_CSV.exists():
        log.error("Test CSV not found at %s", TEST_CSV)
        sys.exit(1)

    samples = []
    with open(TEST_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = PROJECT_ROOT / row["image_path"]
            if not img_path.exists():
                log.warning("Image not found, skipping: %s", img_path)
                continue
            samples.append({
                "image_id": row["image_id"],
                "image_path": str(img_path),
                "true_grade": int(row["dr_grade"]),
                "true_label": row["dr_grade_label"],
            })

    if args.limit:
        samples = samples[:args.limit]

    log.info("Evaluating on %d test images", len(samples))

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

    log.info("Loading model...")
    t0 = time.time()

    processor = PaliGemmaProcessor.from_pretrained(
        str(MODEL_PATH), use_fast=True)

    if device == "mps":
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            str(MODEL_PATH), torch_dtype=dtype, low_cpu_mem_usage=True,
        ).to(device)
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            str(MODEL_PATH), torch_dtype=dtype, device_map="auto",
            low_cpu_mem_usage=True,
        )

    model.eval()
    log.info("Model loaded in %.1fs", time.time() - t0)

    results = []
    correct = 0
    within_one = 0
    class_correct = {g: 0 for g in range(5)}
    class_total = {g: 0 for g in range(5)}
    confusion = [[0] * 5 for _ in range(5)]
    parse_failures = 0

    log.info("Starting evaluation...")
    eval_start = time.time()

    for i, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")

        inputs = processor(
            text=GRADE_PROMPT,
            images=image,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=128, do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_len:]
        raw_output = processor.decode(generated_ids, skip_special_tokens=True)

        pred_grade = extract_grade(raw_output)
        true_grade = sample["true_grade"]

        class_total[true_grade] += 1

        if pred_grade is not None:
            is_correct = pred_grade == true_grade
            is_within_one = abs(pred_grade - true_grade) <= 1

            if is_correct:
                correct += 1
                class_correct[true_grade] += 1
            if is_within_one:
                within_one += 1

            confusion[true_grade][pred_grade] += 1
        else:
            is_correct = False
            is_within_one = False
            parse_failures += 1

        results.append({
            "image_id": sample["image_id"],
            "true_grade": true_grade,
            "true_label": sample["true_label"],
            "pred_grade": pred_grade,
            "raw_output": raw_output,
            "correct": is_correct,
            "within_one": is_within_one,
        })

        status = "Yes" if is_correct else ("≈" if is_within_one else "No")
        log.info(
            "[%d/%d] %s  true=%d  pred=%s  %s  |  %s",
            i + 1, len(samples), sample["image_id"],
            true_grade, pred_grade if pred_grade is not None else "??",
            status, raw_output[:80],
        )

    eval_elapsed = time.time() - eval_start
    n = len(samples)

    def cohens_kappa(conf_matrix, n_classes=5):
        """Compute Cohen's Kappa from a confusion matrix."""
        total = sum(sum(row) for row in conf_matrix)
        if total == 0:
            return 0.0
        po = sum(conf_matrix[i][i] for i in range(n_classes)) / total
        pe = sum(
            sum(conf_matrix[i][j] for j in range(n_classes)) *
            sum(conf_matrix[j][i] for j in range(n_classes))
            for i in range(n_classes)
        ) / (total ** 2)
        if pe == 1.0:
            return 1.0
        return (po - pe) / (1 - pe)

    kappa = cohens_kappa(confusion)

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS — Fine-Tuned PaLiGemma")
    print("=" * 70)

    print(f"\n  Total images:        {n}")
    print(
        f"  Evaluation time:     {eval_elapsed:.1f}s ({eval_elapsed/n:.1f}s per image)")
    print(
        f"  Parse failures:      {parse_failures} ({parse_failures/n*100:.1f}%)")

    print(f"\n  ┌──────────────────────────────────────────┐")
    print(
        f"  │  Exact Accuracy:     {correct}/{n} = {correct/n*100:.1f}%{' ' * 10}│")
    print(
        f"  │  ±1 Accuracy:        {within_one}/{n} = {within_one/n*100:.1f}%{' ' * 10}│")
    print(f"  │  Cohen's Kappa:      {kappa:.3f}{' ' * 19}│")
    print(f"  └──────────────────────────────────────────┘")

    print(f"\n  Per-Class Accuracy:")
    print(f"  {'Grade':<8} {'Label':<25} {'Correct':>8} {'Total':>6} {'Accuracy':>9}")
    print(f"  {'─'*8} {'─'*25} {'─'*8} {'─'*6} {'─'*9}")
    for g in range(5):
        if class_total[g] > 0:
            acc = class_correct[g] / class_total[g] * 100
            print(
                f"  {g:<8} {GRADE_LABELS[g]:<25} {class_correct[g]:>8} {class_total[g]:>6} {acc:>8.1f}%")
        else:
            print(f"  {g:<8} {GRADE_LABELS[g]:<25} {'—':>8} {'0':>6} {'—':>9}")

    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    print(f"  {'':>8}", end="")
    for g in range(5):
        print(f"  P={g}", end="")
    print()
    for g in range(5):
        print(f"  T={g:>3}  ", end="")
        for p in range(5):
            val = confusion[g][p]
            print(f"  {val:>3}", end="")
        print(f"   ({GRADE_LABELS[g]})")

    print("=" * 70)

    if args.output:
        output_path = PROJECT_ROOT / f"{args.output}.json"
        with open(output_path, "w") as f:
            json.dump({
                "metrics": {
                    "total": n,
                    "exact_accuracy": correct / n,
                    "within_one_accuracy": within_one / n,
                    "cohens_kappa": kappa,
                    "parse_failures": parse_failures,
                    "per_class_accuracy": {
                        str(g): class_correct[g] / class_total[g]
                        if class_total[g] > 0 else None
                        for g in range(5)
                    },
                    "confusion_matrix": confusion,
                },
                "per_image": results,
            }, f, indent=2)
        log.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()