"""
paligemma_tool.py

PaliGemma inference tool for retinal image analysis.

Loads the fine-tuned PaliGemma model (google/paligemma-3b-pt-224 + LoRA,
merged) from a local directory and runs inference.  The model is loaded
once on first use and cached for subsequent calls.

The model was fine-tuned for Diabetic Retinopathy VQA — it generates
free-form clinical text conditioned on fundus images.  This module wraps
that capability for use by the pipeline agents (segmenter, diagnosis).

Supported calls:
    run_paligemma_detection(image_path, query_context, ...)
        → Returns structured dict with raw_output, detections, summary, etc.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from pathlib import Path

import torch
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "backend"
    / "models"
    / "paligemma-finetuned"
)


_model = None
_processor = None
_device = None


def _get_device_and_dtype() -> tuple[str, torch.dtype]:
    """Select the best available device and dtype."""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


def _load_model(model_dir: Path | str | None = None):
    """
    Load the PaliGemma model + processor (lazy singleton).

    Args:
        model_dir: Path to the merged model directory.
                   Defaults to backend/models/paligemma-finetuned/.
    """
    global _model, _processor, _device

    if _model is not None:
        return

    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

    model_path = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR

    if not model_path.exists():
        raise FileNotFoundError(
            f"PaliGemma model not found at {model_path}. "
            f"Run `python backend/scripts/merge_adapter.py` first."
        )

    device, dtype = _get_device_and_dtype()
    _device = device

    logger.info("Loading PaliGemma model from %s (device=%s, dtype=%s)",
                model_path, device, dtype)

    _processor = PaliGemmaProcessor.from_pretrained(
        str(model_path), use_fast=True)

    if device == "mps":
        _model = PaliGemmaForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
    else:
        _model = PaliGemmaForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    _model.eval()
    logger.info("PaliGemma model loaded successfully.")


_LOC_PATTERN = re.compile(r"<loc(\d{4})>")


def _parse_loc_tokens(raw_output: str, image_width: int, image_height: int) -> list[dict]:
    """
    Parse <loc####> tokens from PaliGemma output into bounding boxes.

    PaliGemma detection models output sequences of 4 <locXXXX> tokens
    (y_min, x_min, y_max, x_max) normalized to [0, 1024), followed by
    the label text.

    Args:
        raw_output:   Raw decoded string from model generation.
        image_width:  Original image width in pixels.
        image_height: Original image height in pixels.

    Returns:
        List of detection dicts with 'label' and 'bounding_box' keys.
    """
    detections = []
    tokens = _LOC_PATTERN.findall(raw_output)

    if len(tokens) >= 4:
        for i in range(0, len(tokens) - 3, 4):
            y_min = int(tokens[i]) / 1024 * image_height
            x_min = int(tokens[i + 1]) / 1024 * image_width
            y_max = int(tokens[i + 2]) / 1024 * image_height
            x_max = int(tokens[i + 3]) / 1024 * image_width

            loc_end_pattern = rf"<loc{tokens[i+3]}>\s*"
            label_match = re.search(loc_end_pattern + r"([^<]+)", raw_output)
            label = label_match.group(1).strip(
            ) if label_match else f"detection_{i // 4}"

            detections.append({
                "label": label,
                "bounding_box": {
                    "x_min": round(x_min, 1),
                    "y_min": round(y_min, 1),
                    "x_max": round(x_max, 1),
                    "y_max": round(y_max, 1),
                },
            })

    return detections


def _draw_detections(image: Image.Image, detections: list[dict]) -> str:
    """
    Draw bounding boxes on an image and return as base64 PNG.

    Args:
        image:      PIL Image to annotate.
        detections: List of detection dicts from _parse_loc_tokens.

    Returns:
        Base64-encoded PNG string of the annotated image.
    """
    draw = ImageDraw.Draw(image.copy())
    colors = ["#FF4444", "#4444FF", "#44FF44", "#FFAA00", "#AA44FF"]

    for i, det in enumerate(detections):
        bb = det["bounding_box"]
        color = colors[i % len(colors)]
        draw.rectangle(
            [bb["x_min"], bb["y_min"], bb["x_max"], bb["y_max"]],
            outline=color,
            width=3,
        )
        draw.text((bb["x_min"], bb["y_min"] - 15), det["label"], fill=color)

    buf = io.BytesIO()
    draw._image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_paligemma_detection(
    image_path: str,
    query_context: str = "Analyze this retinal fundus image.",
    max_new_tokens: int = 256,
    adapter_dir: Path | str | None = None,
) -> dict:
    """
    Run PaliGemma inference on a retinal image.

    This function handles both:
      - Detection tasks (if model outputs <loc####> tokens → parsed to bounding boxes)
      - VQA/analysis tasks (free-text clinical analysis)

    The adapter_dir parameter is accepted for API compatibility with the
    segmenter but the actual model loaded is the pre-merged model at
    backend/models/paligemma-finetuned/ (since the LoRA weights are
    already merged).

    Args:
        image_path:     Path to the input image file.
        query_context:  The prompt/question to send with the image.
        max_new_tokens: Maximum tokens to generate.
        adapter_dir:    Ignored (kept for API compat). The merged model
                        path is used instead.

    Returns:
        Dict with keys:
            raw_output             (str):   Raw decoded model output.
            detections             (list):  Parsed bounding box detections
                                            (empty if no <loc> tokens found).
            annotated_image_base64 (str):   Base64 PNG with drawn boxes
                                            (empty if no detections).
            summary                (str):   Human-readable summary of results.

    Raises:
        FileNotFoundError: If the model directory doesn't exist.
        RuntimeError:      If inference fails.
    """
    _load_model()

    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    logger.info(
        "PaliGemma inference — image=%s (%dx%d) prompt=%r",
        img_path.name, w, h, query_context[:60],
    )

    inputs = _processor(
        text=query_context,
        images=image,
        return_tensors="pt",
    ).to(_model.device)

    with torch.inference_mode():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_len:]
    raw_output = _processor.decode(generated_ids, skip_special_tokens=False)
    clean_output = _processor.decode(
        generated_ids, skip_special_tokens=True).strip()

    logger.info("PaliGemma raw output: %s", raw_output[:200])

    detections = _parse_loc_tokens(raw_output, w, h)

    annotated_b64 = ""
    if detections:
        try:
            annotated_b64 = _draw_detections(image, detections)
        except Exception as e:
            logger.warning("Failed to draw detections: %s", e)

    if detections:
        labels = [d["label"] for d in detections]
        summary = f"Detected {len(detections)} region(s): {', '.join(labels)}"
    else:
        summary = clean_output[:300] if clean_output else "No output generated."

    return {
        "raw_output": clean_output,
        "detections": detections,
        "annotated_image_base64": annotated_b64,
        "summary": summary,
    }