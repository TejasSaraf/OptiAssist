from __future__ import annotations

"""
segmenter.py

PaliGemma agent

This agent uses the fine-tuned PaliGemma model (google/paligemma-3b-pt-224,
LoRA-adapted, merged) for retinal image analysis.  Given a fundus image
it generates a clinical analysis that can include detection of structures
(optic disc, optic cup) or general DR findings.

The model was fine-tuned on Diabetic Retinopathy VQA — 6 question types:
  1. Classification (grade identification)
  2. Lesion identification
  3. Clinical reasoning
  4. Clinical action
  5. Confidence assessment
  6. Differential diagnosis

Model details
-------------
  Base model:  google/paligemma-3b-pt-224
  Fine-tuning: QLoRA (r=8, alpha=16) on DR VQA dataset
  Merged:      backend/models/paligemma-finetuned/

Agent role in the pipeline
---------------------------
  1. Gemma 3 prescans the image → prescan.py
  2. FunctionGemma routes to the appropriate agent(s) → router.py
  3. ★ THIS AGENT: PaliGemma analyzes the fundus image.
  4. MedGemma runs general medical diagnosis → diagnosis.py
  5. Results merged into final output → merger.py
"""

import asyncio
import io
import logging
import sys
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

_MODEL_DIR = (
    Path(__file__).parent.parent
    / "models"
    / "paligemma-finetuned"
)


_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_DEFAULT_PROMPT = "Analyze this retinal fundus image. What condition is present, what is the severity, and what findings do you see?"


def _run_inference_sync(image_bytes: bytes, query: str = _DEFAULT_PROMPT) -> dict:
    """
    Blocking wrapper — loads the model once (cached inside paligemma_tool)
    then runs retinal image analysis.

    Called via ``asyncio.to_thread`` so the event loop stays free.

    Args:
        image_bytes: Raw bytes of the input fundus image.
        query:       The analysis prompt to send with the image.

    Returns:
        Raw result dict from run_paligemma_detection.
    """
    import os
    import tempfile

    from app.tools.paligemma_tool import run_paligemma_detection

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = run_paligemma_detection(
            image_path=tmp_path,
            query_context=query,
            max_new_tokens=256,
        )
    finally:
        os.unlink(tmp_path)

    return result


async def run_segmentation(image_bytes: bytes, query: str = _DEFAULT_PROMPT) -> dict:
    """
    Analyze a retinal fundus image using the fine-tuned PaliGemma model.

    This agent wraps the merged PaliGemma model for retinal analysis.
    It generates clinical text describing DR grade, lesions, and findings.
    If the model outputs <loc####> tokens (detection mode), those are
    parsed into bounding boxes automatically.

    Args:
        image_bytes: Raw bytes of the input retinal fundus image (JPEG/PNG).
        query:       The analysis prompt/question for the model.

    Returns:
        A dict with keys:
            ``"detections"``            (list[dict]): Parsed bounding box
                detections (empty if model outputs free text instead).
            ``"annotated_image_base64"`` (str):  Base64 PNG with boxes drawn
                (empty string if no detections).
            ``"raw_output"``            (str):  Raw model output text.
            ``"summary"``               (str):  Human-readable analysis summary.

    Raises:
        RuntimeError: If the image cannot be decoded or PaliGemma inference fails.
        FileNotFoundError: If the model directory doesn't exist.
    """
    if not _MODEL_DIR.exists():
        raise FileNotFoundError(
            f"PaliGemma model not found at {_MODEL_DIR}. "
            f"Run `python backend/scripts/merge_adapter.py` first."
        )

    try:
        pil = Image.open(io.BytesIO(image_bytes))
        w, h = pil.size
    except Exception as e:
        raise RuntimeError(f"Failed to decode image: {e}") from e

    logger.info(
        "PaliGemma — analyzing retinal image. image=%dx%d prompt=%r",
        w, h, query[:60],
    )

    try:
        result = await asyncio.to_thread(_run_inference_sync, image_bytes, query)
    except (FileNotFoundError, ImportError):
        raise
    except Exception as e:
        raise RuntimeError(f"PaliGemma inference failed: {e}") from e

    detections = result.get("detections", [])
    summary = result.get("summary", "")
    logger.info(
        "PaliGemma done. detections=%d summary=%s",
        len(detections), summary[:100],
    )
    return result