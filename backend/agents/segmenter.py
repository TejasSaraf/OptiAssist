from __future__ import annotations

"""
segmenter.py

PaliGemma agent — retinal image analysis via fine-tuned PaliGemma.

When OPTIASSIST_PALIGEMMA_URL is set (e.g. http://localhost:8080), inference
is delegated to the dedicated serve_paligemma.py server via HTTP. Otherwise
the merged model is loaded in-process from backend/models/paligemma-finetuned/.
"""

import asyncio
import io
import logging
import os
import sys
from pathlib import Path

import httpx
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

_DEFAULT_PROMPT = (
    "Analyze this retinal fundus image. What condition is present, "
    "what is the severity, and what findings do you see?"
)


# ── Helpers for building result dicts from HTTP responses ─────────────

def _structure_http_response(
    image_bytes: bytes,
    raw_output: str,
    clean_response: str,
) -> dict:
    """Build the same result shape as paligemma_tool.run_paligemma_detection."""
    from app.tools.paligemma_tool import _draw_detections, _parse_loc_tokens

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size
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
        summary = (clean_response or "")[:300] or "No output generated."

    return {
        "raw_output": clean_response,
        "detections": detections,
        "annotated_image_base64": annotated_b64,
        "summary": summary,
    }


async def _run_segmentation_remote(
    base_url: str, image_bytes: bytes, query: str
) -> dict:
    """POST to serve_paligemma /v1/generate and structure the result."""
    url = f"{base_url}/v1/generate"
    files = {"image": ("fundus.jpg", image_bytes, "image/jpeg")}
    data = {"prompt": query, "max_tokens": "256"}
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(url, files=files, data=data)
        resp.raise_for_status()
        body = resp.json()
    if body.get("status") != "success":
        raise RuntimeError(body.get("message", "PaliGemma API error"))
    raw_for_parse = body.get("raw_output") or body.get("response") or ""
    clean = (body.get("response") or "").strip()
    return _structure_http_response(image_bytes, raw_for_parse, clean)


# ── In-process inference (when no remote URL configured) ──────────────

def _run_inference_sync(image_bytes: bytes, query: str = _DEFAULT_PROMPT) -> dict:
    """
    Blocking wrapper — loads the model once (cached inside paligemma_tool)
    then runs retinal image analysis.
    """
    import os as _os
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
        _os.unlink(tmp_path)

    return result


# ── Public entry point ────────────────────────────────────────────────

async def run_segmentation(image_bytes: bytes, query: str = _DEFAULT_PROMPT) -> dict:
    """
    Analyze a retinal fundus image using the fine-tuned PaliGemma model.

    Routes to the HTTP server at OPTIASSIST_PALIGEMMA_URL when set,
    otherwise loads the merged model in-process.
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes))
        w, h = pil.size
    except Exception as e:
        raise RuntimeError(f"Failed to decode image: {e}") from e

    pali_url = os.environ.get("OPTIASSIST_PALIGEMMA_URL", "").rstrip("/")

    logger.info(
        "PaliGemma — analyzing retinal image. image=%dx%d prompt=%r remote=%s",
        w, h, query[:60], bool(pali_url),
    )

    try:
        if pali_url:
            result = await _run_segmentation_remote(pali_url, image_bytes, query)
        else:
            if not _MODEL_DIR.exists():
                raise FileNotFoundError(
                    f"PaliGemma model not found at {_MODEL_DIR}. "
                    f"Run `python backend/scripts/merge_adapter.py` first, "
                    f"or set OPTIASSIST_PALIGEMMA_URL."
                )
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
