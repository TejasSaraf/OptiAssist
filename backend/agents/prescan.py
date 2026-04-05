"""
prescanner.py

Pre-scans a retinal image using Gemma 3 via the Ollama local API and returns
a brief natural-language description of the image content. This description is
passed downstream to the router to help select the appropriate analysis path.
"""

import base64
import logging
import os

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get(
    "OPTIASSIST_OLLAMA_CHAT_URL",
    "http://localhost:11434/api/chat",
)
OLLAMA_MODEL = os.environ.get("OPTIASSIST_GEMMA3_PRESCAN_MODEL", "gemma3:4b")
PRESCAN_PROMPT = (
    "You are an expert ophthalmology image analysis assistant specializing in "
    "diabetic retinopathy screening.\n\n"
    "Analyze this retinal fundus image. Describe the visible anatomical "
    "structures (optic disc, macula, retinal vasculature) and identify any "
    "pathological signs relevant to diabetic retinopathy such as "
    "microaneurysms, hard exudates, cotton-wool spots, hemorrhages, "
    "neovascularization, or macular edema.\n\n"
    "Be factual, clinical, and concise (2-3 sentences). "
    "Do NOT add any disclaimer, legal notice, or statement about not being "
    "a medical professional. Output only the clinical description."
)
FALLBACK_DESCRIPTION = "Retinal fundus image"


async def prescan_image(image_bytes: bytes) -> str:
    """
    Send a retinal image to Gemma 3 (via Ollama) and return a short description.

    Args:
        image_bytes: Raw bytes of the input retinal image.

    Returns:
        A 1-2 sentence plain-text description of the image.
        Falls back to a generic label if the Ollama call fails.
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "user",
                "content": PRESCAN_PROMPT,
                "images": [base64_image],
            }
        ],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            description = data["message"]["content"]
            logger.info("Prescan succeeded. description=%s", description[:80])
            return description
    except Exception as e:
        logger.warning(
            "Prescan failed, using fallback description. reason=%s", str(e))
        return FALLBACK_DESCRIPTION
