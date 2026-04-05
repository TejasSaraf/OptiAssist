from __future__ import annotations

"""
diagnosis.py

MedGemma diagnosis agent — uses Google's MedGemma 4B (google/medgemma-4b-it)
via HuggingFace Transformers to produce a structured medical diagnosis for a
retinal fundus image.

Runtime: HuggingFace Transformers pipeline("image-text-to-text")
Model:   google/medgemma-4b-it

Pipeline role:
    1. Gemma 3 prescans the image          → image_description
    2. FunctionGemma decides route          → run_diagnosis called
    3. PaliGemma 2 runs segmentation        → paligemma_context (if applicable)
    4. ★ THIS AGENT: MedGemma diagnosis     → structured JSON result
    5. Gemma 3 synthesises final summary    → merger.py

This module accepts the PaliGemma segmentation output as additional context
so MedGemma can incorporate optic disc/cup detections into its clinical
reasoning.
"""

import asyncio
import io
import json
import logging
import re
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

_HF_MODEL_ID = "google/medgemma-4b-it"

SYSTEM_PROMPT = (
    "You are MedGemma, a general-purpose medical AI model developed by Google. "
    "You are being used to analyze a retinal fundus image and answer a clinical question.\n\n"
    "CRITICAL INSTRUCTION: Your response must be ONLY a valid JSON object. "
    "Do NOT include any explanation, preamble, markdown, or text outside the JSON. "
    "Your response must start with { and end with }.\n\n"
    "Required JSON fields:\n"
    "  condition: string (diagnosed condition, e.g. 'Glaucoma', 'Diabetic Retinopathy', "
    "'Age-related Macular Degeneration', or 'Normal')\n"
    "  severity: string — one of: 'None', 'Mild', 'Moderate', 'Severe', 'Proliferative'\n"
    "  severity_level: integer 0–4  (0=None, 1=Mild, 2=Moderate, 3=Severe, 4=Proliferative)\n"
    "  confidence: float 0.0–1.0\n"
    "  findings: list of strings — MUST be non-empty. Always list every observable feature "
    "seen in the image, whether pathological or normal. For a Normal image, describe the "
    "normal structures you observe (e.g. 'normal optic disc', 'clear macula', "
    "'normal retinal vasculature'). For an abnormal image, list the specific abnormal findings "
    "(e.g. 'increased cup-to-disc ratio', 'dark pigmented macular lesion', 'microaneurysms', "
    "'hard exudates', 'neovascularisation'). Never return an empty list.\n"
    "  recommendation: string — clinical follow-up advice\n"
    "  disclaimer: string — always exactly: "
    "'For research use only. Not intended for clinical diagnosis.'\n\n"
    "Example — abnormal:\n"
    '{"condition": "Glaucoma", "severity": "Moderate", "severity_level": 2, '
    '"confidence": 0.82, '
    '"findings": ["increased cup-to-disc ratio", "optic disc cupping", "neuroretinal rim thinning"], '
    '"recommendation": "Refer to glaucoma specialist for IOP measurement and visual field testing.", '
    '"disclaimer": "For research use only. Not intended for clinical diagnosis."}\n\n'
    "Example — normal:\n"
    '{"condition": "Normal", "severity": "None", "severity_level": 0, '
    '"confidence": 0.91, '
    '"findings": ["normal optic disc appearance", "clear macula", "normal retinal vasculature", "no haemorrhages"], '
    '"recommendation": "Routine eye exam every 1-2 years.", '
    '"disclaimer": "For research use only. Not intended for clinical diagnosis."}'
)

FALLBACK_RESULT = {
    "condition": "Analysis unavailable",
    "severity": "None",
    "severity_level": 0,
    "confidence": 0.0,
    "findings": [],
    "recommendation": "Please retry or consult a qualified ophthalmologist.",
    "disclaimer": "For research use only. Not intended for clinical diagnosis.",
}

_medgemma_pipe = None


def _load_medgemma():
    """Load MedGemma via the HuggingFace Transformers pipeline (lazy)."""
    global _medgemma_pipe

    if _medgemma_pipe is not None:
        return

    import os
    from pathlib import Path as _Path

    from dotenv import load_dotenv
    from transformers import pipeline

    _env_path = _Path(__file__).parent.parent / ".env"
    load_dotenv(_env_path)
    hf_token = os.environ.get("HF_TOKEN")

    logger.info("Loading MedGemma pipeline from %s", _HF_MODEL_ID)
    _medgemma_pipe = pipeline(
        "image-text-to-text",
        model=_HF_MODEL_ID,
        token=hf_token,
    )
    logger.info("MedGemma pipeline loaded successfully.")


def _run_inference(messages: list[dict]) -> str:
    """
    Execute blocking MedGemma pipeline inference synchronously.

    Args:
        messages: Chat-formatted message list. All content fields must be
            list-of-dicts (never a plain string) so that Gemma3Processor's
            apply_chat_template can iterate over them without hitting
            "string indices must be integers".

    Returns:
        Raw text content of the model's last response turn.
    """
    _load_medgemma()

    output = _medgemma_pipe(text=messages, max_new_tokens=512, do_sample=False)
    generated = output[0]["generated_text"]

    if isinstance(generated, list):
        last = generated[-1]
        if isinstance(last, dict):
            content = last.get("content")
            if isinstance(content, list):
                return "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ) or str(last)
            return str(content) if content else str(last)
        return str(last)

    return str(generated)


def _parse_json(raw_text: str) -> dict:
    """
    Parse a JSON dict from the model's raw output string.

    MedGemma sometimes wraps its JSON in natural language or markdown code
    fences.  We try three progressively looser extraction strategies before
    giving up and returning FALLBACK_RESULT.

    Args:
        raw_text: The raw string output from the MedGemma pipeline.

    Returns:
        Parsed diagnosis dict, or FALLBACK_RESULT on failure.
    """

    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start: end + 1])
        except json.JSONDecodeError:
            pass

    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    logger.warning(
        "JSON parsing failed after 3 attempts. Raw output (first 400 chars): %s",
        raw_text[:400],
    )
    return FALLBACK_RESULT


async def run_diagnosis(
    image_bytes: bytes | None,
    query: str,
    image_description: str = "",
    paligemma_context: str = "",
) -> dict:
    """
    Produce a structured ophthalmological diagnosis using MedGemma.

    Args:
        image_bytes:       Raw bytes of the retinal image, or None for text-only queries.
        query:             The clinician's diagnostic question.
        image_description: Pre-scan description from Gemma 3 (optional). Appended
                           to the user prompt so MedGemma can cross-reference its
                           own image analysis with the prescanner findings.
        paligemma_context: Output from PaliGemma 2 segmentation (optional).
                           When provided, MedGemma incorporates the optic disc/cup
                           detection results into its clinical reasoning.

    Returns:
        A dict with keys: condition, severity, severity_level, confidence,
        findings, recommendation, disclaimer.

    Raises:
        RuntimeError: If inference fails unexpectedly.
    """
    pil_image: Image.Image | None = None
    if image_bytes is not None:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise RuntimeError(
                f"Failed to decode image bytes into PIL Image: {e}") from e

    full_query = query
    if image_description:
        full_query = (
            f"{query}\n\n"
            f"Pre-scan image description (from Gemma 3):\n{image_description}"
        )
    if paligemma_context:
        full_query = (
            f"{full_query}\n\n"
            f"PaliGemma 2 segmentation results:\n{paligemma_context}"
        )

    user_content: list[dict]
    if pil_image is not None:
        user_content = [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": full_query},
        ]
    else:
        user_content = [{"type": "text", "text": full_query}]

    messages = [
        {"role": "system", "content": [
            {"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]

    logger.info(
        "Running MedGemma diagnosis. query=%s has_image=%s has_prescan=%s has_paligemma=%s",
        query[:80],
        pil_image is not None,
        bool(image_description),
        bool(paligemma_context),
    )

    try:
        raw_text = await asyncio.to_thread(_run_inference, messages)
    except Exception as e:
        raise RuntimeError(f"MedGemma inference failed: {e}") from e

    logger.info("Raw MedGemma output (first 400 chars): %s", raw_text[:400])

    result = _parse_json(raw_text)
    logger.info("Diagnosis complete. condition=%s severity=%s",
                result.get("condition"), result.get("severity"))

    return result
