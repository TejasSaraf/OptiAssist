from __future__ import annotations

"""
diagnosis.py

MedGemma diagnosis — structured JSON for a retinal fundus image.

Backends (first match wins for step 5 in ``/api/analyze``):
    1. OPTIASSIST_MEDGEMMA_URL set → HTTP to serve_medgemma.py (HF Transformers)
    2. Else default → Ollama ``/api/chat`` (same as Gemma 3 prescan / FunctionGemma).
       Model tag: OPTIASSIST_MEDGEMMA_OLLAMA_MODEL (default ``medgemma``).
       Build the model from ``backend/ollama/Modelfile``.
    3. OPTIASSIST_MEDGEMMA_BACKEND=hf  (or OPTIASSIST_USE_HF_MEDGEMMA=1)
       → HuggingFace Transformers in-process (google/medgemma-4b-it)
    4. OPTIASSIST_USE_OLLAMA_MEDGEMMA=0 → skip Ollama and use HF when URL empty

Device (HF full-precision only; CUDA 4-bit unchanged):
    OPTIASSIST_MEDGEMMA_DEVICE   — cpu | cuda | mps | auto (default auto)
    OPTIASSIST_MEDGEMMA_USE_MPS  — 1 to allow MPS when auto and no CUDA
        (default 0: use CPU on Apple Silicon — avoids MPS placeholder bugs)

Ollama env:
    OPTIASSIST_OLLAMA_CHAT_URL       — default http://localhost:11434/api/chat
    OPTIASSIST_MEDGEMMA_OLLAMA_MODEL — default medgemma

Ollama model creation: see ``backend/ollama/Modelfile`` (GGUF + mmproj download,
then ``cd backend/ollama && ollama create medgemma -f Modelfile``).

Orchestrator order (see ``main.py``): Input → Gemma 3 prescan → FunctionGemma →
PaliGemma → ★ MedGemma (this module) → Gemma 3 summary (merger).
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
from pathlib import Path

import httpx
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
_medgemma_4bit = False  # tracks whether we loaded with quantization


def _load_medgemma():
    """Load MedGemma via HuggingFace Transformers, with optional 4-bit quantization.

    When ``OPTIASSIST_MEDGEMMA_4BIT`` is ``"1"`` (the default) **and** CUDA is
    available, the model is loaded with NF4 4-bit quantization via
    ``BitsAndBytesConfig``.  This cuts VRAM from ~8 GB to ~2.5 GB and speeds up
    inference on consumer GPUs.

    On MPS (Apple Silicon) or CPU — where ``bitsandbytes`` 4-bit is not
    supported — the function silently falls back to full-precision loading.
    """
    global _medgemma_pipe, _medgemma_4bit

    if _medgemma_pipe is not None:
        return

    from pathlib import Path as _Path

    import torch
    from dotenv import load_dotenv
    from transformers import AutoModelForImageTextToText, AutoProcessor, pipeline

    _env_path = _Path(__file__).parent.parent / ".env"
    load_dotenv(_env_path)
    hf_token = os.environ.get("HF_TOKEN")

    # ── Determine whether to use 4-bit quantization ───────────────────
    want_4bit = os.environ.get("OPTIASSIST_MEDGEMMA_4BIT", "1").strip() == "1"
    use_4bit = want_4bit and torch.cuda.is_available()

    if want_4bit and not torch.cuda.is_available():
        logger.warning(
            "4-bit quantization requested but CUDA is not available "
            "(bitsandbytes NF4 requires CUDA). Falling back to full precision."
        )

    if use_4bit:
        # ── 4-bit NF4 quantized loading ──────────────────────────────
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        logger.info(
            "Loading MedGemma from %s with 4-bit NF4 quantization "
            "(compute_dtype=bfloat16, double_quant=True)",
            _HF_MODEL_ID,
        )

        processor = AutoProcessor.from_pretrained(
            _HF_MODEL_ID, token=hf_token, use_fast=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            _HF_MODEL_ID,
            token=hf_token,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        _medgemma_pipe = pipeline(
            "image-text-to-text",
            model=model,
            tokenizer=processor.tokenizer,
            image_processor=processor.image_processor,
        )
        _medgemma_4bit = True
        logger.info("MedGemma pipeline loaded successfully (4-bit quantized).")

    else:
        # ── Full-precision loading (CUDA / optional MPS / CPU) ───────────
        # Apple MPS + Gemma multimodal pipelines often raise:
        #   Placeholder storage has not been allocated on MPS device!
        # Default on Mac is therefore CPU unless OPTIASSIST_MEDGEMMA_USE_MPS=1
        # or OPTIASSIST_MEDGEMMA_DEVICE=mps.
        has_cuda = torch.cuda.is_available()
        has_mps = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        dev_pref = os.environ.get("OPTIASSIST_MEDGEMMA_DEVICE", "").strip().lower()
        use_mps_flag = os.environ.get(
            "OPTIASSIST_MEDGEMMA_USE_MPS", "0",
        ).strip().lower() in ("1", "true", "yes", "on")

        if dev_pref == "cpu":
            backend = "cpu"
        elif dev_pref == "cuda" and has_cuda:
            backend = "cuda"
        elif dev_pref == "mps" and has_mps:
            backend = "mps"
        elif dev_pref in ("auto", ""):
            if has_cuda:
                backend = "cuda"
            elif has_mps and use_mps_flag:
                backend = "mps"
            else:
                backend = "cpu"
                if has_mps and not use_mps_flag:
                    logger.warning(
                        "MedGemma: skipping MPS (multimodal instability). "
                        "Using CPU. For Apple GPU try OPTIASSIST_MEDGEMMA_USE_MPS=1 "
                        "or OPTIASSIST_MEDGEMMA_DEVICE=mps at your own risk."
                    )
        else:
            backend = "cpu"

        processor = AutoProcessor.from_pretrained(
            _HF_MODEL_ID, token=hf_token, use_fast=True,
        )

        if backend == "cuda":
            dtype = torch.bfloat16
            device_map = "auto"
            logger.info(
                "Loading MedGemma (CUDA, dtype=%s, device_map=%s)",
                dtype, device_map,
            )
            _medgemma_pipe = pipeline(
                "image-text-to-text",
                model=_HF_MODEL_ID,
                token=hf_token,
                torch_dtype=dtype,
                device_map=device_map,
                image_processor=processor.image_processor,
                tokenizer=processor.tokenizer,
            )
        elif backend == "mps":
            dtype = torch.float16
            logger.info(
                "Loading MedGemma (MPS, dtype=%s, device_map=auto) — may be unstable",
                dtype,
            )
            _medgemma_pipe = pipeline(
                "image-text-to-text",
                model=_HF_MODEL_ID,
                token=hf_token,
                torch_dtype=dtype,
                device_map="auto",
                image_processor=processor.image_processor,
                tokenizer=processor.tokenizer,
            )
        else:
            dtype = torch.float32
            logger.info(
                "Loading MedGemma (CPU, dtype=%s) — slower but stable on Apple Silicon",
                dtype,
            )
            _medgemma_pipe = pipeline(
                "image-text-to-text",
                model=_HF_MODEL_ID,
                token=hf_token,
                torch_dtype=dtype,
                device="cpu",
                image_processor=processor.image_processor,
                tokenizer=processor.tokenizer,
            )

        _medgemma_4bit = False
        logger.info("MedGemma pipeline loaded successfully (full precision).")


def _run_inference(messages: list[dict]) -> str:
    """
    Execute blocking MedGemma pipeline inference synchronously.

    All content fields must be list-of-dicts (never a plain string) so
    Gemma3Processor's apply_chat_template can iterate correctly.
    """
    _load_medgemma()

    output = _medgemma_pipe(text=messages, max_new_tokens=256, do_sample=False)
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


def _build_diagnosis_user_text(
    query: str,
    image_description: str,
    paligemma_context: str,
) -> str:
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
    return full_query


def _use_ollama_medgemma() -> bool:
    """
    Default: Ollama (aligns with Gemma 3 + FunctionGemma on the same host).

    Use HuggingFace in-process instead:
        OPTIASSIST_MEDGEMMA_BACKEND=hf
        or OPTIASSIST_USE_HF_MEDGEMMA=1
        or OPTIASSIST_USE_OLLAMA_MEDGEMMA=0
    """
    backend = os.environ.get("OPTIASSIST_MEDGEMMA_BACKEND", "").strip().lower()
    if backend in ("hf", "transformers", "huggingface"):
        return False
    if backend == "ollama":
        return True
    if os.environ.get("OPTIASSIST_USE_HF_MEDGEMMA", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return False
    if os.environ.get("OPTIASSIST_USE_OLLAMA_MEDGEMMA", "").strip().lower() in (
        "0", "false", "no", "off",
    ):
        return False
    if os.environ.get("OPTIASSIST_USE_OLLAMA_MEDGEMMA", "").strip().lower() in (
        "1", "true", "yes", "on",
    ):
        return True
    return True


async def _run_diagnosis_ollama(
    image_bytes: bytes | None,
    query: str,
    image_description: str,
    paligemma_context: str,
) -> dict:
    """
    Call Ollama /api/chat with a vision message (same transport as Gemma 3 prescan).
    One user turn bundles system instructions + clinical text so templates stay happy.
    """
    ollama_url = os.environ.get(
        "OPTIASSIST_OLLAMA_CHAT_URL",
        "http://localhost:11434/api/chat",
    )
    model = os.environ.get(
        "OPTIASSIST_MEDGEMMA_OLLAMA_MODEL",
        "medgemma",
    )
    user_text = _build_diagnosis_user_text(
        query, image_description, paligemma_context,
    )
    combined = (
        f"{SYSTEM_PROMPT}\n\n--- Clinical task ---\n\n{user_text}"
    )
    user_msg: dict = {"role": "user", "content": combined}
    if image_bytes is not None:
        user_msg["images"] = [base64.b64encode(image_bytes).decode("utf-8")]

    payload = {
        "model": model,
        "messages": [user_msg],
        "stream": False,
    }

    logger.info(
        "MedGemma — Ollama model=%s url=%s has_image=%s",
        model, ollama_url, image_bytes is not None,
    )

    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(ollama_url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    raw = (data.get("message") or {}).get("content", "") or ""
    raw = raw.strip()
    if not raw:
        logger.warning("Ollama MedGemma returned empty content")
        return dict(FALLBACK_RESULT)
    return _parse_json(raw)


def _parse_json(raw_text: str) -> dict:
    """
    Parse a JSON dict from the model's raw output string.

    Tries three progressively looser extraction strategies before
    giving up and returning FALLBACK_RESULT.
    """
    try:
        return json.loads(raw_text.strip())
    except json.JSONDecodeError:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            pass

    fence_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL
    )
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


# ── Remote inference via serve_medgemma.py ────────────────────────────

async def _run_diagnosis_remote(
    base_url: str,
    image_bytes: bytes,
    query: str,
    image_description: str,
    paligemma_context: str,
) -> dict:
    url = f"{base_url}/v1/diagnose"
    files = {"image": ("fundus.jpg", image_bytes, "image/jpeg")}
    data = {
        "prompt": query,
        "image_description": image_description,
        "paligemma_context": paligemma_context,
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(url, files=files, data=data)
        resp.raise_for_status()
        body = resp.json()
    if body.get("status") != "success":
        raise RuntimeError(body.get("message", "MedGemma API error"))
    diagnosis = body.get("diagnosis")
    if not isinstance(diagnosis, dict):
        raise RuntimeError("MedGemma API returned invalid diagnosis payload")
    return diagnosis


# ── Public entry point ────────────────────────────────────────────────

async def run_diagnosis(
    image_bytes: bytes | None,
    query: str,
    image_description: str = "",
    paligemma_context: str = "",
) -> dict:
    """
    Structured ophthalmology JSON (pipeline step 5).

    Order: optional HTTP serve → default Ollama → optional HF Transformers.
    """
    # Check for remote inference: use HTTP only when the orchestrator set
    # the URL *and* we are NOT inside the serve_medgemma server process.
    remote_url = os.environ.get("OPTIASSIST_MEDGEMMA_URL", "").rstrip("/")
    is_server = os.environ.get("_MEDGEMMA_SERVER_PROCESS", "")
    if remote_url and not is_server and image_bytes is not None:
        logger.info("MedGemma — remote inference via %s", remote_url)
        try:
            return await _run_diagnosis_remote(
                remote_url, image_bytes, query,
                image_description, paligemma_context,
            )
        except Exception as e:
            raise RuntimeError(f"MedGemma remote inference failed: {e}") from e

    if _use_ollama_medgemma():
        try:
            result = await _run_diagnosis_ollama(
                image_bytes, query, image_description, paligemma_context,
            )
            logger.info(
                "MedGemma Ollama diagnosis complete. condition=%s severity=%s",
                result.get("condition"), result.get("severity"),
            )
            return result
        except Exception as e:
            raise RuntimeError(f"MedGemma Ollama inference failed: {e}") from e

    # ── In-process HuggingFace inference ──────────────────────────────
    pil_image: Image.Image | None = None
    if image_bytes is not None:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            # Downscale large fundus images — MedGemma rescales to 448×448
            # internally; sending multi-megapixel images only wastes time in
            # the image processor.
            _MAX_SIDE = 512
            if max(pil_image.size) > _MAX_SIDE:
                pil_image.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
        except Exception as e:
            raise RuntimeError(
                f"Failed to decode image bytes into PIL Image: {e}") from e

    full_query = _build_diagnosis_user_text(
        query, image_description, paligemma_context,
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
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
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
