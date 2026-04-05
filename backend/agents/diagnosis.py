from __future__ import annotations

"""
diagnosis.py

MedGemma diagnosis agent — uses Google's MedGemma 4B (google/medgemma-4b-it)
via HuggingFace Transformers to produce a structured medical diagnosis for a
retinal fundus image.

Runtime: HuggingFace Transformers pipeline("image-text-to-text")
Model:   google/medgemma-4b-it

When OPTIASSIST_MEDGEMMA_URL is set (e.g. http://localhost:8081), inference
is delegated to the dedicated serve_medgemma.py server via HTTP. Otherwise
the pipeline is loaded in-process.

Pipeline role:
    1. Gemma 3 prescans the image          → image_description
    2. FunctionGemma decides route          → run_diagnosis called
    3. PaliGemma 2 runs segmentation        → paligemma_context (if applicable)
    4. ★ THIS AGENT: MedGemma diagnosis     → structured JSON result
    5. Gemma 3 synthesises final summary    → merger.py
"""

import asyncio
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
_medgemma_4bit = False      # tracks whether we loaded with quantization
_medgemma_finetuned = False  # tracks whether a LoRA adapter was applied

# Default adapter path — override with OPTIASSIST_MEDGEMMA_ADAPTER env var
_DEFAULT_ADAPTER_PATH = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "medgemma" / "final"


def _load_medgemma():
    """Load MedGemma via HuggingFace Transformers, with optional 4-bit quantization.

    When ``OPTIASSIST_MEDGEMMA_4BIT`` is ``"1"`` (the default) **and** CUDA is
    available, the model is loaded with NF4 4-bit quantization via
    ``BitsAndBytesConfig``.  This cuts VRAM from ~8 GB to ~2.5 GB and speeds up
    inference on consumer GPUs.

    On MPS (Apple Silicon) or CPU — where ``bitsandbytes`` 4-bit is not
    supported — the function silently falls back to full-precision loading.
    """
    global _medgemma_pipe, _medgemma_4bit, _medgemma_finetuned

    if _medgemma_pipe is not None:
        return

    from pathlib import Path as _Path

    import torch
    from dotenv import load_dotenv
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor, pipeline

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
        model = Gemma3ForConditionalGeneration.from_pretrained(
            _HF_MODEL_ID,
            token=hf_token,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # ── Check for fine-tuned LoRA adapter ────────────────────────
        adapter_path = os.environ.get(
            "OPTIASSIST_MEDGEMMA_ADAPTER", str(_DEFAULT_ADAPTER_PATH))
        if Path(adapter_path).exists() and (Path(adapter_path) / "adapter_model.safetensors").exists():
            from peft import PeftModel
            logger.info("Loading fine-tuned LoRA adapter from %s", adapter_path)
            model = PeftModel.from_pretrained(model, adapter_path)
            _medgemma_finetuned = True
            logger.info("LoRA adapter loaded successfully.")
        else:
            logger.info("No LoRA adapter found at %s — using base model.", adapter_path)

        _medgemma_pipe = pipeline(
            "image-text-to-text",
            model=model,
            processor=processor,
        )
        _medgemma_4bit = True
        logger.info("MedGemma pipeline loaded successfully (4-bit quantized%s).",
                     " + fine-tuned" if _medgemma_finetuned else "")

    else:
        # ── Full-precision loading (MPS / CPU / 4-bit disabled) ──────
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

        if torch.cuda.is_available():
            dtype = torch.bfloat16
            device_map = "auto"
        elif mps_available:
            dtype = torch.float16
            device_map = "auto"
        else:
            dtype = torch.float32
            device_map = None

        logger.info(
            "Loading MedGemma pipeline from %s (dtype=%s, device_map=%s)",
            _HF_MODEL_ID, dtype, device_map,
        )
        processor = AutoProcessor.from_pretrained(
            _HF_MODEL_ID, token=hf_token, use_fast=True,
        )

        model_or_id: any = _HF_MODEL_ID
        extra_kwargs: dict = {
            "token": hf_token,
            "torch_dtype": dtype,
            "device_map": device_map,
        }

        # ── Check for fine-tuned LoRA adapter ────────────────────────
        adapter_path = os.environ.get(
            "OPTIASSIST_MEDGEMMA_ADAPTER", str(_DEFAULT_ADAPTER_PATH))
        if Path(adapter_path).exists() and (Path(adapter_path) / "adapter_model.safetensors").exists():
            from peft import PeftModel
            logger.info("Loading base model for adapter attachment...")

            base_model_kwargs: dict = {
                "token": hf_token,
                "torch_dtype": dtype,
            }
            post_load_device: str | None = None

            if torch.cuda.is_available():
                base_model_kwargs["device_map"] = device_map
            else:
                # PEFT currently crashes when Gemma 3 is loaded through
                # device_map="auto" and some layers are disk-offloaded/meta.
                base_model_kwargs["device_map"] = "cpu"
                if device_map == "auto":
                    logger.info(
                        "Loading MedGemma on CPU first to avoid the Gemma 3 "
                        "PEFT offload bug triggered by device_map='auto'."
                    )
                if mps_available:
                    post_load_device = "mps"

            base_model = Gemma3ForConditionalGeneration.from_pretrained(
                _HF_MODEL_ID,
                **base_model_kwargs,
            )
            logger.info("Loading fine-tuned LoRA adapter from %s", adapter_path)
            model_or_id = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                low_cpu_mem_usage=False,
            )
            if post_load_device is not None:
                logger.info("Moving MedGemma + LoRA adapter to %s for inference.", post_load_device.upper())
                model_or_id = model_or_id.to(post_load_device)
            _medgemma_finetuned = True
            extra_kwargs = {}  # model already loaded, no extra kwargs needed
            logger.info("LoRA adapter loaded successfully.")
        else:
            logger.info("No LoRA adapter found at %s — using base model.", adapter_path)

        _medgemma_pipe = pipeline(
            "image-text-to-text",
            model=model_or_id,
            processor=processor,
            **extra_kwargs,
        )
        _medgemma_4bit = False
        logger.info("MedGemma pipeline loaded successfully (full precision%s).",
                     " + fine-tuned" if _medgemma_finetuned else "")


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
    Produce a structured ophthalmological diagnosis using MedGemma.

    Routes to the HTTP server at OPTIASSIST_MEDGEMMA_URL when set (and
    we're not inside serve_medgemma itself), otherwise runs in-process.
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

    # ── In-process inference ──────────────────────────────────────────
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
