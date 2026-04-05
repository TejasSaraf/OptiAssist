"""
Serve fine-tuned MedGemma (google/medgemma-4b-it + LoRA) on Modal with FastAPI.

Loads 4-bit NF4 base weights from cache and PEFT adapters from volume
``OpusAI-checkpoints`` at ``/checkpoints/medgemma/final`` (same layout as
training).

Prerequisites
    pip install modal
    modal setup
    modal secret create hf-secret HF_TOKEN=hf_...

Deploy (persistent HTTPS endpoint; no fixed host port — Modal terminates TLS)
    modal deploy backend/scripts/modal_serve_medgemma.py

Interactive / dev (temporary URL)
    modal serve backend/scripts/modal_serve_medgemma.py

Point your stack at the deployment URL + path, e.g. set in ``.env``::

    OpusAI_MEDGEMMA_URL=https://<workspace>--OpusAI-medgemma-serve-serve.modal.run

(Exact hostname is printed after ``modal deploy``; include no trailing slash.)
"""

from __future__ import annotations

import modal

app = modal.App("OpusAI-medgemma-serve")

model_cache = modal.Volume.from_name(
    "OpusAI-model-cache", create_if_missing=True
)
checkpoints_vol = modal.Volume.from_name(
    "OpusAI-checkpoints", create_if_missing=True
)

serve_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "transformers>=4.52.0",
        "peft>=0.12.0",
        "accelerate>=0.34.0",
        "Pillow>=10.0.0",
        "bitsandbytes>=0.43.0",
        "fastapi>=0.115.0",
        "uvicorn[standard]>=0.30.0",
        "python-multipart>=0.0.9",
        "huggingface_hub>=0.23.0",
    )
    .env({"HF_HOME": "/model_cache", "TRANSFORMERS_VERBOSITY": "warning"})
)

MODEL_ID = "google/medgemma-4b-it"
ADAPTER_DIR = "/checkpoints/medgemma/final"


@app.function(
    image=serve_image,
    gpu="L40S",
    volumes={
        "/model_cache": model_cache,
        "/checkpoints": checkpoints_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    scaledown_window=600,
    timeout=60 * 30,
)
@modal.concurrent(max_inputs=1)
@modal.asgi_app(label="medgemma")
def serve():
    import asyncio
    import io
    import json
    import logging
    import os
    import re
    from contextlib import asynccontextmanager

    import torch
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from peft import PeftModel
    from PIL import Image
    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        Gemma3ForConditionalGeneration,
        pipeline,
    )

    log = logging.getLogger("medgemma-serve")

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
        "  findings: list of strings — MUST be non-empty.\n"
        "  recommendation: string — clinical follow-up advice\n"
        "  disclaimer: string — always exactly: "
        "'For research use only. Not intended for clinical diagnosis.'\n"
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

    def _parse_json(raw_text: str) -> dict:
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
        log.warning("JSON parse failed; raw (400 chars): %s", raw_text[:400])
        return dict(FALLBACK_RESULT)

    def _decode_generated(output: list) -> str:
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

    _pipe = None
    _quantized_4bit = True
    _has_lora = False

    @asynccontextmanager
    async def lifespan(web: FastAPI):
        nonlocal _pipe, _has_lora
        hf_token = os.environ.get("HF_TOKEN")
        log.info("Loading %s (4-bit) …", MODEL_ID)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_ID, token=hf_token, use_fast=True
        )
        base = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            token=hf_token,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        adapter_cfg = os.path.join(ADAPTER_DIR, "adapter_config.json")
        if os.path.isfile(adapter_cfg):
            model = PeftModel.from_pretrained(base, ADAPTER_DIR)
            _has_lora = True
            log.info("Merged PEFT adapters from %s", ADAPTER_DIR)
        else:
            model = base
            _has_lora = False
            log.warning(
                "No adapter at %s — serving base model only (train or upload LoRA).",
                ADAPTER_DIR,
            )
        _pipe = pipeline(
            "image-text-to-text",
            model=model,
            tokenizer=processor.tokenizer,
            image_processor=processor.image_processor,
        )
        log.info("MedGemma pipeline ready (4-bit=%s, lora=%s).", True, _has_lora)
        yield

    web = FastAPI(title="MedGemma Modal Inference", lifespan=lifespan)
    web.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": MODEL_ID,
            "quantized_4bit": _quantized_4bit,
            "lora_loaded": _has_lora,
        }

    @web.post("/v1/diagnose")
    async def diagnose(
        image: UploadFile = File(...),
        prompt: str = Form(
            "Analyze this retinal fundus image. What condition is present and what is the severity?"
        ),
        image_description: str = Form(""),
        paligemma_context: str = Form(""),
    ):
        if _pipe is None:
            return {"status": "error", "message": "Model not loaded yet"}

        try:
            image_bytes = await image.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            max_side = 512
            if max(pil_image.size) > max_side:
                pil_image.thumbnail((max_side, max_side), Image.LANCZOS)

            full_query = prompt
            if image_description:
                full_query = (
                    f"{prompt}\n\n"
                    f"Pre-scan image description (from Gemma 3):\n{image_description}"
                )
            if paligemma_context:
                full_query = (
                    f"{full_query}\n\n"
                    f"PaliGemma 2 segmentation results:\n{paligemma_context}"
                )

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": full_query},
                    ],
                },
            ]

            def _infer():
                out = _pipe(text=messages, max_new_tokens=256, do_sample=False)
                return _decode_generated(out)

            raw_text = await asyncio.to_thread(_infer)
            diagnosis = _parse_json(raw_text)
            return {
                "status": "success",
                "model": MODEL_ID,
                "quantized_4bit": _quantized_4bit,
                "lora": _has_lora,
                "request_prompt": prompt,
                "diagnosis": diagnosis,
            }
        except Exception as e:
            log.exception("diagnose failed")
            return {"status": "error", "message": repr(e)}

    return web
