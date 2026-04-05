from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io
import torch
from PIL import Image
from pathlib import Path

from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "backend" / "models" / "paligemma-finetuned"

model = None
processor = None
device = "cpu"
dtype = torch.float32


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, device, dtype
    print(f"Loading PaliGemma model from {MODEL_PATH}...")

    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16

    processor = PaliGemmaProcessor.from_pretrained(str(MODEL_PATH), use_fast=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=dtype,
        device_map="auto" if device != "mps" else None,
    )
    if device == "mps":
        model = model.to(device)

    print(f"Model successfully loaded onto {device.upper()} in memory.")
    yield


app = FastAPI(title="PaliGemma Local Inference Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


@app.post("/v1/generate")
async def generate(
    image: UploadFile = File(...),
    prompt: str = Form("Analyze this retinal fundus image. What condition is present?"),
    max_tokens: int = Form(256),
):
    try:
        image_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(
            text=prompt,
            images=pil_img,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_len:]
        raw_output = processor.decode(generated_ids, skip_special_tokens=False)
        result_text = processor.decode(generated_ids, skip_special_tokens=True).strip()

        return {
            "status": "success",
            "model": "paligemma-finetuned",
            "prompt": prompt,
            "response": result_text,
            "raw_output": raw_output,
        }

    except Exception as e:
        return {"status": "error", "message": repr(e)}


if __name__ == "__main__":
    print("Starting PaliGemma dedicated API server...")
    uvicorn.run("serve_paligemma:app", host="0.0.0.0", port=8080, reload=False)
