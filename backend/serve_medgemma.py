import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.diagnosis import run_diagnosis, _load_medgemma, _medgemma_4bit


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Prevent diagnosis.py from HTTP-calling back into this process when
    # OPTIASSIST_MEDGEMMA_URL is in .env (used by the orchestrator only).
    os.environ["_MEDGEMMA_SERVER_PROCESS"] = "1"
    print("Loading MedGemma 4B into active memory... (This may take a few minutes)")
    _load_medgemma()
    from agents.diagnosis import _medgemma_4bit as q4
    q_label = "4-bit NF4 quantized" if q4 else "full precision"
    print(f"MedGemma successfully loaded and bound to VRAM ({q_label}).")
    yield


app = FastAPI(title="MedGemma Local Inference Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/v1/diagnose")
async def diagnose(
    image: UploadFile = File(...),
    prompt: str = Form("Analyze this retinal fundus image. What condition is present and what is the severity?"),
    image_description: str = Form(""),
    paligemma_context: str = Form(""),
):
    try:
        image_bytes = await image.read()

        diagnosis_result = await run_diagnosis(
            image_bytes=image_bytes,
            query=prompt,
            image_description=image_description,
            paligemma_context=paligemma_context,
        )

        from agents.diagnosis import _medgemma_4bit as q4
        return {
            "status": "success",
            "model": "google/medgemma-4b-it",
            "quantized_4bit": q4,
            "request_prompt": prompt,
            "diagnosis": diagnosis_result,
        }

    except Exception as e:
        return {"status": "error", "message": repr(e)}


if __name__ == "__main__":
    print("Starting MedGemma dedicated API server on Port 8081...")
    uvicorn.run("serve_medgemma:app", host="0.0.0.0", port=8081, reload=False)
