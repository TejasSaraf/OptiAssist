"""
OptiAssist Backend — 6-Stage Linear Pipeline

    1. input        → Image + prompt accepted
    2. gemma3       → Gemma 3 pre-scans the retinal image  (Ollama)
    3. routing      → FunctionGemma decides the analysis route  (Ollama)
    4. paligemma    → Fine-tuned PaliGemma analyses the image
    5. medgemma     → MedGemma 4B produces structured diagnosis
    6. synthesis    → Gemma 3 merges everything into a clinical narrative  (Ollama)

Inference delegation (set in backend/.env):

    OPTIASSIST_PALIGEMMA_URL=http://localhost:8080   → serve_paligemma.py
    OPTIASSIST_MEDGEMMA_URL=http://localhost:8081     → serve_medgemma.py
"""

import sys
import os
import json
import asyncio
import logging
import traceback
from pathlib import Path

from dotenv import load_dotenv

# Load .env BEFORE importing agents — they read OPTIASSIST_* at call time.
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import httpx
import uvicorn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.prescan import prescan_image
from agents.segmenter import run_segmentation
from agents.diagnosis import run_diagnosis
from agents.merger import merge_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="OptiAssist Backend API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Ollama config ────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/chat"
FUNCTIONGEMMA_MODEL = "functiongemma"

# ── SSE helper ───────────────────────────────────────────────────────

def _sse(event: str, data) -> str:
    payload = json.dumps(data) if isinstance(data, dict) else json.dumps({"message": data})
    return f"event: {event}\ndata: {payload}\n\n"

# ── Health check ─────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "OptiAssist API is running"}

# ── FunctionGemma single-turn routing ────────────────────────────────

async def _route_with_functiongemma(question: str, image_description: str) -> str:
    """
    Ask FunctionGemma which analysis route to take.
    Returns a short routing decision string.
    """
    prompt = (
        f"You are a medical routing agent. Based on the clinical question and "
        f"image description, decide the analysis tools to invoke.\n\n"
        f"Question: {question}\n"
        f"Image description: {image_description}\n\n"
        f"Available tools:\n"
        f"  - run_segmentation: PaliGemma 2 — retinal structure analysis\n"
        f"  - run_diagnosis: MedGemma 4B — structured medical diagnosis\n\n"
        f"Reply with ONLY the tool names to call, separated by commas."
    )
    payload = {
        "model": FUNCTIONGEMMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
            logger.info("FunctionGemma route decision: %s", content[:200])
            return content.strip()
    except Exception as e:
        logger.warning("FunctionGemma routing failed: %s — defaulting to full pipeline", e)
        return "run_segmentation, run_diagnosis"


def _parse_route_flags(route: str) -> tuple[bool, bool]:
    """
    Decide whether to run PaliGemma and/or MedGemma from FunctionGemma's
    comma-separated tool reply. Default to both if neither is mentioned.
    """
    rl = (route or "").lower()
    want_seg = "run_segmentation" in rl or "segmentation" in rl
    want_diag = "run_diagnosis" in rl or "diagnosis" in rl
    if not want_seg and not want_diag:
        return True, True
    return want_seg, want_diag

# ── Main pipeline endpoint ───────────────────────────────────────────

@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    question: str = Form(
        "Analyze this retinal fundus image. What condition is present and what is the severity?"
    ),
):
    image_bytes = await file.read()

    async def stream():
        queue: asyncio.Queue = asyncio.Queue()

        async def emit(event: str, data):
            await queue.put((event, data))

        async def pipeline():
            results = {
                "prescan": None,
                "routing": None,
                "segmentation": None,
                "diagnosis": None,
                "synthesis": None,
            }

            try:
                # ── 1. Input ─────────────────────────────────────────
                await emit("stage", {
                    "id": "input", "status": "complete",
                    "message": "Image and prompt received",
                })

                # ── 2. Gemma 3 Pre-scan ──────────────────────────────
                await emit("stage", {
                    "id": "gemma3", "status": "running",
                    "message": "Gemma 3 scanning the retinal image…",
                })
                description = await prescan_image(image_bytes)
                results["prescan"] = description
                await emit("stage", {
                    "id": "gemma3", "status": "complete",
                    "message": "Pre-scan complete",
                    "data": description,
                })

                # ── 3. FunctionGemma Routing ─────────────────────────
                await emit("stage", {
                    "id": "routing", "status": "running",
                    "message": "FunctionGemma deciding analysis route…",
                })
                route = await _route_with_functiongemma(question, description)
                results["routing"] = route
                run_seg, run_diag = _parse_route_flags(route)
                await emit("stage", {
                    "id": "routing", "status": "complete",
                    "message": f"Route: {route[:100]}",
                    "data": {
                        "raw": route,
                        "run_segmentation": run_seg,
                        "run_diagnosis": run_diag,
                    },
                })

                # ── 4. Fine-tuned PaliGemma ──────────────────────────
                paligemma_ctx = ""
                location = None
                if run_seg:
                    await emit("stage", {
                        "id": "paligemma", "status": "running",
                        "message": "PaliGemma analyzing retinal structures…",
                    })
                    try:
                        location = await run_segmentation(image_bytes, question)
                        results["segmentation"] = location
                        if location and location.get("detections"):
                            paligemma_ctx = json.dumps(
                                location["detections"], indent=2)
                        await emit("stage", {
                            "id": "paligemma", "status": "complete",
                            "message": "PaliGemma analysis complete",
                            "data": {
                                "summary": location.get("summary", "") if location else "",
                                "raw_output": (location.get("raw_output", "") if location else "")[:500],
                                "detections_count": len(location.get("detections", [])) if location else 0,
                            },
                        })
                    except Exception as e:
                        logger.error("PaliGemma failed: %s", e)
                        await emit("stage", {
                            "id": "paligemma", "status": "error",
                            "message": f"PaliGemma error: {e}",
                        })
                else:
                    await emit("stage", {
                        "id": "paligemma", "status": "skipped",
                        "message": "Skipped per FunctionGemma route",
                    })

                # ── 5. MedGemma 4B Diagnosis ─────────────────────────
                diagnosis = None
                if run_diag:
                    await emit("stage", {
                        "id": "medgemma", "status": "running",
                        "message": "MedGemma 4B generating medical diagnosis…",
                    })
                    try:
                        diagnosis = await run_diagnosis(
                            image_bytes=image_bytes,
                            query=question,
                            image_description=description or "",
                            paligemma_context=paligemma_ctx,
                        )
                        results["diagnosis"] = diagnosis
                        await emit("stage", {
                            "id": "medgemma", "status": "complete",
                            "message": "Diagnosis complete",
                            "data": diagnosis,
                        })
                    except Exception as e:
                        logger.error("MedGemma failed: %s", e)
                        await emit("stage", {
                            "id": "medgemma", "status": "error",
                            "message": f"MedGemma error: {e}",
                        })
                else:
                    await emit("stage", {
                        "id": "medgemma", "status": "skipped",
                        "message": "Skipped per FunctionGemma route",
                    })

                # ── 6. Gemma 3 Synthesis ─────────────────────────────
                await emit("stage", {
                    "id": "synthesis", "status": "running",
                    "message": "Gemma 3 synthesizing clinical narrative…",
                })
                try:
                    merged = await merge_results(location, diagnosis, question)
                    summary = merged.get("summary", "") if merged else "No summary available."
                    results["synthesis"] = summary
                    await emit("stage", {
                        "id": "synthesis", "status": "complete",
                        "message": "Summary generated",
                        "data": summary,
                    })
                except Exception as e:
                    logger.error("Synthesis failed: %s", e)
                    results["synthesis"] = "Synthesis unavailable."
                    await emit("stage", {
                        "id": "synthesis", "status": "error",
                        "message": f"Synthesis error: {e}",
                    })

                # ── Done ─────────────────────────────────────────────
                await emit("complete", {"results": results})

            except Exception as e:
                logger.error("Pipeline error: %s\n%s", e, traceback.format_exc())
                await emit("error", {"message": str(e)})
            finally:
                await queue.put(None)

        asyncio.create_task(pipeline())

        while True:
            item = await queue.get()
            if item is None:
                break
            event, data = item
            yield _sse(event, data)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
