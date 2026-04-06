# OpusAI 👁️

> **Fully local AI diagnostic assistant for ophthalmologists.** Patient data never leaves the clinic.

OpusAI is a multi-model AI pipeline that analyzes retinal fundus images and answers clinical questions in real time — all on-device, with zero cloud dependency. It combines four specialized Google Gemma family models orchestrated through a 6-stage pipeline, streaming live progress updates to the clinician as each analysis step completes.

---

## Demo

[![Watch the demo]](https://youtu.be/5FZsRmOI36k?si=f8SB4pgFSNyYh-Tx)

- **Image:** `/data/dr_unified_v2_sampled/test/0/134_right.jpg`
- **Question:** `How severe is the diabetic retinopathy in this image?`

---

## Key Features

- **100% Local Inference** — HIPAA/GDPR-friendly; images and patient data never leave the device
- **Multi-Model Pipeline** — Four specialized AI models working in concert: FunctionGemma, Gemma 3, PaliGemma 2, and MedGemma
- **Real-Time Streaming** — Server-Sent Events push live progress updates for every pipeline stage
- **Intelligent Routing** — FunctionGemma 270M autonomously decides which models to invoke based on the clinical question
- **Optic Structure Segmentation** — Fine-tuned PaliGemma 2 detects optic disc and optic cup bounding boxes for diabetic retenopathy
- **Structured Diagnosis** — Structured diagnosis JSON (Ollama, remote server, or Hugging Face, depending on config).  
- **Summary** - Gemma 3 generates summary and merges output into a clinical narrative.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (React.js)                      │
│                                                             │
│  /  Landing Page          /demo  Interactive Demo           │
│     Hero, Problem,               Input Panel (upload +      │
│     HowItWorks, TechStack        question) + SSE feed       │
└──────────────────────────────┬──────────────────────────────┘
                               │  POST /analyze (multipart)
                               │  ← SSE Stream (text/event-stream)
┌──────────────────────────────▼──────────────────────────────┐
│                  BACKEND (FastAPI + Uvicorn)                │
│                                                             │
│  Stage 1  Input Validation                                  │
│      ↓                                                      │
│  Stage 2  prescan.py  →  Gemma 3 4B (Ollama)                │
│      ↓                                                      │
│  Stage 3  router.py      →  FunctionGemma 270M (Ollama)     │
│      ↓                                                      │
│  Stage 4  segmenter.py   →  PaliGemma 2 3B (HuggingFace)    │
│      ↓                                                      │
│  Stage 5  diagnosis.py   →  MedGemma 4B (HuggingFace)       │
│      ↓                                                      │
│  Stage 6  merger.py      →  Gemma 3 4B (Ollama)             │
│                                                             │
│  Local Model Runtime:                                       │
│  · Ollama (localhost:11434): Gemma 3, FunctionGemma         │
│  · HuggingFace Transformers: PaliGemma 2, MedGemma          │
└─────────────────────────────────────────────────────────────┘
```

## Models

| Model | Parameters | Runtime | Role |
|-------|-----------|---------|------|
| **Gemma 3 4B** | 4B | Ollama | Image pre-scan description + final narrative summary |
| **FunctionGemma** | 270M | Ollama | Intelligent routing via function calling |
| **PaliGemma 2** *(custom QLoRA)* | 3B | HuggingFace Transformers | Optic disc/cup bounding box segmentation |
| **MedGemma** | 4B | HuggingFace Transformers | Ophthalmic disease diagnosis + structured report |

### 🔬 Custom Fine-Tuned PaliGemma 2

The PaliGemma 2 model used in OptiAssist is **our own LoRA fine-tune**, trained specifically for retinal optic structure detection (optic disc and optic cup bounding boxes).

| Training Detail | Value |
|----------------|-------|
| Base model | `google/paligemma-3b-pt-224` |
| Fine-tuning method | QLoRA (4-bit NF4 quantization via BitsAndBytes) |
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs | 8 (with Early Stopping patience of 3)|
| Batch size | 4 |
| Input resolution | 224x224|
| Task | Object detection via tokens |

The adapter weights (`adapter_model.safetensors`) are loaded at runtime via `peft.PeftModel`, then merged into the base model for inference. Weights are stored locally under `backend/models/paligemma-finetuned/`.

---

## Tech stack

| Layer | Technologies |
|-------|----------------|
| **Frontend** | React 19, TypeScript, Vite 8, Tailwind CSS 4, React Router 7, Lucide React |
| **Backend** | Python, FastAPI, Uvicorn, HTTPX, Pillow; optional **PyTorch / Transformers** for HF MedGemma and training PaliGemma 3b / LoRA |
| **Inference** | **Ollama** (Gemma 3, FunctionGemma, MedGemma chat paths), optional dedicated servers (`serve_paligemma.py`, `serve_medgemma.py`) |

---

## Repository layout

```
OpusAI/
├── frontend/          # Vite + React app
│   ├── src/
│   │   ├── LandingPage.tsx
│   │   ├── AppPage.tsx          # wraps Dashboard
│   │   └── components/
│   │       └── dashboard.tsx    # SSE client, stage UI
│   └── .env.example             # VITE_API_BASE_URL
│
└── backend/
    ├── main.py                  # FastAPI app, /api/analyze, SSE
    ├── agents/                  # prescan, segmenter, diagnosis, merger, router
    ├── serve_paligemma.py       # optional PaliGemma HTTP service (default :8080)
    ├── serve_medgemma.py        # optional MedGemma HTTP service (default :8081)
    ├── scripts/                 # training, eval, Modal helpers, pipeline tests
    └── requirements.txt         # ML + HTTP client deps (see setup note below)
```

---

## Prerequisites

- **Node.js** 18+ (for the frontend)  
- **Python** 3.10+ (for the backend)  
- **Ollama** running locally (or reachable URL) with models you actually use (e.g. Gemma 3, FunctionGemma, MedGemma tags as configured)  
- Optional: **CUDA** / GPU for faster local HF inference; **PaliGemma** server if you point `OPTIASSIST_PALIGEMMA_URL` at it  

---

## Quick start

### 1. Frontend

```bash
cd frontend
cp .env.example .env          # edit VITE_API_BASE_URL if the API is not on localhost:8000
npm install
npm run dev
```

Open the URL Vite prints (often `http://localhost:5173`).  
- **`/`** — landing page  
- **`/app`** — analysis dashboard  

### 2. Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`main.py` imports **FastAPI**, **Uvicorn**, and **python-multipart** (for file uploads). If anything is missing after `pip install -r requirements.txt`, install explicitly:

```bash
pip install "fastapi>=0.115" "uvicorn[standard]>=0.30" python-multipart
```

Create **`backend/.env`** (there is no committed `.env.example`; use the variables below as a checklist). Then run:

```bash
# From backend/ (same directory as main.py)
python main.py
# or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Health check: **`GET /`** → `{"status": "OptiAssist API is running"}`.

### 3. Point the UI at the API

In `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

---

## Environment variables (backend)

All backend env vars use the **`OPTIASSIST_`** prefix (legacy name). Common ones:

| Variable | Purpose |
|----------|---------|
| `OPTIASSIST_OLLAMA_CHAT_URL` | Ollama chat API (default `http://localhost:11434/api/chat`) — pre-scan, FunctionGemma, MedGemma (Ollama path). |
| `OPTIASSIST_OLLAMA_GENERATE_URL` | Used for the final **Gemma 3 summary** step in the merger. |
| `OPTIASSIST_GEMMA3_PRESCAN_MODEL` | Ollama model tag for pre-scan (e.g. `gemma3:4b`). |
| `OPTIASSIST_FUNCTIONGEMMA_MODEL` | FunctionGemma model tag (default `functiongemma`). |
| `OPTIASSIST_MEDGEMMA_OLLAMA_MODEL` | MedGemma Ollama tag (default `medgemma`). |
| `OPTIASSIST_PALIGEMMA_URL` | Base URL for PaliGemma HTTP service (e.g. `http://localhost:8080`). |
| `OPTIASSIST_MEDGEMMA_URL` | Optional HTTP MedGemma server (`serve_medgemma.py`, often port **8081**). |
| `OPTIASSIST_MEDGEMMA_BACKEND` | Set to `hf` (or related flags in `diagnosis.py`) for Hugging Face in-process MedGemma. |
| `OPTIASSIST_FULL_PIPELINE` | `1` (default) = always run PaliGemma + MedGemma; `0` = respect FunctionGemma routing hints only. |

For MedGemma backends, device selection, and HF model IDs, see **`backend/agents/diagnosis.py`** (docstring at top).

---

## API: analyze (SSE)

**`POST /api/analyze`**

- **Form fields:** `file` (image), `question` (string).  
- **Response:** `text/event-stream` with JSON payloads per **stage** (`input`, `gemma3`, `routing`, `paligemma`, `medgemma`, `synthesis`) and a final completion event with aggregated results.

The dashboard builds a `FormData` request and reads the stream with `fetch` + `ReadableStream`.

---

## Optional services

| Script | Role |
|--------|------|
| `serve_paligemma.py` | Standalone FastAPI app for PaliGemma-style vision calls (default port **8080** in file). |
| `serve_medgemma.py` | Standalone MedGemma HTTP API (default port **8081**). |

Start them when you set the matching `OPTIASSIST_*_URL` values in `.env`.

---

## Training & evaluation

Under **`backend/scripts/`** you’ll find helpers for PaliGemma training/eval, Modal jobs, adapter merge, and **`test_pipeline.py`** for integration-style checks. Read each script’s header before running; they often assume paths, GPUs, or cloud secrets.

---

## Disclaimer

This project is built for learning and demo contexts (e.g. **ScarletHacks**). Retinal imaging and any “diagnosis” output are **not** a substitute for professional medical advice, regulatory clearance, or your own compliance review (HIPAA, GDPR, institutional policy, etc.). The landing page includes product messaging about privacy and on-device use—**verify** those claims against your actual deployment. Future scope will be to test the system with real clinicians.


---

## Credits

**Built for ScarletHacks** · **© 2026 OpusAI**
