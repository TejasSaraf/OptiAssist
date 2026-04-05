# OpusAI

**OpusAI** is a web app for **retinal fundus analysis** aimed at diabetic retinopathy workflows. You upload an image, optionally edit a clinical question, and watch a **multi-model pipeline** stream results in real time. The UI is branded **OpusAI**; the FastAPI service still uses the internal name **OptiAssist** in a few places (API title, env prefixes).

---

## What you get

| Area | Description |
|------|-------------|
| **Landing page** | Marketing copy, pipeline overview, model cards, and privacy/speed messaging. |
| **Dashboard (`/app`)** | Upload fundus image, clinical question, **Run analysis** with live stage updates. |
| **Backend** | **`POST /api/analyze`** accepts image + question and returns **Server-Sent Events (SSE)** as each stage runs. |

---

## How the pipeline works (high level)

1. **Input** — Image and question are received.  
2. **Gemma 3** — Pre-scan of the fundus (via **Ollama**).  
3. **FunctionGemma** — Suggests routing (segmentation vs diagnosis); full pipeline can be forced with env (see below).  
4. **PaliGemma** — Vision / segmentation (HTTP server you configure, or local tooling).  
5. **MedGemma** — Structured diagnosis JSON (Ollama, remote server, or Hugging Face, depending on config).  
6. **Gemma 3 (summary)** — Merges outputs into a clinical narrative.

The dashboard labels match these stages (e.g. “Gemma 3 — Pre-scan”, “MedGemma 4B — Diagnosis”).

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
