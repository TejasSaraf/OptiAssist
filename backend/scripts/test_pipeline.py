"""
test_pipeline.py

End-to-end test of the OpusAI retinal analysis pipeline.

Tests each stage independently so you can see which parts are working
and which need setup:

    Stage 1: Prescan        (Gemma 3 via Ollama)
    Stage 2: Router         (FunctionGemma via Ollama)
    Stage 3: Segmentation   (PaliGemma 2 local — if model available)
    Stage 4: Diagnosis      (MedGemma via HuggingFace Transformers)
    Stage 5: Merger         (Gemma 3 via Ollama)

Usage:
    # Full pipeline test on a retinal image
    python backend/scripts/test_pipeline.py --image data/dr_unified_v2/test/0/31182_left.jpg

    # Skip stages that require missing models (test what's available)
    python backend/scripts/test_pipeline.py --image data/dr_unified_v2/test/0/31182_left.jpg --skip-missing

    # Test only specific stages
    python backend/scripts/test_pipeline.py --image data/dr_unified_v2/test/0/31182_left.jpg --stage prescan
    python backend/scripts/test_pipeline.py --image data/dr_unified_v2/test/0/31182_left.jpg --stage diagnosis
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


sys.path.insert(0, str(PROJECT_ROOT / "backend"))


async def check_prerequisites() -> dict:
    """Check which components are available and return a status dict."""
    status = {}

    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            status["ollama"] = True
            status["ollama_models"] = models
    except Exception:
        status["ollama"] = False
        status["ollama_models"] = []

    model_names_lower = [m.lower().split(":")[0]
                         for m in status["ollama_models"]]
    status["gemma3"] = any("gemma3" in m for m in model_names_lower)
    status["functiongemma"] = any(
        "functiongemma" in m for m in model_names_lower)

    paligemma_segmenter_path = (
        PROJECT_ROOT / "backend" / "models" / "paligemma-finetuned"
    )
    status["paligemma_segmenter"] = paligemma_segmenter_path.exists()

    paligemma_dr_path = PROJECT_ROOT / "backend" / "models" / "paligemma-finetuned"
    status["paligemma_dr"] = paligemma_dr_path.exists()

    try:
        import transformers
        status["medgemma"] = True
        status["transformers_version"] = transformers.__version__
    except ImportError:
        status["medgemma"] = False

    return status


def print_prerequisites(status: dict):
    """Print a formatted prerequisites table."""
    print("\n" + "=" * 60)
    print("  PREREQUISITES CHECK")
    print("=" * 60)

    def icon(ok): return "Yes" if ok else "No"

    print(f"\n  {icon(status['ollama'])} Ollama server running")
    if status["ollama"]:
        print(f"     Models: {', '.join(status['ollama_models']) or 'none'}")

    print(f"\n  Ollama Models Required:")
    print(f"  {icon(status['gemma3'])} gemma3:4b        (prescan + summary)")
    print(f"  {icon(status['functiongemma'])} functiongemma    (router)")

    print(f"\n  Local Models:")
    print(
        f"  {icon(status['paligemma_dr'])} PaliGemma DR grading     (backend/models/paligemma-finetuned/)")
    print(
        f"  {icon(status['paligemma_segmenter'])} PaliGemma analysis       (backend/models/paligemma-finetuned/)")

    print(f"\n  HuggingFace:")
    print(
        f"  {icon(status['medgemma'])} MedGemma 4B              (google/medgemma-4b-it — downloads on first use)")

    can_test = []
    cannot_test = []

    if status["ollama"] and status["gemma3"]:
        can_test.append("prescan")
        can_test.append("merger (summary)")
    else:
        cannot_test.append("prescan (need: ollama + gemma3:4b)")
        cannot_test.append("merger (need: ollama + gemma3:4b)")

    if status["ollama"] and status["functiongemma"]:
        can_test.append("router")
    else:
        cannot_test.append("router (need: ollama + functiongemma)")

    if status["paligemma_segmenter"]:
        can_test.append("segmenter")
    else:
        cannot_test.append("segmenter (need: paligemma fine-tuned model)")

    if status["medgemma"]:
        can_test.append("diagnosis")
    else:
        cannot_test.append("diagnosis (need: transformers)")

    print(f"\n  Can test:    {', '.join(can_test) if can_test else 'nothing'}")
    if cannot_test:
        print(f"  Cannot test: {', '.join(cannot_test)}")

    missing_cmds = []
    if not status["ollama"]:
        missing_cmds.append("# Start Ollama server:\n  ollama serve")
    if status["ollama"] and not status["gemma3"]:
        missing_cmds.append("# Pull Gemma 3:\n  ollama pull gemma3:4b")
    if status["ollama"] and not status["functiongemma"]:
        missing_cmds.append(
            "# Pull FunctionGemma:\n  ollama pull functiongemma")

    if missing_cmds:
        print(f"\n  Setup commands for missing components:")
        for cmd in missing_cmds:
            print(f"  {cmd}")

    print("=" * 60)
    return can_test


async def test_prescan(image_bytes: bytes) -> str | None:
    """Stage 1: Test Gemma 3 prescan."""
    print("\n" + "─" * 60)
    print("  STAGE 1: PRESCAN (Gemma 3 via Ollama)")
    print("─" * 60)

    try:
        from agents.prescan import prescan_image

        t0 = time.time()
        description = await prescan_image(image_bytes)
        elapsed = time.time() - t0

        print(f"  Prescan complete ({elapsed:.1f}s)")
        print(f"  Description: {description[:200]}")
        return description

    except Exception as e:
        print(f"  Prescan failed: {e}")
        return None


async def test_diagnosis(image_bytes: bytes, image_description: str = "", paligemma_context: str = "") -> dict | None:
    """Stage 4: Test MedGemma diagnosis."""
    print("\n" + "─" * 60)
    print("  STAGE 4: DIAGNOSIS (MedGemma via HuggingFace)")
    print("─" * 60)

    try:
        from agents.diagnosis import run_diagnosis

        query = "Analyze this retinal fundus image. What condition is present and what is the severity?"

        t0 = time.time()
        result = await run_diagnosis(
            image_bytes=image_bytes,
            query=query,
            image_description=image_description,
            paligemma_context=paligemma_context,
        )
        elapsed = time.time() - t0

        print(f"  Diagnosis complete ({elapsed:.1f}s)")
        print(f"  Condition:  {result.get('condition')}")
        print(f"  Severity:   {result.get('severity')}")
        print(f"  Confidence: {result.get('confidence')}")
        print(f"  Findings:   {result.get('findings', [])}")
        print(f"  Recommend:  {result.get('recommendation', '')[:100]}")
        return result

    except Exception as e:
        print(f"  Diagnosis failed: {e}")
        return None


async def test_segmentation(image_bytes: bytes) -> dict | None:
    """Stage 3: Test PaliGemma 2 segmentation."""
    print("\n" + "─" * 60)
    print("  STAGE 3: SEGMENTATION (PaliGemma 2 local)")
    print("─" * 60)

    try:
        from agents.segmenter import run_segmentation

        t0 = time.time()
        result = await run_segmentation(image_bytes)
        elapsed = time.time() - t0

        detections = result.get("detections", [])
        labels = [d.get("label", "?") for d in detections]

        print(f"  Segmentation complete ({elapsed:.1f}s)")
        print(f"  Detections: {len(detections)} — {labels}")
        print(f"  Summary:    {result.get('summary', '')[:150]}")
        return result

    except (FileNotFoundError, ImportError) as e:
        print(f"  ⚠️  Segmentation model not available: {e}")
        print(
            f"     This is expected if you haven't set up the PaliGemma 2 detection model.")
        return None
    except Exception as e:
        print(f"  Segmentation failed: {e}")
        return None


async def test_merger(
    location: dict | None,
    diagnosis: dict | None,
    question: str,
) -> dict | None:
    """Stage 5: Test Gemma 3 summary in merger."""
    print("\n" + "─" * 60)
    print("  STAGE 5: MERGER (Gemma 3 summary via Ollama)")
    print("─" * 60)

    try:
        from agents.merger import merge_results

        t0 = time.time()
        result = await merge_results(
            location=location,
            diagnosis=diagnosis,
            question=question,
        )
        elapsed = time.time() - t0

        print(f"  Merge complete ({elapsed:.1f}s)")
        print(f"  Type:    {result.get('type')}")
        print(f"  Summary: {result.get('summary', '')[:300]}")
        return result

    except Exception as e:
        print(f"  Merger failed: {e}")
        return None


async def test_router_standalone(image_description: str) -> None:
    """Stage 2: Test FunctionGemma routing (standalone, no callbacks)."""
    print("\n" + "─" * 60)
    print("  STAGE 2: ROUTER (FunctionGemma via Ollama)")
    print("─" * 60)

    try:
        import httpx

        from agents.router import (
            OLLAMA_URL, OLLAMA_MODEL, TOOLS, DEVELOPER_MESSAGE,
            _ORCHESTRATION_INSTRUCTIONS,
        )

        question = "What is the severity of diabetic retinopathy in this image?"
        user_content = f"{question}\n\nImage context: {image_description}"
        user_content += _ORCHESTRATION_INSTRUCTIONS

        messages = [
            {"role": "developer", "content": DEVELOPER_MESSAGE},
            {"role": "user", "content": user_content},
        ]

        payload = {
            "model": OLLAMA_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "stream": False,
        }

        t0 = time.time()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(OLLAMA_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
        elapsed = time.time() - t0

        message = data.get("message", {})
        tool_calls = message.get("tool_calls", [])
        text = message.get("content", "")

        if tool_calls:
            tools_called = [tc["function"]["name"] for tc in tool_calls]
            print(f"  Router responded with tool calls ({elapsed:.1f}s)")
            print(f"  Tools called: {tools_called}")
        elif text:
            print(
                f"  Router returned text instead of tool calls ({elapsed:.1f}s)")
            print(f"  Text: {text[:200]}")
        else:
            print(f"  Router returned empty response ({elapsed:.1f}s)")

    except Exception as e:
        print(f"  Router failed: {e}")


async def run_full_pipeline(image_bytes: bytes, skip_missing: bool = False):
    """Run the full pipeline end-to-end."""
    print("\n" + "=" * 60)
    print("  FULL PIPELINE TEST")
    print("=" * 60)

    question = "Analyze this retinal fundus image. What condition is present and what is the severity?"

    description = await test_prescan(image_bytes)
    if description is None and not skip_missing:
        print("\n  Pipeline halted — prescan failed (run with --skip-missing to continue)")
        return

    await test_router_standalone(description or "Retinal fundus image")

    location = await test_segmentation(image_bytes)

    paligemma_ctx = ""
    if location:
        paligemma_ctx = json.dumps(location.get("detections", []), indent=2)
    diagnosis = await test_diagnosis(
        image_bytes, image_description=description or "", paligemma_context=paligemma_ctx
    )

    if diagnosis is not None or location is not None:
        await test_merger(location, diagnosis, question)
    else:
        print("\n  ⚠️  Skipping merger — no diagnosis or segmentation results to merge")

    print("\n" + "=" * 60)
    print("  PIPELINE TEST COMPLETE")
    print("=" * 60)


async def async_main():
    parser = argparse.ArgumentParser(description="Test OpusAI pipeline")
    parser.add_argument("--image", "-i", required=True,
                        help="Path to retinal image")
    parser.add_argument("--stage", "-s", default=None,
                        choices=["prescan", "router", "segmenter",
                                 "diagnosis", "merger", "all"],
                        help="Test only a specific stage (default: all)")
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip stages that fail and continue the pipeline")
    parser.add_argument("--skip-prereqs", action="store_true",
                        help="Skip prerequisites check")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        log.error("Image not found: %s", image_path)
        sys.exit(1)
    image_bytes = image_path.read_bytes()
    print(f"\n  Image: {image_path} ({len(image_bytes) / 1024:.0f} KB)")

    if not args.skip_prereqs:
        status = await check_prerequisites()
        can_test = print_prerequisites(status)

    stage = args.stage or "all"

    if stage == "prescan":
        await test_prescan(image_bytes)
    elif stage == "router":
        await test_router_standalone("Retinal fundus image with visible pathology")
    elif stage == "segmenter":
        await test_segmentation(image_bytes)
    elif stage == "diagnosis":
        await test_diagnosis(image_bytes)
    elif stage == "merger":
        mock_diagnosis = {
            "condition": "Diabetic Retinopathy",
            "severity": "Moderate",
            "severity_level": 2,
            "confidence": 0.85,
            "findings": ["microaneurysms", "hard exudates"],
            "recommendation": "Refer to ophthalmologist.",
        }
        await test_merger(None, mock_diagnosis, "What is the diagnosis?")
    elif stage == "all":
        await run_full_pipeline(image_bytes, skip_missing=args.skip_missing)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()