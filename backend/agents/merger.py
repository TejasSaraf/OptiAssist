from __future__ import annotations

"""
merger.py

Merges segmentation results from PaliGemma 2, diagnosis results from MedGemma,
and cup-to-disc ratio metrics into a single unified response.

Gemma 3 (via Ollama) is used to generate the final clinical narrative summary,
synthesising all pipeline outputs into a concise 2–4 sentence paragraph.
Falls back to the raw context string if Gemma 3 is unavailable.

Pipeline role:
    1. Gemma 3 prescans the image         → prescan.py
    2. FunctionGemma decides route         → router.py
    3. PaliGemma 2 segmentation            → segmenter.py
    4. MedGemma diagnosis                  → diagnosis.py
    5. ★ THIS MODULE: Gemma 3 summary      → final unified response
"""

import base64
import logging

import httpx

logger = logging.getLogger(__name__)

DISCLAIMER = "For research use only. Not intended for clinical diagnosis."

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"

_SUMMARY_PROMPT_TEMPLATE = (
    "You are an expert ophthalmology AI assistant.\n"
    "Synthesize the following retinal analysis results into a clear, concise "
    "clinical narrative of 2–4 sentences suitable for a clinician.\n"
    "Include the diagnosis, key findings, any cup-to-disc ratio measurements "
    "with their clinical interpretation, and an overall impression.\n"
    "Write in plain prose — do NOT output JSON. Do NOT repeat the disclaimer.\n\n"
    "Clinical question: {question}\n\n"
    "Analysis results:\n{context}"
)


def _build_context(
    location: dict | None,
    diagnosis: dict | None,
    cdr_metrics: dict | None,
) -> str:
    """
    Construct a plain-text context string from all available analysis results.

    Args:
        location:    Segmentation result dict from PaliGemma 2, or None.
        diagnosis:   Diagnosis result dict from MedGemma, or None.
        cdr_metrics: Dict of CDR metric results keyed by tool name, or None.

    Returns:
        A human-readable string summarising whichever results are present.
    """
    parts: list[str] = []

    # PaliGemma first, then MedGemma — matches clinical flow for Gemma 3 summary.
    if location is not None:
        summary = location.get("summary", "")
        detections = location.get("detections", [])
        raw_snip = (location.get("raw_output") or "")[:800]
        seg_line = (
            f"PaliGemma (image): {summary} ({len(detections)} region(s) detected)"
        )
        if raw_snip.strip():
            seg_line += f"\nPaliGemma raw excerpt: {raw_snip.strip()}"
        parts.append(seg_line)

    if diagnosis is not None:
        condition = diagnosis.get("condition", "Unknown")
        severity = diagnosis.get("severity", "Unknown")
        confidence = diagnosis.get("confidence")
        findings = diagnosis.get("findings", [])
        recommendation = diagnosis.get("recommendation", "")
        findings_text = "; ".join(findings) if findings else "none recorded"
        conf_text = f" (confidence: {confidence:.0%})" if isinstance(
            confidence, float) else ""
        parts.append(
            f"MedGemma diagnosis: {condition}, severity {severity}{conf_text}. "
            f"Findings: {findings_text}. Recommendation: {recommendation}"
        )

    if cdr_metrics:
        cdr_lines: list[str] = []
        for tool_name, result in cdr_metrics.items():
            if result.get("error"):
                cdr_lines.append(f"  {tool_name}: {result['error']}")
                continue
            metric = result.get("metric", tool_name)
            value = result.get("value") or result.get("value_px")
            interp = result.get("interpretation", "")
            if value is not None:
                unit = "px" if "diameter" in metric else ""
                cdr_lines.append(
                    f"  {metric}: {value}{unit}"
                    + (f" — {interp}" if interp else "")
                )
        if cdr_lines:
            parts.append("Cup-to-disc metrics:\n" + "\n".join(cdr_lines))

    return "\n\n".join(parts) if parts else "No analysis results available."


def _determine_result_type(location: dict | None, diagnosis: dict | None) -> str:
    """
    Determine which result types are present in this response.

    Args:
        location:  Segmentation result dict, or None.
        diagnosis: Diagnosis result dict, or None.

    Returns:
        One of: "full", "location", "diagnosis".
    """
    if location is not None and diagnosis is not None:
        return "full"
    if location is not None:
        return "location"
    return "diagnosis"


async def _summarize_with_gemma3(context: str, question: str = "") -> str:
    """
    Generate a clinical narrative summary using Gemma 3 via Ollama.

    Gemma 3 synthesises the MedGemma diagnosis, PaliGemma 2 segmentation
    detections, and cup-to-disc ratio metrics into a 2–4 sentence clinical
    paragraph.

    Args:
        context:  Plain-text string describing all pipeline results
                  (built by _build_context).
        question: The original clinical question (optional, improves relevance).

    Returns:
        A concise clinical narrative string generated by Gemma 3, or the raw
        context string if Gemma 3 is unavailable.
    """
    prompt = _SUMMARY_PROMPT_TEMPLATE.format(
        question=question or "General retinal assessment",
        context=context,
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            summary = data.get("response", "").strip()

        if not summary:
            logger.warning(
                "Gemma 3 returned empty summary. Using raw context.")
            return context

        if summary.startswith("{") or summary.startswith("["):
            logger.warning(
                "Gemma 3 returned JSON instead of prose for summary. Falling back.")
            return context

        logger.info("Gemma 3 summary generated. length=%d", len(summary))
        return summary

    except Exception as e:
        logger.warning(
            "Gemma 3 summarization failed; using raw context. reason=%s", str(
                e)
        )
        return context


async def merge_results(
    location: dict | None,
    diagnosis: dict | None,
    question: str,
    cdr_metrics: dict | None = None,
) -> dict:
    """
    Merge PaliGemma 2, MedGemma, and CDR metric outputs into a unified clinical
    response.  Gemma 3 (via Ollama) generates the final narrative summary.

    Args:
        location:    Segmentation result dict from run_segmentation(), or None.
        diagnosis:   Diagnosis result dict from run_diagnosis(), or None.
        question:    The original clinical question asked by the user.
        cdr_metrics: Dict of CDR metric results from the agentic loop, or None.

    Returns:
        A dict with keys:
            "type"        (str):         "full", "location", or "diagnosis".
            "location"    (dict|None):   Raw segmentation result.
            "diagnosis"   (dict|None):   Raw diagnosis result.
            "cdr_metrics" (dict):        All computed CDR metric results.
            "summary"     (str):         Gemma 3-generated narrative summary.
            "disclaimer"  (str):         Standard research-use disclaimer.
    """
    cdr_metrics = cdr_metrics or {}
    context_string = _build_context(location, diagnosis, cdr_metrics)
    result_type = _determine_result_type(location, diagnosis)

    summary = await _summarize_with_gemma3(
        context=context_string, question=question
    )

    return {
        "type": result_type,
        "location": location,
        "diagnosis": diagnosis,
        "cdr_metrics": cdr_metrics,
        "summary": summary,
        "disclaimer": DISCLAIMER,
    }
