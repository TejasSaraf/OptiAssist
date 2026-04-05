from __future__ import annotations

"""
router.py

Implements a multi-turn FunctionGemma agentic loop for OpusAI.

FunctionGemma maintains a messages conversation history and autonomously decides
which tools to call and in what order, until it either returns a plain text
response or explicitly calls the 'finish' tool.

Message format follows the official FunctionGemma spec:
  - System prompt uses role "developer"
  - Tool results use role "tool" with content [{"name": ..., "response": ...}]
  - After a tool call the assistant message (with tool_calls) is appended before
    the tool result, preserving the correct alternating turn structure.

Reference: https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma
"""

import json
import logging
import os
from typing import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get(
    "OpusAI_OLLAMA_CHAT_URL",
    "http://localhost:11434/api/chat",
)
OLLAMA_MODEL = os.environ.get("OpusAI_FUNCTIONGEMMA_MODEL", "functiongemma")

MAX_LOOP_ITERATIONS = 8
BYPASS_FUNCTIONGEMMA = False

MAX_NUDGE_RETRIES = 2

DEVELOPER_MESSAGE = (
    "You are an ophthalmology pipeline routing agent specialized in diabetic "
    "retinopathy analysis. You do NOT analyze images yourself. Your only job "
    "is to call the provided tools in the correct order to route the image "
    "through the diagnostic pipeline.\n\n"
    "You have access to these tools:\n"
    "- run_diagnosis: sends the image to MedGemma for medical diagnosis\n"
    "- run_segmentation: sends the image to PaliGemma for structural analysis\n"
    "- finish: signals workflow completion\n\n"
    "IMPORTANT: Never refuse. Never say you cannot assist. Never describe your "
    "limitations. You are a routing agent — just call the tools."
)

_ORCHESTRATION_INSTRUCTIONS = (
    "\n\n---\n"
    "MANDATORY RULES — follow exactly, no exceptions:\n"
    "1. NEVER output text. ONLY call tools. Do not explain, apologize, or refuse.\n"
    "2. ALWAYS call run_diagnosis first.\n"
    "3. ALWAYS call run_segmentation after run_diagnosis.\n"
    "4. ALWAYS call finish last.\n"
    "5. The correct sequence is: run_diagnosis → run_segmentation → finish.\n"
    "Start now: call run_diagnosis."
)

_NUDGE_MESSAGE = (
    "You are a routing agent. You do NOT analyze images or describe limitations. "
    "You responded with text instead of calling a tool — this is forbidden. "
    "You MUST call run_diagnosis right now. "
    "Do not output any text — only call the run_diagnosis tool."
)


_NUDGE_SEGMENTATION_MESSAGE = (
    "You responded with text instead of calling run_segmentation. "
    "This is not allowed. "
    "The question is about cup-to-disc ratio / optic disc / glaucoma. "
    "You MUST call run_segmentation right now — do NOT output text."
)


_SEGMENTATION_KEYWORDS = frozenset({
    "optic disc", "optic cup", "cup to disc", "cup-to-disc",
    "cdr", "glaucoma", "disc cupping", "neuroretinal rim",
    "cup disc", "disc cup",
})


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_diagnosis",
            "description": (
                "Run MedGemma 4B — Google's general-purpose medical AI model — "
                "to analyze the retinal fundus image and produce a structured "
                "medical diagnosis. Returns condition, severity (None/Mild/Moderate/"
                "Severe/Proliferative), confidence score, a list of specific findings, "
                "and a clinical recommendation. Always call this tool first before "
                "any other tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_segmentation",
            "description": (
                "Run PaliGemma 2 to detect optic disc and optic cup bounding boxes "
                "from the retinal fundus image. "
                "Call this whenever the question asks about: optic disc, optic cup, "
                "cup-to-disc ratio, CDR, glaucoma, disc cupping, or neuroretinal rim. "
                "Must be called after run_diagnosis and before finish."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Signal that the analysis workflow is complete and all required "
                "tools have been called. Gemma 3 will then produce a final "
                "clinical narrative summarising all outputs. Always call this "
                "as the last step."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


_TOOL_TO_ROUTE: dict[str, str] = {
    "run_diagnosis": "analyze_diagnosis",
    "run_segmentation": "analyze_location",
    "finish": "analyze_diagnosis",
}


def _needs_segmentation(question: str) -> bool:
    """Return True if the question requires PaliGemma segmentation."""
    q = question.lower()
    return any(kw in q for kw in _SEGMENTATION_KEYWORDS)


async def _call_functiongemma(messages: list[dict]) -> dict:
    """
    Send the current messages history to FunctionGemma via Ollama.

    Args:
        messages: Full conversation history, correctly alternating user / assistant
                  / tool turns.

    Returns:
        The "message" object from Ollama's response (contains "content" and/or
        "tool_calls").

    Raises:
        httpx.HTTPStatusError: If Ollama returns a non-2xx response.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
    return data.get("message", {})


async def run_agentic_loop(
    question: str,
    image_description: str,
    run_diagnosis_cb: Callable[[], Awaitable[dict]],
    run_segmentation_cb: Callable[[str], Awaitable[dict]],
    emit: Callable[[str, str], Awaitable[None]],
) -> dict:
    """
    Run the FunctionGemma multi-turn agentic loop.

    FunctionGemma autonomously decides which tools to invoke and in what order
    within the same conversation context, until it either returns a plain-text
    answer or calls the 'finish' tool.

    Args:
        question:             The clinician's question.
        image_description:    Pre-scanned image description from the prescanner.
        run_diagnosis_cb:     Async callback that runs MedGemma diagnosis;
                              emits medgemma_start / medgemma_complete internally.
        run_segmentation_cb:  Async callback(query: str) that runs PaliGemma 2
                              segmentation; emits paligemma_start / paligemma_complete
                              internally.
        emit:                 Async SSE emit callback — called as
                              await emit(event: str, message: str).

    Returns:
        Dict with keys:
            "diagnosis"  (dict | None): MedGemma result, or None if not run.
            "location"   (dict | None): PaliGemma result, or None if not run.
            "final_text" (str):         FunctionGemma's last plain-text response.
    """

    user_content = f"{question}"
    if image_description:
        user_content = f"{question}\n\nImage context: {image_description}"
    user_content += _ORCHESTRATION_INSTRUCTIONS

    messages: list[dict] = [
        {
            "role": "developer",
            "content": DEVELOPER_MESSAGE,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    if BYPASS_FUNCTIONGEMMA:
        logger.info(
            "BYPASS mode: skipping FunctionGemma AND MedGemma, running PaliGemma only.")
        await emit("route_decided", "Route: analyze_location")

        location: dict | None = None
        seg_query = "detect optic-disc ; optic-cup"
        try:
            location = await run_segmentation_cb(seg_query)
        except Exception as exc:
            logger.error("bypass run_segmentation_cb raised: %s", exc)

        return {
            "diagnosis": None,
            "location": location,
            "final_text": "",
        }

    diagnosis: dict | None = None
    location: dict | None = None
    final_text: str = ""
    route_decided_emitted = False
    nudge_count = 0

    for iteration in range(MAX_LOOP_ITERATIONS):
        logger.info(
            "Agentic loop — iteration %d/%d", iteration + 1, MAX_LOOP_ITERATIONS
        )

        try:
            message = await _call_functiongemma(messages)
        except Exception as exc:
            logger.warning(
                "FunctionGemma call failed at iteration %d: %s", iteration + 1, exc
            )
            break

        tool_calls: list[dict] = message.get("tool_calls") or []
        text_content: str = message.get("content") or ""

        if not tool_calls:
            if diagnosis is None and nudge_count < MAX_NUDGE_RETRIES:

                nudge_count += 1
                logger.warning(
                    "FunctionGemma returned text without calling a tool "
                    "(iteration %d, nudge %d/%d). Injecting reminder.",
                    iteration + 1,
                    nudge_count,
                    MAX_NUDGE_RETRIES,
                )
                if text_content:
                    messages.append(
                        {"role": "assistant", "content": text_content})
                messages.append({"role": "user", "content": _NUDGE_MESSAGE})
                continue

            if (
                diagnosis is not None
                and location is None
                and _needs_segmentation(question)
                and nudge_count < MAX_NUDGE_RETRIES
            ):

                nudge_count += 1
                logger.warning(
                    "FunctionGemma skipped run_segmentation for a CDR question "
                    "(iteration %d, nudge %d/%d). Injecting segmentation reminder.",
                    iteration + 1,
                    nudge_count,
                    MAX_NUDGE_RETRIES,
                )
                if text_content:
                    messages.append(
                        {"role": "assistant", "content": text_content})
                messages.append(
                    {"role": "user", "content": _NUDGE_SEGMENTATION_MESSAGE})
                continue

            final_text = text_content
            logger.info(
                "FunctionGemma returned text (no tool calls). "
                "nudge_count=%d diagnosis_done=%s segmentation_done=%s. Exiting loop.",
                nudge_count,
                diagnosis is not None,
                location is not None,
            )
            break

        messages.append(message)

        tool_results: list[dict] = []
        should_finish = False

        for call in tool_calls:
            fn_name: str = call["function"]["name"]
            fn_args: dict = call["function"].get("arguments") or {}

            if not fn_name:
                logger.warning(
                    "FunctionGemma returned a tool_call with an empty function name; skipping.")
                continue

            logger.info(
                "FunctionGemma called tool=%s args=%s", fn_name, fn_args
            )

            if not route_decided_emitted and fn_name in _TOOL_TO_ROUTE:
                route_name = _TOOL_TO_ROUTE[fn_name]
                await emit("route_decided", f"Route: {route_name}")
                logger.info("Route decided: %s", route_name)
                route_decided_emitted = True

            if fn_name == "finish":
                should_finish = True
                tool_results.append(
                    {"name": "finish", "response": "Analysis complete."})

            elif fn_name == "run_diagnosis":
                try:
                    diagnosis = await run_diagnosis_cb()
                    tool_results.append({
                        "name": "run_diagnosis",
                        "response": json.dumps(diagnosis),
                    })
                except Exception as exc:
                    logger.error("run_diagnosis_cb raised: %s", exc)
                    tool_results.append({
                        "name": "run_diagnosis",
                        "response": f"Tool execution failed: {exc}",
                    })

            elif fn_name == "run_segmentation":

                query: str = "detect optic-disc ; optic-cup"
                try:
                    location = await run_segmentation_cb(query)
                    detections_out = location.get("detections", [])

                    raw_out = location.get("raw_output", "")[:200]
                    labels_found = [d.get("label", "<empty>")
                                    for d in detections_out]
                    logger.info(
                        "PaliGemma detections=%d  labels=%s  raw_output=%r",
                        len(detections_out), labels_found, raw_out,
                    )
                    tool_results.append({
                        "name": "run_segmentation",
                        "response": json.dumps({
                            "summary": location.get("summary", ""),
                            "detections_count": len(detections_out),
                            "detections": detections_out,
                        }),
                    })
                except (FileNotFoundError, ImportError) as exc:

                    logger.error(
                        "run_segmentation_cb: model permanently unavailable, skipping: %s", exc)
                    location = None
                    should_finish = True
                    tool_results.append({
                        "name": "run_segmentation",
                        "response": (
                            "Segmentation model is permanently unavailable. "
                            "Skip this tool and call finish immediately."
                        ),
                    })
                except Exception as exc:
                    logger.error("run_segmentation_cb raised: %s", exc)
                    tool_results.append({
                        "name": "run_segmentation",
                        "response": f"Tool execution failed: {exc}",
                    })

            else:
                logger.warning(
                    "FunctionGemma called unknown tool: %s", fn_name)
                tool_results.append({
                    "name": fn_name,
                    "response": f"Unknown tool: {fn_name}",
                })

        messages.append({
            "role": "tool",
            "content": json.dumps(tool_results),
        })

        if should_finish:
            logger.info("FunctionGemma called finish. Exiting agentic loop.")
            break

    else:

        logger.warning(
            "Agentic loop reached maximum iterations (%d). Stopping.", MAX_LOOP_ITERATIONS
        )

    if not route_decided_emitted:
        await emit("route_decided", "Route: analyze_diagnosis")

    return {
        "diagnosis": diagnosis,
        "location": location,
        "final_text": final_text,
    }