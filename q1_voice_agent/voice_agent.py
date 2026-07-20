"""
Q1 — Voice Agent (SmallWebRTCTransport + Gemini Live + KB grounding)

Signaling protocol (pipecat-ai-small-webrtc-prebuilt):
  1. Browser  POST /start                      → { sessionId, iceConfig }
  2. Browser  POST /sessions/{id}/api/offer    → SDP answer
  3. Browser  PATCH /sessions/{id}/api/offer   → ICE candidates

Pipeline pattern (realtime / speech-to-speech):
  Uses PipelineWorker + WorkerRunner (not PipelineTask + PipelineRunner).
  VAD goes in LLMUserAggregatorParams, not TransportParams.
  Greeting is triggered by queuing LLMRunFrame on_client_connected.

KB grounding: search_knowledge_base FunctionSchema tool registered on
GeminiLiveLLMService. Handler calls Q2 /retrieve API and returns grounded
context via result_callback.

Usage:
    pip install "pipecat-ai[webrtc,silero,google]" pipecat-ai-small-webrtc-prebuilt
    uvicorn voice_agent:app --port 7860 --reload
    Open http://localhost:7860 and click Connect
"""

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import (
    FunctionSchema,
    LLMContext,
    ToolsSchema,
)
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.google.gemini_live.llm import (
    GeminiLiveLLMService,
    GeminiLiveLLMSettings,
    GeminiModalities,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import (
    IceCandidate,
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
KB_API_URL = os.getenv("KB_API_URL", "http://localhost:8001")

SYSTEM_PROMPT = """You are Priya, a friendly health insurance advisor for HealthShield Insurance.

Your role: qualify leads for HealthShield health insurance plans.

Rules:
- Keep responses to 1-2 sentences — this is a voice call.
- ALWAYS call search_knowledge_base before answering any question about coverage,
  premiums, waiting periods, claims, eligibility, or policy rules.
- If the KB returns nothing: "I'd need to check that with our specialists."
- If the customer wants a human: "Let me connect you with a senior advisor right now."
- Never state policy details you haven't retrieved from the knowledge base.
- Do not read out record IDs or citations to the customer.

Qualification to collect: age, family size, current insurance, health conditions, interest level.
"""


# ── KB tool ────────────────────────────────────────────────────────────────────

async def _kb_handler(params: FunctionCallParams) -> None:
    """Called by Gemini when it invokes search_knowledge_base."""
    query = params.arguments.get("query", "")
    category = params.arguments.get("category")

    try:
        payload: dict[str, Any] = {"query": query, "top_k": 3}
        if category:
            payload["category_filter"] = category

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{KB_API_URL}/retrieve", json=payload)
            resp.raise_for_status()
            results = resp.json().get("results", [])

        if not results:
            result_text = (
                "No relevant information found. "
                "Tell the customer you will check with a specialist."
            )
        else:
            result_text = "\n\n---\n\n".join(
                f"[{r['record_id']}] {r['title']}\n{r['content']}"
                for r in results
            )
    except Exception as e:
        result_text = (
            f"Knowledge base unavailable ({e}). "
            "Tell the customer you will check with a specialist."
        )

    await params.result_callback(result_text)


KB_TOOL = FunctionSchema(
    name="search_knowledge_base",
    description=(
        "Search HealthShield's knowledge base for product details, policy rules, "
        "FAQs, objection responses, eligibility criteria, and claim procedures. "
        "Call this before answering ANY question about coverage, premiums, waiting "
        "periods, claims, exclusions, or eligibility."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "The customer's question or topic to look up.",
        },
        "category": {
            "type": "string",
            "enum": [
                "product_overview",
                "policy_rules",
                "faq",
                "objection_handling",
                "qualification_rules",
                "escalation_rules",
                "network_info",
            ],
            "description": "Optional: narrow results to a specific category.",
        },
    },
    required=["query"],
    handler=_kb_handler,
)


# ── Pipeline ───────────────────────────────────────────────────────────────────

async def run_voice_pipeline(webrtc_connection: SmallWebRTCConnection) -> None:
    """
    Run one voice session using the canonical realtime pipeline pattern.

    Key differences from cascade (non-realtime) pattern:
    - PipelineWorker + WorkerRunner instead of PipelineTask + PipelineRunner
    - SileroVADAnalyzer in LLMUserAggregatorParams, NOT in TransportParams
    - LLMContextAggregatorPair unpacked to named variables (user_aggregator,
      assistant_aggregator) and referenced directly in the Pipeline list
    - Greeting triggered by queuing LLMRunFrame on_client_connected
    - modalities=GeminiModalities.AUDIO (scalar, not a list)
    """

    # Transport: no VAD here for realtime services
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            # No vad_analyzer here — it belongs in LLMUserAggregatorParams
        ),
    )

    # Gemini Live: STT + LLM + TTS in one service
    settings = GeminiLiveLLMSettings(
        voice="Puck",
        modalities=GeminiModalities.AUDIO,   # scalar, NOT a list
        system_instruction=SYSTEM_PROMPT,
    )

    llm = GeminiLiveLLMService(
        api_key=GEMINI_API_KEY,
        settings=settings,
        tools=ToolsSchema(standard_tools=[KB_TOOL]),
    )

    # Context + aggregators
    # VAD goes here (in user_params), not in TransportParams
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # Canonical realtime pipeline order (from Pipecat's own template):
    # input → user_aggregator → llm → output → assistant_aggregator
    pipeline = Pipeline([
        transport.input(),
        user_aggregator,
        llm,
        transport.output(),
        assistant_aggregator,
    ])

    # PipelineWorker is the correct runner for realtime (speech-to-speech) services
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        print("Client connected — starting voice session")
        # Seed the greeting and queue LLMRunFrame to trigger Gemini's first response
        context.add_message({
            "role": "user",
            "content": "Start by greeting the customer warmly and asking how you can help.",
        })
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        print("Client disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


# ── FastAPI app ────────────────────────────────────────────────────────────────

_active_sessions: dict[str, dict] = {}
_webrtc_handler = SmallWebRTCRequestHandler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await _webrtc_handler.close()


app = FastAPI(title="HealthShield Voice Agent", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/prebuilt", SmallWebRTCPrebuiltUI)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/prebuilt/")


@app.post("/start")
async def start(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}

    session_id = str(uuid.uuid4())
    _active_sessions[session_id] = body

    result: dict[str, Any] = {"sessionId": session_id}
    if body.get("enableDefaultIceServers"):
        result["iceConfig"] = {
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    return JSONResponse(result)


@app.post("/sessions/{session_id}/api/offer")
async def session_offer(
    session_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
):
    if session_id not in _active_sessions:
        return JSONResponse({"error": "Unknown session_id"}, status_code=404)

    data = await request.json()
    webrtc_request = SmallWebRTCRequest(
        sdp=data["sdp"],
        type=data["type"],
        pc_id=data.get("pc_id"),
        restart_pc=data.get("restart_pc"),
        request_data=data.get("requestData") or _active_sessions.get(session_id),
    )

    async def connection_callback(connection: SmallWebRTCConnection):
        background_tasks.add_task(run_voice_pipeline, connection)

    answer = await _webrtc_handler.handle_web_request(
        request=webrtc_request,
        webrtc_connection_callback=connection_callback,
    )
    return JSONResponse(answer)


@app.patch("/sessions/{session_id}/api/offer")
async def session_ice(session_id: str, request: Request):
    data = await request.json()
    patch_request = SmallWebRTCPatchRequest(
        pc_id=data["pc_id"],
        candidates=[IceCandidate(**c) for c in data.get("candidates", [])],
    )
    await _webrtc_handler.handle_patch_request(patch_request)
    return JSONResponse({"status": "ok"})


@app.post("/api/offer")
async def direct_offer(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    session_id = data.get("pc_id") or str(uuid.uuid4())
    _active_sessions[session_id] = {}

    webrtc_request = SmallWebRTCRequest(
        sdp=data["sdp"],
        type=data["type"],
        pc_id=data.get("pc_id"),
        restart_pc=data.get("restart_pc"),
        request_data=data.get("requestData"),
    )

    async def connection_callback(connection: SmallWebRTCConnection):
        background_tasks.add_task(run_voice_pipeline, connection)

    answer = await _webrtc_handler.handle_web_request(
        request=webrtc_request,
        webrtc_connection_callback=connection_callback,
    )
    return JSONResponse(answer)


@app.patch("/api/offer")
async def direct_ice(request: Request):
    data = await request.json()
    patch_request = SmallWebRTCPatchRequest(
        pc_id=data["pc_id"],
        candidates=[IceCandidate(**c) for c in data.get("candidates", [])],
    )
    await _webrtc_handler.handle_patch_request(patch_request)
    return JSONResponse({"status": "ok"})
