# AI Engineer Assessment

Four working AI systems. Zero cost. No credit card required.
<img width="882" height="553" alt="Screenshot 2026-07-20 at 9 14 01 PM" src="https://github.com/user-attachments/assets/209b4c4e-a709-4680-9c7a-936260ef3bc2" />

## Video Walkthrough

![Video Walkthrough](https://drive.google.com/file/d/10V0x--jG_g5PZott8IJib7mYhLfuFrIa/view?usp=sharing)

---

## Tech Stack

| Component | Tool | Cost |
|---|---|---|
| Voice transport | Pipecat SmallWebRTCTransport (aiortc, P2P, no external service) | ₹0 |
| Browser UI | pipecat-ai-small-webrtc-prebuilt | ₹0 |
| Vector DB | Qdrant Cloud free cluster | ₹0 |
| Embeddings | all-MiniLM-L6-v2 (local) | ₹0 |
| LLM + STT + TTS | Gemini 2.0 Flash / Gemini Live | ₹0 (1,500 req/day free) |
| ASR (Q4) | faster-whisper tiny (local CPU) | ₹0 |

**No Daily.co. No ElevenLabs. No Azure. No Retell. No Deepgram. No payment method anywhere.**

---

## Project Structure

```
ai-engineer-assessment/
├── q1_voice_agent/
│   ├── voice_agent.py          # FastAPI + SmallWebRTCTransport + Gemini Live + KB tool
│   └── gemini_proxy.py         # OpenAI-compatible Gemini proxy (alternative path)
├── q2_knowledge_base/
│   ├── kb_data.py              # 24 records (22 non-PII + 2 PII-flagged)
│   ├── ingest.py               # Embed + load into Qdrant + build BM25 index
│   └── retrieval_server.py     # FastAPI hybrid retrieval API (port 8001)
├── q3_multilingual_bots/
│   └── multilingual_agent.py   # PH + ID bots, SmallWebRTCTransport + Gemini Live
├── q4_live_insights/
│   ├── pipeline.py             # faster-whisper + Gemini signals + WebSocket dashboard (port 8003)
│   └── simulate_call.py        # Replay WAV at real-time speed
├── recordings/
│   ├── RECORDING_GUIDE.md      # Exact filenames, scenarios, and what to capture
│   ├── q1_transcript_template.txt
│   └── q4_synthetic_transcript.txt
├── .env.example
├── requirements.txt
└── README.md
```

> **⚠ Recordings required for submission.** The `recordings/` folder contains
> guide files only. You must run the agents, make the calls, and save the audio
> and transcripts yourself. See `recordings/RECORDING_GUIDE.md` for exact
> filenames and scenarios required per question.

---

## Accounts Required (all free, no card)

| Service | URL | Used for |
|---|---|---|
| Google AI Studio | https://aistudio.google.com | Gemini API key |
| Qdrant Cloud | https://cloud.qdrant.io | Free vector DB cluster |

Two accounts. Nothing else.

---

## Setup

```bash
pip install -r requirements.txt
pip install "pipecat-ai[webrtc,silero,google]" pipecat-ai-small-webrtc-prebuilt

cp .env.example .env
# Fill in: GEMINI_API_KEY, QDRANT_URL, QDRANT_API_KEY
```

---

## How the Voice Pipeline Works

### Transport

Pipecat's `SmallWebRTCTransport` (built on `aiortc`) creates a direct peer-to-peer WebRTC connection between the browser and the Python process. No external WebRTC server, no account, no billing.

### Signaling protocol (3 steps)

```
1. Browser  POST /start
           → Backend returns { sessionId, iceConfig? }

2. Browser  POST /sessions/{sessionId}/api/offer   (SDP offer)
           → Backend returns SDP answer

3. Browser  PATCH /sessions/{sessionId}/api/offer  (trickle ICE candidates)
           → Backend returns { status: "ok" }
```

`SmallWebRTCRequestHandler` from `pipecat.transports.smallwebrtc.request_handler` handles steps 2 and 3.

### Pipeline pattern (realtime / speech-to-speech)

GeminiLiveLLMService is a realtime service (STT + LLM + TTS in one). It requires the realtime pipeline pattern, which differs from the cascade (non-realtime) pattern in three ways:

| Aspect | Cascade (non-realtime) | Realtime (GeminiLive) |
|---|---|---|
| Runner | `PipelineTask` + `PipelineRunner` | `PipelineWorker` + `WorkerRunner` |
| VAD location | `TransportParams(vad_analyzer=...)` | `LLMUserAggregatorParams(vad_analyzer=...)` |
| Greeting trigger | STT → transcription starts it | `LLMRunFrame` queued on `on_client_connected` |

Correct pipeline order (from Pipecat's official `bot_realtime.py.jinja2` template):

```python
context = LLMContext()
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
)

pipeline = Pipeline([
    transport.input(),
    user_aggregator,      # NOT context_aggregator.user()
    llm,
    transport.output(),
    assistant_aggregator, # NOT context_aggregator.assistant()
])

worker = PipelineWorker(pipeline, params=PipelineParams(...))

@transport.event_handler("on_client_connected")
async def on_client_connected(transport, client):
    context.add_message({"role": "user", "content": "Start by greeting..."})
    await worker.queue_frames([LLMRunFrame()])  # triggers the greeting

runner = WorkerRunner(handle_sigint=False)
await runner.add_workers(worker)
await runner.run()
```

---

## Q2 — Knowledge Base

Run this first. Everything else depends on it.

```bash
cd q2_knowledge_base

# Step 1: Embed records into Qdrant + build BM25 index
python ingest.py

# Step 2: Start retrieval API
uvicorn retrieval_server:app --port 8001 --reload

# Step 3: Run 5 assessment test queries
curl -X POST http://localhost:8001/retrieve/test | python -m json.tool

# Step 4: Demonstrate PII protection
curl -X POST http://localhost:8001/retrieve/pii-demo | python -m json.tool
```

**Design decisions:**
- **24 records:** 22 non-PII records across 7 categories + 2 PII-flagged records (`kb_pii_001`, `kb_pii_002`) containing names, Aadhaar numbers, and bank account details
- **PII protection:** `pii=True` records excluded from all standard `/retrieve` calls via Qdrant payload filter. `/retrieve/pii-demo` returns a `PASS/FAIL` verdict
- **No chunking:** All records are 50–85 words. One record → one Qdrant point
- **Enriched embeddings:** `"{title}. Category: {category}. {content}"` — title is the strongest retrieval signal
- **Normalized BM25:** `normalize_text()` replaces hyphens with spaces (`pre-existing` → `pre existing`) before tokenizing, applied identically to documents and queries
- **Weighted score fusion:** Vector cosine (0.65) + normalized BM25 score (0.35). Replaces RRF, which discards actual similarity values
- **Zero-score exclusion:** BM25 documents with score ≤ 0 excluded from fusion pool
- **Evaluation (v2):** `correct` if expected record appears anywhere in top-3

**Known data issue:** `kb_policy_001` (Eligibility Criteria) is categorized as `qualification_rules` — should be `policy_rules`. Noted in evaluation output.

---

## Q1 — Voice Agent

```bash
# Terminal 1: KB API must be running
uvicorn q2_knowledge_base/retrieval_server:app --port 8001

# Terminal 2: Start voice agent
cd q1_voice_agent
uvicorn voice_agent:app --port 7860 --reload

# Open http://localhost:7860 — allow mic access — click Connect
# Agent greets you immediately after connection
```

**KB grounding:**

`GeminiLiveLLMService` is given a `search_knowledge_base` tool via `ToolsSchema`. When the customer asks a policy question, Gemini calls the tool. `_kb_handler` calls Q2 `/retrieve` and returns grounded context via `params.result_callback()`. Terminal shows each tool call:

```
[Tool] search_knowledge_base(query="waiting period pre-existing diabetes")
[KB]   → kb_policy_002: Waiting Periods — 2-year wait for pre-existing (Basic)
```

**Key imports (Pipecat 1.5.0):**
```python
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import WorkerRunner
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import FunctionSchema, LLMContext, ToolsSchema
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiLiveLLMSettings, GeminiModalities
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.request_handler import SmallWebRTCRequestHandler, SmallWebRTCRequest, SmallWebRTCPatchRequest, IceCandidate
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.transports.base_transport import TransportParams
```

**Required test calls (record all 5, save to `recordings/`):**
1. `q1_call_cooperative` — Full qualification flow, grounded answers
2. `q1_call_objection` — Price objection handled via `kb_objection_001`
3. `q1_call_incomplete` — Conflicting or missing customer details
4. `q1_call_outofscope` — Car insurance question → safe fallback
5. `q1_call_escalation` — Human agent request → immediate acknowledgement

---

## Q3 — Multilingual Bots

```bash
cd q3_multilingual_bots

# Philippines bot (port 7861)
uvicorn multilingual_agent:ph_app --port 7861 --reload
# Open http://localhost:7861 — click Connect

# Indonesia bot (port 7862)
uvicorn multilingual_agent:id_app --port 7862 --reload
# Open http://localhost:7862 — click Connect

# Print localization evidence (no server started)
python multilingual_agent.py --market ph --report-only
python multilingual_agent.py --market id --report-only
```

**How it works:** Gemini Live handles STT + LLM + TTS natively per language. Language set via `Language.FIL_PH` and `Language.ID_ID` enums. Same `PipelineWorker` + `WorkerRunner` pattern as Q1. Greeting triggered by `LLMRunFrame` on connect.

**Required test calls — Philippines (record 2+):**
1. `q3_ph_call_cooperative` — Full flow in Tagalog/Taglish
2. `q3_ph_call_objection` — "Mahal naman ng premium, hindi ko kaya"
3. Mixed finance terms: rider, beneficiary, coverage in Taglish
4. Colloquial: "Yung insurance ba nyo, okay ba talaga yun?"
5. Escalation: "Gusto ko ng totoong tao"

**Required test calls — Indonesia (record 2+):**
1. `q3_id_call_cooperative` — Full flow in formal Bahasa
2. `q3_id_call_regional` — Regional accent / "Gak bisa reschedule ya?"
3. Payment difficulty: "Saya lagi susah Pak, nggak bisa bayar"
4. Mixed loanwords: "DP berapa? Tenor bisa berapa bulan?"
5. Escalation: "Saya mau bicara sama supervisornya"

**Known limitations:**
- Gemini TTS output is Jakarta-dialect Indonesian only
- Regional accent ASR accuracy drops ~15–20% outside standard Indonesian
- Taglish code-switching quality depends on Gemini's multilingual training

---

## Q4 — Live Insights Pipeline

```bash
# Terminal 1: Start server + dashboard
uvicorn q4_live_insights/pipeline:app --port 8003 --reload

# Terminal 2: Open dashboard
# http://localhost:8003

# Terminal 3: Run simulation
cd q4_live_insights
python simulate_call.py                           # synthetic demo, no audio needed
python simulate_call.py ../recordings/call1.wav   # real WAV (16kHz mono PCM)

# Fetch latency report (save to recordings/q4_latency_report.json)
curl http://localhost:8003/report | python -m json.tool
```

**Pipeline:**
```
WAV file → 1-second chunks (real-time pacing)
    → faster-whisper tiny (local CPU, ~200–400ms/chunk)
    → rolling transcript buffer (30s window)
    → Gemini Flash signal extraction every 5s
    → NudgeController (confidence ≥ 0.70, 30s cooldown, dedup by hash)
    → WebSocket → HTML dashboard
```

**Latency measurement:**
- **ASR latency:** measured per chunk across **all chunks** via `pipeline.asr_latencies_ms` — reported by `/report` as `asr_per_chunk_ms`
- **Signal extraction:** measured per Gemini call (every 5s)
- **End-to-end:** chunk received → nudge delivered, per emitted nudge

**Signal types:** `cross_sell`, `compliance_gap`, `frustration`, `payment_difficulty`, `missed_opportunity`

**Expected latency:**
- ASR per chunk (CPU): P50 ≈ 200–400ms
- Signal extraction (Gemini): P50 ≈ 600–900ms
- End-to-end per nudge: P50 ≈ 1.5–2s, P95 ≈ 3s

---

## Known Limitations and Production Improvements

1. **Synthetic KB data.** Production: parse real policy PDFs with Docling or AWS Textract, 500+ records.
2. **No cross-encoder reranker.** Production: add `cross-encoder/ms-marco-MiniLM-L-6-v2` as a third retrieval stage.
3. **SmallWebRTCTransport is P2P (one user per process).** Production at scale: LiveKit (open-source SFU, self-hostable).
4. **STUN-only ICE.** Works on same-network and most home/office setups. Strict corporate NAT or Docker-on-macOS: add a self-hosted TURN server (coturn, free).
5. **faster-whisper on CPU (Q4):** ~200–400ms/chunk. Production: GPU inference or managed ASR.
6. **Speaker diarization (Q4):** Alternating-speaker heuristic. Production: WhisperX + pyannote.
7. **Gemini TTS Jakarta-dialect only (Q3):** No regional Indonesian voice on any free tier.
8. **Signal extraction every 5s (Q4):** Production: trigger on utterance-end events for ~1–2s improvement.
9. **Gemini free tier (1,500 req/day):** Sufficient for testing; production concurrent calls need paid tier.
10. **KB category issue:** `kb_policy_001` should be `policy_rules` not `qualification_rules`.
