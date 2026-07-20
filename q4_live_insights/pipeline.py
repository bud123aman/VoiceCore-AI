"""
Q4 — Live Insights and Nudges (Zero Cost)
Real-time pipeline using faster-whisper (local, free) instead of Deepgram.

Architecture:
  WAV file replayed in 1s chunks (real-time speed)
      → faster-whisper (local, CPU, tiny model — ~200ms latency per chunk)
      → Rolling transcript buffer (30s window)
      → Gemini Flash signal extraction every 5s (free tier)
      → NudgeController (confidence ≥ 0.70, 30s cooldown, dedup)
      → WebSocket broadcast → HTML dashboard

Usage:
    uvicorn pipeline:app --port 8003 --reload
    # Open http://localhost:8003
    # Then: python simulate_call.py
"""

import asyncio
import hashlib
import io
import json
import os
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import types as genai_types

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

_gemini_client = None
def get_gemini():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client

_whisper_model = None
def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        print("Loading faster-whisper model (tiny)...")
        _whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("  ✓ Whisper model loaded")
    return _whisper_model

SIGNAL_INTERVAL = 5.0        
WINDOW_SECONDS = 30          
NUDGE_COOLDOWN = 30       
CONFIDENCE_THRESHOLD = 0.70
CHUNK_SECONDS = 1.0          
SAMPLE_RATE = 16000

@dataclass
class TranscriptLine:
    speaker: str
    text: str
    timestamp: float

@dataclass
class Signal:
    signal_type: str
    description: str
    nudge: str
    confidence: float
    priority: str
    timestamp: float

@dataclass
class LatencyRecord:
    chunk_received: float = 0.0
    transcription_done: float = 0.0
    signal_extracted: float = 0.0
    nudge_delivered: float = 0.0

    @property
    def asr_ms(self): return (self.transcription_done - self.chunk_received) * 1000
    @property
    def signal_ms(self): return (self.signal_extracted - self.transcription_done) * 1000
    @property
    def total_ms(self): return (self.nudge_delivered - self.chunk_received) * 1000

class TranscriptBuffer:
    def __init__(self):
        self.lines: deque[TranscriptLine] = deque()
        self.start = time.time()

    def add(self, speaker: str, text: str):
        now = time.time() - self.start
        self.lines.append(TranscriptLine(speaker, text, now))
        cutoff = (time.time() - self.start) - WINDOW_SECONDS
        while self.lines and self.lines[0].timestamp < cutoff:
            self.lines.popleft()

    def get_text(self) -> str:
        return "\n".join(f"[{l.speaker}]: {l.text}" for l in self.lines)

    def is_empty(self): return len(self.lines) == 0

class NudgeController:
    def __init__(self):
        self.last: dict[str, float] = {}
        self.hashes: set[str] = set()
        self.log: list[dict] = []

    def should_emit(self, s: Signal) -> bool:
        if s.confidence < CONFIDENCE_THRESHOLD: return False
        if time.time() - self.last.get(s.signal_type, 0) < NUDGE_COOLDOWN: return False
        h = hashlib.md5(s.nudge[:80].encode()).hexdigest()
        if h in self.hashes: return False
        return True

    def record(self, s: Signal):
        self.last[s.signal_type] = time.time()
        self.hashes.add(hashlib.md5(s.nudge[:80].encode()).hexdigest())
        self.log.append({
            "signal_type": s.signal_type, "nudge": s.nudge,
            "confidence": s.confidence, "priority": s.priority,
            "timestamp": s.timestamp,
        })


def transcribe_chunk(audio_bytes: bytes, language: str = "en") -> tuple[str, float]:
    """
    Transcribe a raw PCM audio chunk using faster-whisper locally.
    Returns (transcript, latency_seconds).
    No API calls. No cost. Runs on CPU.
    """
    t0 = time.time()
    model = get_whisper()

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if len(audio_np) < 1600:
        return "", 0.0

    segments, info = model.transcribe(
        audio_np,
        language=language,
        beam_size=1,            
        vad_filter=True,        
        vad_parameters={"min_silence_duration_ms": 300},
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    latency = time.time() - t0
    return text, latency


SIGNAL_PROMPT = """Analyze this call transcript excerpt. Detect actionable signals if clearly present.

Transcript (last {window}s):
{transcript}

Signal types:
1. cross_sell — unaddressed customer need or product mention
2. compliance_gap — agent skipped required disclosure or made unverified claim
3. frustration — clear emotional distress or impatience from customer
4. payment_difficulty — affordability concern or inability to pay
5. missed_opportunity — agent missed a buying signal or question

Return ONLY a JSON array. Each item: {{"signal_type": str, "description": str, "nudge": str, "confidence": float, "priority": "high"|"medium"|"low"}}
If no clear signals: []
No other text."""


async def extract_signals(transcript: str) -> list[Signal]:
    if not transcript.strip():
        return []
    try:
        client = get_gemini()
        prompt = SIGNAL_PROMPT.format(window=WINDOW_SECONDS, transcript=transcript)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.1, max_output_tokens=512),
        )
        raw = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        items = json.loads(raw)
        return [
            Signal(
                signal_type=i.get("signal_type", ""),
                description=i.get("description", ""),
                nudge=i.get("nudge", ""),
                confidence=float(i.get("confidence", 0)),
                priority=i.get("priority", "low"),
                timestamp=time.time(),
            )
            for i in items if isinstance(i, dict)
        ]
    except Exception as e:
        print(f"Signal extraction error: {e}")
        return []


class CallPipeline:
    def __init__(self):
        self.buffer = TranscriptBuffer()
        self.controller = NudgeController()
        self.latency_records: list[LatencyRecord] = []  
        self.asr_latencies_ms: list[float] = []        
        self.clients: list[WebSocket] = []
        self.running = False
        self._chunk_count = 0

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.clients:
            try: await ws.send_json(msg)
            except: dead.append(ws)
        for ws in dead:
            self.clients.remove(ws)

    async def signal_loop(self):
        while self.running:
            await asyncio.sleep(SIGNAL_INTERVAL)
            if self.buffer.is_empty():
                continue
            t_start = time.time()
            signals = await extract_signals(self.buffer.get_text())
            t_extracted = time.time()

            for sig in signals:
                if self.controller.should_emit(sig):
                    self.controller.record(sig)
                    rec = LatencyRecord(
                        chunk_received=t_start - SIGNAL_INTERVAL,
                        transcription_done=t_start,
                        signal_extracted=t_extracted,
                        nudge_delivered=time.time(),
                    )
                    self.latency_records.append(rec)
                    await self.broadcast({
                        "type": "nudge",
                        "signal_type": sig.signal_type,
                        "description": sig.description,
                        "nudge": sig.nudge,
                        "confidence": round(sig.confidence, 2),
                        "priority": sig.priority,
                        "timestamp": round(sig.timestamp, 2),
                        "latency_ms": round(rec.total_ms, 1),
                    })
                    print(f"  NUDGE [{sig.priority.upper()}] {sig.signal_type}: {sig.nudge[:60]}")

    async def run_from_file(self, audio_path: str):
        """Replay WAV file at real-time speed. Uses local Whisper for ASR."""
        self.running = True
        print(f"\nPipeline started. ASR: faster-whisper (local, free)")

        get_whisper()

        sig_task = asyncio.create_task(self.signal_loop())
        chunk_size = SAMPLE_RATE * 2  

        try:
            with wave.open(audio_path, "rb") as wf:
                n = 0
                while True:
                    t_recv = time.time()
                    data = wf.readframes(SAMPLE_RATE)
                    if not data:
                        break

                    loop = asyncio.get_event_loop()
                    text, asr_lat = await loop.run_in_executor(
                        None, transcribe_chunk, data, "en"
                    )

                    if asr_lat > 0:
                        self.asr_latencies_ms.append(asr_lat * 1000)

                    if text:
                        speaker = "Agent" if n % 2 == 0 else "Customer"
                        self.buffer.add(speaker, text)
                        await self.broadcast({"type": "transcript", "speaker": speaker, "text": text})
                        print(f"  [{speaker}] (ASR {asr_lat*1000:.0f}ms): {text[:60]}")

                    n += 1
                    elapsed = time.time() - t_recv
                    await asyncio.sleep(max(0, CHUNK_SECONDS - elapsed))

        except FileNotFoundError:
            print(f"  Audio file not found: {audio_path}")
            print(f"  Running synthetic demo instead...")
            await self._synthetic_demo()

        finally:
            self.running = False
            sig_task.cancel()
            self._print_report()

    async def _synthetic_demo(self):
        """Inject synthetic transcript to test nudge pipeline without audio."""
        script = [
            ("Agent", "Hello, thank you for calling HealthShield. My name is Priya. How can I help?"),
            ("Customer", "Hi, I'm looking for health insurance for my family of four."),
            ("Agent", "Great! I can help with our family floater plans. What's your age?"),
            ("Customer", "I'm 35. We have two kids. Currently only have 1 lakh office coverage."),
            ("Agent", "Our Premium plan gives 25 lakhs coverage for the whole family."),
            ("Customer", "How much is the premium? Seems very expensive from what I've seen."),
            ("Agent", "For your profile it's around 28,000 per year, under 2,500 per month."),
            ("Customer", "That is too expensive. My salary is limited right now."),
            ("Agent", "We have a Basic plan starting at 14,000 per year for families."),
            ("Customer", "My wife has diabetes. Will that be covered?"),
            ("Agent", "Yes, but there's a 2-year waiting period for pre-existing conditions."),
            ("Customer", "We also bought a second car. Does your plan cover vehicles too?"),
            ("Agent", "Health insurance doesn't cover vehicles — that's motor insurance."),
            ("Customer", "I've been on hold for 20 minutes, I'm very frustrated right now."),
            ("Agent", "I sincerely apologize for the wait. Let me resolve this quickly."),
            ("Customer", "I want to speak with your manager immediately."),
        ]
        for speaker, text in script:
            self.buffer.add(speaker, text)
            await self.broadcast({"type": "transcript", "speaker": speaker, "text": text})
            print(f"  [{speaker}]: {text[:70]}")
            await asyncio.sleep(3)

    def _print_report(self):
        print("\n" + "="*50)
        print("LATENCY REPORT (ASR: faster-whisper local)")
        print("="*50)

        def pct(vals, label):
            if vals:
                a = np.array(vals)
                print(f"  {label}: P50={np.percentile(a,50):.0f}ms  P95={np.percentile(a,95):.0f}ms  n={len(vals)}")
            else:
                print(f"  {label}: no data")

        pct(self.asr_latencies_ms, "ASR per chunk (faster-whisper CPU)")

        records = self.latency_records
        sigs = [r.signal_ms for r in records if r.signal_ms > 0]
        totals = [r.total_ms for r in records if r.total_ms > 0]
        pct(sigs, "Signal extraction (Gemini, per nudge batch)")
        pct(totals, "End-to-end approx (chunk→nudge, per nudge batch)")

        print(f"\n  Total chunks processed: {len(self.asr_latencies_ms)}")
        print(f"  Nudges emitted: {len(self.controller.log)}")
        for n in self.controller.log:
            print(f"    [{n['signal_type']}] conf={n['confidence']:.2f} — {n['nudge'][:60]}")


app = FastAPI(title="Q4 Live Insights")
pipeline = CallPipeline()

DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Live Call Insights</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f5f5;color:#1a1a1a}
header{background:#1a1a2e;color:#fff;padding:16px 24px;display:flex;align-items:center;gap:12px}
header h1{font-size:18px;font-weight:500}
.dot{width:10px;height:10px;border-radius:50%;background:#ef4444;flex-shrink:0}
.dot.on{background:#22c55e}
.grid{display:grid;grid-template-columns:1fr 340px;gap:16px;padding:16px;height:calc(100vh - 60px)}
.panel{background:#fff;border-radius:12px;border:1px solid #e5e7eb;overflow:hidden;display:flex;flex-direction:column}
.ph{padding:10px 16px;border-bottom:1px solid #e5e7eb;font-size:12px;font-weight:500;color:#6b7280;text-transform:uppercase;letter-spacing:.04em}
.pb{flex:1;overflow-y:auto;padding:12px 16px}
.tl{padding:5px 0;font-size:14px;line-height:1.5;border-bottom:1px solid #f3f4f6}
.sp{font-weight:500;margin-right:6px}
.sp.Agent{color:#2563eb}.sp.Customer{color:#7c3aed}
.nc{border-radius:10px;padding:10px 14px;margin-bottom:8px;border:1px solid transparent;animation:si .3s ease}
@keyframes si{from{opacity:0;transform:translateY(-6px)}to{opacity:1;transform:translateY(0)}}
.nc.high{background:#fef2f2;border-color:#fca5a5}
.nc.medium{background:#fffbeb;border-color:#fcd34d}
.nc.low{background:#f0fdf4;border-color:#86efac}
.nt{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:3px}
.nc.high .nt{color:#dc2626}.nc.medium .nt{color:#d97706}.nc.low .nt{color:#16a34a}
.nb{font-size:13px;color:#1a1a1a;line-height:1.4;margin-bottom:4px}
.nm{font-size:11px;color:#9ca3af}
.cb{height:3px;border-radius:2px;background:#e5e7eb;margin-top:5px}
.cf{height:3px;border-radius:2px}
.nc.high .cf{background:#ef4444}.nc.medium .cf{background:#f59e0b}.nc.low .cf{background:#22c55e}
#nn{text-align:center;color:#9ca3af;font-size:13px;padding:20px 0}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:12px}
.sc{background:#f9fafb;border-radius:8px;padding:8px 10px}
.sl{font-size:11px;color:#9ca3af;margin-bottom:2px}
.sv{font-size:18px;font-weight:500}
.sr{font-size:11px;color:#6b7280;margin-top:2px}
</style></head>
<body>
<header><div class="dot" id="d"></div><h1>Live Call Insights — HealthShield</h1></header>
<div class="grid">
  <div class="panel">
    <div class="ph">Live transcript</div>
    <div class="pb" id="tr"></div>
  </div>
  <div style="display:flex;flex-direction:column;gap:16px">
    <div class="panel">
      <div class="ph">Stats</div>
      <div class="pb">
        <div class="sg">
          <div class="sc"><div class="sl">ASR</div><div class="sv" id="asr-engine">faster-whisper</div><div class="sr">local · free</div></div>
          <div class="sc"><div class="sl">Nudges</div><div class="sv" id="nc">0</div></div>
          <div class="sc"><div class="sl">Last latency</div><div class="sv" id="lat">—</div></div>
          <div class="sc"><div class="sl">LLM</div><div class="sv" style="font-size:13px">Gemini Flash</div><div class="sr">free tier</div></div>
        </div>
      </div>
    </div>
    <div class="panel" style="flex:1">
      <div class="ph">Agent nudges</div>
      <div class="pb" id="nd"><div id="nn">Listening for signals...</div></div>
    </div>
  </div>
</div>
<script>
let ws,n=0;
function conn(){
  ws=new WebSocket(`ws://${location.host}/ws/dashboard`);
  ws.onopen=()=>document.getElementById('d').classList.add('on');
  ws.onclose=()=>{document.getElementById('d').classList.remove('on');setTimeout(conn,2000)};
  ws.onmessage=e=>{const m=JSON.parse(e.data);m.type==='transcript'?addT(m):m.type==='nudge'?addN(m):null};
}
function addT(m){
  const el=document.createElement('div');el.className='tl';
  el.innerHTML=`<span class="sp ${m.speaker}">${m.speaker}:</span>${m.text}`;
  const b=document.getElementById('tr');b.appendChild(el);b.scrollTop=b.scrollHeight;
}
function addN(m){
  document.getElementById('nn').style.display='none';
  n++;document.getElementById('nc').textContent=n;
  if(m.latency_ms)document.getElementById('lat').textContent=m.latency_ms+'ms';
  const c=document.createElement('div');c.className=`nc ${m.priority}`;
  const p=Math.round(m.confidence*100);
  c.innerHTML=`<div class="nt">${m.signal_type.replace(/_/g,' ')}</div>
    <div class="nb">${m.nudge}</div>
    <div class="nm">Confidence ${p}% · ${m.latency_ms}ms</div>
    <div class="cb"><div class="cf" style="width:${p}%"></div></div>`;
  const b=document.getElementById('nd');b.insertBefore(c,b.firstChild);
  if(m.priority==='low')setTimeout(()=>{if(c.parentNode)c.remove()},30000);
}
conn();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard(): return DASHBOARD

@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket):
    await websocket.accept()
    pipeline.clients.append(websocket)
    try:
        while True: await asyncio.sleep(1)
    except WebSocketDisconnect:
        if websocket in pipeline.clients:
            pipeline.clients.remove(websocket)

@app.post("/simulate")
async def simulate(audio_path: str = ""):
    if pipeline.running:
        return {"error": "Pipeline already running"}
    asyncio.create_task(pipeline.run_from_file(audio_path) if audio_path else pipeline._synthetic_demo())
    return {"status": "started", "asr": "faster-whisper (local, free)"}

@app.get("/report")
async def report():
    if not pipeline.asr_latencies_ms and not pipeline.latency_records:
        return {"message": "No data. Run /simulate first."}

    def pcts(vals):
        if not vals: return {"p50": None, "p95": None, "n": 0}
        a = np.array(vals)
        return {
            "p50": round(float(np.percentile(a, 50)), 1),
            "p95": round(float(np.percentile(a, 95)), 1),
            "n": len(vals),
        }

    records = pipeline.latency_records
    sigs = [r.signal_ms for r in records if r.signal_ms > 0]
    totals = [r.total_ms for r in records if r.total_ms > 0]

    return {
        "asr_engine": "faster-whisper tiny (local CPU, free)",
        "llm_engine": "Gemini 2.0 Flash (free tier)",
        "note": (
            "ASR latency measured per chunk across ALL chunks. "
            "Signal extraction and end-to-end measured per emitted nudge."
        ),
        "latency": {
            "asr_per_chunk_ms": pcts(pipeline.asr_latencies_ms),
            "signal_extraction_ms": pcts(sigs),
            "end_to_end_approx_ms": pcts(totals),
        },
        "nudges_emitted": len(pipeline.controller.log),
        "nudge_log": pipeline.controller.log,
    }
