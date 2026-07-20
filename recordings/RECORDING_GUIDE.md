# Recording Guide

All call recordings must be placed in this folder before submission.
The assessment requires recordings as evidence — code alone is insufficient.

---

## Q1 — Required recordings (minimum 3, aim for 5)

Record each scenario as a separate file. Use any screen/audio recorder
(OBS, QuickTime, Loom). Record the browser tab showing the prebuilt UI.

| File | Scenario | What to demonstrate |
|---|---|---|
| `q1_call_cooperative.mp3` | Cooperative customer | Full qualification flow, KB-grounded answers with citations in logs |
| `q1_call_objection.mp3` | Price objection | Agent uses `kb_objection_001` to respond, not invented text |
| `q1_call_incomplete.mp3` | Conflicting/incomplete details | Agent asks clarifying questions, handles gracefully |
| `q1_call_outofscope.mp3` | Out-of-scope question (car insurance) | Safe fallback: "That's outside what I can help with" |
| `q1_call_escalation.mp3` | Human escalation request | Immediate transfer acknowledgement |

**How to verify KB grounding:**
The server terminal shows `search_knowledge_base` tool calls as they happen:
```
[TOOL] search_knowledge_base(query="waiting period pre-existing")
[KB]   → kb_policy_002: Waiting Periods — What Is and Is Not Covered
```
Include a screenshot of these logs in your submission.

---

## Q1 — Transcripts

Save each call transcript as a `.txt` file. The prebuilt UI shows the
conversation in the browser. Copy and paste into a text file after each call.

Format:
```
Agent: Hello! Thank you for calling HealthShield Insurance...
Customer: Hi, I want to know about health insurance for my family...
Agent: I'd be happy to help. May I ask how many family members...
...
Result: [HIGH_PRIORITY / MEDIUM_PRIORITY / LOW_PRIORITY]
KB records used: [list record IDs from server logs]
```

---

## Q3 — Required recordings (2 per market)

| File | Market | Scenario |
|---|---|---|
| `q3_ph_call_cooperative.mp3` | Philippines | Cooperative flow in Taglish |
| `q3_ph_call_objection.mp3` | Philippines | Price objection in Taglish |
| `q3_id_call_cooperative.mp3` | Indonesia | Cooperative flow in Bahasa |
| `q3_id_call_regional.mp3` | Indonesia | Regional accent / colloquial speech |

For each Q3 recording, note in the transcript:
- Which language the customer spoke in
- Whether code-switching occurred naturally
- Any ASR errors observed
- Any TTS quality issues

---

## Q4 — Required evidence

1. **Dashboard screenshot** — Browser showing live nudges during a call
2. **Latency report** — Run `curl http://localhost:8003/report` and save output as `q4_latency_report.json`
3. **Recording of simulation** — Screen record the dashboard while `simulate_call.py` runs

Required nudge coverage (at least one of each):
- `cross_sell` — customer mentions a second vehicle or uncovered family member
- `compliance_gap` — agent makes an unverified claim
- `frustration` — customer expresses anger/impatience

---

## Architecture diagram

Save as `architecture.png` or `architecture.pdf` in this folder.

Minimum content:
- Browser → SmallWebRTCTransport → GeminiLiveLLMService
- GeminiLiveLLMService ↔ search_knowledge_base tool ↔ KB /retrieve API
- KB /retrieve → Qdrant (vector) + BM25 (local) → RRF fusion
- Q4: WAV → faster-whisper → TranscriptBuffer → Gemini signal extraction → WebSocket → Dashboard

---

## Video walkthrough

Required topics (assessment spec):
1. System overview and live demo
2. Architecture and key design decisions
3. KB/retrieval design and voice agent flow (show KB tool call in terminal)
4. Multilingual handling (show PH and ID bots)
5. Live nudge generation (show dashboard)
6. Error/fallback cases
7. Known limitations and production improvements

Recommended: ~10–15 minutes. Tools: Loom, OBS, QuickTime.
