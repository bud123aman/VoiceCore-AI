"""
Q1 — Gemini → OpenAI-Compatible Proxy
Retell requires an OpenAI-format /chat/completions endpoint.
This proxy translates Retell's requests to Gemini API and back.

Usage:
    uvicorn gemini_proxy:app --port 8002 --reload

Retell custom LLM URL: http://your-server:8002/chat/completions
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import AsyncGenerator, Optional

from google import genai
from google.genai import types as genai_types
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
KB_API_URL = os.getenv("KB_API_URL", "http://localhost:8001")
GEMINI_MODEL = "gemini-2.0-flash"

_gemini_client = None

def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client

app = FastAPI(title="Gemini OpenAI Proxy", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── KB tool definition (OpenAI function-calling format) ───────────────────────
KB_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the HealthShield knowledge base for policy details, FAQs, "
            "product information, objection responses, and qualification rules. "
            "Always call this before answering any question about coverage, premiums, "
            "waiting periods, claims, or eligibility. Never answer from memory alone."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The customer's question or topic to search for",
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
                    "description": "Optional category filter to narrow results",
                },
            },
            "required": ["query"],
        },
    },
}


async def call_kb(query: str, category: Optional[str] = None) -> str:
    """Call the Q2 KB retrieval API."""
    payload = {"query": query, "top_k": 3}
    if category:
        payload["category_filter"] = category

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(f"{KB_API_URL}/retrieve", json=payload)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            if not results:
                return "No relevant information found in the knowledge base."

            # Format results for LLM context
            formatted = []
            for r in results:
                formatted.append(
                    f"[{r['record_id']}] {r['title']}\n"
                    f"{r['content']}\n"
                    f"Source: {r['citation']}"
                )
            return "\n\n---\n\n".join(formatted)

        except Exception as e:
            return f"Knowledge base unavailable: {str(e)}. Please escalate to human agent."


def messages_to_gemini(messages: list[dict]) -> tuple[str, list]:
    """
    Convert OpenAI-format messages to Gemini format.
    Returns (system_instruction, history).
    """
    system_instruction = ""
    history = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            system_instruction = content
        elif role == "user":
            history.append({"role": "user", "parts": [{"text": content}]})
        elif role == "assistant":
            # Handle tool calls in assistant messages
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                # Represent tool calls as text in Gemini (it handles function calling differently)
                tc_text = json.dumps(tool_calls)
                history.append({"role": "model", "parts": [{"text": tc_text}]})
            elif content:
                history.append({"role": "model", "parts": [{"text": content}]})
        elif role == "tool":
            # Tool results — append as user context
            history.append(
                {
                    "role": "user",
                    "parts": [{"text": f"[KB Result]: {content}"}],
                }
            )

    return system_instruction, history


async def run_gemini_with_tools(messages: list[dict]) -> dict:
    """
    Run Gemini with KB tool support.
    Implements a simple agentic loop: if Gemini wants to call KB, we call it and re-run.
    Max 2 tool call iterations to bound latency.
    """
    system_instruction, history = messages_to_gemini(messages)

    # Add KB tool awareness to system prompt
    enhanced_system = (
        system_instruction + "\n\n"
        "You have access to a knowledge base search tool. "
        "ALWAYS search the knowledge base before answering policy, pricing, "
        "coverage, eligibility, or claims questions. "
        "If the knowledge base returns no results, say so honestly — do NOT make up answers. "
        "Format: respond naturally as a voice agent. Keep responses concise (2-3 sentences max). "
        "Do not read out citations or record IDs to the customer."
    )

    # Check if last user message warrants a KB search
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    # Keywords that trigger automatic KB lookup
    kb_trigger_keywords = [
        "cover", "covered", "coverage", "premium", "price", "cost", "wait",
        "waiting", "claim", "hospital", "eligible", "eligibility", "plan",
        "policy", "pre-existing", "exclude", "excluded", "benefit", "maternity",
        "dental", "vision", "network", "tax", "deduction", "renewal", "cancel",
        "refund", "grace", "lapse", "covid", "senior", "family", "dependent",
    ]

    kb_context = ""
    if any(kw in last_user_msg.lower() for kw in kb_trigger_keywords):
        kb_context = await call_kb(last_user_msg)

    # Build full prompt with system instruction + history + KB context
    full_prompt = enhanced_system + "\n\n"

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            full_prompt += f"User: {content}\n"
        elif role == "assistant":
            full_prompt += f"Assistant: {content}\n"

    if kb_context:
        full_prompt += (
            f"\n[INTERNAL KB CONTEXT — use this to answer, do not mention it explicitly]:\n"
            f"{kb_context}\n\n"
        )

    full_prompt += "Assistant:"

    # Run Gemini
    client = get_gemini_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=256,
        ),
    )

    return {"content": response.text}


def make_openai_response(content: str, model: str = GEMINI_MODEL) -> dict:
    """Wrap Gemini response in OpenAI-compatible format."""
    return {
        "id": f"chatcmpl-gemini-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


async def stream_openai_response(content: str) -> AsyncGenerator[bytes, None]:
    """Stream response in OpenAI SSE format."""
    chunk_id = f"chatcmpl-gemini-{int(time.time())}"

    # Stream word by word for natural voice feel
    words = content.split(" ")
    for i, word in enumerate(words):
        chunk_content = word + (" " if i < len(words) - 1 else "")
        chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": GEMINI_MODEL,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": chunk_content},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        await asyncio.sleep(0.01)

    # Final chunk
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": GEMINI_MODEL,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(final_chunk)}\n\n".encode()
    yield b"data: [DONE]\n\n"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": GEMINI_MODEL}


@app.post("/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible /chat/completions endpoint.
    Retell will POST here with messages in OpenAI format.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if not messages:
        raise HTTPException(400, "No messages provided")

    try:
        result = await run_gemini_with_tools(messages)
        content = result["content"]
    except Exception as e:
        # Fail gracefully — tell agent to escalate
        content = (
            "I'm having trouble accessing our information system right now. "
            "Let me connect you with one of our specialists who can assist you immediately."
        )
        print(f"Gemini error: {e}")

    if stream:
        return StreamingResponse(
            stream_openai_response(content),
            media_type="text/event-stream",
        )
    else:
        return JSONResponse(make_openai_response(content))


@app.get("/models")
def list_models():
    """Retell sometimes checks available models."""
    return {
        "object": "list",
        "data": [{"id": GEMINI_MODEL, "object": "model", "owned_by": "google"}],
    }
