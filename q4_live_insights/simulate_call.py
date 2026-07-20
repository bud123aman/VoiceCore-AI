"""
Q4 — Call Simulation Script
Replays a WAV file at real-time speed through the pipeline,
or runs a synthetic demo if no audio file is provided.

Usage:
    # Synthetic demo (no audio needed):
    python simulate_call.py

    # With real audio file:
    python simulate_call.py ../recordings/call1.wav

    # First start the server:
    uvicorn pipeline:app --port 8003
    # Then open http://localhost:8003 in browser
    # Then run this script
"""

import asyncio
import sys
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")


async def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else ""
    server_url = "http://localhost:8003"

    print(f"Starting simulation...")
    if audio_path:
        print(f"  Audio file: {audio_path}")
    else:
        print(f"  Mode: synthetic demo (no audio file)")

    print(f"  Dashboard: {server_url}")
    print(f"  Open the dashboard in your browser before continuing.")

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.post(
                f"{server_url}/simulate",
                params={"audio_path": audio_path},
            )
            data = resp.json()
            if "error" in data:
                print(f"Error: {data['error']}")
            else:
                print(f"\n✓ {data['message']}")
                print("Watch the dashboard for real-time nudges.")
                print("\nWaiting for simulation to complete...")
                await asyncio.sleep(90) 

                resp = await client.get(f"{server_url}/report")
                report = resp.json()
                print("\n" + "=" * 50)
                print("FINAL LATENCY REPORT")
                print("=" * 50)
                print(f"  ASR latency:    P50={report['latency']['asr_ms']['p50']}ms  P95={report['latency']['asr_ms']['p95']}ms")
                print(f"  Signal extract: P50={report['latency']['signal_extraction_ms']['p50']}ms  P95={report['latency']['signal_extraction_ms']['p95']}ms")
                print(f"  End-to-end:     P50={report['latency']['end_to_end_ms']['p50']}ms  P95={report['latency']['end_to_end_ms']['p95']}ms")
                print(f"  Nudges emitted: {report['nudges_emitted']}")

        except httpx.ConnectError:
            print(f"\nError: Cannot connect to {server_url}")
            print("Make sure the server is running: uvicorn pipeline:app --port 8003")


if __name__ == "__main__":
    asyncio.run(main())
