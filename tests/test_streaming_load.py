"""
Streaming load test harness.

Usage:
  python tests/test_streaming_load.py --url ws://localhost:8002/v1/stream --token <api-key> --clients 10
"""

import argparse
import asyncio
import json
import statistics
import sys
import time

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)


async def _run_client(client_id: int, url: str, token: str, audio_seconds: int, chunk_ms: int):
    ws_url = f"{url}?token={token}" if token else url
    session_id = f"load-{client_id}-{int(time.time() * 1000)}"
    first_transcript_latency = None
    start_time = time.perf_counter()

    # 8kHz, 16-bit PCM silence
    sample_rate = 8000
    bytes_per_chunk = int(sample_rate * (chunk_ms / 1000.0) * 2)
    total_chunks = int((audio_seconds * 1000) / chunk_ms)
    silence_chunk = b"\x00" * bytes_per_chunk

    async with websockets.connect(ws_url) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.start",
                    "session_id": session_id,
                    "call_metadata": {"source": "load-test"},
                    "audio_config": {"sample_rate": sample_rate, "encoding": "pcm_s16le"},
                }
            )
        )

        started = json.loads(await ws.recv())
        if started.get("type") == "error":
            raise RuntimeError(f"session.start failed: {started.get('message')}")

        async def receive_updates():
            nonlocal first_transcript_latency
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") == "transcript.update" and first_transcript_latency is None:
                    first_transcript_latency = time.perf_counter() - start_time
                if msg.get("type") == "session.ended":
                    break

        receiver = asyncio.create_task(receive_updates())
        for _ in range(total_chunks):
            await ws.send(silence_chunk)
            await asyncio.sleep(chunk_ms / 1000.0)
        await ws.send(json.dumps({"type": "session.end"}))
        await asyncio.wait_for(receiver, timeout=60)

    return first_transcript_latency


async def run_load(url: str, token: str, clients: int, audio_seconds: int, chunk_ms: int):
    tasks = [
        _run_client(i, url, token, audio_seconds=audio_seconds, chunk_ms=chunk_ms)
        for i in range(clients)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    latencies = []
    errors = 0
    clients_without_transcripts = 0
    for result in results:
        if isinstance(result, Exception):
            errors += 1
            continue
        if result is not None:
            latencies.append(result)
        else:
            clients_without_transcripts += 1

    if not latencies:
        if errors == clients:
            raise RuntimeError(f"No successful sessions. errors={errors}/{clients}")
        return {
            "clients": clients,
            "errors": errors,
            "clients_without_transcripts": clients_without_transcripts,
            "samples": 0,
            "p50_first_transcript_s": None,
            "p95_first_transcript_s": None,
        }

    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted)
    p95_idx = max(0, min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.95) - 1))
    p95 = latencies_sorted[p95_idx]

    return {
        "clients": clients,
        "errors": errors,
        "clients_without_transcripts": clients_without_transcripts,
        "samples": len(latencies_sorted),
        "p50_first_transcript_s": round(p50, 3),
        "p95_first_transcript_s": round(p95, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Streaming load test")
    parser.add_argument("--url", required=True, help="WebSocket streaming URL")
    parser.add_argument("--token", default=None, help="API key for non-staging environments")
    parser.add_argument("--clients", type=int, default=10, help="Concurrent clients")
    parser.add_argument("--audio-seconds", type=int, default=12, help="Audio seconds per client")
    parser.add_argument("--chunk-ms", type=int, default=100, help="Chunk size in ms")
    parser.add_argument("--p95-target", type=float, default=3.0, help="Target p95 latency (seconds)")
    args = parser.parse_args()

    summary = asyncio.run(
        run_load(
            url=args.url,
            token=args.token,
            clients=args.clients,
            audio_seconds=args.audio_seconds,
            chunk_ms=args.chunk_ms,
        )
    )
    print(json.dumps(summary, indent=2))

    if summary["p95_first_transcript_s"] is None:
        print("WARN: no transcript updates observed; skipping p95 target check")
        return

    if summary["p95_first_transcript_s"] > args.p95_target:
        print(
            f"FAIL: p95 latency {summary['p95_first_transcript_s']}s "
            f"exceeds target {args.p95_target}s"
        )
        sys.exit(1)
    print("PASS: p95 latency target met")


if __name__ == "__main__":
    main()
