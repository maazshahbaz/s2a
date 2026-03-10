"""
End-to-end test for streaming STT with diarization.
Connects via WebSocket, streams a WAV file as PCM chunks, and prints diarized transcription.

Usage:
    python tests/test_streaming_e2e.py --audio tests/test_audio/test.wav
    python tests/test_streaming_e2e.py --audio tests/test_audio/test.wav --url ws://localhost:8002/v1/stream
    python tests/test_streaming_e2e.py --audio tests/test_audio/test.wav --url ws://localhost:8002/v1/stream --token bp-proj-xxx
"""

import argparse
import asyncio
import json
import time
import sys
import struct

try:
    import websockets
except ImportError:
    print("ERROR: websockets not installed. Run: pip install websockets")
    sys.exit(1)

try:
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample
except ImportError:
    print("ERROR: Missing deps. Run: pip install soundfile numpy scipy")
    sys.exit(1)


async def stream_audio_websocket(url, audio_path, token=None, sample_rate=8000, chunk_ms=100):
    """Stream audio over WebSocket and print live transcription."""

    # Load audio
    print(f"Loading audio: {audio_path}")
    audio_data, sr = sf.read(audio_path, dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Downsample to target sample_rate (simulate telephony)
    if sr != sample_rate:
        num_samples = int(len(audio_data) * sample_rate / sr)
        audio_data = resample(audio_data, num_samples).astype(np.float32)

    # Convert to 16-bit PCM bytes
    audio_int16 = (audio_data * 32768).clip(-32768, 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    duration = len(audio_data) / sample_rate
    print(f"  Duration: {duration:.1f}s, Sample rate: {sample_rate}Hz")
    print(f"  Chunk size: {chunk_ms}ms")

    # Add token to URL if provided
    ws_url = url
    if token:
        ws_url = f"{url}?token={token}"

    print(f"\nConnecting to {ws_url}...")

    async with websockets.connect(ws_url) as ws:
        # Send session.start
        start_msg = {
            "type": "session.start",
            "session_id": f"test-{int(time.time())}",
            "call_metadata": {
                "src": "+1234567890",
                "calldate": "2026-02-23 12:00:00",
                "direction": "INBOUND",
            },
            "callback_url": f"http://localhost:8002/v1/webhook",
            "audio_config": {
                "sample_rate": sample_rate,
                "encoding": "pcm_s16le",
            },
        }
        await ws.send(json.dumps(start_msg))

        # Wait for session.started
        response = await ws.recv()
        resp = json.loads(response)
        print(f"  Session started: {json.dumps(resp, indent=2)}")

        if resp.get("type") == "error":
            print(f"  ERROR: {resp.get('message')}")
            return

        # Stream audio chunks
        chunk_bytes = int(sample_rate * chunk_ms / 1000 * 2)  # 2 bytes per sample
        total_chunks = len(audio_bytes) // chunk_bytes
        start_time = time.time()

        print(f"\n--- Streaming {total_chunks} chunks ---\n")

        # Create a task to receive transcription updates
        update_count = 0
        async def receive_updates():
            nonlocal update_count
            try:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)

                    if data.get("type") == "transcript.update":
                        update_count += 1
                        num_speakers = data.get("num_speakers", 1)
                        is_final = data.get("is_final", False)

                        # Clear line and show live-updating transcript
                        print(f"\r  --- Update #{update_count} (speakers: {num_speakers}, final: {is_final}) ---")
                        for seg in data.get("segments", []):
                            speaker = seg.get("speaker", "?")
                            text = seg.get("text", "")
                            start = seg.get("start", 0)
                            end = seg.get("end", 0)
                            print(f"    [{speaker}] ({start:.1f}s-{end:.1f}s): {text}")

                    elif data.get("type") == "session.ended":
                        print(f"\n--- Session ended ---")
                        print(f"  Duration: {data.get('total_duration', 0):.1f}s")
                        print(f"  Segments: {data.get('total_segments', 0)}")
                        print(f"  Intelligence: {data.get('intelligence_status', 'N/A')}")
                        break

                    elif data.get("type") == "error":
                        print(f"  ERROR: {data.get('message')}")

                    elif data.get("type") == "pong":
                        pass

            except websockets.exceptions.ConnectionClosed:
                pass

        receiver = asyncio.create_task(receive_updates())

        # Send audio chunks at real-time pace
        for i in range(total_chunks):
            chunk_start = i * chunk_bytes
            chunk_end = chunk_start + chunk_bytes
            chunk = audio_bytes[chunk_start:chunk_end]

            await ws.send(chunk)

            # Simulate real-time: wait chunk_ms before sending next
            await asyncio.sleep(chunk_ms / 1000.0)

        # Send remaining bytes
        remaining = audio_bytes[total_chunks * chunk_bytes:]
        if remaining:
            await ws.send(remaining)

        elapsed = time.time() - start_time
        print(f"\n  Audio streamed in {elapsed:.1f}s (real-time: {duration:.1f}s)")

        # Send session.end
        await ws.send(json.dumps({"type": "session.end"}))

        # Wait for receiver to finish
        await asyncio.wait_for(receiver, timeout=30)

    print("\n  Done!")


async def run_audiosocket_client(host, port, audio_path, sample_rate=8000, chunk_ms=20):
    """Simulate an Asterisk AudioSocket connection."""
    import uuid as uuid_mod

    print(f"\nLoading audio: {audio_path}")
    audio_data, sr = sf.read(audio_path, dtype="float32")
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    if sr != sample_rate:
        num_samples = int(len(audio_data) * sample_rate / sr)
        audio_data = resample(audio_data, num_samples).astype(np.float32)

    audio_int16 = (audio_data * 32768).clip(-32768, 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    duration = len(audio_data) / sample_rate
    print(f"  Duration: {duration:.1f}s")

    print(f"\nConnecting to AudioSocket at {host}:{port}...")
    reader, writer = await asyncio.open_connection(host, port)

    try:
        # Send UUID frame (type=0x01)
        call_uuid = uuid_mod.uuid4()
        uuid_bytes = call_uuid.bytes
        header = struct.pack("!BH", 0x01, len(uuid_bytes))
        writer.write(header + uuid_bytes)
        await writer.drain()
        print(f"  Sent UUID: {call_uuid}")

        # Send audio frames (type=0x10, 320 bytes = 20ms at 8kHz)
        chunk_bytes = int(sample_rate * chunk_ms / 1000 * 2)
        total_chunks = len(audio_bytes) // chunk_bytes

        print(f"  Streaming {total_chunks} audio frames ({chunk_ms}ms each)...")

        for i in range(total_chunks):
            chunk_start = i * chunk_bytes
            chunk_end = chunk_start + chunk_bytes
            chunk = audio_bytes[chunk_start:chunk_end]

            header = struct.pack("!BH", 0x10, len(chunk))
            writer.write(header + chunk)
            await writer.drain()

            await asyncio.sleep(chunk_ms / 1000.0)

            # Try to read any responses (non-blocking)
            try:
                resp_data = await asyncio.wait_for(reader.read(4096), timeout=0.01)
                if resp_data and len(resp_data) >= 3:
                    resp_type = resp_data[0]
                    if resp_type == 0x20:  # Custom transcript frame
                        resp_len = struct.unpack("!H", resp_data[1:3])[0]
                        transcript = resp_data[3:3 + resp_len].decode("utf-8")
                        data = json.loads(transcript)
                        for seg in data.get("segments", []):
                            print(f"    [{seg.get('speaker', '?')}]: {seg.get('text', '')}")
            except (asyncio.TimeoutError, Exception):
                pass

        # Send hangup frame (type=0x00)
        header = struct.pack("!BH", 0x00, 0)
        writer.write(header)
        await writer.drain()
        print(f"\n  Hangup sent. Session complete.")

    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    print("  Done!")


def main():
    parser = argparse.ArgumentParser(description="E2E Streaming STT Test")
    parser.add_argument("--audio", required=True, help="Path to test WAV file")
    parser.add_argument("--url", default="ws://localhost:8002/v1/stream", help="WebSocket URL")
    parser.add_argument("--token", default=None, help="API key for auth")
    parser.add_argument("--sample-rate", type=int, default=8000, help="Simulate this sample rate (default: 8000)")
    parser.add_argument("--chunk-ms", type=int, default=100, help="Chunk size in ms (default: 100)")
    parser.add_argument("--mode", choices=["websocket", "audiosocket", "both"], default="websocket", help="Test mode")
    args = parser.parse_args()

    if args.mode in ("websocket", "both"):
        print("=" * 60)
        print("WebSocket Streaming Test")
        print("=" * 60)
        asyncio.run(stream_audio_websocket(
            args.url, args.audio, args.token, args.sample_rate, args.chunk_ms
        ))

    if args.mode in ("audiosocket", "both"):
        print("\n" + "=" * 60)
        print("AudioSocket Streaming Test")
        print("=" * 60)
        # Extract host from URL or default to localhost
        host = "localhost"
        port = 8003
        asyncio.run(run_audiosocket_client(host, port, args.audio, args.sample_rate))


if __name__ == "__main__":
    main()
