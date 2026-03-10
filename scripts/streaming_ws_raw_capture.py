#!/usr/bin/env python3
"""
Stream a WAV file to the staging WebSocket API and capture raw streaming model outputs.

Why two passes:
- WebSocket pass exercises the real API flow.
- Direct Triton pass captures raw ASR/diar outputs per chunk.

This avoids session-cache collisions in Triton (do not reuse the same session_id
for both API and direct model calls at the same time).
"""

import argparse
import asyncio
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import soundfile as sf
import tritonclient.grpc.aio as grpcclient_aio
import websockets
from scipy.signal import resample


TARGET_SR = 16000


def load_wav_mono(audio_path: str) -> Tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    return audio, int(sample_rate)


def to_pcm16_bytes(audio: np.ndarray) -> bytes:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    return pcm16.tobytes()


def iter_audio_frames_pcm(audio: np.ndarray, sample_rate: int, frame_ms: int):
    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    pcm = to_pcm16_bytes(audio)
    frame_bytes = frame_samples * 2  # int16
    for i in range(0, len(pcm), frame_bytes):
        chunk = pcm[i : i + frame_bytes]
        if chunk:
            yield chunk


def resample_to_16k(audio_chunk: np.ndarray, input_sr: int) -> np.ndarray:
    if input_sr == TARGET_SR:
        return audio_chunk.astype(np.float32)
    num_target_samples = int(len(audio_chunk) * TARGET_SR / input_sr)
    return resample(audio_chunk, num_target_samples).astype(np.float32)


def iter_model_chunks(audio: np.ndarray, sample_rate: int, chunk_seconds: float):
    chunk_samples = max(1, int(sample_rate * chunk_seconds))
    total = len(audio)
    index = 0
    for start in range(0, total, chunk_samples):
        end = min(start + chunk_samples, total)
        chunk = audio[start:end]
        is_final = end >= total
        yield index, start / sample_rate, end / sample_rate, chunk, is_final
        index += 1


def decode_triton_json_payload(raw_output: np.ndarray) -> Tuple[str, Optional[Dict]]:
    value = raw_output
    while isinstance(value, np.ndarray):
        if value.size == 0:
            return "", None
        value = value[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if not isinstance(value, str):
        value = str(value)
    try:
        return value, json.loads(value)
    except json.JSONDecodeError:
        return value, None


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_model_delta_for_report(model_text: str, previous_model_text: str) -> Tuple[str, str]:
    """
    Lightweight delta extractor for report rendering.
    """
    new_text = normalize_text(model_text)
    previous = normalize_text(previous_model_text)

    if not new_text:
        return "", previous
    if not previous:
        return new_text, new_text
    if new_text == previous:
        return "", previous
    if new_text.startswith(previous):
        return new_text[len(previous):].lstrip(), new_text

    # Sliding overlap handling (avoid repeating the same prefix again).
    prev_words = previous.split()
    new_words = new_text.split()
    max_k = min(len(prev_words), len(new_words), 64)
    for k in range(max_k, 1, -1):
        if prev_words[-k:] == new_words[:k]:
            suffix = new_words[k:]
            return (" ".join(suffix), new_text) if suffix else ("", new_text)

    if previous.endswith(new_text):
        return "", previous
    return new_text, new_text


def _to_speaker_id(raw_speaker) -> int:
    if isinstance(raw_speaker, int):
        return max(0, raw_speaker)
    if isinstance(raw_speaker, float):
        return max(0, int(raw_speaker))
    text = str(raw_speaker or "")
    match = re.search(r"\d+", text)
    if match:
        return max(0, int(match.group()))
    return 0


def _largest_overlap_speaker(
    speaker_segments: List[Dict],
    start_t: float,
    end_t: float,
) -> str:
    if not speaker_segments:
        return "spk_0"

    best_speaker = "spk_0"
    best_overlap = -1.0
    for seg in speaker_segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", 0.0))
        overlap = max(0.0, min(end_t, seg_end) - max(start_t, seg_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = seg.get("speaker", "spk_0")

    # If no overlap, pick nearest by midpoint.
    if best_overlap <= 0:
        midpoint = (start_t + end_t) / 2.0
        nearest_dist = float("inf")
        for seg in speaker_segments:
            seg_mid = (float(seg.get("start", 0.0)) + float(seg.get("end", 0.0))) / 2.0
            dist = abs(midpoint - seg_mid)
            if dist < nearest_dist:
                nearest_dist = dist
                best_speaker = seg.get("speaker", "spk_0")

    return best_speaker


def _format_clock(seconds: float) -> str:
    total = max(0.0, float(seconds))
    mins = int(total // 60)
    secs = total - (mins * 60)
    return f"{mins:02d}:{secs:05.2f}"


def build_streaming_report(raw_rows: List[Dict]) -> Dict:
    diar_stream_segments: List[Dict] = []
    diar_segments_for_overlap: List[Dict] = []
    asr_events: List[Dict] = []
    fused: List[Dict] = []
    human_lines: List[str] = []

    seen_diar_segments = set()
    previous_model_text = ""

    for row in raw_rows:
        diar_parsed = row.get("diar_parsed") or {}
        asr_parsed = row.get("asr_parsed") or {}

        diar_status = "final" if bool(diar_parsed.get("is_final") or row.get("is_final")) else "partial"
        for seg in (diar_parsed.get("segments") or []):
            speaker_id = _to_speaker_id(seg.get("speaker", 0))
            start_t = round(float(seg.get("start", 0.0)), 3)
            end_t = round(float(seg.get("end", 0.0)), 3)
            key = (speaker_id, start_t, end_t)
            if key in seen_diar_segments:
                continue
            seen_diar_segments.add(key)

            speaker = f"spk_{speaker_id}"
            diar_stream_segments.append(
                {
                    "speaker": speaker,
                    "start": start_t,
                    "end": end_t,
                    "status": diar_status,
                }
            )
            diar_segments_for_overlap.append(
                {
                    "speaker": speaker,
                    "start": start_t,
                    "end": end_t,
                }
            )

        model_text = normalize_text(asr_parsed.get("text", ""))
        if model_text:
            delta_text, previous_model_text = extract_model_delta_for_report(
                model_text,
                previous_model_text,
            )
            if delta_text:
                asr_events.append(
                    {
                        "start": round(float(row.get("chunk_start_sec", 0.0)), 3),
                        "end": round(float(row.get("chunk_end_sec", 0.0)), 3),
                        "text": delta_text,
                        "is_final": bool(asr_parsed.get("is_final") or row.get("is_final")),
                    }
                )

    for evt in asr_events:
        speaker = _largest_overlap_speaker(
            diar_segments_for_overlap,
            float(evt["start"]),
            float(evt["end"]),
        )
        fused.append(
            {
                "speaker": speaker,
                "start": evt["start"],
                "end": evt["end"],
                "text": evt["text"],
                "is_final": evt["is_final"],
            }
        )

    for item in fused:
        text = item["text"]
        if not item["is_final"] and not text.endswith("..."):
            text = f"{text}..."
        human_lines.append(
            f"[{_format_clock(item['start'])} - {_format_clock(item['end'])}] "
            f"{item['speaker']}: {text}"
        )

    return {
        "raw_streaming_outputs_parallel": {
            "sortformer_output_stream": {"speaker_segments": diar_stream_segments},
            "streaming_asr_output": {"asr_events": asr_events},
        },
        "fused_diarized_streaming_transcript": {"diarized_transcript": fused},
        "human_readable_live_output": human_lines,
        "merge_logic": [
            "collect streaming ASR hypotheses in chunk order",
            "collect streaming diarization speaker segments in parallel",
            "assign each ASR event to speaker segment with largest timestamp overlap",
            "keep partial utterances editable until they finalize",
            "allow recent-window speaker correction if diarization stabilizes later",
        ],
    }


def render_streaming_report_text(report: Dict) -> str:
    raw_parallel = report.get("raw_streaming_outputs_parallel", {})
    sortformer = raw_parallel.get("sortformer_output_stream", {"speaker_segments": []})
    asr_stream = raw_parallel.get("streaming_asr_output", {"asr_events": []})
    fused = report.get("fused_diarized_streaming_transcript", {"diarized_transcript": []})
    human = report.get("human_readable_live_output", [])
    merge_logic = report.get("merge_logic", [])

    parts = [
        "1) Raw streaming outputs in parallel",
        "A. Sortformer output stream",
        json.dumps(sortformer, indent=2, ensure_ascii=False),
        "B. Streaming ASR output",
        json.dumps(asr_stream, indent=2, ensure_ascii=False),
        "2) Fused diarized streaming transcript",
        json.dumps(fused, indent=2, ensure_ascii=False),
        "3) Human-readable live output",
        "\n".join(human) if human else "(no transcript lines produced)",
        "4) Merge logic used",
        "\n".join(f"- {step}" for step in merge_logic) if merge_logic else "- (not available)",
    ]
    return "\n\n".join(parts)


async def infer_streaming_asr_raw(
    client: grpcclient_aio.InferenceServerClient,
    model_name: str,
    audio_16k: np.ndarray,
    session_id: str,
    is_final: bool,
    request_id: str,
):
    audio_data = audio_16k.astype(np.float32).flatten()
    audio_input = grpcclient_aio.InferInput("audio_data", [1, len(audio_data)], "FP32")
    audio_input.set_data_from_numpy(audio_data.reshape(1, -1))

    sr_input = grpcclient_aio.InferInput("sample_rate", [1, 1], "INT32")
    sr_input.set_data_from_numpy(np.array([[TARGET_SR]], dtype=np.int32))

    sid_input = grpcclient_aio.InferInput("session_id", [1, 1], "BYTES")
    sid_input.set_data_from_numpy(np.array([[session_id]], dtype=object))

    final_input = grpcclient_aio.InferInput("is_final", [1, 1], "BOOL")
    final_input.set_data_from_numpy(np.array([[is_final]], dtype=bool))

    output = grpcclient_aio.InferRequestedOutput("transcription")
    response = await client.infer(
        model_name=model_name,
        inputs=[audio_input, sr_input, sid_input, final_input],
        outputs=[output],
        request_id=request_id,
    )
    raw_np = response.as_numpy("transcription")
    raw_text, parsed = decode_triton_json_payload(raw_np)
    return raw_np, raw_text, parsed


async def infer_streaming_diar_raw(
    client: grpcclient_aio.InferenceServerClient,
    model_name: str,
    audio_16k: np.ndarray,
    session_id: str,
    is_final: bool,
    request_id: str,
):
    audio_data = audio_16k.astype(np.float32).flatten()
    audio_input = grpcclient_aio.InferInput("audio_data", [1, len(audio_data)], "FP32")
    audio_input.set_data_from_numpy(audio_data.reshape(1, -1))

    sr_input = grpcclient_aio.InferInput("sample_rate", [1, 1], "INT32")
    sr_input.set_data_from_numpy(np.array([[TARGET_SR]], dtype=np.int32))

    sid_input = grpcclient_aio.InferInput("session_id", [1, 1], "BYTES")
    sid_input.set_data_from_numpy(np.array([[session_id]], dtype=object))

    final_input = grpcclient_aio.InferInput("is_final", [1, 1], "BOOL")
    final_input.set_data_from_numpy(np.array([[is_final]], dtype=bool))

    output = grpcclient_aio.InferRequestedOutput("diarization_output")
    response = await client.infer(
        model_name=model_name,
        inputs=[audio_input, sr_input, sid_input, final_input],
        outputs=[output],
        request_id=request_id,
    )
    raw_np = response.as_numpy("diarization_output")
    raw_text, parsed = decode_triton_json_payload(raw_np)
    return raw_np, raw_text, parsed


async def websocket_stream_pass(
    ws_url: str,
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int,
    send_speed: float,
    session_id: str,
    token: Optional[str],
    call_metadata: Optional[Dict],
    output_jsonl_path: Path,
):
    final_url = ws_url
    if token:
        sep = "&" if "?" in ws_url else "?"
        final_url = f"{ws_url}{sep}token={quote(token)}"

    start_message = {
        "type": "session.start",
        "session_id": session_id,
        "audio_config": {"sample_rate": sample_rate},
    }
    if call_metadata:
        start_message["call_metadata"] = call_metadata

    messages: List[Dict] = []

    async with websockets.connect(final_url, max_size=20 * 1024 * 1024) as ws:
        await ws.send(json.dumps(start_message))

        async def recv_loop():
            while True:
                try:
                    msg = await ws.recv()
                except websockets.ConnectionClosed:
                    break

                ts = datetime.now(timezone.utc).isoformat()
                if isinstance(msg, (bytes, bytearray)):
                    messages.append(
                        {
                            "timestamp_utc": ts,
                            "kind": "bytes",
                            "size_bytes": len(msg),
                        }
                    )
                    continue

                try:
                    payload = json.loads(msg)
                except json.JSONDecodeError:
                    payload = {"raw_text": msg}

                messages.append(
                    {
                        "timestamp_utc": ts,
                        "kind": "json",
                        "payload": payload,
                    }
                )

                msg_type = payload.get("type")
                if msg_type in {"session.ended", "error"}:
                    break

        recv_task = asyncio.create_task(recv_loop())

        try:
            loop = asyncio.get_running_loop()
            frame_interval_s = (frame_ms / 1000.0) / max(send_speed, 1e-6)
            next_send_at = loop.time()
            for frame in iter_audio_frames_pcm(audio, sample_rate, frame_ms):
                sleep_for = next_send_at - loop.time()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                await ws.send(frame)
                next_send_at = max(next_send_at + frame_interval_s, loop.time())
            await ws.send(json.dumps({"type": "session.end"}))
        except websockets.ConnectionClosed:
            pass
        await recv_task

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl_path.open("w", encoding="utf-8") as f:
        for row in messages:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "ws_url": final_url,
        "session_id": session_id,
        "message_count": len(messages),
        "output_jsonl": str(output_jsonl_path),
    }


async def direct_triton_raw_pass(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    asr_url: str,
    asr_model: str,
    diar_url: str,
    diar_model: str,
    session_id: str,
    output_jsonl_path: Path,
):
    asr_client = grpcclient_aio.InferenceServerClient(url=asr_url)
    diar_client = grpcclient_aio.InferenceServerClient(url=diar_url)

    rows: List[Dict] = []
    try:
        for chunk_index, start_s, end_s, chunk, is_final in iter_model_chunks(
            audio, sample_rate, chunk_seconds
        ):
            chunk_16k = resample_to_16k(chunk, sample_rate)
            request_base = f"{session_id}_{chunk_index}"

            asr_task = infer_streaming_asr_raw(
                asr_client,
                asr_model,
                chunk_16k,
                session_id,
                is_final,
                f"{request_base}_asr",
            )
            diar_task = infer_streaming_diar_raw(
                diar_client,
                diar_model,
                chunk_16k,
                session_id,
                is_final,
                f"{request_base}_diar",
            )
            (asr_np, asr_raw_text, asr_parsed), (
                diar_np,
                diar_raw_text,
                diar_parsed,
            ) = await asyncio.gather(asr_task, diar_task)

            rows.append(
                {
                    "chunk_index": chunk_index,
                    "chunk_start_sec": round(start_s, 3),
                    "chunk_end_sec": round(end_s, 3),
                    "chunk_samples_input_sr": int(len(chunk)),
                    "chunk_samples_16k": int(len(chunk_16k)),
                    "is_final": is_final,
                    "asr_raw_shape": list(asr_np.shape),
                    "diar_raw_shape": list(diar_np.shape),
                    "asr_raw_text": asr_raw_text,
                    "diar_raw_text": diar_raw_text,
                    "asr_parsed": asr_parsed,
                    "diar_parsed": diar_parsed,
                }
            )
    finally:
        await asr_client.close()
        await diar_client.close()

    output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "asr_url": asr_url,
        "diar_url": diar_url,
        "asr_model": asr_model,
        "diar_model": diar_model,
        "session_id": session_id,
        "chunk_count": len(rows),
        "output_jsonl": str(output_jsonl_path),
    }
    return summary, rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stream WAV to staging WebSocket and capture raw streaming model outputs."
    )
    parser.add_argument("--audio", required=True, help="Path to WAV file")
    parser.add_argument("--ws-url", default="ws://localhost:8002/v1/stream", help="Streaming WS URL")
    parser.add_argument("--token", default=None, help="API token (optional in staging)")
    parser.add_argument("--frame-ms", type=int, default=100, help="WebSocket audio frame size in ms")
    parser.add_argument(
        "--send-speed",
        type=float,
        default=1.0,
        help="WebSocket send speed multiplier vs realtime (1.0 = realtime)",
    )

    parser.add_argument("--asr-url", default="localhost:3901", help="Streaming ASR Triton gRPC URL")
    parser.add_argument("--diar-url", default="localhost:4001", help="Streaming diar Triton gRPC URL")
    parser.add_argument("--asr-model", default="streaming_asr", help="ASR Triton model name")
    parser.add_argument("--diar-model", default="streaming_diar", help="Diar Triton model name")
    parser.add_argument(
        "--model-chunk-seconds",
        type=float,
        default=1.0,
        help="Chunk size in seconds for direct Triton raw-capture replay",
    )

    parser.add_argument("--call-metadata-json", default=None, help="Optional JSON string for call_metadata")
    parser.add_argument("--out-dir", default="results/streaming_raw_capture", help="Output directory root")
    return parser.parse_args()


async def main():
    args = parse_args()
    audio_path = os.path.abspath(args.audio)
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if args.send_speed <= 0:
        raise ValueError("--send-speed must be > 0")

    call_metadata = None
    if args.call_metadata_json:
        call_metadata = json.loads(args.call_metadata_json)

    audio, sample_rate = load_wav_mono(audio_path)
    duration_sec = len(audio) / sample_rate if sample_rate > 0 else 0.0

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base_dir = Path(args.out_dir) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    ws_session_id = f"ws-{uuid.uuid4()}"
    raw_session_id = f"raw-{uuid.uuid4()}"

    ws_out = base_dir / "websocket_messages.jsonl"
    raw_out = base_dir / "raw_model_outputs.jsonl"
    summary_out = base_dir / "summary.json"

    ws_summary = await websocket_stream_pass(
        ws_url=args.ws_url,
        audio=audio,
        sample_rate=sample_rate,
        frame_ms=args.frame_ms,
        send_speed=args.send_speed,
        session_id=ws_session_id,
        token=args.token,
        call_metadata=call_metadata,
        output_jsonl_path=ws_out,
    )

    raw_summary, raw_rows = await direct_triton_raw_pass(
        audio=audio,
        sample_rate=sample_rate,
        chunk_seconds=args.model_chunk_seconds,
        asr_url=args.asr_url,
        asr_model=args.asr_model,
        diar_url=args.diar_url,
        diar_model=args.diar_model,
        session_id=raw_session_id,
        output_jsonl_path=raw_out,
    )

    report = build_streaming_report(raw_rows)
    report_text = render_streaming_report_text(report)
    report_json_out = base_dir / "streaming_report.json"
    report_txt_out = base_dir / "streaming_report.txt"
    report_json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_txt_out.write_text(report_text, encoding="utf-8")

    summary = {
        "audio_path": audio_path,
        "audio_sample_rate": sample_rate,
        "audio_duration_sec": round(duration_sec, 3),
        "notes": [
            "WebSocket and direct Triton capture use separate session_ids on purpose.",
            "Direct Triton raw capture is the source of model raw outputs.",
        ],
        "websocket_pass": ws_summary,
        "direct_triton_raw_pass": raw_summary,
        "artifacts": {
            "websocket_messages_jsonl": str(ws_out),
            "raw_model_outputs_jsonl": str(raw_out),
            "streaming_report_json": str(report_json_out),
            "streaming_report_txt": str(report_txt_out),
            "summary_json": str(summary_out),
        },
    }

    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(report_text)
    print("\n---\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
