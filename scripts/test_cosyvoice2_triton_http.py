"""
Offline HTTP smoke test for the official CosyVoice Triton runtime.

This matches the supported request contract from the upstream
`runtime/triton_trtllm/client_http.py` example:
- reference_wav
- reference_wav_len
- reference_text
- target_text

It is intentionally separate from the local Python CosyVoice2 test because the
official Triton runtime currently exposes a reference-based path rather than the
`instruct2` API used in local evaluation.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


DEFAULT_SERVER_URL = "http://localhost:3950"
DEFAULT_MODEL_NAME = "cosyvoice2"
DEFAULT_TARGET_SR = 16000
DEFAULT_OUTPUT_SR = 24000


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def import_dependencies():
    try:
        import requests
        import soundfile as sf
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for CosyVoice Triton HTTP testing. Install the project test"
            " dependencies before running this script.\n"
            f"Original import error: {exc}"
        ) from exc
    return requests, sf, resample_poly


def load_reference_audio(reference_audio: Path, target_sr: int, max_seconds: float) -> tuple[np.ndarray, int]:
    _, sf, resample_poly = import_dependencies()
    waveform, sample_rate = sf.read(str(reference_audio))

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    waveform = np.asarray(waveform, dtype=np.float32)

    if sample_rate != target_sr:
        gcd = np.gcd(sample_rate, target_sr)
        waveform = resample_poly(waveform, target_sr // gcd, sample_rate // gcd).astype(np.float32)
        sample_rate = target_sr

    max_samples = int(max_seconds * sample_rate)
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]

    return waveform, sample_rate


def build_request(waveform: np.ndarray, reference_text: str, target_text: str) -> dict:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(1, -1)
    lengths = np.array([[waveform.shape[1]]], dtype=np.int32)
    return {
        "inputs": [
            {
                "name": "reference_wav",
                "shape": list(waveform.shape),
                "datatype": "FP32",
                "data": waveform.tolist(),
            },
            {
                "name": "reference_wav_len",
                "shape": list(lengths.shape),
                "datatype": "INT32",
                "data": lengths.tolist(),
            },
            {
                "name": "reference_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [reference_text],
            },
            {
                "name": "target_text",
                "shape": [1, 1],
                "datatype": "BYTES",
                "data": [target_text],
            },
        ]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a CosyVoice2 Triton offline HTTP smoke test")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help=f"Server base URL (default: {DEFAULT_SERVER_URL})")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"Triton model name (default: {DEFAULT_MODEL_NAME})")
    parser.add_argument("--reference-audio", type=Path, required=True, help="Reference speaker WAV path")
    parser.add_argument("--reference-text", required=True, help="Transcript of the reference speaker audio")
    parser.add_argument("--target-text", required=True, help="Target synthesis text")
    parser.add_argument("--output-audio", type=Path, default=Path("results/tts_cosyvoice2_triton/output.wav"), help="Output WAV path")
    parser.add_argument("--request-id", default="0", help="Optional Triton request id")
    parser.add_argument("--max-reference-seconds", type=float, default=30.0, help="Trim reference audio to this duration")
    parser.add_argument("--timeout-sec", type=float, default=600.0, help="HTTP timeout in seconds")
    args = parser.parse_args()
    requests, sf, _ = import_dependencies()

    ensure_dir(args.output_audio)
    server_url = args.server_url.rstrip("/")
    infer_url = f"{server_url}/v2/models/{args.model_name}/infer"

    waveform, sample_rate = load_reference_audio(
        reference_audio=args.reference_audio,
        target_sr=DEFAULT_TARGET_SR,
        max_seconds=args.max_reference_seconds,
    )
    payload = build_request(waveform, args.reference_text, args.target_text)

    print("\n=== CosyVoice2 Triton HTTP Test ===")
    print(f"Server: {infer_url}")
    print(f"Reference audio: {args.reference_audio}")
    print(f"Reference sample rate sent: {sample_rate} Hz")

    started = time.perf_counter()
    response = requests.post(
        infer_url,
        headers={"Content-Type": "application/json"},
        json=payload,
        params={"request_id": args.request_id},
        timeout=args.timeout_sec,
    )
    elapsed = time.perf_counter() - started

    response.raise_for_status()
    result = response.json()
    outputs = result.get("outputs", [])
    if not outputs:
        raise RuntimeError(f"No outputs returned by Triton: {result}")

    audio = np.asarray(outputs[0]["data"], dtype=np.float32)
    sf.write(str(args.output_audio), audio, DEFAULT_OUTPUT_SR, subtype="PCM_16")

    audio_duration_sec = audio.shape[0] / DEFAULT_OUTPUT_SR
    rtf = elapsed / audio_duration_sec if audio_duration_sec > 0 else None

    print(f"Wrote audio to {args.output_audio}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Audio duration: {audio_duration_sec:.2f}s")
    if rtf is not None:
        print(f"RTF: {rtf:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
