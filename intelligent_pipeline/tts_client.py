from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import aiohttp
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config_loader import config


class AsyncTTSClient:
    """Async HTTP client for the separate CosyVoice Triton service."""

    def __init__(self, url: str = None):
        service_config = config.get_service_config("tts")
        self.url = (url or service_config.get("url", "http://localhost:3950")).rstrip("/")
        self.model_name = service_config.get("model_name", "cosyvoice2")
        self.output_sample_rate = int(service_config.get("sample_rate", 24000))
        self.reference_sample_rate = int(service_config.get("reference_sample_rate", 16000))
        self.max_reference_seconds = float(service_config.get("max_reference_seconds", 30.0))
        self.timeout_sec = float(service_config.get("timeout_sec", 600.0))
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self):
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_sec)
            self.session = aiohttp.ClientSession(timeout=timeout)

    def _load_reference_audio(self, reference_audio_path: str) -> np.ndarray:
        waveform, sample_rate = sf.read(reference_audio_path)

        if getattr(waveform, "ndim", 1) > 1:
            waveform = np.mean(waveform, axis=1)

        waveform = np.asarray(waveform, dtype=np.float32)
        if sample_rate != self.reference_sample_rate:
            gcd = np.gcd(sample_rate, self.reference_sample_rate)
            waveform = resample_poly(
                waveform,
                self.reference_sample_rate // gcd,
                sample_rate // gcd,
            ).astype(np.float32)
            sample_rate = self.reference_sample_rate

        max_samples = int(self.max_reference_seconds * sample_rate)
        if waveform.shape[0] > max_samples:
            waveform = waveform[:max_samples]

        return waveform

    def _build_payload(self, waveform: np.ndarray, reference_text: str, target_text: str) -> Dict:
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

    async def synthesize(
        self,
        reference_audio_path: str,
        reference_text: str,
        target_text: str,
        request_id: str = "0",
        output_path: str | None = None,
    ) -> Dict:
        await self.initialize()
        waveform = self._load_reference_audio(reference_audio_path)
        payload = self._build_payload(waveform, reference_text, target_text)
        infer_url = f"{self.url}/v2/models/{self.model_name}/infer"

        started = time.perf_counter()
        async with self.session.post(
            infer_url,
            headers={"Content-Type": "application/json"},
            json=payload,
            params={"request_id": request_id},
        ) as response:
            response.raise_for_status()
            result = await response.json()
        elapsed = time.perf_counter() - started

        outputs = result.get("outputs", [])
        if not outputs:
            raise RuntimeError(f"No outputs returned by TTS service: {result}")

        audio = np.asarray(outputs[0]["data"], dtype=np.float32)
        audio_duration_sec = audio.shape[0] / self.output_sample_rate if audio.size else 0.0

        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_file), audio, self.output_sample_rate, subtype="PCM_16")

        return {
            "sample_rate": self.output_sample_rate,
            "audio": audio,
            "audio_duration_sec": round(audio_duration_sec, 3),
            "elapsed_sec": round(elapsed, 3),
            "rtf": round(elapsed / audio_duration_sec, 3) if audio_duration_sec else None,
            "output_path": output_path,
        }

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()
