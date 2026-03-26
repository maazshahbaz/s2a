import os
import tritonclient.grpc.aio as grpcclient_aio
import numpy as np
import json
from typing import Dict
from .config_loader import config


class AsyncStreamingDiarClient:
    """Async gRPC Client for Streaming Diarization Triton Server."""

    def __init__(self, url: str = None):
        service_config = config.get_service_config("streaming_diar")
        env_url = os.getenv("S2A_STREAMING_DIAR_URL") or os.getenv("STREAMING_DIAR_URL")
        env_model_name = os.getenv("S2A_STREAMING_DIAR_MODEL_NAME")
        self.url = url or env_url or service_config.get("url", "localhost:4001")
        self.model_name = env_model_name or service_config.get("model_name", "streaming_diar")
        self.client = None

    async def connect(self):
        """Connect to Triton streaming diarization server."""
        self.client = grpcclient_aio.InferenceServerClient(url=self.url)

        if not await self.client.is_server_live():
            raise Exception(f"Streaming diar server at {self.url} is not live")
        if not await self.client.is_server_ready():
            raise Exception(f"Streaming diar server at {self.url} is not ready")
        if not await self.client.is_model_ready(self.model_name):
            raise Exception(f"Model {self.model_name} is not ready")

    async def diarize_chunk(
        self,
        audio_data: np.ndarray,
        session_id: str,
        sample_rate: int = 16000,
        is_final: bool = False,
        request_id: str = "stream",
    ) -> Dict:
        """
        Send an audio chunk for streaming diarization.

        Args:
            audio_data: Float32 audio waveform
            session_id: Unique session identifier (per call)
            sample_rate: Audio sample rate
            is_final: True if this is the last chunk in the session
            request_id: Request identifier for tracking

        Returns:
            {"segments": [{"speaker": 0, "start": 1.2, "end": 3.4}], "num_speakers": 2}
        """
        audio_data = audio_data.astype(np.float32).flatten()

        audio_input = grpcclient_aio.InferInput(
            "audio_data", [1, len(audio_data)], "FP32"
        )
        audio_input.set_data_from_numpy(audio_data.reshape(1, -1))

        sr_input = grpcclient_aio.InferInput("sample_rate", [1, 1], "INT32")
        sr_input.set_data_from_numpy(np.array([[sample_rate]], dtype=np.int32))

        sid_input = grpcclient_aio.InferInput("session_id", [1, 1], "BYTES")
        sid_input.set_data_from_numpy(np.array([[session_id]], dtype=object))

        final_input = grpcclient_aio.InferInput("is_final", [1, 1], "BOOL")
        final_input.set_data_from_numpy(np.array([[is_final]], dtype=bool))

        output = grpcclient_aio.InferRequestedOutput("diarization_output")

        response = await self.client.infer(
            model_name=self.model_name,
            inputs=[audio_input, sr_input, sid_input, final_input],
            outputs=[output],
            request_id=request_id,
        )

        return self._decode_json_output(response.as_numpy("diarization_output"))

    @staticmethod
    def _decode_json_output(raw_output: np.ndarray) -> Dict:
        """Decode Triton JSON payload with tolerant shape handling."""
        value = raw_output
        while isinstance(value, np.ndarray):
            if value.size == 0:
                return {}
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str):
            value = str(value)
        return json.loads(value)

    async def close(self):
        """Close the gRPC connection."""
        if self.client:
            await self.client.close()
            self.client = None
