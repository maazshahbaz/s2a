from client_triton import TritonClient
from loguru import logger
import os


class TritonService:
    def __init__(self):
        # URL like "triton-server:2001" (from Docker Compose network)
        triton_url = os.getenv("TRITON_URL", "triton-server:2001")
        model_name = os.getenv("TRITON_MODEL", "llama-8b-instruct")

        logger.info(f"Initializing Triton client (url={triton_url}, model={model_name})")
        self.client = TritonClient(url=triton_url, model_name=model_name)

        # Optional: pre-flight check
        try:
            if self.client.client.is_server_ready() and self.client.client.is_model_ready(model_name):
                logger.info(f"✅ Triton model '{model_name}' is ready.")
            else:
                logger.warning(f"⚠️ Triton model '{model_name}' not ready.")
        except Exception as e:
            logger.error(f"Triton connection failed: {e}")

    def analyze(self, transcription: str, max_tokens: int = 512, temperature: float = 0.3, top_p: float = 0.9):
        try:
            return self.client.analyze_call(
                transcription=transcription,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        except Exception as e:
            logger.error(f"Triton inference failed: {e}")
            return {"error": str(e)}
