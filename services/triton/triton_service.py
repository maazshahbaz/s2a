from .triton_asr import ASRTritonClient
from .triton_mistral import TritonMistralClient
from loguru import logger
import os


class TritonService:
    def __init__(self):
        # URL like "triton-server:2001" (from Docker Compose network)
        triton_url = os.getenv("TRITON_URL", "host.docker.internal:2001")
        model_asr_name = os.getenv("TRITON_ASR_MODEL", "asr_model")
        model_intelligence_name = os.getenv("TRITON_INTELLIGENCE_MODEL", "mistral-nemo")

        logger.info(f"Initializing Triton client (url={triton_url}, model={model_asr_name},{model_intelligence_name})")
        self.client_asr = ASRTritonClient(triton_url=triton_url, model_name=model_asr_name)
        self.client_intelligence = TritonMistralClient(triton_url=triton_url, model_name=model_intelligence_name)

        # Optional: pre-flight check
        try:
            if self.client_asr.client.is_server_ready() and self.client_asr.client.is_model_ready(model_asr_name):
                logger.info(f"✅ Triton ASR model '{model_asr_name}' is ready.")
            else:
                logger.warning(f"⚠️ Triton ASR model '{model_asr_name}' not ready.")
        except Exception as e:
            logger.error(f"Triton ASR connection failed: {e}")
        try:
            if self.client_asr.client.is_server_ready() and self.client_intelligence.client.is_model_ready(model_intelligence_name):
                logger.info(f"✅ Triton INTELLIGENCE model '{model_intelligence_name}' is ready.")
            else:
                logger.warning(f"⚠️ Triton INTELLIGENCE model '{model_intelligence_name}' not ready.")
        except Exception as e:
            logger.error(f"Triton INTELLIGENCE connection failed: {e}")

    def process_asr(self,  audio_file_path: str, request_id: str = None, on_complete=None):
        try:
            return self.client_asr.transcribe_async(
               audio_file_path,
               request_id,
               on_complete
            )
        except Exception as e:
            logger.error(f"Triton inference failed: {e}")
            return {"error": str(e)}
            
    def process_intelligence(self,  prompt: str, request_id: str = None, on_complete=None):
        try:
            return self.client_intelligence.analyze_call_async(
               prompt,
               request_id,
               on_complete
            )
        except Exception as e:
            logger.error(f"Triton inference failed: {e}")
            return {"error": str(e)}