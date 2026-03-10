from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional

class ASRConfig(BaseSettings):
    # ASR validation limits
    min_audio_duration: float = Field(default=1.0, description="Minimum audio duration to process")
    max_audio_duration: float = Field(default=5 * 60 * 60, description="Maximum audio duration (5 hours)")
    
    # Staging
    staging_mode: bool = Field(default=False, description="Staging mode - skip auth and database")

    # Streaming
    streaming_max_concurrent_sessions: int = Field(default=50, description="Max concurrent streaming sessions")
    streaming_chunk_duration: float = Field(default=1.0, description="Audio chunk duration in seconds for streaming inference")
    streaming_audiosocket_port: int = Field(default=8003, description="TCP port for AudioSocket connections")
    streaming_session_start_timeout_seconds: float = Field(
        default=15.0,
        description="Timeout waiting for session.start control message",
    )
    streaming_idle_timeout_seconds: float = Field(
        default=30.0,
        description="Timeout for idle streaming connections",
    )
    streaming_max_session_duration_seconds: float = Field(
        default=4 * 60 * 60,
        description="Hard limit for streaming session duration in seconds",
    )
    streaming_max_bytes_per_second: int = Field(
        default=64000,
        description="Maximum sustained ingress audio bytes per second per session",
    )
    streaming_max_frame_bytes: int = Field(
        default=32768,
        description="Maximum bytes allowed in a single streaming audio frame",
    )
    streaming_inference_timeout_seconds: float = Field(
        default=20.0,
        description="Timeout for a single streaming ASR/diarization inference request",
    )
    streaming_default_callback_url: Optional[str] = Field(
        default=None,
        description="Default callback URL used when stream clients do not provide one",
    )
    streaming_allowed_ips: Optional[str] = Field(
        default=None,
        description="Optional comma-separated IP/CIDR allowlist for streaming endpoints",
    )
    
    model_config = {
        "env_prefix": "S2A_",
        "case_sensitive": False,
        "extra": "ignore"  # Allow extra fields in .env
    }



# Global settings instances
_settings = None

def get_settings() -> ASRConfig:
    global _settings
    if _settings is None:
        _settings = ASRConfig()
    return _settings
