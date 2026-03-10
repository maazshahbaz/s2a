from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import torch

class ASRConfig(BaseSettings):
    # Model configuration
    model_name: str = Field(default="nvidia/parakeet-tdt-0.6b-v2", description="HuggingFace model name")
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu", description="Device to use")
    # ASR Model parameters
    batch_size: int = Field(default=128, description="Batch size for GPU processing")
    max_chunk_duration: float = Field(default=24 * 60, description="Maximum chunk duration in seconds (24 min for Parakeet)")
    min_audio_duration: float = Field(default=1.0, description="Minimum audio duration to process")
    max_audio_duration: float = Field(default=5 * 60 * 60, description="Maximum audio duration (5 hours)")
    target_sample_rate: int = Field(default=16000, description="Target sample rate for audio")
    overlap_duration: float = Field(default=5.0, description="Overlap between chunks in seconds")

    # GPU optimization
    gpu_memory_fraction: float = Field(default=0.8, description="Fraction of GPU memory to use")
    enable_mixed_precision: bool = Field(default=True, description="Enable mixed precision training")

    # Chunking and stitching parameters
    words_per_second: float = Field(default=3.0, description="Average speaking rate (words/second) for overlap estimation (3.0 = 180 WPM)")
    overlap_similarity_threshold: float = Field(default=0.8, description="Minimum similarity (0-1) for fuzzy overlap detection in stitching")

    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
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

    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    model_config = {
        "env_prefix": "S2A_",
        "case_sensitive": False,
        "extra": "ignore"  # Allow extra fields in .env
    }

# Performance monitoring configuration
class PerformanceConfig(BaseSettings):
    enable_metrics: bool = Field(default=True, description="Enable performance metrics collection")
    metrics_interval: float = Field(default=10.0, description="Metrics collection interval in seconds")
    rtf_warning_threshold: float = Field(default=0.5, description="RTF threshold for warnings")
    rtf_error_threshold: float = Field(default=1.0, description="RTF threshold for errors")
    
    # Memory monitoring
    memory_warning_threshold: float = Field(default=0.8, description="Memory usage warning threshold")
    memory_error_threshold: float = Field(default=0.9, description="Memory usage error threshold")
    
    # Queue monitoring
    queue_warning_threshold: int = Field(default=50, description="Queue size warning threshold")
    queue_error_threshold: int = Field(default=80, description="Queue size error threshold")
    
    model_config = {
        "env_prefix": "S2A_PERF_",
        "extra": "ignore"
    }



# Global settings instances
_settings = None

def get_settings() -> ASRConfig:
    global _settings
    if _settings is None:
        _settings = ASRConfig()
    return _settings
