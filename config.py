from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import torch

class ASRConfig(BaseSettings):
    # Model configuration
    model_name: str = Field(default="nvidia/parakeet-tdt-0.6b-v2", description="HuggingFace model name")
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu", description="Device to use")
    batch_size: int = Field(default=4, description="Batch size")
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
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    model_config = {
        "env_prefix": "S2A_",
        "case_sensitive": False,
        "extra": "ignore"  # Allow extra fields in .env
    }

# Redis configuration for chunk queue system (Required)
class RedisConfig(BaseSettings):
    # Redis connection (required)
    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    db: int = Field(default=0, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")

    # Queue configuration
    queue_prefix: str = Field(default="stt", description="Prefix for Redis keys")
    chunk_ttl: int = Field(default=86400, description="TTL for chunk data in seconds (24 hours)")

    # Batch processing configuration
    batch_size: int = Field(default=128, description="Max chunks per GPU batch (can mix jobs)")
    num_workers: int = Field(default=1, description="Number of concurrent workers (1 is sufficient with batch_size=128)")

    # Audio caching
    audio_cache_size: int = Field(default=10, description="Number of audio files to cache in memory")

    model_config = {
        "env_prefix": "S2A_REDIS_",
        "case_sensitive": False,
        "extra": "ignore"
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
_redis_settings = None
_diarization_settings = None

def get_settings() -> ASRConfig:
    global _settings
    if _settings is None:
        _settings = ASRConfig()
    return _settings

def get_redis_settings() -> RedisConfig:
    global _redis_settings
    if _redis_settings is None:
        _redis_settings = RedisConfig()
    return _redis_settings


# Diarization configuration
class DiarizationConfig(BaseSettings):
    enabled: bool = Field(default=True, description="Enable diarization pipeline (mandatory in API)")
    model_name: str = Field(default="nvidia/diar_sortformer_4spk-v1", description="Diarization model from HuggingFace")
    max_speakers: int = Field(default=4, description="Maximum number of speakers")
    timeout_seconds: float = Field(default=120.0, description="Diarization timeout budget")

    model_config = {
        "env_prefix": "S2A_DIAR_",
        "case_sensitive": False,
        "extra": "ignore"
    }


def get_diarization_settings() -> DiarizationConfig:
    global _diarization_settings
    if _diarization_settings is None:
        _diarization_settings = DiarizationConfig()
    return _diarization_settings
