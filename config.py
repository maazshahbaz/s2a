from pydantic import BaseSettings, Field
from typing import Optional
import torch

class ASRConfig(BaseSettings):
    # Model configuration
    model_name: str = Field(default="nvidia/parakeet-tdt-0.6b-v2", description="HuggingFace model name")
    device: str = Field(default="cuda" if torch.cuda.is_available() else "cpu", description="Device to use")
    
    # Processing parameters
    batch_size: int = Field(default=4, description="Batch size for processing")
    max_chunk_duration: float = Field(default=24 * 60, description="Maximum chunk duration in seconds")
    min_audio_duration: float = Field(default=5.0, description="Minimum audio duration to process")
    target_sample_rate: int = Field(default=16000, description="Target sample rate for audio")
    
    # Batch processing
    max_queue_size: int = Field(default=100, description="Maximum queue size for batch processing")
    processing_timeout: float = Field(default=300.0, description="Processing timeout in seconds")
    dynamic_batching: bool = Field(default=True, description="Enable dynamic batching")
    batch_timeout_ms: int = Field(default=100, description="Batch collection timeout in milliseconds")
    num_workers: int = Field(default=2, description="Number of worker processes")
    
    # GPU optimization
    gpu_memory_fraction: float = Field(default=0.8, description="Fraction of GPU memory to use")
    enable_mixed_precision: bool = Field(default=True, description="Enable mixed precision training")
    
    # Audio processing
    vad_aggressiveness: int = Field(default=3, description="Voice activity detection aggressiveness (0-3)")
    overlap_duration: float = Field(default=5.0, description="Overlap duration between chunks in seconds")
    
    # API configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=1, description="Number of API workers")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    class Config:
        env_prefix = "S2A_"
        env_file = ".env"
        case_sensitive = False

# Global settings instance
_settings = None

def get_settings() -> ASRConfig:
    global _settings
    if _settings is None:
        _settings = ASRConfig()
    return _settings

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
    
    class Config:
        env_prefix = "S2A_PERF_"
        env_file = ".env"