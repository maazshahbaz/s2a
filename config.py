from pydantic_settings import BaseSettings
from pydantic import Field
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
    max_sync_audio_duration: float = Field(default=2 * 60, description="Maximum audio duration for sync API (2 minutes)")
    max_async_audio_duration: float = Field(default=2 * 60 * 60, description="Maximum audio duration for async API (2 hours)")
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
    
    model_config = {
        "env_prefix": "S2A_",
        "env_file": ".env", 
        "case_sensitive": False,
        "extra": "ignore"  # Allow extra fields in .env
    }

# Global settings instances
_settings = None
_intelligence_settings = None
_intelligence_metrics_settings = None

def get_settings() -> ASRConfig:
    global _settings
    if _settings is None:
        _settings = ASRConfig()
    return _settings

def get_intelligence_settings() -> IntelligenceConfig:
    global _intelligence_settings
    if _intelligence_settings is None:
        _intelligence_settings = IntelligenceConfig()
    return _intelligence_settings

def get_intelligence_metrics_settings() -> IntelligenceMetricsConfig:
    global _intelligence_metrics_settings
    if _intelligence_metrics_settings is None:
        _intelligence_metrics_settings = IntelligenceMetricsConfig()
    return _intelligence_metrics_settings

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
        "env_file": ".env",
        "extra": "ignore"
    }


# Intelligence processing configuration
class IntelligenceConfig(BaseSettings):
    # Core intelligence settings
    enabled: bool = Field(default=True, description="Enable intelligence extraction pipeline")
    vllm_base_url: str = Field(default="http://localhost:8000/v1", description="vLLM API server base URL")
    model_name: str = Field(default="Qwen/Qwen2.5-7B-Instruct", description="Language model for intelligence extraction")
    
    # Processing parameters
    temperature: float = Field(default=0.2, description="Model temperature for extraction")
    max_tokens: int = Field(default=500, description="Maximum tokens for model response")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    timeout_seconds: float = Field(default=30.0, description="API request timeout in seconds")
    
    # Memory and performance
    gpu_memory_utilization: float = Field(default=0.6, description="GPU memory fraction for vLLM (when co-located with ASR)")
    max_model_len: int = Field(default=16000, description="Maximum model context length")
    
    # Processing behavior
    auto_process: bool = Field(default=True, description="Automatically process transcriptions for intelligence")
    retry_attempts: int = Field(default=2, description="Number of retry attempts for failed extractions")
    batch_processing: bool = Field(default=True, description="Enable batch processing of intelligence jobs")
    max_batch_size: int = Field(default=8, description="Maximum batch size for intelligence processing")
    
    # Queue configuration
    queue_max_size: int = Field(default=200, description="Maximum intelligence processing queue size")
    processing_timeout: float = Field(default=600.0, description="Intelligence processing timeout in seconds")
    queue_check_interval: float = Field(default=5.0, description="Interval to check queue for new jobs")
    
    # Output configuration  
    save_raw_responses: bool = Field(default=False, description="Save raw LLM responses for debugging")
    confidence_threshold: float = Field(default=0.6, description="Minimum confidence score for valid extractions")
    
    # Integration settings
    webhook_intelligence_enabled: bool = Field(default=True, description="Include intelligence data in webhook callbacks")
    storage_enabled: bool = Field(default=True, description="Store extracted intelligence data persistently")
    
    model_config = {
        "env_prefix": "S2A_INTEL_",
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore"
    }


# Intelligence metrics and monitoring configuration
class IntelligenceMetricsConfig(BaseSettings):
    # Metrics collection
    enable_metrics: bool = Field(default=True, description="Enable intelligence metrics collection")
    metrics_interval: float = Field(default=15.0, description="Metrics collection interval in seconds")
    
    # Performance thresholds
    extraction_latency_warning: float = Field(default=5.0, description="Extraction latency warning threshold (seconds)")
    extraction_latency_error: float = Field(default=15.0, description="Extraction latency error threshold (seconds)")
    
    # Success rate monitoring
    success_rate_warning: float = Field(default=0.7, description="Success rate warning threshold (0-1)")
    success_rate_error: float = Field(default=0.5, description="Success rate error threshold (0-1)")
    
    # Queue monitoring
    queue_warning_threshold: int = Field(default=100, description="Intelligence queue warning threshold")
    queue_error_threshold: int = Field(default=150, description="Intelligence queue error threshold")
    
    # Field extraction rates
    field_hit_rate_warning: float = Field(default=0.4, description="Field hit rate warning threshold (0-1)")
    
    model_config = {
        "env_prefix": "S2A_INTEL_METRICS_",
        "env_file": ".env",
        "extra": "ignore"
    }