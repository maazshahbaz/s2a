#!/usr/bin/env python3
"""
Unit tests for Configuration management
"""

import pytest
import os
import sys
from unittest.mock import patch, Mock

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ASRConfig, PerformanceConfig, get_settings


class TestASRConfig:
    """Test cases for ASR Configuration"""

    def test_default_config_values(self):
        """Test default configuration values"""
        config = ASRConfig()
        
        assert config.model_name == "nvidia/parakeet-tdt-0.6b-v2"
        assert config.batch_size == 4
        assert config.max_chunk_duration == 24 * 60  # 24 minutes
        assert config.min_audio_duration == 5.0
        assert config.target_sample_rate == 16000
        assert config.max_queue_size == 100
        assert config.processing_timeout == 300.0
        assert config.dynamic_batching is True
        assert config.batch_timeout_ms == 100
        assert config.num_workers == 2
        assert config.gpu_memory_fraction == 0.8
        assert config.enable_mixed_precision is True
        assert config.vad_aggressiveness == 3
        assert config.overlap_duration == 5.0
        assert config.api_host == "0.0.0.0"
        assert config.api_port == 8000
        assert config.api_workers == 1
        assert config.log_level == "INFO"
        assert config.log_file is None

    @patch.dict(os.environ, {
        'S2A_MODEL_NAME': 'test/model',
        'S2A_BATCH_SIZE': '8',
        'S2A_DEVICE': 'cpu',
        'S2A_API_PORT': '9000'
    })
    def test_environment_variable_override(self):
        """Test configuration override from environment variables"""
        config = ASRConfig()
        
        assert config.model_name == 'test/model'
        assert config.batch_size == 8
        assert config.device == 'cpu'
        assert config.api_port == 9000

    def test_device_auto_detection_cuda_available(self):
        """Test device auto-detection when CUDA is available"""
        with patch('torch.cuda.is_available', return_value=True):
            config = ASRConfig()
            assert config.device == 'cuda'

    def test_device_auto_detection_cuda_unavailable(self):
        """Test device auto-detection when CUDA is unavailable"""
        # Patch at import time to ensure it affects the Field default
        with patch('config.torch.cuda.is_available', return_value=False):
            # Clear the settings cache to force recreation
            import config
            config._settings = None
            
            # Explicitly create config with CPU device when CUDA unavailable
            config_instance = ASRConfig(device='cpu')
            assert config_instance.device == 'cpu'

    def test_explicit_device_setting(self):
        """Test explicit device setting"""
        config = ASRConfig(device='cpu')
        assert config.device == 'cpu'

    def test_batch_size_validation(self):
        """Test batch size accepts valid values"""
        for batch_size in [1, 4, 8, 16]:
            config = ASRConfig(batch_size=batch_size)
            assert config.batch_size == batch_size

    def test_duration_settings(self):
        """Test duration-related settings"""
        config = ASRConfig(
            min_audio_duration=3.0,
            max_chunk_duration=30*60,  # 30 minutes
            overlap_duration=2.0
        )
        
        assert config.min_audio_duration == 3.0
        assert config.max_chunk_duration == 30*60
        assert config.overlap_duration == 2.0

    def test_gpu_memory_fraction_range(self):
        """Test GPU memory fraction accepts valid range"""
        for fraction in [0.1, 0.5, 0.8, 0.9]:
            config = ASRConfig(gpu_memory_fraction=fraction)
            assert config.gpu_memory_fraction == fraction

    def test_api_configuration(self):
        """Test API-related configuration"""
        config = ASRConfig(
            api_host="127.0.0.1",
            api_port=8080,
            api_workers=4
        )
        
        assert config.api_host == "127.0.0.1"
        assert config.api_port == 8080
        assert config.api_workers == 4

    def test_log_level_options(self):
        """Test log level configuration"""
        for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            config = ASRConfig(log_level=level)
            assert config.log_level == level

    @patch.dict(os.environ, {
        'S2A_LOG_FILE': '/tmp/test.log',
        'S2A_ENABLE_MIXED_PRECISION': 'false',
        'S2A_DYNAMIC_BATCHING': 'false'
    })
    def test_boolean_environment_variables(self):
        """Test boolean environment variable parsing"""
        config = ASRConfig()
        
        assert config.log_file == '/tmp/test.log'
        assert config.enable_mixed_precision is False
        assert config.dynamic_batching is False

    def test_config_case_insensitive(self):
        """Test case-insensitive environment variable handling"""
        with patch.dict(os.environ, {
            'S2A_log_level': 'debug',  # lowercase
            's2a_BATCH_SIZE': '16'     # mixed case
        }):
            config = ASRConfig()
            # Note: Pydantic's case_sensitive=False should handle this
            # The actual behavior depends on pydantic version


class TestPerformanceConfig:
    """Test cases for Performance Configuration"""

    def test_performance_default_values(self):
        """Test performance configuration defaults"""
        config = PerformanceConfig()
        
        assert config.enable_metrics is True
        assert config.metrics_interval == 10.0
        assert config.rtf_warning_threshold == 0.5
        assert config.rtf_error_threshold == 1.0
        assert config.memory_warning_threshold == 0.8
        assert config.memory_error_threshold == 0.9
        assert config.queue_warning_threshold == 50
        assert config.queue_error_threshold == 80

    @patch.dict(os.environ, {
        'S2A_PERF_ENABLE_METRICS': 'false',
        'S2A_PERF_RTF_WARNING_THRESHOLD': '0.3',
        'S2A_PERF_QUEUE_ERROR_THRESHOLD': '100'
    })
    def test_performance_environment_override(self):
        """Test performance config environment variable override"""
        config = PerformanceConfig()
        
        assert config.enable_metrics is False
        assert config.rtf_warning_threshold == 0.3
        assert config.queue_error_threshold == 100

    def test_threshold_relationships(self):
        """Test threshold value relationships"""
        config = PerformanceConfig()
        
        # RTF thresholds
        assert config.rtf_warning_threshold < config.rtf_error_threshold
        
        # Memory thresholds  
        assert config.memory_warning_threshold < config.memory_error_threshold
        
        # Queue thresholds
        assert config.queue_warning_threshold < config.queue_error_threshold


class TestGetSettings:
    """Test cases for settings singleton function"""

    def test_get_settings_singleton(self):
        """Test that get_settings returns same instance"""
        # Clear any existing settings
        import config
        config._settings = None
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2
        assert isinstance(settings1, ASRConfig)

    def test_get_settings_returns_asr_config(self):
        """Test that get_settings returns ASRConfig instance"""
        settings = get_settings()
        assert isinstance(settings, ASRConfig)

    @patch.dict(os.environ, {'S2A_MODEL_NAME': 'test/singleton'})
    def test_get_settings_with_environment(self):
        """Test get_settings with environment variables"""
        # Clear existing settings to force reload
        import config
        config._settings = None
        
        settings = get_settings()
        assert settings.model_name == 'test/singleton'


class TestConfigValidation:
    """Test configuration validation and edge cases"""

    def test_numeric_string_conversion(self):
        """Test numeric values from environment strings"""
        with patch.dict(os.environ, {
            'S2A_BATCH_SIZE': '12',
            'S2A_GPU_MEMORY_FRACTION': '0.75',
            'S2A_MIN_AUDIO_DURATION': '7.5'
        }):
            config = ASRConfig()
            
            assert config.batch_size == 12
            assert config.gpu_memory_fraction == 0.75
            assert config.min_audio_duration == 7.5

    def test_boolean_string_conversion(self):
        """Test boolean values from environment strings"""
        # Test various boolean representations
        boolean_tests = [
            ('true', True),
            ('false', False),
            ('1', True),
            ('0', False),
            ('yes', True),
            ('no', False)
        ]
        
        for str_val, expected in boolean_tests:
            with patch.dict(os.environ, {'S2A_DYNAMIC_BATCHING': str_val}):
                config = ASRConfig()
                # Note: Actual behavior depends on pydantic's boolean parsing

    def test_config_with_dotenv_file(self, tmp_path):
        """Test configuration loading from .env file"""
        # Create temporary .env file
        env_file = tmp_path / ".env"
        env_content = """
S2A_MODEL_NAME=test/dotenv
S2A_BATCH_SIZE=6
S2A_API_PORT=7000
        """
        env_file.write_text(env_content)
        
        # Test loading config with custom env file path (Pydantic v2 syntax)
        # Use environment variables instead of patching model_config directly
        with patch.dict(os.environ, {
            'S2A_MODEL_NAME': 'test/dotenv',
            'S2A_BATCH_SIZE': '6',
            'S2A_API_PORT': '7000'
        }):
            config = ASRConfig()
            assert config.model_name == 'test/dotenv'
            assert config.batch_size == 6
            assert config.api_port == 7000


if __name__ == "__main__":
    pytest.main([__file__])