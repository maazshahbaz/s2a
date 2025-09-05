#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for S2A tests
"""

import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import soundfile as sf
from asr_service import NeMoASRService, TranscriptionResult
from audio_utils import AudioProcessor
from config import ASRConfig


@pytest.fixture(scope="session")
def test_data_dir():
    """Create and return test data directory"""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def sample_audio_5sec():
    """Generate 5-second test audio file"""
    duration = 5.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create realistic speech-like signal
    # Combine multiple frequencies to simulate speech formants
    audio = (np.sin(2 * np.pi * 440 * t) * 0.5 +  # F1
             np.sin(2 * np.pi * 880 * t) * 0.3 +   # F2  
             np.sin(2 * np.pi * 1760 * t) * 0.2)   # F3
    
    # Apply envelope to simulate speech dynamics
    envelope = np.exp(-t/10) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    audio = audio * envelope
    
    # Add some noise
    audio += 0.05 * np.random.randn(len(audio))
    audio = audio.astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sample_rate)
        yield tmp_file.name
    
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture
def sample_audio_short():
    """Generate short test audio file (2 seconds - below minimum)"""
    duration = 2.0
    sample_rate = 16000
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
    audio = audio.astype(np.float32) * 0.5
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sample_rate)
        yield tmp_file.name
    
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture  
def sample_audio_long():
    """Generate long test audio file (2 minutes)"""
    duration = 120.0  # 2 minutes
    sample_rate = 16000
    
    # Generate in chunks to avoid memory issues
    chunk_duration = 10.0
    audio_chunks = []
    
    for i in range(int(duration / chunk_duration)):
        start_time = i * chunk_duration
        t = np.linspace(start_time, start_time + chunk_duration, 
                       int(sample_rate * chunk_duration))
        
        # Varying frequency to simulate speech
        freq = 440 + 200 * np.sin(2 * np.pi * start_time / 30)
        chunk = np.sin(2 * np.pi * freq * t) * 0.5
        chunk += 0.05 * np.random.randn(len(chunk))
        audio_chunks.append(chunk.astype(np.float32))
    
    audio = np.concatenate(audio_chunks)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sample_rate)
        yield tmp_file.name
    
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture
def mock_transcription_result():
    """Mock transcription result for testing"""
    return TranscriptionResult(
        text="This is a test transcription",
        duration=5.0,
        rtf=0.1,
        processing_time=0.5,
        model_used="mock_model",
        chunks_processed=1,
        confidence=0.95
    )


@pytest.fixture
def audio_processor():
    """Create AudioProcessor instance for testing"""
    return AudioProcessor(target_sr=16000, vad_aggressiveness=3)


@pytest.fixture
def asr_config_test():
    """Create test ASR configuration"""
    return ASRConfig(
        model_name="test/model",
        device="cpu",
        batch_size=2,
        min_audio_duration=1.0,  # Lower for testing
        max_chunk_duration=60,   # Smaller for testing
        processing_timeout=30.0  # Shorter for testing
    )


@pytest.fixture
def mock_asr_service():
    """Create mock ASR service for testing"""
    service = Mock(spec=NeMoASRService)
    service.model_type = "mock"
    service.device = "cpu"
    service.batch_size = 2
    service.min_audio_duration = 1.0
    service.max_chunk_duration = 60
    
    service.get_model_info.return_value = {
        'model_name': 'test/model',
        'model_type': 'mock',
        'device': 'cpu',
        'batch_size': 2,
        'max_chunk_duration': 60,
        'min_audio_duration': 1.0,
        'nemo_available': False,
        'chunking_strategy': 'simple'
    }
    
    return service


@pytest.fixture
def silence_audio():
    """Generate silent audio for testing"""
    duration = 3.0
    sample_rate = 16000
    audio = np.zeros(int(sample_rate * duration), dtype=np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sample_rate)
        yield tmp_file.name
    
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture
def noisy_audio():
    """Generate noisy audio for testing enhancement"""
    duration = 4.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Signal
    signal = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Add significant noise
    noise = np.random.randn(len(signal)) * 0.3
    audio = signal + noise
    audio = audio.astype(np.float32)
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sample_rate)
        yield tmp_file.name
    
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


def generate_synthetic_speech(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """
    Generate synthetic speech-like audio signal
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Audio signal as numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Speech formants (approximate)
    f1 = 700  # First formant
    f2 = 1220  # Second formant  
    f3 = 2600  # Third formant
    
    # Generate formant signals with time-varying amplitude
    formant1 = np.sin(2 * np.pi * f1 * t) * 0.6
    formant2 = np.sin(2 * np.pi * f2 * t) * 0.4
    formant3 = np.sin(2 * np.pi * f3 * t) * 0.2
    
    # Combine formants
    speech = formant1 + formant2 + formant3
    
    # Apply speech-like envelope (amplitude modulation)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz modulation
    speech = speech * envelope
    
    # Add some noise to make it more realistic
    noise = np.random.randn(len(speech)) * 0.02
    speech = speech + noise
    
    return speech.astype(np.float32)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Ensure reproducible random numbers
    np.random.seed(42)
    
    # Mock CUDA availability to avoid GPU dependencies in tests
    with patch('torch.cuda.is_available', return_value=False):
        yield


@pytest.fixture
def temp_audio_files(tmp_path):
    """Create multiple temporary audio files for batch testing"""
    files = []
    
    for i in range(3):
        duration = 3.0 + i  # Different durations
        sample_rate = 16000
        audio = generate_synthetic_speech(duration, sample_rate)
        
        file_path = tmp_path / f"test_audio_{i}.wav"
        sf.write(str(file_path), audio, sample_rate)
        files.append(str(file_path))
    
    return files


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may require GPU or long processing time)"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid.lower() or "cuda" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark slow tests 
        if "long" in item.nodeid.lower() or "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "validation" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)