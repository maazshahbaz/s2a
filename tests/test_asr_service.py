#!/usr/bin/env python3
"""
Unit tests for ASR Service core functionality
"""

import pytest
import asyncio
import numpy as np
import tempfile
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asr_service import NeMoASRService, TranscriptionResult
import soundfile as sf


class TestNeMoASRService:
    """Test cases for NeMo ASR Service"""

    @pytest.fixture
    def mock_audio_file(self):
        """Create a temporary audio file for testing"""
        # Generate 5 seconds of test audio
        duration = 5.0
        sample_rate = 16000
        audio = np.random.randn(int(duration * sample_rate)).astype(np.float32) * 0.1
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            yield tmp_file.name
        
        # Cleanup
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)

    @pytest.fixture
    def asr_service_cpu(self):
        """Create ASR service for CPU testing"""
        with patch('torch.cuda.is_available', return_value=False):
            service = NeMoASRService(
                model_name="nvidia/parakeet-tdt-0.6b-v2",
                device="cpu",
                batch_size=1,
                min_audio_duration=5.0  # Use default 5.0s minimum
            )
            return service

    def test_model_info_structure(self, asr_service_cpu):
        """Test model info returns correct structure"""
        info = asr_service_cpu.get_model_info()
        
        required_keys = [
            'model_name', 'model_type', 'device', 'batch_size',
            'max_chunk_duration', 'min_audio_duration', 'nemo_available'
        ]
        
        for key in required_keys:
            assert key in info
        
        assert info['device'] == 'cpu'
        assert info['batch_size'] == 1
        assert isinstance(info['nemo_available'], bool)

    def test_preprocess_audio_valid_file(self, asr_service_cpu, mock_audio_file):
        """Test audio preprocessing with valid file"""
        audio, duration, is_valid = asr_service_cpu.preprocess_audio(mock_audio_file)
        
        assert isinstance(audio, np.ndarray)
        assert duration > 0
        assert is_valid is True
        assert audio.dtype == np.float32 or audio.dtype == np.float64

    def test_preprocess_audio_short_duration(self, asr_service_cpu):
        """Test preprocessing with audio shorter than minimum duration"""
        # Create very short audio (1 second, below 5s minimum)
        duration = 1.0
        sample_rate = 16000
        audio = np.random.randn(int(duration * sample_rate)).astype(np.float32) * 0.1
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            
            try:
                audio_result, duration_result, is_valid = asr_service_cpu.preprocess_audio(tmp_file.name)
                assert is_valid is False
                assert duration_result < asr_service_cpu.min_audio_duration
            finally:
                os.unlink(tmp_file.name)

    def test_preprocess_audio_nonexistent_file(self, asr_service_cpu):
        """Test preprocessing with non-existent file"""
        with pytest.raises(FileNotFoundError):
            asr_service_cpu.preprocess_audio("nonexistent_file.wav")

    def test_chunk_audio_simple_short(self, asr_service_cpu):
        """Test simple chunking with short audio"""
        # 10 seconds of audio
        duration = 10.0
        sample_rate = 16000
        audio = np.random.randn(int(duration * sample_rate)).astype(np.float32)
        
        chunks = asr_service_cpu.chunk_audio_simple(audio, sample_rate)
        
        assert len(chunks) == 1  # Should be single chunk for short audio
        assert len(chunks[0]) == len(audio)

    def test_chunk_audio_simple_long(self, asr_service_cpu):
        """Test simple chunking with long audio"""
        # Test with audio longer than NeMo's 24-minute chunk size to force chunking
        duration = 30 * 60  # 30 minutes (longer than 24-minute NeMo chunks)
        sample_rate = 16000
        audio = np.random.randn(int(duration * sample_rate)).astype(np.float32)
        
        chunks = asr_service_cpu.chunk_audio_simple(audio, sample_rate)
        
        # For NeMo model, 30-minute audio should create multiple 24-minute chunks
        if asr_service_cpu.model_type == "nemo":
            # 30 minutes should create at least 2 chunks (24min + 6min)
            assert len(chunks) > 1, f"Expected multiple chunks for 30-min audio, got {len(chunks)}"
        else:
            # For Whisper, much smaller chunks expected
            assert len(chunks) > 1
        
        # Verify all chunks meet minimum duration requirement
        for chunk in chunks:
            chunk_duration = len(chunk) / sample_rate
            assert chunk_duration >= asr_service_cpu.min_audio_duration

    def test_transcription_result_structure(self):
        """Test TranscriptionResult structure"""
        result = TranscriptionResult(
            text="test transcription",
            duration=10.0,
            rtf=0.1,
            processing_time=1.0,
            model_used="whisper"
        )
        
        assert result.text == "test transcription"
        assert result.duration == 10.0
        assert result.rtf == 0.1
        assert result.processing_time == 1.0
        assert result.model_used == "whisper"
        assert result.chunks is None  # Default value

    @pytest.mark.asyncio
    async def test_transcribe_audio_mock(self, asr_service_cpu, mock_audio_file):
        """Test audio transcription with mocked NeMo model"""
        # Mock the NeMo transcription method since we're NeMo-only now
        with patch.object(asr_service_cpu, '_transcribe_with_nemo') as mock_transcribe:
            mock_transcribe.return_value = TranscriptionResult(
                text="mocked transcription",
                duration=5.0,
                rtf=0.1,
                processing_time=0.5,
                model_used="nemo",
                chunks_processed=1
            )
            
            result = await asr_service_cpu.transcribe_audio(mock_audio_file)
            
            assert isinstance(result, TranscriptionResult)
            assert result.text == "mocked transcription"
            assert result.duration == 5.0
            assert result.model_used == "nemo"

    def test_stitch_transcriptions_simple(self, asr_service_cpu):
        """Test simple transcription stitching"""
        chunk_results = [
            {"text": "Hello", "duration": 2.0, "rtf": 0.1},
            {"text": "world", "duration": 2.0, "rtf": 0.1},
            {"text": "test", "duration": 1.0, "rtf": 0.1}
        ]
        
        stitched = asr_service_cpu.stitch_transcriptions(chunk_results)
        assert stitched == "Hello world test"

    def test_stitch_transcriptions_with_errors(self, asr_service_cpu):
        """Test stitching with some failed chunks"""
        chunk_results = [
            {"text": "Hello", "duration": 2.0, "rtf": 0.1},
            {"text": "", "duration": 2.0, "rtf": 0.1, "error": "processing failed"},
            {"text": "world", "duration": 1.0, "rtf": 0.1}
        ]
        
        stitched = asr_service_cpu.stitch_transcriptions(chunk_results)
        assert stitched == "Hello world"

    def test_stitch_transcriptions_empty_results(self, asr_service_cpu):
        """Test stitching with empty results"""
        chunk_results = []
        
        stitched = asr_service_cpu.stitch_transcriptions(chunk_results)
        assert stitched == ""

    def test_device_configuration(self):
        """Test device configuration"""
        # Test CPU device
        service_cpu = NeMoASRService(device="cpu")
        assert service_cpu.device == "cpu"
        
        # Test auto device selection
        with patch('torch.cuda.is_available', return_value=False):
            service_auto = NeMoASRService(device="cuda")
            # Should fall back to available device based on torch

    def test_batch_size_configuration(self):
        """Test batch size configuration"""
        service = NeMoASRService(batch_size=8)
        assert service.batch_size == 8
        
        info = service.get_model_info()
        assert info['batch_size'] == 8

    def test_duration_limits(self):
        """Test duration limit configurations"""
        service = NeMoASRService(
            min_audio_duration=3.0,
            max_chunk_duration=30 * 60  # 30 minutes
        )
        
        assert service.min_audio_duration == 3.0
        assert service.max_chunk_duration == 30 * 60
        
        info = service.get_model_info()
        assert info['min_audio_duration'] == 3.0
        assert info['max_chunk_duration'] == 30 * 60


if __name__ == "__main__":
    pytest.main([__file__])