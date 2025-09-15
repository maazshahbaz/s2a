#!/usr/bin/env python3
"""
Unit tests for Audio Processing utilities
"""

import pytest
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, Mock

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.audio_utils import AudioProcessor
import soundfile as sf
from pydub import AudioSegment


class TestAudioProcessor:
    """Test cases for Audio Processor"""

    @pytest.fixture
    def audio_processor(self):
        """Create audio processor instance"""
        return AudioProcessor(target_sr=16000, vad_aggressiveness=3)

    @pytest.fixture
    def sample_audio_wav(self):
        """Create sample WAV file"""
        duration = 5.0
        sample_rate = 16000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        audio = audio.astype(np.float32)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sample_rate)
            yield tmp_file.name
        
        if os.path.exists(tmp_file.name):
            os.unlink(tmp_file.name)

    @pytest.fixture
    def sample_audio_mp3(self):
        """Create sample MP3 file"""
        duration = 3.0
        sample_rate = 16000
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        
        # Create WAV first, then convert to MP3
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            sf.write(wav_file.name, audio.astype(np.float32), sample_rate)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
                audio_segment = AudioSegment.from_wav(wav_file.name)
                audio_segment.export(mp3_file.name, format="mp3")
                yield mp3_file.name
        
        # Cleanup
        for file_path in [wav_file.name, mp3_file.name]:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_audio_processor_initialization(self):
        """Test AudioProcessor initialization"""
        processor = AudioProcessor(target_sr=22050, vad_aggressiveness=2)
        
        assert processor.target_sr == 22050
        assert processor.vad.aggressiveness == 2

    def test_convert_to_wav_already_wav(self, audio_processor, sample_audio_wav):
        """Test WAV conversion when file is already WAV"""
        result_path = audio_processor.convert_to_wav(sample_audio_wav)
        
        assert result_path == Path(sample_audio_wav)

    def test_convert_to_wav_from_mp3(self, audio_processor, sample_audio_mp3):
        """Test WAV conversion from MP3"""
        result_path = audio_processor.convert_to_wav(sample_audio_mp3)
        
        assert result_path.suffix == '.wav'
        assert result_path.exists()
        
        # Verify audio can be loaded
        audio, sr = sf.read(str(result_path))
        assert sr == audio_processor.target_sr
        assert len(audio) > 0
        
        # Cleanup
        if result_path.exists():
            result_path.unlink()

    def test_normalize_audio_peak(self, audio_processor):
        """Test peak normalization"""
        audio = np.array([0.5, -0.8, 0.3, -0.2], dtype=np.float32)
        normalized = audio_processor.normalize_audio(audio, method="peak")
        
        assert np.max(np.abs(normalized)) == 1.0
        assert normalized.dtype == audio.dtype

    def test_normalize_audio_rms(self, audio_processor):
        """Test RMS normalization"""
        audio = np.array([0.5, -0.8, 0.3, -0.2], dtype=np.float32)
        normalized = audio_processor.normalize_audio(audio, method="rms")
        
        rms = np.sqrt(np.mean(normalized**2))
        assert abs(rms - 0.1) < 0.01  # Target RMS is 0.1

    def test_apply_preemphasis(self, audio_processor):
        """Test preemphasis filter"""
        audio = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        preemphasized = audio_processor.apply_preemphasis(audio, coeff=0.97)
        
        assert len(preemphasized) == len(audio)
        assert preemphasized[0] == audio[0]  # First sample unchanged

    def test_remove_silence(self, audio_processor):
        """Test silence removal"""
        # Create audio with silence (zeros) and signal
        sample_rate = 16000
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sample_rate))
        silence = np.zeros(sample_rate)
        audio = np.concatenate([silence, signal, silence])
        
        trimmed, compression_ratio = audio_processor.remove_silence(audio, sample_rate)
        
        assert len(trimmed) < len(audio)
        assert 0 < compression_ratio < 1

    def test_detect_voice_activity(self, audio_processor):
        """Test voice activity detection"""
        sample_rate = 16000
        duration = 2.0
        
        # Create audio with speech-like characteristics
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        audio = audio.astype(np.float32)
        
        vad_frames = audio_processor.detect_voice_activity(audio, sample_rate)
        
        assert isinstance(vad_frames, np.ndarray)
        assert len(vad_frames) > 0
        assert all(isinstance(frame, (bool, np.bool_)) for frame in vad_frames)

    def test_apply_noise_reduction(self, audio_processor):
        """Test noise reduction"""
        sample_rate = 16000
        duration = 2.0
        
        # Create audio with noise
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        noise = np.random.randn(len(signal)) * 0.1
        audio = signal + noise
        
        enhanced = audio_processor.apply_noise_reduction(audio.astype(np.float32), sample_rate)
        
        assert isinstance(enhanced, np.ndarray)
        assert len(enhanced) > 0

    def test_apply_band_pass_filter(self, audio_processor):
        """Test band-pass filtering"""
        sample_rate = 16000
        duration = 1.0
        audio = np.random.randn(int(sample_rate * duration)).astype(np.float32)
        
        filtered = audio_processor.apply_band_pass_filter(
            audio, sample_rate, low_freq=300, high_freq=3400
        )
        
        assert isinstance(filtered, np.ndarray)
        assert len(filtered) == len(audio)

    def test_enhance_audio(self, audio_processor):
        """Test complete audio enhancement pipeline"""
        sample_rate = 16000
        duration = 2.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        audio = audio.astype(np.float32)
        
        enhanced, info = audio_processor.enhance_audio(
            audio, sample_rate,
            apply_noise_reduction=True,
            apply_filtering=True,
            remove_silence=False
        )
        
        assert isinstance(enhanced, np.ndarray)
        assert isinstance(info, dict)
        assert 'filtered' in info
        assert 'noise_reduced' in info
        assert 'original_length' in info
        assert 'final_length' in info

    def test_validate_audio_quality(self, audio_processor):
        """Test audio quality validation"""
        sample_rate = 16000
        duration = 2.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
        audio = audio.astype(np.float32)
        
        metrics = audio_processor.validate_audio_quality(audio, sample_rate)
        
        assert isinstance(metrics, dict)
        
        expected_keys = [
            'dynamic_range_db', 'mean_zcr', 'rms_energy',
            'spectral_centroid_hz', 'voice_activity_ratio'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))

    def test_process_audio_file_wav(self, audio_processor, sample_audio_wav):
        """Test processing WAV file"""
        audio, sr, info = audio_processor.process_audio_file(
            sample_audio_wav, enhance=True, validate=True
        )
        
        assert isinstance(audio, np.ndarray)
        assert sr == audio_processor.target_sr
        assert isinstance(info, dict)
        
        # Check expected info keys
        expected_keys = ['original_format', 'sample_rate', 'duration', 'channels']
        for key in expected_keys:
            assert key in info
        
        assert 'quality_metrics' in info
        assert isinstance(info['quality_metrics'], dict)

    def test_process_audio_file_nonexistent(self, audio_processor):
        """Test processing non-existent file"""
        with pytest.raises(Exception):  # Should raise some exception
            audio_processor.process_audio_file("nonexistent_file.wav")

    def test_process_audio_file_no_enhancement(self, audio_processor, sample_audio_wav):
        """Test processing without enhancement"""
        audio, sr, info = audio_processor.process_audio_file(
            sample_audio_wav, enhance=False, validate=False
        )
        
        assert isinstance(audio, np.ndarray)
        assert sr == audio_processor.target_sr
        assert 'quality_metrics' not in info

    def test_vad_aggressiveness_levels(self):
        """Test different VAD aggressiveness levels"""
        for level in [0, 1, 2, 3]:
            processor = AudioProcessor(vad_aggressiveness=level)
            assert processor.vad.aggressiveness == level

    def test_target_sample_rate_configuration(self):
        """Test different target sample rates"""
        for sr in [8000, 16000, 22050, 44100]:
            processor = AudioProcessor(target_sr=sr)
            assert processor.target_sr == sr


if __name__ == "__main__":
    pytest.main([__file__])