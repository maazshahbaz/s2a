#!/usr/bin/env python3
"""
Debug script to test diarization service directly.
Run this to identify why diarization returns 1 speaker for 2-speaker audio.
"""

import asyncio
import sys
import os
import soundfile as sf
import numpy as np
from loguru import logger

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.diarization_service import DiarizationService

# Simple config without torch dependency
class SimpleDiarConfig:
    def __init__(self):
        self.model_name = "nvidia/diar_sortformer_4spk-v1"
        self.max_speakers = 4

async def test_diarization(audio_path: str):
    """Test diarization with detailed logging"""
    
    logger.info(f"Testing diarization on: {audio_path}")
    
    # Check if file exists and get basic info
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return
    
    try:
        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr
        logger.info(f"Audio info: duration={duration:.2f}s, sample_rate={sr}, channels={audio.ndim}")
    except Exception as e:
        logger.error(f"Failed to read audio file: {e}")
        return
    
    # Get diarization settings
    diar_cfg = SimpleDiarConfig()
    logger.info(f"Diarization config: model={diar_cfg.model_name}, max_speakers={diar_cfg.max_speakers}")
    
    # Initialize diarization service
    diar_service = DiarizationService(
        model_name=diar_cfg.model_name,
        max_speakers=diar_cfg.max_speakers
    )
    
    try:
        # Run diarization
        logger.info("Running diarization...")
        segments = await diar_service.run(audio_path, max_speakers=diar_cfg.max_speakers)
        
        logger.info(f"Diarization completed. Found {len(segments)} segments:")
        
        if not segments:
            logger.error("No segments returned!")
            return
        
        # Analyze segments
        speakers = set()
        for i, seg in enumerate(segments):
            logger.info(f"  Segment {i+1}: start={seg.start:.2f}, end={seg.end:.2f}, speaker={seg.speaker}")
            speakers.add(seg.speaker)
        
        unique_speakers = len(speakers)
        logger.info(f"Unique speakers detected: {unique_speakers}")
        logger.info(f"Speaker labels: {sorted(speakers)}")
        
        # Check for issues
        if unique_speakers == 1 and duration > 30:
            logger.warning("⚠️  Only 1 speaker detected for longer audio - this may indicate a problem")
            
            # Additional diagnostics
            total_speech_time = sum(seg.end - seg.start for seg in segments)
            logger.info(f"Total speech time: {total_speech_time:.2f}s ({total_speech_time/duration*100:.1f}% of audio)")
            
            # Check segment distribution
            if len(segments) > 1:
                logger.info("Multiple segments but same speaker - checking timing patterns...")
                gaps = []
                for i in range(1, len(segments)):
                    gap = segments[i].start - segments[i-1].end
                    gaps.append(gap)
                avg_gap = sum(gaps) / len(gaps) if gaps else 0
                logger.info(f"Average gap between segments: {avg_gap:.2f}s")
        
        return segments
        
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        logger.exception("Full exception:")
        return None

def create_test_audio(output_path: str, duration: int = 60, sample_rate: int = 16000):
    """Create a simple test audio with two speakers (simulated by different patterns)"""
    logger.info(f"Creating test audio: {output_path} ({duration}s)")
    
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Create two different frequency patterns to simulate different speakers
    # Speaker 1: 200-400 Hz range (lower pitch)
    # Speaker 2: 400-800 Hz range (higher pitch)
    
    audio = np.zeros_like(t)
    
    # Alternate between speakers every 10 seconds
    segment_duration = 10
    for i in range(0, duration, segment_duration):
        start_sample = int(i * sample_rate)
        end_sample = min(int((i + segment_duration) * sample_rate), len(t))
        
        if i // segment_duration % 2 == 0:
            # Speaker 1: lower frequency with some modulation
            freq1 = 300 + 50 * np.sin(2 * np.pi * 0.5 * t[start_sample:end_sample])
            audio[start_sample:end_sample] = 0.3 * np.sin(2 * np.pi * freq1 * t[start_sample:end_sample])
        else:
            # Speaker 2: higher frequency with different modulation
            freq2 = 600 + 100 * np.sin(2 * np.pi * 0.3 * t[start_sample:end_sample])
            audio[start_sample:end_sample] = 0.3 * np.sin(2 * np.pi * freq2 * t[start_sample:end_sample])
    
    # Add some noise to make it more realistic
    noise = 0.02 * np.random.normal(0, 1, len(t))
    audio = audio + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save as WAV
    sf.write(output_path, audio, sample_rate)
    logger.info(f"Test audio saved to: {output_path}")

async def main():
    """Main test function"""
    
    # Check for provided audio file or create test audio
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        # Create test audio
        audio_path = "/tmp/test_two_speakers.wav"
        create_test_audio(audio_path, duration=60)  # 1 minute test audio
    
    # Test diarization
    await test_diarization(audio_path)

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    
    asyncio.run(main())
