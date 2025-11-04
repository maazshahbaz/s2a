#!/usr/bin/env python3
"""
End-to-end test of diarization through the full S2A pipeline
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from services.diarization_service import DiarizationService
from config import get_diarization_settings
from loguru import logger

async def test_diarization_e2e():
    """Test diarization with the actual audio file"""
    
    audio_path = "/tmp/test_audio.wav"
    
    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        logger.info("Please ensure the audio file is copied to the container")
        return False
    
    logger.info(f"Testing diarization on: {audio_path}")
    
    # Get config
    diar_cfg = get_diarization_settings()
    logger.info(f"Config: model={diar_cfg.model_name}, max_speakers={diar_cfg.max_speakers}")
    
    # Initialize service
    diar_service = DiarizationService(
        model_name=diar_cfg.model_name,
        max_speakers=diar_cfg.max_speakers
    )
    
    try:
        # Run diarization
        logger.info("Running diarization...")
        segments = await diar_service.run(audio_path)
        
        # Analyze results
        speakers = set(seg.speaker for seg in segments)
        num_speakers = len(speakers)
        
        logger.info(f"✅ Diarization completed successfully!")
        logger.info(f"   Total segments: {len(segments)}")
        logger.info(f"   Unique speakers: {num_speakers}")
        logger.info(f"   Speaker labels: {sorted(speakers)}")
        
        # Show sample segments for each speaker
        for speaker in sorted(speakers):
            speaker_segs = [s for s in segments if s.speaker == speaker]
            total_time = sum(s.end - s.start for s in speaker_segs)
            logger.info(f"   {speaker}: {len(speaker_segs)} segments, {total_time:.1f}s total")
            
            # Show first 3 segments
            for i, seg in enumerate(speaker_segs[:3]):
                logger.info(f"      [{i+1}] {seg.start:.2f}s - {seg.end:.2f}s")
        
        # Verify we got multiple speakers
        if num_speakers >= 2:
            logger.success(f"✅ TEST PASSED: Detected {num_speakers} speakers")
            return True
        else:
            logger.error(f"❌ TEST FAILED: Only detected {num_speakers} speaker(s)")
            return False
            
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}")
        logger.exception("Full exception:")
        return False

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    result = asyncio.run(test_diarization_e2e())
    sys.exit(0 if result else 1)
