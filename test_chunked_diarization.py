#!/usr/bin/env python3
"""
Test chunked diarization implementation with 24-minute chunks
"""

import asyncio
import sys
from pathlib import Path
from loguru import logger

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from services.diarization_service import DiarizationService

async def test_chunked_diarization():
    """Test chunked diarization on your 2-speaker audio"""
    
    audio_path = "/tmp/your_audio.wav"
    
    if not Path(audio_path).exists():
        logger.error(f"Audio file not found: {audio_path}")
        logger.info("Please ensure the audio file is available in the container")
        return False
    
    logger.info("="*70)
    logger.info("TESTING CHUNKED DIARIZATION (24-min chunks, 30s overlap)")
    logger.info("="*70)
    
    # Initialize service
    diar_service = DiarizationService(
        model_name='nvidia/diar_sortformer_4spk-v1',
        max_speakers=4
    )
    
    # Show configuration
    logger.info(f"\nConfiguration:")
    logger.info(f"  Chunk duration: {diar_service.chunk_duration}s (24 minutes)")
    logger.info(f"  Overlap duration: {diar_service.overlap_duration}s (30 seconds)")
    logger.info(f"  Similarity threshold: {diar_service.similarity_threshold}")
    
    try:
        # Run chunked diarization
        logger.info(f"\nRunning chunked diarization on: {audio_path}")
        segments = await diar_service.run(audio_path)
        
        # Analyze results
        speakers = {}
        for seg in segments:
            if seg.speaker not in speakers:
                speakers[seg.speaker] = {'count': 0, 'total_time': 0.0}
            speakers[seg.speaker]['count'] += 1
            speakers[seg.speaker]['total_time'] += (seg.end - seg.start)
        
        num_speakers = len(speakers)
        
        logger.info("\n" + "="*70)
        logger.info("CHUNKED DIARIZATION RESULTS")
        logger.info("="*70)
        logger.info(f"Total segments: {len(segments)}")
        logger.info(f"Unique speakers: {num_speakers}")
        logger.info(f"Speaker labels: {sorted(speakers.keys())}")
        logger.info("")
        
        for speaker in sorted(speakers.keys()):
            data = speakers[speaker]
            percentage = (data['total_time'] / sum(s['total_time'] for s in speakers.values())) * 100
            logger.info(f"{speaker}:")
            logger.info(f"  Segments: {data['count']}")
            logger.info(f"  Total time: {data['total_time']:.1f}s ({percentage:.1f}%)")
        
        # Show first 10 segments
        logger.info(f"\nFirst 10 segments:")
        logger.info("-"*70)
        for i, seg in enumerate(segments[:10], 1):
            logger.info(f"{i:2d}. [{seg.start:7.2f}s - {seg.end:7.2f}s] {seg.speaker}")
        
        if len(segments) > 10:
            logger.info(f"... and {len(segments)-10} more segments")
        
        # Verify success
        logger.info("\n" + "="*70)
        if num_speakers >= 2:
            logger.success(f"✅ TEST PASSED: Detected {num_speakers} speakers with chunked approach!")
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
    
    result = asyncio.run(test_chunked_diarization())
    sys.exit(0 if result else 1)
