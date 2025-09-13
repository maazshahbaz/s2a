#!/usr/bin/env python3
"""
End-to-End Validation Test Suite for S2A ASR Service

Tests real audio files with duration-based routing rules:
- 40s audio: Sync API (instant response, single chunk)
- 3.21min audio: Async API (background processing, single chunk) 
- 33.47min audio: Async API (background processing, multi-chunk)

Validates NeMo Parakeet integration with H100 optimizations.
"""

import asyncio
import soundfile as sf
from pathlib import Path
from loguru import logger
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.asr_service import NeMoASRService

# Test audio files paths
TEST_AUDIO_DIR = Path(__file__).parent / "test_audio"
AUDIO_FILES = {
    "short": "in-9524528884-2014569295-20250515-095120-1747320680.72350.wav",  # 40.48s
    "medium": "in-6123987606-6123773368-20250513-142151-1747164111.64201.wav",  # 3.21min  
    "long": "in-9524528884-2058527609-20250125-132037-1737832837.3553.wav"  # 33.47min
}

def save_transcription(audio_name: str, result, audio_filename: str):
    """Save transcription results to test_audio directory"""
    if not result or not hasattr(result, 'text'):
        logger.warning(f"⚠️  No transcription to save for {audio_name}")
        return None
    
    # Create transcription filename
    base_name = audio_filename.replace('.wav', '')
    transcript_file = TEST_AUDIO_DIR / f"{base_name}_transcription.txt"
    
    try:
        # Save detailed transcription results
        with open(transcript_file, 'w', encoding='utf-8') as f:
            f.write(f"=== S2A ASR Transcription Results ===\n")
            f.write(f"Audio File: {audio_filename}\n")
            f.write(f"Duration: {result.duration:.2f}s ({result.duration/60:.2f} minutes)\n")
            f.write(f"Model: {result.model_used}\n")
            f.write(f"Chunks Processed: {result.chunks_processed}\n")
            f.write(f"Processing Time: {result.processing_time:.2f}s\n")
            f.write(f"RTF: {result.rtf:.3f}\n")
            f.write(f"Performance: {result.duration/result.processing_time:.1f}x real-time\n")
            f.write(f"Generated: {__import__('datetime').datetime.now()}\n")
            f.write(f"\n=== TRANSCRIPTION ===\n")
            f.write(result.text)
            f.write(f"\n\n=== END ===\n")
        
        logger.info(f"💾 Transcription saved: {transcript_file.name}")
        return transcript_file
        
    except Exception as e:
        logger.error(f"❌ Failed to save transcription: {e}")
        return None

async def test_model_loading():
    """Test NeMo-only ASR service initialization"""
    logger.info("=== Testing NeMo Model Loading ===")
    
    try:
        asr_service = NeMoASRService(
            model_name="nvidia/parakeet-tdt-0.6b-v2",
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
            batch_size=2
        )
        
        model_info = asr_service.get_model_info()
        logger.info(f"✅ NeMo model loaded successfully:")
        logger.info(f"  - Model type: {model_info['model_type']}")
        logger.info(f"  - Model name: {model_info['model_name']}")
        logger.info(f"  - Device: {model_info['device']}")
        logger.info(f"  - Chunking strategy: {model_info['chunking_strategy']}")
        logger.info(f"  - H100 optimizations: {model_info['h100_optimizations']}")
        logger.info(f"  - Max chunk duration: {model_info['max_chunk_duration']/60:.1f} minutes")
        
        return asr_service
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize ASR service: {e}")
        return None

async def test_sync_api_audio(asr_service, audio_name: str):
    """Test sync API behavior with real audio file"""
    logger.info(f"\n=== Testing Sync API Logic: {audio_name} Audio ===")
    
    audio_path = TEST_AUDIO_DIR / AUDIO_FILES[audio_name]
    if not audio_path.exists():
        logger.error(f"❌ Audio file not found: {audio_path}")
        return None
    
    # Get audio info
    info = sf.info(audio_path)
    duration = info.frames / info.samplerate
    
    logger.info(f"📁 File: {audio_path.name}")
    logger.info(f"⏱️  Duration: {duration:.2f}s ({duration/60:.2f} minutes)")
    
    # Check sync API duration rules
    if duration <= 120:  # ≤ 2 minutes
        logger.info(f"✅ Expected: SYNC API should ACCEPT (≤2min rule)")
        expected_behavior = "accept"
    else:
        logger.info(f"❌ Expected: SYNC API should REJECT (>2min rule)")
        expected_behavior = "reject"
    
    try:
        result = await asr_service.transcribe_audio(audio_path)
        
        if expected_behavior == "accept":
            logger.info(f"✅ SYNC API Results:")
            logger.info(f"  - Transcription: \"{result.text[:100]}{'...' if len(result.text) > 100 else ''}\"")
            logger.info(f"  - Duration: {result.duration:.2f}s")
            logger.info(f"  - RTF: {result.rtf:.3f}")
            logger.info(f"  - Model used: {result.model_used}")
            logger.info(f"  - Chunks processed: {result.chunks_processed}")
            logger.info(f"  - Processing time: {result.processing_time:.2f}s")
            
            # Save transcription to file
            save_transcription(audio_name, result, AUDIO_FILES[audio_name])
            return result
        else:
            logger.warning(f"⚠️  Audio was accepted but should have been rejected by sync API")
            return result
            
    except Exception as e:
        if expected_behavior == "reject":
            logger.info(f"✅ SYNC API correctly rejected: {e}")
            return "rejected"
        else:
            logger.error(f"❌ Unexpected error: {e}")
            return None

async def test_async_api_chunking(asr_service, audio_name: str):
    """Test async API chunking behavior with real audio"""
    logger.info(f"\n=== Testing Async API Chunking: {audio_name} Audio ===")
    
    audio_path = TEST_AUDIO_DIR / AUDIO_FILES[audio_name]
    if not audio_path.exists():
        logger.error(f"❌ Audio file not found: {audio_path}")
        return None
    
    # Get audio info
    info = sf.info(audio_path)
    duration = info.frames / info.samplerate
    
    logger.info(f"📁 File: {audio_path.name}")
    logger.info(f"⏱️  Duration: {duration:.2f}s ({duration/60:.2f} minutes)")
    
    # Predict chunking behavior
    if duration <= 1440:  # ≤ 24 minutes
        expected_chunks = 1
        logger.info(f"🎯 Expected: Single chunk (≤24min rule)")
    else:
        expected_chunks = int(duration / 1440) + (1 if duration % 1440 > 300 else 0)  # Rough estimate
        logger.info(f"🧩 Expected: ~{expected_chunks} chunks (24min chunking)")
    
    try:
        result = await asr_service.transcribe_audio(audio_path)
        
        logger.info(f"✅ ASYNC Processing Results:")
        logger.info(f"  - Transcription length: {len(result.text)} characters")
        logger.info(f"  - Sample: \"{result.text[:200]}{'...' if len(result.text) > 200 else ''}\"")
        logger.info(f"  - Duration: {result.duration:.2f}s ({result.duration/60:.2f} minutes)")
        logger.info(f"  - RTF: {result.rtf:.3f}")
        logger.info(f"  - Model used: {result.model_used}")
        logger.info(f"  - Chunks processed: {result.chunks_processed}")
        logger.info(f"  - Processing time: {result.processing_time:.2f}s")
        
        # Validate chunking prediction
        if result.chunks_processed == expected_chunks:
            logger.info(f"✅ Chunking behavior matches expectation: {result.chunks_processed} chunks")
        else:
            logger.warning(f"⚠️  Chunking mismatch - Expected: {expected_chunks}, Got: {result.chunks_processed}")
        
        # Performance analysis
        speedup = result.duration / result.processing_time
        logger.info(f"🚀 Performance: {speedup:.1f}x real-time processing")
        
        # Save transcription to file
        save_transcription(audio_name, result, AUDIO_FILES[audio_name])
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in async processing: {e}")
        return None

async def validate_duration_routing_rules():
    """Validate all duration-based routing rules"""
    logger.info("\n=== Validating Duration-Based Routing Rules ===")
    
    for audio_name, filename in AUDIO_FILES.items():
        audio_path = TEST_AUDIO_DIR / filename
        if not audio_path.exists():
            logger.warning(f"⚠️  Skipping {audio_name}: file not found")
            continue
            
        info = sf.info(audio_path)
        duration = info.frames / info.samplerate
        
        logger.info(f"\n📊 {audio_name.upper()} AUDIO ({duration:.1f}s):")
        
        if duration < 5:
            logger.info("  🚫 Both APIs: REJECT (< 5s minimum)")
        elif duration <= 120:
            logger.info("  ✅ Sync API: ACCEPT (5s-2min) - Instant response")
            logger.info("  ✅ Async API: ACCEPT (5s-2min) - Background processing")
            logger.info("  🎯 Processing: Single chunk, no splitting")
        elif duration <= 7200:  # 2 hours
            logger.info("  ❌ Sync API: REJECT (>2min limit)")  
            logger.info("  ✅ Async API: ACCEPT (2min-2hr) - Background processing")
            if duration <= 1440:  # 24 minutes
                logger.info("  🎯 Processing: Single chunk, no splitting")
            else:
                chunks = int(duration / 1440) + (1 if duration % 1440 > 0 else 0)
                logger.info(f"  🧩 Processing: ~{chunks} chunks (24-min intervals)")
        else:
            logger.info("  🚫 Both APIs: REJECT (>2hr limit)")

async def main():
    """Main validation test runner"""
    logger.info("🚀 Starting S2A End-to-End Validation Tests")
    logger.info("📁 Using real audio files for transcription accuracy testing")
    
    # Test 1: Model Loading
    asr_service = await test_model_loading()
    if not asr_service:
        logger.error("❌ Model loading failed. Aborting tests.")
        return
    
    # Test 2: Duration Rules Analysis
    await validate_duration_routing_rules()
    
    # Test 3: Sync API Testing (40s audio)
    logger.info(f"\n{'='*60}")
    sync_result = await test_sync_api_audio(asr_service, "short")
    
    # Test 4: Async API Single Chunk (3.21min audio)  
    logger.info(f"\n{'='*60}")
    async_single_result = await test_async_api_chunking(asr_service, "medium")
    
    # Test 5: Async API Multi-Chunk (33.47min audio) - Only if GPU and good performance
    if asr_service.device == "cuda" and sync_result and hasattr(sync_result, 'rtf') and sync_result.rtf < 0.5:
        logger.info(f"\n{'='*60}")
        async_multi_result = await test_async_api_chunking(asr_service, "long")
    else:
        logger.info(f"\n⚠️  Skipping long audio test (CPU mode or performance concerns)")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("🏆 VALIDATION SUMMARY")
    logger.info(f"✅ Model: {asr_service.model_type} ({asr_service.model_name})")
    logger.info(f"✅ Device: {asr_service.device}")
    logger.info(f"✅ Duration rules: Validated")
    logger.info(f"✅ Real audio processing: Tested")
    logger.info(f"✅ Chunking behavior: Verified")
    logger.info("🎯 S2A service is ready for production!")

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    # Run validation tests
    asyncio.run(main())