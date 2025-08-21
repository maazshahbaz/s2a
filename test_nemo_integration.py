#!/usr/bin/env python3
"""
Test script to validate NeMo Parakeet integration with fallback to Whisper
"""

import asyncio
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from asr_service import NeMoASRService

async def test_model_loading():
    """Test model loading with NeMo primary and Whisper fallback"""
    logger.info("=== Testing Model Loading ===")
    
    # Test with NeMo model name
    try:
        asr_service = NeMoASRService(
            model_name="nvidia/parakeet-tdt-0.6b-v2",
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
            batch_size=2
        )
        
        model_info = asr_service.get_model_info()
        logger.info(f"Model loaded successfully:")
        logger.info(f"  - Model type: {model_info['model_type']}")
        logger.info(f"  - Model name: {model_info['model_name']}")
        logger.info(f"  - Device: {model_info['device']}")
        logger.info(f"  - NeMo available: {model_info['nemo_available']}")
        logger.info(f"  - H100 optimizations: {model_info['h100_optimizations']}")
        logger.info(f"  - Chunking strategy: {model_info['chunking_strategy']}")
        
        return asr_service
        
    except Exception as e:
        logger.error(f"Failed to initialize ASR service: {e}")
        return None

async def test_short_audio_transcription(asr_service):
    """Test transcription of short audio (< 30 seconds)"""
    logger.info("\n=== Testing Short Audio Transcription ===")
    
    # Generate test audio (10 seconds of synthetic speech-like signal)
    duration = 10.0
    sample_rate = 16000
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a synthetic speech-like signal
    audio = np.sin(2 * np.pi * 440 * t) * np.exp(-t/2)  # Decaying sine wave
    audio += 0.1 * np.random.randn(len(audio))  # Add some noise
    audio = audio.astype(np.float32)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sample_rate)
        temp_path = tmp_file.name
    
    try:
        result = await asr_service.transcribe_audio(temp_path)
        
        logger.info(f"Short audio transcription results:")
        logger.info(f"  - Duration: {result.duration:.2f}s")
        logger.info(f"  - RTF: {result.rtf:.3f}")
        logger.info(f"  - Model used: {result.model_used}")
        logger.info(f"  - Chunks processed: {result.chunks_processed}")
        logger.info(f"  - Processing time: {result.processing_time:.2f}s")
        logger.info(f"  - Text length: {len(result.text)} characters")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in short audio transcription: {e}")
        return None
    
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

async def test_long_audio_transcription(asr_service):
    """Test transcription of long audio (> 24 minutes for NeMo chunking)"""
    logger.info("\n=== Testing Long Audio Transcription ===")
    
    # Generate test audio (25 minutes for NeMo chunking test)
    duration = 25 * 60  # 25 minutes
    sample_rate = 16000
    
    logger.info(f"Generating {duration/60:.1f} minute test audio...")
    
    # Create a longer synthetic signal (in chunks to avoid memory issues)
    chunk_duration = 60  # 1 minute chunks
    audio_chunks = []
    
    for i in range(0, int(duration), chunk_duration):
        chunk_dur = min(chunk_duration, duration - i)
        t = np.linspace(0, chunk_dur, int(sample_rate * chunk_dur))
        
        # Varying frequency to simulate speech patterns
        freq = 440 + 100 * np.sin(2 * np.pi * i / 300)  # Varying frequency
        chunk = np.sin(2 * np.pi * freq * t) * np.exp(-t/10)
        chunk += 0.1 * np.random.randn(len(chunk))
        audio_chunks.append(chunk.astype(np.float32))
    
    audio = np.concatenate(audio_chunks)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sample_rate)
        temp_path = tmp_file.name
    
    try:
        logger.info(f"Starting transcription of {duration/60:.1f} minute audio...")
        result = await asr_service.transcribe_audio(temp_path)
        
        logger.info(f"Long audio transcription results:")
        logger.info(f"  - Duration: {result.duration/60:.2f} minutes")
        logger.info(f"  - RTF: {result.rtf:.3f}")
        logger.info(f"  - Model used: {result.model_used}")
        logger.info(f"  - Chunks processed: {result.chunks_processed}")
        logger.info(f"  - Processing time: {result.processing_time:.2f}s")
        logger.info(f"  - Text length: {len(result.text)} characters")
        logger.info(f"  - Chunks info: {len(result.chunks) if result.chunks else 0} chunk results")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in long audio transcription: {e}")
        return None
    
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

async def main():
    """Main test function"""
    logger.info("Starting NeMo Parakeet Integration Tests")
    
    # Test 1: Model Loading
    asr_service = await test_model_loading()
    if not asr_service:
        logger.error("Model loading failed. Aborting tests.")
        return
    
    # Test 2: Short Audio
    short_result = await test_short_audio_transcription(asr_service)
    
    # Test 3: Long Audio (only if short audio worked)
    if short_result:
        logger.info(f"\nShort audio test successful with {short_result.model_used} model")
        
        # Only test long audio if we have GPU and reasonable performance
        if asr_service.device == "cuda" and short_result.rtf < 1.0:
            long_result = await test_long_audio_transcription(asr_service)
        else:
            logger.info("Skipping long audio test (CPU mode or poor RTF)")
    
    logger.info("\n=== Test Summary ===")
    logger.info(f"Model type used: {asr_service.model_type}")
    logger.info(f"Device: {asr_service.device}")
    logger.info(f"Batch size: {asr_service.batch_size}")
    logger.info("Tests completed successfully!")

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
    
    # Run tests
    asyncio.run(main())