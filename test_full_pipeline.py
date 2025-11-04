#!/usr/bin/env python3
"""
Test complete transcription + diarization pipeline
"""

import asyncio
import uuid
import shutil
from pathlib import Path
from loguru import logger
import sys

# Silence verbose logs
logger.remove()
logger.add(sys.stderr, level="INFO")

async def test_full_pipeline():
    """Test the complete pipeline: transcription + diarization"""
    
    from generated.prisma import Prisma
    from db_services.transcription import TranscriptionJobService
    from dependencies import process_audio_background_db
    
    print("="*70)
    print("COMPLETE PIPELINE TEST: Transcription + Diarization")
    print("="*70)
    
    # Initialize database
    db = Prisma()
    await db.connect()
    
    job_service = TranscriptionJobService(db)
    
    # Create a test job
    job_id = str(uuid.uuid4())
    audio_path = "/tmp/your_audio.wav"
    
    print(f"\nCreating transcription job...")
    print(f"Job ID: {job_id}")
    print(f"Audio: badcb4b0-f69f-4384-863e-09b09ce0582f.wav (2 speakers)")
    
    job = await job_service.create_job(
        job_id=job_id,
        audio_path=audio_path,
        is_async=False,  # Synchronous for testing
        enhance_audio=True,
        remove_silence=False,
        priority=0,
        callback_url=None,
        audio_duration=928.0
    )
    
    print(f"\n✓ Job created with status: {job.status}")
    
    # Process the audio (this runs transcription + diarization)
    print(f"\nProcessing audio through complete pipeline...")
    print("  - ASR transcription")
    print("  - Speaker diarization")
    print("  - Alignment (matching transcription to speakers)")
    print()
    
    try:
        # Import the background processing function
        from services.nemo_asr_service import NeMoASRService
        from services.diarization_service import DiarizationService
        from services.alignment_service import align_sentence_segments, render_speaker_attributed_text
        from config import get_asr_settings, get_diarization_settings
        import soundfile as sf
        import json
        
        # Initialize services
        asr_cfg = get_asr_settings()
        asr_service = NeMoASRService(
            model_name=asr_cfg.model_name,
            device=asr_cfg.device,
            batch_size=asr_cfg.batch_size
        )
        
        diar_cfg = get_diarization_settings()
        diar_service = DiarizationService(
            model_name=diar_cfg.model_name,
            max_speakers=diar_cfg.max_speakers
        )
        
        # Update job status
        await job_service.update_job_status(job_id, 'processing')
        
        # 1. Run ASR transcription
        print("Step 1: Running ASR transcription...")
        audio, sr = sf.read(audio_path)
        duration = len(audio) / sr
        
        transcription_result = asr_service.transcribe_nemo(audio_path)
        transcription_text = transcription_result.get('text', '')
        confidence = transcription_result.get('confidence', 0.0)
        
        print(f"  ✓ Transcription complete")
        print(f"    Text length: {len(transcription_text)} chars")
        print(f"    Confidence: {confidence:.2f}")
        print(f"    Preview: {transcription_text[:100]}...")
        
        # 2. Run diarization
        print(f"\nStep 2: Running speaker diarization...")
        diar_segments = await diar_service.run(audio_path, max_speakers=diar_cfg.max_speakers)
        
        speakers = set(seg.speaker for seg in diar_segments)
        num_speakers = len(speakers)
        
        print(f"  ✓ Diarization complete")
        print(f"    Speakers detected: {num_speakers}")
        print(f"    Total segments: {len(diar_segments)}")
        print(f"    Speaker labels: {sorted(speakers)}")
        
        # 3. Align transcription with diarization
        print(f"\nStep 3: Aligning transcription with speakers...")
        
        # Create ASR segments (simplified - using full text as one segment)
        asr_segments = [{
            'start_time': 0.0,
            'end_time': duration,
            'text': transcription_text
        }]
        
        # Convert diar segments to dict format
        diar_segments_dict = [
            {'start': seg.start, 'end': seg.end, 'speaker': seg.speaker}
            for seg in diar_segments
        ]
        
        speaker_blocks, aligned_num_speakers = align_sentence_segments(
            asr_segments, diar_segments_dict
        )
        
        speaker_attributed_text = render_speaker_attributed_text(speaker_blocks)
        
        print(f"  ✓ Alignment complete")
        print(f"    Speaker blocks: {len(speaker_blocks)}")
        print(f"    Aligned speakers: {aligned_num_speakers}")
        
        # 4. Save results
        print(f"\nStep 4: Saving results to database...")
        
        audio_quality = {
            'speakerTranscript': speaker_blocks,
            'numSpeakers': num_speakers,
            'diarModel': diar_cfg.model_name
        }
        
        await job_service.save_transcription_result(
            job_id=job_id,
            text=speaker_attributed_text,
            confidence=confidence,
            rtf=0.0,
            processing_time=0.0,
            chunks=1,
            audio_quality=audio_quality
        )
        
        print(f"  ✓ Results saved")
        
        # 5. Retrieve and display final result
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS (as returned by Status API)")
        print(f"{'='*70}")
        
        final_job = await db.transcriptionjob.find_unique(
            where={'jobId': job_id},
            include={'transcriptionResult': True}
        )
        
        if final_job and final_job.transcriptionResult:
            result = final_job.transcriptionResult
            quality = result.audioQuality
            if isinstance(quality, str):
                quality = json.loads(quality)
            
            print(f"\nJob Status: {final_job.status}")
            print(f"Transcription length: {len(result.text)} chars")
            print(f"\nDiarization Info:")
            print(f"  Number of speakers: {quality.get('numSpeakers')}")
            print(f"  Diarization model: {quality.get('diarModel')}")
            print(f"  Speaker blocks: {len(quality.get('speakerTranscript', []))}")
            
            # Show speaker breakdown
            from collections import Counter, defaultdict
            speaker_transcript = quality.get('speakerTranscript', [])
            speaker_counts = Counter(block['speaker'] for block in speaker_transcript)
            speaker_times = defaultdict(float)
            
            for block in speaker_transcript:
                speaker_times[block['speaker']] += (block['end'] - block['start'])
            
            print(f"\n  Speaker breakdown:")
            for speaker in sorted(speaker_counts.keys()):
                print(f"    {speaker}: {speaker_counts[speaker]} blocks, {speaker_times[speaker]:.1f}s")
            
            # Show sample speaker-attributed transcript
            print(f"\n  Sample speaker-attributed transcript:")
            print(f"  {'-'*66}")
            for block in speaker_transcript[:3]:
                text_preview = block['text'][:80] + "..." if len(block['text']) > 80 else block['text']
                print(f"  {block['speaker']} [{block['start']:.1f}s]: {text_preview}")
            
            print(f"\n{'='*70}")
            if quality.get('numSpeakers', 0) >= 2:
                print(f"✅ PIPELINE TEST PASSED: Multiple speakers detected and aligned!")
            else:
                print(f"⚠️  Only {quality.get('numSpeakers', 0)} speaker(s) detected")
            print(f"{'='*70}")
        
    except Exception as e:
        print(f"\n❌ Error during pipeline processing: {e}")
        import traceback
        traceback.print_exc()
        await job_service.update_job_status(job_id, 'failed', error_message=str(e))
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())
