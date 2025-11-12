#!/usr/bin/env python3
"""Submit a test transcription job and display results"""

import asyncio
import uuid
import shutil
from pathlib import Path
from generated.prisma import Prisma
from services.batch_processor import BatchProcessor, BatchProcessorConfig
from services.asr_service import ASRService
from config import get_asr_settings, get_redis_settings

async def submit_and_monitor():
    # Initialize services
    print("Initializing services...")
    db = Prisma()
    await db.connect()
    
    asr_cfg = get_asr_settings()
    asr_service = ASRService(
        model_name=asr_cfg.model_name,
        device=asr_cfg.device,
        batch_size=asr_cfg.batch_size
    )
    
    redis_cfg = get_redis_settings()
    batch_cfg = BatchProcessorConfig(
        redis_host=redis_cfg.host,
        redis_port=redis_cfg.port,
        redis_db=redis_cfg.db,
        batch_size=redis_cfg.batch_size,
        num_workers=redis_cfg.num_workers
    )
    
    batch_processor = BatchProcessor(asr_service, db, batch_cfg)
    await batch_processor.start()
    
    # Copy test audio to uploads
    job_id = str(uuid.uuid4())
    source = Path("/tmp/test_audio.wav")
    dest_dir = Path("uploads/2025-10-30")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{job_id}.wav"
    
    print(f"\nCopying audio file...")
    print(f"Source: {source}")
    print(f"Dest: {dest}")
    shutil.copy(source, dest)
    
    # Create job in database
    from db_services.transcription import TranscriptionJobService
    job_service = TranscriptionJobService(db)
    
    print(f"\nCreating job: {job_id}")
    job = await job_service.create_job(
        job_id=job_id,
        audio_path=str(dest),
        is_async=True,
        enhance_audio=True,
        remove_silence=False,
        priority=0,
        callback_url=None,
        audio_duration=928.0
    )
    
    # Submit to batch processor
    print(f"\nSubmitting to batch processor...")
    result = await batch_processor.submit_job(
        job_id=job_id,
        audio_path=str(dest),
        callback_url=None
    )
    
    print(f"\nJob submitted:")
    print(f"  Status: {result['status']}")
    print(f"  Chunks: {result['num_chunks']}")
    print(f"  Estimated time: {result['estimated_processing_time']:.1f}s")
    
    # Wait for completion
    print(f"\nWaiting for job to complete...")
    for i in range(60):  # Wait up to 60 seconds
        await asyncio.sleep(1)
        
        job_status = await batch_processor.get_job_status(job_id)
        if job_status['status'] == 'completed':
            print(f"\n✅ Job completed in {i+1} seconds!")
            break
        elif i % 5 == 0:
            print(f"  Status: {job_status['status']}, chunks: {job_status.get('completed_chunks', 0)}/{job_status.get('total_chunks', 0)}")
    
    # Get final result from database
    print(f"\n" + "="*70)
    print(f"FINAL RESULT FROM STATUS API")
    print("="*70)
    
    final_job = await db.transcriptionjob.find_unique(
        where={'jobId': job_id},
        include={'transcriptionResult': True}
    )
    
    if final_job and final_job.transcriptionResult:
        result = final_job.transcriptionResult
        
        print(f"\nJob ID: {job_id}")
        print(f"Status: {final_job.status}")
        print(f"Processing Time: {result.processingTime:.2f}s")
        print(f"RTF: {result.rtf:.4f}")
        print(f"Confidence: {result.confidence:.2f}")
        
        # Display diarization
        if result.audioQuality:
            import json
            quality = result.audioQuality
            if isinstance(quality, str):
                quality = json.loads(quality)
            
            print(f"\n{'='*70}")
            print(f"DIARIZATION RESULTS")
            print(f"{'='*70}")
            print(f"Number of Speakers: {quality.get('numSpeakers', 'N/A')}")
            print(f"Diarization Model: {quality.get('diarModel', 'N/A')}")
            
            speaker_transcript = quality.get('speakerTranscript', [])
            if speaker_transcript:
                print(f"\nTotal Speaker Blocks: {len(speaker_transcript)}")
                
                # Count segments per speaker
                from collections import Counter
                speaker_counts = Counter(block['speaker'] for block in speaker_transcript)
                print(f"\nSegments per speaker:")
                for speaker, count in sorted(speaker_counts.items()):
                    total_time = sum(
                        block['end'] - block['start'] 
                        for block in speaker_transcript 
                        if block['speaker'] == speaker
                    )
                    print(f"  {speaker}: {count} segments, {total_time:.1f}s total")
                
                print(f"\nFirst 3 speaker blocks:")
                print("-"*70)
                for block in speaker_transcript[:3]:
                    print(f"{block['speaker']} [{block['start']:.2f}s - {block['end']:.2f}s]:")
                    print(f"  {block['text'][:150]}...")
                    print()
    
    # Cleanup
    await batch_processor.stop()
    await db.disconnect()
    print(f"\n✅ Test complete!")

if __name__ == "__main__":
    asyncio.run(submit_and_monitor())
