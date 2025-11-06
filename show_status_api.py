#!/usr/bin/env python3
"""Display transcription status in API format"""

import asyncio
import json
from generated.prisma import Prisma

async def show_status():
    db = Prisma()
    await db.connect()
    
    # Get a completed job
    job = await db.transcriptionjob.find_first(
        where={'status': 'completed'},
        include={'transcriptionResult': True}
    )
    
    if not job:
        print("No completed jobs found")
        return
    
    # Format as API response
    response = {
        "job_id": job.jobId,
        "status": job.status,
        "audio_path": job.audioPath,
        "is_async": job.isAsync,
        "created_at": job.createdAt.isoformat() if job.createdAt else None,
        "started_at": job.startedAt.isoformat() if job.startedAt else None,
        "completed_at": job.completedAt.isoformat() if job.completedAt else None,
        "error_message": job.errorMessage
    }
    
    if job.transcriptionResult:
        result = job.transcriptionResult
        
        # Parse audioQuality JSON
        audio_quality = result.audioQuality
        if isinstance(audio_quality, str):
            audio_quality = json.loads(audio_quality)
        
        response["result"] = {
            "text": result.text[:200] + "..." if len(result.text) > 200 else result.text,
            "full_text_length": len(result.text),
            "confidence": result.confidence,
            "rtf": result.rtf,
            "processing_time": result.processingTime,
            "chunks": result.chunks,
            "diarization": {
                "num_speakers": audio_quality.get('numSpeakers') if audio_quality else None,
                "diar_model": audio_quality.get('diarModel') if audio_quality else None,
                "speaker_transcript_blocks": len(audio_quality.get('speakerTranscript', [])) if audio_quality else 0
            }
        }
        
        # Add speaker transcript preview
        if audio_quality and 'speakerTranscript' in audio_quality:
            speaker_transcript = audio_quality['speakerTranscript']
            response["result"]["speaker_transcript_preview"] = speaker_transcript[:3]  # First 3 blocks
    
    print("="*80)
    print("STATUS API RESPONSE FORMAT")
    print("="*80)
    print(json.dumps(response, indent=2, default=str))
    print()
    
    # Show detailed diarization if available
    if job.transcriptionResult and audio_quality and 'speakerTranscript' in audio_quality:
        print("="*80)
        print("DETAILED DIARIZATION INFO")
        print("="*80)
        
        speaker_transcript = audio_quality['speakerTranscript']
        
        # Count by speaker
        from collections import Counter, defaultdict
        speaker_counts = Counter(block['speaker'] for block in speaker_transcript)
        speaker_times = defaultdict(float)
        
        for block in speaker_transcript:
            speaker = block['speaker']
            duration = block['end'] - block['start']
            speaker_times[speaker] += duration
        
        print(f"\nTotal Speakers: {audio_quality.get('numSpeakers', 'N/A')}")
        print(f"Total Blocks: {len(speaker_transcript)}")
        print(f"\nBreakdown by speaker:")
        for speaker in sorted(speaker_counts.keys()):
            print(f"  {speaker}:")
            print(f"    Segments: {speaker_counts[speaker]}")
            print(f"    Total time: {speaker_times[speaker]:.1f}s")
        
        print(f"\nFirst 5 speaker blocks:")
        print("-"*80)
        for i, block in enumerate(speaker_transcript[:5], 1):
            text_preview = block['text'][:100] + "..." if len(block['text']) > 100 else block['text']
            print(f"{i}. {block['speaker']} [{block['start']:.2f}s - {block['end']:.2f}s]")
            print(f"   {text_preview}")
            print()
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(show_status())
