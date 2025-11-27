from typing import Optional, Dict, Any
from datetime import datetime
from generated.prisma import Prisma, Json
from generated.prisma.models import TranscriptionJob, TranscriptionResult
import os
import math
from pathlib import Path
from loguru import logger


def sanitize_json_data(data: Any) -> Any:
    """Sanitize data to be JSON-compatible by handling NaN, inf, and numpy types"""
    if data is None:
        return None
    elif isinstance(data, dict):
        return {key: sanitize_json_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_json_data(item) for item in data]
    elif isinstance(data, (float, complex)):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)
    elif hasattr(data, 'item'):  # numpy types
        return sanitize_json_data(data.item())
    elif hasattr(data, 'tolist'):  # numpy arrays
        return sanitize_json_data(data.tolist())
    else:
        return data


class TranscriptionJobService:
    def __init__(self, db: Prisma):
        self.db = db

    async def create_job(
        self,
        job_id: str,
        audio_path: str,
        is_async: bool,
        enhance_audio: bool = True,
        remove_silence: bool = False,
        priority: int = 0,
        callback_url: Optional[str] = None,
        audio_duration: Optional[float] = None
    ) -> TranscriptionJob:
        """Create a new transcription job"""
        job = await self.db.transcriptionjob.create(
            data={
                'jobId': job_id,
                'audioPath': audio_path,
                'isAsync': is_async,
                'enhanceAudio': enhance_audio,
                'removeSilence': remove_silence,
                'priority': priority,
                'callbackUrl': callback_url,
                'audioDuration': audio_duration,
                'status': 'pending'
            }
        )
        logger.info(f"Created transcription job {job_id} with path {audio_path}")
        return job

    async def get_job(self, job_id: str) -> Optional[TranscriptionJob]:
        """Get a job by job_id"""
        return await self.db.transcriptionjob.find_unique(
            where={'jobId': job_id},
            include={'transcriptionResult': True}
        )

    async def get_callback_url(self, job_id: str) -> Optional[str]:
        """Get callback URL for a job"""
        job = await self.db.transcriptionjob.find_unique(
            where={'jobId': job_id}
        )
        return job.callbackUrl if job else None

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        error_message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> Optional[TranscriptionJob]:
        """Update job status"""
        update_data = {'status': status}
        
        if error_message:
            update_data['errorMessage'] = error_message
        if started_at:
            update_data['startedAt'] = started_at
        if completed_at:
            update_data['completedAt'] = completed_at
            
        job = await self.db.transcriptionjob.update(
            where={'jobId': job_id},
            data=update_data
        )
        logger.info(f"Updated job {job_id} status to {status}")
        return job

    async def save_transcription_result(
        self,
        job_id: str,
        text: str,
        confidence: Optional[float] = None,
        rtf: Optional[float] = None,
        processing_time: Optional[float] = None,
        chunks: Optional[int] = None,
        audio_quality: Optional[Dict[str, Any]] = None,
        diarization: Optional[Dict[str, Any]] = None,
        intelligence: Optional[Dict[str, Any]] = None,
    ) -> TranscriptionResult:
        """Save transcription result"""
        # Build data dict, excluding None values for optional fields
        data = {
            'jobId': job_id,
            'text': text,
        }
        
        # Only include optional fields if they have values
        if confidence is not None:
            data['confidence'] = confidence
        if rtf is not None:
            data['rtf'] = rtf
        if processing_time is not None:
            data['processingTime'] = processing_time
        if chunks is not None:
            data['chunks'] = chunks
        if diarization is not None:
            try:
                # Use Prisma Json wrapper
                sanitized_diarization = sanitize_json_data(diarization)
                data['diarization'] = Json(sanitized_diarization)
                logger.info(f"Prepared diarization data with Json wrapper")
            except Exception as e:
                logger.warning(f"Failed to process diarization data: {e}, skipping")
        if intelligence is not None:
            try:
                # Use Prisma Json wrapper
                sanitized_intelligence = sanitize_json_data(intelligence)
                data['intelligence'] = Json(sanitized_intelligence)
                logger.info(f"Prepared intelligence data with Json wrapper")
            except Exception as e:
                logger.warning(f"Failed to process intelligence data: {e}, skipping")
        if audio_quality is not None:
            try:
                # Use Prisma Json wrapper
                sanitized_quality = sanitize_json_data(audio_quality)
                data['audioQuality'] = Json(sanitized_quality)
                logger.info(f"Prepared audio quality data with Json wrapper")
            except Exception as e:
                logger.warning(f"Failed to process audio quality data: {e}, skipping")
            
        result = await self.db.transcriptionresult.create(data=data)
        
        # Update job status to completed
        await self.update_job_status(job_id, 'completed', completed_at=datetime.utcnow())
        
        logger.info(f"Saved transcription result for job {job_id}")
        return result

    async def get_pending_async_jobs(self, limit: int = 10) -> list[TranscriptionJob]:
        """Get pending async jobs ordered by priority and creation time"""
        return await self.db.transcriptionjob.find_many(
            where={
                'status': 'pending',
                'isAsync': True
            },
            order_by=[
                {'priority': 'desc'},
                {'createdAt': 'asc'}
            ],
            take=limit
        )

    async def cleanup_old_audio_files(self, days_old: int = 7):
        """Clean up old audio files that are completed or failed"""
        from datetime import timedelta
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        old_jobs = await self.db.transcriptionjob.find_many(
            where={
                'status': {'in': ['completed', 'failed']},
                'completedAt': {'lt': cutoff_date}
            }
        )
        
        cleaned_count = 0
        for job in old_jobs:
            try:
                if os.path.exists(job.audioPath):
                    os.unlink(job.audioPath)
                    cleaned_count += 1
                    logger.info(f"Cleaned up audio file: {job.audioPath}")
            except Exception as e:
                logger.error(f"Failed to clean up {job.audioPath}: {e}")
        
        return cleaned_count

    async def get_job_statistics(self) -> Dict[str, Any]:
        """Get job statistics"""
        total_jobs = await self.db.transcriptionjob.count()
        pending_jobs = await self.db.transcriptionjob.count(where={'status': 'pending'})
        processing_jobs = await self.db.transcriptionjob.count(where={'status': 'processing'})
        completed_jobs = await self.db.transcriptionjob.count(where={'status': 'completed'})
        failed_jobs = await self.db.transcriptionjob.count(where={'status': 'failed'})
        
        return {
            'total': total_jobs,
            'pending': pending_jobs,
            'processing': processing_jobs,
            'completed': completed_jobs,
            'failed': failed_jobs
        }


# Utility functions for file storage
def get_audio_storage_path(job_id: str, filename: str) -> str:
    """Generate storage path for audio file"""
    
    # Create uploads directory if it doesn't exist
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    # Create subdirectory based on date to avoid too many files in one directory
    date_dir = uploads_dir / datetime.now().strftime("%Y-%m-%d")
    date_dir.mkdir(exist_ok=True)
    
    # Use job_id as part of filename to ensure uniqueness
    file_extension = Path(filename).suffix
    storage_filename = f"{job_id}{file_extension}"
    
    return str(date_dir / storage_filename)

async def delete_audio_file(storage_path: str) -> bool:
    """
    Delete a stored audio file.
    Returns True if deleted, False if file not found.
    """
    try:
        path = Path(storage_path)
        if path.exists():
            path.unlink()  # removes the file
            logger.info(f"Deleted audio file at {storage_path}")
            return True
        else:
            logger.warning(f"Audio file not found at {storage_path}")
            return False
    except Exception as e:
        logger.error(f"Error deleting audio file at {storage_path}: {e}")
        return False


async def store_uploaded_file(audio_file, job_id: str) -> str:
    """Store uploaded audio file and return path"""
    storage_path = get_audio_storage_path(job_id, audio_file.filename)
    
    content = await audio_file.read()
    with open(storage_path, 'wb') as f:
        f.write(content)
    
    logger.info(f"Stored audio file for job {job_id} at {storage_path}")
    return storage_path