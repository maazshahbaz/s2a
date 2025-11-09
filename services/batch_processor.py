"""
Redis-based batch processor with chunk queue architecture.
Maximizes GPU utilization by batching chunks from multiple jobs.
"""

import asyncio
import redis.asyncio as redis
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

from .redis_queue_manager import RedisQueueManager
from .chunk_generator import ChunkGenerator
from .chunk_worker import ChunkWorker, AudioCache
from .stitching_service import StitchingService


@dataclass
class BatchProcessorConfig:
    """Configuration for batch processor"""
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    batch_size: int = 128  # Maximum chunks per batch
    num_workers: int = 1  # 1 worker is sufficient with batch_size=128 (processes 3000+ min/min)
    max_chunk_duration: float = 1440.0  # 24 minutes (Parakeet limit)
    overlap_duration: float = 5.0  # 5 seconds overlap

    audio_cache_size: int = 10  # Number of audio files to cache
    worker_executor_threads: int = 4  # Threads per worker


class BatchProcessor:
    """
    Redis-based batch processor with chunk queue.

    Architecture:
    1. Audio files stored once on disk
    2. Chunks are metadata only (no physical files)
    3. Workers pull chunks from Redis queue
    4. Chunks from different jobs batched together
    5. GPU processes mixed batches for maximum efficiency
    6. Automatic stitching when job completes
    """

    def __init__(
        self,
        asr_service,
        db,
        triton_service,
        config: BatchProcessorConfig = None
    ):
        self.db=db
        self.asr_service = asr_service
        self.triton_service=triton_service
        self.config = config or BatchProcessorConfig()

        # Redis connection
        self.redis_client = None
        self.redis_queue = None

        # Workers
        self.workers: List[ChunkWorker] = []
        self.worker_tasks: List[asyncio.Task] = []

        # Shared resources
        self.audio_cache = AudioCache(max_cache_size=self.config.audio_cache_size)
        self.executor = ThreadPoolExecutor(max_workers=self.config.worker_executor_threads)

        self._running = False

    async def start(self):
        """Start the batch processor"""
        if self._running:
            logger.warning("Batch processor already running")
            return

        logger.info("Starting new batch processor")

        # Connect to Redis
        self.redis_client = await redis.Redis(
            host=self.config.redis_host,
            port=self.config.redis_port,
            db=self.config.redis_db,
            password=self.config.redis_password,
            decode_responses=False  # We'll handle encoding/decoding
        )

        # Test connection
        await self.redis_client.ping()
        logger.info(f"Connected to Redis at {self.config.redis_host}:{self.config.redis_port}")

        # Create queue manager
        self.redis_queue = RedisQueueManager(self.redis_client)

        # Start workers
        for i in range(self.config.num_workers):
            worker = ChunkWorker(
                db=self.db,
                triton_service=self.triton_service,
                worker_id=f"worker_{i}",
                asr_service=self.asr_service,
                redis_queue=self.redis_queue,
                batch_size=self.config.batch_size,
                audio_cache=self.audio_cache,
                executor=self.executor
            )
            self.workers.append(worker)

            # Start worker task
            task = asyncio.create_task(worker.start())
            self.worker_tasks.append(task)

        self._running = True
        logger.info(f"Started {self.config.num_workers} workers with batch_size={self.config.batch_size}")

    async def stop(self):
        """Stop the batch processor"""
        if not self._running:
            return

        logger.info("Stopping batch processor")

        # Stop workers
        for worker in self.workers:
            await worker.stop()

        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()

        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)

        # Clear caches
        await self.audio_cache.clear()

        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        self._running = False
        logger.info("Batch processor stopped")

    async def submit_job(
        self,
        job_id: str,
        audio_path: str,
        include_intelligence: bool = False,
        callback_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit an audio file for processing.

        Args:
            job_id: Unique job identifier
            audio_path: Path to audio file (stored once)
            callback_url: Webhook URL for completion notification

        Returns:
            Job submission result
        """
        if not self._running:
            raise RuntimeError("Batch processor not running")

        try:
            # Get audio duration
            duration = await self._get_audio_duration(audio_path)
            logger.info(f"Job {job_id}: Audio duration {duration:.1f}s")

            # Generate chunk metadata (no physical files)
            chunks = ChunkGenerator.generate_chunks(
                job_id=job_id,
                audio_path=audio_path,
                audio_duration=duration,
                sample_rate=16000,  # Will be verified when loading
                max_chunk_duration=self.config.max_chunk_duration,
                overlap_duration=self.config.overlap_duration,
                include_intelligence=include_intelligence,
                callback_url=callback_url
            )

            # Enqueue chunks to Redis
            await self.redis_queue.enqueue_chunks(chunks)

            # Estimate processing time
            estimated_time = ChunkGenerator.estimate_processing_time(chunks)

            return {
                'job_id': job_id,
                'status': 'queued',
                'audio_duration': duration,
                'num_chunks': len(chunks),
                'estimated_processing_time': estimated_time
            }

        except Exception as e:
            logger.error(f"Failed to submit job {job_id}: {e}")
            raise

    async def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using librosa"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._get_audio_duration_sync,
            audio_path
        )

    def _get_audio_duration_sync(self, audio_path: str) -> float:
        """Get audio duration (synchronous)"""
        try:
            # Try librosa first (faster for getting duration)
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            # Fallback to soundfile
            audio, sr = sf.read(audio_path)
            return len(audio) / sr

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a job"""
        if not self.redis_queue:
            raise RuntimeError("Batch processor not initialized")

        # Get job metadata from Redis
        job_key = self.redis_queue.job_status_key(job_id)
        job_data = await self.redis_client.hgetall(job_key)

        if not job_data:
            return {'job_id': job_id, 'status': 'not_found'}

        # Decode and parse
        job_info = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in job_data.items()
        }

        # Get completion stats
        completed_count = await self.redis_client.scard(
            self.redis_queue.completed_set(job_id)
        )

        return {
            'job_id': job_id,
            'status': job_info.get('status', 'unknown'),
            'total_chunks': int(job_info.get('total_chunks', 0)),
            'completed_chunks': completed_count,
            'audio_path': job_info.get('audio_path'),
            'created_at': float(job_info.get('created_at', 0))
        }

    async def get_job_result(self, job_id: str) -> Optional[str]:
        """
        Get final transcription result for a completed job.

        Args:
            job_id: Job identifier

        Returns:
            Final stitched transcription or None if not ready
        """
        if not self.redis_queue:
            raise RuntimeError("Batch processor not initialized")

        # Check job status
        status = await self.get_job_status(job_id)

        if status['status'] != 'completed':
            return None

        # Get all chunk results
        chunk_results = await self.redis_queue.get_job_results(job_id)

        if not chunk_results:
            return None

        # Stitch results (with config values from asr_service)
        stitching_service = StitchingService(
            words_per_second=self.asr_service.words_per_second,
            overlap_similarity_threshold=self.asr_service.overlap_similarity_threshold
        )
        final_text = await stitching_service.stitch_transcriptions(
            chunk_results,
            remove_overlap=True
        )

        return final_text

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        if not self.redis_queue:
            return {}

        stats = await self.redis_queue.get_queue_stats()

        # Add worker stats
        worker_stats = []
        for worker in self.workers:
            worker_stats.append({
                'worker_id': worker.worker_id,
                **worker.get_stats()
            })

        stats['workers'] = worker_stats
        stats['batch_size'] = self.config.batch_size
        stats['num_workers'] = self.config.num_workers

        return stats


# Compatibility wrapper for existing code
async def create_batch_processor(asr_service, config_dict: Optional[Dict] = None):
    """
    Create and start a batch processor.

    Args:
        asr_service: ASR service instance
        config_dict: Optional configuration dictionary

    Returns:
        Started batch processor instance
    """
    # Convert dict to config object
    if config_dict:
        config = BatchProcessorConfig(**config_dict)
    else:
        config = BatchProcessorConfig()

    processor = BatchProcessor(asr_service, config)
    await processor.start()

    return processor