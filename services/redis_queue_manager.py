"""
Redis queue manager for chunk processing.
Uses Redis Lists for FIFO queue with simple RPUSH/LPOP operations.
"""

import redis.asyncio as redis
import json
from typing import List, Optional, Dict
from loguru import logger
from .chunk_metadata import ChunkMetadata, ChunkResult, JobMetadata


class RedisQueueManager:
    """
    Manages chunk queues in Redis.

    Queue structure:
    - stt:chunks:pending (List) - FIFO queue of chunk IDs
    - stt:chunks:processing:{worker_id} (Set) - chunks being processed
    - stt:chunks:completed:{job_id} (Set) - completed chunk IDs per job
    - stt:results:{chunk_id} (String) - chunk results
    - stt:jobs:{job_id}:status (Hash) - job metadata
    - stt:chunks:data:{chunk_id} (String) - chunk metadata
    """

    def __init__(self, redis_client: redis.Redis, prefix: str = "stt"):
        self.redis = redis_client
        self.prefix = prefix

    # Queue keys
    @property
    def pending_queue(self) -> str:
        return f"{self.prefix}:chunks:pending"

    def processing_set(self, worker_id: str) -> str:
        return f"{self.prefix}:chunks:processing:{worker_id}"

    def completed_set(self, job_id: str) -> str:
        return f"{self.prefix}:chunks:completed:{job_id}"

    def result_key(self, chunk_id: str) -> str:
        return f"{self.prefix}:results:{chunk_id}"

    def job_status_key(self, job_id: str) -> str:
        return f"{self.prefix}:jobs:{job_id}:status"

    def chunk_data_key(self, chunk_id: str) -> str:
        return f"{self.prefix}:chunks:data:{chunk_id}"

    async def enqueue_chunks(self, chunks: List[ChunkMetadata]) -> None:
        """
        Add chunks to the pending queue.
        Stores chunk metadata and adds chunk_id to queue.
        """
        pipeline = self.redis.pipeline()

        job_metadata = {}  # Track job metadata

        for chunk in chunks:
            # Store chunk metadata
            pipeline.set(
                self.chunk_data_key(chunk.chunk_id),
                chunk.to_json(),
                ex=86400  # 24 hour expiry
            )

            # Add chunk_id to pending queue (FIFO)
            pipeline.rpush(self.pending_queue, chunk.chunk_id)

            # Collect job metadata
            if chunk.job_id not in job_metadata:
                job_metadata[chunk.job_id] = JobMetadata(
                    job_id=chunk.job_id,
                    audio_path=chunk.audio_path,
                    total_chunks=chunk.total_chunks,
                    sample_rate=chunk.sample_rate,
                    audio_duration=0,  # Will be calculated later
                    callback_url=chunk.callback_url,
                    status="processing"
                )

        # Store job metadata
        for job_id, metadata in job_metadata.items():
            pipeline.hset(
                self.job_status_key(job_id),
                mapping=metadata.to_dict()
            )
            pipeline.expire(self.job_status_key(job_id), 86400)

        await pipeline.execute()
        logger.info(f"Enqueued {len(chunks)} chunks from {len(job_metadata)} jobs")

    async def dequeue_chunks(
        self,
        worker_id: str,
        batch_size: int = 128,
        timeout: float = 1.0
    ) -> List[ChunkMetadata]:
        """
        Pull chunks from pending queue for processing.
        Can return chunks from different jobs to maximize batch utilization.
        """
        chunks = []
        chunk_ids_to_process = []

        # Pull up to batch_size chunks from queue
        # Use LPOP in a pipeline for efficiency
        pipeline = self.redis.pipeline()
        for _ in range(batch_size):
            pipeline.lpop(self.pending_queue)

        results = await pipeline.execute()

        # Filter out None results and decode bytes
        for result in results:
            if result:
                chunk_id = result.decode() if isinstance(result, bytes) else result
                chunk_ids_to_process.append(chunk_id)

        if not chunk_ids_to_process:
            return []

        # Fetch chunk metadata
        pipeline = self.redis.pipeline()
        for chunk_id in chunk_ids_to_process:
            pipeline.get(self.chunk_data_key(chunk_id))

        chunk_data_list = await pipeline.execute()

        # Parse chunks and mark as processing
        pipeline = self.redis.pipeline()
        processing_set_key = self.processing_set(worker_id)

        for chunk_id, chunk_data in zip(chunk_ids_to_process, chunk_data_list):
            if chunk_data:
                chunk = ChunkMetadata.from_json(chunk_data)
                chunks.append(chunk)
                # Mark as processing by this worker
                pipeline.sadd(processing_set_key, chunk_id)

        if chunks:
            pipeline.expire(processing_set_key, 3600)  # 1 hour expiry
            await pipeline.execute()

            # Log statistics
            unique_jobs = len(set(c.job_id for c in chunks))
            logger.info(f"Worker {worker_id} dequeued {len(chunks)} chunks from {unique_jobs} jobs")

        return chunks

    async def mark_chunk_completed(
        self,
        worker_id: str,
        chunk_result: ChunkResult
    ) -> bool:
        """
        Mark chunk as completed and check if job is ready for stitching.
        Returns True if all chunks for the job are completed.
        """
        chunk_id = chunk_result.chunk_id
        job_id = chunk_result.job_id

        pipeline = self.redis.pipeline()

        # Remove from processing set
        pipeline.srem(self.processing_set(worker_id), chunk_id)

        # Store result
        pipeline.set(
            self.result_key(chunk_id),
            chunk_result.to_json(),
            ex=86400  # 24 hour expiry
        )

        # Add to completed set for this job
        pipeline.sadd(self.completed_set(job_id), chunk_id)
        pipeline.expire(self.completed_set(job_id), 86400)

        # Get counts to check if job is complete
        pipeline.scard(self.completed_set(job_id))  # Completed count
        pipeline.hget(self.job_status_key(job_id), 'total_chunks')  # Total chunks

        results = await pipeline.execute()

        completed_count = results[-2]
        total_chunks = int(results[-1]) if results[-1] else 0

        # Check if all chunks are done
        all_done = completed_count == total_chunks and total_chunks > 0

        if all_done:
            # Update job status to stitching
            await self.redis.hset(
                self.job_status_key(job_id),
                'status', 'stitching'
            )
            logger.info(f"Job {job_id} ready for stitching: {completed_count}/{total_chunks} chunks complete")

        return all_done

    async def get_job_results(self, job_id: str) -> List[ChunkResult]:
        """
        Get all chunk results for a job in order.
        Used for stitching.
        """
        # Get all completed chunk IDs
        chunk_ids = await self.redis.smembers(self.completed_set(job_id))

        if not chunk_ids:
            return []

        # Fetch all results
        pipeline = self.redis.pipeline()
        for chunk_id in chunk_ids:
            chunk_id_str = chunk_id.decode() if isinstance(chunk_id, bytes) else chunk_id
            pipeline.get(self.result_key(chunk_id_str))

        results_data = await pipeline.execute()

        # Parse results
        results = []
        for data in results_data:
            if data:
                results.append(ChunkResult.from_json(data))

        # Sort by chunk index for proper ordering
        results.sort(key=lambda x: x.chunk_index)

        return results

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        error: Optional[str] = None
    ) -> None:
        """Update job status in Redis"""
        updates = {'status': status}
        if error:
            updates['error'] = error

        await self.redis.hset(
            self.job_status_key(job_id),
            mapping=updates
        )
        logger.info(f"Job {job_id} status updated to: {status}")

    async def cleanup_worker_processing(self, worker_id: str) -> List[str]:
        """
        Clean up processing set for a worker and return chunks to queue.
        Used when worker stops or crashes.
        """
        processing_key = self.processing_set(worker_id)
        chunk_ids = await self.redis.smembers(processing_key)

        if chunk_ids:
            pipeline = self.redis.pipeline()

            # Return chunks to pending queue
            for chunk_id in chunk_ids:
                chunk_id_str = chunk_id.decode() if isinstance(chunk_id, bytes) else chunk_id
                pipeline.lpush(self.pending_queue, chunk_id_str)  # Push to front

            # Clear processing set
            pipeline.delete(processing_key)

            await pipeline.execute()
            logger.warning(f"Returned {len(chunk_ids)} chunks from worker {worker_id} to queue")

        return [c.decode() if isinstance(c, bytes) else c for c in chunk_ids]

    async def get_queue_stats(self) -> Dict:
        """Get queue statistics"""
        pipeline = self.redis.pipeline()
        pipeline.llen(self.pending_queue)
        pipeline.keys(f"{self.prefix}:chunks:processing:*")
        pipeline.keys(f"{self.prefix}:chunks:completed:*")

        results = await pipeline.execute()

        pending_count = results[0]
        processing_workers = len(results[1])
        jobs_with_completions = len(results[2])

        return {
            'pending_chunks': pending_count,
            'active_workers': processing_workers,
            'jobs_in_progress': jobs_with_completions
        }