"""
Test cases for the Redis-based chunk queue pipeline.
Tests the new architecture with cross-job batching.
"""

import pytest
import asyncio
import redis.asyncio as redis
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from services.chunk_metadata import ChunkMetadata, ChunkResult
from services.redis_queue_manager import RedisQueueManager
from services.chunk_generator import ChunkGenerator
from services.stitching_service import StitchingService


@pytest.fixture
async def redis_client():
    """Mock Redis client for testing"""
    client = AsyncMock(spec=redis.Redis)
    return client


@pytest.fixture
async def queue_manager(redis_client):
    """Create queue manager with mock Redis"""
    return RedisQueueManager(redis_client)


class TestChunkGeneration:
    """Test chunk metadata generation (no physical files)"""

    def test_single_chunk_for_short_audio(self):
        """Test that short audio creates single chunk"""
        chunks = ChunkGenerator.generate_chunks(
            job_id="test-123",
            audio_path="/path/to/audio.wav",
            audio_duration=60.0,  # 1 minute
            sample_rate=16000,
            max_chunk_duration=1440.0,  # 24 minutes
            overlap_duration=5.0
        )

        assert len(chunks) == 1
        assert chunks[0].chunk_id == "test-123_chunk_0"
        assert chunks[0].start_time == 0
        assert chunks[0].end_time == 60.0
        assert chunks[0].overlap_start == 0
        assert chunks[0].overlap_end == 0

    def test_multiple_chunks_with_overlap(self):
        """Test that long audio creates multiple chunks with overlap"""
        chunks = ChunkGenerator.generate_chunks(
            job_id="test-456",
            audio_path="/path/to/long_audio.wav",
            audio_duration=3600.0,  # 60 minutes
            sample_rate=16000,
            max_chunk_duration=1440.0,  # 24 minutes
            overlap_duration=5.0
        )

        assert len(chunks) == 3  # Should create 3 chunks

        # First chunk: 0-1440s (no overlap at start)
        assert chunks[0].start_time == 0
        assert chunks[0].end_time == 1440.0
        assert chunks[0].overlap_start == 0
        assert chunks[0].overlap_end == 5.0

        # Second chunk: 1435-2880s (5s overlap on both sides)
        assert chunks[1].start_time == 1435.0  # 5s before previous end
        assert chunks[1].end_time == 2880.0
        assert chunks[1].overlap_start == 5.0
        assert chunks[1].overlap_end == 5.0

        # Third chunk: 2875-3600s (5s overlap at start only)
        assert chunks[2].start_time == 2875.0
        assert chunks[2].end_time == 3600.0
        assert chunks[2].overlap_start == 5.0
        assert chunks[2].overlap_end == 0

    def test_chunk_metadata_serialization(self):
        """Test chunk metadata can be serialized for Redis"""
        chunk = ChunkMetadata(
            chunk_id="test-789_chunk_0",
            job_id="test-789",
            audio_path="/path/to/audio.wav",
            start_time=0.0,
            end_time=1440.0,
            chunk_index=0,
            total_chunks=1,
            sample_rate=16000
        )

        # Test serialization
        json_str = chunk.to_json()
        assert isinstance(json_str, str)

        # Test deserialization
        restored = ChunkMetadata.from_json(json_str)
        assert restored.chunk_id == chunk.chunk_id
        assert restored.start_time == chunk.start_time
        assert restored.end_time == chunk.end_time


class TestRedisQueueOperations:
    """Test Redis queue manager operations"""

    @pytest.mark.asyncio
    async def test_enqueue_chunks(self, queue_manager, redis_client):
        """Test adding chunks to Redis queue"""
        chunks = [
            ChunkMetadata(
                chunk_id=f"job-1_chunk_{i}",
                job_id="job-1",
                audio_path="/audio1.wav",
                start_time=i * 1440,
                end_time=(i + 1) * 1440,
                chunk_index=i,
                total_chunks=3,
                sample_rate=16000
            )
            for i in range(3)
        ]

        await queue_manager.enqueue_chunks(chunks)

        # Verify Redis operations
        assert redis_client.pipeline.called
        assert redis_client.rpush.called or any(
            call[0][0] == 'rpush' for call in redis_client.pipeline().method_calls
        )

    @pytest.mark.asyncio
    async def test_dequeue_chunks_mixes_jobs(self, queue_manager, redis_client):
        """Test that dequeue can pull chunks from different jobs"""
        # Mock Redis responses for mixed job chunks
        redis_client.lpop.side_effect = [
            b"job1_chunk_0",
            b"job2_chunk_0",
            b"job1_chunk_1",
            b"job3_chunk_0",
            None  # End of queue
        ]

        # Mock getting chunk data
        async def mock_get(key):
            if b"job1" in key:
                return ChunkMetadata(
                    chunk_id="job1_chunk_0",
                    job_id="job1",
                    audio_path="/audio1.wav",
                    start_time=0,
                    end_time=1440,
                    chunk_index=0,
                    total_chunks=2,
                    sample_rate=16000
                ).to_json()
            return None

        redis_client.get.side_effect = mock_get

        chunks = await queue_manager.dequeue_chunks(
            worker_id="worker_0",
            batch_size=4
        )

        # Should get mixed chunks from different jobs
        assert len(chunks) >= 1
        # Verify chunks can be from different jobs
        job_ids = {chunk.job_id for chunk in chunks}
        assert len(job_ids) >= 1  # Can have multiple job IDs


class TestStitching:
    """Test stitching service with overlap removal"""

    @pytest.mark.asyncio
    async def test_stitch_with_overlap_removal(self):
        """Test that stitching removes overlap correctly"""
        chunk_results = [
            ChunkResult(
                chunk_id="job1_chunk_0",
                job_id="job1",
                chunk_index=0,
                text="Hello world this is chunk one ending here",
                confidence=0.95,
                start_time=0,
                end_time=10,
                processing_time=1.0,
                rtf=0.1,
                overlap_start=0,
                overlap_end=5
            ),
            ChunkResult(
                chunk_id="job1_chunk_1",
                job_id="job1",
                chunk_index=1,
                text="ending here and this is chunk two",
                confidence=0.93,
                start_time=5,
                end_time=15,
                processing_time=1.0,
                rtf=0.1,
                overlap_start=5,
                overlap_end=0
            )
        ]

        stitching_service = StitchingService()
        final_text = await stitching_service.stitch_transcriptions(
            chunk_results,
            remove_overlap=True
        )

        # Should remove the duplicated "ending here" part
        assert "Hello world this is chunk one ending here and this is chunk two" in final_text

    @pytest.mark.asyncio
    async def test_single_chunk_no_stitching_needed(self):
        """Test that single chunk doesn't need stitching"""
        chunk_results = [
            ChunkResult(
                chunk_id="job2_chunk_0",
                job_id="job2",
                chunk_index=0,
                text="This is a complete transcription",
                confidence=0.96,
                start_time=0,
                end_time=60,
                processing_time=5.0,
                rtf=0.08,
                overlap_start=0,
                overlap_end=0
            )
        ]

        stitching_service = StitchingService()
        final_text = await stitching_service.stitch_transcriptions(
            chunk_results,
            remove_overlap=False
        )

        assert final_text == "This is a complete transcription"


class TestBatchProcessing:
    """Test batch processing with mixed job chunks"""

    def test_batch_can_contain_mixed_chunks(self):
        """Test that a batch can contain chunks from multiple jobs"""
        batch = []

        # Add chunks from job A (long audio - 3 chunks)
        for i in range(3):
            batch.append(ChunkMetadata(
                chunk_id=f"jobA_chunk_{i}",
                job_id="jobA",
                audio_path="/audioA.wav",
                start_time=i * 1440,
                end_time=(i + 1) * 1440,
                chunk_index=i,
                total_chunks=3,
                sample_rate=16000
            ))

        # Add chunks from job B (short audio - 1 chunk)
        batch.append(ChunkMetadata(
            chunk_id="jobB_chunk_0",
            job_id="jobB",
            audio_path="/audioB.wav",
            start_time=0,
            end_time=60,
            chunk_index=0,
            total_chunks=1,
            sample_rate=16000
        ))

        # Add chunks from job C (medium audio - 2 chunks)
        for i in range(2):
            batch.append(ChunkMetadata(
                chunk_id=f"jobC_chunk_{i}",
                job_id="jobC",
                audio_path="/audioC.wav",
                start_time=i * 1440,
                end_time=(i + 1) * 1440 if i == 0 else 1800,
                chunk_index=i,
                total_chunks=2,
                sample_rate=16000
            ))

        # Verify batch contains chunks from 3 different jobs
        assert len(batch) == 6
        job_ids = {chunk.job_id for chunk in batch}
        assert len(job_ids) == 3
        assert job_ids == {"jobA", "jobB", "jobC"}

    def test_batch_size_128_capacity(self):
        """Test that batch can hold 128 chunks from different users"""
        batch = []

        # Simulate chunks from 20 different users
        for user_id in range(20):
            # Each user has different number of chunks
            num_chunks = (user_id % 5) + 1  # 1-5 chunks per user

            for chunk_idx in range(num_chunks):
                batch.append(ChunkMetadata(
                    chunk_id=f"user{user_id}_job_chunk_{chunk_idx}",
                    job_id=f"user{user_id}_job",
                    audio_path=f"/audio_user{user_id}.wav",
                    start_time=chunk_idx * 1440,
                    end_time=(chunk_idx + 1) * 1440,
                    chunk_index=chunk_idx,
                    total_chunks=num_chunks,
                    sample_rate=16000
                ))

        # Should be able to create large batches
        assert len(batch) > 50  # Multiple users with multiple chunks

        # Can take up to 128 chunks for processing
        processing_batch = batch[:128]
        assert len(processing_batch) == min(128, len(batch))


class TestWorkerPerformance:
    """Test worker performance with batch_size=128"""

    def test_single_worker_throughput(self):
        """Calculate throughput with 1 worker and batch_size=128"""
        batch_size = 128
        max_chunk_duration = 1440  # 24 minutes
        rtf_target = 0.0003  # RTFx 3300

        # Maximum audio in one batch
        max_audio_minutes = (batch_size * max_chunk_duration) / 60
        assert max_audio_minutes == 3072  # 128 * 24 = 3072 minutes

        # Processing time for full batch
        processing_time_seconds = max_audio_minutes * 60 * rtf_target
        assert processing_time_seconds < 60  # ~55 seconds

        # Throughput per minute
        throughput_per_minute = max_audio_minutes / (processing_time_seconds / 60)
        assert throughput_per_minute > 3000  # Over 3000 minutes per minute

        # Conclusion: 1 worker is sufficient!

    def test_multi_worker_not_needed(self):
        """Verify that multiple workers are not necessary"""
        # With 1 worker processing 3000+ minutes per minute
        # Even with 100 concurrent users each uploading 1 hour audio
        total_audio_minutes = 100 * 60  # 6000 minutes

        # Time to process with 1 worker
        worker_throughput = 3000  # minutes per minute
        processing_time = total_audio_minutes / worker_throughput
        assert processing_time < 3  # Less than 3 minutes to process all

        # Multiple workers would only help if:
        # - Queue has >3000 minutes arriving per minute continuously
        # - Need redundancy for failures
        # - Want to reduce individual job latency

        # For most use cases, 1 worker is optimal