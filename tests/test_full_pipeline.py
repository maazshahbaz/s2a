"""
End-to-end test cases for the complete STT pipeline.
Tests with real audio files from test_audio directory.
"""

import pytest
import asyncio
import redis.asyncio as redis
import os
import uuid
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List

from services.batch_processor import BatchProcessor, BatchProcessorConfig
from services.chunk_generator import ChunkGenerator
from services.redis_queue_manager import RedisQueueManager
from services.chunk_metadata import ChunkMetadata, ChunkResult
from services.stitching_service import StitchingService
from services.asr_service import NeMoASRService


# Test audio files with their expected durations and chunk counts
TEST_AUDIO_FILES = {
    "short": {
        "path": "tests/test_audio/in-9524528884-2014569295-20250515-095120-1747320680.72350.wav",
        "duration": 40.5,
        "expected_chunks": 1,
        "description": "40.5 seconds - No chunking needed"
    },
    "medium": {
        "path": "tests/test_audio/in-6123987606-6123773368-20250513-142151-1747164111.64201.wav",
        "duration": 192.7,
        "expected_chunks": 1,
        "description": "3.2 minutes - No chunking needed"
    },
    "long": {
        "path": "tests/test_audio/in-9524528884-2058527609-20250125-132037-1737832837.3553.wav",
        "duration": 2008.0,
        "expected_chunks": 2,
        "description": "33.5 minutes - Needs 2 chunks"
    }
}


@pytest.fixture(scope="module")
async def redis_client():
    """Create Redis client for testing"""
    client = await redis.Redis(
        host=os.getenv("S2A_REDIS_HOST", "localhost"),
        port=int(os.getenv("S2A_REDIS_PORT", 6379)),
        db=int(os.getenv("S2A_REDIS_DB", 1)),  # Use DB 1 for tests
        decode_responses=False
    )

    # Clear test database
    await client.flushdb()

    yield client

    # Cleanup
    await client.flushdb()
    await client.close()


@pytest.fixture(scope="module")
def asr_service():
    """Create ASR service (mock or real depending on GPU availability)"""
    try:
        # Try to create real ASR service
        service = NeMoASRService(
            model_name="nvidia/parakeet-tdt-0.6b-v2",
            device="cuda",
            min_audio_duration=1.0
        )
        return service
    except Exception:
        # Fall back to mock if GPU not available
        from unittest.mock import Mock
        mock_service = Mock()
        mock_service.transcribe_batch_nemo = Mock(return_value=[
            {"text": "Mock transcription result", "confidence": 0.95}
        ])
        return mock_service


@pytest.fixture
async def batch_processor(asr_service, redis_client):
    """Create batch processor for testing"""
    config = BatchProcessorConfig(
        redis_host="localhost",
        redis_port=6379,
        redis_db=1,  # Test DB
        batch_size=128,
        num_workers=1,
        max_chunk_duration=1440.0,
        overlap_duration=5.0
    )

    processor = BatchProcessor(asr_service, config)
    await processor.start()

    yield processor

    await processor.stop()


class TestChunkGeneration:
    """Test chunk generation for different audio lengths"""

    def test_short_audio_single_chunk(self):
        """Test that 40.5 second audio generates single chunk"""
        audio_info = TEST_AUDIO_FILES["short"]

        chunks = ChunkGenerator.generate_chunks(
            job_id="test-short-001",
            audio_path=audio_info["path"],
            audio_duration=audio_info["duration"],
            sample_rate=16000
        )

        assert len(chunks) == audio_info["expected_chunks"]
        assert chunks[0].start_time == 0
        assert chunks[0].end_time == audio_info["duration"]
        assert chunks[0].overlap_start == 0
        assert chunks[0].overlap_end == 0

    def test_medium_audio_single_chunk(self):
        """Test that 3.2 minute audio generates single chunk"""
        audio_info = TEST_AUDIO_FILES["medium"]

        chunks = ChunkGenerator.generate_chunks(
            job_id="test-medium-001",
            audio_path=audio_info["path"],
            audio_duration=audio_info["duration"],
            sample_rate=16000
        )

        assert len(chunks) == audio_info["expected_chunks"]
        assert chunks[0].start_time == 0
        assert chunks[0].end_time == audio_info["duration"]
        assert chunks[0].overlap_start == 0
        assert chunks[0].overlap_end == 0

    def test_long_audio_multiple_chunks(self):
        """Test that 33.5 minute audio generates 2 chunks with overlap"""
        audio_info = TEST_AUDIO_FILES["long"]

        chunks = ChunkGenerator.generate_chunks(
            job_id="test-long-001",
            audio_path=audio_info["path"],
            audio_duration=audio_info["duration"],
            sample_rate=16000,
            max_chunk_duration=1440.0,  # 24 minutes
            overlap_duration=5.0
        )

        assert len(chunks) == audio_info["expected_chunks"]

        # First chunk: 0-1440 seconds
        assert chunks[0].start_time == 0
        assert chunks[0].end_time == 1440.0
        assert chunks[0].overlap_start == 0
        assert chunks[0].overlap_end == 5.0

        # Second chunk: 1435-2008 seconds (5s overlap)
        assert chunks[1].start_time == 1435.0  # 5 seconds before first chunk ends
        assert chunks[1].end_time == audio_info["duration"]
        assert chunks[1].overlap_start == 5.0
        assert chunks[1].overlap_end == 0


@pytest.mark.asyncio
class TestRedisQueueOperations:
    """Test Redis queue operations with real chunks"""

    async def test_enqueue_all_test_files(self, redis_client):
        """Test enqueueing chunks from all test files"""
        queue_manager = RedisQueueManager(redis_client)
        all_chunks = []

        for file_type, audio_info in TEST_AUDIO_FILES.items():
            job_id = f"test-{file_type}-{uuid.uuid4().hex[:8]}"

            chunks = ChunkGenerator.generate_chunks(
                job_id=job_id,
                audio_path=audio_info["path"],
                audio_duration=audio_info["duration"],
                sample_rate=16000,
                callback_url=f"http://test.webhook/{job_id}"
            )

            all_chunks.extend(chunks)

        # Enqueue all chunks
        await queue_manager.enqueue_chunks(all_chunks)

        # Check queue depth
        queue_depth = await redis_client.llen(queue_manager.pending_queue)

        # Should have 1 + 1 + 2 = 4 chunks total
        expected_total = sum(info["expected_chunks"] for info in TEST_AUDIO_FILES.values())
        assert queue_depth == expected_total

    async def test_mixed_job_batching(self, redis_client):
        """Test that worker can pull chunks from different jobs in one batch"""
        queue_manager = RedisQueueManager(redis_client)

        # Create chunks from different jobs
        chunks_job1 = ChunkGenerator.generate_chunks(
            job_id="job1",
            audio_path=TEST_AUDIO_FILES["short"]["path"],
            audio_duration=TEST_AUDIO_FILES["short"]["duration"],
            sample_rate=16000
        )

        chunks_job2 = ChunkGenerator.generate_chunks(
            job_id="job2",
            audio_path=TEST_AUDIO_FILES["long"]["path"],
            audio_duration=TEST_AUDIO_FILES["long"]["duration"],
            sample_rate=16000
        )

        # Enqueue all
        await queue_manager.enqueue_chunks(chunks_job1 + chunks_job2)

        # Worker pulls batch
        batch = await queue_manager.dequeue_chunks(
            worker_id="test_worker",
            batch_size=128
        )

        # Should get mixed chunks
        job_ids = {chunk.job_id for chunk in batch}
        assert len(job_ids) == 2  # Chunks from both jobs
        assert "job1" in job_ids
        assert "job2" in job_ids


@pytest.mark.asyncio
@pytest.mark.integration
class TestFullPipeline:
    """Integration tests for the complete pipeline"""

    async def test_single_file_no_chunking(self, batch_processor):
        """Test processing a short audio file that doesn't need chunking"""
        audio_info = TEST_AUDIO_FILES["short"]
        job_id = f"test-short-{uuid.uuid4().hex[:8]}"

        # Submit job
        result = await batch_processor.submit_job(
            job_id=job_id,
            audio_path=audio_info["path"],
            callback_url=f"http://test.webhook/{job_id}"
        )

        assert result["status"] == "queued"
        assert result["num_chunks"] == 1
        assert result["audio_duration"] == pytest.approx(audio_info["duration"], rel=0.1)

        # Wait for processing (mock or real)
        await asyncio.sleep(2)

        # Check job status
        status = await batch_processor.get_job_status(job_id)
        assert status["job_id"] == job_id
        assert status["total_chunks"] == 1

    async def test_long_file_with_chunking(self, batch_processor):
        """Test processing a long audio file that needs chunking"""
        audio_info = TEST_AUDIO_FILES["long"]
        job_id = f"test-long-{uuid.uuid4().hex[:8]}"

        # Submit job
        result = await batch_processor.submit_job(
            job_id=job_id,
            audio_path=audio_info["path"],
            callback_url=f"http://test.webhook/{job_id}"
        )

        assert result["status"] == "queued"
        assert result["num_chunks"] == 2  # Should create 2 chunks
        assert result["audio_duration"] == pytest.approx(audio_info["duration"], rel=0.1)

        # Wait for processing
        await asyncio.sleep(5)

        # Check job status
        status = await batch_processor.get_job_status(job_id)
        assert status["job_id"] == job_id
        assert status["total_chunks"] == 2

    async def test_concurrent_jobs(self, batch_processor):
        """Test processing multiple audio files concurrently"""
        jobs = []

        # Submit all three test files
        for file_type, audio_info in TEST_AUDIO_FILES.items():
            job_id = f"test-concurrent-{file_type}-{uuid.uuid4().hex[:8]}"

            result = await batch_processor.submit_job(
                job_id=job_id,
                audio_path=audio_info["path"],
                callback_url=f"http://test.webhook/{job_id}"
            )

            jobs.append({
                "job_id": job_id,
                "expected_chunks": audio_info["expected_chunks"],
                "result": result
            })

        # All jobs should be queued
        for job in jobs:
            assert job["result"]["status"] == "queued"
            assert job["result"]["num_chunks"] == job["expected_chunks"]

        # Wait for processing
        await asyncio.sleep(10)

        # Check all job statuses
        for job in jobs:
            status = await batch_processor.get_job_status(job["job_id"])
            assert status["total_chunks"] == job["expected_chunks"]


class TestStitching:
    """Test stitching functionality"""

    @pytest.mark.asyncio
    async def test_stitch_two_chunks(self):
        """Test stitching two chunks with overlap removal"""
        # Simulate chunk results from the long audio file
        chunk_results = [
            ChunkResult(
                chunk_id="job1_chunk_0",
                job_id="job1",
                chunk_index=0,
                text="This is the first chunk of audio ending with some words",
                confidence=0.95,
                start_time=0,
                end_time=1440,
                processing_time=10.0,
                rtf=0.007,
                overlap_start=0,
                overlap_end=5.0
            ),
            ChunkResult(
                chunk_id="job1_chunk_1",
                job_id="job1",
                chunk_index=1,
                text="ending with some words and continuing with the second chunk",
                confidence=0.93,
                start_time=1435,
                end_time=2008,
                processing_time=4.0,
                rtf=0.007,
                overlap_start=5.0,
                overlap_end=0
            )
        ]

        stitching_service = StitchingService()
        final_text = await stitching_service.stitch_transcriptions(
            chunk_results,
            remove_overlap=True
        )

        # Should remove the duplicate "ending with some words"
        assert "This is the first chunk of audio ending with some words and continuing with the second chunk" in final_text

        # Calculate overall confidence
        confidence = stitching_service.calculate_confidence(chunk_results)
        assert 0.9 < confidence < 1.0

        # Calculate overall RTF
        rtf = stitching_service.calculate_rtf(chunk_results)
        assert rtf < 0.1  # Should be very efficient


@pytest.mark.asyncio
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for the pipeline"""

    async def test_throughput_calculation(self):
        """Calculate theoretical throughput with batch_size=128"""
        batch_size = 128
        max_chunk_duration = 1440  # 24 minutes
        target_rtf = 0.0003  # RTFx 3300

        # Total audio in one batch
        total_audio_seconds = batch_size * max_chunk_duration
        total_audio_minutes = total_audio_seconds / 60

        # Processing time
        processing_time = total_audio_seconds * target_rtf

        # Throughput
        throughput_per_minute = total_audio_minutes / (processing_time / 60)

        assert total_audio_minutes == 3072  # 128 * 24
        assert processing_time < 60  # Should process in under a minute
        assert throughput_per_minute > 3000  # Over 3000 minutes per minute

    async def test_queue_capacity(self, redis_client):
        """Test that queue can handle large number of chunks"""
        queue_manager = RedisQueueManager(redis_client)

        # Simulate 100 jobs, each with 2 chunks
        all_chunks = []
        for i in range(100):
            chunks = ChunkGenerator.generate_chunks(
                job_id=f"load-test-{i}",
                audio_path="/dummy/path.wav",
                audio_duration=2000,  # ~33 minutes (needs 2 chunks)
                sample_rate=16000
            )
            all_chunks.extend(chunks)

        # Should have 200 chunks total
        assert len(all_chunks) == 200

        # Enqueue all
        await queue_manager.enqueue_chunks(all_chunks)

        # Check queue depth
        queue_depth = await redis_client.llen(queue_manager.pending_queue)
        assert queue_depth == 200

        # Worker can pull up to 128 at once
        batch = await queue_manager.dequeue_chunks(
            worker_id="test_worker",
            batch_size=128
        )

        assert len(batch) == 128  # Got full batch

        # Different jobs in batch
        job_ids = {chunk.job_id for chunk in batch}
        assert len(job_ids) > 1  # Mixed jobs in batch