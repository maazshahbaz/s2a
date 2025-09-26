#!/usr/bin/env python3
"""
Intelligence Service Integration for S2A Pipeline
Integrates enhanced business intelligence extraction with the main S2A transcription pipeline
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from loguru import logger
from pydantic import BaseModel

from enhanced_extractor import EnhancedExtractor, ExtractionMode
from enhanced_schema import EnhancedBusinessIntelligence, SalesIntelligence, SupportIntelligence
from config import get_intelligence_settings


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class IntelligenceJob:
    """Intelligence processing job"""
    job_id: str
    transcript_id: str
    transcript_text: str
    mode: ExtractionMode = ExtractionMode.AUTO_DETECT
    priority: str = "normal"  # high, normal, low
    created_at: datetime = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class IntelligenceMetrics(BaseModel):
    """Intelligence service metrics"""
    total_jobs_processed: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    average_processing_time: float = 0.0
    queue_size: int = 0
    active_workers: int = 0
    uptime_hours: float = 0.0
    last_job_processed: Optional[str] = None

    # Mode-specific metrics
    sales_jobs: int = 0
    support_jobs: int = 0
    general_jobs: int = 0

    # Quality metrics
    avg_confidence_score: float = 0.0
    extraction_field_rates: Dict[str, float] = {}


class IntelligenceService:
    """
    Async intelligence processing service for S2A pipeline
    Processes transcriptions to extract comprehensive business intelligence
    """

    def __init__(self):
        self.settings = get_intelligence_settings()
        self.extractor = None
        self.job_queue: List[IntelligenceJob] = []
        self.processing_jobs: Dict[str, IntelligenceJob] = {}
        self.completed_jobs: Dict[str, IntelligenceJob] = {}

        self.metrics = IntelligenceMetrics()
        self.is_running = False
        self.start_time = None

        # Processing control
        self.max_concurrent_jobs = 3
        self.job_timeout = self.settings.processing_timeout

        logger.info("Intelligence service initialized")

    async def start(self) -> None:
        """Start the intelligence service"""
        if self.is_running:
            logger.warning("Intelligence service is already running")
            return

        try:
            # Initialize extractor
            self.extractor = EnhancedExtractor(
                base_url=self.settings.vllm_base_url,
                mode=ExtractionMode.AUTO_DETECT
            )

            self.is_running = True
            self.start_time = datetime.now()

            logger.info(f"Intelligence service started with vLLM at {self.settings.vllm_base_url}")

            # Start background processing task
            asyncio.create_task(self._process_queue())

        except Exception as e:
            logger.error(f"Failed to start intelligence service: {e}")
            raise

    async def stop(self) -> None:
        """Stop the intelligence service"""
        if not self.is_running:
            return

        self.is_running = False

        if self.extractor:
            self.extractor.close()

        logger.info("Intelligence service stopped")

    async def submit_job(self,
                        transcript_id: str,
                        transcript_text: str,
                        mode: ExtractionMode = ExtractionMode.AUTO_DETECT,
                        priority: str = "normal") -> str:
        """Submit a transcription for intelligence processing"""

        if not self.is_running:
            raise RuntimeError("Intelligence service is not running")

        # Generate job ID
        job_id = f"intel_{transcript_id}_{int(time.time())}"

        # Create job
        job = IntelligenceJob(
            job_id=job_id,
            transcript_id=transcript_id,
            transcript_text=transcript_text,
            mode=mode,
            priority=priority
        )

        # Add to queue (priority jobs go to front)
        if priority == "high":
            self.job_queue.insert(0, job)
        else:
            self.job_queue.append(job)

        self.metrics.queue_size = len(self.job_queue)

        logger.info(f"Submitted intelligence job {job_id} for transcript {transcript_id}")
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a specific job"""

        # Check processing jobs
        if job_id in self.processing_jobs:
            job = self.processing_jobs[job_id]
            return self._job_to_dict(job)

        # Check completed jobs
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return self._job_to_dict(job)

        # Check queue
        for job in self.job_queue:
            if job.job_id == job_id:
                return self._job_to_dict(job)

        return None

    async def get_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the result of a completed job"""
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            if job.status == ProcessingStatus.COMPLETED:
                return job.result
        return None

    async def _process_queue(self) -> None:
        """Background task to process the job queue"""
        logger.info("Started intelligence queue processing")

        while self.is_running:
            try:
                # Update uptime
                if self.start_time:
                    uptime = (datetime.now() - self.start_time).total_seconds() / 3600
                    self.metrics.uptime_hours = uptime

                # Process jobs if queue is not empty and we have capacity
                if (self.job_queue and
                    len(self.processing_jobs) < self.max_concurrent_jobs):

                    job = self.job_queue.pop(0)
                    self.metrics.queue_size = len(self.job_queue)

                    # Start processing
                    asyncio.create_task(self._process_job(job))

                # Clean up old completed jobs (keep last 100)
                if len(self.completed_jobs) > 100:
                    oldest_jobs = sorted(
                        self.completed_jobs.values(),
                        key=lambda j: j.created_at
                    )[:50]
                    for old_job in oldest_jobs:
                        del self.completed_jobs[old_job.job_id]

                # Sleep before next iteration
                await asyncio.sleep(self.settings.queue_check_interval)

            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
                await asyncio.sleep(5)  # Wait a bit on error

        logger.info("Intelligence queue processing stopped")

    async def _process_job(self, job: IntelligenceJob) -> None:
        """Process a single intelligence job"""
        start_time = time.time()

        try:
            # Mark as processing
            job.status = ProcessingStatus.PROCESSING
            self.processing_jobs[job.job_id] = job
            self.metrics.active_workers = len(self.processing_jobs)

            logger.info(f"Processing intelligence job {job.job_id}")

            # Extract intelligence (run in thread pool to avoid blocking)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.extractor.extract,
                job.transcript_text,
                job.mode
            )

            # Process result
            processing_time = time.time() - start_time
            job.processing_time = processing_time

            if result["success"]:
                job.status = ProcessingStatus.COMPLETED
                job.result = result

                # Update metrics
                self.metrics.successful_extractions += 1
                self._update_mode_metrics(result["mode"])
                self._update_quality_metrics(result)

                logger.info(f"Completed intelligence job {job.job_id} in {processing_time:.2f}s")

            else:
                job.status = ProcessingStatus.FAILED
                job.error = result.get("error", "Unknown error")
                self.metrics.failed_extractions += 1

                logger.error(f"Failed intelligence job {job.job_id}: {job.error}")

        except asyncio.TimeoutError:
            job.status = ProcessingStatus.FAILED
            job.error = "Processing timeout"
            self.metrics.failed_extractions += 1
            logger.error(f"Intelligence job {job.job_id} timed out")

        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.error = str(e)
            self.metrics.failed_extractions += 1
            logger.error(f"Error processing intelligence job {job.job_id}: {e}")

        finally:
            # Move to completed jobs
            if job.job_id in self.processing_jobs:
                del self.processing_jobs[job.job_id]

            self.completed_jobs[job.job_id] = job
            self.metrics.active_workers = len(self.processing_jobs)
            self.metrics.total_jobs_processed += 1
            self.metrics.last_job_processed = job.job_id

            # Update average processing time
            if job.processing_time:
                current_avg = self.metrics.average_processing_time
                total_jobs = self.metrics.total_jobs_processed

                self.metrics.average_processing_time = (
                    (current_avg * (total_jobs - 1) + job.processing_time) / total_jobs
                )

    def _update_mode_metrics(self, mode: str) -> None:
        """Update mode-specific metrics"""
        if mode == "sales":
            self.metrics.sales_jobs += 1
        elif mode == "support":
            self.metrics.support_jobs += 1
        else:
            self.metrics.general_jobs += 1

    def _update_quality_metrics(self, result: Dict[str, Any]) -> None:
        """Update quality metrics from successful extraction"""
        if not result.get("success"):
            return

        data = result.get("data", {})

        # Update confidence score average
        confidence = data.get("confidence_score", 0.0)
        current_avg = self.metrics.avg_confidence_score
        successful_jobs = self.metrics.successful_extractions

        self.metrics.avg_confidence_score = (
            (current_avg * (successful_jobs - 1) + confidence) / successful_jobs
        )

        # Update field extraction rates
        entities = data.get("entities", {})
        fields_to_track = [
            "action_items", "emails", "phones", "companies", "products",
            "invoice_ids", "order_ids", "dates", "financial_info"
        ]

        for field in fields_to_track:
            has_field = bool(data.get(field)) or bool(entities.get(field))

            if field not in self.metrics.extraction_field_rates:
                self.metrics.extraction_field_rates[field] = 0.0

            current_rate = self.metrics.extraction_field_rates[field]
            new_rate = (
                (current_rate * (successful_jobs - 1) + (1.0 if has_field else 0.0)) /
                successful_jobs
            )
            self.metrics.extraction_field_rates[field] = new_rate

    def _job_to_dict(self, job: IntelligenceJob) -> Dict[str, Any]:
        """Convert job to dictionary for API responses"""
        return {
            "job_id": job.job_id,
            "transcript_id": job.transcript_id,
            "mode": job.mode.value,
            "priority": job.priority,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "processing_time": job.processing_time,
            "error": job.error,
            "has_result": job.result is not None
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """Get current service metrics"""
        return self.metrics.model_dump()

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            "queue_size": len(self.job_queue),
            "processing_jobs": len(self.processing_jobs),
            "completed_jobs": len(self.completed_jobs),
            "is_running": self.is_running,
            "uptime_hours": self.metrics.uptime_hours
        }

    async def health_check(self) -> Dict[str, Any]:
        """Health check for the intelligence service"""
        is_healthy = self.is_running and self.extractor is not None

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "is_running": self.is_running,
            "extractor_available": self.extractor is not None,
            "queue_size": len(self.job_queue),
            "active_workers": len(self.processing_jobs),
            "total_processed": self.metrics.total_jobs_processed,
            "success_rate": (
                self.metrics.successful_extractions / max(1, self.metrics.total_jobs_processed)
            ) * 100
        }


# Global intelligence service instance
_intelligence_service: Optional[IntelligenceService] = None


async def get_intelligence_service() -> IntelligenceService:
    """Get the global intelligence service instance"""
    global _intelligence_service

    if _intelligence_service is None:
        _intelligence_service = IntelligenceService()

    return _intelligence_service


async def start_intelligence_service() -> None:
    """Start the global intelligence service"""
    service = await get_intelligence_service()
    await service.start()


async def stop_intelligence_service() -> None:
    """Stop the global intelligence service"""
    global _intelligence_service

    if _intelligence_service:
        await _intelligence_service.stop()
        _intelligence_service = None