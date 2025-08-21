import asyncio
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import time
from loguru import logger
from collections import deque
import threading
from enum import Enum

class BatchStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class BatchJob:
    job_id: str
    audio_data: np.ndarray
    metadata: Dict = field(default_factory=dict)
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    status: BatchStatus = BatchStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None

@dataclass
class BatchConfig:
    max_batch_size: int = 8
    max_queue_size: int = 100
    processing_timeout: float = 300.0  # 5 minutes
    dynamic_batching: bool = True
    batch_timeout_ms: int = 100
    gpu_memory_fraction: float = 0.8
    enable_mixed_precision: bool = True

class GPUMemoryManager:
    def __init__(self, target_utilization: float = 0.8):
        self.target_utilization = target_utilization
        self._lock = threading.Lock()
        
    def get_available_memory(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        
        with self._lock:
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated(0)
            available = (total_memory - allocated_memory) / total_memory
            
        return available
    
    def can_process_batch(self, batch_size: int, estimated_memory_per_item: float = 0.1) -> bool:
        available = self.get_available_memory()
        required = batch_size * estimated_memory_per_item
        
        return available >= required
    
    def optimize_batch_size(self, requested_size: int, max_size: int) -> int:
        if not torch.cuda.is_available():
            return min(requested_size, max_size)
        
        available_memory = self.get_available_memory()
        
        # Estimate memory usage per item (rough approximation)
        memory_per_item = 0.1  # 10% of GPU memory per item (conservative)
        
        max_items_by_memory = int(available_memory * self.target_utilization / memory_per_item)
        optimal_size = min(requested_size, max_size, max_items_by_memory)
        
        return max(1, optimal_size)

class DynamicBatcher:
    def __init__(self, config: BatchConfig):
        self.config = config
        self.job_queue = asyncio.Queue(maxsize=config.max_queue_size)
        self.result_store: Dict[str, BatchJob] = {}
        self.processing_jobs: Dict[str, BatchJob] = {}
        self.gpu_manager = GPUMemoryManager(config.gpu_memory_fraction)
        self._stats = {
            'jobs_processed': 0,
            'total_processing_time': 0.0,
            'average_batch_size': 0.0,
            'gpu_utilization': 0.0
        }
        self._lock = asyncio.Lock()
        
    async def add_job(self, job: BatchJob) -> None:
        try:
            await asyncio.wait_for(
                self.job_queue.put(job), 
                timeout=1.0
            )
            logger.debug(f"Added job {job.job_id} to queue")
        except asyncio.TimeoutError:
            raise RuntimeError("Batch queue is full")
    
    async def get_result(self, job_id: str, timeout: float = 30.0) -> Optional[BatchJob]:
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with self._lock:
                if job_id in self.result_store:
                    return self.result_store.pop(job_id)
                
                if job_id in self.processing_jobs:
                    job = self.processing_jobs[job_id]
                    if job.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
                        self.processing_jobs.pop(job_id)
                        return job
            
            await asyncio.sleep(0.1)
        
        return None
    
    def create_optimal_batches(self, jobs: List[BatchJob]) -> List[List[BatchJob]]:
        if not jobs:
            return []
        
        # Sort by priority and creation time
        jobs.sort(key=lambda x: (-x.priority, x.created_at))
        
        batches = []
        current_batch = []
        
        for job in jobs:
            # Check if adding this job would exceed limits
            if (len(current_batch) >= self.config.max_batch_size or
                (current_batch and not self._can_batch_together(current_batch[0], job))):
                
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
            
            current_batch.append(job)
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _can_batch_together(self, job1: BatchJob, job2: BatchJob) -> bool:
        # Check if jobs can be batched together based on audio characteristics
        audio1_duration = len(job1.audio_data) / job1.metadata.get('sample_rate', 16000)
        audio2_duration = len(job2.audio_data) / job2.metadata.get('sample_rate', 16000)
        
        # Don't batch if duration difference is too large
        max_duration = max(audio1_duration, audio2_duration)
        min_duration = min(audio1_duration, audio2_duration)
        
        if max_duration / min_duration > 3.0:  # Max 3x duration difference
            return False
        
        return True
    
    async def collect_batch(self, timeout_ms: int = 100) -> List[BatchJob]:
        jobs = []
        deadline = time.time() + timeout_ms / 1000.0
        
        # Get at least one job
        try:
            job = await asyncio.wait_for(
                self.job_queue.get(), 
                timeout=timeout_ms / 1000.0
            )
            jobs.append(job)
        except asyncio.TimeoutError:
            return jobs
        
        # Collect additional jobs until timeout or batch is full
        while (len(jobs) < self.config.max_batch_size and 
               time.time() < deadline):
            
            try:
                job = await asyncio.wait_for(
                    self.job_queue.get(),
                    timeout=max(0.01, deadline - time.time())
                )
                jobs.append(job)
            except asyncio.TimeoutError:
                break
        
        return jobs
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            'queue_size': self.job_queue.qsize(),
            'processing_jobs': len(self.processing_jobs),
            'gpu_memory_available': self.gpu_manager.get_available_memory()
        }

class BatchProcessor:
    def __init__(self, 
                 asr_service,
                 config: BatchConfig = None,
                 num_workers: int = 2):
        
        self.asr_service = asr_service
        self.config = config or BatchConfig()
        self.num_workers = num_workers
        self.batcher = DynamicBatcher(self.config)
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self._running = False
        self._worker_tasks = []
        
    async def start(self):
        if self._running:
            return
        
        self._running = True
        logger.info(f"Starting batch processor with {self.num_workers} workers")
        
        # Start worker tasks
        for i in range(self.num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._worker_tasks.append(task)
    
    async def stop(self):
        self._running = False
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        self.executor.shutdown(wait=True)
        logger.info("Batch processor stopped")
    
    async def _worker_loop(self, worker_name: str):
        logger.info(f"Worker {worker_name} started")
        
        while self._running:
            try:
                # Collect a batch of jobs
                jobs = await self.batcher.collect_batch(self.config.batch_timeout_ms)
                
                if not jobs:
                    await asyncio.sleep(0.01)
                    continue
                
                # Optimize batch size based on GPU memory
                optimal_size = self.batcher.gpu_manager.optimize_batch_size(
                    len(jobs), self.config.max_batch_size
                )
                
                if optimal_size < len(jobs):
                    # Put excess jobs back in queue
                    excess_jobs = jobs[optimal_size:]
                    jobs = jobs[:optimal_size]
                    
                    for job in reversed(excess_jobs):  # LIFO for priority
                        await self.batcher.job_queue.put(job)
                
                # Process the batch
                await self._process_batch(jobs, worker_name)
                
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in worker {worker_name}: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_batch(self, jobs: List[BatchJob], worker_name: str):
        if not jobs:
            return
        
        batch_start_time = time.time()
        batch_id = f"batch-{int(batch_start_time)}-{len(jobs)}"
        
        logger.info(f"Worker {worker_name} processing {batch_id} with {len(jobs)} jobs")
        
        # Mark jobs as processing
        async with self.batcher._lock:
            for job in jobs:
                job.status = BatchStatus.PROCESSING
                self.batcher.processing_jobs[job.job_id] = job
        
        try:
            # Extract audio data for batch processing
            audio_chunks = [job.audio_data for job in jobs]
            
            # Process batch using ASR service
            results = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.asr_service.transcribe_batch,
                audio_chunks
            )
            
            # Update jobs with results
            processing_time = time.time() - batch_start_time
            
            for job, result in zip(jobs, results):
                job.result = result
                job.status = BatchStatus.COMPLETED if not result.get('error') else BatchStatus.FAILED
                job.error = result.get('error')
                
                # Move to result store
                async with self.batcher._lock:
                    self.batcher.result_store[job.job_id] = job
                    if job.job_id in self.batcher.processing_jobs:
                        del self.batcher.processing_jobs[job.job_id]
            
            # Update stats
            self.batcher._stats['jobs_processed'] += len(jobs)
            self.batcher._stats['total_processing_time'] += processing_time
            self.batcher._stats['average_batch_size'] = (
                (self.batcher._stats['average_batch_size'] * 
                 (self.batcher._stats['jobs_processed'] - len(jobs)) + len(jobs)) /
                self.batcher._stats['jobs_processed']
            )
            
            avg_rtf = sum(r.get('rtf', 0) for r in results) / len(results)
            logger.info(f"Batch {batch_id} completed: {processing_time:.2f}s, "
                       f"avg RTF: {avg_rtf:.3f}")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            
            # Mark all jobs as failed
            async with self.batcher._lock:
                for job in jobs:
                    job.status = BatchStatus.FAILED
                    job.error = str(e)
                    self.batcher.result_store[job.job_id] = job
                    if job.job_id in self.batcher.processing_jobs:
                        del self.batcher.processing_jobs[job.job_id]
    
    async def transcribe_async(self, 
                             job_id: str,
                             audio_data: np.ndarray,
                             metadata: Dict = None,
                             priority: int = 0,
                             timeout: float = 300.0) -> Optional[Dict]:
        
        job = BatchJob(
            job_id=job_id,
            audio_data=audio_data,
            metadata=metadata or {},
            priority=priority
        )
        
        try:
            # Add job to queue
            await self.batcher.add_job(job)
            
            # Wait for result
            result_job = await self.batcher.get_result(job_id, timeout)
            
            if result_job is None:
                logger.warning(f"Job {job_id} timed out")
                return None
            
            if result_job.status == BatchStatus.FAILED:
                logger.error(f"Job {job_id} failed: {result_job.error}")
                return None
            
            return result_job.result
            
        except Exception as e:
            logger.error(f"Error processing job {job_id}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        return self.batcher.get_stats()