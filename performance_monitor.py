import time
import psutil
import torch
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import threading
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import json

@dataclass
class PerformanceMetrics:
    timestamp: float
    rtf: float
    gpu_utilization: float
    gpu_memory_used: float
    gpu_memory_total: float
    cpu_utilization: float
    memory_used: float
    memory_total: float
    queue_size: int
    active_jobs: int
    processing_time: float
    audio_duration: float
    batch_size: int
    model_name: str

class PrometheusMetrics:
    def __init__(self):
        # Counters
        self.transcription_requests = Counter(
            'transcription_requests_total', 
            'Total transcription requests', 
            ['status', 'model']
        )
        
        self.transcription_errors = Counter(
            'transcription_errors_total', 
            'Total transcription errors', 
            ['error_type', 'model']
        )
        
        # Histograms
        self.processing_time = Histogram(
            'transcription_processing_seconds',
            'Time spent processing transcriptions',
            ['model']
        )
        
        self.rtf_metric = Histogram(
            'transcription_rtf',
            'Real-time factor of transcriptions',
            ['model']
        )
        
        self.audio_duration = Histogram(
            'transcription_audio_duration_seconds',
            'Duration of processed audio',
            ['model']
        )
        
        self.batch_size_metric = Histogram(
            'transcription_batch_size',
            'Batch size for transcriptions',
            ['model']
        )
        
        # Gauges
        self.queue_size = Gauge(
            'transcription_queue_size',
            'Current queue size'
        )
        
        self.active_jobs = Gauge(
            'transcription_active_jobs',
            'Current number of active jobs'
        )
        
        self.gpu_utilization = Gauge(
            'gpu_utilization_percent',
            'GPU utilization percentage'
        )
        
        self.gpu_memory_used = Gauge(
            'gpu_memory_used_bytes',
            'GPU memory used in bytes'
        )
        
        self.cpu_utilization = Gauge(
            'cpu_utilization_percent',
            'CPU utilization percentage'
        )
        
        self.memory_used = Gauge(
            'memory_used_bytes',
            'System memory used in bytes'
        )

class PerformanceMonitor:
    def __init__(self, 
                 metrics_interval: float = 10.0,
                 history_size: int = 1000,
                 enable_prometheus: bool = True,
                 prometheus_port: int = 9090):
        
        self.metrics_interval = metrics_interval
        self.history_size = history_size
        self.enable_prometheus = enable_prometheus
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.current_metrics = {}
        self._lock = threading.Lock()
        
        # Prometheus metrics
        if enable_prometheus:
            self.prometheus_metrics = PrometheusMetrics()
            try:
                start_http_server(prometheus_port)
                logger.info(f"Prometheus metrics server started on port {prometheus_port}")
            except Exception as e:
                logger.warning(f"Failed to start Prometheus server: {e}")
        
        # Monitoring thread
        self._monitoring = False
        self._monitor_thread = None
        
        # Alert thresholds
        self.rtf_warning_threshold = 0.5
        self.rtf_error_threshold = 1.0
        self.memory_warning_threshold = 0.8
        self.memory_error_threshold = 0.9
        self.gpu_memory_warning_threshold = 0.8
        self.gpu_memory_error_threshold = 0.9
        
    def start_monitoring(self):
        if self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        while self._monitoring:
            try:
                self._collect_system_metrics()
                time.sleep(self.metrics_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1.0)
    
    def _collect_system_metrics(self):
        with self._lock:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU metrics
            gpu_utilization = 0
            gpu_memory_used = 0
            gpu_memory_total = 0
            
            if torch.cuda.is_available():
                try:
                    # GPU utilization (requires nvidia-ml-py)
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_utilization = gpu_util.gpu
                    
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_used = memory_info.used
                    gpu_memory_total = memory_info.total
                    
                except ImportError:
                    # Fallback to PyTorch memory info
                    gpu_memory_used = torch.cuda.memory_allocated(0)
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                except Exception as e:
                    logger.debug(f"GPU metrics collection failed: {e}")
            
            # Update current metrics
            self.current_metrics.update({
                'timestamp': time.time(),
                'cpu_utilization': cpu_percent,
                'memory_used': memory.used,
                'memory_total': memory.total,
                'memory_percent': memory.percent,
                'gpu_utilization': gpu_utilization,
                'gpu_memory_used': gpu_memory_used,
                'gpu_memory_total': gpu_memory_total,
                'gpu_memory_percent': (gpu_memory_used / gpu_memory_total * 100) if gpu_memory_total > 0 else 0
            })
            
            # Update Prometheus metrics
            if self.enable_prometheus:
                self.prometheus_metrics.cpu_utilization.set(cpu_percent)
                self.prometheus_metrics.memory_used.set(memory.used)
                self.prometheus_metrics.gpu_utilization.set(gpu_utilization)
                self.prometheus_metrics.gpu_memory_used.set(gpu_memory_used)
            
            # Check for alerts
            self._check_alerts()
    
    def record_transcription(self, 
                           processing_time: float,
                           audio_duration: float,
                           batch_size: int,
                           model_name: str,
                           queue_size: int = 0,
                           active_jobs: int = 0,
                           status: str = "success",
                           error_type: str = None):
        
        rtf = processing_time / audio_duration if audio_duration > 0 else float('inf')
        
        # Create metrics object
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            rtf=rtf,
            gpu_utilization=self.current_metrics.get('gpu_utilization', 0),
            gpu_memory_used=self.current_metrics.get('gpu_memory_used', 0),
            gpu_memory_total=self.current_metrics.get('gpu_memory_total', 0),
            cpu_utilization=self.current_metrics.get('cpu_utilization', 0),
            memory_used=self.current_metrics.get('memory_used', 0),
            memory_total=self.current_metrics.get('memory_total', 0),
            queue_size=queue_size,
            active_jobs=active_jobs,
            processing_time=processing_time,
            audio_duration=audio_duration,
            batch_size=batch_size,
            model_name=model_name
        )
        
        # Store in history
        with self._lock:
            self.metrics_history.append(metrics)
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self.prometheus_metrics.transcription_requests.labels(
                status=status, model=model_name
            ).inc()
            
            if error_type:
                self.prometheus_metrics.transcription_errors.labels(
                    error_type=error_type, model=model_name
                ).inc()
            
            self.prometheus_metrics.processing_time.labels(model=model_name).observe(processing_time)
            self.prometheus_metrics.rtf_metric.labels(model=model_name).observe(rtf)
            self.prometheus_metrics.audio_duration.labels(model=model_name).observe(audio_duration)
            self.prometheus_metrics.batch_size_metric.labels(model=model_name).observe(batch_size)
            self.prometheus_metrics.queue_size.set(queue_size)
            self.prometheus_metrics.active_jobs.set(active_jobs)
        
        # Log performance info
        logger.info(f"Transcription completed: duration={audio_duration:.2f}s, "
                   f"processing={processing_time:.2f}s, RTF={rtf:.3f}, "
                   f"batch_size={batch_size}")
        
        # Check RTF thresholds
        if rtf > self.rtf_error_threshold:
            logger.error(f"RTF {rtf:.3f} exceeds error threshold {self.rtf_error_threshold}")
        elif rtf > self.rtf_warning_threshold:
            logger.warning(f"RTF {rtf:.3f} exceeds warning threshold {self.rtf_warning_threshold}")
    
    def _check_alerts(self):
        # Memory alerts
        memory_percent = self.current_metrics.get('memory_percent', 0)
        if memory_percent > self.memory_error_threshold * 100:
            logger.error(f"System memory usage {memory_percent:.1f}% exceeds error threshold")
        elif memory_percent > self.memory_warning_threshold * 100:
            logger.warning(f"System memory usage {memory_percent:.1f}% exceeds warning threshold")
        
        # GPU memory alerts
        gpu_memory_percent = self.current_metrics.get('gpu_memory_percent', 0)
        if gpu_memory_percent > self.gpu_memory_error_threshold * 100:
            logger.error(f"GPU memory usage {gpu_memory_percent:.1f}% exceeds error threshold")
        elif gpu_memory_percent > self.gpu_memory_warning_threshold * 100:
            logger.warning(f"GPU memory usage {gpu_memory_percent:.1f}% exceeds warning threshold")
    
    def get_current_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self.current_metrics)
    
    def get_performance_summary(self, minutes: int = 60) -> Dict[str, Any]:
        cutoff_time = time.time() - (minutes * 60)
        
        with self._lock:
            recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]
        
        if not recent_metrics:
            return {"message": "No recent metrics available"}
        
        # Calculate statistics
        rtfs = [m.rtf for m in recent_metrics if m.rtf != float('inf')]
        processing_times = [m.processing_time for m in recent_metrics]
        audio_durations = [m.audio_duration for m in recent_metrics]
        batch_sizes = [m.batch_size for m in recent_metrics]
        
        return {
            "time_window_minutes": minutes,
            "total_transcriptions": len(recent_metrics),
            "average_rtf": sum(rtfs) / len(rtfs) if rtfs else 0,
            "min_rtf": min(rtfs) if rtfs else 0,
            "max_rtf": max(rtfs) if rtfs else 0,
            "average_processing_time": sum(processing_times) / len(processing_times),
            "total_audio_duration": sum(audio_durations),
            "average_batch_size": sum(batch_sizes) / len(batch_sizes),
            "throughput_hours_per_hour": sum(audio_durations) / 3600 / (minutes / 60) if minutes > 0 else 0,
            "current_system_stats": self.get_current_stats()
        }
    
    def export_metrics(self, format: str = "json") -> str:
        with self._lock:
            metrics_data = [
                {
                    "timestamp": m.timestamp,
                    "rtf": m.rtf,
                    "processing_time": m.processing_time,
                    "audio_duration": m.audio_duration,
                    "batch_size": m.batch_size,
                    "model_name": m.model_name,
                    "gpu_utilization": m.gpu_utilization,
                    "gpu_memory_percent": (m.gpu_memory_used / m.gpu_memory_total * 100) if m.gpu_memory_total > 0 else 0,
                    "cpu_utilization": m.cpu_utilization,
                    "memory_percent": (m.memory_used / m.memory_total * 100) if m.memory_total > 0 else 0,
                }
                for m in self.metrics_history
            ]
        
        if format.lower() == "json":
            return json.dumps(metrics_data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Global performance monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
        _performance_monitor.start_monitoring()
    return _performance_monitor