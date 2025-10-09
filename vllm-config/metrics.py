#!/usr/bin/env python3
"""
Metrics collection for vLLM service
"""

import time
import logging
from prometheus_client import start_http_server, Gauge, Counter
import httpx
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Prometheus metrics
vllm_requests_total = Counter('vllm_requests_total', 'Total number of requests')
vllm_gpu_memory_usage = Gauge('vllm_gpu_memory_usage_bytes', 'GPU memory usage in bytes')
vllm_response_time = Gauge('vllm_response_time_seconds', 'Average response time')
vllm_queue_size = Gauge('vllm_queue_size', 'Current queue size')

class VLLMMetricsCollector:
    """Collect metrics from vLLM service"""

    def __init__(self):
        self.client = httpx.Client(timeout=5.0)

    def collect_metrics(self):
        """Collect and update metrics"""
        try:
            # Get vLLM metrics if available
            response = self.client.get("http://localhost:8000/metrics")
            if response.status_code == 200:
                # Parse vLLM internal metrics
                self._parse_vllm_metrics(response.text)

        except Exception as e:
            logging.warning(f"Failed to collect vLLM metrics: {e}")

        try:
            # Get GPU metrics using nvidia-ml-py if available
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vllm_gpu_memory_usage.set(meminfo.used)

        except ImportError:
            logging.warning("pynvml not available, skipping GPU metrics")
        except Exception as e:
            logging.warning(f"Failed to collect GPU metrics: {e}")

    def _parse_vllm_metrics(self, metrics_text):
        """Parse vLLM Prometheus metrics"""
        # Basic parsing of Prometheus format
        lines = metrics_text.split('\n')
        for line in lines:
            if line.startswith('vllm_'):
                try:
                    # Simple metric parsing
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        metric_name = parts[0]
                        metric_value = float(parts[1])

                        if 'request' in metric_name:
                            vllm_requests_total._value._value = metric_value
                        elif 'queue' in metric_name:
                            vllm_queue_size.set(metric_value)

                except (ValueError, IndexError):
                    continue

    def run(self):
        """Run metrics collection loop"""
        logging.info("Starting vLLM metrics collector")

        # Start Prometheus metrics server
        start_http_server(8001)  # Different port from main vLLM service

        while True:
            try:
                self.collect_metrics()
                time.sleep(10)  # Collect metrics every 10 seconds
            except KeyboardInterrupt:
                logging.info("Metrics collector stopped")
                break
            except Exception as e:
                logging.error(f"Metrics collection error: {e}")
                time.sleep(30)

if __name__ == "__main__":
    collector = VLLMMetricsCollector()
    collector.run()