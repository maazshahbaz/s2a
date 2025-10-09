# S2A: Speech-to-Actions Microservice

A high-performance ASR (Automatic Speech Recognition) microservice built with NVIDIA NeMo Parakeet models, designed for production deployment with H100 GPU optimization.

## Features

### Core ASR Capabilities
- **NVIDIA NeMo Parakeet Integration**: Utilizes `nvidia/parakeet-tdt-0.6b-v2` model for state-of-the-art transcription accuracy
- **Long Audio Support**: Handles audio files from 5 seconds up to 2 hours with intelligent chunking
- **24-minute Chunk Processing**: Optimal chunk size for maximum GPU utilization
- **Intelligent Audio Stitching**: Seamless reconstruction of long audio transcriptions

### Performance Optimization
- **Dynamic Batch Processing**: Adaptive batching based on GPU memory and audio characteristics
- **Real-time Factor (RTF) < 0.1**: Optimized for low-latency transcription
- **GPU Memory Management**: Intelligent memory allocation and cleanup
- **Mixed Precision Support**: FP16/BF16 for faster inference

### Audio Preprocessing
- **Format Support**: WAV, MP3, FLAC, M4A, OGG
- **Audio Enhancement**: Noise reduction, filtering, normalization
- **Voice Activity Detection**: WebRTC VAD for silence removal
- **Quality Validation**: SNR calculation and audio quality metrics

### API & Integration
- **FastAPI REST API**: Production-ready async endpoints
- **Bearer Token Authentication**: OpenAI-style API key authentication with rate limiting
- **Sync & Async Processing**: Sync API (≤2 minutes), Async API (≤2 hours)
- **Webhook Support**: Callback URLs for async results
- **Prometheus Metrics**: Comprehensive performance monitoring
- **Docker Deployment**: GPU-optimized containerization

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA 11.8+
- Docker & docker-compose
- Python 3.10+ (for local development)
- 32GB+ system RAM (recommended)
- 100GB+ storage for models and logs

### Docker Deployment (Recommended)

1. **Clone and setup**:
```bash
git clone <repository>
cd s2a
```

2. **Deploy with docker-compose**:
```bash
docker-compose up -d
```

3. **Set up authentication** (create your first API key):
```bash
python key_manager.py create --name "my-project-key" --type project
```

4. **Verify deployment**:
```bash
curl https://bytepulseai.com/v1/statistics/health
```

### Local Development

1. **Install dependencies**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Run the service**:
```bash
python main.py
```

## Docker Specifications

**Last Updated**: January 2025

### Container Details

**Base Image**: `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel`

**Key Features**:
- CUDA 12.1 with cuDNN 8 support
- PyTorch 2.3.1 pre-installed
- Optimized for NVIDIA H100/A100 GPUs
- Multi-stage builds for smaller production images

### Resource Requirements

| Component | Minimum | Recommended | Production |
|-----------|---------|-------------|-----------|
| **GPU Memory** | 8GB VRAM | 24GB VRAM | 40GB+ VRAM |
| **System RAM** | 16GB | 32GB | 64GB+ |
| **Storage** | 50GB | 100GB | 500GB+ |
| **CPU Cores** | 8 cores | 16 cores | 32+ cores |

### Port Configuration

| Port | Service | Protocol | Description |
|------|---------|----------|--------------|
| `8001` | API Server | HTTP | Main FastAPI application |
| `9090` | Metrics | HTTP | Prometheus metrics endpoint |

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|----------|
| `./logs` | `/app/logs` | Application logs |
| `/tmp/s2a` | `/tmp/s2a` | Temporary audio processing files |

### Environment Variables

See [Configuration](#configuration) section for complete list.

### GPU Configuration

The service requires NVIDIA Container Toolkit:

```yaml
# docker-compose.yml GPU configuration
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Health Checks

- **Interval**: 30 seconds
- **Timeout**: 10 seconds  
- **Start Period**: 60 seconds
- **Retries**: 3

## Authentication

### API Key Management

The S2A service uses OpenAI-style Bearer token authentication with comprehensive API key management.

#### Creating API Keys

1. **Using the Key Manager CLI**:
```bash
# Create a project key
python key_manager.py create --name "production-api" --type project

# Create with custom rate limits
python key_manager.py create \
  --name "high-volume-client" \
  --type project \
  --rpm 120 \
  --rph 2000 \
  --rpd 20000
```

2. **Key Types Available**:
   - `project`: `bp-proj-*` - Standard project keys
   - `user`: `bp-*` - User-specific keys  
   - `service`: `bp-svc-*` - Service-to-service keys

#### Managing API Keys

```bash
# List all keys
python key_manager.py list

# Show key details
python key_manager.py show <key_id>

# View usage statistics  
python key_manager.py stats

# Test API endpoints
python key_manager.py test
```

### Authentication Configuration

Set the API key secret for production:

```bash
# Required: Set a strong secret for key signing
export API_KEY_SECRET="your-secure-secret-min-32-chars"

# Optional: Custom key storage location
export API_KEYS_FILE="./api_keys.json"
```

### Rate Limiting

Each API key has configurable rate limits:

- **Per Minute**: Default 60 requests
- **Per Hour**: Default 1,000 requests  
- **Per Day**: Default 10,000 requests

Rate limit headers are included in all responses:

```http
X-RateLimit-Limit-Minute: 60
X-RateLimit-Remaining-Minute: 59
X-RateLimit-Reset-Minute: 1640995200
```

### Security Features

- **HMAC-SHA256 Key Hashing**: Keys are never stored in plaintext
- **Atomic File Operations**: Race-condition safe key storage
- **Permission-Based Access**: Granular endpoint permissions
- **Usage Tracking**: Comprehensive audit logs
- **Automatic Revocation**: Instant key deactivation

## API Usage

### Authentication Required

All API endpoints (except `/v1/statistics/health`) require Bearer token authentication:

```bash
curl -H "Authorization: Bearer bp-proj-YOUR_API_KEY" \
     https://bytepulseai.com/v1/transcribe
```
### Asynchronous Transcription
**Duration Limit**: Minimum 1 sec and Maximum 5 hours
**Response**: Job ID with webhook callback when complete
```bash
# Submit job
curl -X POST "https://bytepulseai.com/v1/transcribe" \
  -H "Authorization: Bearer bp-proj-YOUR_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@long_audio.mp3" \
  -F "callback_url=https://your-app.com/webhook" \
  -F "priority=1"

# Check status
curl "https://bytepulseai.com/v1/transcribe/status/{job_id}" \
  -H "Authorization: Bearer bp-proj-YOUR_API_KEY"
```

### Health Check
```bash
# Health check doesn't require authentication
curl "https://bytepulseai.com/v1/statistics/health"
```

## CLI Usage

The CLI provides a convenient interface for local transcription:

### Basic Transcription
```bash
python cli.py transcribe audio.wav
```

### Batch Processing
```bash
python cli.py batch-transcribe *.wav --output-dir results/
```

### API Client
```bash
python cli.py api-transcribe audio.wav --url https://bytepulseai.com --api-key bp-proj-YOUR_API_KEY
```

### Server Status
```bash
python cli.py status --url https://bytepulseai.com --api-key bp-proj-YOUR_API_KEY
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `S2A_MODEL_NAME` | `nvidia/parakeet-tdt-0.6b-v2` | HuggingFace model name |
| `S2A_DEVICE` | `cuda` | Processing device |
| `S2A_BATCH_SIZE` | `8` | Batch size for processing |
| `S2A_MAX_CHUNK_DURATION` | `1440` | Max chunk duration (seconds) |
| `S2A_MIN_AUDIO_DURATION` | `5.0` | Min audio duration to process |
| `S2A_GPU_MEMORY_FRACTION` | `0.8` | GPU memory utilization |
| `S2A_NUM_WORKERS` | `2` | Number of worker processes |
| `S2A_LOG_LEVEL` | `INFO` | Logging level |
| `S2A_ENABLE_MIXED_PRECISION` | `true` | Enable mixed precision inference |
| `API_KEY_SECRET` | `change-me` | Secret for API key HMAC signing |
| `API_KEYS_FILE` | `./api_keys.json` | Path to API keys storage file |

### .env File
```bash
# Copy example configuration
cp .env.example .env
# Edit configuration
nano .env
```

## Architecture

### Core Components

1. **ASR Service** (`asr_service.py`)
   - NeMo model integration
   - Batch processing coordination
   - Performance monitoring

2. **Audio Processing** (`audio_utils.py`)
   - Format conversion and normalization
   - Enhancement algorithms
   - Quality validation

3. **Batch Processor** (`batch_processor.py`)
   - Dynamic GPU memory management
   - Intelligent job queuing
   - Concurrent processing

4. **Chunking Manager** (`chunking_utils.py`)
   - Speech-aware segmentation
   - Overlap management
   - Intelligent stitching

5. **Performance Monitor** (`performance_monitor.py`)
   - Real-time metrics collection
   - Prometheus integration
   - Alert system

### Data Flow

```
Audio Input → Preprocessing → Chunking → Batch Processing → Transcription → Stitching → Output
     ↓              ↓            ↓            ↓              ↓            ↓         ↓
  Validation   Enhancement   VAD Split    GPU Queue     NeMo Model   Overlap    Result
     ↓              ↓            ↓            ↓              ↓        Removal       ↓
Format Check   Noise Filter  Boundaries   Memory Mgmt   Inference   Text Join   Webhook
```

## Performance Benchmarks

### Hardware Configuration
- **GPU**: NVIDIA H100 80GB
- **CPU**: 64-core AMD EPYC
- **RAM**: 512GB DDR4
- **Storage**: NVMe SSD

### Performance Metrics
- **RTF**: 0.05-0.15 (depending on audio quality)
- **Throughput**: 20-30x real-time
- **GPU Utilization**: 80-95%
- **Memory Efficiency**: <8GB VRAM for batch size 8
- **Latency**: <2s for 60s audio

### Scalability
- **Max Concurrent Jobs**: 100+
- **Queue Throughput**: 1000+ jobs/hour
- **Long Audio Support**: Tested up to 2 hours (async API limit)
- **Batch Efficiency**: 90%+ GPU utilization

## Monitoring

### Prometheus Metrics
Access metrics at `http://localhost:9090/metrics`:

- `transcription_requests_total`: Total requests by status
- `transcription_processing_seconds`: Processing time histogram
- `transcription_rtf`: Real-time factor distribution
- `gpu_utilization_percent`: GPU usage
- `transcription_queue_size`: Current queue size

### Grafana Dashboard
- Real-time performance monitoring
- GPU utilization tracking
- Queue depth analysis
- Error rate monitoring

Access at `http://localhost:3000` (admin/admin)

### Health Endpoints
- `/v1/statistics/health`: Service health check
- `/v1/statistics/stats`: Performance statistics
- `/metrics`: Prometheus metrics

## Error Handling

### Common Issues

1. **GPU Memory Errors**
   - Reduce batch size
   - Increase `gpu_memory_fraction`
   - Monitor memory usage

2. **Long Processing Times**
   - Check RTF metrics
   - Verify GPU utilization
   - Consider model quantization

3. **Audio Format Issues**
   - Ensure ffmpeg is installed
   - Check supported formats
   - Verify file integrity

### Troubleshooting

```bash
# Check GPU status
nvidia-smi

# View service logs
docker-compose logs -f s2a-api

# Test with sample audio
curl -X POST "https://bytepulseai.com/v1/transcribe" \
  -H "Authorization: Bearer bp-proj-YOUR_API_KEY" \
  -F "audio_file=@test_audio.wav" \
  -F "callback_url=https://your-app.com/webhook"

# Monitor performance
curl "https://bytepulseai.com/v1/statistics/stats" \
  -H "Authorization: Bearer bp-proj-YOUR_API_KEY" | jq
```

## Development

### Setup Development Environment
```bash
# Install development dependencies
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
flake8 .
mypy .
```

### Adding New Features

1. **Audio Preprocessing**: Extend `AudioProcessor` class
2. **Model Integration**: Add new models in `ASRService`
3. **API Endpoints**: Add routes in `main.py`
4. **Monitoring**: Extend `PerformanceMonitor`

### Testing
```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Performance tests
pytest tests/performance/ --benchmark
```

## Deployment

### Production Deployment

1. **Hardware Requirements**:
   - NVIDIA H100/A100 GPU
   - 32GB+ system RAM
   - 100GB+ storage
   - High-speed internet for model downloads

2. **Security Considerations**:
   - ✅ **API Authentication**: Bearer token system implemented
   - ✅ **Rate Limiting**: Per-minute/hour/day limits enforced
   - ✅ **Input Validation**: Comprehensive audio file validation
   - ✅ **Secure Key Storage**: HMAC-SHA256 hashed keys
   - ✅ **Secret Management**: Use environment variables for API_KEY_SECRET
   - **Network Isolation**: Configure firewall rules
   - **HTTPS/TLS**: Requires reverse proxy setup (not included)

3. **Scaling**:
   - Multiple GPU support
   - Kubernetes deployment
   - Load balancing
   - Auto-scaling

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: s2a-asr
spec:
  replicas: 2
  selector:
    matchLabels:
      app: s2a-asr
  template:
    metadata:
      labels:
        app: s2a-asr
    spec:
      containers:
      - name: s2a-asr
        image: s2a-asr:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: See `/docs` folder for detailed guides
- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Commercial Support**: Contact [your-email@company.com]

## Changelog

### v1.0.0 (Current)
- Initial release with NeMo Parakeet integration
- Batch processing and GPU optimization
- Docker deployment support
- Comprehensive monitoring

### Upcoming Features
- Multi-language support
- Real-time streaming
- Speaker diarization
- Advanced noise cancellation
- Cloud provider integrations