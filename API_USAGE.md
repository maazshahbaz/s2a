# S2A Speech-to-Actions API Usage Guide

## Quick Start

Your S2A microservice provides REST APIs for speech transcription with GPU acceleration.

### Service Endpoints

- **Base URL**: `http://localhost:8001`
- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics` (Prometheus format)
- **Stats**: `GET /v1/stats`

### API Endpoints

#### 1. Synchronous Transcription
**Endpoint**: `POST /v1/transcribe`

```bash
curl -X POST "http://localhost:8001/v1/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@your_audio.wav" \
  -F "enhance_audio=true" \
  -F "remove_silence=false" \
  -F "priority=1"
```

**Response**:
```json
{
  "job_id": "uuid-here",
  "status": "completed",
  "text": "Your transcribed text here",
  "duration": 30.5,
  "rtf": 0.08,
  "processing_time": 2.4,
  "chunks": 2,
  "confidence": 0.95,
  "audio_quality": {
    "snr": 25.3,
    "sample_rate": 16000
  }
}
```

#### 2. Asynchronous Transcription
**Submit Job**: `POST /v1/transcribe/async`

```bash
# Submit long audio for processing
curl -X POST "http://localhost:8001/v1/transcribe/async" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@long_audio.mp3" \
  -F "priority=5" \
  -F "callback_url=https://your-app.com/webhook"
```

**Check Status**: `GET /v1/status/{job_id}`

```bash
curl "http://localhost:8001/v1/status/your-job-id-here"
```

### Supported Audio Formats

- **WAV** (recommended)
- **MP3** 
- **FLAC**
- **M4A**
- **OGG**

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_file` | file | required | Audio file to transcribe |
| `enhance_audio` | boolean | true | Apply noise reduction and filtering |
| `remove_silence` | boolean | false | Remove silence segments |
| `priority` | integer | 0 | Processing priority (0-10, higher = faster) |
| `callback_url` | string | null | Webhook URL for async results |

### Performance & Limits

- **Max file size**: 500MB
- **Max duration**: 5+ hours  
- **Batch processing**: Dynamic (1-8 jobs per batch)
- **Queue capacity**: 100 concurrent jobs
- **RTF**: < 0.1 (10x faster than real-time)
- **Supported sample rates**: 8kHz - 48kHz (auto-converted to 16kHz)

### Error Handling

**Common HTTP Status Codes**:
- `200` - Success
- `400` - Bad request (invalid audio format/size)
- `429` - Queue full (retry later)
- `500` - Processing error
- `503` - Service unavailable

**Error Response**:
```json
{
  "error": "Queue is full",
  "code": "QUEUE_FULL",
  "details": "Current queue size: 100/100"
}
```

### Monitoring

#### Health Check
```bash
curl http://localhost:8001/health
```

Response:
```json
{
  "status": "healthy",
  "model_info": {
    "name": "nvidia/parakeet-tdt-0.6b-v2",
    "device": "cuda"
  },
  "gpu_available": true,
  "batch_processor_stats": {
    "queue_size": 5,
    "processing_jobs": 2,
    "jobs_processed": 1247
  },
  "uptime": 3600.5
}
```

#### Performance Stats
```bash
curl http://localhost:8001/v1/stats
```

### Integration Examples

#### Python Client
```python
import requests
import json

def transcribe_audio(file_path, async_mode=False):
    url = "http://localhost:8001/v1/transcribe"
    if async_mode:
        url += "/async"
    
    files = {"audio_file": open(file_path, "rb")}
    data = {"enhance_audio": True, "priority": 1}
    
    response = requests.post(url, files=files, data=data)
    return response.json()

# Sync transcription
result = transcribe_audio("audio.wav")
print(result["text"])

# Async transcription
job = transcribe_audio("long_audio.mp3", async_mode=True)
job_id = job["job_id"]

# Check status
status_url = f"http://localhost:8001/v1/status/{job_id}"
result = requests.get(status_url).json()
```

#### JavaScript/Node.js
```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function transcribeAudio(filePath) {
  const form = new FormData();
  form.append('audio_file', fs.createReadStream(filePath));
  form.append('enhance_audio', 'true');
  
  try {
    const response = await axios.post(
      'http://localhost:8001/v1/transcribe',
      form,
      { headers: form.getHeaders() }
    );
    return response.data;
  } catch (error) {
    console.error('Transcription failed:', error.response.data);
  }
}

// Usage
transcribeAudio('./audio.wav')
  .then(result => console.log(result.text));
```

#### cURL with Webhook
```bash
# Submit with webhook callback
curl -X POST "http://localhost:8001/v1/transcribe/async" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@presentation.mp3" \
  -F "callback_url=https://myapp.com/transcription-complete" \
  -F "priority=3"

# Your webhook will receive:
# POST https://myapp.com/transcription-complete
# {
#   "job_id": "uuid",
#   "status": "completed", 
#   "text": "transcription...",
#   "processing_time": 45.2
# }
```

### Production Deployment

For production use:

1. **Enable authentication** (implement API keys)
2. **Set up rate limiting** (nginx/API gateway)
3. **Configure load balancing** (multiple instances)
4. **Monitor with Prometheus/Grafana** (port 9090)
5. **Set up log aggregation**
6. **Configure SSL/TLS termination**

### Troubleshooting

**Service won't start**:
```bash
# Check GPU access
nvidia-smi

# View logs
docker compose logs -f s2a-api

# Test health
curl http://localhost:8001/health
```

**Poor transcription quality**:
- Use WAV format for best results
- Enable `enhance_audio=true`
- Ensure good audio quality (SNR > 20dB)
- Check supported languages (currently English)

**High latency**:
- Check GPU utilization in logs
- Monitor queue size via `/v1/stats`
- Consider increasing batch size for throughput
- Use async mode for long audio files