# S2A - Speech-to-Actions

A high-performance ASR microservice that transcribes audio/video files and enriches them with an intelligent processing pipeline — speaker diarization, LLM-powered analysis, sentiment detection, fraud detection, CSR scoring, task generation, and text-to-speech.

## Tech Stack

- **Backend:** FastAPI, PyTorch 2.3.1 (CUDA 12.1), NVIDIA NeMo Parakeet TDT 0.6B-v2
- **LLM:** Qwen/Qwen2.5-7B-Instruct via vLLM / Triton Inference Server
- **TTS:** CosyVoice2
- **Database:** PostgreSQL 15 (Prisma ORM)
- **Queue/Cache:** Redis
- **Frontend:** Next.js, NextAuth (Azure AD)
- **SDKs:** Python, JavaScript/TypeScript

## Project Structure

```
├── api/routers/          # FastAPI route handlers
├── intelligent_pipeline/ # Core processing pipeline
├── db_services/          # Database & auth services
├── security/             # HMAC authentication
├── api_portal/           # Next.js frontend
├── sdk/                  # Python & JS client SDKs
├── triton-service/       # Triton model servers (ASR, diarization, LLM, TTS)
├── prisma/               # Database schema & migrations
└── scripts/              # Test & utility scripts
```

## Quick Start

```bash
# Configure environment
cp .env.example .env
# Set DATABASE_URL and API_KEY_SECRET in .env

# Run with Docker
docker compose up -d --build

# Health check
curl http://localhost:8001/v1/statistics/health
```

### Other modes

```bash
# Development (hot reload, debug logging)
docker compose -f docker-compose.dev.yml up -d --build

# Staging (no database, no auth)
docker compose -f docker-compose.staging.yml up -d --build

# With Triton inference servers
docker compose -f docker-compose.triton.yml up -d --build
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/transcribe` | Submit audio for async transcription |
| `GET` | `/v1/transcribe/status/{job_id}` | Check job status |
| `DELETE` | `/v1/transcribe/jobs/{job_id}` | Cancel a job |
| `GET` | `/v1/statistics/health` | Health check |
| `GET` | `/v1/statistics/stats` | Service statistics |
| `POST` | `/v1/api-keys/` | Create API key |
| `GET` | `/v1/api-keys/` | List API keys |
| `DELETE` | `/v1/api-keys/{key_id}` | Revoke API key |
| `POST` | `/v1/users` | Create user |

## Pipeline

Audio is processed through these stages:

1. **Chunking** — split audio into segments for parallel processing
2. **ASR** — transcribe with NVIDIA Parakeet TDT
3. **Diarization** — identify and label speakers globally
4. **Transcript Merging** — align chunks with speaker labels
5. **LLM Analysis** — extract summary, intent, sentiment, action items
6. **Scoring** — CSR performance and fraud detection
7. **Task Generation** — extract follow-up tasks
8. **TTS** — generate speech from text (CosyVoice2)
9. **Webhook Delivery** — send results to callback URL

## Requirements

- NVIDIA GPU with CUDA 12.1+
- Docker & Docker Compose
- PostgreSQL 15 (provided via docker-compose)
