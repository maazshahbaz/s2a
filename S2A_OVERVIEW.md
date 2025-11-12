## S2A Service Overview

This document is a concise reference for the S2A (Speech-to-Actions) microservice, covering its purpose, architecture, runtime flow, key endpoints, and how to run/configure it.

### What it does
- High-performance speech-to-text microservice using NVIDIA NeMo Parakeet.
- Async job processing with Redis-backed chunk queue and GPU-optimized batching.
- Prisma/Postgres job tracking and OpenAI-style Bearer auth with rate limiting.
- Optional “Intelligence” pipeline using a vLLM-compatible LLM to extract entities, actions, and metrics.

### Tech stack
- API: FastAPI
- ML: PyTorch, NVIDIA NeMo (Parakeet)
- Queue/Cache: Redis (asyncio client)
- DB/ORM: Postgres via Prisma Python (required)
- Logging/Monitoring: Loguru, Prometheus metrics (exposed)

### Process flow
1) Client uploads audio to POST /v1/transcribe with a callback URL.
2) File saved under `uploads/YYYY-MM-DD/<job_id>.<ext>`, job persisted via Prisma.
3) Two parallel paths start:
   - ASR: Chunk metadata enqueued in Redis; workers batch across jobs for GPU efficiency, then stitch.
   - Diarization (mandatory): Full-audio diarization (`nvidia/diar_sortformer_4spk-v1`) runs asynchronously and stores speaker segments.
4) Alignment: When ASR stitching completes and diarization is available, the alignment service assigns ASR segments to speakers via timestamp overlap and renders speaker-attributed text.
5) A single final webhook is sent with speaker-attributed transcript, speaker segments, and metrics; status can be polled.

### Key components (paths)
- App entry: `main.py` (sets up DB, auth, ASR service, Redis batch processor, routers)
- ASR: `services/asr_service.py` (NeMo load, intelligent 24-min chunking, stitching)
- Batch processing: `services/batch_processor.py` (Redis queues, workers, result stitching)
- Diarization: `services/diarization_service.py` (NeMo diarization, Redis persistence of segments)
- Alignment: `services/alignment_service.py` (map ASR timestamps to diar segments and render speaker transcript)
- Auth: `db_services/auth.py` (Bearer tokens, permissions, rate limiting, Prisma-backed)
- Jobs/Results: `db_services/transcription.py` (CRUD, file storage helpers)
- Intelligence: `intelligence/intelligence_service.py` (+ `enhanced_extractor.py`) optional vLLM pipeline
- API routers:
  - `api/routers/transcribe.py`
  - `api/routers/stats.py`
  - `api/routers/intelligence.py`

### API endpoints (prefix defaults to /v1)
- Health/Stats
  - GET `/v1/statistics/health` (public)
  - GET `/v1/statistics/stats` (auth: `stats`)
- Transcription
  - POST `/v1/transcribe` (multipart; auth: `transcribe`) → `{ job_id, status: accepted }` (always includes diarization in final output)
  - GET `/v1/transcribe/status/{job_id}` (auth: `status`) → state/result
  - DELETE `/v1/transcribe/jobs/{job_id}` (auth: `transcribe`) → best-effort cancel
- Intelligence (if enabled)
  - POST `/v1/intelligence/extract` (async)
  - GET `/v1/intelligence/job/{job_id}/status`
  - GET `/v1/intelligence/job/{job_id}/result`
  - POST `/v1/intelligence/extract/sync`
  - GET `/v1/intelligence/metrics`, `/v1/intelligence/health`, `/v1/intelligence/modes`

### Configuration (env)
- ASR: `S2A_*` (e.g., `S2A_MODEL_NAME`, `S2A_BATCH_SIZE`, `S2A_MAX_CHUNK_DURATION`, `S2A_OVERLAP_DURATION`)
- Redis: `S2A_REDIS_*` (host, port, db, batch sizes, workers)
- Diarization: `S2A_DIAR_ENABLED=true` (default mandatory), `S2A_DIAR_MODEL=nvidia/diar_sortformer_4spk-v1`, `S2A_DIAR_MAX_SPEAKERS=4`
- Intelligence: `S2A_INTEL_*` (enabled=true by default, `vllm_base_url`, model, timeouts)
- Auth: `API_KEY_SECRET` (required for production), Prisma `DATABASE_URL`
- API: `API_VERSION` (default `/v1`)

See `config.py` for defaults and descriptions.

### Running locally
1) Python: `pip install -r requirements.txt`
2) Ensure Redis is running (or use Docker Compose).
3) Ensure Postgres is reachable via `DATABASE_URL`; run Prisma migrations: `prisma migrate deploy && prisma py fetch`
4) Start app: `python main.py` (port 8001)
5) Health check: `GET /v1/statistics/health`

### Docker Compose
- `docker-compose up -d` builds the API and starts Redis.
- Exposes ports: API `8001`, metrics `9090`.
- Mounts HuggingFace cache to persist models.
- Runs `prisma migrate deploy && prisma py fetch` before starting.

### Authentication
- OpenAI-style Bearer tokens required for all endpoints except health.
- Keys are stored hashed (HMAC-SHA256 with `API_KEY_SECRET`) via Prisma.
- In-memory rate limiting; headers returned on authenticated responses.

### Notes & gotchas
- NeMo is required: `nemo_toolkit[asr]` must be installed and GPU recommended.
- Long audio is handled by 24-min chunks with intelligent stitching.
- Postgres/Prisma are required for auth and job persistence; the service will not start without a reachable database (`DATABASE_URL`).
- Webhooks are required for async `POST /transcribe` (`callback_url` must be valid HTTP/HTTPS).
- Diarization is mandatory in the pipeline; final webhook waits for diarization+alignment. If diarization is unavailable, a single-speaker fallback is emitted.
- Callback URLs are validated: localhost/private network URLs are rejected for security.

### Quick links (source)
- Entry and app wiring: `main.py`
- Config: `config.py`
- Redis queue manager and workers: `services/*`
- DB services: `db_services/*`
- API: `api/routers/*`, `api/schemas/*`
- Intelligence: `intelligence/*`


