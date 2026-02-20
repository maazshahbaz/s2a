# S2A Staging Environment

Staging API running on **port 8002**, sharing GPU models with production. No database, no authentication.

## Architecture

```
Production (port 8001)          Staging (port 8002)
  - Auth required                 - No auth
  - PostgreSQL                    - No database
  - Full DB writes                - Skip all DB ops
         \                        /
          \________ Shared _______/
          ASR Triton       :3501
          Diarization      :3601
          Mistral LLM      :3701
          Qwen Task Gen    :3801
```

## Setup

```bash
# Clone repo into staging directory
git clone git@github.com:99Technologies-ai/s2a.git s2a-staging
cd s2a-staging
git checkout feature/staging

# Build and launch
docker compose -f docker-compose.staging.yml up -d --build
```

## Usage

### Health check
```bash
curl http://localhost:8002/
```

### Submit transcription (no auth needed)
```bash
curl -X POST "http://localhost:8002/v1/transcribe" \
  -F "audio_file=@recording.wav;type=audio/wav" \
  -F "callback_url=http://localhost:8001/v1/webhook" \
  -F 'call_metadata={"src":"+1234567890","calldate":"2026-02-18 12:00:00","agentExtension":"1003-rti.talkloop.ai","direction":"INBOUND"}'
```

Note: Use `callback_url=http://localhost:8001/v1/webhook` (port 8001) when sending webhooks to the staging container itself, since port 8001 is the internal port.

### Check webhook logs
```bash
curl http://localhost:8002/v1/webhook/logs | python3 -m json.tool
```

### View container logs
```bash
docker logs s2a-api-staging --tail 50 -f
```

## What works in staging

- Full audio pipeline (chunking, ASR, diarization, merging)
- LLM intelligence (speaker correction, analysis, scoring, fraud detection)
- Task generation (Qwen)
- Follow-up email generation
- Webhook delivery to callback_url

## What is skipped in staging

- Authentication (mock APIKey with all permissions)
- Database connection (no PostgreSQL needed)
- Usage tracking
- Job creation/status updates in DB
- Result persistence in DB

## Updating staging

```bash
cd s2a-staging
git pull
docker compose -f docker-compose.staging.yml up -d --build
```

## Testing a feature branch on staging

When you want to test a feature branch (e.g. `feature/exy` with updated prompts), switch the staging repo to that branch and rebuild.

### Prerequisites

Your feature branch must include the staging code (auth bypass, DB skip). Either:

1. **Merge `feature/staging` into your feature branch** (recommended before `feature/staging` is merged to main):
   ```bash
   git checkout feature/exy
   git merge feature/staging
   ```

2. **Or branch from `main`** after `feature/staging` has been merged — all future branches will inherit staging support automatically.

### Deploy your feature branch to staging

```bash
cd ~/Desktop/data/back2/bytepulse-ai/s2a-staging

# Switch to your feature branch
git fetch origin
git checkout feature/exy
git pull origin feature/exy

# Rebuild staging container with the new code
docker compose -f docker-compose.staging.yml up -d --build

# Test
curl -X POST "http://localhost:8002/v1/transcribe" \
  -F "audio_file=@test.wav;type=audio/wav" \
  -F "callback_url=http://localhost:8001/v1/webhook" \
  -F 'call_metadata={"src":"+1234567890","calldate":"2026-02-18 12:00:00","agentExtension":"1003-rti.talkloop.ai","direction":"INBOUND"}'

# Check results
docker logs s2a-api-staging --tail 50
```

### Switch back when done

```bash
git checkout feature/staging
docker compose -f docker-compose.staging.yml up -d --build
```

## Tearing down

```bash
docker compose -f docker-compose.staging.yml down
```


## Configuration

Staging mode is controlled by `S2A_STAGING_MODE=true` in `docker-compose.staging.yml`. All other env vars (`DATABASE_URL`, `API_KEY_SECRET`, `HMAC_SECRET`) are dummy values required by pydantic-settings but never used.
