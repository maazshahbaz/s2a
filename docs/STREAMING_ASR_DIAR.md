# Streaming ASR / Diar Core

This branch contains only the streaming ASR / diarization core.

It intentionally includes:
- Triton Python backend models for streaming ASR and streaming diarization
- Thin async Python clients for calling those Triton models
- Config entries for `streaming_asr` and `streaming_diar`
- A Docker Compose stack for the two streaming Triton services

Does not include:
- WebSocket streaming API
- AudioSocket / Asterisk server integration
- FastAPI startup wiring in `main.py`
- Auth, routing, or staging API glue

## Files:

- `intelligent_pipeline/streaming_asr_client.py`
- `intelligent_pipeline/streaming_diar_client.py`
- `intelligent_pipeline/config.json`
- `docker-compose.streaming.yml`
- `triton-service/streaming_asr_triton/...`
- `triton-service/streaming_diar_triton/...`

## What The Models Expect

Both streaming models take:
- `audio_data`: float32 waveform
- `sample_rate`: input sample rate
- `session_id`: stable ID per call/session
- `is_final`: whether this is the last chunk

Both models keep per-session state in memory, so one session must stay routed to the same model instance.

## Run The Stack

From the repo root:

```powershell
docker compose -f docker-compose.streaming.yml up -d --build
```

Ports:
- Streaming ASR Triton HTTP/gRPC/metrics: `3900/3901/3902`
- Streaming Diar Triton HTTP/gRPC/metrics: `4000/4001/4002`

## Health Checks

```powershell
curl http://localhost:3900/v2/models/streaming_asr/ready
curl http://localhost:4000/v2/models/streaming_diar/ready
```

## Notes For Backend Integration

- The clients read endpoints from `intelligent_pipeline/config.json`.
- The clients also support env overrides:
  - `S2A_STREAMING_ASR_URL`
  - `S2A_STREAMING_DIAR_URL`
  - `S2A_STREAMING_ASR_MODEL_NAME`
  - `S2A_STREAMING_DIAR_MODEL_NAME`
- The default config uses `localhost` for streaming endpoints. Container-to-container integrations should override that with service DNS or another explicit address.
- This branch is model-serving core only. Transport, session orchestration, and transcript delivery are expected to be handled by the backend integration layer.
