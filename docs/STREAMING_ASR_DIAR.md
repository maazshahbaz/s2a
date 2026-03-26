# Streaming ASR / Diar Core

This branch contains the streaming speech core only.

It is meant to provide:
- a Triton-served streaming ASR model
- a Triton-served streaming diarization model
- thin Python clients that call those models
- minimal config and compose wiring to run the two services

It does not include:
- WebSocket streaming APIs
- AudioSocket / Asterisk handling
- FastAPI startup wiring
- auth, routing, or staging API glue
- live transcript delivery or session orchestration in the API layer

## Purpose

The goal of this branch is to isolate the reusable streaming inference layer from the backend transport layer.

The intended ownership split is:
- this branch: model-serving core
- backend integration: session transport, request lifecycle, transcript delivery, retries, and external API behavior

That makes this branch easier to review, test, and hand off.

## Included Files

Core runtime:
- `intelligent_pipeline/streaming_asr_client.py`
- `intelligent_pipeline/streaming_diar_client.py`
- `intelligent_pipeline/config.json`
- `docker-compose.streaming.yml`

Streaming ASR Triton service:
- `triton-service/streaming_asr_triton/dockerfile`
- `triton-service/streaming_asr_triton/models/streaming_asr/config.pbtxt`
- `triton-service/streaming_asr_triton/models/streaming_asr/1/model.py`

Streaming diarization Triton service:
- `triton-service/streaming_diar_triton/dockerfile`
- `triton-service/streaming_diar_triton/models/streaming_diar/config.pbtxt`
- `triton-service/streaming_diar_triton/models/streaming_diar/1/model.py`

## High-Level Architecture

There are two independent Triton Python-backend models:

1. `streaming_asr`
- wraps the NeMo streaming ASR model
- accepts chunked audio plus a stable `session_id`
- keeps ASR cache state in memory per session
- returns cumulative transcription text for the current session

2. `streaming_diar`
- wraps the NeMo Sortformer streaming diarization model
- accepts chunked audio plus a stable `session_id`
- keeps a rolling audio window and speaker mapping per session
- returns speaker segments for the recent session state

The Python clients are intentionally thin:
- they load endpoint/model configuration
- they verify server/model readiness
- they send one inference request per chunk
- they decode JSON results returned by Triton

## Model Input / Output Contract

Both Triton models use the same input pattern:

Inputs:
- `audio_data`: `float32` waveform
- `sample_rate`: integer sample rate
- `session_id`: stable session identifier
- `is_final`: whether this is the last chunk of the session

Streaming ASR output:
- `transcription`: JSON string with fields such as:
  - `text`
  - `word_timestamps`
  - `session_id`
  - `is_final`
  - `step_num`

Streaming diarization output:
- `diarization_output`: JSON string with fields such as:
  - `segments`
  - `num_speakers`
  - `session_id`
  - `audio_duration`
  - `is_final`
  - `diar_ran`
  - `diar_window_seconds`

## Session Behavior

Both models are stateful.

That means:
- all chunks for one call must use the same `session_id`
- all chunks for one call must stay routed to the same model instance
- `is_final=true` should be sent on the last chunk so session state is released promptly

Because of that design:
- both Triton model configs use `count: 1`
- multi-instance scaling would require sticky routing or a different session-state strategy

## Preprocessing / Runtime Behavior

### Streaming ASR

The ASR model:
- loads `nvidia/nemotron-speech-streaming-en-0.6b`
- keeps encoder and pre-encode cache state per session
- resamples incoming audio to 16 kHz when needed
- runs greedy streaming decoding
- returns cumulative text for the session

Important behavior:
- the model does not return word timestamps in the current implementation
- stale ASR sessions are cleaned periodically
- pre-encode overlap is dropped during decode to reduce duplicate overlap effects

### Streaming Diarization

The diarization model:
- loads `nvidia/diar_streaming_sortformer_4spk-v2.1`
- resamples incoming audio to 16 kHz when needed
- keeps only a rolling audio window in memory
- reruns diarization periodically instead of on every chunk
- maintains a stable per-session speaker ID map

Important behavior:
- diarization only runs after enough context is available
- it operates on a bounded rolling window to avoid unbounded memory growth
- speaker IDs are stabilized per session, but speaker labels are still heuristic model output, not guaranteed identity truth

## Configuration

Default service locations are in:
- `intelligent_pipeline/config.json`

Current defaults:
- streaming ASR: `localhost:3901`
- streaming diarization: `localhost:4001`

The thin clients also support env overrides:
- `S2A_STREAMING_ASR_URL`
- `S2A_STREAMING_DIAR_URL`
- `S2A_STREAMING_ASR_MODEL_NAME`
- `S2A_STREAMING_DIAR_MODEL_NAME`

Use env overrides when:
- backend integration runs in containers
- service DNS should be used instead of `localhost`
- the model name changes

## Running The Stack

From the repo root:

```powershell
docker compose -f docker-compose.streaming.yml up -d --build
```

Ports:
- ASR Triton HTTP: `3900`
- ASR Triton gRPC: `3901`
- ASR Triton metrics: `3902`
- Diar Triton HTTP: `4000`
- Diar Triton gRPC: `4001`
- Diar Triton metrics: `4002`

## Health Checks

Model-specific readiness checks:

```powershell
curl http://localhost:3900/v2/models/streaming_asr/ready
curl http://localhost:4000/v2/models/streaming_diar/ready
```

This branch uses model-specific readiness instead of generic server readiness so “healthy” more closely matches actual model availability.

## Integration Notes

If this branch is used by a backend service, that backend should own:
- chunk scheduling
- session creation and cleanup policy
- retry/backoff around inference calls
- transport protocol
- transcript assembly and downstream business logic

Recommended integration assumptions:
- send audio as `float32` chunks
- keep chunk duration consistent
- preserve one stable `session_id` per call
- always send a final chunk with `is_final=true`

## Known Limitations

These are known tradeoffs in the current branch:

- ASR does not emit word timestamps
- diarization still uses temp-file I/O internally
- both Triton Docker images install NeMo from pip at build time
- model state is local to one Triton instance
- this branch is not a full end-to-end product path by itself

## Handoff Notes

If someone is reviewing this branch for KT, the recommended reading order is:

1. `triton-service/streaming_asr_triton/models/streaming_asr/1/model.py`
2. `triton-service/streaming_diar_triton/models/streaming_diar/1/model.py`
3. `intelligent_pipeline/streaming_asr_client.py`
4. `intelligent_pipeline/streaming_diar_client.py`
5. `intelligent_pipeline/config.json`
6. `docker-compose.streaming.yml`

In short:
- Triton model files contain the actual streaming logic
- client files contain the calling contract
- config and compose files contain the operational wiring
