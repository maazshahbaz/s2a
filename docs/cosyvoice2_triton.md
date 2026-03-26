# CosyVoice2 Triton Deployment

This is the production-oriented deployment path for `FunAudioLLM/CosyVoice2-0.5B`
in S2A when it is run as a separate service from the existing Triton stack.

Primary sources:

- CosyVoice repo: https://github.com/FunAudioLLM/CosyVoice
- Official Triton runtime README: https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/runtime/triton_trtllm/README.md
- Official Triton runtime compose file: https://raw.githubusercontent.com/FunAudioLLM/CosyVoice/main/runtime/triton_trtllm/docker-compose.yml

## Why Separate

S2A already has ASR, diarization, and LLM Triton services running on shared GPU
infrastructure. CosyVoice2 should be deployed in its own compose stack so:

- it can be rolled independently
- engine build and model warmup do not interfere with the current Triton stack
- TTS GPU scheduling can be tuned separately from ASR and diarization

The repo-side compose file for this path is:

- [`docker-compose.cosyvoice-triton.yml`](../docker-compose.cosyvoice-triton.yml)

## Important Runtime Limitation

The official `runtime/triton_trtllm` path for CosyVoice2 is not the same API as
the local Python `instruct2` path we used for evaluation.

The supported Triton request contract is currently:

- `reference_wav`
- `reference_wav_len`
- `reference_text`
- `target_text`

That means this production path is a `reference-audio + reference-transcript +
target-text` synthesis flow. It does **not** expose the free-form `instruction`
field from `inference_instruct2(...)` out of the box.

For S2A, that means:

- fixed approved speakers can work well
- zero-shot/reference-based cloning works
- explicit customer-support style prompting needs either runtime customization or
  the Python service path instead of this Triton runtime

## Environment

Copy the example env file and adjust ports/GPU as needed:

- [`cosyvoice-triton.env.example`](../cosyvoice-triton.env.example)

Default ports:

- HTTP: `3950`
- gRPC: `3951`
- Metrics: `3952`

## First Start

```bash
docker compose \
  --env-file cosyvoice-triton.env.example \
  -f docker-compose.cosyvoice-triton.yml \
  up -d
```

The first start can take a long time because the container will:

1. clone `FunAudioLLM/CosyVoice`
2. download the CosyVoice2 checkpoints
3. convert the Hugging Face LLM checkpoint into TensorRT-LLM weights
4. build the TensorRT engines
5. create the Triton model repository
6. launch the Triton server

This follows upstream `run.sh 0 3`.

## Health Check

```bash
curl http://localhost:3950/v2/health/ready
```

## Smoke Test

Use the repo-side client:

```bash
python scripts/test_cosyvoice2_triton_http.py \
  --server-url http://localhost:3950 \
  --reference-audio /path/to/reference_16k.wav \
  --reference-text "Transcript of the reference clip." \
  --target-text "Thank you for calling support. I can help you with that today." \
  --output-audio results/tts_cosyvoice2_triton/output.wav
```

The client will:

- resample the reference WAV to `16 kHz`
- trim it to `30s` if needed
- send the official offline HTTP request shape
- save the returned `24 kHz` WAV

## Production Recommendation For S2A

Use this Triton path when:

- you want a separate GPU-backed TTS service
- reference-based speaker control is acceptable
- you want the official TensorRT-LLM optimization path

Do not assume this path preserves the exact `instruct2` behavior from the local
Python evaluation. If product requires promptable styles like "calm empathetic
customer-support agent" in production, validate that separately before rolling
this as the final user-facing TTS path.
