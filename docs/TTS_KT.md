# TTS KT

This branch is the production TTS path for S2A.

It contains only the code needed to run CosyVoice2 as a separate Triton service
and call it from S2A.

## What This Branch Includes

- [docker-compose.cosyvoice-triton.yml](/W:/99Technologies/s2a/docker-compose.cosyvoice-triton.yml)
- [cosyvoice-triton.env.example](/W:/99Technologies/s2a/cosyvoice-triton.env.example)
- [intelligent_pipeline/config.json](/W:/99Technologies/s2a/intelligent_pipeline/config.json)
- [intelligent_pipeline/tts_client.py](/W:/99Technologies/s2a/intelligent_pipeline/tts_client.py)
- [docs/cosyvoice2_triton.md](/W:/99Technologies/s2a/docs/cosyvoice2_triton.md)

## What Has Been Implemented

### 1. Separate CosyVoice2 Triton deployment

CosyVoice2 runs outside the main S2A Triton stack in its own compose file.

This keeps TTS separate from:

- ASR
- diarization
- LLM workloads

Main file:

- [docker-compose.cosyvoice-triton.yml](/W:/99Technologies/s2a/docker-compose.cosyvoice-triton.yml)

### 2. S2A-side TTS service configuration

S2A now has a `tts` service entry in:

- [intelligent_pipeline/config.json](/W:/99Technologies/s2a/intelligent_pipeline/config.json)

This defines:

- service URL
- Triton model name
- output sample rate
- reference sample rate
- max reference duration
- timeout

### 3. Async TTS client in S2A

The app-side caller is:

- [intelligent_pipeline/tts_client.py](/W:/99Technologies/s2a/intelligent_pipeline/tts_client.py)

This client:

- loads a reference speaker WAV
- converts stereo to mono if needed
- resamples to `16 kHz`
- trims the reference audio if needed
- sends a Triton infer request
- receives synthesized audio back
- can optionally write output to disk

## Current Runtime Contract

The current Triton path is reference-based synthesis.

Inputs:

- `reference_wav`
- `reference_wav_len`
- `reference_text`
- `target_text`

So the current TTS flow is not:

- `text -> choose speaker id -> audio`

It is:

- `reference speaker audio + transcript + target text -> audio`

That means speaker selection currently depends on choosing the correct approved
reference WAV and transcript pair.

## What Is Still Missing

This branch does not yet provide:

- LLM agent integration
- a speaker-profile registry
- live audio streaming back into calls
- telephony/media output handling
- interruption or barge-in support