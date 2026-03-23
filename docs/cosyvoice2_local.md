# CosyVoice2 Local Trial

This is the second TTS comparison path for S2A.

Unlike the Magpie test, this path uses the official `FunAudioLLM/CosyVoice`
runtime code plus the `FunAudioLLM/CosyVoice2-0.5B` model.

Primary sources:

- CosyVoice repo: https://github.com/FunAudioLLM/CosyVoice
- CosyVoice2 model: https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B

## Why This Path

The goal is to test whether CosyVoice2 is a better fit for customer-support
speech than Magpie because it supports free-form instructions like:

- calm empathetic customer support agent
- medium pace, clear pronunciation
- reassuring and professional

The first pass is still local Python inference, not a service deployment.

## What This Script Covers

[`scripts/test_cosyvoice2_local.py`](../scripts/test_cosyvoice2_local.py) supports:

- `instruct` mode via `inference_instruct2(...)`
- `zero_shot` mode via `inference_zero_shot(...)`
- single-prompt CLI runs
- JSONL manifest runs
- stitched WAV output even when audio-out streaming is enabled

This is enough to compare voice quality, style control, and rough latency before
we decide whether to move forward with Triton or another serving path.

## Setup

Follow the official CosyVoice install in a separate Linux GPU environment. Their
README currently recommends Python 3.10 and cloning the repo recursively.

```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git /opt/CosyVoice
cd /opt/CosyVoice
conda create -n cosyvoice -y python=3.10
conda activate cosyvoice
pip install -r requirements.txt
```

Optional but useful if you want to let the S2A script download the model:

```bash
pip install huggingface_hub
```

## Model Download

You can either let the S2A script download the model, or download it yourself:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='FunAudioLLM/CosyVoice2-0.5B', local_dir='/opt/CosyVoice/pretrained_models/CosyVoice2-0.5B')"
```

## Single-Prompt Run

You need a short reference speaker WAV for both `instruct` and `zero_shot`
testing.

```bash
python scripts/test_cosyvoice2_local.py \
  --cosyvoice-repo /opt/CosyVoice \
  --model-dir /opt/CosyVoice/pretrained_models/CosyVoice2-0.5B \
  --prompt-wav /path/to/reference.wav \
  --out-dir results/tts_cosyvoice2
```

To compare the same reference voice without free-form instructions:

```bash
python scripts/test_cosyvoice2_local.py \
  --cosyvoice-repo /opt/CosyVoice \
  --model-dir /opt/CosyVoice/pretrained_models/CosyVoice2-0.5B \
  --mode zero_shot \
  --prompt-text "This is the transcript of the reference clip." \
  --prompt-wav /path/to/reference.wav \
  --out-dir results/tts_cosyvoice2_zero_shot
```

## Manifest Run

Start from [`scripts/cosyvoice2_local_example.jsonl`](../scripts/cosyvoice2_local_example.jsonl),
replace `REPLACE_WITH_REFERENCE.wav`, then run:

```bash
python scripts/test_cosyvoice2_local.py \
  --cosyvoice-repo /opt/CosyVoice \
  --model-dir /opt/CosyVoice/pretrained_models/CosyVoice2-0.5B \
  --manifest scripts/cosyvoice2_local_example.jsonl \
  --out-dir results/tts_cosyvoice2_manifest
```

## Notes

- CosyVoice2 `instruct` mode is the relevant one for customer-support style
  testing.
- The reference WAV still matters because this model uses a reference speaker
  clip for `instruct2` and `zero_shot`.
- The official repo also mentions Triton + TensorRT-LLM runtime support, but
  this script intentionally stays at the local inference layer first.
