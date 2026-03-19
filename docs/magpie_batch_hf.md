# Magpie HF Batch Test

This is the current first-test path for TTS in S2A.

We are not using NIM for this phase.
We are not using Triton for this phase.
We are loading the public Hugging Face checkpoint directly through NeMo on a Linux GPU host.

Model:

- `nvidia/magpie_tts_multilingual_357m`

Primary source:

- https://huggingface.co/nvidia/magpie_tts_multilingual_357m

## Why This Path

The immediate goal is simple local validation:

- load the model
- synthesize a small batch of prompts
- save WAV files
- measure rough latency and RTF

This avoids:

- NGC authentication
- NIM container setup
- Triton deployment work

We only decide on the final serving path after we hear the outputs and see the runtime behavior.

## Host Requirements

Use a Linux machine with an NVIDIA GPU.

The model card says:

- preferred OS: Linux
- supported hardware: NVIDIA GPUs
- runtime engine: NeMo Framework
- acceleration engine: none

This is not meant for the current Windows no-GPU dev machine.

## Install

Use a fresh Python environment on the Linux GPU host.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "nemo_toolkit[tts]@main"
pip install kaldialign soundfile
```

If PyTorch is not already installed appropriately for the GPU host, install the correct CUDA build first, then install NeMo.

## Run

Default sample batch:

```bash
python scripts/test_magpie_batch_hf.py
```

Custom batch manifest:

```bash
python scripts/test_magpie_batch_hf.py \
  --manifest scripts/magpie_batch_example.jsonl \
  --out-dir results/tts_magpie_hf \
  --batch-size 2
```

## Manifest Format

JSONL, one prompt per line:

```json
{"text":"Hello world","language":"en","speaker":"Aria","apply_tn":false,"output_name":"hello.wav"}
```

Fields:

- `text`: required
- `language`: optional, default `en`
- `speaker`: optional, one of `John`, `Sofia`, `Aria`, `Jason`, `Leo`
- `apply_tn`: optional, default `false`
- `output_name`: optional, output WAV filename

## What The Script Does

[`scripts/test_magpie_batch_hf.py`](../scripts/test_magpie_batch_hf.py) does the following:

1. loads the HF model through `MagpieTTSModel.from_pretrained(...)`
2. moves it to GPU if CUDA is available
3. reads a batch manifest
4. processes prompts in batch groups
5. tries list-input inference for homogeneous batches
6. falls back to per-item synthesis if the installed NeMo build rejects direct list batching
7. writes WAV files and `report.json`

That fallback is intentional. It makes the first validation path robust across NeMo revisions while still trying to use batch-shaped inputs when possible.

## Outputs

The script writes:

- one WAV per prompt
- `report.json` with timing and output metadata

Example output directory:

- `results/tts_magpie_hf/`

## What We Evaluate First

For the initial pass, only answer these questions:

- Does the model load successfully on the Linux GPU host?
- Are the voices acceptable for S2A use?
- Is latency acceptable for offline generation?
- Does throughput look reasonable enough to continue?

If yes, then we decide whether to:

- keep direct HF inference for internal testing
- wrap it in a service
- or move to a different deployment path later
