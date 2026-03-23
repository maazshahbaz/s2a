"""
Local CosyVoice2 trial script for S2A.

This proof-of-concept stays outside the main application stack. It is meant for
side-by-side evaluation against the existing Magpie batch test on a Linux GPU
host.

What it does:
- loads CosyVoice2 from a local model directory
- runs either `instruct2` or `zero_shot` inference
- accepts a single prompt from CLI args or a JSONL manifest
- writes one WAV per prompt and a compact report.json

This script intentionally does not start a service or Triton runtime yet.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_MODEL_ID = "FunAudioLLM/CosyVoice2-0.5B"
DEFAULT_MODEL_DIR = Path("pretrained_models/CosyVoice2-0.5B")
DEFAULT_TEXT = "Thank you for calling support. I can help you with that today."
DEFAULT_INSTRUCTION = (
    "Speak like a calm, empathetic customer support agent. "
    "Medium pace. Clear pronunciation. Warm but professional tone.<|endofprompt|>"
)
DEFAULT_PROMPT_TEXT = "Hello, this is a reference voice sample for zero shot synthesis."


@dataclass
class PromptItem:
    mode: str
    text: str
    output_name: str
    prompt_wav: Path
    instruction: Optional[str] = None
    prompt_text: Optional[str] = None
    stream: bool = False
    text_frontend: Optional[bool] = None


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "sample"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_path(path_value: str, base_dir: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def normalize_bool(value, *, field_name: str) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Unsupported boolean value for {field_name}: {value!r}")


def validate_item(item: PromptItem) -> PromptItem:
    if item.mode not in {"instruct", "zero_shot"}:
        raise ValueError(f"Unsupported mode '{item.mode}'. Use 'instruct' or 'zero_shot'.")
    if not item.text.strip():
        raise ValueError("Prompt text cannot be empty.")
    if item.mode == "instruct" and not (item.instruction or "").strip():
        raise ValueError("Instruct mode requires a non-empty instruction.")
    if item.mode == "zero_shot" and not (item.prompt_text or "").strip():
        raise ValueError("Zero-shot mode requires a non-empty prompt_text.")
    if not item.prompt_wav.exists():
        raise FileNotFoundError(f"Reference audio not found: {item.prompt_wav}")
    return item


def load_manifest(manifest_path: Path) -> List[PromptItem]:
    base_dir = manifest_path.parent.resolve()
    items: List[PromptItem] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = str(record["text"]).strip()
            mode = str(record.get("mode", "instruct")).strip()
            prompt_wav = resolve_path(str(record["prompt_wav"]).strip(), base_dir)
            output_name = str(record.get("output_name", f"{index:03d}_{slugify(text[:40])}.wav")).strip()
            item = PromptItem(
                mode=mode,
                text=text,
                output_name=output_name,
                prompt_wav=prompt_wav,
                instruction=str(record.get("instruction", "")).strip() or None,
                prompt_text=str(record.get("prompt_text", "")).strip() or None,
                stream=bool(record.get("stream", False)),
                text_frontend=normalize_bool(record.get("text_frontend"), field_name="text_frontend"),
            )
            items.append(validate_item(item))
    if not items:
        raise ValueError(f"No prompt items found in {manifest_path}")
    return items


def build_single_item_from_args(args: argparse.Namespace) -> PromptItem:
    if not args.prompt_wav:
        raise SystemExit(
            "Single-prompt mode requires --prompt-wav.\n"
            "For batch mode, use --manifest with JSONL entries."
        )
    item = PromptItem(
        mode=args.mode,
        text=args.text,
        output_name=args.output_name or f"001_{slugify(args.text[:40])}.wav",
        prompt_wav=args.prompt_wav.resolve(),
        instruction=args.instruction,
        prompt_text=args.prompt_text,
        stream=args.stream,
        text_frontend=False if args.no_text_frontend else None,
    )
    return validate_item(item)


def import_dependencies(cosyvoice_repo: Path):
    repo_path = cosyvoice_repo.resolve()
    matcha_path = repo_path / "third_party" / "Matcha-TTS"

    sys.path.insert(0, str(repo_path))
    if matcha_path.exists():
        sys.path.insert(0, str(matcha_path))

    try:
        import torch
        import torchaudio
        from huggingface_hub import snapshot_download
        from cosyvoice.cli.cosyvoice import AutoModel
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for CosyVoice2 local inference. Follow the official setup in a"
            " dedicated CosyVoice environment, then rerun this script.\n"
            f"CosyVoice repo: {repo_path}\n"
            f"Original import error: {exc}"
        ) from exc
    return torch, torchaudio, snapshot_download, AutoModel


def ensure_model_dir(model_dir: Path, model_id: str, download_model: bool, snapshot_download) -> Path:
    if model_dir.exists():
        return model_dir.resolve()
    if not download_model:
        raise SystemExit(
            f"Model directory not found: {model_dir}\n"
            "Either download the model first or rerun with --download-model."
        )
    print(f"Downloading {model_id} to {model_dir} ...")
    snapshot_download(repo_id=model_id, local_dir=str(model_dir), local_dir_use_symlinks=False)
    return model_dir.resolve()


def normalize_audio_chunk(torch_mod, chunk) -> "torch.Tensor":
    if not hasattr(chunk, "dim"):
        chunk = torch_mod.as_tensor(chunk)
    chunk = chunk.detach().cpu().float()
    if chunk.dim() == 1:
        chunk = chunk.unsqueeze(0)
    return chunk


def collect_audio(torch_mod, generator: Iterable[dict]):
    chunks = []
    first_chunk_sec = None
    started = time.perf_counter()
    chunk_count = 0

    for result in generator:
        audio = normalize_audio_chunk(torch_mod, result["tts_speech"])
        if first_chunk_sec is None:
            first_chunk_sec = time.perf_counter() - started
        chunks.append(audio)
        chunk_count += 1

    if not chunks:
        raise RuntimeError("CosyVoice returned no audio chunks.")

    combined = torch_mod.cat(chunks, dim=-1)
    return combined, chunk_count, first_chunk_sec


def run_item(cosyvoice, torch_mod, torchaudio_mod, item: PromptItem, out_dir: Path) -> dict:
    kwargs = {"stream": item.stream}
    if item.text_frontend is not None:
        kwargs["text_frontend"] = item.text_frontend

    started = time.perf_counter()
    if item.mode == "instruct":
        generator = cosyvoice.inference_instruct2(
            item.text,
            item.instruction,
            str(item.prompt_wav),
            **kwargs,
        )
    else:
        generator = cosyvoice.inference_zero_shot(
            item.text,
            item.prompt_text,
            str(item.prompt_wav),
            **kwargs,
        )

    audio, chunk_count, first_chunk_sec = collect_audio(torch_mod, generator)
    elapsed = time.perf_counter() - started

    output_path = out_dir / item.output_name
    torchaudio_mod.save(str(output_path), audio, cosyvoice.sample_rate)

    sample_count = int(audio.shape[-1])
    audio_duration_sec = sample_count / cosyvoice.sample_rate if cosyvoice.sample_rate else None
    return {
        "mode": item.mode,
        "text": item.text,
        "instruction": item.instruction,
        "prompt_text": item.prompt_text,
        "prompt_wav": str(item.prompt_wav),
        "stream": item.stream,
        "text_frontend": item.text_frontend,
        "output_path": str(output_path),
        "sample_rate": cosyvoice.sample_rate,
        "audio_samples": sample_count,
        "audio_duration_sec": round(audio_duration_sec, 3) if audio_duration_sec is not None else None,
        "elapsed_sec": round(elapsed, 3),
        "first_chunk_sec": round(first_chunk_sec, 3) if first_chunk_sec is not None else None,
        "rtf": round(elapsed / audio_duration_sec, 3) if audio_duration_sec else None,
        "chunk_count": chunk_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local CosyVoice2 trial from S2A.")
    parser.add_argument("--cosyvoice-repo", type=Path, default=Path(os.environ["COSYVOICE_REPO"]) if os.environ.get("COSYVOICE_REPO") else None, help="Path to a cloned FunAudioLLM/CosyVoice repo")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help=f"Hugging Face model id to download (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help=f"Local CosyVoice2 model directory (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--download-model", action="store_true", help="Download the CosyVoice2 model if --model-dir does not exist")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSONL manifest for multiple prompts")
    parser.add_argument("--mode", choices=["instruct", "zero_shot"], default="instruct", help="Single-prompt mode only")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Target synthesis text for single-prompt mode")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="Instruction text for single-prompt instruct mode")
    parser.add_argument("--prompt-text", default=DEFAULT_PROMPT_TEXT, help="Reference transcript for zero-shot mode")
    parser.add_argument("--prompt-wav", type=Path, default=None, help="Reference speaker WAV for single-prompt mode")
    parser.add_argument("--output-name", default=None, help="Output WAV filename for single-prompt mode")
    parser.add_argument("--out-dir", type=Path, default=Path("results/tts_cosyvoice2"), help="Output directory for WAVs and report")
    parser.add_argument("--stream", action="store_true", help="Use audio-out streaming and stitch chunks into one WAV")
    parser.add_argument("--no-text-frontend", action="store_true", help="Pass text_frontend=False during inference")
    args = parser.parse_args()

    if not args.cosyvoice_repo:
        raise SystemExit(
            "CosyVoice runtime code is required. Pass --cosyvoice-repo or set COSYVOICE_REPO."
        )

    ensure_dir(args.out_dir)
    torch, torchaudio, snapshot_download, AutoModel = import_dependencies(args.cosyvoice_repo)
    model_dir = ensure_model_dir(args.model_dir, args.model_id, args.download_model, snapshot_download)
    items = load_manifest(args.manifest.resolve()) if args.manifest else [build_single_item_from_args(args)]

    print("\n=== CosyVoice2 Local Trial ===")
    print(f"CosyVoice repo: {args.cosyvoice_repo.resolve()}")
    print(f"Model dir: {model_dir}")
    print(f"Prompts: {len(items)}")
    print(f"Streaming: {any(item.stream for item in items)}")

    torch.set_grad_enabled(False)
    cosyvoice = AutoModel(model_dir=str(model_dir))
    print(f"Sample rate: {cosyvoice.sample_rate} Hz")

    records = []
    total_started = time.perf_counter()
    with torch.inference_mode():
        for index, item in enumerate(items, start=1):
            print(f"Processing item {index}/{len(items)}: {item.output_name}")
            record = run_item(cosyvoice, torch, torchaudio, item, args.out_dir)
            records.append(record)

    total_elapsed = time.perf_counter() - total_started
    total_audio_sec = sum(item["audio_duration_sec"] for item in records if item["audio_duration_sec"] is not None)
    summary = {
        "model_id": args.model_id,
        "model_dir": str(model_dir),
        "prompt_count": len(records),
        "total_elapsed_sec": round(total_elapsed, 3),
        "total_audio_sec": round(total_audio_sec, 3),
        "overall_rtf": round(total_elapsed / total_audio_sec, 3) if total_audio_sec else None,
        "items": records,
    }

    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(records)} WAV file(s) to {args.out_dir}")
    print(f"Summary report: {report_path}")
    print(f"Total elapsed: {total_elapsed:.2f}s")
    if total_audio_sec:
        print(f"Overall RTF: {total_elapsed / total_audio_sec:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
