"""
Direct Hugging Face + NeMo batch test for Magpie TTS.

This script is intentionally simple:
- loads the public HF checkpoint
- reads a JSONL batch manifest
- synthesizes one or more prompts
- writes WAV files and a small summary report

It is designed for Linux GPU hosts used for local/staging validation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


SPEAKER_MAP = {
    "John": 0,
    "Sofia": 1,
    "Aria": 2,
    "Jason": 3,
    "Leo": 4,
}
DEFAULT_MODEL_ID = "nvidia/magpie_tts_multilingual_357m"
DEFAULT_SAMPLE_RATE = 22050


@dataclass
class PromptItem:
    text: str
    language: str
    speaker: str
    apply_tn: bool
    output_name: str


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return value or "sample"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_manifest(manifest_path: Path) -> List[PromptItem]:
    items: List[PromptItem] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = str(record["text"]).strip()
            if not text:
                raise ValueError(f"Manifest line {index} has empty text")
            speaker = str(record.get("speaker", "Aria")).strip()
            if speaker not in SPEAKER_MAP:
                raise ValueError(
                    f"Manifest line {index} uses unsupported speaker '{speaker}'. "
                    f"Supported values: {', '.join(SPEAKER_MAP)}"
                )
            language = str(record.get("language", "en")).strip()
            apply_tn = bool(record.get("apply_tn", False))
            output_name = str(record.get("output_name", f"{index:03d}_{slugify(text[:40])}.wav")).strip()
            items.append(
                PromptItem(
                    text=text,
                    language=language,
                    speaker=speaker,
                    apply_tn=apply_tn,
                    output_name=output_name,
                )
            )
    if not items:
        raise ValueError(f"No prompt items found in {manifest_path}")
    return items


def default_manifest_items() -> List[PromptItem]:
    return [
        PromptItem(
            text="Hello from S2A. This is a local Magpie batch test running through the Hugging Face checkpoint.",
            language="en",
            speaker="Aria",
            apply_tn=False,
            output_name="001_hello_from_s2a.wav",
        ),
        PromptItem(
            text="This second prompt lets us validate repeated generation in a single batch job.",
            language="en",
            speaker="Aria",
            apply_tn=False,
            output_name="002_repeated_generation.wav",
        ),
    ]


def chunked(items: Sequence[PromptItem], batch_size: int) -> Iterable[Sequence[PromptItem]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def import_dependencies():
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from nemo.collections.tts.models import MagpieTTSModel
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for Magpie HF inference. Install on the Linux GPU host with:\n"
            "  pip install nemo_toolkit[tts]@main\n"
            "  pip install kaldialign soundfile\n"
            f"Original import error: {exc}"
        ) from exc
    return np, sf, torch, MagpieTTSModel


def detect_sample_rate(model) -> int:
    cfg = getattr(model, "cfg", None)
    for attr in ("sample_rate", "sampling_rate"):
        direct = getattr(model, attr, None)
        if isinstance(direct, int):
            return direct
        cfg_value = getattr(cfg, attr, None) if cfg is not None else None
        if isinstance(cfg_value, int):
            return cfg_value
    return DEFAULT_SAMPLE_RATE


def to_numpy_audio(np_mod, audio):
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    else:
        audio = np_mod.asarray(audio, dtype=np_mod.float32)
    return np_mod.squeeze(audio).astype(np_mod.float32)


def normalize_lengths(lengths, batch_len: int) -> List[int | None]:
    if lengths is None:
        return [None] * batch_len
    if hasattr(lengths, "detach"):
        lengths = lengths.detach().cpu().tolist()
    elif not isinstance(lengths, (list, tuple)):
        lengths = [lengths]
    return [int(x) if x is not None else None for x in lengths]


def save_audio(sf_mod, audio_np, output_path: Path, sample_rate: int) -> int:
    sf_mod.write(str(output_path), audio_np, sample_rate)
    return int(audio_np.shape[-1]) if hasattr(audio_np, "shape") and audio_np.shape else len(audio_np)


def synthesize_batch(
    model,
    np_mod,
    sf_mod,
    prompts: Sequence[PromptItem],
    out_dir: Path,
    sample_rate: int,
) -> List[dict]:
    first = prompts[0]
    homogeneous = all(
        p.language == first.language and p.speaker == first.speaker and p.apply_tn == first.apply_tn
        for p in prompts
    )

    started = time.perf_counter()
    batch_mode = "sequential"
    records: List[dict] = []

    if homogeneous:
        transcripts = [p.text for p in prompts]
        try:
            audios, lengths = model.do_tts(
                transcripts,
                language=first.language,
                apply_TN=first.apply_tn,
                speaker_index=SPEAKER_MAP[first.speaker],
            )
            batch_mode = "list_input"
            length_values = normalize_lengths(lengths, len(prompts))
            for prompt, audio, audio_len in zip(prompts, audios, length_values):
                output_path = out_dir / prompt.output_name
                audio_np = to_numpy_audio(np_mod, audio)
                sample_count = save_audio(sf_mod, audio_np, output_path, sample_rate)
                if audio_len is None or audio_len <= 0:
                    audio_len = sample_count
                records.append(
                    {
                        "text": prompt.text,
                        "language": prompt.language,
                        "speaker": prompt.speaker,
                        "apply_tn": prompt.apply_tn,
                        "output_path": str(output_path),
                        "audio_samples": int(audio_len),
                        "audio_duration_sec": round(audio_len / sample_rate, 3),
                    }
                )
        except Exception:
            batch_mode = "sequential_fallback"

    if not records:
        for prompt in prompts:
            audio, audio_len = model.do_tts(
                prompt.text,
                language=prompt.language,
                apply_TN=prompt.apply_tn,
                speaker_index=SPEAKER_MAP[prompt.speaker],
            )
            output_path = out_dir / prompt.output_name
            audio_np = to_numpy_audio(np_mod, audio)
            sample_count = save_audio(sf_mod, audio_np, output_path, sample_rate)
            if hasattr(audio_len, "detach"):
                audio_len = int(audio_len.detach().cpu().item())
            elif isinstance(audio_len, (list, tuple)):
                audio_len = int(audio_len[0])
            else:
                audio_len = int(audio_len) if audio_len else sample_count
            records.append(
                {
                    "text": prompt.text,
                    "language": prompt.language,
                    "speaker": prompt.speaker,
                    "apply_tn": prompt.apply_tn,
                    "output_path": str(output_path),
                    "audio_samples": int(audio_len),
                    "audio_duration_sec": round(audio_len / sample_rate, 3),
                }
            )

    elapsed = time.perf_counter() - started
    total_audio_sec = sum(r["audio_duration_sec"] for r in records)
    per_item_elapsed = elapsed / max(1, len(records))
    rtf = elapsed / total_audio_sec if total_audio_sec > 0 else None

    for record in records:
        record["batch_mode"] = batch_mode
        record["batch_elapsed_sec"] = round(elapsed, 3)
        record["per_item_elapsed_sec"] = round(per_item_elapsed, 3)
        record["batch_rtf"] = round(rtf, 3) if rtf is not None else None

    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local HF + NeMo batch test for Magpie TTS")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help=f"Hugging Face model id (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--manifest", type=Path, default=None, help="JSONL manifest with text/language/speaker/apply_tn entries")
    parser.add_argument("--out-dir", type=Path, default=Path("results/tts_magpie_hf"), help="Output directory for WAVs and report")
    parser.add_argument("--batch-size", type=int, default=2, help="Number of prompts to process per batch group")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Inference device (default: auto)")
    args = parser.parse_args()

    np_mod, sf_mod, torch, MagpieTTSModel = import_dependencies()
    prompts = load_manifest(args.manifest) if args.manifest else default_manifest_items()
    ensure_dir(args.out_dir)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("\n=== Magpie HF Batch Test ===")
    print(f"Model: {args.model_id}")
    print(f"Prompts: {len(prompts)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {device}")

    torch.set_grad_enabled(False)
    model = MagpieTTSModel.from_pretrained(args.model_id)
    model = model.eval()
    if device == "cuda":
        model = model.cuda()

    sample_rate = detect_sample_rate(model)
    print(f"Sample rate: {sample_rate} Hz")

    all_records: List[dict] = []
    total_started = time.perf_counter()
    total_batches = math.ceil(len(prompts) / max(1, args.batch_size))

    for batch_index, batch_items in enumerate(chunked(prompts, max(1, args.batch_size)), start=1):
        print(f"Processing batch {batch_index}/{total_batches} with {len(batch_items)} item(s)")
        with torch.inference_mode():
            batch_records = synthesize_batch(
                model=model,
                np_mod=np_mod,
                sf_mod=sf_mod,
                prompts=batch_items,
                out_dir=args.out_dir,
                sample_rate=sample_rate,
            )
        all_records.extend(batch_records)

    total_elapsed = time.perf_counter() - total_started
    total_audio_sec = sum(r["audio_duration_sec"] for r in all_records)
    summary = {
        "model_id": args.model_id,
        "device": device,
        "sample_rate": sample_rate,
        "prompt_count": len(all_records),
        "batch_size": args.batch_size,
        "total_elapsed_sec": round(total_elapsed, 3),
        "total_audio_sec": round(total_audio_sec, 3),
        "overall_rtf": round(total_elapsed / total_audio_sec, 3) if total_audio_sec > 0 else None,
        "items": all_records,
    }

    report_path = args.out_dir / "report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(all_records)} WAV file(s) to {args.out_dir}")
    print(f"Summary report: {report_path}")
    print(f"Total elapsed: {total_elapsed:.2f}s")
    if total_audio_sec > 0:
        print(f"Overall RTF: {total_elapsed / total_audio_sec:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
