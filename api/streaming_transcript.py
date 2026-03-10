import re
from typing import Dict, Optional, Tuple


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_model_delta(model_text: str, previous_model_text: str) -> Tuple[str, str]:
    """
    Extract incremental text from the latest model output using simple, low-risk rules.

    Rules:
    - Empty output => no delta
    - Exact repeat => no delta
    - Prefix growth => emit only appended suffix
    - Otherwise treat as reset and emit full model_text
    """
    new_text = normalize_text(model_text)
    previous = normalize_text(previous_model_text)

    if not new_text:
        return "", previous
    if not previous:
        return new_text, new_text
    if new_text == previous:
        return "", previous
    if new_text.startswith(previous):
        return new_text[len(previous):].lstrip(), new_text
    return new_text, new_text


def append_transcript(existing_text: str, new_text: str) -> str:
    clean_new = normalize_text(new_text)
    if not clean_new:
        return normalize_text(existing_text)
    if not existing_text:
        return clean_new
    return f"{normalize_text(existing_text)} {clean_new}".strip()


def choose_speaker_id(diar_result: Dict, time_point: float) -> int:
    def _coerce_speaker_id(raw_speaker) -> int:
        if isinstance(raw_speaker, int):
            return raw_speaker
        if isinstance(raw_speaker, str):
            match = re.search(r"\d+", raw_speaker)
            if match:
                return int(match.group())
        try:
            return int(raw_speaker)
        except (TypeError, ValueError):
            return 0

    segments = (diar_result or {}).get("segments", [])
    parsed_segments = []
    parsed_speaker_ids = []

    for seg in segments:
        try:
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))
            speaker = _coerce_speaker_id(seg.get("speaker", 0))
        except (TypeError, ValueError, AttributeError):
            continue
        parsed_segments.append((start, end, speaker))
        parsed_speaker_ids.append(speaker)

    if not parsed_segments:
        return 0

    speaker_base = 1 if min(parsed_speaker_ids) >= 1 else 0
    best_speaker = 0
    best_distance = float("inf")

    for start, end, raw_speaker in parsed_segments:
        speaker = max(0, raw_speaker - speaker_base)
        if start <= time_point <= end:
            return speaker
        distance = min(abs(time_point - start), abs(time_point - end))
        if distance < best_distance:
            best_distance = distance
            best_speaker = speaker

    return best_speaker


def build_chunk_segment(
    text: str,
    diar_result: Dict,
    chunk_start: float,
    chunk_end: float,
) -> Optional[Dict]:
    clean_text = normalize_text(text)
    if not clean_text:
        return None

    start = max(0.0, float(chunk_start))
    end = max(start, float(chunk_end))
    if end <= start:
        end = start + 0.001

    speaker_id = choose_speaker_id(diar_result, (start + end) / 2.0)
    return {
        "speaker": f"Speaker {speaker_id + 1}",
        "speaker_id": speaker_id,
        "text": clean_text,
        "start": round(start, 3),
        "end": round(end, 3),
    }
