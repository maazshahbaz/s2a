import re
from typing import Dict, Optional, Tuple


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _word_overlap_suffix_prefix(left_text: str, right_text: str, max_words: int = 64) -> int:
    """
    Return the largest K such that the last K words of left_text match
    the first K words of right_text.
    """
    left_words = normalize_text(left_text).split()
    right_words = normalize_text(right_text).split()
    if not left_words or not right_words:
        return 0

    max_k = min(len(left_words), len(right_words), max_words)
    for k in range(max_k, 0, -1):
        if left_words[-k:] == right_words[:k]:
            return k
    return 0


def _prefix_overlap_with_recent_tail(
    existing_text: str,
    new_text: str,
    max_words: int = 48,
    tail_chars: int = 1200,
) -> int:
    """
    Return the largest K such that the first K words of new_text appear
    somewhere in the recent tail of existing_text.
    """
    existing_clean = normalize_text(existing_text)
    new_words = normalize_text(new_text).split()
    if not existing_clean or not new_words:
        return 0

    recent_tail = existing_clean[-tail_chars:]
    padded_tail = f" {recent_tail} "
    max_k = min(len(new_words), max_words)
    for k in range(max_k, 1, -1):
        prefix = " ".join(new_words[:k])
        if f" {prefix} " in padded_tail:
            return k
    return 0


def extract_model_delta(model_text: str, previous_model_text: str) -> Tuple[str, str]:
    """
    Extract incremental text from the latest model output using simple, low-risk rules.

    Rules:
    - Empty output => no delta
    - Exact repeat => no delta
    - Prefix growth => emit only appended suffix
    - Sliding-window growth => emit only non-overlapping suffix
    - Truncated rollback => no delta
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

    # Handle models that return sliding windows (tail of previous + new words).
    overlap_words = _word_overlap_suffix_prefix(previous, new_text)
    if overlap_words >= 2:
        new_words = new_text.split()
        suffix_words = new_words[overlap_words:]
        if suffix_words:
            return " ".join(suffix_words), new_text
        return "", new_text

    # If output rolls back to an earlier truncated phrase, don't re-emit.
    if previous.endswith(new_text):
        return "", previous

    return new_text, new_text


def append_transcript(existing_text: str, new_text: str) -> str:
    existing_clean = normalize_text(existing_text)
    clean_new = normalize_text(new_text)
    if not clean_new:
        return existing_clean
    if not existing_clean:
        return clean_new

    # Drop obvious immediate duplicates.
    if existing_clean == clean_new or existing_clean.endswith(clean_new):
        return existing_clean

    # Drop stale repeats seen in the recent transcript tail (common with
    # model instance hopping / reset-like outputs).
    tail_chars = max(512, len(clean_new) * 3)
    recent_tail = existing_clean[-tail_chars:]
    if f" {clean_new} " in f" {recent_tail} ":
        return existing_clean

    # If the new chunk starts with a phrase already present in the recent tail,
    # trim that repeated prefix and keep only novel continuation.
    recent_prefix_overlap = _prefix_overlap_with_recent_tail(existing_clean, clean_new)
    if recent_prefix_overlap >= 2:
        suffix_words = clean_new.split()[recent_prefix_overlap:]
        if not suffix_words:
            return existing_clean
        clean_new = " ".join(suffix_words)

    # Merge with overlap to keep a smooth flow without duplicated boundary words.
    overlap_words = _word_overlap_suffix_prefix(existing_clean, clean_new)
    if overlap_words > 0:
        suffix_words = clean_new.split()[overlap_words:]
        if not suffix_words:
            return existing_clean
        clean_new = " ".join(suffix_words)

    return f"{existing_clean} {clean_new}".strip()


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
