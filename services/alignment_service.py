"""
Alignment service to map ASR timestamps to diarization segments and
produce a speaker-attributed transcript representation.
"""

from __future__ import annotations

from typing import List, Dict, Any, Tuple


def _interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0.0, end - start)


def align_sentence_segments(
    asr_segments: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Align sentence/segment-level ASR results to diarization segments using overlap.
    If ASR segments are large, split them based on diarization boundaries.

    asr_segments: [{start_time, end_time, text}]
    diar_segments: [{start, end, speaker}]

    Returns: (speaker_blocks, num_speakers)
      speaker_blocks: [{speaker, start, end, text}]
    """
    blocks: List[Dict[str, Any]] = []
    
    # Sort diarization segments by start time
    sorted_diar = sorted(diar_segments, key=lambda x: float(x["start"]))
    
    for seg in asr_segments:
        s_start = float(seg.get("start_time", 0.0))
        s_end = float(seg.get("end_time", 0.0))
        s_text = seg.get("text", "").strip()
        if not s_text:
            continue
        
        # Find all diarization segments that overlap with this ASR segment
        overlapping_diar = []
        for d in sorted_diar:
            d_start = float(d["start"])
            d_end = float(d["end"])
            overlap = _interval_overlap(s_start, s_end, d_start, d_end)
            if overlap > 0:
                overlapping_diar.append({
                    'speaker': d["speaker"],
                    'start': d_start,
                    'end': d_end,
                    'overlap': overlap
                })
        
        if not overlapping_diar:
            # No overlap, assign to default speaker
            blocks.append({
                "speaker": "SPK_1",
                "start": s_start,
                "end": s_end,
                "text": s_text,
            })
            continue
        
        # If ASR segment is large and spans multiple diarization segments,
        # split the text proportionally
        if len(overlapping_diar) > 1:
            # Split text based on time proportions
            words = s_text.split()
            total_duration = s_end - s_start
            word_duration = total_duration / len(words) if words else 0
            
            word_idx = 0
            for diar in overlapping_diar:
                # Calculate how many words belong to this diarization segment
                diar_duration = min(diar['end'], s_end) - max(diar['start'], s_start)
                num_words = int((diar_duration / word_duration)) if word_duration > 0 else len(words)
                num_words = max(1, min(num_words, len(words) - word_idx))
                
                segment_words = words[word_idx:word_idx + num_words]
                segment_text = " ".join(segment_words)
                
                if segment_text:
                    # Merge with previous block if same speaker
                    if blocks and blocks[-1]["speaker"] == diar['speaker']:
                        blocks[-1]["end"] = min(diar['end'], s_end)
                        blocks[-1]["text"] = (blocks[-1]["text"] + " " + segment_text).strip()
                    else:
                        blocks.append({
                            "speaker": diar['speaker'],
                            "start": max(diar['start'], s_start),
                            "end": min(diar['end'], s_end),
                            "text": segment_text,
                        })
                
                word_idx += num_words
                if word_idx >= len(words):
                    break
        else:
            # Single diarization segment, use it directly
            diar = overlapping_diar[0]
            # Merge with previous block if same speaker
            if blocks and blocks[-1]["speaker"] == diar['speaker']:
                blocks[-1]["end"] = s_end
                blocks[-1]["text"] = (blocks[-1]["text"] + " " + s_text).strip()
            else:
                blocks.append({
                    "speaker": diar['speaker'],
                    "start": s_start,
                    "end": s_end,
                    "text": s_text,
                })
    
    num_speakers = len(set(b["speaker"] for b in blocks))
    return blocks, num_speakers


def render_speaker_attributed_text(blocks: List[Dict[str, Any]]) -> str:
    lines = []
    for b in blocks:
        lines.append(f"{b['speaker']}: {b['text']}")
    return "\n".join(lines)


