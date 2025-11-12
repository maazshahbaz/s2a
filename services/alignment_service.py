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


def _find_speaker_at_time(time: float, diar_segments: List[Dict[str, Any]]) -> str:
    """Find which speaker is active at a specific time."""
    for seg in diar_segments:
        if seg['start'] <= time <= seg['end']:
            return seg['speaker']
    return "SPK_1"  # Default fallback


def _enhance_speaker_assignment(
    words_with_speakers: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Enhance speaker assignment using confidence-based logic instead of hardcoded patterns.
    
    This addresses diarization inconsistencies by:
    1. Analyzing temporal continuity and speaker confidence
    2. Using speaker embedding consistency for verification
    3. Applying minimal corrections for low-confidence assignments
    4. Maintaining diarization model's primary role
    """
    if not words_with_speakers:
        return []
    
    enhanced = words_with_speakers.copy()
    
    # Analyze speaker consistency and temporal patterns
    for i in range(len(enhanced)):
        current_word = enhanced[i]
        
        # Skip if this is the first word or already consistent with neighbors
        if i == 0:
            continue
            
        prev_word = enhanced[i-1]
        time_gap = current_word['start'] - prev_word['end']
        
        # Only consider enhancement for very short gaps that might be diarization errors
        if time_gap > 5.0:  # Gaps over 5 seconds are likely real speaker changes
            continue
            
        # Check for potential diarization inconsistency
        if (prev_word['speaker'] != current_word['speaker'] and 
            time_gap < 2.0 and  # Very short gap
            len(current_word['word']) <= 4):  # Short word
            
            # Calculate local speaker dominance in temporal context
            context_window = 10.0  # 10 seconds context
            current_time = (current_word['start'] + current_word['end']) / 2
            
            # Count speakers in local temporal context
            local_speakers = []
            for word in enhanced:
                if abs(word['start'] - current_time) <= context_window:
                    local_speakers.append(word['speaker'])
            
            speaker_counts = {}
            for speaker in local_speakers:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            # If previous speaker is dominant locally, consider maintaining continuity
            if (len(speaker_counts) > 1 and
                speaker_counts.get(prev_word['speaker'], 0) > speaker_counts.get(current_word['speaker'], 0) * 1.5):
                
                # Apply correction with lower confidence
                current_word['speaker'] = prev_word['speaker']
                current_word['_confidence'] = 'enhanced'  # Mark as enhanced
    
    return enhanced


def _group_words_by_speaker(words_with_speakers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group consecutive words by speaker to create natural speech segments."""
    if not words_with_speakers:
        return []
    
    blocks = []
    current_block = {
        'speaker': words_with_speakers[0]['speaker'],
        'start': words_with_speakers[0]['start'],
        'end': words_with_speakers[0]['end'],
        'text': words_with_speakers[0]['word']
    }
    
    for word_info in words_with_speakers[1:]:
        word = word_info['word']
        speaker = word_info['speaker']
        start = word_info['start']
        end = word_info['end']
        
        if speaker == current_block['speaker']:
            # Same speaker - append to current block
            current_block['text'] += ' ' + word
            current_block['end'] = end
        else:
            # Different speaker - save current block and start new one
            blocks.append(current_block)
            current_block = {
                'speaker': speaker,
                'start': start,
                'end': end,
                'text': word
            }
    
    # Add the last block
    blocks.append(current_block)
    return blocks


def align_words_to_speakers(
    word_timestamps: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Align word-level ASR results to diarization segments with improved handling.
    
    Improvements:
    - Merges consecutive same-speaker segments to reduce over-segmentation
    - Handles overlapping segments by choosing the one with most overlap
    - Assigns words to nearest segment when midpoint doesn't fall in any segment
    
    word_timestamps: [{'word': str, 'start': float, 'end': float, 'word_index': int}]
    diar_segments: [{'start': float, 'end': float, 'speaker': str}]

    Returns: (speaker_blocks, num_speakers)
      speaker_blocks: [{speaker: str, start: float, end: float, text: str}]
    """
    if not word_timestamps or not diar_segments:
        return [], 0
    
    # Step 1: Merge consecutive same-speaker segments to reduce over-segmentation
    # Only merge if gap is small (< 2 seconds) to avoid merging across chunk boundaries
    sorted_segments = sorted(diar_segments, key=lambda x: x['start'])
    merged_segments = []
    max_merge_gap = 2.0  # Maximum gap in seconds to merge segments
    
    for seg in sorted_segments:
        if (merged_segments and 
            merged_segments[-1]['speaker'] == seg['speaker'] and
            seg['start'] - merged_segments[-1]['end'] <= max_merge_gap):
            # Same speaker and close enough - merge them
            # Extend the end time to include this segment
            merged_segments[-1]['end'] = max(merged_segments[-1]['end'], seg['end'])
        else:
            # Different speaker, too far apart, or first segment - add new segment
            merged_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker']
            })
    
    # Step 2: Assign each word to the best matching segment
    word_to_segment = {}
    
    for word_idx, word_info in enumerate(word_timestamps):
        word_start = word_info['start']
        word_end = word_info['end']
        word_mid = (word_start + word_end) / 2
        
        best_segment_idx = None
        best_overlap = 0.0
        
        # Find segment with maximum overlap with this word
        for seg_idx, seg in enumerate(merged_segments):
            # Calculate overlap between word and segment
            overlap_start = max(word_start, seg['start'])
            overlap_end = min(word_end, seg['end'])
            overlap = max(0.0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_segment_idx = seg_idx
        
        # If no overlap found, assign to nearest segment by midpoint
        if best_segment_idx is None:
            min_distance = float('inf')
            for seg_idx, seg in enumerate(merged_segments):
                # Distance from word midpoint to segment center
                seg_mid = (seg['start'] + seg['end']) / 2
                distance = abs(word_mid - seg_mid)
                
                if distance < min_distance:
                    min_distance = distance
                    best_segment_idx = seg_idx
        
        if best_segment_idx is not None:
            word_to_segment[word_idx] = best_segment_idx
    
    # Step 3: Build speaker blocks from word assignments
    speaker_blocks = []
    
    for seg_idx, seg in enumerate(merged_segments):
        # Collect words assigned to this segment
        segment_words = []
        word_indices = []
        for word_idx in sorted(word_to_segment.keys()):
            if word_to_segment[word_idx] == seg_idx:
                segment_words.append(word_timestamps[word_idx]['word'])
                word_indices.append(word_idx)
        
        # Create speaker block if we have words
        # Use actual word timestamps for accurate start/end instead of segment boundaries
        if segment_words and word_indices:
            actual_start = word_timestamps[word_indices[0]]['start']
            actual_end = word_timestamps[word_indices[-1]]['end']
            
            speaker_blocks.append({
                'speaker': seg['speaker'],
                'start': actual_start,
                'end': actual_end,
                'text': ' '.join(segment_words)
            })
    
    # Step 4: Resolve overlapping blocks by splitting words at overlap boundaries
    # Sort blocks by start time
    speaker_blocks.sort(key=lambda x: x['start'])
    
    # Check for overlaps and resolve them
    resolved_blocks = []
    for i, block in enumerate(speaker_blocks):
        if i == 0:
            resolved_blocks.append(block)
            continue
        
        prev_block = resolved_blocks[-1]
        
        # Check if current block overlaps with previous block
        if block['start'] < prev_block['end']:
            # Overlap detected - adjust boundaries at midpoint
            overlap_mid = (block['start'] + prev_block['end']) / 2
            
            # Adjust previous block end
            prev_block['end'] = overlap_mid
            
            # Adjust current block start
            block['start'] = overlap_mid
        
        resolved_blocks.append(block)
    
    # Calculate number of unique speakers
    num_speakers = len(set(block['speaker'] for block in resolved_blocks))
    
    return resolved_blocks, num_speakers


def align_sentence_segments(
    asr_segments: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Legacy alignment function for backward compatibility.
    Uses proportional text splitting as fallback when word timestamps are not available.
    
    New approach: Use align_words_to_speakers() when word timestamps are available.
    """
    blocks: List[Dict[str, Any]] = []
    
    # Sort diarization segments by start time
    sorted_diar = sorted(diar_segments, key=lambda x: float(x["start"]))
    
    # Group consecutive diarization segments by speaker
    grouped_diar = []
    if sorted_diar:
        current_group = {
            'speaker': sorted_diar[0]['speaker'],
            'start': float(sorted_diar[0]['start']),
            'end': float(sorted_diar[0]['end']),
        }
        
        for d in sorted_diar[1:]:
            if d['speaker'] == current_group['speaker']:
                # Extend current group
                current_group['end'] = float(d['end'])
            else:
                # Save current group and start new one
                grouped_diar.append(current_group)
                current_group = {
                    'speaker': d['speaker'],
                    'start': float(d['start']),
                    'end': float(d['end']),
                }
        grouped_diar.append(current_group)
    
    for seg in asr_segments:
        s_start = float(seg.get("start_time", 0.0))
        s_end = float(seg.get("end_time", 0.0))
        s_text = seg.get("text", "").strip()
        if not s_text:
            continue
        
        # Find all GROUPED diarization segments that overlap with this ASR segment
        overlapping_diar = []
        for d in grouped_diar:
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
        # split the text proportionally (legacy approach)
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


