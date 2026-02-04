from typing import List, Tuple, Dict, Optional
from .speaker_corrector import LLMSpeakerCorrector
from .config_loader import config


class TranscriptMerger:
    """
    Merges chunk-based transcriptions with diarization results.
    Uses LLM to assign Agent/Customer labels instead of speaker_0/speaker_1.

    """
    
    # Common IVR/automated system phrases to detect
    IVR_PATTERNS = [
        # Google Voice patterns
        "please state your name",
        "google voice will try to connect you",
        "after the tone",
        "after the beep",
        "press 1",
        "press 2", 
        "press #",
        "press *",
        "press pound",
        "press star",
        # Generic IVR patterns
        "for english",
        "para español",
        "your call may be recorded",
        "your call is being recorded",
        "this call may be monitored",
        "for quality assurance",
        "please hold",
        "please wait",
        "all of our representatives are busy",
        "your call is important to us",
        "leave a message after",
        "please leave your message",
        "at the tone please record",
        "mailbox is full",
        "is not available",
        "has a voicemail box",
        "to leave a callback number",
        "thank you for calling",
        "welcome to",
        "you have reached",
        "office hours are",
        "we are currently closed",
        "main menu",
        "return to the main menu",
        "goodbye",
    ]
    
    def __init__(self, triton_url: str = None):
        if triton_url is None:
            service_config = config.get_service_config('speaker_correction')
            triton_url = service_config.get('url', 'localhost:3701')
        
        self.llm_corrector = LLMSpeakerCorrector(triton_url=triton_url)
        print("[Merger] Initialized with LLM-based Agent/Customer labeling + IVR detection")
    
    @staticmethod
    def _renumber_speakers_sequentially(aligned_words: List[Dict]) -> List[Dict]:
        if not aligned_words:
            return aligned_words
        
        unique_speakers = sorted(set(
            w['speaker'] for w in aligned_words 
            if w['speaker'] != 'IVR'
        ))
        
        expected_speakers = [f'speaker_{i}' for i in range(len(unique_speakers))]
        
        if unique_speakers == expected_speakers:
            return aligned_words
        
        speaker_mapping = {}
        for new_idx, old_speaker in enumerate(unique_speakers):
            speaker_mapping[old_speaker] = f'speaker_{new_idx}'
        
        speaker_mapping['IVR'] = 'IVR'
        
        print(f"[Renumber] Renumbering speakers: {speaker_mapping}")
        
        result = []
        for word in aligned_words:
            new_word = word.copy()
            old_speaker = new_word['speaker']
            if old_speaker in speaker_mapping:
                new_word['speaker'] = speaker_mapping[old_speaker]
                if old_speaker != new_word['speaker']:
                    new_word['original_diar_speaker'] = old_speaker
            result.append(new_word)
        
        return result
    
    @staticmethod
    def _analyze_speaker_statistics(aligned_words: List[Dict]) -> Dict[str, Dict]:
        if not aligned_words:
            return {}
        
        stats = {}
        
        for word in aligned_words:
            speaker = word['speaker']
            if speaker == 'IVR':
                continue
                
            if speaker not in stats:
                stats[speaker] = {
                    'word_count': 0,
                    'total_duration': 0.0,
                    'first_appearance': float('inf'),
                    'last_appearance': 0.0,
                    'words': [],
                    'segments': []
                }
            
            stats[speaker]['word_count'] += 1
            stats[speaker]['total_duration'] += word['global_end'] - word['global_start']
            stats[speaker]['first_appearance'] = min(
                stats[speaker]['first_appearance'], 
                word['global_start']
            )
            stats[speaker]['last_appearance'] = max(
                stats[speaker]['last_appearance'], 
                word['global_end']
            )
            stats[speaker]['words'].append(word)
        
        for speaker in stats:
            words = stats[speaker]['words']
            if not words:
                continue
            
            segments = []
            seg_start = words[0]['global_start']
            seg_end = words[0]['global_end']
            seg_word_count = 1
            
            for i in range(1, len(words)):
                gap = words[i]['global_start'] - words[i-1]['global_end']
                
                if gap > 1.0:
                    segments.append((seg_start, seg_end, seg_word_count))
                    seg_start = words[i]['global_start']
                    seg_end = words[i]['global_end']
                    seg_word_count = 1
                else:
                    seg_end = words[i]['global_end']
                    seg_word_count += 1
            
            segments.append((seg_start, seg_end, seg_word_count))
            
            stats[speaker]['segments'] = segments
            stats[speaker]['segment_count'] = len(segments)
            stats[speaker]['avg_word_duration'] = (
                stats[speaker]['total_duration'] / stats[speaker]['word_count']
                if stats[speaker]['word_count'] > 0 else 0
            )
            
            del stats[speaker]['words']
        
        return stats
    
    @staticmethod
    def _find_speaker_to_merge(
        speaker_stats: Dict[str, Dict],
        aligned_words: List[Dict]
    ) -> Tuple[str, str]:
        speakers = list(speaker_stats.keys())
        
        if len(speakers) <= 2:
            return None, None
        
        speakers_by_size = sorted(speakers, key=lambda s: speaker_stats[s]['word_count'])
        
        smallest_speaker = speakers_by_size[0]
        smallest_stats = speaker_stats[smallest_speaker]
        
        other_speakers = [s for s in speakers if s != smallest_speaker]
        
        print(f"\n[Consolidation] Analyzing speakers for merge:")
        for spk in speakers:
            s = speaker_stats[spk]
            print(f"  {spk}: {s['word_count']} words, {s['segment_count']} segments, "
                  f"first@{s['first_appearance']:.1f}s, last@{s['last_appearance']:.1f}s")
        
        total_duration = max(s['last_appearance'] for s in speaker_stats.values())
        
        early_threshold = total_duration * 0.15
        late_threshold = total_duration * 0.85
        
        appears_only_early = smallest_stats['last_appearance'] < early_threshold
        appears_only_late = smallest_stats['first_appearance'] > late_threshold
        
        if appears_only_early:
            print(f"[Consolidation] {smallest_speaker} only appears in first 15% of call")
            merge_target = min(
                other_speakers,
                key=lambda s: speaker_stats[s]['first_appearance']
            )
            return smallest_speaker, merge_target
        
        if appears_only_late:
            print(f"[Consolidation] {smallest_speaker} only appears in last 15% of call")
            merge_target = max(
                other_speakers,
                key=lambda s: speaker_stats[s]['last_appearance']
            )
            return smallest_speaker, merge_target
        
        adjacency_count = {s: 0 for s in other_speakers}
        
        prev_speaker = None
        for word in aligned_words:
            curr_speaker = word['speaker']
            if curr_speaker == 'IVR':
                continue
            if prev_speaker == smallest_speaker and curr_speaker in other_speakers:
                adjacency_count[curr_speaker] += 1
            elif curr_speaker == smallest_speaker and prev_speaker in other_speakers:
                adjacency_count[prev_speaker] += 1
            prev_speaker = curr_speaker
        
        if any(adjacency_count.values()):
            most_adjacent = max(adjacency_count.items(), key=lambda x: x[1])
            if most_adjacent[1] > 0:
                print(f"[Consolidation] {smallest_speaker} most adjacent to {most_adjacent[0]} "
                      f"({most_adjacent[1]} transitions)")
                return smallest_speaker, most_adjacent[0]
        
        all_segments = []
        for spk, stats in speaker_stats.items():
            for seg_start, seg_end, word_count in stats['segments']:
                all_segments.append((seg_start, seg_end, spk, word_count))
        
        all_segments.sort(key=lambda x: x[0])
        
        before_counts = {s: 0 for s in other_speakers}
        after_counts = {s: 0 for s in other_speakers}
        
        for i, (start, end, spk, wc) in enumerate(all_segments):
            if spk == smallest_speaker:
                if i > 0 and all_segments[i-1][2] in other_speakers:
                    before_counts[all_segments[i-1][2]] += 1
                if i < len(all_segments) - 1 and all_segments[i+1][2] in other_speakers:
                    after_counts[all_segments[i+1][2]] += 1
        
        combined_counts = {s: before_counts[s] + after_counts[s] for s in other_speakers}
        most_surrounding = max(combined_counts.items(), key=lambda x: x[1])
        
        if most_surrounding[1] > 0:
            print(f"[Consolidation] {smallest_speaker} most surrounded by {most_surrounding[0]} "
                  f"({most_surrounding[1]} times)")
            return smallest_speaker, most_surrounding[0]
        
        merge_target = max(other_speakers, key=lambda s: speaker_stats[s]['word_count'])
        print(f"[Consolidation] Default: merging {smallest_speaker} into {merge_target} (largest)")
        return smallest_speaker, merge_target
    
    @staticmethod
    def _consolidate_speakers(
        aligned_words: List[Dict],
        speaker_to_remove: str,
        speaker_to_merge_into: str
    ) -> List[Dict]:
        result = []
        
        for word in aligned_words:
            new_word = word.copy()
            if new_word['speaker'] == speaker_to_remove:
                new_word['speaker'] = speaker_to_merge_into
                new_word['merged_from'] = speaker_to_remove
            result.append(new_word)
        
        unique_speakers = sorted(set(w['speaker'] for w in result if w['speaker'] != 'IVR'))
        speaker_mapping = {old: f'speaker_{i}' for i, old in enumerate(unique_speakers)}
        
        speaker_mapping['IVR'] = 'IVR'
        
        for word in result:
            old_speaker = word['speaker']
            if old_speaker in speaker_mapping:
                word['speaker'] = speaker_mapping[old_speaker]
        
        print(f"[Consolidation] Speaker mapping: {speaker_mapping}")
        
        return result
    
    @staticmethod
    def _should_consolidate_speakers(
        aligned_words: List[Dict],
        ivr_detected: bool,
        diarization_speaker_count: int
    ) -> bool:
        if ivr_detected:
            return False
        
        unique_speakers = set(w['speaker'] for w in aligned_words if w['speaker'] != 'IVR')
        
        return len(unique_speakers) == 3
    
    # ==================== IVR DETECTION ====================
    
    @staticmethod
    def _detect_ivr_phrase_boundaries(aligned_words: List[Dict]) -> List[Tuple[int, int, str]]:
        if not aligned_words:
            return []
        
        full_text = ' '.join(w['text'] for w in aligned_words).lower()
        
        ivr_regions = []
        
        for pattern in TranscriptMerger.IVR_PATTERNS:
            pattern_lower = pattern.lower()
            start_pos = 0
            
            while True:
                idx = full_text.find(pattern_lower, start_pos)
                if idx == -1:
                    break
                
                end_pos = idx + len(pattern_lower)
                
                char_count = 0
                start_word_idx = None
                end_word_idx = None
                
                for i, word in enumerate(aligned_words):
                    word_start_char = char_count
                    word_end_char = char_count + len(word['text'])
                    
                    if start_word_idx is None and word_end_char > idx:
                        start_word_idx = i
                    
                    if word_end_char >= end_pos:
                        end_word_idx = i
                        break
                    
                    char_count = word_end_char + 1
                
                if start_word_idx is not None and end_word_idx is not None:
                    ivr_regions.append((start_word_idx, end_word_idx, pattern))
                
                start_pos = end_pos
        
        if not ivr_regions:
            return []
        
        ivr_regions.sort(key=lambda x: x[0])
        merged = [ivr_regions[0]]
        
        for region in ivr_regions[1:]:
            last = merged[-1]
            if region[0] <= last[1] + 1:
                merged[-1] = (last[0], max(last[1], region[1]), last[2] + " + " + region[2])
            else:
                merged.append(region)
        
        return merged
    
    @staticmethod
    def _expand_ivr_regions_to_speaker_segments(
        aligned_words: List[Dict],
        ivr_regions: List[Tuple[int, int, str]]
    ) -> List[Tuple[int, int, str]]:
        """
        Expand IVR regions to cover full contiguous speaker segments.
        
        Logic:
        - For each IVR region, find the original speaker of those words
        - Extend the region to cover all contiguous words by that same speaker
          that are adjacent to or between IVR matches
        - Bridge gaps between nearby IVR regions from the same speaker
        
        This ensures that "Thank you for calling MTC home of the Fixture-Free 
        Guarantee. Please wait while we answer your call." gets fully tagged as 
        IVR, not just the "thank you for calling" and "please wait" fragments.
        """
        if not ivr_regions or not aligned_words:
            return ivr_regions
        
        # Step 1: Find the dominant original speaker for each IVR region
        region_speakers = []
        for start_idx, end_idx, pattern in ivr_regions:
            speaker_counts = {}
            for i in range(start_idx, min(end_idx + 1, len(aligned_words))):
                spk = aligned_words[i]['speaker']
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            dominant_speaker = max(speaker_counts, key=speaker_counts.get) if speaker_counts else None
            region_speakers.append(dominant_speaker)
        
        # Step 2: Bridge nearby IVR regions from the same speaker
        # If two IVR regions are close and share a speaker, the gap between
        # them is almost certainly part of the same IVR message
        bridged = []
        i = 0
        while i < len(ivr_regions):
            start_idx, end_idx, pattern = ivr_regions[i]
            speaker = region_speakers[i]
            
            # Look ahead to see if next region(s) can be bridged
            j = i + 1
            while j < len(ivr_regions):
                next_start, next_end, next_pattern = ivr_regions[j]
                next_speaker = region_speakers[j]
                
                # Check if gap between regions is all from the same speaker
                gap_same_speaker = True
                if next_start > end_idx + 1:
                    for k in range(end_idx + 1, next_start):
                        if k < len(aligned_words) and aligned_words[k]['speaker'] != speaker:
                            gap_same_speaker = False
                            break
                
                # Bridge if: same speaker AND (gap is small OR gap is same speaker)
                gap_words = next_start - end_idx - 1
                time_gap = 0.0
                if end_idx < len(aligned_words) and next_start < len(aligned_words):
                    time_gap = (aligned_words[next_start]['global_start'] - 
                               aligned_words[end_idx]['global_end'])
                
                should_bridge = (
                    next_speaker == speaker and 
                    gap_same_speaker and
                    (gap_words <= 20 or time_gap <= 5.0)  # reasonable gap limits
                )
                
                if should_bridge:
                    end_idx = next_end
                    pattern = pattern + " + " + next_pattern
                    j += 1
                else:
                    break
            
            bridged.append((start_idx, end_idx, pattern, speaker))
            i = j
        
        # Step 3: Expand each bridged region to cover the full contiguous 
        # speaker segment from the same speaker
        expanded = []
        for start_idx, end_idx, pattern, speaker in bridged:
            # Expand backward: include contiguous words from same speaker
            new_start = start_idx
            while new_start > 0 and aligned_words[new_start - 1]['speaker'] == speaker:
                # Stop if we hit a large time gap (> 2 seconds) - likely a different utterance
                time_gap = (aligned_words[new_start]['global_start'] - 
                           aligned_words[new_start - 1]['global_end'])
                if time_gap > 2.0:
                    break
                new_start -= 1
            
            # Expand forward: include contiguous words from same speaker
            new_end = end_idx
            while (new_end < len(aligned_words) - 1 and 
                   aligned_words[new_end + 1]['speaker'] == speaker):
                # Stop if we hit a large time gap (> 2 seconds)
                time_gap = (aligned_words[new_end + 1]['global_start'] - 
                           aligned_words[new_end]['global_end'])
                if time_gap > 2.0:
                    break
                new_end += 1
            
            if new_start != start_idx or new_end != end_idx:
                expanded_text = ' '.join(
                    aligned_words[k]['text'] 
                    for k in range(new_start, min(new_end + 1, len(aligned_words)))
                )
                print(f"[IVR Expand] Expanded region [{start_idx}:{end_idx}] -> "
                      f"[{new_start}:{new_end}] for speaker {speaker}")
                print(f"[IVR Expand] Full text: \"{expanded_text}\"")
            
            expanded.append((new_start, new_end, pattern))
        
        # Step 4: Final merge of any overlapping expanded regions
        if not expanded:
            return expanded
        
        expanded.sort(key=lambda x: x[0])
        final = [expanded[0]]
        
        for region in expanded[1:]:
            last = final[-1]
            if region[0] <= last[1] + 1:
                final[-1] = (last[0], max(last[1], region[1]), last[2] + " + " + region[2])
            else:
                final.append(region)
        
        return final
    
    @staticmethod
    def _split_ivr_from_human_speech(
        aligned_words: List[Dict],
        ivr_regions: List[Tuple[int, int, str]]
    ) -> List[Dict]:
        if not ivr_regions:
            return aligned_words
        
        result = [word.copy() for word in aligned_words]
        
        for start_idx, end_idx, pattern in ivr_regions:
            greeting_words = {'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening'}
            
            pre_ivr_greeting_idx = None
            if start_idx > 0:
                for check_idx in range(max(0, start_idx - 2), start_idx):
                    word_text = result[check_idx]['text'].lower().rstrip('.,!?')
                    if word_text in greeting_words:
                        pre_ivr_greeting_idx = check_idx
                        break
            
            for i in range(start_idx, end_idx + 1):
                if i < len(result):
                    original_speaker = result[i]['speaker']
                    result[i]['speaker'] = 'IVR'
                    result[i]['original_speaker'] = original_speaker
                    result[i]['ivr_pattern'] = pattern
            
            if pre_ivr_greeting_idx is not None:
                result[pre_ivr_greeting_idx]['pre_ivr_greeting'] = True
                print(f"[IVR Detection] Found greeting '{result[pre_ivr_greeting_idx]['text']}' "
                      f"at {result[pre_ivr_greeting_idx]['global_start']:.2f}s before IVR")
        
        return result
    
    # ==================== WORD ALIGNMENT ====================
    
    @staticmethod
    def _align_words_with_diarization(
        word_timestamps: List[Dict],
        diarization_segments: List[Dict],
        chunk_offset: float
    ) -> List[Dict]:
        if not diarization_segments:
            aligned_words = []
            for word in word_timestamps:
                aligned_words.append({
                    'text': word.get('text', word.get('word', '')),
                    'start': word['start'],
                    'end': word['end'],
                    'global_start': chunk_offset + word['start'],
                    'global_end': chunk_offset + word['end'],
                    'speaker': 'speaker_0'
                })
            return aligned_words
        
        aligned_words = []
        
        for word in word_timestamps:
            word_start = word['start']
            word_end = word['end']
            word_mid = (word_start + word_end) / 2
            word_duration = word_end - word_start
            
            best_speaker = None
            max_overlap_ratio = 0
            
            for seg in diarization_segments:
                overlap_start = max(word_start, seg['start'])
                overlap_end = min(word_end, seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > 0:
                    overlap_ratio = overlap / word_duration if word_duration > 0 else 0
                    if overlap_ratio > max_overlap_ratio:
                        max_overlap_ratio = overlap_ratio
                        best_speaker = seg['speaker']
            
            if best_speaker is None:
                min_distance = float('inf')
                
                for seg in diarization_segments:
                    if word_mid < seg['start']:
                        distance = seg['start'] - word_mid
                    elif word_mid > seg['end']:
                        distance = word_mid - seg['end']
                    else:
                        distance = 0
                        best_speaker = seg['speaker']
                        break
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_speaker = seg['speaker']
                
                if best_speaker is None:
                    best_speaker = diarization_segments[0]['speaker']
            
            aligned_words.append({
                'text': word.get('text', word.get('word', '')),
                'start': word_start,
                'end': word_end,
                'global_start': chunk_offset + word_start,
                'global_end': chunk_offset + word['end'],
                'speaker': best_speaker
            })
        
        return aligned_words
    
    # ==================== SPEAKER SMOOTHING ====================
    
    @staticmethod
    def _smooth_speaker_changes(
        aligned_words: List[Dict], 
        min_duration: float = 0.5,
        diarization_speaker_count: int = 2
    ) -> List[Dict]:
        if len(aligned_words) < 3:
            return aligned_words
        
        non_ivr_speakers = set(
            word['speaker'] for word in aligned_words 
            if word['speaker'] != 'IVR'
        )
        num_speakers = len(non_ivr_speakers)
        
        effective_speaker_count = max(diarization_speaker_count, num_speakers)
        
        if effective_speaker_count > 2:
            print(f"[Smoother] {effective_speaker_count} speakers detected - using conservative smoothing")
            return TranscriptMerger._smooth_speaker_changes_conservative(
                aligned_words, 
                min_duration=0.15
            )
        
        print(f"[Smoother] {effective_speaker_count} speakers detected - using standard smoothing")
        
        smoothed = [word.copy() for word in aligned_words]
        
        segments = []
        current_speaker = None
        seg_start_idx = 0
        
        for i, word in enumerate(smoothed):
            if word['speaker'] != current_speaker:
                if current_speaker is not None:
                    segments.append({
                        'speaker': current_speaker,
                        'start_idx': seg_start_idx,
                        'end_idx': i - 1,
                        'start_time': smoothed[seg_start_idx]['global_start'],
                        'end_time': smoothed[i - 1]['global_end']
                    })
                current_speaker = word['speaker']
                seg_start_idx = i
        
        if current_speaker is not None:
            segments.append({
                'speaker': current_speaker,
                'start_idx': seg_start_idx,
                'end_idx': len(smoothed) - 1,
                'start_time': smoothed[seg_start_idx]['global_start'],
                'end_time': smoothed[-1]['global_end']
            })
        
        for i in range(1, len(segments) - 1):
            seg = segments[i]
            
            if seg['speaker'] == 'IVR':
                continue
            
            duration = seg['end_time'] - seg['start_time']
            prev_speaker = segments[i - 1]['speaker']
            next_speaker = segments[i + 1]['speaker']
            
            if prev_speaker == 'IVR' or next_speaker == 'IVR':
                continue
            
            if duration < min_duration and prev_speaker == next_speaker:
                for idx in range(seg['start_idx'], seg['end_idx'] + 1):
                    smoothed[idx]['speaker'] = prev_speaker
                
                print(f"[Smoother] Fixed rapid speaker change at {seg['start_time']:.2f}s "
                      f"(duration: {duration:.2f}s, {seg['speaker']} → {prev_speaker})")
        
        return smoothed
    
    @staticmethod
    def _smooth_speaker_changes_conservative(
        aligned_words: List[Dict], 
        min_duration: float = 0.15
    ) -> List[Dict]:
        if len(aligned_words) < 3:
            return aligned_words
        
        smoothed = [word.copy() for word in aligned_words]
        
        segments = []
        current_speaker = None
        seg_start_idx = 0
        
        for i, word in enumerate(smoothed):
            if word['speaker'] != current_speaker:
                if current_speaker is not None:
                    segments.append({
                        'speaker': current_speaker,
                        'start_idx': seg_start_idx,
                        'end_idx': i - 1,
                        'start_time': smoothed[seg_start_idx]['global_start'],
                        'end_time': smoothed[i - 1]['global_end'],
                        'word_count': i - seg_start_idx
                    })
                current_speaker = word['speaker']
                seg_start_idx = i
        
        if current_speaker is not None:
            segments.append({
                'speaker': current_speaker,
                'start_idx': seg_start_idx,
                'end_idx': len(smoothed) - 1,
                'start_time': smoothed[seg_start_idx]['global_start'],
                'end_time': smoothed[-1]['global_end'],
                'word_count': len(smoothed) - seg_start_idx
            })
        
        for i in range(1, len(segments) - 1):
            seg = segments[i]
            
            if seg['speaker'] == 'IVR':
                continue
            
            duration = seg['end_time'] - seg['start_time']
            word_count = seg['word_count']
            prev_speaker = segments[i - 1]['speaker']
            next_speaker = segments[i + 1]['speaker']
            
            if prev_speaker == 'IVR' or next_speaker == 'IVR':
                continue
            
            is_glitch = (
                duration < min_duration and 
                word_count <= 1 and 
                prev_speaker == next_speaker
            )
            
            if is_glitch:
                for idx in range(seg['start_idx'], seg['end_idx'] + 1):
                    smoothed[idx]['speaker'] = prev_speaker
                
                print(f"[Smoother-Conservative] Fixed glitch at {seg['start_time']:.2f}s")
        
        return smoothed
    
    # ==================== FORMATTING ====================
    
    @staticmethod
    def _format_with_speakers(aligned_words: List[Dict]) -> str:
        if not aligned_words:
            return ""
        
        sentence_enders = {'.', '?', '!'}
        
        result = []
        current_speaker = None
        current_words = []
        
        for i, word_info in enumerate(aligned_words):
            speaker = word_info['speaker']
            text = word_info['text']
            
            if speaker != current_speaker:
                if current_words and current_speaker:
                    result.append(f"[{current_speaker}] {' '.join(current_words)}")
                
                current_speaker = speaker
                current_words = [text]
            else:
                current_words.append(text)
                
                if any(text.endswith(p) for p in sentence_enders):
                    next_is_different_speaker = (
                        i + 1 < len(aligned_words) and 
                        aligned_words[i + 1]['speaker'] != current_speaker
                    )
                    
                    if next_is_different_speaker or len(current_words) > 8:
                        result.append(f"[{current_speaker}] {' '.join(current_words)}")
                        current_words = []
        
        if current_words and current_speaker:
            result.append(f"[{current_speaker}] {' '.join(current_words)}")
        
        return '\n'.join(result)
    
    # ==================== MAIN MERGE FUNCTION ====================
    
    async def merge_transcriptions(
        self,
        request_id,
        transcription_results: List[Dict],
        diarization_results: List[Dict],
        chunk_timings: List[Tuple[float, float]]
    ) -> Tuple[str, str]:
        all_raw_text = []
        all_aligned_words = []
        
        all_speakers_in_diarization = set()
        for diar in diarization_results:
            for seg in diar.get('segments', []):
                all_speakers_in_diarization.add(seg['speaker'])
        
        num_total_speakers = len(all_speakers_in_diarization)
        print(f"[Merger] Total unique speakers in diarization: {num_total_speakers} "
              f"({all_speakers_in_diarization})")
        
        print(diarization_results)
        
        # Step 1: Align words with diarization
        for i, (trans, diar, (chunk_start, chunk_end)) in enumerate(zip(
            transcription_results, 
            diarization_results, 
            chunk_timings
        )):
            text = trans.get('text', '')
            all_raw_text.append(text)
            
            word_timestamps = trans.get('word_timestamps', [])
            diarization_segments = diar.get('segments', [])
            
            print(f"[Chunk {i}] Time: {chunk_start:.1f}s-{chunk_end:.1f}s, "
                  f"Words: {len(word_timestamps)}, Diar segments: {len(diarization_segments)}")
            
            if word_timestamps and diarization_segments:
                aligned_words = self._align_words_with_diarization(
                    word_timestamps,
                    diarization_segments,
                    chunk_start
                )
                all_aligned_words.extend(aligned_words)
            elif word_timestamps:
                print(f"[Chunk {i}] Warning: No diarization segments, using default speaker")
                for word in word_timestamps:
                    all_aligned_words.append({
                        'text': word.get('text', word.get('word', '')),
                        'start': word['start'],
                        'end': word['end'],
                        'global_start': chunk_start + word['start'],
                        'global_end': chunk_start + word['end'],
                        'speaker': 'speaker_0'
                    })
        
        raw_transcription = ' '.join(all_raw_text)
        
        # Step 2: Renumber speakers sequentially AFTER alignment
        print("[Merger] Renumbering speakers sequentially...")
        all_aligned_words = self._renumber_speakers_sequentially(all_aligned_words)
        
        if all_aligned_words:
            post_renumber_speakers = {}
            for word in all_aligned_words:
                spk = word['speaker']
                post_renumber_speakers[spk] = post_renumber_speakers.get(spk, 0) + 1
            print(f"[Merger] Speaker distribution AFTER renumbering: {post_renumber_speakers}")
        
        # Step 3: Detect IVR phrases based on content
        print("[Merger] Detecting IVR phrases based on content patterns...")
        ivr_regions = self._detect_ivr_phrase_boundaries(all_aligned_words)
        ivr_detected = len(ivr_regions) > 0
        
        if ivr_detected:
            # NEW: Expand IVR regions to cover full speaker segments
            print(f"[Merger] Found {len(ivr_regions)} initial IVR regions, expanding to full segments...")
            ivr_regions = self._expand_ivr_regions_to_speaker_segments(
                all_aligned_words, ivr_regions
            )
            
            print(f"[Merger] {len(ivr_regions)} IVR regions after expansion:")
            for start_idx, end_idx, pattern in ivr_regions:
                start_time = all_aligned_words[start_idx]['global_start']
                end_time = all_aligned_words[end_idx]['global_end']
                ivr_text = ' '.join(all_aligned_words[k]['text'] for k in range(start_idx, end_idx + 1))
                print(f"  - {start_time:.2f}s to {end_time:.2f}s: '{ivr_text}'")
            
            # Mark IVR words
            all_aligned_words = self._split_ivr_from_human_speech(all_aligned_words, ivr_regions)
        else:
            print("[Merger] No IVR phrases detected")
        
        # Step 4: Consolidate speakers if needed (3 -> 2 when no IVR)
        if self._should_consolidate_speakers(all_aligned_words, ivr_detected, num_total_speakers):
            print("\n[Merger] *** SPEAKER CONSOLIDATION REQUIRED ***")
            print("[Merger] No IVR detected but 3 speakers found - consolidating to 2...")
            
            speaker_stats = self._analyze_speaker_statistics(all_aligned_words)
            
            speaker_to_remove, speaker_to_merge_into = self._find_speaker_to_merge(
                speaker_stats, 
                all_aligned_words
            )
            
            if speaker_to_remove and speaker_to_merge_into:
                print(f"[Merger] Merging {speaker_to_remove} into {speaker_to_merge_into}")
                all_aligned_words = self._consolidate_speakers(
                    all_aligned_words,
                    speaker_to_remove,
                    speaker_to_merge_into
                )
                
                unique_after = set(w['speaker'] for w in all_aligned_words if w['speaker'] != 'IVR')
                print(f"[Merger] After consolidation: {len(unique_after)} speakers")
        
        if all_aligned_words:
            pre_smooth_speakers = {}
            for word in all_aligned_words:
                spk = word['speaker']
                pre_smooth_speakers[spk] = pre_smooth_speakers.get(spk, 0) + 1
            print(f"[Merger] Speaker distribution BEFORE smoothing: {pre_smooth_speakers}")
        
        # Step 5: Apply speaker smoothing
        if all_aligned_words:
            print("[Merger] Smoothing rapid speaker changes...")
            current_speaker_count = len(set(
                w['speaker'] for w in all_aligned_words if w['speaker'] != 'IVR'
            ))
            all_aligned_words = self._smooth_speaker_changes(
                all_aligned_words, 
                min_duration=0.5,
                diarization_speaker_count=current_speaker_count
            )
        
        # Step 6: Final renumbering to ensure speaker_0, speaker_1 for LLM
        print("[Merger] Final speaker renumbering for LLM...")
        all_aligned_words = self._renumber_speakers_sequentially(all_aligned_words)
        
        if all_aligned_words:
            post_smooth_speakers = {}
            for word in all_aligned_words:
                spk = word['speaker']
                post_smooth_speakers[spk] = post_smooth_speakers.get(spk, 0) + 1
            print(f"[Merger] Speaker distribution AFTER smoothing: {post_smooth_speakers}")
        
        # Step 7: Create formatted transcript
        labeled_transcription = self._format_with_speakers(all_aligned_words)
        
        # Step 8: Apply LLM to convert speaker labels to Agent/Customer
        if all_aligned_words:
            print("[Merger] Converting speaker labels to Agent/Customer using LLM...")
            
            labeled_transcription, all_aligned_words = await self.llm_corrector.assign_agent_customer_labels(
                labeled_transcription,
                request_id,
                all_aligned_words,
                sample_size=20
            )
            
            speaker_counts = {}
            for word in all_aligned_words:
                spk = word['speaker']
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            print(f"[Merge] Final speaker distribution: {speaker_counts}")
        
        return raw_transcription, labeled_transcription