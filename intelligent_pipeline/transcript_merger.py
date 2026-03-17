from typing import List, Tuple, Dict, Optional
from .speaker_corrector import LLMSpeakerCorrector
from .config_loader import config


class TranscriptMerger:
    """
    Merges chunk-based transcriptions with diarization results.
    Uses LLM to:
    1. Detect IVR/automated system messages
    2. Assign Agent/Customer labels to human speakers
    
    All done in a single LLM call for efficiency.
    """
    
    def __init__(self, triton_url: str = None):
        if triton_url is None:
            service_config = config.get_service_config('speaker_correction')
            triton_url = service_config.get('url', 'localhost:3701')
        
        self.llm_corrector = LLMSpeakerCorrector(triton_url=triton_url)
        print("[Merger] Initialized with LLM-based IVR detection + Agent/Customer labeling")
    
    @staticmethod
    def _renumber_speakers_sequentially(aligned_words: List[Dict]) -> List[Dict]:
        """Renumber speakers to be sequential (speaker_0, speaker_1, etc.)"""
        if not aligned_words:
            return aligned_words
        
        # Get unique speakers excluding IVR, in order of first appearance
        seen_speakers = []
        for w in aligned_words:
            spk = w['speaker']
            if spk != 'IVR' and spk not in seen_speakers:
                seen_speakers.append(spk)
        
        # Create mapping based on order of appearance
        speaker_mapping = {old: f'speaker_{i}' for i, old in enumerate(seen_speakers)}
        speaker_mapping['IVR'] = 'IVR'
        
        # Check if renumbering is needed
        needs_renumber = any(old != new for old, new in speaker_mapping.items() if old != 'IVR')
        
        if not needs_renumber:
            return aligned_words
        
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
    
    # ==================== SINGLE-SPEAKER FALLBACK ====================
    
    @staticmethod
    def _detect_single_speaker(aligned_words: List[Dict]) -> bool:
        """Check if all words are assigned to a single speaker."""
        if not aligned_words:
            return False
        speakers = set(w['speaker'] for w in aligned_words if w['speaker'] != 'IVR')
        return len(speakers) <= 1
    
    # ==================== FIX #1: Heavily imbalanced speaker detection ====================
    # Detects when diarization produced 2+ speakers but one has <5% of words.
    # Uses 5% threshold (not 10%) to avoid false triggers on genuine one-sided calls
    # where one party gives long explanations with brief acknowledgments from the other.
    
    @staticmethod
    def _detect_heavily_imbalanced_speakers(
        aligned_words: List[Dict],
        imbalance_threshold: float = 0.05
    ) -> bool:
        """
        Detect when diarization produced 2+ speakers but one has very few words.
        
        This catches cases where SpectralClustering technically produced 2 labels
        but assigned 95%+ of embeddings to one cluster — effectively a single-speaker
        failure that _detect_single_speaker doesn't catch.
        
        Uses a conservative 5% threshold to avoid false positives on legitimate
        calls where one speaker dominates (e.g., long technical explanation with
        brief "uh-huh" responses).
        """
        if not aligned_words:
            return False
        
        speakers = {}
        for w in aligned_words:
            spk = w.get('speaker', '')
            if spk != 'IVR':
                speakers[spk] = speakers.get(spk, 0) + 1
        
        if len(speakers) < 2:
            return False  # Already handled by _detect_single_speaker
        
        total = sum(speakers.values())
        min_count = min(speakers.values())
        ratio = min_count / total
        
        if ratio < imbalance_threshold:
            print(f"[Merger] WARNING: Heavily imbalanced speakers detected! "
                  f"Distribution: {speakers} (minority ratio: {ratio:.1%})")
            return True
        return False
    
    @staticmethod
    def _apply_silence_based_speaker_splitting(
        aligned_words: List[Dict],
        min_gap_for_turn: float = 0.8,
        min_segment_words: int = 3
    ) -> List[Dict]:
        """
        FIX for single-speaker output: Split into 2 speakers using silence gaps.
        
        When diarization fails to distinguish speakers, use timing gaps between
        words as indicators of speaker turns. In telephonic conversations, speaker
        turns typically have a noticeable pause (0.5-2.0 seconds).
        """
        if not aligned_words or len(aligned_words) < 5:
            return aligned_words
        
        print("[Fallback] Applying silence-based speaker splitting...")
        
        # Step 1: Calculate gaps between consecutive words
        gaps = []
        for i in range(1, len(aligned_words)):
            gap = aligned_words[i]['global_start'] - aligned_words[i-1]['global_end']
            gaps.append({'index': i, 'gap': gap})
        
        if not gaps:
            return aligned_words
        
        # Step 2: Find significant gaps that likely indicate speaker turns
        gap_values = sorted([g['gap'] for g in gaps if g['gap'] > 0])
        if not gap_values:
            return aligned_words
        
        median_gap = gap_values[len(gap_values) // 2]
        adaptive_threshold = max(min_gap_for_turn, median_gap * 3.0)
        
        turn_points = [g['index'] for g in gaps if g['gap'] >= adaptive_threshold]
        
        # If very few turn points, lower threshold
        if len(turn_points) < 3 and gap_values:
            threshold_75 = gap_values[int(len(gap_values) * 0.75)]
            adaptive_threshold = max(0.3, threshold_75)
            turn_points = [g['index'] for g in gaps if g['gap'] >= adaptive_threshold]
        
        if not turn_points:
            print("[Fallback] No significant gaps found, cannot split speakers")
            return aligned_words
        
        print(f"[Fallback] Found {len(turn_points)} potential turn points "
              f"(threshold: {adaptive_threshold:.2f}s)")
        
        # Step 3: Create segments between turn points
        segments = []
        prev_idx = 0
        for tp in turn_points:
            if tp > prev_idx:
                segments.append((prev_idx, tp))
            prev_idx = tp
        if prev_idx < len(aligned_words):
            segments.append((prev_idx, len(aligned_words)))
        
        # Step 4: Merge very short segments into neighbors
        merged_segments = []
        for start, end in segments:
            word_count = end - start
            if word_count < min_segment_words and merged_segments:
                prev_start, prev_end = merged_segments[-1]
                merged_segments[-1] = (prev_start, end)
            else:
                merged_segments.append((start, end))
        
        # Step 5: Alternate speakers across segments
        result = [w.copy() for w in aligned_words]
        current_speaker = 0
        
        for i, (seg_start, seg_end) in enumerate(merged_segments):
            speaker = f"speaker_{current_speaker}"
            for j in range(seg_start, seg_end):
                result[j]['speaker'] = speaker
            if i < len(merged_segments) - 1:
                current_speaker = 1 - current_speaker
        
        final_speakers = set(w['speaker'] for w in result)
        if len(final_speakers) >= 2:
            spk_counts = {}
            for w in result:
                spk_counts[w['speaker']] = spk_counts.get(w['speaker'], 0) + 1
            print(f"[Fallback] Successfully split into speakers: {spk_counts}")
        else:
            print("[Fallback] Warning: Could not create 2-speaker split")
        
        return result
    
    # ==================== MISSING TIMESTAMPS HANDLER ====================
    
    @staticmethod
    def _generate_approximate_words(
        text: str,
        chunk_start: float,
        chunk_end: float,
        speaker: str = 'speaker_0'
    ) -> List[Dict]:
        """
        Generate approximate word-level entries when ASR returns text but no timestamps.
        Prevents content from being silently dropped from the labeled transcription.
        """
        if not text or not text.strip():
            return []
        
        words = text.strip().split()
        if not words:
            return []
        
        chunk_duration = chunk_end - chunk_start
        if chunk_duration <= 0:
            chunk_duration = len(words) * 0.3
        
        word_duration = chunk_duration / len(words)
        
        result = []
        for i, word_text in enumerate(words):
            word_start = i * word_duration
            word_end = (i + 1) * word_duration
            result.append({
                'text': word_text,
                'start': word_start,
                'end': word_end,
                'global_start': chunk_start + word_start,
                'global_end': chunk_start + word_end,
                'speaker': speaker,
                'approximate_timing': True
            })
        
        print(f"[Merger] Generated approximate timestamps for {len(words)} words "
              f"(chunk {chunk_start:.1f}s-{chunk_end:.1f}s)")
        
        return result
    
    # ==================== Boundary segment cleanup ====================
    
    @staticmethod
    def _fix_boundary_segments(
        aligned_words: List[Dict],
        min_words: int = 2,
        max_duration: float = 0.4
    ) -> List[Dict]:
        """
        Merge very short segments at the start/end of conversation into
        the adjacent segment. Handles stray words from diarization noise.
        """
        if len(aligned_words) < 3:
            return aligned_words
        
        result = [w.copy() for w in aligned_words]
        
        # Find the first speaker change
        first_speaker = result[0]['speaker']
        first_change_idx = None
        for i in range(1, len(result)):
            if result[i]['speaker'] != first_speaker:
                first_change_idx = i
                break
        
        if first_change_idx is not None and first_change_idx <= min_words:
            duration = result[first_change_idx - 1]['global_end'] - result[0]['global_start']
            if duration < max_duration:
                # Do NOT merge if the stray word is a phone-answering greeting.
                # "Hello?", "Yes?", "Hi!" at the very start of a call is the
                # customer picking up — merging it into the agent's channel
                # creates a spurious [Agent] Hello? as T00 and causes the
                # first-appearance renumber to swap all speaker labels.
                PHONE_GREETINGS = {
                    'hello', 'hi', 'yes', 'yeah', 'yep', 'yup', 'speaking', 'yo'
                }
                stray_text = ' '.join(
                    result[j]['text'] for j in range(first_change_idx)
                ).lower().strip().rstrip('.,?!')
                if stray_text in PHONE_GREETINGS:
                    print(
                        f"[Smoother] Keeping '{stray_text}' as separate turn "
                        f"at start (phone-answering greeting — not merged)"
                    )
                else:
                    new_speaker = result[first_change_idx]['speaker']
                    for j in range(first_change_idx):
                        result[j]['speaker'] = new_speaker
                    print(f"[Smoother] Merged {first_change_idx} stray word(s) at start "
                          f"into {new_speaker}")
        
        # Same for the end
        last_speaker = result[-1]['speaker']
        last_change_idx = None
        for i in range(len(result) - 2, -1, -1):
            if result[i]['speaker'] != last_speaker:
                last_change_idx = i + 1
                break
        
        if last_change_idx is not None and (len(result) - last_change_idx) <= min_words:
            duration = result[-1]['global_end'] - result[last_change_idx]['global_start']
            if duration < max_duration:
                new_speaker = result[last_change_idx - 1]['speaker']
                for j in range(last_change_idx, len(result)):
                    result[j]['speaker'] = new_speaker
                print(f"[Smoother] Merged {len(result) - last_change_idx} stray word(s) at end "
                      f"into {new_speaker}")
        
        return result
    
    # ==================== Anti-fragmentation ====================
    
    @staticmethod
    def _fix_mid_sentence_speaker_changes(
        aligned_words: List[Dict],
        min_segment_words: int = 4
    ) -> List[Dict]:
        """
        FIX for over-fragmentation: Prevent speaker changes mid-sentence.
        
        Short segments that occur within a continuous sentence are likely diarization
        errors. Merge them with the surrounding speaker.
        """
        if len(aligned_words) < 5:
            return aligned_words
        
        result = [w.copy() for w in aligned_words]
        sentence_enders = {'.', '?', '!'}
        
        # Build segments
        segments = []
        current_speaker = None
        seg_start_idx = 0
        
        for i, word in enumerate(result):
            if word['speaker'] != current_speaker:
                if current_speaker is not None:
                    segments.append({
                        'speaker': current_speaker,
                        'start_idx': seg_start_idx,
                        'end_idx': i - 1,
                        'word_count': i - seg_start_idx
                    })
                current_speaker = word['speaker']
                seg_start_idx = i
        
        if current_speaker is not None:
            segments.append({
                'speaker': current_speaker,
                'start_idx': seg_start_idx,
                'end_idx': len(result) - 1,
                'word_count': len(result) - seg_start_idx
            })
        
        for i in range(1, len(segments) - 1):
            seg = segments[i]
            if seg['speaker'] == 'IVR':
                continue
            if seg['word_count'] >= min_segment_words:
                continue
            
            prev_seg = segments[i - 1]
            next_seg = segments[i + 1]
            
            if prev_seg['speaker'] == 'IVR' or next_seg['speaker'] == 'IVR':
                continue
            
            prev_last_word = result[prev_seg['end_idx']]['text']
            prev_ends_sentence = any(prev_last_word.rstrip().endswith(p) for p in sentence_enders)

            # Check if this short segment itself ends a sentence — if so it is
            # a complete response (e.g. "Okay.", "Yes.", "Right.") and must NOT
            # be merged into the surrounding speaker.
            seg_last_word = result[seg['end_idx']]['text']
            seg_ends_sentence = any(seg_last_word.rstrip().endswith(p) for p in sentence_enders)

            # If previous didn't end a sentence and this is a short fragment
            # surrounded by the same speaker — merge
            if not prev_ends_sentence and prev_seg['speaker'] == next_seg['speaker']:
                for idx in range(seg['start_idx'], seg['end_idx'] + 1):
                    result[idx]['speaker'] = prev_seg['speaker']
                print(f"[Anti-Fragment] Merged {seg['word_count']}-word mid-sentence segment "
                      f"at word {seg['start_idx']}")
            elif not prev_ends_sentence and seg['word_count'] <= 2 and not seg_ends_sentence:
                # Only merge 1-2 word fragments that are genuinely mid-sentence
                # (no sentence-ender). Standalone responses like "Okay." are kept.
                for idx in range(seg['start_idx'], seg['end_idx'] + 1):
                    result[idx]['speaker'] = prev_seg['speaker']
                print(f"[Anti-Fragment] Merged {seg['word_count']}-word fragment "
                      f"at word {seg['start_idx']}")
        
        return result
    
    # ==================== WORD ALIGNMENT ====================
    
    @staticmethod
    def _align_words_with_diarization(
        word_timestamps: List[Dict],
        diarization_segments: List[Dict],
        chunk_offset: float
    ) -> List[Dict]:
        """Align transcribed words with diarization speaker segments."""
        if not diarization_segments:
            return [
                {
                    'text': word.get('text', word.get('word', '')),
                    'start': word['start'],
                    'end': word['end'],
                    'global_start': chunk_offset + word['start'],
                    'global_end': chunk_offset + word['end'],
                    'speaker': 'speaker_0'
                }
                for word in word_timestamps
            ]
        
        aligned_words = []
        
        for word in word_timestamps:
            word_start = word['start']
            word_end = word['end']
            word_mid = (word_start + word_end) / 2
            word_duration = word_end - word_start
            
            best_speaker = None
            max_overlap_ratio = 0
            
            # Find speaker with maximum overlap
            for seg in diarization_segments:
                overlap_start = max(word_start, seg['start'])
                overlap_end = min(word_end, seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > 0:
                    overlap_ratio = overlap / word_duration if word_duration > 0 else 0
                    if overlap_ratio > max_overlap_ratio:
                        max_overlap_ratio = overlap_ratio
                        best_speaker = seg['speaker']
            
            # If no overlap, find nearest segment
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
        """Smooth out rapid speaker changes that are likely diarization errors."""
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
        
        # Build segments
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
        
        # Fix short segments surrounded by same speaker
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
        
        # Additional pass - merge stray words at conversation boundaries
        smoothed = TranscriptMerger._fix_boundary_segments(smoothed, min_words=2, max_duration=0.4)
        
        return smoothed
    
    @staticmethod
    def _smooth_speaker_changes_conservative(
        aligned_words: List[Dict], 
        min_duration: float = 0.15
    ) -> List[Dict]:
        """Conservative smoothing for multi-speaker scenarios."""
        if len(aligned_words) < 3:
            return aligned_words
        
        smoothed = [word.copy() for word in aligned_words]
        
        # Build segments with word count
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
        
        # Only fix single-word glitches
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
        """Format aligned words into a speaker-labeled transcript."""
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
    
    # ==================== SHARED POST-PROCESSING ====================

    async def _apply_post_processing(
        self,
        all_aligned_words: List[Dict],
        raw_transcription: str,
        request_id: str,
    ) -> Tuple[str, str]:
        """
        Shared post-processing pipeline applied identically by both
        merge_transcriptions() and merge_global().

        Steps (unchanged from original merge_transcriptions):
          2. Renumber speakers sequentially
          2a. Single-speaker fallback splitting
          2b. Heavily-imbalanced speaker fallback
          3. Smooth rapid speaker changes
          3a. Anti-fragmentation pass
          4. Final renumbering before LLM
          5. Format transcript for LLM
          6. LLM IVR detection + Agent/Customer labeling

        Args:
            all_aligned_words: Words already aligned to global diarization segments.
                               Each entry must have 'text', 'start', 'end',
                               'global_start', 'global_end', 'speaker'.
            raw_transcription: Plain concatenated text (no speaker labels).
            request_id:        Unique identifier passed through to the LLM corrector.

        Returns:
            (raw_transcription, labeled_transcription)
        """
        # Step 2: Renumber speakers sequentially
        print("[Merger] Renumbering speakers sequentially...")
        all_aligned_words = self._renumber_speakers_sequentially(all_aligned_words)

        # Step 2a: Single-speaker fallback
        if all_aligned_words and self._detect_single_speaker(all_aligned_words):
            print("[Merger] WARNING: All words assigned to single speaker! "
                  "Applying silence-based fallback splitting...")
            all_aligned_words = self._apply_silence_based_speaker_splitting(
                all_aligned_words,
                min_gap_for_turn=0.8,
                min_segment_words=3
            )
            all_aligned_words = self._renumber_speakers_sequentially(all_aligned_words)

        # Step 2b: Heavily-imbalanced 2-speaker fallback
        if all_aligned_words and not self._detect_single_speaker(all_aligned_words):
            if self._detect_heavily_imbalanced_speakers(all_aligned_words):
                print("[Merger] Heavily imbalanced split detected — "
                      "applying silence-based fallback splitting...")
                for w in all_aligned_words:
                    if w.get('speaker') != 'IVR':
                        w['speaker'] = 'speaker_0'
                all_aligned_words = self._apply_silence_based_speaker_splitting(
                    all_aligned_words,
                    min_gap_for_turn=0.8,
                    min_segment_words=3
                )
                all_aligned_words = self._renumber_speakers_sequentially(all_aligned_words)

        # Step 3: Smooth rapid speaker changes
        if all_aligned_words:
            print("[Merger] Smoothing rapid speaker changes...")
            current_speaker_count = len(set(
                w['speaker'] for w in all_aligned_words if w['speaker'] != 'IVR'
            ))
            all_aligned_words = self._smooth_speaker_changes(
                all_aligned_words,
                min_duration=0.3,   # reduced from 0.5 — preserves short "Yes"/"Okay" turns
                diarization_speaker_count=current_speaker_count
            )

        # Step 3a: Anti-fragmentation pass
        if all_aligned_words:
            print("[Merger] Fixing mid-sentence speaker changes...")
            all_aligned_words = self._fix_mid_sentence_speaker_changes(
                all_aligned_words,
                min_segment_words=4
            )

        # Step 4: Final renumbering before LLM
        print("[Merger] Final speaker renumbering for LLM...")
        all_aligned_words = self._renumber_speakers_sequentially(all_aligned_words)

        if all_aligned_words:
            pre_llm_speakers = {}
            for word in all_aligned_words:
                spk = word['speaker']
                pre_llm_speakers[spk] = pre_llm_speakers.get(spk, 0) + 1
            print(f"[Merger] Speaker distribution BEFORE LLM: {pre_llm_speakers}")

        # Step 5: Format for LLM
        labeled_transcription = self._format_with_speakers(all_aligned_words)

        # Step 6: LLM — IVR detection + Agent/Customer labeling
        if all_aligned_words:
            print("[Merger] Analyzing transcript with LLM (IVR detection + speaker roles)...")
            labeled_transcription, all_aligned_words, analysis_result = \
                await self.llm_corrector.analyze_transcript(
                    labeled_transcription,
                    request_id,
                    all_aligned_words,
                    sample_size=30
                )
            if analysis_result.get('has_ivr', False):
                print("[Merger] LLM detected IVR in transcript")
            else:
                print("[Merger] No IVR detected by LLM")

            speaker_counts = {}
            for word in all_aligned_words:
                spk = word['speaker']
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            print(f"[Merger] Final speaker distribution: {speaker_counts}")

        return raw_transcription, labeled_transcription

    # ==================== MAIN MERGE FUNCTION (chunk-based) ====================

    async def merge_transcriptions(
        self,
        request_id,
        transcription_results: List[Dict],
        diarization_results: List[Dict],
        chunk_timings: List[Tuple[float, float]]
    ) -> Tuple[str, str]:
        """
        Chunk-based merge: each ASR chunk is aligned against its corresponding
        (already chunk-clipped) diarization result, then all chunks are
        concatenated before the shared post-processing pipeline runs.

        Use this when pipeline.py clips global diarization segments down to
        per-chunk local coordinates via _align_segments_to_global().

        Args:
            request_id:            Unique request identifier.
            transcription_results: One transcription dict per ASR chunk.
            diarization_results:   One diarization dict per chunk (local coords).
            chunk_timings:         [(start, end), ...] for each chunk in global time.

        Returns:
            (raw_transcription, labeled_transcription)
        """
        all_raw_text = []
        all_aligned_words = []

        # Log total unique speakers across all chunk-level diarization results
        all_speakers_in_diarization = set()
        for diar in diarization_results:
            for seg in diar.get('segments', []):
                all_speakers_in_diarization.add(seg['speaker'])

        print(f"[Merger] Total unique speakers in diarization: "
              f"{len(all_speakers_in_diarization)} ({all_speakers_in_diarization})")

        # Step 1: Align each chunk's words against its local diarization segments
        for i, (trans, diar, (chunk_start, chunk_end)) in enumerate(zip(
            transcription_results,
            diarization_results,
            chunk_timings
        )):
            text = trans.get('text', '')
            all_raw_text.append(text)

            word_timestamps    = trans.get('word_timestamps', [])
            diarization_segments = diar.get('segments', [])

            print(f"[Chunk {i}] Time: {chunk_start:.1f}s-{chunk_end:.1f}s, "
                  f"Words: {len(word_timestamps)}, "
                  f"Diar segments: {len(diarization_segments)}")

            if word_timestamps and diarization_segments:
                aligned_words = self._align_words_with_diarization(
                    word_timestamps,
                    diarization_segments,
                    chunk_start
                )
                all_aligned_words.extend(aligned_words)

            elif word_timestamps:
                print(f"[Chunk {i}] Warning: No diarization segments, "
                      f"using default speaker")
                for word in word_timestamps:
                    all_aligned_words.append({
                        'text':         word.get('text', word.get('word', '')),
                        'start':        word['start'],
                        'end':          word['end'],
                        'global_start': chunk_start + word['start'],
                        'global_end':   chunk_start + word['end'],
                        'speaker':      'speaker_0'
                    })

            elif text and text.strip():
                print(f"[Chunk {i}] Warning: Text exists but no word timestamps — "
                      f"generating approximate timestamps")
                default_speaker = 'speaker_0'
                if diarization_segments:
                    speaker_coverage = {}
                    for seg in diarization_segments:
                        spk = seg['speaker']
                        dur = seg.get('duration', seg['end'] - seg['start'])
                        speaker_coverage[spk] = speaker_coverage.get(spk, 0) + dur
                    if speaker_coverage:
                        default_speaker = max(speaker_coverage,
                                              key=speaker_coverage.get)

                approx_words = self._generate_approximate_words(
                    text, chunk_start, chunk_end, default_speaker
                )
                if approx_words and diarization_segments:
                    approx_words = self._align_words_with_diarization(
                        approx_words, diarization_segments, chunk_start
                    )
                all_aligned_words.extend(approx_words)

        raw_transcription = ' '.join(all_raw_text)
        return await self._apply_post_processing(
            all_aligned_words, raw_transcription, request_id
        )

    # ==================== GLOBAL MERGE FUNCTION ====================

    async def merge_global(
        self,
        request_id: str,
        transcription_results: List[Dict],
        chunk_timings: List[Tuple[float, float]],
        global_diarization: Dict,
    ) -> Tuple[str, str]:
        """
        Global merge: ASR chunks are offset to a single global timeline, then
        aligned in one pass against the full-audio diarization result.

        This eliminates the segment-clipping artifacts that occur when
        _align_segments_to_global() splits a diarization segment that spans
        a 5-minute ASR chunk boundary.  A speaker segment at 298s–302s stays
        intact; the words on either side of the seam are both matched to it.

        How it works
        ────────────
        1. For each ASR chunk, shift every word timestamp by chunk_start so
           that all words sit on a single 0→duration timeline.
        2. Call _align_words_with_diarization() once with chunk_offset=0.0,
           passing the offsetted words and the raw global diarization segments.
           Because word['start'] is already global time and chunk_offset is 0,
           the existing alignment logic (overlap pass + midpoint fallback) works
           identically — it just operates across the whole call at once instead
           of per chunk.
        3. Hand the resulting all_aligned_words to _apply_post_processing(),
           which is the identical smoothing + anti-frag + LLM chain used by
           merge_transcriptions().

        Args:
            request_id:            Unique request identifier.
            transcription_results: One transcription dict per ASR chunk
                                   (words in chunk-local time coordinates).
            chunk_timings:         [(start, end), ...] for each chunk in
                                   global seconds.  Used to offset word times.
            global_diarization:    Full-audio diarization result dict with a
                                   'segments' key containing global-time segments
                                   (exactly what model.py / diarization_client
                                   returns before any clipping).

        Returns:
            (raw_transcription, labeled_transcription)
        """
        global_segments = global_diarization.get('segments', [])

        if not global_segments:
            # Diarization returned nothing — fall back gracefully with no labels
            print(f"[Merger/Global] WARNING: global diarization returned 0 segments "
                  f"for {request_id}. Returning raw transcript only.")
            raw_parts = []
            for trans in transcription_results:
                if trans:
                    raw_parts.append(trans.get('text', ''))
            raw = ' '.join(raw_parts)
            return raw, raw

        # Log unique speakers from the global result
        unique_speakers = {seg['speaker'] for seg in global_segments}
        print(f"[Merger/Global] Global diarization: {len(global_segments)} segments, "
              f"{len(unique_speakers)} speakers {unique_speakers}")

        # ── Step 1: Build a single global word list ───────────────────────────
        # Each word's 'start'/'end' is shifted to global time.
        # 'global_start'/'global_end' are set to the same value (chunk_offset=0.0
        # in the alignment call means no second shift is applied).
        all_raw_text   = []
        global_words   = []   # words with globally-offsetted start/end

        for i, (trans, (chunk_start, chunk_end)) in enumerate(
            zip(transcription_results, chunk_timings)
        ):
            if not trans:
                print(f"[Merger/Global] Chunk {i} ({chunk_start:.1f}s–{chunk_end:.1f}s): "
                      f"no result, skipping")
                continue

            text = trans.get('text', '')
            all_raw_text.append(text)

            word_timestamps = trans.get('word_timestamps', [])

            if word_timestamps:
                for word in word_timestamps:
                    g_start = chunk_start + word['start']
                    g_end   = chunk_start + word['end']
                    global_words.append({
                        'text':  word.get('text', word.get('word', '')),
                        # start/end in global time — _align_words_with_diarization
                        # uses these for overlap matching against global segments
                        'start': g_start,
                        'end':   g_end,
                    })
                print(f"[Merger/Global] Chunk {i} ({chunk_start:.1f}s–{chunk_end:.1f}s): "
                      f"{len(word_timestamps)} words offset to global time")

            elif text and text.strip():
                # No word timestamps — distribute words evenly across the chunk
                print(f"[Merger/Global] Chunk {i}: no word timestamps, "
                      f"generating approximate global timestamps")
                words_in_text = text.strip().split()
                if words_in_text:
                    chunk_dur   = max(chunk_end - chunk_start, len(words_in_text) * 0.3)
                    word_dur    = chunk_dur / len(words_in_text)
                    for j, w in enumerate(words_in_text):
                        g_start = chunk_start + j * word_dur
                        g_end   = chunk_start + (j + 1) * word_dur
                        global_words.append({
                            'text':               w,
                            'start':              g_start,
                            'end':                g_end,
                            'approximate_timing': True,
                        })

        raw_transcription = ' '.join(all_raw_text)

        if not global_words:
            print(f"[Merger/Global] No words produced for {request_id}")
            return raw_transcription, raw_transcription

        # ── Step 2: Single alignment pass across the full timeline ────────────
        # chunk_offset=0.0 because word['start'] is already in global time.
        # _align_words_with_diarization sets:
        #   global_start = chunk_offset + word['start'] = 0 + global_time = global_time  ✓
        #   global_end   = chunk_offset + word['end']   = 0 + global_time = global_time  ✓
        print(f"[Merger/Global] Aligning {len(global_words)} words against "
              f"{len(global_segments)} global diarization segments...")

        all_aligned_words = self._align_words_with_diarization(
            word_timestamps=global_words,
            diarization_segments=global_segments,
            chunk_offset=0.0,
        )

        print(f"[Merger/Global] Alignment complete — "
              f"{len(all_aligned_words)} words assigned to speakers")

        # ── Step 3: Shared post-processing (smoothing + LLM) ──────────────────
        return await self._apply_post_processing(
            all_aligned_words, raw_transcription, request_id
        )