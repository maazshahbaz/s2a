from typing import List, Tuple, Dict, Optional
from .speaker_corrector import LLMSpeakerCorrector
from .config_loader import config


class TranscriptMerger:
    """
    Merges chunk-based transcriptions with diarization results.
    Uses LLM to:
    1. Detect IVR/automated system messages
    2. Assign Agent/Customer labels to human speakers
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
    
    # ==================== MAIN MERGE FUNCTION ====================
    
    async def merge_transcriptions(
        self,
        request_id,
        transcription_results: List[Dict],
        diarization_results: List[Dict],
        chunk_timings: List[Tuple[float, float]]
    ) -> Tuple[str, str]:
        """
        Main function to merge transcriptions with diarization and apply LLM labeling.
        
        Args:
            request_id: Unique request identifier
            transcription_results: List of transcription results per chunk
            diarization_results: List of diarization results per chunk
            chunk_timings: List of (start, end) tuples for each chunk
            
        Returns:
            (raw_transcription, labeled_transcription)
        """
        all_raw_text = []
        all_aligned_words = []
        
        # Count unique speakers from diarization
        all_speakers_in_diarization = set()
        for diar in diarization_results:
            for seg in diar.get('segments', []):
                all_speakers_in_diarization.add(seg['speaker'])
        
        num_total_speakers = len(all_speakers_in_diarization)
        print(f"[Merger] Total unique speakers in diarization: {num_total_speakers} "
              f"({all_speakers_in_diarization})")
        
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
        
        # Step 2: Renumber speakers sequentially
        print("[Merger] Renumbering speakers sequentially...")
        all_aligned_words = self._renumber_speakers_sequentially(all_aligned_words)
        
        # Step 3: Apply speaker smoothing (before LLM analysis)
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
        
        # Step 4: Final renumbering before LLM
        print("[Merger] Final speaker renumbering for LLM...")
        all_aligned_words = self._renumber_speakers_sequentially(all_aligned_words)
        
        if all_aligned_words:
            pre_llm_speakers = {}
            for word in all_aligned_words:
                spk = word['speaker']
                pre_llm_speakers[spk] = pre_llm_speakers.get(spk, 0) + 1
            print(f"[Merger] Speaker distribution BEFORE LLM: {pre_llm_speakers}")
        
        # Step 5: Create formatted transcript for LLM analysis
        labeled_transcription = self._format_with_speakers(all_aligned_words)
        
        # Step 6: Apply LLM to detect IVR and assign Agent/Customer labels
        if all_aligned_words:
            print("[Merger] Analyzing transcript with LLM (IVR detection + speaker roles)...")
            
            labeled_transcription, all_aligned_words, analysis_result = await self.llm_corrector.analyze_transcript(
                labeled_transcription,
                request_id,
                all_aligned_words,
                sample_size=30
            )
            
            print(labeled_transcription)
            
            # Log results
            if analysis_result.get('has_ivr', False):
                print("[Merger] LLM detected IVR in transcript")
            else:
                print("[Merger] No IVR detected by LLM")
            
            # Final speaker count
            speaker_counts = {}
            for word in all_aligned_words:
                spk = word['speaker']
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            print(f"[Merger] Final speaker distribution: {speaker_counts}")
        
        return raw_transcription, labeled_transcription