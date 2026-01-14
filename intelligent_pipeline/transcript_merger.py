from typing import List, Tuple, Dict
from .speaker_corrector import LLMSpeakerCorrector
from .config_loader import config


class TranscriptMerger:
    """
    Merges chunk-based transcriptions with diarization results.
    Uses LLM to assign Agent/Customer labels instead of speaker_0/speaker_1.
    """
    
    def __init__(self, triton_url: str = None):
        """
        Args:
            triton_url: Triton server URL for LLM inference (default: from config)
        """
        # Load configuration if not provided
        if triton_url is None:
            service_config = config.get_service_config('speaker_correction')
            triton_url = service_config.get('url', 'localhost:3701')
        
        self.llm_corrector = LLMSpeakerCorrector(triton_url=triton_url)
        print("[Merger] Initialized with LLM-based Agent/Customer labeling")
    
    @staticmethod
    def _align_words_with_diarization(
        word_timestamps: List[Dict],
        diarization_segments: List[Dict],
        chunk_offset: float
    ) -> List[Dict]:
        """
        Align word timestamps with diarization segments.
        
        Args:
            word_timestamps: List of {'text': str, 'start': float, 'end': float}
            diarization_segments: List of {'speaker': str, 'start': float, 'end': float}
            chunk_offset: Offset in seconds for this chunk in the full audio
            
        Returns:
            List of aligned words with speaker labels
        """
        if not diarization_segments:
            # No diarization available - assign default speaker
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
            
            # Find the speaker segment that best matches this word
            best_speaker = None
            max_overlap_ratio = 0
            
            # First pass: Find best overlap
            for seg in diarization_segments:
                # Calculate overlap between word and segment
                overlap_start = max(word_start, seg['start'])
                overlap_end = min(word_end, seg['end'])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > 0:
                    overlap_ratio = overlap / word_duration if word_duration > 0 else 0
                    if overlap_ratio > max_overlap_ratio:
                        max_overlap_ratio = overlap_ratio
                        best_speaker = seg['speaker']
            
            # If no overlap found (word in gap between segments), find nearest segment
            if best_speaker is None:
                min_distance = float('inf')
                
                for seg in diarization_segments:
                    # Distance from word midpoint to segment
                    if word_mid < seg['start']:
                        distance = seg['start'] - word_mid
                    elif word_mid > seg['end']:
                        distance = word_mid - seg['end']
                    else:
                        # Word midpoint is inside segment
                        distance = 0
                        best_speaker = seg['speaker']
                        break
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_speaker = seg['speaker']
                
                # Final fallback: if still no speaker (shouldn't happen), use first speaker
                if best_speaker is None:
                    best_speaker = diarization_segments[0]['speaker']
            
            aligned_words.append({
                'text': word.get('text', word.get('word', '')),
                'start': word_start,
                'end': word_end,
                'global_start': chunk_offset + word_start,
                'global_end': chunk_offset + word_end,
                'speaker': best_speaker
            })
        
        return aligned_words
    
    @staticmethod
    def _smooth_speaker_changes(aligned_words: List[Dict], min_duration: float = 0.5) -> List[Dict]:
        """
        Smooth out rapid speaker changes (speaker ping-pong).
        
        If a speaker segment is very short (< min_duration seconds), and surrounded
        by the same other speaker, it's likely a diarization error.
        
        Args:
            aligned_words: Words with speaker labels
            min_duration: Minimum segment duration in seconds (default: 0.5s)
            
        Returns:
            Smoothed aligned words
        """
        if len(aligned_words) < 3:
            return aligned_words
        
        smoothed = aligned_words.copy()
        
        # Find segments (continuous same-speaker regions)
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
        
        # Last segment
        if current_speaker is not None:
            segments.append({
                'speaker': current_speaker,
                'start_idx': seg_start_idx,
                'end_idx': len(smoothed) - 1,
                'start_time': smoothed[seg_start_idx]['global_start'],
                'end_time': smoothed[-1]['global_end']
            })
        
        # Smooth very short segments
        for i in range(1, len(segments) - 1):
            seg = segments[i]
            duration = seg['end_time'] - seg['start_time']
            
            prev_speaker = segments[i - 1]['speaker']
            next_speaker = segments[i + 1]['speaker']
            
            # If this segment is very short AND surrounded by same speaker, merge it
            if duration < min_duration and prev_speaker == next_speaker:
                # Reassign this segment to the surrounding speaker
                for idx in range(seg['start_idx'], seg['end_idx'] + 1):
                    smoothed[idx]['speaker'] = prev_speaker
                
                print(f"[Smoother] Fixed rapid speaker change at {seg['start_time']:.2f}s "
                      f"(duration: {duration:.2f}s, {seg['speaker']} → {prev_speaker})")
        
        return smoothed
    
    @staticmethod
    def _format_with_speakers(aligned_words: List[Dict]) -> str:
        """
        Format aligned words into readable transcription with speaker labels.
        Creates new lines at speaker changes AND natural sentence boundaries.
        
        Args:
            aligned_words: Words with speaker assignments (Agent/Customer or speaker_0/speaker_1)
            
        Returns:
            Formatted transcription string with proper line breaks
        """
        if not aligned_words:
            return ""
        
        # Sentence-ending punctuation
        sentence_enders = {'.', '?', '!'}
        
        result = []
        current_speaker = None
        current_words = []
        
        for i, word_info in enumerate(aligned_words):
            speaker = word_info['speaker']
            text = word_info['text']
            
            # Check if this is a speaker change
            if speaker != current_speaker:
                # Save previous segment if exists
                if current_words and current_speaker:
                    result.append(f"[{current_speaker}] {' '.join(current_words)}")
                
                # Start new segment
                current_speaker = speaker
                current_words = [text]
            else:
                # Same speaker - add word
                current_words.append(text)
                
                # Check if this word ends with sentence-ending punctuation
                if any(text.endswith(p) for p in sentence_enders):
                    # Look ahead to see if next speaker is different
                    next_is_different_speaker = (
                        i + 1 < len(aligned_words) and 
                        aligned_words[i + 1]['speaker'] != current_speaker
                    )
                    
                    # Break at sentence boundary if next speaker is different
                    # OR if sentence is long (>8 words) to avoid super long lines
                    if next_is_different_speaker or len(current_words) > 8:
                        result.append(f"[{current_speaker}] {' '.join(current_words)}")
                        current_words = []
        
        # Add last segment
        if current_words and current_speaker:
            result.append(f"[{current_speaker}] {' '.join(current_words)}")
        
        return '\n'.join(result)
    
    async def merge_transcriptions(
        self,
        request_id,
        transcription_results: List[Dict],
        diarization_results: List[Dict],
        chunk_timings: List[Tuple[float, float]]
    ) -> Tuple[str, str]:
        """
        Merge transcriptions with diarization results and assign Agent/Customer labels.
        
        Args:
            transcription_results: List of {'text': str, 'word_timestamps': [...]}
            diarization_results: List of {'segments': [{'speaker': str, 'start': float, 'end': float}]}
            chunk_timings: List of (start_time, end_time) for each chunk
            
        Returns:
            (raw_transcription, labeled_transcription)
            where labeled_transcription has [Agent] and [Customer] labels
        """
        all_raw_text = []
        all_aligned_words = []
        
        for i, (trans, diar, (chunk_start, chunk_end)) in enumerate(zip(
            transcription_results, 
            diarization_results, 
            chunk_timings
        )):
            # Extract text
            text = trans.get('text', '')
            all_raw_text.append(text)
            
            # Extract word timestamps and diarization segments
            word_timestamps = trans.get('word_timestamps', [])
            diarization_segments = diar.get('segments', [])
            
            # Debug: Log chunk info
            print(f"[Chunk {i}] Time: {chunk_start:.1f}s-{chunk_end:.1f}s, "
                  f"Words: {len(word_timestamps)}, Diar segments: {len(diarization_segments)}")
            
            if word_timestamps and diarization_segments:
                # Align words with diarization
                aligned_words = self._align_words_with_diarization(
                    word_timestamps,
                    diarization_segments,
                    chunk_start
                )
                all_aligned_words.extend(aligned_words)
            elif word_timestamps:
                # No diarization for this chunk - use default speaker
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
        
        # Create raw transcription
        raw_transcription = ' '.join(all_raw_text)
        
        # Apply speaker smoothing to fix rapid ping-ponging
        if all_aligned_words:
            print("[Merger] Smoothing rapid speaker changes...")
            all_aligned_words = self._smooth_speaker_changes(all_aligned_words, min_duration=0.5)
        
        # Create initial labeled transcription with speaker_0/speaker_1
        labeled_transcription = self._format_with_speakers(all_aligned_words)
        
        # Apply LLM to convert speaker_0/speaker_1 to Agent/Customer
        if all_aligned_words:
            print("[Merger] Converting speaker labels to Agent/Customer using LLM...")
            
            labeled_transcription, all_aligned_words = await self.llm_corrector.assign_agent_customer_labels(
                labeled_transcription,
                request_id,
                all_aligned_words,
                sample_size=20  # Analyze first 20 lines
            )
            
            # Debug: Count final speaker distribution
            speaker_counts = {}
            for word in all_aligned_words:
                spk = word['speaker']
                speaker_counts[spk] = speaker_counts.get(spk, 0) + 1
            print(f"[Merge] Final speaker distribution: {speaker_counts}")
        
        return raw_transcription, labeled_transcription