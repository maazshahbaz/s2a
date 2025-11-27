from typing import List, Tuple, Dict
from diarization import GlobalDiarizationManager

class WordLevelDiarizationMerger:
    """
    Merges ASR transcription with GLOBAL diarization using word/segment-level timestamps
    """
    
    @staticmethod
    def align_words_with_global_diarization(
        word_timestamps: List[Dict],
        global_diar_manager: GlobalDiarizationManager,
        chunk_start_time: float = 0.0
    ) -> List[Dict]:
        """
        Align word/segment timestamps with GLOBAL speaker segments.
        
        Args:
            word_timestamps: [{'text': 'hello', 'start': 0.5, 'end': 0.8}, ...] (chunk-relative times)
            global_diar_manager: Manager with global diarization data
            chunk_start_time: Start time of this chunk in the global timeline
            
        Returns:
            [{'text': 'hello', 'start': 0.5, 'end': 0.8, 'speaker': 'SPEAKER_00', 'global_start': 10.5}, ...]
        """
        aligned_words = []
        
        for word_info in word_timestamps:
            # Convert chunk-relative time to global time
            word_start_global = chunk_start_time + word_info['start']
            word_end_global = chunk_start_time + word_info['end']
            word_mid_global = (word_start_global + word_end_global) / 2
            
            # Get speaker from global diarization
            speaker = global_diar_manager.get_speaker_at_time(word_mid_global)
            
            aligned_word = word_info.copy()
            aligned_word['speaker'] = speaker
            aligned_word['global_start'] = word_start_global
            aligned_word['global_end'] = word_end_global
            aligned_words.append(aligned_word)
        
        return aligned_words
    
    @staticmethod
    def format_transcription_with_speakers(aligned_words: List[Dict]) -> str:
        """
        Format aligned words/segments into readable transcription with speaker labels
        
        Args:
            aligned_words: Words/segments with speaker assignments
            
        Returns:
            Formatted transcription string
        """
        if not aligned_words:
            return ""
        
        result = []
        current_speaker = None
        current_text_parts = []
        
        for word_info in aligned_words:
            speaker = word_info['speaker']
            text = word_info.get('text', word_info.get('segment', ''))
            
            if speaker != current_speaker:
                # Speaker change - save previous segment
                if current_text_parts and current_speaker:
                    result.append(f"[{current_speaker}] {' '.join(current_text_parts)}")
                current_speaker = speaker
                current_text_parts = [text]
            else:
                current_text_parts.append(text)
        
        # Add last segment
        if current_text_parts and current_speaker:
            result.append(f"[{current_speaker}] {' '.join(current_text_parts)}")
        
        return '\n'.join(result)
    
    @staticmethod
    def merge_all_chunks_with_global_diarization(
        chunk_transcriptions: List[Dict],
        global_diar_manager: GlobalDiarizationManager,
        chunk_timings: List[Tuple[float, float]]
    ) -> Tuple[str, str, List[Dict]]:
        """
        Merge all chunks using GLOBAL diarization for consistent speaker labels.
        
        Args:
            chunk_transcriptions: List of {'text': '...', 'word_timestamps': [...]}
            global_diar_manager: Manager with global diarization from complete audio
            chunk_timings: Chunk start/end times in seconds
            
        Returns:
            (raw_transcription, labeled_transcription, all_aligned_words)
        """
        all_raw_text = []
        all_aligned_words = []
        
        for i, (trans_data, timing) in enumerate(zip(chunk_transcriptions, chunk_timings)):
            chunk_start = timing[0]
            
            # Extract text and word timestamps
            text = trans_data.get('text', '')
            word_timestamps = trans_data.get('word_timestamps', [])
            
            all_raw_text.append(text)
            
            # Align words with GLOBAL diarization
            if word_timestamps:
                aligned_words = WordLevelDiarizationMerger.align_words_with_global_diarization(
                    word_timestamps,
                    global_diar_manager,
                    chunk_start
                )
                all_aligned_words.extend(aligned_words)
        
        # Create formatted transcriptions
        raw_transcription = ' '.join(all_raw_text)
        labeled_transcription = WordLevelDiarizationMerger.format_transcription_with_speakers(
            all_aligned_words
        )
        
        return raw_transcription, labeled_transcription, all_aligned_words

