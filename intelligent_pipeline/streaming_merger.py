from typing import Dict, List


class StreamingMerger:
    """
    Real-time merger that aligns ASR word timestamps with diarization speaker segments.
    Simplified for phone calls (2 speakers, no IVR detection).
    """

    def __init__(self):
        self._current_speaker = None

    def merge(
        self,
        asr_result: Dict,
        diar_result: Dict,
        time_offset: float = 0.0,
    ) -> List[Dict]:
        """
        Merge ASR text with diarization speaker segments.

        Args:
            asr_result: {"text": "...", "word_timestamps": [{"text": "hello", "start": 0.1, "end": 0.5}, ...]}
            diar_result: {"segments": [{"speaker": 0, "start": 0.0, "end": 2.1}, ...], "num_speakers": 2}
            time_offset: Offset in seconds from the start of the call (for absolute timestamps)

        Returns:
            List of segments: [{"speaker": "Speaker 1", "text": "hello world", "start": 0.0, "end": 2.1}]
        """
        text = asr_result.get("text", "").strip()
        if not text:
            return []

        word_timestamps = asr_result.get("word_timestamps", [])
        diar_segments = diar_result.get("segments", [])

        # If no word timestamps, return entire text with best-guess speaker
        if not word_timestamps:
            estimated_end = time_offset + len(text) * 0.05
            if diar_segments:
                speaker = self._find_speaker_at((time_offset + estimated_end) / 2.0, diar_segments)
            else:
                speaker = 0
            return [
                {
                    "speaker": f"Speaker {speaker + 1}",
                    "text": text,
                    "start": round(time_offset, 3),
                    "end": round(estimated_end, 3),  # rough estimate
                }
            ]

        # If no diarization segments, return text without speaker labels
        if not diar_segments:
            return [
                {
                    "speaker": self._current_speaker or "Speaker 1",
                    "text": text,
                    "start": round(time_offset + self._get_word_start(word_timestamps[0]), 3),
                    "end": round(time_offset + self._get_word_end(word_timestamps[-1]), 3),
                }
            ]

        # Align each word to a speaker
        merged_segments = []
        current_speaker = None
        current_text = []
        segment_start = None
        segment_end = None

        for word_info in word_timestamps:
            word_text = self._get_word_text(word_info)
            word_start = self._get_word_start(word_info)
            word_end = self._get_word_end(word_info)

            # Find which speaker was active at this word's midpoint
            word_mid = (word_start + word_end) / 2
            speaker = self._find_speaker_at(word_mid, diar_segments)

            if speaker != current_speaker and current_text:
                # Flush the current segment
                merged_segments.append(
                    {
                        "speaker": f"Speaker {current_speaker + 1}" if current_speaker is not None else "Speaker 1",
                        "text": " ".join(current_text),
                        "start": round(time_offset + segment_start, 3),
                        "end": round(time_offset + max(segment_start, segment_end or segment_start), 3),
                    }
                )
                current_text = []
                segment_start = None
                segment_end = None

            if not current_text:
                segment_start = word_start

            current_speaker = speaker
            current_text.append(word_text)
            segment_end = word_end

        # Flush remaining
        if current_text:
            merged_segments.append(
                {
                    "speaker": f"Speaker {current_speaker + 1}" if current_speaker is not None else "Speaker 1",
                    "text": " ".join(current_text),
                    "start": round(time_offset + segment_start, 3),
                    "end": round(time_offset + max(segment_start, segment_end or segment_start), 3),
                }
            )

        # Update running speaker
        if current_speaker is not None:
            self._current_speaker = f"Speaker {current_speaker + 1}"

        return merged_segments

    def _find_speaker_at(self, time_point: float, segments: List[Dict]) -> int:
        """Find which speaker was active at a given time point."""
        best_speaker = 0
        best_overlap = -1

        for seg in segments:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            speaker = seg.get("speaker", 0)

            if start <= time_point <= end:
                return speaker

            # Find nearest segment if no exact match
            distance = min(abs(time_point - start), abs(time_point - end))
            if best_overlap == -1 or distance < best_overlap:
                best_overlap = distance
                best_speaker = speaker

        return best_speaker

    def _get_dominant_speaker(self, segments: List[Dict]) -> int:
        """Get the speaker with the most total duration."""
        if not segments:
            return 0
        speaker_durations = {}
        for seg in segments:
            spk = seg.get("speaker", 0)
            dur = seg.get("end", 0) - seg.get("start", 0)
            speaker_durations[spk] = speaker_durations.get(spk, 0) + dur
        return max(speaker_durations, key=speaker_durations.get)

    def _get_word_text(self, word_info) -> str:
        if isinstance(word_info, dict):
            return word_info.get("text", word_info.get("word", ""))
        if hasattr(word_info, "text"):
            return word_info.text
        return str(word_info)

    def _get_word_start(self, word_info) -> float:
        if isinstance(word_info, dict):
            return word_info.get("start", word_info.get("start_time", 0.0))
        if hasattr(word_info, "start"):
            return word_info.start
        return 0.0

    def _get_word_end(self, word_info) -> float:
        if isinstance(word_info, dict):
            return word_info.get("end", word_info.get("end_time", 0.0))
        if hasattr(word_info, "end"):
            return word_info.end
        return 0.0
