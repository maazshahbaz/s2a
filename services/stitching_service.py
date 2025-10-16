"""
Stitching service for combining chunk transcriptions with overlap removal.
Overlaps are removed during stitching, not during processing.
"""

import re
from typing import List, Tuple
from difflib import SequenceMatcher
from loguru import logger
from .chunk_metadata import ChunkResult


class StitchingService:
    """
    Service for stitching chunk transcriptions together.

    Key features:
    - Removes overlapping text between chunks
    - Preserves word boundaries
    - Handles various overlap scenarios
    - Maintains transcription accuracy
    """

    def __init__(self,
                 words_per_second: float = 3.0,
                 overlap_similarity_threshold: float = 0.8):
        """
        Initialize stitching service with config values.

        Args:
            words_per_second: Average speaking rate for overlap estimation
            overlap_similarity_threshold: Minimum similarity for fuzzy matching
        """
        self.words_per_second = words_per_second
        self.overlap_similarity_threshold = overlap_similarity_threshold

    async def stitch_transcriptions(
        self,
        chunk_results: List[ChunkResult],
        remove_overlap: bool = True
    ) -> str:
        """
        Stitch chunk transcriptions together.

        Args:
            chunk_results: List of chunk results sorted by chunk_index
            remove_overlap: Whether to remove overlapping text

        Returns:
            Final stitched transcription
        """
        if not chunk_results:
            return ""

        # Sort by chunk index to ensure correct order
        chunk_results.sort(key=lambda x: x.chunk_index)

        if len(chunk_results) == 1:
            # Single chunk, no stitching needed
            return chunk_results[0].text.strip()

        # Stitch chunks together
        if remove_overlap:
            return self._stitch_with_overlap_removal(chunk_results)
        else:
            # Simple concatenation (no overlap removal)
            return " ".join(chunk.text.strip() for chunk in chunk_results)

    def _stitch_with_overlap_removal(self, chunk_results: List[ChunkResult]) -> str:
        """
        Stitch chunks with intelligent overlap removal.

        The overlap was added during chunking to ensure we don't lose words
        at chunk boundaries. Now we remove the duplicated text.
        """
        final_text = []

        for i, chunk in enumerate(chunk_results):
            text = chunk.text.strip()

            if i == 0:
                # First chunk - no overlap at start
                final_text.append(text)

            else:
                # Find and remove overlap with previous chunk
                prev_text = final_text[-1] if final_text else ""

                # Get overlap duration in seconds
                overlap_seconds = chunk.overlap_start

                if overlap_seconds > 0 and prev_text and text:
                    # Find overlapping text
                    overlap_text = self._find_overlap(
                        prev_text,
                        text,
                        overlap_seconds,
                        chunk.duration
                    )

                    if overlap_text:
                        # Remove overlap from current chunk
                        text = self._remove_overlap_from_text(
                            text,
                            overlap_text
                        )

                # Append processed text
                if text:
                    final_text.append(text)

        # Join all chunks with space
        result = " ".join(final_text)

        # Clean up multiple spaces
        result = re.sub(r'\s+', ' ', result)

        logger.info(f"Stitched {len(chunk_results)} chunks into {len(result)} chars")

        return result.strip()

    def _find_overlap(
        self,
        prev_text: str,
        curr_text: str,
        overlap_seconds: float,
        chunk_duration: float
    ) -> str:
        """
        Find overlapping text between two chunks.

        Uses multiple strategies:
        1. Exact suffix/prefix matching
        2. Fuzzy matching with SequenceMatcher
        3. Word-boundary matching
        """
        # Estimate overlap length (from config)
        expected_overlap_words = int(overlap_seconds * self.words_per_second)

        # Get potential overlap regions
        prev_words = prev_text.split()
        curr_words = curr_text.split()

        if len(prev_words) < expected_overlap_words or len(curr_words) < expected_overlap_words:
            # Not enough words for expected overlap
            return ""

        # Look for overlap in the last N words of prev and first N words of curr
        search_window = expected_overlap_words * 2  # Search wider window

        prev_suffix = " ".join(prev_words[-search_window:])
        curr_prefix = " ".join(curr_words[:search_window])

        # Try exact matching first
        overlap = self._find_exact_overlap(prev_suffix, curr_prefix)

        if not overlap:
            # Try fuzzy matching (use config threshold)
            overlap = self._find_fuzzy_overlap(
                prev_suffix,
                curr_prefix,
                min_similarity=self.overlap_similarity_threshold
            )

        return overlap

    def _find_exact_overlap(self, text1: str, text2: str) -> str:
        """
        Find exact overlapping substring between end of text1 and start of text2.
        """
        min_overlap = 10  # Minimum characters to consider as overlap
        max_overlap = min(len(text1), len(text2))

        best_overlap = ""

        for i in range(min_overlap, max_overlap + 1):
            suffix = text1[-i:]
            if text2.startswith(suffix):
                best_overlap = suffix

        return best_overlap

    def _find_fuzzy_overlap(
        self,
        text1: str,
        text2: str,
        min_similarity: float = 0.8
    ) -> str:
        """
        Find fuzzy overlapping text using sequence matching.
        Handles minor transcription differences between chunks.
        """
        # Use SequenceMatcher to find longest common subsequence
        matcher = SequenceMatcher(None, text1, text2)
        match = matcher.find_longest_match(
            len(text1) - len(text1) // 2,  # Search in second half of text1
            0,  # Start of text2
            len(text1),
            len(text2) // 2  # Search in first half of text2
        )

        if match.size > 20 and matcher.ratio() >= min_similarity:
            # Found significant fuzzy match
            return text1[match.a:match.a + match.size]

        return ""

    def _remove_overlap_from_text(self, text: str, overlap: str) -> str:
        """
        Remove overlap from the beginning of text.
        Preserves word boundaries.
        """
        if not overlap:
            return text

        # Try to remove exact overlap
        if text.startswith(overlap):
            return text[len(overlap):].lstrip()

        # Try fuzzy removal (in case of minor differences)
        overlap_words = overlap.split()
        text_words = text.split()

        # Find where overlap ends in text
        for i in range(len(text_words)):
            window = text_words[i:i + len(overlap_words)]
            if len(window) == len(overlap_words):
                # Check similarity
                similarity = SequenceMatcher(
                    None,
                    " ".join(overlap_words),
                    " ".join(window)
                ).ratio()

                if similarity >= self.overlap_similarity_threshold:
                    # Found overlap position, remove it (using config threshold)
                    return " ".join(text_words[i + len(overlap_words):])

        # If no clear overlap found, return original text
        return text

    def calculate_confidence(self, chunk_results: List[ChunkResult]) -> float:
        """Calculate overall confidence from chunk confidences"""
        if not chunk_results:
            return 0.0

        # Weighted average by duration
        total_duration = sum(c.duration for c in chunk_results)
        if total_duration == 0:
            return 0.0

        weighted_confidence = sum(
            c.confidence * c.duration for c in chunk_results
        )

        return weighted_confidence / total_duration

    def calculate_rtf(self, chunk_results: List[ChunkResult]) -> float:
        """Calculate overall RTF from chunk RTFs"""
        if not chunk_results:
            return 0.0

        # Average RTF across all chunks
        rtfs = [c.rtf for c in chunk_results if c.rtf > 0]
        return sum(rtfs) / len(rtfs) if rtfs else 0.0