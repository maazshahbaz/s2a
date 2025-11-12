"""
Chunk metadata models for Redis-based processing.
No physical chunk files are created - only metadata.
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ChunkMetadata:
    """
    Metadata for an audio chunk (NOT the actual audio data).
    This is what gets stored in Redis queue.
    """
    chunk_id: str  # Format: "{job_id}_chunk_{index}"
    job_id: str
    audio_path: str  # Path to original full audio file
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    duration: float  # Chunk duration
    chunk_index: int  # 0, 1, 2, ...
    total_chunks: int  # Total chunks for this job
    sample_rate: int
    callback_url: Optional[str] = None
    created_at: float = None
    overlap_start: float = 0  # Overlap at start (seconds) - for stitching
    overlap_end: float = 0  # Overlap at end (seconds) - for stitching
    include_intelligence: bool = False

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        self.duration = self.end_time - self.start_time

    def to_json(self) -> str:
        """Serialize to JSON for Redis storage"""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'ChunkMetadata':
        """Deserialize from JSON"""
        if isinstance(json_str, bytes):
            json_str = json_str.decode()
        return cls(**json.loads(json_str))


@dataclass
class ChunkResult:
    """Result of processing a chunk"""
    chunk_id: str
    job_id: str
    chunk_index: int
    text: str
    confidence: float
    start_time: float
    end_time: float
    processing_time: float
    rtf: float
    overlap_start: float = 0
    overlap_end: float = 0
    include_intelligence: bool = False
    word_timestamps: Optional[list] = None  # List of {'word': str, 'start': float, 'end': float, 'word_index': int}

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'ChunkResult':
        if isinstance(json_str, bytes):
            json_str = json_str.decode()
        return cls(**json.loads(json_str))


@dataclass
class JobMetadata:
    """Metadata for an entire transcription job"""
    job_id: str
    audio_path: str
    total_chunks: int
    completed_chunks: int = 0
    sample_rate: int = 16000
    audio_duration: float = 0
    callback_url: Optional[str] = None
    created_at: float = None
    status: str = "pending"  # pending, processing, stitching, completed, failed

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'JobMetadata':
        return cls(**data)