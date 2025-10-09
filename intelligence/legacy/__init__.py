"""
Legacy Intelligence Pipeline

Simple action item extraction pipeline for backward compatibility.
This module is maintained for existing integrations but new development
should use the enhanced intelligence pipeline.

Legacy Components:
- ActionPipelineExtractor: Simple extraction engine
- TranscriptionProcessor: Batch processing utilities
"""

from .action_pipeline import (
    ActionPipelineExtractor,
    UnifiedPayload,
    QuickIntelligence,
    ActionItem,
    Entity,
    Priority,
    Sentiment,
    Intent
)

from .process_transcriptions import TranscriptionProcessor

__version__ = "1.0.0"
__all__ = [
    "ActionPipelineExtractor",
    "UnifiedPayload",
    "QuickIntelligence",
    "ActionItem",
    "Entity",
    "Priority",
    "Sentiment",
    "Intent",
    "TranscriptionProcessor"
]