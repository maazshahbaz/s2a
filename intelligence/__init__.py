"""
Enhanced Intelligence Pipeline for S2A

This module provides comprehensive business intelligence extraction from transcriptions
for sales, customer support, and general business contexts.

Main Components:
- Enhanced Schema: Comprehensive data models for business intelligence
- Enhanced Extractor: Advanced extraction engine with auto-mode detection
- Intelligence Service: Async service integration with queue management
- Legacy Pipeline: Simple action item extraction (backward compatibility)
"""

# Enhanced Intelligence Pipeline (Recommended)
from .enhanced_schema import (
    EnhancedBusinessIntelligence,
    SalesIntelligence,
    SupportIntelligence,
)

from .enhanced_extractor import EnhancedExtractor
from .intelligence_service import (
    IntelligenceService,
    get_intelligence_service,
    start_intelligence_service,
    stop_intelligence_service
)

# Legacy pipeline for backward compatibility
from .legacy.action_pipeline import ActionPipelineExtractor
from .legacy.process_transcriptions import TranscriptionProcessor

__version__ = "2.0.0"
__all__ = [
    # Enhanced Pipeline
    "EnhancedBusinessIntelligence",
    "SalesIntelligence",
    "SupportIntelligence",
    "ExtractionMode",
    "EnhancedExtractor",
    "IntelligenceService",
    "get_intelligence_service",
    "start_intelligence_service",
    "stop_intelligence_service",

    # Legacy Pipeline
    "ActionPipelineExtractor",
    "TranscriptionProcessor"
]