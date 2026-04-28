"""
TDAS: Tool Description Augmentation System
==========================================
Automated detection and remediation of MCP tool description anti-patterns.
"""

from tdas.core import TDAS
from tdas.models import (
    AugmentationResult,
    QualityScore,
    Severity,
    SmellReport,
    SmellType,
    ToolDescription,
)

__version__ = "1.0.0"
__all__ = [
    "TDAS",
    "ToolDescription",
    "SmellReport",
    "AugmentationResult",
    "QualityScore",
    "SmellType",
    "Severity",
]
