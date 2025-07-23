"""FLEXT Core Type Definitions - Unified Type System.

Standardized type aliases for consistent typing across FLEXT ecosystem.
"""

from __future__ import annotations

from typing import Any
from typing import NewType

# =============================================================================
# PATTERN TYPE ALIASES - Standardized types for enterprise
# patterns
# =============================================================================

# Handler System Types
FlextHandlerId = NewType("FlextHandlerId", str)
FlextHandlerName = NewType("FlextHandlerName", str)
FlextMessageType = NewType("FlextMessageType", str)

# Command Pattern Types
FlextCommandId = NewType("FlextCommandId", str)
FlextCommandName = NewType("FlextCommandName", str)
FlextCommandType = NewType("FlextCommandType", str)

# Validation System Types
FlextValidatorId = NewType("FlextValidatorId", str)
FlextValidatorName = NewType("FlextValidatorName", str)
FlextRuleName = NewType("FlextRuleName", str)

# Field System Types
FlextFieldId = NewType("FlextFieldId", str)
FlextFieldName = NewType("FlextFieldName", str)
FlextFieldPath = NewType("FlextFieldPath", str)

# Logging System Types
FlextLoggerName = NewType("FlextLoggerName", str)
FlextLoggerContext = NewType("FlextLoggerContext", str)
FlextLogTag = NewType("FlextLogTag", str)

# Generic System Types
FlextPatternName = NewType("FlextPatternName", str)
FlextPatternId = NewType("FlextPatternId", str)
FlextMetadataKey = NewType("FlextMetadataKey", str)

# Data Processing Types
FlextDataPath = NewType("FlextDataPath", str)
FlextDataKey = NewType("FlextDataKey", str)
FlextDataValue = Any

# =============================================================================
# EXPORTS - Clean public API
# =============================================================================

__all__ = [
    # Command Pattern Types
    "FlextCommandId",
    "FlextCommandName",
    "FlextCommandType",
    "FlextDataKey",
    # Data Processing Types
    "FlextDataPath",
    "FlextDataValue",
    # Field System Types
    "FlextFieldId",
    "FlextFieldName",
    "FlextFieldPath",
    # Handler System Types
    "FlextHandlerId",
    "FlextHandlerName",
    "FlextLogTag",
    "FlextLoggerContext",
    # Logging System Types
    "FlextLoggerName",
    "FlextMessageType",
    "FlextMetadataKey",
    "FlextPatternId",
    # Generic System Types
    "FlextPatternName",
    "FlextRuleName",
    # Validation System Types
    "FlextValidatorId",
    "FlextValidatorName",
]
