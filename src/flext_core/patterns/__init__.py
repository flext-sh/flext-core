"""FLEXT Core Patterns - Unified Patterns.

Standardized patterns for handlers, commands, validation, logging,
and fields that FLEXT libraries can use consistently across the ecosystem.
"""

from __future__ import annotations

# Command Pattern
from flext_core.patterns.commands import FlextCommand
from flext_core.patterns.commands import FlextCommandBus
from flext_core.patterns.commands import FlextCommandHandler
from flext_core.patterns.commands import FlextCommandResult

# Field System
from flext_core.patterns.fields import FlextField
from flext_core.patterns.fields import FlextFieldMetadata
from flext_core.patterns.fields import FlextFieldType

# Handler Pattern
from flext_core.patterns.handlers import FlextEventHandler
from flext_core.patterns.handlers import FlextHandler
from flext_core.patterns.handlers import FlextMessageHandler
from flext_core.patterns.handlers import FlextRequestHandler

# Logging System
from flext_core.patterns.logging import FlextLogContext
from flext_core.patterns.logging import FlextLogger
from flext_core.patterns.logging import FlextLoggerFactory
from flext_core.patterns.logging import FlextLogLevel
from flext_core.patterns.logging import configure_logging
from flext_core.patterns.logging import get_logger

# Type Definitions
from flext_core.patterns.typedefs import FlextCommandId
from flext_core.patterns.typedefs import FlextFieldId
from flext_core.patterns.typedefs import FlextHandlerId
from flext_core.patterns.typedefs import FlextLoggerName
from flext_core.patterns.typedefs import FlextValidatorId

# Validation System
from flext_core.patterns.validation import FlextFieldValidator
from flext_core.patterns.validation import FlextValidationResult
from flext_core.patterns.validation import FlextValidationRule
from flext_core.patterns.validation import FlextValidator

__all__ = [
    # Command Pattern
    "FlextCommand",
    "FlextCommandBus",
    "FlextCommandHandler",
    "FlextCommandId",
    "FlextCommandResult",
    "FlextEventHandler",
    # Field System
    "FlextField",
    "FlextFieldId",
    "FlextFieldMetadata",
    "FlextFieldType",
    "FlextFieldValidator",
    # Validation System
    "FlextFieldValidator",
    # Handler Pattern
    "FlextHandler",
    # Type Definitions
    "FlextHandlerId",
    "FlextLogContext",
    "FlextLogLevel",
    # Logging System
    "FlextLogger",
    "FlextLoggerFactory",
    "FlextLoggerName",
    "FlextMessageHandler",
    "FlextRequestHandler",
    "FlextValidationResult",
    "FlextValidationRule",
    "FlextValidator",
    "FlextValidatorId",
    "configure_logging",
    "get_logger",
]
