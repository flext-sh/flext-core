"""FLEXT Mixins Package - Modular behavioral patterns.

This package provides reusable behavioral patterns organized into focused modules:
- logging: Structured logging capabilities
- timestamps: Creation and update timestamp tracking
- validation: Data validation and error tracking
- cache: Memoization and caching functionality
- identification: Entity ID management
- serialization: JSON and dictionary conversion
- state: Object lifecycle and state management
- timing: Performance tracking and timing

The main entry point is FlextMixins which provides all functionality.
"""

from __future__ import annotations

from flext_core.mixins.core import FlextMixins

from flext_core.mixins.cache import FlextCache
from flext_core.mixins.identification import FlextIdentification
from flext_core.mixins.logging import FlextLogging
from flext_core.mixins.serialization import FlextSerialization
from flext_core.mixins.state import FlextState
from flext_core.mixins.timestamps import FlextTimestamps
from flext_core.mixins.timing import FlextTiming
from flext_core.mixins.validation import FlextValidation

__all__ = [
    # Main facade class that provides everything
    "FlextMixins",
    # Individual module classes for direct access
    "FlextCache",
    "FlextIdentification",
    "FlextLogging",
    "FlextSerialization",
    "FlextState",
    "FlextTimestamps",
    "FlextTiming",
    "FlextValidation",
]
