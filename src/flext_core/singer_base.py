"""FLEXT Core Singer Base - Legacy Singer Exception Compatibility.

Backward compatibility layer providing Singer-specific exception classes that were
historically part of FLEXT Core but have been architecturally relocated to the
appropriate flext-meltano module where Singer functionality belongs.

Module Role in Architecture:
    Legacy Compatibility Layer → Singer Exceptions → Singer Ecosystem Projects

    This compatibility module provides:
    - Backward compatibility for existing Singer tap and target projects
    - Migration pathway from core library to specialized meltano module
    - Deprecation warnings to guide developers to correct module usage
    - Minimal exception stubs to prevent breaking changes during transition

Deprecation Status (v0.9.0 → 1.0.0):
    🚨 DEPRECATED: Singer functionality moved to flext-meltano.singer_base
    📋 Removal Timeline: v3.0.0 (allows 2 major versions for migration)
    ✅ Current Status: Backward compatibility maintained with deprecation warnings

Ecosystem Migration Impact:
    - 15 Singer projects (5 taps + 5 targets + 4 DBT + 1 extension) affected
    - All Singer projects should migrate imports to flext-meltano.singer_base
    - Exception hierarchy maintained for seamless migration
    - Error codes and context preserved for operational continuity

Architectural Rationale:
    Singer Protocol Base Classes belong in flext-meltano, not flext-core because:
    - Singer is a specialized data integration protocol, not core architectural pattern
    - Meltano orchestrates Singer taps/targets, making it the logical home
    - Core library should focus on foundational patterns (FlextResult, FlextContainer)
    - Separation of concerns improves maintainability and reduces coupling

Migration Path for Projects:
    # Old import (deprecated, will be removed in v3.0.0)
    from flext_core.singer_base import FlextSingerError

    # New import (recommended for all Singer projects)
    from flext_meltano.singer_base import FlextSingerError

    # Exception hierarchy and behavior remain identical

Legacy Exception Classes:
    FlextSingerError: Base Singer error with operational context
    FlextTapError: Tap-specific errors for data extraction failures
    FlextTargetError: Target-specific errors for data loading failures
    FlextTransformError: Transformation errors for data processing
    FlextSingerConnectionError: Connection failures in Singer protocols
    FlextSingerValidationError: Data validation errors in Singer pipelines

Quality Standards:
    - Deprecation warnings must be clear and actionable
    - Exception behavior must remain identical during transition
    - Migration timeline must be clearly communicated
    - Backward compatibility must be maintained until removal

See Also:
    flext-meltano.singer_base: New location for Singer functionality
    docs/TODO.md: Architectural cleanup and deprecation timeline

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import warnings

from flext_core.exceptions import FlextError

# Issue deprecation warning for any import from this module
warnings.warn(
    "flext_core.singer_base is deprecated and will be removed in v3.0. "
    "Use flext_meltano.singer_base instead for Singer functionality.",
    DeprecationWarning,
    stacklevel=2,
)


class FlextSingerError(FlextError):
    """Base Singer error with appropriate defaults."""

    def __init__(
        self,
        message: str = "Singer operation error",
        **kwargs: object,
    ) -> None:
        """Initialize Singer error with default message."""
        super().__init__(message, error_code="SINGER_ERROR", context=kwargs)


class FlextTapError(FlextSingerError):
    """Tap-specific error."""

    def __init__(self, message: str = "Tap operation error", **kwargs: object) -> None:
        """Initialize Tap error with default message."""
        kwargs["component_type"] = "tap"
        super().__init__(message, error_code="TAP_ERROR", **kwargs)


class FlextTargetError(FlextSingerError):
    """Target-specific error."""

    def __init__(
        self,
        message: str = "Target operation error",
        **kwargs: object,
    ) -> None:
        """Initialize Target error with default message."""
        kwargs["component_type"] = "target"
        super().__init__(message, error_code="TARGET_ERROR", **kwargs)


class FlextTransformError(FlextSingerError):
    """Transform-specific error."""

    def __init__(
        self,
        message: str = "Transform operation error",
        **kwargs: object,
    ) -> None:
        """Initialize Transform error with default message."""
        kwargs["component_type"] = "transform"
        super().__init__(message, error_code="TRANSFORM_ERROR", **kwargs)


# Create stubs for other Singer classes (proper subclasses)
class FlextSingerConnectionError(FlextSingerError):
    """Connection-specific Singer error."""


class FlextSingerAuthenticationError(FlextSingerError):
    """Authentication-specific Singer error."""


class FlextSingerValidationError(FlextSingerError):
    """Validation-specific Singer error."""


class FlextSingerConfigurationError(FlextSingerError):
    """Configuration-specific Singer error."""


class FlextSingerProcessingError(FlextSingerError):
    """Processing-specific Singer error."""


# Export all classes for backward compatibility
__all__ = [
    "FlextSingerAuthenticationError",
    "FlextSingerConfigurationError",
    "FlextSingerConnectionError",
    "FlextSingerError",
    "FlextSingerProcessingError",
    "FlextSingerValidationError",
    "FlextTapError",
    "FlextTargetError",
    "FlextTransformError",
]
