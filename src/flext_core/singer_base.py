"""Singer Protocol Base Classes - DEPRECATED: Moved to flext-meltano.

ARCHITECTURAL CORRECTION: This module has been moved to flext-meltano.singer_base
following proper logical hierarchy. Singer functionality belongs in Meltano module,
not in the core foundation library.

This module now provides backward compatibility imports only.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings

# Backward compatibility imports from the new proper location
try:
    from flext_meltano.singer_base import (
        FlextSingerAuthenticationError,
        FlextSingerConfigurationError,
        FlextSingerConnectionError,
        FlextSingerError,
        FlextSingerProcessingError,
        FlextSingerValidationError,
        FlextTapError,
        FlextTargetError,
        FlextTransformError,
    )

    # Issue deprecation warning
    warnings.warn(
        "Importing Singer classes from flext_core.singer_base is deprecated. "
        "Import from flext_meltano.singer_base instead. "
        "This compatibility import will be removed in v3.0.",
        DeprecationWarning,
        stacklevel=2,
    )

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

except ImportError:
    # If flext-meltano is not available, provide a minimal stub
    from flext_core.exceptions import FlextError

    class FlextSingerError(FlextError):
        """Base Singer error with appropriate defaults."""

        def __init__(
            self, message: str = "Singer operation error", **kwargs: object
        ) -> None:
            """Initialize Singer error with default message."""
            super().__init__(message, error_code="SINGER_ERROR", context=kwargs)

    class FlextTapError(FlextSingerError):
        """Tap-specific error."""

        def __init__(
            self, message: str = "Tap operation error", **kwargs: object
        ) -> None:
            """Initialize Tap error with default message."""
            kwargs["component_type"] = "tap"
            super().__init__(message, **kwargs)
            self.error_code = "TAP_ERROR"

    class FlextTargetError(FlextSingerError):
        """Target-specific error."""

        def __init__(
            self, message: str = "Target operation error", **kwargs: object
        ) -> None:
            """Initialize Target error with default message."""
            kwargs["component_type"] = "target"
            super().__init__(message, **kwargs)
            self.error_code = "TARGET_ERROR"

    class FlextTransformError(FlextSingerError):
        """Transform-specific error."""

        def __init__(
            self, message: str = "Transform operation error", **kwargs: object
        ) -> None:
            """Initialize Transform error with default message."""
            kwargs["component_type"] = "transform"
            super().__init__(message, **kwargs)
            self.error_code = "TRANSFORM_ERROR"

    # Create stubs for other Singer classes (just aliases)
    FlextSingerConnectionError = FlextSingerError
    FlextSingerAuthenticationError = FlextSingerError
    FlextSingerValidationError = FlextSingerError
    FlextSingerConfigurationError = FlextSingerError
    FlextSingerProcessingError = FlextSingerError

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
