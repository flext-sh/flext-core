"""Import fallback patterns for FLX Core compatibility.

This module provides import fallback patterns for optional dependencies
to maintain compatibility between flx and flext naming conventions.
"""

# Import from the flext_core implementation
from flext_core.utils.import_fallback_patterns import (
    SQLALCHEMY_DEPENDENCY,
    DependencyFallback,
)

__all__ = ["DependencyFallback", "SQLALCHEMY_DEPENDENCY"]
