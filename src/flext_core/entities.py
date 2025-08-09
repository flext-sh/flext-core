"""Domain entities with identity and lifecycle management.

⚠️ DEPRECATION NOTICE: This module is deprecated as of v1.0.0.
Use flext_core.models.FlextEntity and FlextEntityFactory instead.
"""

from __future__ import annotations

import warnings

from flext_core.models import FlextEntity, FlextEntityFactory

# Issue deprecation warning when module is imported
warnings.warn(
    "flext_core.entities module is deprecated. "
    "Use 'from flext_core.models import FlextEntity, FlextEntityFactory' instead.",
    DeprecationWarning,
    stacklevel=2,
)


# Legacy aliases with warnings
def _deprecated_factory() -> FlextEntityFactory:
    """Return deprecated factory (deprecated)."""
    warnings.warn(
        "Import FlextEntityFactory from flext_core.models instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return FlextEntityFactory()


# Backward compatibility exports
__all__ = [
    "FlextEntity",  # Re-exported from models
    "FlextEntityFactory",  # Re-exported from models
]
