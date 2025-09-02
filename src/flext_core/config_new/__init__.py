"""New configuration pipeline package.

Exports the new typed facade plus a compatibility alias `FlextConfig` that maps
to the legacy configuration module, allowing drop-in replacement of imports.
"""

from .schema_app import FlextConfigSchemaAppConfig
from .core import FlextConfigCore
from flext_core.config import FlextConfig  # Compatibility alias

__all__ = [
    "FlextConfigCore",
    "FlextConfigSchemaAppConfig",
    "FlextConfig",
]
