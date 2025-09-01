"""New configuration pipeline package.

Expose only the facade and schema at package level.
"""

from .schema_app import FlextConfigSchemaAppConfig
from .core import FlextConfigCore

__all__ = [
    "FlextConfigCore",
    "FlextConfigSchemaAppConfig",
]
