"""New typed context package.

Expose only the facade and typed config at package level.
"""

from flext_core.context_new.config import FlextContextConfig
from flext_core.context_new.core import FlextContextCore

__all__ = [
    "FlextContextConfig",
    "FlextContextCore",
]
