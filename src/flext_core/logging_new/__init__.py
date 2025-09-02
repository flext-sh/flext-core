"""New logging facade package.

Exports a compatibility alias `FlextLogger` so it can fully replace imports of
`flext_core.loggings.FlextLogger` without call-site changes. The new facade
`FlextLoggingCore` remains available for typed, functional-style operations and
mixins are exposed via `FlextLoggingMixin`.
"""

from .core import FlextLoggingCore
from .mixin import FlextLoggingMixin
from flext_core.loggings import FlextLogger  # Compatibility alias (drop-in)

__all__ = [
    "FlextLoggingCore",
    "FlextLoggingMixin",
    "FlextLogger",
]
