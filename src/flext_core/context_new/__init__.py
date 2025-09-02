"""New typed context package.

Provides typed config + a drop-in compatible alias for the legacy API so teams can
opt-in gradually without touching call sites. Importing `FlextContext` from this
package yields the same public surface as `flext_core.context.FlextContext`.
"""

from flext_core.context_new.config import FlextContextConfig
from flext_core.context_new.core import FlextContextCore
from flext_core.context import FlextContext  # Compatibility alias (full API)

__all__ = [
    "FlextContextConfig",
    "FlextContextCore",
    "FlextContext",
]
