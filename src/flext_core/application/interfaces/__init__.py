"""Application interfaces for FLEXT Core.

ARCHITECTURAL RULE: Only abstract domain interfaces allowed here.
ALL concrete technology-specific interfaces moved to their respective projects.

These interfaces define domain contracts that enterprise applications
can depend on without coupling to concrete implementations.
"""

from __future__ import annotations

# Domain interfaces for dependency injection
from flext_core.application.interfaces.ldif_services import LDIFAdapterInterface
from flext_core.application.interfaces.ldif_services import LDIFProcessorInterface
from flext_core.application.interfaces.plugin import PluginHealthResult

# ✅ CLEAN ARCHITECTURE COMPLIANCE:
# - Only abstract domain interfaces exported
# - No concrete implementations or technology coupling
# - Enterprise applications depend on these contracts
# - Concrete implementations registered via DI at runtime

__all__ = [
    "LDIFAdapterInterface",
    # LDIF services interfaces
    "LDIFProcessorInterface",
    # Plugin interfaces
    "PluginHealthResult",
]
