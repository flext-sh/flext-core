"""Check services for quality gate execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.check.services import (
        DEFAULT_GATES,
        FlextInfraConfigFixer,
        FlextInfraWorkspaceChecker,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_GATES": ("flext_infra.check.services", "DEFAULT_GATES"),
    "FlextInfraConfigFixer": ("flext_infra.check.services", "FlextInfraConfigFixer"),
    "FlextInfraWorkspaceChecker": (
        "flext_infra.check.services",
        "FlextInfraWorkspaceChecker",
    ),
}

__all__ = [
    "DEFAULT_GATES",
    "FlextInfraConfigFixer",
    "FlextInfraWorkspaceChecker",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
