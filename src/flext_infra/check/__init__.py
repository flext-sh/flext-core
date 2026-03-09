# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Check services for quality gate execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.check.services import (
        FlextInfraConfigFixer,
        _CheckIssue,
        _GateExecution,
        _ProjectResult,
        _ProjectResult as r,
    )
    from flext_infra.check.workspace_check import (
        FlextInfraWorkspaceChecker,
        build_parser,
        main,
        run_cli,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextInfraConfigFixer": ("flext_infra.check.services", "FlextInfraConfigFixer"),
    "FlextInfraWorkspaceChecker": (
        "flext_infra.check.workspace_check",
        "FlextInfraWorkspaceChecker",
    ),
    "_CheckIssue": ("flext_infra.check.services", "_CheckIssue"),
    "_GateExecution": ("flext_infra.check.services", "_GateExecution"),
    "_ProjectResult": ("flext_infra.check.services", "_ProjectResult"),
    "build_parser": ("flext_infra.check.workspace_check", "build_parser"),
    "main": ("flext_infra.check.workspace_check", "main"),
    "r": ("flext_infra.check.services", "_ProjectResult"),
    "run_cli": ("flext_infra.check.workspace_check", "run_cli"),
}

__all__ = [
    "FlextInfraConfigFixer",
    "FlextInfraWorkspaceChecker",
    "_CheckIssue",
    "_GateExecution",
    "_ProjectResult",
    "build_parser",
    "main",
    "r",
    "run_cli",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
