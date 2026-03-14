# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Phase modules for pyproject dependency detector standardization."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from flext_infra.deps._phases.consolidate_groups import ConsolidateGroupsPhase
    from flext_infra.deps._phases.ensure_coverage import EnsureCoverageConfigPhase
    from flext_infra.deps._phases.ensure_extra_paths import EnsureExtraPathsPhase
    from flext_infra.deps._phases.ensure_formatting import EnsureFormattingToolingPhase
    from flext_infra.deps._phases.ensure_mypy import EnsureMypyConfigPhase
    from flext_infra.deps._phases.ensure_namespace import EnsureNamespaceToolingPhase
    from flext_infra.deps._phases.ensure_pydantic_mypy import (
        EnsurePydanticMypyConfigPhase,
    )
    from flext_infra.deps._phases.ensure_pyrefly import EnsurePyreflyConfigPhase
    from flext_infra.deps._phases.ensure_pyright import EnsurePyrightConfigPhase
    from flext_infra.deps._phases.ensure_pytest import EnsurePytestConfigPhase
    from flext_infra.deps._phases.ensure_ruff import EnsureRuffConfigPhase
    from flext_infra.deps._phases.inject_comments import InjectCommentsPhase

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ConsolidateGroupsPhase": (
        "flext_infra.deps._phases.consolidate_groups",
        "ConsolidateGroupsPhase",
    ),
    "EnsureCoverageConfigPhase": (
        "flext_infra.deps._phases.ensure_coverage",
        "EnsureCoverageConfigPhase",
    ),
    "EnsureExtraPathsPhase": (
        "flext_infra.deps._phases.ensure_extra_paths",
        "EnsureExtraPathsPhase",
    ),
    "EnsureFormattingToolingPhase": (
        "flext_infra.deps._phases.ensure_formatting",
        "EnsureFormattingToolingPhase",
    ),
    "EnsureMypyConfigPhase": (
        "flext_infra.deps._phases.ensure_mypy",
        "EnsureMypyConfigPhase",
    ),
    "EnsureNamespaceToolingPhase": (
        "flext_infra.deps._phases.ensure_namespace",
        "EnsureNamespaceToolingPhase",
    ),
    "EnsurePydanticMypyConfigPhase": (
        "flext_infra.deps._phases.ensure_pydantic_mypy",
        "EnsurePydanticMypyConfigPhase",
    ),
    "EnsurePyreflyConfigPhase": (
        "flext_infra.deps._phases.ensure_pyrefly",
        "EnsurePyreflyConfigPhase",
    ),
    "EnsurePyrightConfigPhase": (
        "flext_infra.deps._phases.ensure_pyright",
        "EnsurePyrightConfigPhase",
    ),
    "EnsurePytestConfigPhase": (
        "flext_infra.deps._phases.ensure_pytest",
        "EnsurePytestConfigPhase",
    ),
    "EnsureRuffConfigPhase": (
        "flext_infra.deps._phases.ensure_ruff",
        "EnsureRuffConfigPhase",
    ),
    "InjectCommentsPhase": (
        "flext_infra.deps._phases.inject_comments",
        "InjectCommentsPhase",
    ),
}

__all__ = [
    "ConsolidateGroupsPhase",
    "EnsureCoverageConfigPhase",
    "EnsureExtraPathsPhase",
    "EnsureFormattingToolingPhase",
    "EnsureMypyConfigPhase",
    "EnsureNamespaceToolingPhase",
    "EnsurePydanticMypyConfigPhase",
    "EnsurePyreflyConfigPhase",
    "EnsurePyrightConfigPhase",
    "EnsurePytestConfigPhase",
    "EnsureRuffConfigPhase",
    "InjectCommentsPhase",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
