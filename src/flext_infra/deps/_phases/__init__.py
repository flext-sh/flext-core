"""Phase modules for pyproject dependency detector standardization."""

from __future__ import annotations

from flext_infra.deps._phases.consolidate_groups import ConsolidateGroupsPhase
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

__all__ = [
    "ConsolidateGroupsPhase",
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
