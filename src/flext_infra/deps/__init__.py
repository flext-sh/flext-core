# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Dependency management services.

Provides the pyproject modernizer for workspace-wide dependency
synchronization and formatting.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

from flext_core import t
from flext_core.lazy import cleanup_submodule_namespace

if TYPE_CHECKING:
    from flext_infra.deps._phases import (
        ConsolidateGroupsPhase,
        EnsureFormattingToolingPhase,
        EnsureMypyConfigPhase,
        EnsureNamespaceToolingPhase,
        EnsurePydanticMypyConfigPhase,
        EnsurePyreflyConfigPhase,
        EnsurePyrightConfigPhase,
        EnsurePytestConfigPhase,
        EnsureRuffConfigPhase,
        InjectCommentsPhase,
    )
    from flext_infra.deps.detection import (
        FlextInfraDependencyDetectionService,
        FlextInfraDependencyDetectionService as s,
        build_project_report,
        classify_issues,
        discover_project_paths,
        get_current_typings_from_pyproject,
        get_required_typings,
        load_dependency_limits,
        module_to_types_package,
        run_deptry,
        run_mypy_stub_hints,
        run_pip_check,
    )
    from flext_infra.deps.detector import (
        FlextInfraRuntimeDevDependencyDetector,
    )
    from flext_infra.deps.internal_sync import FlextInfraInternalDependencySyncService
    from flext_infra.deps.modernizer import (
        FlextInfraPyprojectModernizer,
    )
    from flext_infra.deps.path_sync import main
    from flext_infra.deps.tool_config import load_tool_config

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "EnsureFormattingToolingPhase": (
        "flext_infra.deps._phases.ensure_formatting",
        "EnsureFormattingToolingPhase",
    ),
    "EnsureMypyConfigPhase": (
        "flext_infra.deps._phases.ensure_mypy",
        "EnsureMypyConfigPhase",
    ),
    "EnsurePyrightConfigPhase": (
        "flext_infra.deps._phases",
        "EnsurePyrightConfigPhase",
    ),
    "EnsureNamespaceToolingPhase": (
        "flext_infra.deps._phases.ensure_namespace",
        "EnsureNamespaceToolingPhase",
    ),
    "EnsurePytestConfigPhase": (
        "flext_infra.deps._phases",
        "EnsurePytestConfigPhase",
    ),
    "EnsurePydanticMypyConfigPhase": (
        "flext_infra.deps._phases.ensure_pydantic_mypy",
        "EnsurePydanticMypyConfigPhase",
    ),
    "EnsureRuffConfigPhase": (
        "flext_infra.deps._phases.ensure_ruff",
        "EnsureRuffConfigPhase",
    ),
    "FlextInfraDependencyDetectionService": (
        "flext_infra.deps.detection",
        "FlextInfraDependencyDetectionService",
    ),
    "FlextInfraInternalDependencySyncService": (
        "flext_infra.deps.internal_sync",
        "FlextInfraInternalDependencySyncService",
    ),
    "FlextInfraPyprojectModernizer": (
        "flext_infra.deps.modernizer",
        "FlextInfraPyprojectModernizer",
    ),
    "FlextInfraRuntimeDevDependencyDetector": (
        "flext_infra.deps.detector",
        "FlextInfraRuntimeDevDependencyDetector",
    ),
    "build_project_report": ("flext_infra.deps.detection", "build_project_report"),
    "classify_issues": ("flext_infra.deps.detection", "classify_issues"),
    "discover_project_paths": ("flext_infra.deps.detection", "discover_project_paths"),
    "get_current_typings_from_pyproject": (
        "flext_infra.deps.detection",
        "get_current_typings_from_pyproject",
    ),
    "get_required_typings": ("flext_infra.deps.detection", "get_required_typings"),
    "load_dependency_limits": ("flext_infra.deps.detection", "load_dependency_limits"),
    "load_tool_config": ("flext_infra.deps.tool_config", "load_tool_config"),
    "main": ("flext_infra.deps.path_sync", "main"),
    "module_to_types_package": (
        "flext_infra.deps.detection",
        "module_to_types_package",
    ),
    "run_deptry": ("flext_infra.deps.detection", "run_deptry"),
    "run_mypy_stub_hints": ("flext_infra.deps.detection", "run_mypy_stub_hints"),
    "run_pip_check": ("flext_infra.deps.detection", "run_pip_check"),
    "s": ("flext_infra.deps.detection", "FlextInfraDependencyDetectionService"),
}

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
    "FlextInfraDependencyDetectionService",
    "FlextInfraInternalDependencySyncService",
    "FlextInfraPyprojectModernizer",
    "FlextInfraRuntimeDevDependencyDetector",
    "InjectCommentsPhase",
    "build_project_report",
    "classify_issues",
    "discover_project_paths",
    "get_current_typings_from_pyproject",
    "get_required_typings",
    "load_dependency_limits",
    "load_tool_config",
    "main",
    "module_to_types_package",
    "run_deptry",
    "run_mypy_stub_hints",
    "run_pip_check",
    "s",
]


def __getattr__(name: str) -> t.ContainerValue:
    """Lazy-load module attributes on first access (PEP 562)."""
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        module_globals: MutableMapping[str, t.ContainerValue] = globals()
        module_globals[name] = value
        return value
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
