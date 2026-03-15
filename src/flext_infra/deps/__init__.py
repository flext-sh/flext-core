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

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from flext_infra.deps import _phases
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
    from flext_infra.deps.detection import (
        FlextInfraDependencyDetectionService,
        FlextInfraDependencyDetectionService as s,
        build_project_report,
        classify_issues,
        discover_project_paths,
        dm,
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
        FlextInfraUtilitiesIo,
        FlextInfraUtilitiesPaths,
        FlextInfraUtilitiesReporting,
        FlextInfraUtilitiesSubprocess,
        main,
    )
    from flext_infra.deps.extra_paths import FlextInfraExtraPathsManager
    from flext_infra.deps.internal_sync import (
        FlextInfraInternalDependencySyncService,
        shutil,
    )
    from flext_infra.deps.modernizer import FlextInfraPyprojectModernizer, u
    from flext_infra.deps.path_sync import FlextInfraDependencyPathSync
    from flext_infra.deps.tool_config import FlextInfraDependencyToolConfig

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
    "FlextInfraDependencyDetectionService": (
        "flext_infra.deps.detection",
        "FlextInfraDependencyDetectionService",
    ),
    "FlextInfraDependencyPathSync": (
        "flext_infra.deps.path_sync",
        "FlextInfraDependencyPathSync",
    ),
    "FlextInfraDependencyToolConfig": (
        "flext_infra.deps.tool_config",
        "FlextInfraDependencyToolConfig",
    ),
    "FlextInfraExtraPathsManager": (
        "flext_infra.deps.extra_paths",
        "FlextInfraExtraPathsManager",
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
    "FlextInfraUtilitiesIo": ("flext_infra.deps.detector", "FlextInfraUtilitiesIo"),
    "FlextInfraUtilitiesPaths": (
        "flext_infra.deps.detector",
        "FlextInfraUtilitiesPaths",
    ),
    "FlextInfraUtilitiesReporting": (
        "flext_infra.deps.detector",
        "FlextInfraUtilitiesReporting",
    ),
    "FlextInfraUtilitiesSubprocess": (
        "flext_infra.deps.detector",
        "FlextInfraUtilitiesSubprocess",
    ),
    "InjectCommentsPhase": (
        "flext_infra.deps._phases.inject_comments",
        "InjectCommentsPhase",
    ),
    "_phases": ("flext_infra.deps._phases", ""),
    "build_project_report": ("flext_infra.deps.detection", "build_project_report"),
    "classify_issues": ("flext_infra.deps.detection", "classify_issues"),
    "discover_project_paths": ("flext_infra.deps.detection", "discover_project_paths"),
    "dm": ("flext_infra.deps.detection", "dm"),
    "get_current_typings_from_pyproject": (
        "flext_infra.deps.detection",
        "get_current_typings_from_pyproject",
    ),
    "get_required_typings": ("flext_infra.deps.detection", "get_required_typings"),
    "load_dependency_limits": ("flext_infra.deps.detection", "load_dependency_limits"),
    "main": ("flext_infra.deps.detector", "main"),
    "module_to_types_package": (
        "flext_infra.deps.detection",
        "module_to_types_package",
    ),
    "run_deptry": ("flext_infra.deps.detection", "run_deptry"),
    "run_mypy_stub_hints": ("flext_infra.deps.detection", "run_mypy_stub_hints"),
    "run_pip_check": ("flext_infra.deps.detection", "run_pip_check"),
    "s": ("flext_infra.deps.detection", "FlextInfraDependencyDetectionService"),
    "shutil": ("flext_infra.deps.internal_sync", "shutil"),
    "u": ("flext_infra.deps.modernizer", "u"),
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
    "FlextInfraDependencyDetectionService",
    "FlextInfraDependencyPathSync",
    "FlextInfraDependencyToolConfig",
    "FlextInfraExtraPathsManager",
    "FlextInfraInternalDependencySyncService",
    "FlextInfraPyprojectModernizer",
    "FlextInfraRuntimeDevDependencyDetector",
    "FlextInfraUtilitiesIo",
    "FlextInfraUtilitiesPaths",
    "FlextInfraUtilitiesReporting",
    "FlextInfraUtilitiesSubprocess",
    "InjectCommentsPhase",
    "_phases",
    "build_project_report",
    "classify_issues",
    "discover_project_paths",
    "dm",
    "get_current_typings_from_pyproject",
    "get_required_typings",
    "load_dependency_limits",
    "main",
    "module_to_types_package",
    "run_deptry",
    "run_mypy_stub_hints",
    "run_pip_check",
    "s",
    "shutil",
    "u",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
