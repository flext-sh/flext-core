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

from typing import TYPE_CHECKING, Any

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.deps.detection import (
        FlextInfraDependencyDetectionModels,
        FlextInfraDependencyDetectionModels as m,
        FlextInfraDependencyDetectionService,
        FlextInfraDependencyDetectionService as s,
        build_project_report,
        classify_issues,
        discover_projects,
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
        FlextInfraDependencyDetectorModels,
        FlextInfraRuntimeDevDependencyDetector,
        ddm,
    )
    from flext_infra.deps.extra_paths import (
        ROOT,
        get_dep_paths,
        path_dep_paths,
        path_dep_paths_pep621,
        path_dep_paths_poetry,
        sync_extra_paths,
        sync_one,
    )
    from flext_infra.deps.internal_sync import FlextInfraInternalDependencySyncService
    from flext_infra.deps.modernizer import (
        ConsolidateGroupsPhase,
        EnsurePyreflyConfigPhase,
        EnsurePyrightConfigPhase,
        EnsurePytestConfigPhase,
        FlextInfraPyprojectModernizer,
        InjectCommentsPhase,
    )
    from flext_infra.deps.path_sync import (
        detect_mode,
        extract_dep_name,
        main,
        rewrite_dep_paths,
    )
    from flext_infra.deps.tool_config import (
        FlextInfraToolConfigDocument,
        load_tool_config,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ConsolidateGroupsPhase": ("flext_infra.deps.modernizer", "ConsolidateGroupsPhase"),
    "EnsurePyreflyConfigPhase": (
        "flext_infra.deps.modernizer",
        "EnsurePyreflyConfigPhase",
    ),
    "EnsurePyrightConfigPhase": (
        "flext_infra.deps.modernizer",
        "EnsurePyrightConfigPhase",
    ),
    "EnsurePytestConfigPhase": (
        "flext_infra.deps.modernizer",
        "EnsurePytestConfigPhase",
    ),
    "FlextInfraDependencyDetectionModels": (
        "flext_infra.deps.detection",
        "FlextInfraDependencyDetectionModels",
    ),
    "FlextInfraDependencyDetectionService": (
        "flext_infra.deps.detection",
        "FlextInfraDependencyDetectionService",
    ),
    "FlextInfraDependencyDetectorModels": (
        "flext_infra.deps.detector",
        "FlextInfraDependencyDetectorModels",
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
    "FlextInfraToolConfigDocument": (
        "flext_infra.deps.tool_config",
        "FlextInfraToolConfigDocument",
    ),
    "InjectCommentsPhase": ("flext_infra.deps.modernizer", "InjectCommentsPhase"),
    "ROOT": ("flext_infra.deps.extra_paths", "ROOT"),
    "build_project_report": ("flext_infra.deps.detection", "build_project_report"),
    "classify_issues": ("flext_infra.deps.detection", "classify_issues"),
    "ddm": ("flext_infra.deps.detector", "ddm"),
    "detect_mode": ("flext_infra.deps.path_sync", "detect_mode"),
    "discover_projects": ("flext_infra.deps.detection", "discover_projects"),
    "dm": ("flext_infra.deps.detection", "dm"),
    "extract_dep_name": ("flext_infra.deps.path_sync", "extract_dep_name"),
    "get_current_typings_from_pyproject": (
        "flext_infra.deps.detection",
        "get_current_typings_from_pyproject",
    ),
    "get_dep_paths": ("flext_infra.deps.extra_paths", "get_dep_paths"),
    "get_required_typings": ("flext_infra.deps.detection", "get_required_typings"),
    "load_dependency_limits": ("flext_infra.deps.detection", "load_dependency_limits"),
    "load_tool_config": ("flext_infra.deps.tool_config", "load_tool_config"),
    "m": ("flext_infra.deps.detection", "FlextInfraDependencyDetectionModels"),
    "main": ("flext_infra.deps.path_sync", "main"),
    "module_to_types_package": (
        "flext_infra.deps.detection",
        "module_to_types_package",
    ),
    "path_dep_paths": ("flext_infra.deps.extra_paths", "path_dep_paths"),
    "path_dep_paths_pep621": ("flext_infra.deps.extra_paths", "path_dep_paths_pep621"),
    "path_dep_paths_poetry": ("flext_infra.deps.extra_paths", "path_dep_paths_poetry"),
    "rewrite_dep_paths": ("flext_infra.deps.path_sync", "rewrite_dep_paths"),
    "run_deptry": ("flext_infra.deps.detection", "run_deptry"),
    "run_mypy_stub_hints": ("flext_infra.deps.detection", "run_mypy_stub_hints"),
    "run_pip_check": ("flext_infra.deps.detection", "run_pip_check"),
    "s": ("flext_infra.deps.detection", "FlextInfraDependencyDetectionService"),
    "sync_extra_paths": ("flext_infra.deps.extra_paths", "sync_extra_paths"),
    "sync_one": ("flext_infra.deps.extra_paths", "sync_one"),
}

__all__ = [
    "ROOT",
    "ConsolidateGroupsPhase",
    "EnsurePyreflyConfigPhase",
    "EnsurePyrightConfigPhase",
    "EnsurePytestConfigPhase",
    "FlextInfraDependencyDetectionModels",
    "FlextInfraDependencyDetectionService",
    "FlextInfraDependencyDetectorModels",
    "FlextInfraInternalDependencySyncService",
    "FlextInfraPyprojectModernizer",
    "FlextInfraRuntimeDevDependencyDetector",
    "FlextInfraToolConfigDocument",
    "InjectCommentsPhase",
    "build_project_report",
    "classify_issues",
    "ddm",
    "detect_mode",
    "discover_projects",
    "dm",
    "extract_dep_name",
    "get_current_typings_from_pyproject",
    "get_dep_paths",
    "get_required_typings",
    "load_dependency_limits",
    "load_tool_config",
    "m",
    "main",
    "module_to_types_package",
    "path_dep_paths",
    "path_dep_paths_pep621",
    "path_dep_paths_poetry",
    "rewrite_dep_paths",
    "run_deptry",
    "run_mypy_stub_hints",
    "run_pip_check",
    "s",
    "sync_extra_paths",
    "sync_one",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
