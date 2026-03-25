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

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr


if TYPE_CHECKING:
    from flext_core import FlextTypes
    from flext_infra.deps.detection import (
        DependencyDetectionModels,
        DependencyDetectionService,
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
        DependencyDetectorModels,
        RuntimeDevDependencyDetector,
        ddm,
    )
    from flext_infra.deps.extra_paths import (
        MYPY_BASE_PROJECT,
        MYPY_BASE_ROOT,
        PYRIGHT_BASE_PROJECT,
        PYRIGHT_BASE_ROOT,
        get_dep_paths,
        sync_extra_paths,
        sync_one,
    )
    from flext_infra.deps.internal_sync import InternalDependencySyncService, RepoUrls
    from flext_infra.deps.modernizer import (
        ConsolidateGroupsPhase,
        InjectCommentsPhase,
        PyprojectModernizer,
        ROOT,
        SKIP_DIRS,
        main,
    )
    from flext_infra.deps.path_sync import (
        FLEXT_DEPS_DIR,
        detect_mode,
        extract_dep_name,
        rewrite_dep_paths,
    )

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "ConsolidateGroupsPhase": ["flext_infra.deps.modernizer", "ConsolidateGroupsPhase"],
    "DependencyDetectionModels": ["flext_infra.deps.detection", "DependencyDetectionModels"],
    "DependencyDetectionService": ["flext_infra.deps.detection", "DependencyDetectionService"],
    "DependencyDetectorModels": ["flext_infra.deps.detector", "DependencyDetectorModels"],
    "FLEXT_DEPS_DIR": ["flext_infra.deps.path_sync", "FLEXT_DEPS_DIR"],
    "InjectCommentsPhase": ["flext_infra.deps.modernizer", "InjectCommentsPhase"],
    "InternalDependencySyncService": ["flext_infra.deps.internal_sync", "InternalDependencySyncService"],
    "MYPY_BASE_PROJECT": ["flext_infra.deps.extra_paths", "MYPY_BASE_PROJECT"],
    "MYPY_BASE_ROOT": ["flext_infra.deps.extra_paths", "MYPY_BASE_ROOT"],
    "PYRIGHT_BASE_PROJECT": ["flext_infra.deps.extra_paths", "PYRIGHT_BASE_PROJECT"],
    "PYRIGHT_BASE_ROOT": ["flext_infra.deps.extra_paths", "PYRIGHT_BASE_ROOT"],
    "PyprojectModernizer": ["flext_infra.deps.modernizer", "PyprojectModernizer"],
    "ROOT": ["flext_infra.deps.modernizer", "ROOT"],
    "RepoUrls": ["flext_infra.deps.internal_sync", "RepoUrls"],
    "RuntimeDevDependencyDetector": ["flext_infra.deps.detector", "RuntimeDevDependencyDetector"],
    "SKIP_DIRS": ["flext_infra.deps.modernizer", "SKIP_DIRS"],
    "build_project_report": ["flext_infra.deps.detection", "build_project_report"],
    "classify_issues": ["flext_infra.deps.detection", "classify_issues"],
    "ddm": ["flext_infra.deps.detector", "ddm"],
    "detect_mode": ["flext_infra.deps.path_sync", "detect_mode"],
    "discover_projects": ["flext_infra.deps.detection", "discover_projects"],
    "dm": ["flext_infra.deps.detection", "dm"],
    "extract_dep_name": ["flext_infra.deps.path_sync", "extract_dep_name"],
    "get_current_typings_from_pyproject": ["flext_infra.deps.detection", "get_current_typings_from_pyproject"],
    "get_dep_paths": ["flext_infra.deps.extra_paths", "get_dep_paths"],
    "get_required_typings": ["flext_infra.deps.detection", "get_required_typings"],
    "load_dependency_limits": ["flext_infra.deps.detection", "load_dependency_limits"],
    "main": ["flext_infra.deps.modernizer", "main"],
    "module_to_types_package": ["flext_infra.deps.detection", "module_to_types_package"],
    "rewrite_dep_paths": ["flext_infra.deps.path_sync", "rewrite_dep_paths"],
    "run_deptry": ["flext_infra.deps.detection", "run_deptry"],
    "run_mypy_stub_hints": ["flext_infra.deps.detection", "run_mypy_stub_hints"],
    "run_pip_check": ["flext_infra.deps.detection", "run_pip_check"],
    "sync_extra_paths": ["flext_infra.deps.extra_paths", "sync_extra_paths"],
    "sync_one": ["flext_infra.deps.extra_paths", "sync_one"],
}

__all__ = [
    "ConsolidateGroupsPhase",
    "DependencyDetectionModels",
    "DependencyDetectionService",
    "DependencyDetectorModels",
    "FLEXT_DEPS_DIR",
    "InjectCommentsPhase",
    "InternalDependencySyncService",
    "MYPY_BASE_PROJECT",
    "MYPY_BASE_ROOT",
    "PYRIGHT_BASE_PROJECT",
    "PYRIGHT_BASE_ROOT",
    "PyprojectModernizer",
    "ROOT",
    "RepoUrls",
    "RuntimeDevDependencyDetector",
    "SKIP_DIRS",
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
    "main",
    "module_to_types_package",
    "rewrite_dep_paths",
    "run_deptry",
    "run_mypy_stub_hints",
    "run_pip_check",
    "sync_extra_paths",
    "sync_one",
]


_LAZY_CACHE: MutableMapping[str, FlextTypes.ModuleExport] = {}


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562).

    A local cache ``_LAZY_CACHE`` persists resolved objects across repeated
    accesses during process lifetime.

    Args:
        name: Attribute name requested by dir()/import.

    Returns:
        Lazy-loaded module export type.

    Raises:
        AttributeError: If attribute not registered.
    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    value = lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)
    _LAZY_CACHE[name] = value
    return value


def __dir__() -> Sequence[str]:
    """Return list of available attributes for dir() and autocomplete.

    Returns:
        List of public names from module exports.
    """
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)