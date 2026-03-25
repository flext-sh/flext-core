# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Flext infra package."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr


if TYPE_CHECKING:
    from flext_core import FlextTypes
    import flext_infra.check as check
    from flext_infra.check.services import (
        DEFAULT_GATES,
        PyreflyConfigFixer,
        WorkspaceChecker,
        run_cli,
    )
    import flext_infra.deps as deps
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
    import flext_infra.workspace as workspace
    from flext_infra.workspace.detector import WorkspaceDetector, WorkspaceMode
    from flext_infra.workspace.migrator import ProjectMigrator
    from flext_infra.workspace.sync import SyncService

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "ConsolidateGroupsPhase": ["flext_infra.deps.modernizer", "ConsolidateGroupsPhase"],
    "DEFAULT_GATES": ["flext_infra.check.services", "DEFAULT_GATES"],
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
    "ProjectMigrator": ["flext_infra.workspace.migrator", "ProjectMigrator"],
    "PyprojectModernizer": ["flext_infra.deps.modernizer", "PyprojectModernizer"],
    "PyreflyConfigFixer": ["flext_infra.check.services", "PyreflyConfigFixer"],
    "ROOT": ["flext_infra.deps.modernizer", "ROOT"],
    "RepoUrls": ["flext_infra.deps.internal_sync", "RepoUrls"],
    "RuntimeDevDependencyDetector": ["flext_infra.deps.detector", "RuntimeDevDependencyDetector"],
    "SKIP_DIRS": ["flext_infra.deps.modernizer", "SKIP_DIRS"],
    "SyncService": ["flext_infra.workspace.sync", "SyncService"],
    "WorkspaceChecker": ["flext_infra.check.services", "WorkspaceChecker"],
    "WorkspaceDetector": ["flext_infra.workspace.detector", "WorkspaceDetector"],
    "WorkspaceMode": ["flext_infra.workspace.detector", "WorkspaceMode"],
    "build_project_report": ["flext_infra.deps.detection", "build_project_report"],
    "check": ["flext_infra.check", ""],
    "classify_issues": ["flext_infra.deps.detection", "classify_issues"],
    "ddm": ["flext_infra.deps.detector", "ddm"],
    "deps": ["flext_infra.deps", ""],
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
    "run_cli": ["flext_infra.check.services", "run_cli"],
    "run_deptry": ["flext_infra.deps.detection", "run_deptry"],
    "run_mypy_stub_hints": ["flext_infra.deps.detection", "run_mypy_stub_hints"],
    "run_pip_check": ["flext_infra.deps.detection", "run_pip_check"],
    "sync_extra_paths": ["flext_infra.deps.extra_paths", "sync_extra_paths"],
    "sync_one": ["flext_infra.deps.extra_paths", "sync_one"],
    "workspace": ["flext_infra.workspace", ""],
}

__all__ = [
    "ConsolidateGroupsPhase",
    "DEFAULT_GATES",
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
    "ProjectMigrator",
    "PyprojectModernizer",
    "PyreflyConfigFixer",
    "ROOT",
    "RepoUrls",
    "RuntimeDevDependencyDetector",
    "SKIP_DIRS",
    "SyncService",
    "WorkspaceChecker",
    "WorkspaceDetector",
    "WorkspaceMode",
    "build_project_report",
    "check",
    "classify_issues",
    "ddm",
    "deps",
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
    "run_cli",
    "run_deptry",
    "run_mypy_stub_hints",
    "run_pip_check",
    "sync_extra_paths",
    "sync_one",
    "workspace",
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