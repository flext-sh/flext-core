"""Dependency management services.

Provides the pyproject modernizer for workspace-wide dependency
synchronization and formatting.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_infra.deps.detection import (
        DependencyDetectionModels,
        DependencyDetectionService,
        dm,
    )
    from flext_infra.deps.detector import (
        DependencyDetectorModels,
        RuntimeDevDependencyDetector,
        ddm,
        main,
    )
    from flext_infra.deps.extra_paths import (
        MYPY_BASE_PROJECT,
        MYPY_BASE_ROOT,
        PYRIGHT_BASE_PROJECT,
        PYRIGHT_BASE_ROOT,
        ROOT as EXTRA_PATHS_ROOT,
        get_dep_paths,
        sync_extra_paths,
        sync_one,
    )
    from flext_infra.deps.internal_sync import InternalDependencySyncService
    from flext_infra.deps.modernizer import PyprojectModernizer
    from flext_infra.deps.path_sync import (
        FLEXT_DEPS_DIR,
        detect_mode,
        extract_dep_name,
        rewrite_dep_paths,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "DependencyDetectionModels": ("flext_infra.deps.detection", "DependencyDetectionModels"),
    "DependencyDetectionService": ("flext_infra.deps.detection", "DependencyDetectionService"),
    "DependencyDetectorModels": ("flext_infra.deps.detector", "DependencyDetectorModels"),
    "EXTRA_PATHS_ROOT": ("flext_infra.deps.extra_paths", "ROOT"),
    "FLEXT_DEPS_DIR": ("flext_infra.deps.path_sync", "FLEXT_DEPS_DIR"),
    "InternalDependencySyncService": ("flext_infra.deps.internal_sync", "InternalDependencySyncService"),
    "MYPY_BASE_PROJECT": ("flext_infra.deps.extra_paths", "MYPY_BASE_PROJECT"),
    "MYPY_BASE_ROOT": ("flext_infra.deps.extra_paths", "MYPY_BASE_ROOT"),
    "PYRIGHT_BASE_PROJECT": ("flext_infra.deps.extra_paths", "PYRIGHT_BASE_PROJECT"),
    "PYRIGHT_BASE_ROOT": ("flext_infra.deps.extra_paths", "PYRIGHT_BASE_ROOT"),
    "PyprojectModernizer": ("flext_infra.deps.modernizer", "PyprojectModernizer"),
    "RuntimeDevDependencyDetector": ("flext_infra.deps.detector", "RuntimeDevDependencyDetector"),
    "ddm": ("flext_infra.deps.detector", "ddm"),
    "detect_mode": ("flext_infra.deps.path_sync", "detect_mode"),
    "dm": ("flext_infra.deps.detection", "dm"),
    "extract_dep_name": ("flext_infra.deps.path_sync", "extract_dep_name"),
    "get_dep_paths": ("flext_infra.deps.extra_paths", "get_dep_paths"),
    "main": ("flext_infra.deps.detector", "main"),
    "rewrite_dep_paths": ("flext_infra.deps.path_sync", "rewrite_dep_paths"),
    "sync_extra_paths": ("flext_infra.deps.extra_paths", "sync_extra_paths"),
    "sync_one": ("flext_infra.deps.extra_paths", "sync_one"),
}

__all__ = [
    "EXTRA_PATHS_ROOT",
    "FLEXT_DEPS_DIR",
    "MYPY_BASE_PROJECT",
    "MYPY_BASE_ROOT",
    "PYRIGHT_BASE_PROJECT",
    "PYRIGHT_BASE_ROOT",
    "DependencyDetectionModels",
    "DependencyDetectionService",
    "DependencyDetectorModels",
    "InternalDependencySyncService",
    "PyprojectModernizer",
    "RuntimeDevDependencyDetector",
    "ddm",
    "detect_mode",
    "dm",
    "extract_dep_name",
    "get_dep_paths",
    "main",
    "rewrite_dep_paths",
    "sync_extra_paths",
    "sync_one",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
