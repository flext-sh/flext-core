"""Dependency management services.

Provides the pyproject modernizer for workspace-wide dependency
synchronization and formatting.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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
from flext_infra.deps.internal_sync import InternalDependencySyncService, RepoUrls
from flext_infra.deps.modernizer import PyprojectModernizer

FLEXT_DEPS_DIR = ".flext-deps"


def detect_mode(project_root):
    from flext_infra.deps.path_sync import detect_mode as _detect_mode

    return _detect_mode(project_root)


def extract_dep_name(raw_path):
    from flext_infra.deps.path_sync import extract_dep_name as _extract_dep_name

    return _extract_dep_name(raw_path)


def rewrite_dep_paths(pyproject_path, **kwargs):
    from flext_infra.deps.path_sync import rewrite_dep_paths as _rewrite_dep_paths

    return _rewrite_dep_paths(pyproject_path, **kwargs)


__all__ = [
    "DependencyDetectionModels",
    "DependencyDetectionService",
    "DependencyDetectorModels",
    "EXTRA_PATHS_ROOT",
    "FLEXT_DEPS_DIR",
    "InternalDependencySyncService",
    "MYPY_BASE_PROJECT",
    "MYPY_BASE_ROOT",
    "PYRIGHT_BASE_PROJECT",
    "PYRIGHT_BASE_ROOT",
    "PyprojectModernizer",
    "RepoUrls",
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
