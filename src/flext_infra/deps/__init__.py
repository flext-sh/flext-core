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
from flext_infra.deps.internal_sync import InternalDependencySyncService
from flext_infra.deps.modernizer import PyprojectModernizer
from flext_infra.deps.path_sync import (
    FLEXT_DEPS_DIR,
    detect_mode,
    extract_dep_name,
    rewrite_dep_paths,
)

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
