"""Dependency management services.

Provides the pyproject modernizer for workspace-wide dependency
synchronization and formatting.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

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


def __getattr__(name: str):
    if name in {"DependencyDetectionModels", "DependencyDetectionService", "dm"}:
        from flext_infra.deps.detection import (
            DependencyDetectionModels,
            DependencyDetectionService,
            dm,
        )

        return {
            "DependencyDetectionModels": DependencyDetectionModels,
            "DependencyDetectionService": DependencyDetectionService,
            "dm": dm,
        }[name]
    if name in {
        "DependencyDetectorModels",
        "RuntimeDevDependencyDetector",
        "ddm",
        "main",
    }:
        from flext_infra.deps.detector import (
            DependencyDetectorModels,
            RuntimeDevDependencyDetector,
            ddm,
            main,
        )

        return {
            "DependencyDetectorModels": DependencyDetectorModels,
            "RuntimeDevDependencyDetector": RuntimeDevDependencyDetector,
            "ddm": ddm,
            "main": main,
        }[name]
    if name in {
        "MYPY_BASE_PROJECT",
        "MYPY_BASE_ROOT",
        "PYRIGHT_BASE_PROJECT",
        "PYRIGHT_BASE_ROOT",
        "EXTRA_PATHS_ROOT",
        "get_dep_paths",
        "sync_extra_paths",
        "sync_one",
    }:
        from flext_infra.deps.extra_paths import (
            MYPY_BASE_PROJECT,
            MYPY_BASE_ROOT,
            PYRIGHT_BASE_PROJECT,
            PYRIGHT_BASE_ROOT,
            ROOT,
            get_dep_paths,
            sync_extra_paths,
            sync_one,
        )

        return {
            "MYPY_BASE_PROJECT": MYPY_BASE_PROJECT,
            "MYPY_BASE_ROOT": MYPY_BASE_ROOT,
            "PYRIGHT_BASE_PROJECT": PYRIGHT_BASE_PROJECT,
            "PYRIGHT_BASE_ROOT": PYRIGHT_BASE_ROOT,
            "EXTRA_PATHS_ROOT": ROOT,
            "get_dep_paths": get_dep_paths,
            "sync_extra_paths": sync_extra_paths,
            "sync_one": sync_one,
        }[name]
    if name in {"InternalDependencySyncService", "RepoUrls"}:
        from flext_infra.deps.internal_sync import (
            InternalDependencySyncService,
            RepoUrls,
        )

        return {
            "InternalDependencySyncService": InternalDependencySyncService,
            "RepoUrls": RepoUrls,
        }[name]
    if name == "PyprojectModernizer":
        from flext_infra.deps.modernizer import PyprojectModernizer

        return PyprojectModernizer
    raise AttributeError(name)


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
