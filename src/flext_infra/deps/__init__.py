"""Dependency management services.

Provides services for dependency analysis, modernization, and synchronization
across the workspace.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra.deps.detection import DependencyDetector
from flext_infra.deps.detector import RuntimeDevDetector
from flext_infra.deps.extra_paths import ExtraPathsSyncer
from flext_infra.deps.internal_sync import InternalDepsSyncer
from flext_infra.deps.modernizer import PyprojectModernizer
from flext_infra.deps.path_sync import DepPathSyncer

__all__ = [
    "DepPathSyncer",
    "DependencyDetector",
    "ExtraPathsSyncer",
    "InternalDepsSyncer",
    "PyprojectModernizer",
    "RuntimeDevDetector",
]
