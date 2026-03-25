# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Workspace management services.

Provides services for workspace detection, synchronization, and orchestration
across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr


if TYPE_CHECKING:
    from flext_core import FlextTypes
    from flext_infra.workspace.detector import WorkspaceDetector, WorkspaceMode
    from flext_infra.workspace.migrator import ProjectMigrator
    from flext_infra.workspace.sync import SyncService, main

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "ProjectMigrator": ["flext_infra.workspace.migrator", "ProjectMigrator"],
    "SyncService": ["flext_infra.workspace.sync", "SyncService"],
    "WorkspaceDetector": ["flext_infra.workspace.detector", "WorkspaceDetector"],
    "WorkspaceMode": ["flext_infra.workspace.detector", "WorkspaceMode"],
    "main": ["flext_infra.workspace.sync", "main"],
}

__all__ = [
    "ProjectMigrator",
    "SyncService",
    "WorkspaceDetector",
    "WorkspaceMode",
    "main",
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