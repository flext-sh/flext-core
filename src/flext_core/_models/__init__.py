# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Internal module for FlextModels nested classes.

This module contains extracted nested classes from FlextModels to improve
maintainability.

All classes are re-exported through FlextModels in models.py - users should
NEVER import from this module directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core._models.base import FlextModelFoundation
    from flext_core._models.collections import FlextModelsCollections
    from flext_core._models.container import FlextModelsContainer
    from flext_core._models.containers import FlextModelsContainers
    from flext_core._models.context import FlextModelsContext
    from flext_core._models.cqrs import FlextModelsCqrs
    from flext_core._models.decorators import FlextModelsDecorators
    from flext_core._models.dispatcher import FlextModelsDispatcher
    from flext_core._models.domain_event import FlextModelsDomainEvent
    from flext_core._models.entity import FlextModelsEntity
    from flext_core._models.generic import FlextGenericModels
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.result import FlextModelsResult
    from flext_core._models.service import FlextModelsService
    from flext_core._models.settings import FlextModelsConfig
    from flext_core.typings import FlextTypes

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FlextGenericModels": ("flext_core._models.generic", "FlextGenericModels"),
    "FlextModelFoundation": ("flext_core._models.base", "FlextModelFoundation"),
    "FlextModelsCollections": (
        "flext_core._models.collections",
        "FlextModelsCollections",
    ),
    "FlextModelsConfig": ("flext_core._models.settings", "FlextModelsConfig"),
    "FlextModelsContainer": ("flext_core._models.container", "FlextModelsContainer"),
    "FlextModelsContainers": ("flext_core._models.containers", "FlextModelsContainers"),
    "FlextModelsContext": ("flext_core._models.context", "FlextModelsContext"),
    "FlextModelsCqrs": ("flext_core._models.cqrs", "FlextModelsCqrs"),
    "FlextModelsDecorators": ("flext_core._models.decorators", "FlextModelsDecorators"),
    "FlextModelsDispatcher": ("flext_core._models.dispatcher", "FlextModelsDispatcher"),
    "FlextModelsDomainEvent": (
        "flext_core._models.domain_event",
        "FlextModelsDomainEvent",
    ),
    "FlextModelsEntity": ("flext_core._models.entity", "FlextModelsEntity"),
    "FlextModelsHandler": ("flext_core._models.handler", "FlextModelsHandler"),
    "FlextModelsResult": ("flext_core._models.result", "FlextModelsResult"),
    "FlextModelsService": ("flext_core._models.service", "FlextModelsService"),
}

__all__ = [
    "FlextGenericModels",
    "FlextModelFoundation",
    "FlextModelsCollections",
    "FlextModelsConfig",
    "FlextModelsContainer",
    "FlextModelsContainers",
    "FlextModelsContext",
    "FlextModelsCqrs",
    "FlextModelsDecorators",
    "FlextModelsDispatcher",
    "FlextModelsDomainEvent",
    "FlextModelsEntity",
    "FlextModelsHandler",
    "FlextModelsResult",
    "FlextModelsService",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
