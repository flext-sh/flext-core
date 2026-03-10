# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Public API for flext-core.

Runtime aliases: simple assignments only (c = FlextConstants, m = FlextModels, etc.).
Never use FlextRuntime.Aliases or any alias registry for c, m, r, t, u, p, d, e, h, s, x.

Access via project runtime alias only; no subdivision. Subprojects: nested classes
for organization, then class-level aliases at facade root so call sites use m.Foo,
m.Bar only (never m.ProjectName.Foo). MRO protocol only; direct methods.

Use at call sites: from flext_core import c, m, r, t, u, p, d, e, h, s, x
Classes (FlextContainer, FlextModels, etc.) are for inheritance and type annotations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.__version__ import (
        __all__,
        __author__,
        __author_email__,
        __description__,
        __license__,
        __title__,
        __url__,
        __version__,
        __version_info__,
    )
    from flext_core._decorators.discovery import FactoryDecoratorsDiscovery
    from flext_core._dispatcher.config import FlextModelsConfig
    from flext_core._dispatcher.reliability import (
        CircuitBreakerManager,
        RateLimiterManager,
        RetryPolicy,
    )
    from flext_core._dispatcher.timeout import TimeoutEnforcer
    from flext_core._models.base import FlextModelFoundation
    from flext_core._models.collections import FlextModelsCollections
    from flext_core._models.container import FlextModelsContainer
    from flext_core._models.containers import FlextModelsContainers
    from flext_core._models.context import FlextModelsContext
    from flext_core._models.cqrs import FlextModelsCqrs
    from flext_core._models.decorators import FlextModelsDecorators
    from flext_core._models.domain_event import FlextModelsDomainEvent
    from flext_core._models.entity import FlextModelsEntity
    from flext_core._models.generic import FlextGenericModels
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.service import FlextModelsService
    from flext_core._utilities.args import FlextUtilitiesArgs
    from flext_core._utilities.cache import FlextUtilitiesCache
    from flext_core._utilities.checker import FlextUtilitiesChecker
    from flext_core._utilities.collection import FlextUtilitiesCollection
    from flext_core._utilities.configuration import FlextUtilitiesConfiguration
    from flext_core._utilities.context import FlextUtilitiesContext
    from flext_core._utilities.conversion import FlextUtilitiesConversion
    from flext_core._utilities.deprecation import FlextUtilitiesDeprecation
    from flext_core._utilities.domain import FlextUtilitiesDomain
    from flext_core._utilities.enum import FlextUtilitiesEnum
    from flext_core._utilities.file_ops import FlextUtilitiesFileOps
    from flext_core._utilities.generators import FlextUtilitiesGenerators
    from flext_core._utilities.guards import FlextUtilitiesGuards
    from flext_core._utilities.mapper import FlextUtilitiesMapper
    from flext_core._utilities.model import FlextUtilitiesModel
    from flext_core._utilities.pagination import FlextUtilitiesPagination
    from flext_core._utilities.parser import FlextUtilitiesParser
    from flext_core._utilities.reliability import FlextUtilitiesReliability
    from flext_core._utilities.result_helpers import ResultHelpers
    from flext_core._utilities.text import FlextUtilitiesText
    from flext_core.constants import FlextConstants, c
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecorators, d
    from flext_core.dispatcher import (
        DispatchMessageProtocol,
        ExecuteProtocol,
        FlextDispatcher,
        HandleProtocol,
    )
    from flext_core.exceptions import FlextExceptions, Metadata, e
    from flext_core.handlers import FlextHandlers, h
    from flext_core.loggings import FlextLogger
    from flext_core.mixins import FlextMixins, x
    from flext_core.models import FlextModels, m
    from flext_core.protocols import FlextProtocols, p
    from flext_core.registry import FlextRegistry
    from flext_core.result import FlextResult, r
    from flext_core.runtime import FlextRuntime
    from flext_core.service import FlextService, s
    from flext_core.settings import FlextSettings
    from flext_core.typings import (
        FlextTypes,
        FlextTypes as t,
        MessageT_contra,
        P,
        R,
        ResultT,
        T,
        T_co,
        T_contra,
        T_Model,
        T_Namespace,
        T_Settings,
        U,
    )
    from flext_core.utilities import FlextUtilities, u, validate_pydantic_model

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "CircuitBreakerManager": (
        "flext_core._dispatcher.reliability",
        "CircuitBreakerManager",
    ),
    "DispatchMessageProtocol": ("flext_core.dispatcher", "DispatchMessageProtocol"),
    "ExecuteProtocol": ("flext_core.dispatcher", "ExecuteProtocol"),
    "FactoryDecoratorsDiscovery": (
        "flext_core._decorators.discovery",
        "FactoryDecoratorsDiscovery",
    ),
    "FlextConstants": ("flext_core.constants", "FlextConstants"),
    "FlextContainer": ("flext_core.container", "FlextContainer"),
    "FlextContext": ("flext_core.context", "FlextContext"),
    "FlextDecorators": ("flext_core.decorators", "FlextDecorators"),
    "FlextDispatcher": ("flext_core.dispatcher", "FlextDispatcher"),
    "FlextExceptions": ("flext_core.exceptions", "FlextExceptions"),
    "FlextGenericModels": ("flext_core._models.generic", "FlextGenericModels"),
    "FlextHandlers": ("flext_core.handlers", "FlextHandlers"),
    "FlextLogger": ("flext_core.loggings", "FlextLogger"),
    "FlextMixins": ("flext_core.mixins", "FlextMixins"),
    "FlextModelFoundation": ("flext_core._models.base", "FlextModelFoundation"),
    "FlextModels": ("flext_core.models", "FlextModels"),
    "FlextModelsCollections": (
        "flext_core._models.collections",
        "FlextModelsCollections",
    ),
    "FlextModelsConfig": ("flext_core._dispatcher.config", "FlextModelsConfig"),
    "FlextModelsContainer": ("flext_core._models.container", "FlextModelsContainer"),
    "FlextModelsContainers": ("flext_core._models.containers", "FlextModelsContainers"),
    "FlextModelsContext": ("flext_core._models.context", "FlextModelsContext"),
    "FlextModelsCqrs": ("flext_core._models.cqrs", "FlextModelsCqrs"),
    "FlextModelsDecorators": ("flext_core._models.decorators", "FlextModelsDecorators"),
    "FlextModelsDomainEvent": (
        "flext_core._models.domain_event",
        "FlextModelsDomainEvent",
    ),
    "FlextModelsEntity": ("flext_core._models.entity", "FlextModelsEntity"),
    "FlextModelsHandler": ("flext_core._models.handler", "FlextModelsHandler"),
    "FlextModelsService": ("flext_core._models.service", "FlextModelsService"),
    "FlextProtocols": ("flext_core.protocols", "FlextProtocols"),
    "FlextRegistry": ("flext_core.registry", "FlextRegistry"),
    "FlextResult": ("flext_core.result", "FlextResult"),
    "FlextRuntime": ("flext_core.runtime", "FlextRuntime"),
    "FlextService": ("flext_core.service", "FlextService"),
    "FlextSettings": ("flext_core.settings", "FlextSettings"),
    "FlextTypes": ("flext_core.typings", "FlextTypes"),
    "FlextUtilities": ("flext_core.utilities", "FlextUtilities"),
    "FlextUtilitiesArgs": ("flext_core._utilities.args", "FlextUtilitiesArgs"),
    "FlextUtilitiesCache": ("flext_core._utilities.cache", "FlextUtilitiesCache"),
    "FlextUtilitiesChecker": ("flext_core._utilities.checker", "FlextUtilitiesChecker"),
    "FlextUtilitiesCollection": (
        "flext_core._utilities.collection",
        "FlextUtilitiesCollection",
    ),
    "FlextUtilitiesConfiguration": (
        "flext_core._utilities.configuration",
        "FlextUtilitiesConfiguration",
    ),
    "FlextUtilitiesContext": ("flext_core._utilities.context", "FlextUtilitiesContext"),
    "FlextUtilitiesConversion": (
        "flext_core._utilities.conversion",
        "FlextUtilitiesConversion",
    ),
    "FlextUtilitiesDeprecation": (
        "flext_core._utilities.deprecation",
        "FlextUtilitiesDeprecation",
    ),
    "FlextUtilitiesDomain": ("flext_core._utilities.domain", "FlextUtilitiesDomain"),
    "FlextUtilitiesEnum": ("flext_core._utilities.enum", "FlextUtilitiesEnum"),
    "FlextUtilitiesFileOps": (
        "flext_core._utilities.file_ops",
        "FlextUtilitiesFileOps",
    ),
    "FlextUtilitiesGenerators": (
        "flext_core._utilities.generators",
        "FlextUtilitiesGenerators",
    ),
    "FlextUtilitiesGuards": ("flext_core._utilities.guards", "FlextUtilitiesGuards"),
    "FlextUtilitiesMapper": ("flext_core._utilities.mapper", "FlextUtilitiesMapper"),
    "FlextUtilitiesModel": ("flext_core._utilities.model", "FlextUtilitiesModel"),
    "FlextUtilitiesPagination": (
        "flext_core._utilities.pagination",
        "FlextUtilitiesPagination",
    ),
    "FlextUtilitiesParser": ("flext_core._utilities.parser", "FlextUtilitiesParser"),
    "FlextUtilitiesReliability": (
        "flext_core._utilities.reliability",
        "FlextUtilitiesReliability",
    ),
    "FlextUtilitiesText": ("flext_core._utilities.text", "FlextUtilitiesText"),
    "HandleProtocol": ("flext_core.dispatcher", "HandleProtocol"),
    "MessageT_contra": ("flext_core.typings", "MessageT_contra"),
    "P": ("flext_core.typings", "P"),
    "R": ("flext_core.typings", "R"),
    "RateLimiterManager": ("flext_core._dispatcher.reliability", "RateLimiterManager"),
    "ResultHelpers": ("flext_core._utilities.result_helpers", "ResultHelpers"),
    "ResultT": ("flext_core.typings", "ResultT"),
    "RetryPolicy": ("flext_core._dispatcher.reliability", "RetryPolicy"),
    "T": ("flext_core.typings", "T"),
    "T_Model": ("flext_core.typings", "T_Model"),
    "T_Namespace": ("flext_core.typings", "T_Namespace"),
    "T_Settings": ("flext_core.typings", "T_Settings"),
    "T_co": ("flext_core.typings", "T_co"),
    "T_contra": ("flext_core.typings", "T_contra"),
    "TimeoutEnforcer": ("flext_core._dispatcher.timeout", "TimeoutEnforcer"),
    "U": ("flext_core.typings", "U"),
    "Metadata": ("flext_core.exceptions", "Metadata"),
    "__all__": ("flext_core.__version__", "__all__"),
    "__author__": ("flext_core.__version__", "__author__"),
    "__author_email__": ("flext_core.__version__", "__author_email__"),
    "__description__": ("flext_core.__version__", "__description__"),
    "__license__": ("flext_core.__version__", "__license__"),
    "__title__": ("flext_core.__version__", "__title__"),
    "__url__": ("flext_core.__version__", "__url__"),
    "__version__": ("flext_core.__version__", "__version__"),
    "__version_info__": ("flext_core.__version__", "__version_info__"),
    "c": ("flext_core.constants", "c"),
    "d": ("flext_core.decorators", "d"),
    "e": ("flext_core.exceptions", "e"),
    "h": ("flext_core.handlers", "h"),
    "m": ("flext_core.models", "m"),
    "p": ("flext_core.protocols", "p"),
    "r": ("flext_core.result", "r"),
    "s": ("flext_core.service", "s"),
    "t": ("flext_core.typings", "FlextTypes"),
    "u": ("flext_core.utilities", "u"),
    "validate_pydantic_model": ("flext_core.utilities", "validate_pydantic_model"),
    "x": ("flext_core.mixins", "x"),
}

__all__ = [
    "CircuitBreakerManager",
    "DispatchMessageProtocol",
    "ExecuteProtocol",
    "FactoryDecoratorsDiscovery",
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextGenericModels",
    "FlextHandlers",
    "FlextLogger",
    "FlextMixins",
    "FlextModelFoundation",
    "FlextModels",
    "FlextModelsCollections",
    "FlextModelsConfig",
    "FlextModelsContainer",
    "FlextModelsContainers",
    "FlextModelsContext",
    "FlextModelsCqrs",
    "FlextModelsDecorators",
    "FlextModelsDomainEvent",
    "FlextModelsEntity",
    "FlextModelsHandler",
    "FlextModelsService",
    "FlextProtocols",
    "FlextRegistry",
    "FlextResult",
    "FlextRuntime",
    "FlextService",
    "FlextSettings",
    "FlextTypes",
    "FlextUtilities",
    "FlextUtilitiesArgs",
    "FlextUtilitiesCache",
    "FlextUtilitiesChecker",
    "FlextUtilitiesCollection",
    "FlextUtilitiesConfiguration",
    "FlextUtilitiesContext",
    "FlextUtilitiesConversion",
    "FlextUtilitiesDeprecation",
    "FlextUtilitiesDomain",
    "FlextUtilitiesEnum",
    "FlextUtilitiesFileOps",
    "FlextUtilitiesGenerators",
    "FlextUtilitiesGuards",
    "FlextUtilitiesMapper",
    "FlextUtilitiesModel",
    "FlextUtilitiesPagination",
    "FlextUtilitiesParser",
    "FlextUtilitiesReliability",
    "FlextUtilitiesText",
    "HandleProtocol",
    "MessageT_contra",
    "Metadata",
    "P",
    "R",
    "RateLimiterManager",
    "ResultHelpers",
    "ResultT",
    "RetryPolicy",
    "T",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "TimeoutEnforcer",
    "U",
    "__all__",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "c",
    "d",
    "e",
    "h",
    "m",
    "p",
    "r",
    "s",
    "t",
    "u",
    "validate_pydantic_model",
    "x",
]


def __getattr__(name: str) -> Any:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
