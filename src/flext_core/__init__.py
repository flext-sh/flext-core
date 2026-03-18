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

from typing import TYPE_CHECKING

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core import (
        _constants,
        _decorators,
        _dispatcher,
        _models,
        _protocols,
        _typings,
        _utilities,
    )
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
    from flext_core._constants.base import FlextConstantsBase
    from flext_core._constants.cqrs import FlextConstantsCqrs
    from flext_core._constants.domain import FlextConstantsDomain
    from flext_core._constants.infrastructure import FlextConstantsInfrastructure
    from flext_core._constants.mixins import FlextConstantsMixins
    from flext_core._constants.platform import FlextConstantsPlatform
    from flext_core._constants.settings import FlextConstantsSettings
    from flext_core._constants.validation import FlextConstantsValidation
    from flext_core._decorators.discovery import FactoryDecoratorsDiscovery
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
    from flext_core._models.dispatcher import FlextModelsDispatcher
    from flext_core._models.domain_event import FlextModelsDomainEvent
    from flext_core._models.entity import FlextModelsEntity
    from flext_core._models.generic import FlextGenericModels
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.result import FlextModelsResult
    from flext_core._models.service import FlextModelsService
    from flext_core._models.settings import FlextModelsConfig
    from flext_core._protocols.base import FlextProtocolsBase
    from flext_core._protocols.config import FlextProtocolsConfig
    from flext_core._protocols.context import FlextProtocolsContext
    from flext_core._protocols.di import FlextProtocolsDI
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._protocols.logging import FlextProtocolsLogging
    from flext_core._protocols.registry import FlextProtocolsRegistry
    from flext_core._protocols.result import FlextProtocolsResult
    from flext_core._protocols.service import FlextProtocolsService
    from flext_core._typings.base import FlextTypingBase
    from flext_core._typings.containers import FlextTypingContainers
    from flext_core._typings.core import FlextTypesCore
    from flext_core._typings.services import FlextTypesServices
    from flext_core._typings.validation import FlextTypesValidation
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
    from flext_core._utilities.guards_ensure import FlextUtilitiesGuardsEnsure
    from flext_core._utilities.guards_type import FlextUtilitiesGuardsType
    from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
    from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
    from flext_core._utilities.guards_type_protocol import (
        FlextUtilitiesGuardsTypeProtocol,
    )
    from flext_core._utilities.guards_validation import FlextUtilitiesGuardsValidation
    from flext_core._utilities.mapper import FlextUtilitiesMapper
    from flext_core._utilities.model import FlextUtilitiesModel
    from flext_core._utilities.pagination import FlextUtilitiesPagination
    from flext_core._utilities.parser import FlextUtilitiesParser
    from flext_core._utilities.reliability import FlextUtilitiesReliability
    from flext_core._utilities.result_helpers import FlextUtilitiesResultHelpers
    from flext_core._utilities.text import FlextUtilitiesText
    from flext_core.constants import (
        PROJECT_KIND_APPLICATION,
        PROJECT_KIND_LIBRARY,
        PROJECT_KIND_SERVICE,
        FlextConstants,
        c,
    )
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecorators, d
    from flext_core.dispatcher import FlextDispatcher
    from flext_core.errors import ErrorDomain, FlextError
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
        TV,
        BaseModel,
        EnumT,
        FlextTypes,
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
        TRuntime,
        TV_co,
        U,
        ValidatedParams,
        ValidatedReturn,
        t,
    )
    from flext_core.utilities import FlextUtilities, u

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseModel": ("flext_core.typings", "BaseModel"),
    "CircuitBreakerManager": (
        "flext_core._dispatcher.reliability",
        "CircuitBreakerManager",
    ),
    "EnumT": ("flext_core.typings", "EnumT"),
    "ErrorDomain": ("flext_core.errors", "ErrorDomain"),
    "FactoryDecoratorsDiscovery": (
        "flext_core._decorators.discovery",
        "FactoryDecoratorsDiscovery",
    ),
    "FlextConstants": ("flext_core.constants", "FlextConstants"),
    "FlextConstantsBase": ("flext_core._constants.base", "FlextConstantsBase"),
    "FlextConstantsCqrs": ("flext_core._constants.cqrs", "FlextConstantsCqrs"),
    "FlextConstantsDomain": ("flext_core._constants.domain", "FlextConstantsDomain"),
    "FlextConstantsInfrastructure": (
        "flext_core._constants.infrastructure",
        "FlextConstantsInfrastructure",
    ),
    "FlextConstantsMixins": ("flext_core._constants.mixins", "FlextConstantsMixins"),
    "FlextConstantsPlatform": (
        "flext_core._constants.platform",
        "FlextConstantsPlatform",
    ),
    "FlextConstantsSettings": (
        "flext_core._constants.settings",
        "FlextConstantsSettings",
    ),
    "FlextConstantsValidation": (
        "flext_core._constants.validation",
        "FlextConstantsValidation",
    ),
    "FlextContainer": ("flext_core.container", "FlextContainer"),
    "FlextContext": ("flext_core.context", "FlextContext"),
    "FlextDecorators": ("flext_core.decorators", "FlextDecorators"),
    "FlextDispatcher": ("flext_core.dispatcher", "FlextDispatcher"),
    "FlextError": ("flext_core.errors", "FlextError"),
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
    "FlextProtocols": ("flext_core.protocols", "FlextProtocols"),
    "FlextProtocolsBase": ("flext_core._protocols.base", "FlextProtocolsBase"),
    "FlextProtocolsConfig": ("flext_core._protocols.config", "FlextProtocolsConfig"),
    "FlextProtocolsContext": ("flext_core._protocols.context", "FlextProtocolsContext"),
    "FlextProtocolsDI": ("flext_core._protocols.di", "FlextProtocolsDI"),
    "FlextProtocolsHandler": ("flext_core._protocols.handler", "FlextProtocolsHandler"),
    "FlextProtocolsLogging": ("flext_core._protocols.logging", "FlextProtocolsLogging"),
    "FlextProtocolsRegistry": (
        "flext_core._protocols.registry",
        "FlextProtocolsRegistry",
    ),
    "FlextProtocolsResult": ("flext_core._protocols.result", "FlextProtocolsResult"),
    "FlextProtocolsService": ("flext_core._protocols.service", "FlextProtocolsService"),
    "FlextRegistry": ("flext_core.registry", "FlextRegistry"),
    "FlextResult": ("flext_core.result", "FlextResult"),
    "FlextRuntime": ("flext_core.runtime", "FlextRuntime"),
    "FlextService": ("flext_core.service", "FlextService"),
    "FlextSettings": ("flext_core.settings", "FlextSettings"),
    "FlextTypes": ("flext_core.typings", "FlextTypes"),
    "FlextTypesCore": ("flext_core._typings.core", "FlextTypesCore"),
    "FlextTypesServices": ("flext_core._typings.services", "FlextTypesServices"),
    "FlextTypesValidation": ("flext_core._typings.validation", "FlextTypesValidation"),
    "FlextTypingBase": ("flext_core._typings.base", "FlextTypingBase"),
    "FlextTypingContainers": (
        "flext_core._typings.containers",
        "FlextTypingContainers",
    ),
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
    "FlextUtilitiesGuardsEnsure": (
        "flext_core._utilities.guards_ensure",
        "FlextUtilitiesGuardsEnsure",
    ),
    "FlextUtilitiesGuardsType": (
        "flext_core._utilities.guards_type",
        "FlextUtilitiesGuardsType",
    ),
    "FlextUtilitiesGuardsTypeCore": (
        "flext_core._utilities.guards_type_core",
        "FlextUtilitiesGuardsTypeCore",
    ),
    "FlextUtilitiesGuardsTypeModel": (
        "flext_core._utilities.guards_type_model",
        "FlextUtilitiesGuardsTypeModel",
    ),
    "FlextUtilitiesGuardsTypeProtocol": (
        "flext_core._utilities.guards_type_protocol",
        "FlextUtilitiesGuardsTypeProtocol",
    ),
    "FlextUtilitiesGuardsValidation": (
        "flext_core._utilities.guards_validation",
        "FlextUtilitiesGuardsValidation",
    ),
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
    "FlextUtilitiesResultHelpers": (
        "flext_core._utilities.result_helpers",
        "FlextUtilitiesResultHelpers",
    ),
    "FlextUtilitiesText": ("flext_core._utilities.text", "FlextUtilitiesText"),
    "MessageT_contra": ("flext_core.typings", "MessageT_contra"),
    "Metadata": ("flext_core.exceptions", "Metadata"),
    "P": ("flext_core.typings", "P"),
    "PROJECT_KIND_APPLICATION": ("flext_core.constants", "PROJECT_KIND_APPLICATION"),
    "PROJECT_KIND_LIBRARY": ("flext_core.constants", "PROJECT_KIND_LIBRARY"),
    "PROJECT_KIND_SERVICE": ("flext_core.constants", "PROJECT_KIND_SERVICE"),
    "R": ("flext_core.typings", "R"),
    "RateLimiterManager": ("flext_core._dispatcher.reliability", "RateLimiterManager"),
    "ResultT": ("flext_core.typings", "ResultT"),
    "RetryPolicy": ("flext_core._dispatcher.reliability", "RetryPolicy"),
    "T": ("flext_core.typings", "T"),
    "TRuntime": ("flext_core.typings", "TRuntime"),
    "TV": ("flext_core.typings", "TV"),
    "TV_co": ("flext_core.typings", "TV_co"),
    "T_Model": ("flext_core.typings", "T_Model"),
    "T_Namespace": ("flext_core.typings", "T_Namespace"),
    "T_Settings": ("flext_core.typings", "T_Settings"),
    "T_co": ("flext_core.typings", "T_co"),
    "T_contra": ("flext_core.typings", "T_contra"),
    "TimeoutEnforcer": ("flext_core._dispatcher.timeout", "TimeoutEnforcer"),
    "U": ("flext_core.typings", "U"),
    "ValidatedParams": ("flext_core.typings", "ValidatedParams"),
    "ValidatedReturn": ("flext_core.typings", "ValidatedReturn"),
    "__all__": ("flext_core.__version__", "__all__"),
    "__author__": ("flext_core.__version__", "__author__"),
    "__author_email__": ("flext_core.__version__", "__author_email__"),
    "__description__": ("flext_core.__version__", "__description__"),
    "__license__": ("flext_core.__version__", "__license__"),
    "__title__": ("flext_core.__version__", "__title__"),
    "__url__": ("flext_core.__version__", "__url__"),
    "__version__": ("flext_core.__version__", "__version__"),
    "__version_info__": ("flext_core.__version__", "__version_info__"),
    "_constants": ("flext_core._constants", ""),
    "_decorators": ("flext_core._decorators", ""),
    "_dispatcher": ("flext_core._dispatcher", ""),
    "_models": ("flext_core._models", ""),
    "_protocols": ("flext_core._protocols", ""),
    "_typings": ("flext_core._typings", ""),
    "_utilities": ("flext_core._utilities", ""),
    "c": ("flext_core.constants", "c"),
    "d": ("flext_core.decorators", "d"),
    "e": ("flext_core.exceptions", "e"),
    "h": ("flext_core.handlers", "h"),
    "m": ("flext_core.models", "m"),
    "p": ("flext_core.protocols", "p"),
    "r": ("flext_core.result", "r"),
    "s": ("flext_core.service", "s"),
    "t": ("flext_core.typings", "t"),
    "u": ("flext_core.utilities", "u"),
    "x": ("flext_core.mixins", "x"),
}

__all__ = [
    "PROJECT_KIND_APPLICATION",
    "PROJECT_KIND_LIBRARY",
    "PROJECT_KIND_SERVICE",
    "TV",
    "BaseModel",
    "CircuitBreakerManager",
    "EnumT",
    "ErrorDomain",
    "FactoryDecoratorsDiscovery",
    "FlextConstants",
    "FlextConstantsBase",
    "FlextConstantsCqrs",
    "FlextConstantsDomain",
    "FlextConstantsInfrastructure",
    "FlextConstantsMixins",
    "FlextConstantsPlatform",
    "FlextConstantsSettings",
    "FlextConstantsValidation",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextError",
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
    "FlextModelsDispatcher",
    "FlextModelsDomainEvent",
    "FlextModelsEntity",
    "FlextModelsHandler",
    "FlextModelsResult",
    "FlextModelsService",
    "FlextProtocols",
    "FlextProtocolsBase",
    "FlextProtocolsConfig",
    "FlextProtocolsContext",
    "FlextProtocolsDI",
    "FlextProtocolsHandler",
    "FlextProtocolsLogging",
    "FlextProtocolsRegistry",
    "FlextProtocolsResult",
    "FlextProtocolsService",
    "FlextRegistry",
    "FlextResult",
    "FlextRuntime",
    "FlextService",
    "FlextSettings",
    "FlextTypes",
    "FlextTypesCore",
    "FlextTypesServices",
    "FlextTypesValidation",
    "FlextTypingBase",
    "FlextTypingContainers",
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
    "FlextUtilitiesGuardsEnsure",
    "FlextUtilitiesGuardsType",
    "FlextUtilitiesGuardsTypeCore",
    "FlextUtilitiesGuardsTypeModel",
    "FlextUtilitiesGuardsTypeProtocol",
    "FlextUtilitiesGuardsValidation",
    "FlextUtilitiesMapper",
    "FlextUtilitiesModel",
    "FlextUtilitiesPagination",
    "FlextUtilitiesParser",
    "FlextUtilitiesReliability",
    "FlextUtilitiesResultHelpers",
    "FlextUtilitiesText",
    "MessageT_contra",
    "Metadata",
    "P",
    "R",
    "RateLimiterManager",
    "ResultT",
    "RetryPolicy",
    "T",
    "TRuntime",
    "TV_co",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
    "TimeoutEnforcer",
    "U",
    "ValidatedParams",
    "ValidatedReturn",
    "__all__",
    "__author__",
    "__author_email__",
    "__description__",
    "__license__",
    "__title__",
    "__url__",
    "__version__",
    "__version_info__",
    "_constants",
    "_decorators",
    "_dispatcher",
    "_models",
    "_protocols",
    "_typings",
    "_utilities",
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
    "x",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
