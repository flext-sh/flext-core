# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Flext core package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core._utilities.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core import _constants, _models, _protocols, _typings, _utilities
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
    from flext_core._protocols.container import FlextProtocolsContainer
    from flext_core._protocols.context import FlextProtocolsContext
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
    from flext_core._utilities.discovery import FlextUtilitiesDiscovery
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
        FlextConstants as c,
    )
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecorators, FlextDecorators as d
    from flext_core.dispatcher import FlextDispatcher
    from flext_core.errors import ErrorDomain, FlextError
    from flext_core.exceptions import FlextExceptions, FlextExceptions as e, Metadata
    from flext_core.handlers import FlextHandlers, FlextHandlers as h
    from flext_core.loggings import FlextLogger
    from flext_core.mixins import FlextMixins, FlextMixins as x
    from flext_core.models import FlextModels, FlextModels as m
    from flext_core.protocols import FlextProtocols, FlextProtocols as p
    from flext_core.registry import FlextRegistry
    from flext_core.result import FlextResult, FlextResult as r
    from flext_core.runtime import FlextRuntime
    from flext_core.service import FlextService, FlextService as s
    from flext_core.settings import FlextSettings
    from flext_core.typings import (
        TV,
        BaseModel,
        EnumT,
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
        TRuntime,
        TV_co,
        U,
    )
    from flext_core.utilities import FlextUtilities, FlextUtilities as u

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseModel": ("flext_core.typings", "BaseModel"),
    "EnumT": ("flext_core.typings", "EnumT"),
    "ErrorDomain": ("flext_core.errors", "ErrorDomain"),
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
    "FlextProtocolsContainer": (
        "flext_core._protocols.container",
        "FlextProtocolsContainer",
    ),
    "FlextProtocolsContext": ("flext_core._protocols.context", "FlextProtocolsContext"),
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
    "FlextUtilitiesDiscovery": (
        "flext_core._utilities.discovery",
        "FlextUtilitiesDiscovery",
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
    "ResultT": ("flext_core.typings", "ResultT"),
    "T": ("flext_core.typings", "T"),
    "TRuntime": ("flext_core.typings", "TRuntime"),
    "TV": ("flext_core.typings", "TV"),
    "TV_co": ("flext_core.typings", "TV_co"),
    "T_Model": ("flext_core.typings", "T_Model"),
    "T_Namespace": ("flext_core.typings", "T_Namespace"),
    "T_Settings": ("flext_core.typings", "T_Settings"),
    "T_co": ("flext_core.typings", "T_co"),
    "T_contra": ("flext_core.typings", "T_contra"),
    "U": ("flext_core.typings", "U"),
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
    "_models": ("flext_core._models", ""),
    "_protocols": ("flext_core._protocols", ""),
    "_typings": ("flext_core._typings", ""),
    "_utilities": ("flext_core._utilities", ""),
    "c": ("flext_core.constants", "FlextConstants"),
    "d": ("flext_core.decorators", "FlextDecorators"),
    "e": ("flext_core.exceptions", "FlextExceptions"),
    "h": ("flext_core.handlers", "FlextHandlers"),
    "m": ("flext_core.models", "FlextModels"),
    "p": ("flext_core.protocols", "FlextProtocols"),
    "r": ("flext_core.result", "FlextResult"),
    "s": ("flext_core.service", "FlextService"),
    "t": ("flext_core.typings", "FlextTypes"),
    "u": ("flext_core.utilities", "FlextUtilities"),
    "x": ("flext_core.mixins", "FlextMixins"),
}

__all__ = [
    "PROJECT_KIND_APPLICATION",
    "PROJECT_KIND_LIBRARY",
    "PROJECT_KIND_SERVICE",
    "TV",
    "BaseModel",
    "EnumT",
    "ErrorDomain",
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
    "FlextProtocolsContainer",
    "FlextProtocolsContext",
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
    "FlextUtilitiesDiscovery",
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
    "ResultT",
    "T",
    "TRuntime",
    "TV_co",
    "T_Model",
    "T_Namespace",
    "T_Settings",
    "T_co",
    "T_contra",
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
    "_constants",
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
