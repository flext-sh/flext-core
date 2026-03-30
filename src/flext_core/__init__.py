# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Flext core package."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

from flext_core.__version__ import (
    FlextVersion,
    __author__,
    __author_email__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
    __version_info__,
)
from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core import (
        _constants,
        _models,
        _protocols,
        _typings,
        _utilities,
        constants,
        container,
        context,
        decorators,
        dispatcher,
        errors,
        exceptions,
        handlers,
        lazy,
        loggings,
        mixins,
        models,
        protocols,
        registry,
        result,
        runtime,
        service,
        settings,
        typings,
        utilities,
    )
    from flext_core._constants import (
        base,
        cqrs,
        domain,
        infrastructure,
        platform,
        validation,
    )
    from flext_core._constants.base import FlextConstantsBase
    from flext_core._constants.cqrs import FlextConstantsCqrs
    from flext_core._constants.domain import FlextConstantsDomain
    from flext_core._constants.errors import FlextConstantsErrors
    from flext_core._constants.infrastructure import FlextConstantsInfrastructure
    from flext_core._constants.mixins import FlextConstantsMixins
    from flext_core._constants.platform import FlextConstantsPlatform
    from flext_core._constants.settings import FlextConstantsSettings
    from flext_core._constants.validation import FlextConstantsValidation
    from flext_core._models import (
        collections,
        containers,
        domain_event,
        entity,
        exception_params,
        generic,
        handler,
    )
    from flext_core._models._context._data import FlextModelsContextData
    from flext_core._models._context._export import FlextModelsContextExport
    from flext_core._models._context._metadata import FlextModelsContextMetadata
    from flext_core._models._context._proxy_var import FlextModelsContextProxyVar
    from flext_core._models._context._scope import FlextModelsContextScope
    from flext_core._models._context._tokens import FlextModelsContextTokens
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
    from flext_core._models.errors import FlextModelsErrors
    from flext_core._models.exception_params import FlextModelsExceptionParams
    from flext_core._models.generic import FlextGenericModels
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.service import FlextModelsService
    from flext_core._models.settings import FlextModelsConfig
    from flext_core._protocols import config, logging
    from flext_core._protocols.base import FlextProtocolsBase
    from flext_core._protocols.config import FlextProtocolsConfig
    from flext_core._protocols.container import FlextProtocolsContainer
    from flext_core._protocols.context import FlextProtocolsContext
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._protocols.logging import FlextProtocolsLogging
    from flext_core._protocols.registry import FlextProtocolsRegistry
    from flext_core._protocols.result import FlextProtocolsResult
    from flext_core._protocols.service import FlextProtocolsService
    from flext_core._typings import core, generics, services
    from flext_core._typings.base import FlextTypingBase
    from flext_core._typings.containers import FlextTypingContainers
    from flext_core._typings.core import FlextTypesCore
    from flext_core._typings.generics import (
        TV,
        EnumT,
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
    from flext_core._typings.services import FlextTypesServices
    from flext_core._typings.validation import FlextTypesValidation
    from flext_core._utilities import (
        args,
        cache,
        checker,
        collection,
        configuration,
        conversion,
        deprecation,
        discovery,
        enum,
        file_ops,
        generators,
        guards,
        guards_ensure,
        guards_type,
        guards_type_core,
        guards_type_model,
        guards_type_protocol,
        mapper,
        model,
        pagination,
        parser,
        reliability,
        result_helpers,
        text,
    )
    from flext_core._utilities.args import FlextUtilitiesArgs
    from flext_core._utilities.cache import FlextUtilitiesCache
    from flext_core._utilities.checker import FlextUtilitiesChecker
    from flext_core._utilities.collection import FlextUtilitiesCollection
    from flext_core._utilities.configuration import FlextUtilitiesConfiguration
    from flext_core._utilities.context import FlextUtilitiesContext
    from flext_core._utilities.conversion import FlextUtilitiesConversion
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
    from flext_core.constants import FlextConstants, FlextConstants as c
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecorators, FlextDecorators as d
    from flext_core.dispatcher import FlextDispatcher
    from flext_core.errors import FlextError, FlextErrorDomain
    from flext_core.exceptions import FlextExceptions, FlextExceptions as e
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
    from flext_core.typings import BaseModel, FlextTypes, FlextTypes as t
    from flext_core.utilities import FlextUtilities, FlextUtilities as u

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "BaseModel": ["flext_core.typings", "BaseModel"],
    "EnumT": ["flext_core._typings.generics", "EnumT"],
    "FlextConstants": ["flext_core.constants", "FlextConstants"],
    "FlextConstantsBase": ["flext_core._constants.base", "FlextConstantsBase"],
    "FlextConstantsCqrs": ["flext_core._constants.cqrs", "FlextConstantsCqrs"],
    "FlextConstantsDomain": ["flext_core._constants.domain", "FlextConstantsDomain"],
    "FlextConstantsErrors": ["flext_core._constants.errors", "FlextConstantsErrors"],
    "FlextConstantsInfrastructure": [
        "flext_core._constants.infrastructure",
        "FlextConstantsInfrastructure",
    ],
    "FlextConstantsMixins": ["flext_core._constants.mixins", "FlextConstantsMixins"],
    "FlextConstantsPlatform": [
        "flext_core._constants.platform",
        "FlextConstantsPlatform",
    ],
    "FlextConstantsSettings": [
        "flext_core._constants.settings",
        "FlextConstantsSettings",
    ],
    "FlextConstantsValidation": [
        "flext_core._constants.validation",
        "FlextConstantsValidation",
    ],
    "FlextContainer": ["flext_core.container", "FlextContainer"],
    "FlextContext": ["flext_core.context", "FlextContext"],
    "FlextDecorators": ["flext_core.decorators", "FlextDecorators"],
    "FlextDispatcher": ["flext_core.dispatcher", "FlextDispatcher"],
    "FlextError": ["flext_core.errors", "FlextError"],
    "FlextErrorDomain": ["flext_core.errors", "FlextErrorDomain"],
    "FlextExceptions": ["flext_core.exceptions", "FlextExceptions"],
    "FlextGenericModels": ["flext_core._models.generic", "FlextGenericModels"],
    "FlextHandlers": ["flext_core.handlers", "FlextHandlers"],
    "FlextLogger": ["flext_core.loggings", "FlextLogger"],
    "FlextMixins": ["flext_core.mixins", "FlextMixins"],
    "FlextModelFoundation": ["flext_core._models.base", "FlextModelFoundation"],
    "FlextModels": ["flext_core.models", "FlextModels"],
    "FlextModelsCollections": [
        "flext_core._models.collections",
        "FlextModelsCollections",
    ],
    "FlextModelsConfig": ["flext_core._models.settings", "FlextModelsConfig"],
    "FlextModelsContainer": ["flext_core._models.container", "FlextModelsContainer"],
    "FlextModelsContainers": ["flext_core._models.containers", "FlextModelsContainers"],
    "FlextModelsContext": ["flext_core._models.context", "FlextModelsContext"],
    "FlextModelsContextData": [
        "flext_core._models._context._data",
        "FlextModelsContextData",
    ],
    "FlextModelsContextExport": [
        "flext_core._models._context._export",
        "FlextModelsContextExport",
    ],
    "FlextModelsContextMetadata": [
        "flext_core._models._context._metadata",
        "FlextModelsContextMetadata",
    ],
    "FlextModelsContextProxyVar": [
        "flext_core._models._context._proxy_var",
        "FlextModelsContextProxyVar",
    ],
    "FlextModelsContextScope": [
        "flext_core._models._context._scope",
        "FlextModelsContextScope",
    ],
    "FlextModelsContextTokens": [
        "flext_core._models._context._tokens",
        "FlextModelsContextTokens",
    ],
    "FlextModelsCqrs": ["flext_core._models.cqrs", "FlextModelsCqrs"],
    "FlextModelsDecorators": ["flext_core._models.decorators", "FlextModelsDecorators"],
    "FlextModelsDispatcher": ["flext_core._models.dispatcher", "FlextModelsDispatcher"],
    "FlextModelsDomainEvent": [
        "flext_core._models.domain_event",
        "FlextModelsDomainEvent",
    ],
    "FlextModelsEntity": ["flext_core._models.entity", "FlextModelsEntity"],
    "FlextModelsErrors": ["flext_core._models.errors", "FlextModelsErrors"],
    "FlextModelsExceptionParams": [
        "flext_core._models.exception_params",
        "FlextModelsExceptionParams",
    ],
    "FlextModelsHandler": ["flext_core._models.handler", "FlextModelsHandler"],
    "FlextModelsService": ["flext_core._models.service", "FlextModelsService"],
    "FlextProtocols": ["flext_core.protocols", "FlextProtocols"],
    "FlextProtocolsBase": ["flext_core._protocols.base", "FlextProtocolsBase"],
    "FlextProtocolsConfig": ["flext_core._protocols.config", "FlextProtocolsConfig"],
    "FlextProtocolsContainer": [
        "flext_core._protocols.container",
        "FlextProtocolsContainer",
    ],
    "FlextProtocolsContext": ["flext_core._protocols.context", "FlextProtocolsContext"],
    "FlextProtocolsHandler": ["flext_core._protocols.handler", "FlextProtocolsHandler"],
    "FlextProtocolsLogging": ["flext_core._protocols.logging", "FlextProtocolsLogging"],
    "FlextProtocolsRegistry": [
        "flext_core._protocols.registry",
        "FlextProtocolsRegistry",
    ],
    "FlextProtocolsResult": ["flext_core._protocols.result", "FlextProtocolsResult"],
    "FlextProtocolsService": ["flext_core._protocols.service", "FlextProtocolsService"],
    "FlextRegistry": ["flext_core.registry", "FlextRegistry"],
    "FlextResult": ["flext_core.result", "FlextResult"],
    "FlextRuntime": ["flext_core.runtime", "FlextRuntime"],
    "FlextService": ["flext_core.service", "FlextService"],
    "FlextSettings": ["flext_core.settings", "FlextSettings"],
    "FlextTypes": ["flext_core.typings", "FlextTypes"],
    "FlextTypesCore": ["flext_core._typings.core", "FlextTypesCore"],
    "FlextTypesServices": ["flext_core._typings.services", "FlextTypesServices"],
    "FlextTypesValidation": ["flext_core._typings.validation", "FlextTypesValidation"],
    "FlextTypingBase": ["flext_core._typings.base", "FlextTypingBase"],
    "FlextTypingContainers": [
        "flext_core._typings.containers",
        "FlextTypingContainers",
    ],
    "FlextUtilities": ["flext_core.utilities", "FlextUtilities"],
    "FlextUtilitiesArgs": ["flext_core._utilities.args", "FlextUtilitiesArgs"],
    "FlextUtilitiesCache": ["flext_core._utilities.cache", "FlextUtilitiesCache"],
    "FlextUtilitiesChecker": ["flext_core._utilities.checker", "FlextUtilitiesChecker"],
    "FlextUtilitiesCollection": [
        "flext_core._utilities.collection",
        "FlextUtilitiesCollection",
    ],
    "FlextUtilitiesConfiguration": [
        "flext_core._utilities.configuration",
        "FlextUtilitiesConfiguration",
    ],
    "FlextUtilitiesContext": ["flext_core._utilities.context", "FlextUtilitiesContext"],
    "FlextUtilitiesConversion": [
        "flext_core._utilities.conversion",
        "FlextUtilitiesConversion",
    ],
    "FlextUtilitiesDiscovery": [
        "flext_core._utilities.discovery",
        "FlextUtilitiesDiscovery",
    ],
    "FlextUtilitiesDomain": ["flext_core._utilities.domain", "FlextUtilitiesDomain"],
    "FlextUtilitiesEnum": ["flext_core._utilities.enum", "FlextUtilitiesEnum"],
    "FlextUtilitiesFileOps": [
        "flext_core._utilities.file_ops",
        "FlextUtilitiesFileOps",
    ],
    "FlextUtilitiesGenerators": [
        "flext_core._utilities.generators",
        "FlextUtilitiesGenerators",
    ],
    "FlextUtilitiesGuards": ["flext_core._utilities.guards", "FlextUtilitiesGuards"],
    "FlextUtilitiesGuardsEnsure": [
        "flext_core._utilities.guards_ensure",
        "FlextUtilitiesGuardsEnsure",
    ],
    "FlextUtilitiesGuardsType": [
        "flext_core._utilities.guards_type",
        "FlextUtilitiesGuardsType",
    ],
    "FlextUtilitiesGuardsTypeCore": [
        "flext_core._utilities.guards_type_core",
        "FlextUtilitiesGuardsTypeCore",
    ],
    "FlextUtilitiesGuardsTypeModel": [
        "flext_core._utilities.guards_type_model",
        "FlextUtilitiesGuardsTypeModel",
    ],
    "FlextUtilitiesGuardsTypeProtocol": [
        "flext_core._utilities.guards_type_protocol",
        "FlextUtilitiesGuardsTypeProtocol",
    ],
    "FlextUtilitiesMapper": ["flext_core._utilities.mapper", "FlextUtilitiesMapper"],
    "FlextUtilitiesModel": ["flext_core._utilities.model", "FlextUtilitiesModel"],
    "FlextUtilitiesPagination": [
        "flext_core._utilities.pagination",
        "FlextUtilitiesPagination",
    ],
    "FlextUtilitiesParser": ["flext_core._utilities.parser", "FlextUtilitiesParser"],
    "FlextUtilitiesReliability": [
        "flext_core._utilities.reliability",
        "FlextUtilitiesReliability",
    ],
    "FlextUtilitiesResultHelpers": [
        "flext_core._utilities.result_helpers",
        "FlextUtilitiesResultHelpers",
    ],
    "FlextUtilitiesText": ["flext_core._utilities.text", "FlextUtilitiesText"],
    "MessageT_contra": ["flext_core._typings.generics", "MessageT_contra"],
    "P": ["flext_core._typings.generics", "P"],
    "R": ["flext_core._typings.generics", "R"],
    "ResultT": ["flext_core._typings.generics", "ResultT"],
    "T": ["flext_core._typings.generics", "T"],
    "TRuntime": ["flext_core._typings.generics", "TRuntime"],
    "TV": ["flext_core._typings.generics", "TV"],
    "TV_co": ["flext_core._typings.generics", "TV_co"],
    "T_Model": ["flext_core._typings.generics", "T_Model"],
    "T_Namespace": ["flext_core._typings.generics", "T_Namespace"],
    "T_Settings": ["flext_core._typings.generics", "T_Settings"],
    "T_co": ["flext_core._typings.generics", "T_co"],
    "T_contra": ["flext_core._typings.generics", "T_contra"],
    "U": ["flext_core._typings.generics", "U"],
    "_constants": ["flext_core._constants", ""],
    "_models": ["flext_core._models", ""],
    "_protocols": ["flext_core._protocols", ""],
    "_typings": ["flext_core._typings", ""],
    "_utilities": ["flext_core._utilities", ""],
    "args": ["flext_core._utilities.args", ""],
    "base": ["flext_core._constants.base", ""],
    "c": ["flext_core.constants", "FlextConstants"],
    "cache": ["flext_core._utilities.cache", ""],
    "checker": ["flext_core._utilities.checker", ""],
    "collection": ["flext_core._utilities.collection", ""],
    "collections": ["flext_core._models.collections", ""],
    "config": ["flext_core._protocols.config", ""],
    "configuration": ["flext_core._utilities.configuration", ""],
    "constants": ["flext_core.constants", ""],
    "container": ["flext_core.container", ""],
    "containers": ["flext_core._models.containers", ""],
    "context": ["flext_core.context", ""],
    "conversion": ["flext_core._utilities.conversion", ""],
    "core": ["flext_core._typings.core", ""],
    "cqrs": ["flext_core._constants.cqrs", ""],
    "d": ["flext_core.decorators", "FlextDecorators"],
    "decorators": ["flext_core.decorators", ""],
    "deprecation": ["flext_core._utilities.deprecation", ""],
    "discovery": ["flext_core._utilities.discovery", ""],
    "dispatcher": ["flext_core.dispatcher", ""],
    "domain": ["flext_core._constants.domain", ""],
    "domain_event": ["flext_core._models.domain_event", ""],
    "e": ["flext_core.exceptions", "FlextExceptions"],
    "entity": ["flext_core._models.entity", ""],
    "enum": ["flext_core._utilities.enum", ""],
    "errors": ["flext_core.errors", ""],
    "exception_params": ["flext_core._models.exception_params", ""],
    "exceptions": ["flext_core.exceptions", ""],
    "file_ops": ["flext_core._utilities.file_ops", ""],
    "generators": ["flext_core._utilities.generators", ""],
    "generic": ["flext_core._models.generic", ""],
    "generics": ["flext_core._typings.generics", ""],
    "guards": ["flext_core._utilities.guards", ""],
    "guards_ensure": ["flext_core._utilities.guards_ensure", ""],
    "guards_type": ["flext_core._utilities.guards_type", ""],
    "guards_type_core": ["flext_core._utilities.guards_type_core", ""],
    "guards_type_model": ["flext_core._utilities.guards_type_model", ""],
    "guards_type_protocol": ["flext_core._utilities.guards_type_protocol", ""],
    "h": ["flext_core.handlers", "FlextHandlers"],
    "handler": ["flext_core._models.handler", ""],
    "handlers": ["flext_core.handlers", ""],
    "infrastructure": ["flext_core._constants.infrastructure", ""],
    "lazy": ["flext_core.lazy", ""],
    "logging": ["flext_core._protocols.logging", ""],
    "loggings": ["flext_core.loggings", ""],
    "m": ["flext_core.models", "FlextModels"],
    "mapper": ["flext_core._utilities.mapper", ""],
    "mixins": ["flext_core.mixins", ""],
    "model": ["flext_core._utilities.model", ""],
    "models": ["flext_core.models", ""],
    "p": ["flext_core.protocols", "FlextProtocols"],
    "pagination": ["flext_core._utilities.pagination", ""],
    "parser": ["flext_core._utilities.parser", ""],
    "platform": ["flext_core._constants.platform", ""],
    "protocols": ["flext_core.protocols", ""],
    "r": ["flext_core.result", "FlextResult"],
    "registry": ["flext_core.registry", ""],
    "reliability": ["flext_core._utilities.reliability", ""],
    "result": ["flext_core.result", ""],
    "result_helpers": ["flext_core._utilities.result_helpers", ""],
    "runtime": ["flext_core.runtime", ""],
    "s": ["flext_core.service", "FlextService"],
    "service": ["flext_core.service", ""],
    "services": ["flext_core._typings.services", ""],
    "settings": ["flext_core.settings", ""],
    "t": ["flext_core.typings", "FlextTypes"],
    "text": ["flext_core._utilities.text", ""],
    "typings": ["flext_core.typings", ""],
    "u": ["flext_core.utilities", "FlextUtilities"],
    "utilities": ["flext_core.utilities", ""],
    "validation": ["flext_core._constants.validation", ""],
    "x": ["flext_core.mixins", "FlextMixins"],
}

__all__ = [
    "TV",
    "BaseModel",
    "EnumT",
    "FlextConstants",
    "FlextConstantsBase",
    "FlextConstantsCqrs",
    "FlextConstantsDomain",
    "FlextConstantsErrors",
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
    "FlextErrorDomain",
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
    "FlextModelsContextData",
    "FlextModelsContextExport",
    "FlextModelsContextMetadata",
    "FlextModelsContextProxyVar",
    "FlextModelsContextScope",
    "FlextModelsContextTokens",
    "FlextModelsCqrs",
    "FlextModelsDecorators",
    "FlextModelsDispatcher",
    "FlextModelsDomainEvent",
    "FlextModelsEntity",
    "FlextModelsErrors",
    "FlextModelsExceptionParams",
    "FlextModelsHandler",
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
    "FlextVersion",
    "MessageT_contra",
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
    "args",
    "base",
    "c",
    "cache",
    "checker",
    "collection",
    "collections",
    "config",
    "configuration",
    "constants",
    "container",
    "containers",
    "context",
    "conversion",
    "core",
    "cqrs",
    "d",
    "decorators",
    "deprecation",
    "discovery",
    "dispatcher",
    "domain",
    "domain_event",
    "e",
    "entity",
    "enum",
    "errors",
    "exception_params",
    "exceptions",
    "file_ops",
    "generators",
    "generic",
    "generics",
    "guards",
    "guards_ensure",
    "guards_type",
    "guards_type_core",
    "guards_type_model",
    "guards_type_protocol",
    "h",
    "handler",
    "handlers",
    "infrastructure",
    "lazy",
    "logging",
    "loggings",
    "m",
    "mapper",
    "mixins",
    "model",
    "models",
    "p",
    "pagination",
    "parser",
    "platform",
    "protocols",
    "r",
    "registry",
    "reliability",
    "result",
    "result_helpers",
    "runtime",
    "s",
    "service",
    "services",
    "settings",
    "t",
    "text",
    "typings",
    "u",
    "utilities",
    "validation",
    "x",
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
