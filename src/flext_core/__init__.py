# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Flext core package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.__version__ import (
    FlextVersion as FlextVersion,
    __author__ as __author__,
    __author_email__ as __author_email__,
    __description__ as __description__,
    __license__ as __license__,
    __title__ as __title__,
    __url__ as __url__,
    __version__ as __version__,
    __version_info__ as __version_info__,
)
from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core import (
        _constants as _constants,
        _models as _models,
        _protocols as _protocols,
        _typings as _typings,
        _utilities as _utilities,
        constants as constants,
        container as container,
        context as context,
        decorators as decorators,
        dispatcher as dispatcher,
        errors as errors,
        exceptions as exceptions,
        handlers as handlers,
        lazy as lazy,
        loggings as loggings,
        mixins as mixins,
        models as models,
        protocols as protocols,
        registry as registry,
        result as result,
        runtime as runtime,
        service as service,
        settings as settings,
        typings as typings,
        utilities as utilities,
    )
    from flext_core._constants import (
        base as base,
        cqrs as cqrs,
        domain as domain,
        infrastructure as infrastructure,
        platform as platform,
        validation as validation,
    )
    from flext_core._constants.base import FlextConstantsBase as FlextConstantsBase
    from flext_core._constants.cqrs import FlextConstantsCqrs as FlextConstantsCqrs
    from flext_core._constants.domain import (
        FlextConstantsDomain as FlextConstantsDomain,
    )
    from flext_core._constants.errors import (
        FlextConstantsErrors as FlextConstantsErrors,
    )
    from flext_core._constants.infrastructure import (
        FlextConstantsInfrastructure as FlextConstantsInfrastructure,
    )
    from flext_core._constants.mixins import (
        FlextConstantsMixins as FlextConstantsMixins,
    )
    from flext_core._constants.platform import (
        FlextConstantsPlatform as FlextConstantsPlatform,
    )
    from flext_core._constants.settings import (
        FlextConstantsSettings as FlextConstantsSettings,
    )
    from flext_core._constants.validation import (
        FlextConstantsValidation as FlextConstantsValidation,
    )
    from flext_core._models import (
        collections as collections,
        containers as containers,
        domain_event as domain_event,
        entity as entity,
        exception_params as exception_params,
        generic as generic,
        handler as handler,
    )
    from flext_core._models._context._data import (
        FlextModelsContextData as FlextModelsContextData,
    )
    from flext_core._models._context._export import (
        FlextModelsContextExport as FlextModelsContextExport,
    )
    from flext_core._models._context._metadata import (
        FlextModelsContextMetadata as FlextModelsContextMetadata,
    )
    from flext_core._models._context._proxy_var import (
        FlextModelsContextProxyVar as FlextModelsContextProxyVar,
    )
    from flext_core._models._context._scope import (
        FlextModelsContextScope as FlextModelsContextScope,
    )
    from flext_core._models._context._tokens import (
        FlextModelsContextTokens as FlextModelsContextTokens,
    )
    from flext_core._models.base import FlextModelFoundation as FlextModelFoundation
    from flext_core._models.collections import (
        FlextModelsCollections as FlextModelsCollections,
    )
    from flext_core._models.container import (
        FlextModelsContainer as FlextModelsContainer,
    )
    from flext_core._models.containers import (
        FlextModelsContainers as FlextModelsContainers,
    )
    from flext_core._models.context import FlextModelsContext as FlextModelsContext
    from flext_core._models.cqrs import FlextModelsCqrs as FlextModelsCqrs
    from flext_core._models.decorators import (
        FlextModelsDecorators as FlextModelsDecorators,
    )
    from flext_core._models.dispatcher import (
        FlextModelsDispatcher as FlextModelsDispatcher,
    )
    from flext_core._models.domain_event import (
        FlextModelsDomainEvent as FlextModelsDomainEvent,
    )
    from flext_core._models.entity import FlextModelsEntity as FlextModelsEntity
    from flext_core._models.errors import FlextModelsErrors as FlextModelsErrors
    from flext_core._models.exception_params import (
        FlextModelsExceptionParams as FlextModelsExceptionParams,
    )
    from flext_core._models.generic import FlextGenericModels as FlextGenericModels
    from flext_core._models.handler import FlextModelsHandler as FlextModelsHandler
    from flext_core._models.service import FlextModelsService as FlextModelsService
    from flext_core._models.settings import FlextModelsConfig as FlextModelsConfig
    from flext_core._protocols import config as config, logging as logging
    from flext_core._protocols.base import FlextProtocolsBase as FlextProtocolsBase
    from flext_core._protocols.config import (
        FlextProtocolsConfig as FlextProtocolsConfig,
    )
    from flext_core._protocols.container import (
        FlextProtocolsContainer as FlextProtocolsContainer,
    )
    from flext_core._protocols.context import (
        FlextProtocolsContext as FlextProtocolsContext,
    )
    from flext_core._protocols.handler import (
        FlextProtocolsHandler as FlextProtocolsHandler,
    )
    from flext_core._protocols.logging import (
        FlextProtocolsLogging as FlextProtocolsLogging,
    )
    from flext_core._protocols.registry import (
        FlextProtocolsRegistry as FlextProtocolsRegistry,
    )
    from flext_core._protocols.result import (
        FlextProtocolsResult as FlextProtocolsResult,
    )
    from flext_core._protocols.service import (
        FlextProtocolsService as FlextProtocolsService,
    )
    from flext_core._typings import (
        core as core,
        generics as generics,
        services as services,
    )
    from flext_core._typings.base import FlextTypingBase as FlextTypingBase
    from flext_core._typings.containers import (
        FlextTypingContainers as FlextTypingContainers,
    )
    from flext_core._typings.core import FlextTypesCore as FlextTypesCore
    from flext_core._typings.generics import (
        TV as TV,
        EnumT as EnumT,
        MessageT_contra as MessageT_contra,
        P as P,
        R as R,
        ResultT as ResultT,
        T as T,
        T_co as T_co,
        T_contra as T_contra,
        T_Model as T_Model,
        T_Namespace as T_Namespace,
        T_Settings as T_Settings,
        TRuntime as TRuntime,
        TV_co as TV_co,
        U as U,
    )
    from flext_core._typings.services import FlextTypesServices as FlextTypesServices
    from flext_core._typings.validation import (
        FlextTypesValidation as FlextTypesValidation,
    )
    from flext_core._utilities import (
        args as args,
        cache as cache,
        checker as checker,
        collection as collection,
        configuration as configuration,
        conversion as conversion,
        deprecation as deprecation,
        discovery as discovery,
        enum as enum,
        file_ops as file_ops,
        generators as generators,
        guards as guards,
        guards_ensure as guards_ensure,
        guards_type as guards_type,
        guards_type_core as guards_type_core,
        guards_type_model as guards_type_model,
        guards_type_protocol as guards_type_protocol,
        mapper as mapper,
        model as model,
        pagination as pagination,
        parser as parser,
        reliability as reliability,
        result_helpers as result_helpers,
        text as text,
    )
    from flext_core._utilities.args import FlextUtilitiesArgs as FlextUtilitiesArgs
    from flext_core._utilities.cache import FlextUtilitiesCache as FlextUtilitiesCache
    from flext_core._utilities.checker import (
        FlextUtilitiesChecker as FlextUtilitiesChecker,
    )
    from flext_core._utilities.collection import (
        FlextUtilitiesCollection as FlextUtilitiesCollection,
    )
    from flext_core._utilities.configuration import (
        FlextUtilitiesConfiguration as FlextUtilitiesConfiguration,
    )
    from flext_core._utilities.context import (
        FlextUtilitiesContext as FlextUtilitiesContext,
    )
    from flext_core._utilities.conversion import (
        FlextUtilitiesConversion as FlextUtilitiesConversion,
    )
    from flext_core._utilities.discovery import (
        FlextUtilitiesDiscovery as FlextUtilitiesDiscovery,
    )
    from flext_core._utilities.domain import (
        FlextUtilitiesDomain as FlextUtilitiesDomain,
    )
    from flext_core._utilities.enum import FlextUtilitiesEnum as FlextUtilitiesEnum
    from flext_core._utilities.file_ops import (
        FlextUtilitiesFileOps as FlextUtilitiesFileOps,
    )
    from flext_core._utilities.generators import (
        FlextUtilitiesGenerators as FlextUtilitiesGenerators,
    )
    from flext_core._utilities.guards import (
        FlextUtilitiesGuards as FlextUtilitiesGuards,
    )
    from flext_core._utilities.guards_ensure import (
        FlextUtilitiesGuardsEnsure as FlextUtilitiesGuardsEnsure,
    )
    from flext_core._utilities.guards_type import (
        FlextUtilitiesGuardsType as FlextUtilitiesGuardsType,
    )
    from flext_core._utilities.guards_type_core import (
        FlextUtilitiesGuardsTypeCore as FlextUtilitiesGuardsTypeCore,
    )
    from flext_core._utilities.guards_type_model import (
        FlextUtilitiesGuardsTypeModel as FlextUtilitiesGuardsTypeModel,
    )
    from flext_core._utilities.guards_type_protocol import (
        FlextUtilitiesGuardsTypeProtocol as FlextUtilitiesGuardsTypeProtocol,
    )
    from flext_core._utilities.mapper import (
        FlextUtilitiesMapper as FlextUtilitiesMapper,
    )
    from flext_core._utilities.model import FlextUtilitiesModel as FlextUtilitiesModel
    from flext_core._utilities.pagination import (
        FlextUtilitiesPagination as FlextUtilitiesPagination,
    )
    from flext_core._utilities.parser import (
        FlextUtilitiesParser as FlextUtilitiesParser,
    )
    from flext_core._utilities.reliability import (
        FlextUtilitiesReliability as FlextUtilitiesReliability,
    )
    from flext_core._utilities.result_helpers import (
        FlextUtilitiesResultHelpers as FlextUtilitiesResultHelpers,
    )
    from flext_core._utilities.text import FlextUtilitiesText as FlextUtilitiesText
    from flext_core.constants import (
        FlextConstants as FlextConstants,
        FlextConstants as c,
    )
    from flext_core.container import FlextContainer as FlextContainer
    from flext_core.context import FlextContext as FlextContext
    from flext_core.decorators import (
        FlextDecorators as FlextDecorators,
        FlextDecorators as d,
    )
    from flext_core.dispatcher import FlextDispatcher as FlextDispatcher
    from flext_core.errors import (
        FlextError as FlextError,
        FlextErrorDomain as FlextErrorDomain,
    )
    from flext_core.exceptions import (
        FlextExceptions as FlextExceptions,
        FlextExceptions as e,
    )
    from flext_core.handlers import FlextHandlers as FlextHandlers, FlextHandlers as h
    from flext_core.loggings import FlextLogger as FlextLogger
    from flext_core.mixins import FlextMixins as FlextMixins, FlextMixins as x
    from flext_core.models import FlextModels as FlextModels, FlextModels as m
    from flext_core.protocols import (
        FlextProtocols as FlextProtocols,
        FlextProtocols as p,
    )
    from flext_core.registry import FlextRegistry as FlextRegistry
    from flext_core.result import FlextResult as FlextResult, FlextResult as r
    from flext_core.runtime import FlextRuntime as FlextRuntime
    from flext_core.service import FlextService as FlextService, FlextService as s
    from flext_core.settings import FlextSettings as FlextSettings
    from flext_core.typings import (
        BaseModel as BaseModel,
        FlextTypes as FlextTypes,
        FlextTypes as t,
    )
    from flext_core.utilities import (
        FlextUtilities as FlextUtilities,
        FlextUtilities as u,
    )

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
    "install_lazy_exports": ["flext_core.lazy", "install_lazy_exports"],
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

_EXPORTS: Sequence[str] = [
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
    "TV",
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
    "install_lazy_exports",
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
