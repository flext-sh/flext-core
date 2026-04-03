# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Flext core package."""

from __future__ import annotations

import typing as _t

from flext_core.__version__ import *
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
from flext_core._constants.base import FlextConstantsBase
from flext_core._constants.cqrs import FlextConstantsCqrs
from flext_core._constants.domain import FlextConstantsDomain
from flext_core._constants.errors import FlextConstantsErrors
from flext_core._constants.infrastructure import FlextConstantsInfrastructure
from flext_core._constants.mixins import FlextConstantsMixins
from flext_core._constants.platform import FlextConstantsPlatform
from flext_core._constants.settings import FlextConstantsSettings
from flext_core._constants.validation import FlextConstantsValidation
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
from flext_core._utilities.parser import FlextUtilitiesParser
from flext_core._utilities.reliability import FlextUtilitiesReliability
from flext_core._utilities.result_helpers import FlextUtilitiesResultHelpers
from flext_core._utilities.text import FlextUtilitiesText
from flext_core.constants import FlextConstants, FlextConstants as c
from flext_core.container import FlextContainer
from flext_core.context import FlextContext
from flext_core.decorators import FlextDecorators, FlextDecorators as d
from flext_core.dispatcher import FlextDispatcher
from flext_core.exceptions import FlextExceptions, FlextExceptions as e
from flext_core.handlers import FlextHandlers, FlextHandlers as h
from flext_core.lazy import install_lazy_exports, merge_lazy_imports
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

if _t.TYPE_CHECKING:
    import flext_core._constants as _flext_core__constants

    _constants = _flext_core__constants
    import flext_core._constants.base as _flext_core__constants_base

    base = _flext_core__constants_base
    import flext_core._constants.cqrs as _flext_core__constants_cqrs

    cqrs = _flext_core__constants_cqrs
    import flext_core._constants.domain as _flext_core__constants_domain

    domain = _flext_core__constants_domain
    import flext_core._constants.errors as _flext_core__constants_errors

    errors = _flext_core__constants_errors
    import flext_core._constants.infrastructure as _flext_core__constants_infrastructure

    infrastructure = _flext_core__constants_infrastructure
    import flext_core._constants.platform as _flext_core__constants_platform

    platform = _flext_core__constants_platform
    import flext_core._constants.validation as _flext_core__constants_validation

    validation = _flext_core__constants_validation
    import flext_core._models as _flext_core__models

    _models = _flext_core__models
    import flext_core._models.collections as _flext_core__models_collections

    collections = _flext_core__models_collections
    import flext_core._models.containers as _flext_core__models_containers

    containers = _flext_core__models_containers
    import flext_core._models.domain_event as _flext_core__models_domain_event

    domain_event = _flext_core__models_domain_event
    import flext_core._models.entity as _flext_core__models_entity

    entity = _flext_core__models_entity
    import flext_core._models.exception_params as _flext_core__models_exception_params

    exception_params = _flext_core__models_exception_params
    import flext_core._models.generic as _flext_core__models_generic

    generic = _flext_core__models_generic
    import flext_core._models.handler as _flext_core__models_handler

    handler = _flext_core__models_handler
    import flext_core._protocols as _flext_core__protocols

    _protocols = _flext_core__protocols
    import flext_core._protocols.config as _flext_core__protocols_config

    config = _flext_core__protocols_config
    import flext_core._protocols.logging as _flext_core__protocols_logging

    logging = _flext_core__protocols_logging
    import flext_core._typings as _flext_core__typings

    _typings = _flext_core__typings
    import flext_core._typings.core as _flext_core__typings_core

    core = _flext_core__typings_core
    import flext_core._typings.generics as _flext_core__typings_generics

    generics = _flext_core__typings_generics
    import flext_core._typings.services as _flext_core__typings_services

    services = _flext_core__typings_services
    import flext_core._utilities as _flext_core__utilities

    _utilities = _flext_core__utilities
    import flext_core._utilities.args as _flext_core__utilities_args

    args = _flext_core__utilities_args
    import flext_core._utilities.cache as _flext_core__utilities_cache

    cache = _flext_core__utilities_cache
    import flext_core._utilities.checker as _flext_core__utilities_checker

    checker = _flext_core__utilities_checker
    import flext_core._utilities.collection as _flext_core__utilities_collection

    collection = _flext_core__utilities_collection
    import flext_core._utilities.configuration as _flext_core__utilities_configuration

    configuration = _flext_core__utilities_configuration
    import flext_core._utilities.conversion as _flext_core__utilities_conversion

    conversion = _flext_core__utilities_conversion
    import flext_core._utilities.discovery as _flext_core__utilities_discovery

    discovery = _flext_core__utilities_discovery
    import flext_core._utilities.enum as _flext_core__utilities_enum

    enum = _flext_core__utilities_enum
    import flext_core._utilities.file_ops as _flext_core__utilities_file_ops

    file_ops = _flext_core__utilities_file_ops
    import flext_core._utilities.generators as _flext_core__utilities_generators

    generators = _flext_core__utilities_generators
    import flext_core._utilities.guards as _flext_core__utilities_guards

    guards = _flext_core__utilities_guards
    import flext_core._utilities.guards_ensure as _flext_core__utilities_guards_ensure

    guards_ensure = _flext_core__utilities_guards_ensure
    import flext_core._utilities.guards_type as _flext_core__utilities_guards_type

    guards_type = _flext_core__utilities_guards_type
    import flext_core._utilities.guards_type_core as _flext_core__utilities_guards_type_core

    guards_type_core = _flext_core__utilities_guards_type_core
    import flext_core._utilities.guards_type_model as _flext_core__utilities_guards_type_model

    guards_type_model = _flext_core__utilities_guards_type_model
    import flext_core._utilities.guards_type_protocol as _flext_core__utilities_guards_type_protocol

    guards_type_protocol = _flext_core__utilities_guards_type_protocol
    import flext_core._utilities.mapper as _flext_core__utilities_mapper

    mapper = _flext_core__utilities_mapper
    import flext_core._utilities.model as _flext_core__utilities_model

    model = _flext_core__utilities_model
    import flext_core._utilities.parser as _flext_core__utilities_parser

    parser = _flext_core__utilities_parser
    import flext_core._utilities.reliability as _flext_core__utilities_reliability

    reliability = _flext_core__utilities_reliability
    import flext_core._utilities.result_helpers as _flext_core__utilities_result_helpers

    result_helpers = _flext_core__utilities_result_helpers
    import flext_core._utilities.text as _flext_core__utilities_text

    text = _flext_core__utilities_text
    import flext_core.constants as _flext_core_constants

    constants = _flext_core_constants
    import flext_core.container as _flext_core_container

    container = _flext_core_container
    import flext_core.context as _flext_core_context

    context = _flext_core_context
    import flext_core.decorators as _flext_core_decorators

    decorators = _flext_core_decorators
    import flext_core.dispatcher as _flext_core_dispatcher

    dispatcher = _flext_core_dispatcher
    import flext_core.exceptions as _flext_core_exceptions

    exceptions = _flext_core_exceptions
    import flext_core.handlers as _flext_core_handlers

    handlers = _flext_core_handlers
    import flext_core.lazy as _flext_core_lazy

    lazy = _flext_core_lazy
    import flext_core.loggings as _flext_core_loggings

    loggings = _flext_core_loggings
    import flext_core.mixins as _flext_core_mixins

    mixins = _flext_core_mixins
    import flext_core.models as _flext_core_models

    models = _flext_core_models
    import flext_core.protocols as _flext_core_protocols

    protocols = _flext_core_protocols
    import flext_core.registry as _flext_core_registry

    registry = _flext_core_registry
    import flext_core.result as _flext_core_result

    result = _flext_core_result
    import flext_core.runtime as _flext_core_runtime

    runtime = _flext_core_runtime
    import flext_core.service as _flext_core_service

    service = _flext_core_service
    import flext_core.settings as _flext_core_settings

    settings = _flext_core_settings
    import flext_core.typings as _flext_core_typings

    typings = _flext_core_typings
    import flext_core.utilities as _flext_core_utilities

    utilities = _flext_core_utilities

    _ = (
        BaseModel,
        EnumT,
        FlextConstants,
        FlextConstantsBase,
        FlextConstantsCqrs,
        FlextConstantsDomain,
        FlextConstantsErrors,
        FlextConstantsInfrastructure,
        FlextConstantsMixins,
        FlextConstantsPlatform,
        FlextConstantsSettings,
        FlextConstantsValidation,
        FlextContainer,
        FlextContext,
        FlextDecorators,
        FlextDispatcher,
        FlextExceptions,
        FlextGenericModels,
        FlextHandlers,
        FlextLogger,
        FlextMixins,
        FlextModelFoundation,
        FlextModels,
        FlextModelsCollections,
        FlextModelsConfig,
        FlextModelsContainer,
        FlextModelsContainers,
        FlextModelsContext,
        FlextModelsContextData,
        FlextModelsContextExport,
        FlextModelsContextMetadata,
        FlextModelsContextProxyVar,
        FlextModelsContextScope,
        FlextModelsContextTokens,
        FlextModelsCqrs,
        FlextModelsDecorators,
        FlextModelsDispatcher,
        FlextModelsDomainEvent,
        FlextModelsEntity,
        FlextModelsErrors,
        FlextModelsExceptionParams,
        FlextModelsHandler,
        FlextModelsService,
        FlextProtocols,
        FlextProtocolsBase,
        FlextProtocolsConfig,
        FlextProtocolsContainer,
        FlextProtocolsContext,
        FlextProtocolsHandler,
        FlextProtocolsLogging,
        FlextProtocolsRegistry,
        FlextProtocolsResult,
        FlextProtocolsService,
        FlextRegistry,
        FlextResult,
        FlextRuntime,
        FlextService,
        FlextSettings,
        FlextTypes,
        FlextTypesCore,
        FlextTypesServices,
        FlextTypesValidation,
        FlextTypingBase,
        FlextTypingContainers,
        FlextUtilities,
        FlextUtilitiesArgs,
        FlextUtilitiesCache,
        FlextUtilitiesChecker,
        FlextUtilitiesCollection,
        FlextUtilitiesConfiguration,
        FlextUtilitiesContext,
        FlextUtilitiesConversion,
        FlextUtilitiesDiscovery,
        FlextUtilitiesDomain,
        FlextUtilitiesEnum,
        FlextUtilitiesFileOps,
        FlextUtilitiesGenerators,
        FlextUtilitiesGuards,
        FlextUtilitiesGuardsEnsure,
        FlextUtilitiesGuardsType,
        FlextUtilitiesGuardsTypeCore,
        FlextUtilitiesGuardsTypeModel,
        FlextUtilitiesGuardsTypeProtocol,
        FlextUtilitiesMapper,
        FlextUtilitiesModel,
        FlextUtilitiesParser,
        FlextUtilitiesReliability,
        FlextUtilitiesResultHelpers,
        FlextUtilitiesText,
        FlextVersion,
        MessageT_contra,
        P,
        R,
        ResultT,
        T,
        TRuntime,
        TV,
        TV_co,
        T_Model,
        T_Namespace,
        T_Settings,
        T_co,
        T_contra,
        U,
        __author__,
        __author_email__,
        __description__,
        __license__,
        __title__,
        __url__,
        __version__,
        __version_info__,
        _constants,
        _models,
        _protocols,
        _typings,
        _utilities,
        args,
        base,
        c,
        cache,
        checker,
        collection,
        collections,
        config,
        configuration,
        constants,
        container,
        containers,
        context,
        conversion,
        core,
        cqrs,
        d,
        decorators,
        discovery,
        dispatcher,
        domain,
        domain_event,
        e,
        entity,
        enum,
        errors,
        exception_params,
        exceptions,
        file_ops,
        generators,
        generic,
        generics,
        guards,
        guards_ensure,
        guards_type,
        guards_type_core,
        guards_type_model,
        guards_type_protocol,
        h,
        handler,
        handlers,
        infrastructure,
        lazy,
        logging,
        loggings,
        m,
        mapper,
        mixins,
        model,
        models,
        p,
        parser,
        platform,
        protocols,
        r,
        registry,
        reliability,
        result,
        result_helpers,
        runtime,
        s,
        service,
        services,
        settings,
        t,
        text,
        typings,
        u,
        utilities,
        validation,
        x,
    )
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "flext_core._constants",
        "flext_core._models",
        "flext_core._protocols",
        "flext_core._typings",
        "flext_core._utilities",
    ),
    {
        "BaseModel": "flext_core.typings",
        "FlextConstants": "flext_core.constants",
        "FlextContainer": "flext_core.container",
        "FlextContext": "flext_core.context",
        "FlextDecorators": "flext_core.decorators",
        "FlextDispatcher": "flext_core.dispatcher",
        "FlextExceptions": "flext_core.exceptions",
        "FlextHandlers": "flext_core.handlers",
        "FlextLogger": "flext_core.loggings",
        "FlextMixins": "flext_core.mixins",
        "FlextModels": "flext_core.models",
        "FlextProtocols": "flext_core.protocols",
        "FlextRegistry": "flext_core.registry",
        "FlextResult": "flext_core.result",
        "FlextRuntime": "flext_core.runtime",
        "FlextService": "flext_core.service",
        "FlextSettings": "flext_core.settings",
        "FlextTypes": "flext_core.typings",
        "FlextUtilities": "flext_core.utilities",
        "FlextVersion": "flext_core.__version__",
        "__author__": "flext_core.__version__",
        "__author_email__": "flext_core.__version__",
        "__description__": "flext_core.__version__",
        "__license__": "flext_core.__version__",
        "__title__": "flext_core.__version__",
        "__url__": "flext_core.__version__",
        "__version__": "flext_core.__version__",
        "__version_info__": "flext_core.__version__",
        "_constants": "flext_core._constants",
        "_models": "flext_core._models",
        "_protocols": "flext_core._protocols",
        "_typings": "flext_core._typings",
        "_utilities": "flext_core._utilities",
        "c": ("flext_core.constants", "FlextConstants"),
        "constants": "flext_core.constants",
        "container": "flext_core.container",
        "context": "flext_core.context",
        "d": ("flext_core.decorators", "FlextDecorators"),
        "decorators": "flext_core.decorators",
        "dispatcher": "flext_core.dispatcher",
        "e": ("flext_core.exceptions", "FlextExceptions"),
        "exceptions": "flext_core.exceptions",
        "h": ("flext_core.handlers", "FlextHandlers"),
        "handlers": "flext_core.handlers",
        "lazy": "flext_core.lazy",
        "loggings": "flext_core.loggings",
        "m": ("flext_core.models", "FlextModels"),
        "mixins": "flext_core.mixins",
        "models": "flext_core.models",
        "p": ("flext_core.protocols", "FlextProtocols"),
        "protocols": "flext_core.protocols",
        "r": ("flext_core.result", "FlextResult"),
        "registry": "flext_core.registry",
        "result": "flext_core.result",
        "runtime": "flext_core.runtime",
        "s": ("flext_core.service", "FlextService"),
        "service": "flext_core.service",
        "settings": "flext_core.settings",
        "t": ("flext_core.typings", "FlextTypes"),
        "typings": "flext_core.typings",
        "u": ("flext_core.utilities", "FlextUtilities"),
        "utilities": "flext_core.utilities",
        "x": ("flext_core.mixins", "FlextMixins"),
    },
)

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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
