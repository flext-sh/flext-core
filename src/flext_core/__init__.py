# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Flext core package."""

from __future__ import annotations

import typing as _t

from flext_core.__version__ import *
from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _t.TYPE_CHECKING:
    import flext_core._constants as _flext_core__constants

    _constants = _flext_core__constants
    import flext_core._models as _flext_core__models
    from flext_core._constants import (
        FlextConstantsBase,
        FlextConstantsCqrs,
        FlextConstantsDomain,
        FlextConstantsEnforcement,
        FlextConstantsErrors,
        FlextConstantsInfrastructure,
        FlextConstantsMixins,
        FlextConstantsPlatform,
        FlextConstantsSettings,
        FlextConstantsValidation,
        base,
        cqrs,
        domain,
        enforcement,
        errors,
        infrastructure,
        platform,
        validation,
    )

    _models = _flext_core__models
    import flext_core._protocols as _flext_core__protocols
    from flext_core._models import (
        FlextGenericModels,
        FlextModelsBase,
        FlextModelsBuilder,
        FlextModelsCollections,
        FlextModelsConfig,
        FlextModelsContainer,
        FlextModelsContainers,
        FlextModelsContext,
        FlextModelsCqrs,
        FlextModelsDecorators,
        FlextModelsDispatcher,
        FlextModelsDomainEvent,
        FlextModelsEntity,
        FlextModelsErrors,
        FlextModelsExceptionParams,
        FlextModelsHandler,
        FlextModelsNamespace,
        FlextModelsRegistry,
        FlextModelsService,
        builder,
        collections,
        containers,
        domain_event,
        entity,
        exception_params,
        generic,
        handler,
        namespace,
    )
    from flext_core._models._context import (
        FlextModelsContextData,
        FlextModelsContextExport,
        FlextModelsContextMetadata,
        FlextModelsContextProxyVar,
        FlextModelsContextScope,
        FlextModelsContextTokens,
    )

    _protocols = _flext_core__protocols
    import flext_core._typings as _flext_core__typings
    from flext_core._protocols import (
        FlextProtocolsBase,
        FlextProtocolsConfig,
        FlextProtocolsContainer,
        FlextProtocolsContext,
        FlextProtocolsHandler,
        FlextProtocolsLogging,
        FlextProtocolsRegistry,
        FlextProtocolsResult,
        FlextProtocolsService,
        config,
        logging,
    )

    _typings = _flext_core__typings
    import flext_core._utilities as _flext_core__utilities
    from flext_core._typings import (
        FlextTypesAnnotateds,
        FlextTypesCore,
        FlextTypesServices,
        FlextTypesTypeAdapters,
        FlextTypesValidation,
        FlextTypingBase,
        FlextTypingContainers,
        annotateds,
        core,
        services,
        typeadapters,
    )
    from flext_core.typings import (
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

    _utilities = _flext_core__utilities
    import flext_core.constants as _flext_core_constants
    from flext_core._utilities import (
        FlextUtilitiesArgs,
        FlextUtilitiesBeartypeConf,
        FlextUtilitiesBeartypeEngine,
        FlextUtilitiesCache,
        FlextUtilitiesChecker,
        FlextUtilitiesCollection,
        FlextUtilitiesConfiguration,
        FlextUtilitiesContext,
        FlextUtilitiesConversion,
        FlextUtilitiesDiscovery,
        FlextUtilitiesDomain,
        FlextUtilitiesEnforcement,
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
        ParseOptions,
        args,
        beartype_conf,
        beartype_engine,
        cache,
        checker,
        collection,
        configuration,
        conversion,
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
        parser,
        reliability,
        result_helpers,
        text,
    )

    constants = _flext_core_constants
    import flext_core.container as _flext_core_container
    from flext_core.constants import FlextConstants, FlextConstants as c

    container = _flext_core_container
    import flext_core.context as _flext_core_context
    from flext_core.container import FlextContainer

    context = _flext_core_context
    import flext_core.decorators as _flext_core_decorators
    from flext_core.context import FlextContext

    decorators = _flext_core_decorators
    import flext_core.dispatcher as _flext_core_dispatcher
    from flext_core.decorators import FlextDecorators, FlextDecorators as d

    dispatcher = _flext_core_dispatcher
    import flext_core.exceptions as _flext_core_exceptions
    from flext_core.dispatcher import FlextDispatcher

    exceptions = _flext_core_exceptions
    import flext_core.handlers as _flext_core_handlers
    from flext_core.exceptions import FlextExceptions, FlextExceptions as e

    handlers = _flext_core_handlers
    import flext_core.lazy as _flext_core_lazy
    from flext_core.handlers import FlextHandlers, FlextHandlers as h

    lazy = _flext_core_lazy
    import flext_core.loggings as _flext_core_loggings

    loggings = _flext_core_loggings
    import flext_core.mixins as _flext_core_mixins
    from flext_core.loggings import FlextLogger

    mixins = _flext_core_mixins
    import flext_core.models as _flext_core_models
    from flext_core.mixins import FlextMixins, FlextMixins as x

    models = _flext_core_models
    import flext_core.protocols as _flext_core_protocols
    from flext_core.models import FlextModels, FlextModels as m

    protocols = _flext_core_protocols
    import flext_core.registry as _flext_core_registry
    from flext_core.protocols import FlextProtocols, FlextProtocols as p

    registry = _flext_core_registry
    import flext_core.result as _flext_core_result
    from flext_core.registry import FlextRegistry

    result = _flext_core_result
    import flext_core.runtime as _flext_core_runtime
    from flext_core.result import FlextResult, FlextResult as r

    runtime = _flext_core_runtime
    import flext_core.service as _flext_core_service
    from flext_core.runtime import FlextRuntime

    service = _flext_core_service
    import flext_core.settings as _flext_core_settings
    from flext_core.service import FlextService, FlextService as s

    settings = _flext_core_settings
    import flext_core.typings as _flext_core_typings
    from flext_core.settings import FlextSettings

    typings = _flext_core_typings
    import flext_core.utilities as _flext_core_utilities
    from flext_core.typings import BaseModel, FlextTypes, FlextTypes as t

    utilities = _flext_core_utilities
    from flext_core.utilities import FlextUtilities, FlextUtilities as u
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "flext_core._constants",
        "flext_core._models",
        "flext_core._protocols",
        "flext_core._typings",
        "flext_core._utilities",
    ),
    {
        "BaseModel": ("flext_core.typings", "BaseModel"),
        "FlextConstants": ("flext_core.constants", "FlextConstants"),
        "FlextContainer": ("flext_core.container", "FlextContainer"),
        "FlextContext": ("flext_core.context", "FlextContext"),
        "FlextDecorators": ("flext_core.decorators", "FlextDecorators"),
        "FlextDispatcher": ("flext_core.dispatcher", "FlextDispatcher"),
        "FlextExceptions": ("flext_core.exceptions", "FlextExceptions"),
        "FlextHandlers": ("flext_core.handlers", "FlextHandlers"),
        "FlextLogger": ("flext_core.loggings", "FlextLogger"),
        "FlextMixins": ("flext_core.mixins", "FlextMixins"),
        "FlextModels": ("flext_core.models", "FlextModels"),
        "FlextProtocols": ("flext_core.protocols", "FlextProtocols"),
        "FlextRegistry": ("flext_core.registry", "FlextRegistry"),
        "FlextResult": ("flext_core.result", "FlextResult"),
        "FlextRuntime": ("flext_core.runtime", "FlextRuntime"),
        "FlextService": ("flext_core.service", "FlextService"),
        "FlextSettings": ("flext_core.settings", "FlextSettings"),
        "FlextTypes": ("flext_core.typings", "FlextTypes"),
        "FlextUtilities": ("flext_core.utilities", "FlextUtilities"),
        "FlextVersion": ("flext_core.__version__", "FlextVersion"),
        "__author__": ("flext_core.__version__", "__author__"),
        "__author_email__": ("flext_core.__version__", "__author_email__"),
        "__description__": ("flext_core.__version__", "__description__"),
        "__license__": ("flext_core.__version__", "__license__"),
        "__title__": ("flext_core.__version__", "__title__"),
        "__url__": ("flext_core.__version__", "__url__"),
        "__version__": ("flext_core.__version__", "__version__"),
        "__version_info__": ("flext_core.__version__", "__version_info__"),
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
_ = _LAZY_IMPORTS.pop("cleanup_submodule_namespace", None)
_ = _LAZY_IMPORTS.pop("install_lazy_exports", None)
_ = _LAZY_IMPORTS.pop("lazy_getattr", None)
_ = _LAZY_IMPORTS.pop("merge_lazy_imports", None)
_ = _LAZY_IMPORTS.pop("output", None)
_ = _LAZY_IMPORTS.pop("output_reporting", None)

__all__ = [
    "TV",
    "BaseModel",
    "EnumT",
    "FlextConstants",
    "FlextConstantsBase",
    "FlextConstantsCqrs",
    "FlextConstantsDomain",
    "FlextConstantsEnforcement",
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
    "FlextModels",
    "FlextModelsBase",
    "FlextModelsBuilder",
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
    "FlextModelsNamespace",
    "FlextModelsRegistry",
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
    "FlextTypesAnnotateds",
    "FlextTypesCore",
    "FlextTypesServices",
    "FlextTypesTypeAdapters",
    "FlextTypesValidation",
    "FlextTypingBase",
    "FlextTypingContainers",
    "FlextUtilities",
    "FlextUtilitiesArgs",
    "FlextUtilitiesBeartypeConf",
    "FlextUtilitiesBeartypeEngine",
    "FlextUtilitiesCache",
    "FlextUtilitiesChecker",
    "FlextUtilitiesCollection",
    "FlextUtilitiesConfiguration",
    "FlextUtilitiesContext",
    "FlextUtilitiesConversion",
    "FlextUtilitiesDiscovery",
    "FlextUtilitiesDomain",
    "FlextUtilitiesEnforcement",
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
    "ParseOptions",
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
    "annotateds",
    "args",
    "base",
    "beartype_conf",
    "beartype_engine",
    "builder",
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
    "enforcement",
    "entity",
    "enum",
    "errors",
    "exception_params",
    "exceptions",
    "file_ops",
    "generators",
    "generic",
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
    "namespace",
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
    "typeadapters",
    "typings",
    "u",
    "utilities",
    "validation",
    "x",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
