# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Flext core package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.__version__ import *
from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _TYPE_CHECKING:
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
    from flext_core._constants import (
        FlextConstantsBase,
        FlextConstantsCqrs,
        FlextConstantsDomain,
        FlextConstantsErrors,
        FlextConstantsInfrastructure,
        FlextConstantsMixins,
        FlextConstantsPlatform,
        FlextConstantsSettings,
        FlextConstantsValidation,
        base,
        cqrs,
        domain,
        errors,
        infrastructure,
        platform,
        validation,
    )
    from flext_core._models import (
        FlextGenericModels,
        FlextModelFoundation,
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
        FlextModelsService,
        collections,
        containers,
        domain_event,
        entity,
        exception_params,
        generic,
        handler,
    )
    from flext_core._models._context import (
        FlextModelsContextData,
        FlextModelsContextExport,
        FlextModelsContextMetadata,
        FlextModelsContextProxyVar,
        FlextModelsContextScope,
        FlextModelsContextTokens,
    )
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
    from flext_core._typings import (
        TV,
        EnumT,
        FlextTypesCore,
        FlextTypesServices,
        FlextTypesValidation,
        FlextTypingBase,
        FlextTypingContainers,
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
        core,
        generics,
        services,
    )
    from flext_core._utilities import (
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
        args,
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
    from flext_core.constants import FlextConstants, FlextConstants as c
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecorators, FlextDecorators as d
    from flext_core.dispatcher import FlextDispatcher
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

_LAZY_IMPORTS: FlextTypes.LazyImportIndex = merge_lazy_imports(
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
