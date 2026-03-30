# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Flext core package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from flext_core.lazy import install_lazy_exports

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
    from flext_core._constants.base import *
    from flext_core._constants.cqrs import *
    from flext_core._constants.domain import *
    from flext_core._constants.errors import *
    from flext_core._constants.infrastructure import *
    from flext_core._constants.mixins import *
    from flext_core._constants.platform import *
    from flext_core._constants.settings import *
    from flext_core._constants.validation import *
    from flext_core._models import (
        collections,
        containers,
        domain_event,
        entity,
        exception_params,
        generic,
        handler,
    )
    from flext_core._models._context._data import *
    from flext_core._models._context._export import *
    from flext_core._models._context._metadata import *
    from flext_core._models._context._proxy_var import *
    from flext_core._models._context._scope import *
    from flext_core._models._context._tokens import *
    from flext_core._models.base import *
    from flext_core._models.collections import *
    from flext_core._models.container import *
    from flext_core._models.containers import *
    from flext_core._models.context import *
    from flext_core._models.cqrs import *
    from flext_core._models.decorators import *
    from flext_core._models.dispatcher import *
    from flext_core._models.domain_event import *
    from flext_core._models.entity import *
    from flext_core._models.errors import *
    from flext_core._models.exception_params import *
    from flext_core._models.generic import *
    from flext_core._models.handler import *
    from flext_core._models.service import *
    from flext_core._models.settings import *
    from flext_core._protocols import config, logging
    from flext_core._protocols.base import *
    from flext_core._protocols.config import *
    from flext_core._protocols.container import *
    from flext_core._protocols.context import *
    from flext_core._protocols.handler import *
    from flext_core._protocols.logging import *
    from flext_core._protocols.registry import *
    from flext_core._protocols.result import *
    from flext_core._protocols.service import *
    from flext_core._typings import core, generics, services
    from flext_core._typings.base import *
    from flext_core._typings.containers import *
    from flext_core._typings.core import *
    from flext_core._typings.generics import *
    from flext_core._typings.services import *
    from flext_core._typings.validation import *
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
    from flext_core._utilities.args import *
    from flext_core._utilities.cache import *
    from flext_core._utilities.checker import *
    from flext_core._utilities.collection import *
    from flext_core._utilities.configuration import *
    from flext_core._utilities.context import *
    from flext_core._utilities.conversion import *
    from flext_core._utilities.discovery import *
    from flext_core._utilities.domain import *
    from flext_core._utilities.enum import *
    from flext_core._utilities.file_ops import *
    from flext_core._utilities.generators import *
    from flext_core._utilities.guards import *
    from flext_core._utilities.guards_ensure import *
    from flext_core._utilities.guards_type import *
    from flext_core._utilities.guards_type_core import *
    from flext_core._utilities.guards_type_model import *
    from flext_core._utilities.guards_type_protocol import *
    from flext_core._utilities.mapper import *
    from flext_core._utilities.model import *
    from flext_core._utilities.pagination import *
    from flext_core._utilities.parser import *
    from flext_core._utilities.reliability import *
    from flext_core._utilities.result_helpers import *
    from flext_core._utilities.text import *
    from flext_core.constants import *
    from flext_core.container import *
    from flext_core.context import *
    from flext_core.decorators import *
    from flext_core.dispatcher import *
    from flext_core.errors import *
    from flext_core.exceptions import *
    from flext_core.handlers import *
    from flext_core.loggings import *
    from flext_core.mixins import *
    from flext_core.models import *
    from flext_core.protocols import *
    from flext_core.registry import *
    from flext_core.result import *
    from flext_core.runtime import *
    from flext_core.service import *
    from flext_core.settings import *
    from flext_core.typings import *
    from flext_core.utilities import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    "BaseModel": "flext_core.typings",
    "EnumT": "flext_core._typings.generics",
    "FlextConstants": "flext_core.constants",
    "FlextConstantsBase": "flext_core._constants.base",
    "FlextConstantsCqrs": "flext_core._constants.cqrs",
    "FlextConstantsDomain": "flext_core._constants.domain",
    "FlextConstantsErrors": "flext_core._constants.errors",
    "FlextConstantsInfrastructure": "flext_core._constants.infrastructure",
    "FlextConstantsMixins": "flext_core._constants.mixins",
    "FlextConstantsPlatform": "flext_core._constants.platform",
    "FlextConstantsSettings": "flext_core._constants.settings",
    "FlextConstantsValidation": "flext_core._constants.validation",
    "FlextContainer": "flext_core.container",
    "FlextContext": "flext_core.context",
    "FlextDecorators": "flext_core.decorators",
    "FlextDispatcher": "flext_core.dispatcher",
    "FlextError": "flext_core.errors",
    "FlextErrorDomain": "flext_core.errors",
    "FlextExceptions": "flext_core.exceptions",
    "FlextGenericModels": "flext_core._models.generic",
    "FlextHandlers": "flext_core.handlers",
    "FlextLogger": "flext_core.loggings",
    "FlextMixins": "flext_core.mixins",
    "FlextModelFoundation": "flext_core._models.base",
    "FlextModels": "flext_core.models",
    "FlextModelsCollections": "flext_core._models.collections",
    "FlextModelsConfig": "flext_core._models.settings",
    "FlextModelsContainer": "flext_core._models.container",
    "FlextModelsContainers": "flext_core._models.containers",
    "FlextModelsContext": "flext_core._models.context",
    "FlextModelsContextData": "flext_core._models._context._data",
    "FlextModelsContextExport": "flext_core._models._context._export",
    "FlextModelsContextMetadata": "flext_core._models._context._metadata",
    "FlextModelsContextProxyVar": "flext_core._models._context._proxy_var",
    "FlextModelsContextScope": "flext_core._models._context._scope",
    "FlextModelsContextTokens": "flext_core._models._context._tokens",
    "FlextModelsCqrs": "flext_core._models.cqrs",
    "FlextModelsDecorators": "flext_core._models.decorators",
    "FlextModelsDispatcher": "flext_core._models.dispatcher",
    "FlextModelsDomainEvent": "flext_core._models.domain_event",
    "FlextModelsEntity": "flext_core._models.entity",
    "FlextModelsErrors": "flext_core._models.errors",
    "FlextModelsExceptionParams": "flext_core._models.exception_params",
    "FlextModelsHandler": "flext_core._models.handler",
    "FlextModelsService": "flext_core._models.service",
    "FlextProtocols": "flext_core.protocols",
    "FlextProtocolsBase": "flext_core._protocols.base",
    "FlextProtocolsConfig": "flext_core._protocols.config",
    "FlextProtocolsContainer": "flext_core._protocols.container",
    "FlextProtocolsContext": "flext_core._protocols.context",
    "FlextProtocolsHandler": "flext_core._protocols.handler",
    "FlextProtocolsLogging": "flext_core._protocols.logging",
    "FlextProtocolsRegistry": "flext_core._protocols.registry",
    "FlextProtocolsResult": "flext_core._protocols.result",
    "FlextProtocolsService": "flext_core._protocols.service",
    "FlextRegistry": "flext_core.registry",
    "FlextResult": "flext_core.result",
    "FlextRuntime": "flext_core.runtime",
    "FlextService": "flext_core.service",
    "FlextSettings": "flext_core.settings",
    "FlextTypes": "flext_core.typings",
    "FlextTypesCore": "flext_core._typings.core",
    "FlextTypesServices": "flext_core._typings.services",
    "FlextTypesValidation": "flext_core._typings.validation",
    "FlextTypingBase": "flext_core._typings.base",
    "FlextTypingContainers": "flext_core._typings.containers",
    "FlextUtilities": "flext_core.utilities",
    "FlextUtilitiesArgs": "flext_core._utilities.args",
    "FlextUtilitiesCache": "flext_core._utilities.cache",
    "FlextUtilitiesChecker": "flext_core._utilities.checker",
    "FlextUtilitiesCollection": "flext_core._utilities.collection",
    "FlextUtilitiesConfiguration": "flext_core._utilities.configuration",
    "FlextUtilitiesContext": "flext_core._utilities.context",
    "FlextUtilitiesConversion": "flext_core._utilities.conversion",
    "FlextUtilitiesDiscovery": "flext_core._utilities.discovery",
    "FlextUtilitiesDomain": "flext_core._utilities.domain",
    "FlextUtilitiesEnum": "flext_core._utilities.enum",
    "FlextUtilitiesFileOps": "flext_core._utilities.file_ops",
    "FlextUtilitiesGenerators": "flext_core._utilities.generators",
    "FlextUtilitiesGuards": "flext_core._utilities.guards",
    "FlextUtilitiesGuardsEnsure": "flext_core._utilities.guards_ensure",
    "FlextUtilitiesGuardsType": "flext_core._utilities.guards_type",
    "FlextUtilitiesGuardsTypeCore": "flext_core._utilities.guards_type_core",
    "FlextUtilitiesGuardsTypeModel": "flext_core._utilities.guards_type_model",
    "FlextUtilitiesGuardsTypeProtocol": "flext_core._utilities.guards_type_protocol",
    "FlextUtilitiesMapper": "flext_core._utilities.mapper",
    "FlextUtilitiesModel": "flext_core._utilities.model",
    "FlextUtilitiesPagination": "flext_core._utilities.pagination",
    "FlextUtilitiesParser": "flext_core._utilities.parser",
    "FlextUtilitiesReliability": "flext_core._utilities.reliability",
    "FlextUtilitiesResultHelpers": "flext_core._utilities.result_helpers",
    "FlextUtilitiesText": "flext_core._utilities.text",
    "MessageT_contra": "flext_core._typings.generics",
    "P": "flext_core._typings.generics",
    "R": "flext_core._typings.generics",
    "ResultT": "flext_core._typings.generics",
    "T": "flext_core._typings.generics",
    "TRuntime": "flext_core._typings.generics",
    "TV": "flext_core._typings.generics",
    "TV_co": "flext_core._typings.generics",
    "T_Model": "flext_core._typings.generics",
    "T_Namespace": "flext_core._typings.generics",
    "T_Settings": "flext_core._typings.generics",
    "T_co": "flext_core._typings.generics",
    "T_contra": "flext_core._typings.generics",
    "U": "flext_core._typings.generics",
    "_constants": "flext_core._constants",
    "_models": "flext_core._models",
    "_protocols": "flext_core._protocols",
    "_typings": "flext_core._typings",
    "_utilities": "flext_core._utilities",
    "args": "flext_core._utilities.args",
    "base": "flext_core._constants.base",
    "c": ["flext_core.constants", "FlextConstants"],
    "cache": "flext_core._utilities.cache",
    "checker": "flext_core._utilities.checker",
    "collection": "flext_core._utilities.collection",
    "collections": "flext_core._models.collections",
    "config": "flext_core._protocols.config",
    "configuration": "flext_core._utilities.configuration",
    "constants": "flext_core.constants",
    "container": "flext_core.container",
    "containers": "flext_core._models.containers",
    "context": "flext_core.context",
    "conversion": "flext_core._utilities.conversion",
    "core": "flext_core._typings.core",
    "cqrs": "flext_core._constants.cqrs",
    "d": ["flext_core.decorators", "FlextDecorators"],
    "decorators": "flext_core.decorators",
    "deprecation": "flext_core._utilities.deprecation",
    "discovery": "flext_core._utilities.discovery",
    "dispatcher": "flext_core.dispatcher",
    "domain": "flext_core._constants.domain",
    "domain_event": "flext_core._models.domain_event",
    "e": ["flext_core.exceptions", "FlextExceptions"],
    "entity": "flext_core._models.entity",
    "enum": "flext_core._utilities.enum",
    "errors": "flext_core.errors",
    "exception_params": "flext_core._models.exception_params",
    "exceptions": "flext_core.exceptions",
    "file_ops": "flext_core._utilities.file_ops",
    "generators": "flext_core._utilities.generators",
    "generic": "flext_core._models.generic",
    "generics": "flext_core._typings.generics",
    "guards": "flext_core._utilities.guards",
    "guards_ensure": "flext_core._utilities.guards_ensure",
    "guards_type": "flext_core._utilities.guards_type",
    "guards_type_core": "flext_core._utilities.guards_type_core",
    "guards_type_model": "flext_core._utilities.guards_type_model",
    "guards_type_protocol": "flext_core._utilities.guards_type_protocol",
    "h": ["flext_core.handlers", "FlextHandlers"],
    "handler": "flext_core._models.handler",
    "handlers": "flext_core.handlers",
    "infrastructure": "flext_core._constants.infrastructure",
    "lazy": "flext_core.lazy",
    "logging": "flext_core._protocols.logging",
    "loggings": "flext_core.loggings",
    "m": ["flext_core.models", "FlextModels"],
    "mapper": "flext_core._utilities.mapper",
    "mixins": "flext_core.mixins",
    "model": "flext_core._utilities.model",
    "models": "flext_core.models",
    "p": ["flext_core.protocols", "FlextProtocols"],
    "pagination": "flext_core._utilities.pagination",
    "parser": "flext_core._utilities.parser",
    "platform": "flext_core._constants.platform",
    "protocols": "flext_core.protocols",
    "r": ["flext_core.result", "FlextResult"],
    "registry": "flext_core.registry",
    "reliability": "flext_core._utilities.reliability",
    "result": "flext_core.result",
    "result_helpers": "flext_core._utilities.result_helpers",
    "runtime": "flext_core.runtime",
    "s": ["flext_core.service", "FlextService"],
    "service": "flext_core.service",
    "services": "flext_core._typings.services",
    "settings": "flext_core.settings",
    "t": ["flext_core.typings", "FlextTypes"],
    "text": "flext_core._utilities.text",
    "typings": "flext_core.typings",
    "u": ["flext_core.utilities", "FlextUtilities"],
    "utilities": "flext_core.utilities",
    "validation": "flext_core._constants.validation",
    "x": ["flext_core.mixins", "FlextMixins"],
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, sorted(_LAZY_IMPORTS))
