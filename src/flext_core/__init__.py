# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Flext core package."""

from __future__ import annotations

import typing as _t

from flext_core.__version__ import *
from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _t.TYPE_CHECKING:
    from flext_core._constants.base import FlextConstantsBase
    from flext_core._constants.cqrs import FlextConstantsCqrs
    from flext_core._constants.domain import FlextConstantsDomain
    from flext_core._constants.enforcement import FlextConstantsEnforcement
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
    from flext_core._models.base import FlextModelsBase
    from flext_core._models.builder import FlextModelsBuilder
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
    from flext_core._models.namespace import FlextModelsNamespace
    from flext_core._models.registry import FlextModelsRegistry
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
    from flext_core._typings.annotateds import FlextTypesAnnotateds
    from flext_core._typings.base import FlextTypingBase
    from flext_core._typings.containers import FlextTypingContainers
    from flext_core._typings.core import FlextTypesCore
    from flext_core._typings.services import FlextTypesServices
    from flext_core._typings.typeadapters import FlextTypesTypeAdapters
    from flext_core._typings.validation import FlextTypesValidation
    from flext_core._utilities.args import FlextUtilitiesArgs
    from flext_core._utilities.beartype_conf import FlextUtilitiesBeartypeConf
    from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine
    from flext_core._utilities.cache import FlextUtilitiesCache
    from flext_core._utilities.checker import FlextUtilitiesChecker
    from flext_core._utilities.collection import FlextUtilitiesCollection
    from flext_core._utilities.configuration import FlextUtilitiesConfiguration
    from flext_core._utilities.context import FlextUtilitiesContext
    from flext_core._utilities.conversion import FlextUtilitiesConversion
    from flext_core._utilities.discovery import FlextUtilitiesDiscovery
    from flext_core._utilities.domain import FlextUtilitiesDomain
    from flext_core._utilities.enforcement import FlextUtilitiesEnforcement
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
        "EnumT": ("flext_core.typings", "EnumT"),
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
        "MessageT_contra": ("flext_core.typings", "MessageT_contra"),
        "P": ("flext_core.typings", "P"),
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
        "__author__": ("flext_core.__version__", "__author__"),
        "__author_email__": ("flext_core.__version__", "__author_email__"),
        "__description__": ("flext_core.__version__", "__description__"),
        "__license__": ("flext_core.__version__", "__license__"),
        "__title__": ("flext_core.__version__", "__title__"),
        "__url__": ("flext_core.__version__", "__url__"),
        "__version__": ("flext_core.__version__", "__version__"),
        "__version_info__": ("flext_core.__version__", "__version_info__"),
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
    },
)
_ = _LAZY_IMPORTS.pop("cleanup_submodule_namespace", None)
_ = _LAZY_IMPORTS.pop("install_lazy_exports", None)
_ = _LAZY_IMPORTS.pop("lazy_getattr", None)
_ = _LAZY_IMPORTS.pop("logger", None)
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


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
