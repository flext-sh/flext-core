# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core package."""

from __future__ import annotations

import typing as _t

from flext_core.__version__ import *
from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

if _t.TYPE_CHECKING:
    from _constants.base import FlextConstantsBase
    from _constants.cqrs import FlextConstantsCqrs
    from _constants.enforcement import FlextConstantsEnforcement
    from _constants.infrastructure import FlextConstantsInfrastructure
    from _constants.mixins import FlextConstantsMixins
    from _constants.platform import FlextConstantsPlatform
    from _constants.validation import FlextConstantsValidation
    from _models._context._data import FlextModelsContextData
    from _models._context._export import FlextModelsContextExport
    from _models._context._metadata import FlextModelsContextMetadata
    from _models._context._proxy_var import FlextModelsContextProxyVar
    from _models._context._scope import FlextModelsContextScope
    from _models._context._tokens import FlextModelsContextTokens
    from _models.base import FlextModelsBase
    from _models.builder import FlextModelsBuilder
    from _models.collections import FlextModelsCollections
    from _models.container import FlextModelsContainer
    from _models.containers import FlextModelsContainers
    from _models.context import FlextModelsContext
    from _models.cqrs import FlextModelsCqrs
    from _models.decorators import FlextModelsDecorators
    from _models.dispatcher import FlextModelsDispatcher
    from _models.domain_event import FlextModelsDomainEvent
    from _models.entity import FlextModelsEntity
    from _models.errors import FlextModelsErrors
    from _models.exception_params import FlextModelsExceptionParams
    from _models.generic import FlextGenericModels
    from _models.handler import FlextModelsHandler
    from _models.namespace import FlextModelsNamespace
    from _models.registry import FlextModelsRegistry
    from _models.service import FlextModelsService
    from _models.settings import FlextModelsSettings
    from _protocols.base import FlextProtocolsBase
    from _protocols.config import FlextProtocolsSettings
    from _protocols.container import FlextProtocolsContainer
    from _protocols.context import FlextProtocolsContext
    from _protocols.handler import FlextProtocolsHandler
    from _protocols.logging import FlextProtocolsLogging
    from _protocols.registry import FlextProtocolsRegistry
    from _protocols.result import FlextProtocolsResult
    from _protocols.service import FlextProtocolsService
    from _typings.annotateds import FlextTypesAnnotateds
    from _typings.base import FlextTypingBase
    from _typings.containers import FlextTypingContainers
    from _typings.core import FlextTypesCore
    from _typings.services import FlextTypesServices
    from _typings.typeadapters import FlextTypesTypeAdapters
    from _typings.validation import FlextTypesValidation
    from _utilities.args import FlextUtilitiesArgs
    from _utilities.beartype_conf import FlextUtilitiesBeartypeConf
    from _utilities.beartype_engine import FlextUtilitiesBeartypeEngine
    from _utilities.cache import FlextUtilitiesCache
    from _utilities.checker import FlextUtilitiesChecker
    from _utilities.collection import FlextUtilitiesCollection
    from _utilities.configuration import FlextUtilitiesConfiguration
    from _utilities.context import FlextUtilitiesContext
    from _utilities.conversion import FlextUtilitiesConversion
    from _utilities.discovery import FlextUtilitiesDiscovery
    from _utilities.domain import FlextUtilitiesDomain
    from _utilities.enforcement import FlextUtilitiesEnforcement
    from _utilities.enum import FlextUtilitiesEnum
    from _utilities.file_ops import FlextUtilitiesFileOps
    from _utilities.generators import FlextUtilitiesGenerators
    from _utilities.guards import FlextUtilitiesGuards
    from _utilities.guards_ensure import FlextUtilitiesGuardsEnsure
    from _utilities.guards_type import FlextUtilitiesGuardsType
    from _utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
    from _utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
    from _utilities.guards_type_protocol import FlextUtilitiesGuardsTypeProtocol
    from _utilities.mapper import FlextUtilitiesMapper
    from _utilities.model import FlextUtilitiesModel
    from _utilities.parser import FlextUtilitiesParser
    from _utilities.reliability import FlextUtilitiesReliability
    from _utilities.result_helpers import FlextUtilitiesResultHelpers
    from _utilities.text import FlextUtilitiesText
    from pydantic.main import BaseModel

    from _constants.domain import FlextConstantsDomain
    from _constants.errors import FlextConstantsErrors
    from _constants.settings import FlextConstantsSettings
    from flext_core.constants import FlextConstants, c
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecorators, d
    from flext_core.dispatcher import FlextDispatcher
    from flext_core.exceptions import FlextExceptions, e
    from flext_core.handlers import FlextHandlers, h
    from flext_core.lazy import build_lazy_import_map
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
        t,
    )
    from flext_core.utilities import FlextUtilities, u
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._constants",
        "._models",
        "._protocols",
        "._typings",
        "._utilities",
    ),
    build_lazy_import_map(
        {
            ".__version__": (
                "FlextVersion",
                "__author__",
                "__author_email__",
                "__description__",
                "__license__",
                "__title__",
                "__url__",
                "__version__",
                "__version_info__",
            ),
            ".constants": (
                "FlextConstants",
                "c",
            ),
            ".container": ("FlextContainer",),
            ".context": ("FlextContext",),
            ".decorators": (
                "FlextDecorators",
                "d",
            ),
            ".dispatcher": ("FlextDispatcher",),
            ".exceptions": (
                "FlextExceptions",
                "e",
            ),
            ".handlers": (
                "FlextHandlers",
                "h",
            ),
            ".lazy": ("build_lazy_import_map",),
            ".loggings": ("FlextLogger",),
            ".mixins": (
                "FlextMixins",
                "x",
            ),
            ".models": (
                "FlextModels",
                "m",
            ),
            ".protocols": (
                "FlextProtocols",
                "p",
            ),
            ".registry": ("FlextRegistry",),
            ".result": (
                "FlextResult",
                "r",
            ),
            ".runtime": ("FlextRuntime",),
            ".service": (
                "FlextService",
                "s",
            ),
            ".settings": ("FlextSettings",),
            ".typings": (
                "EnumT",
                "FlextTypes",
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
                "t",
            ),
            ".utilities": (
                "FlextUtilities",
                "u",
            ),
            "_constants.base": ("FlextConstantsBase",),
            "_constants.cqrs": ("FlextConstantsCqrs",),
            "_constants.domain": ("FlextConstantsDomain",),
            "_constants.enforcement": ("FlextConstantsEnforcement",),
            "_constants.errors": ("FlextConstantsErrors",),
            "_constants.infrastructure": ("FlextConstantsInfrastructure",),
            "_constants.mixins": ("FlextConstantsMixins",),
            "_constants.platform": ("FlextConstantsPlatform",),
            "_constants.settings": ("FlextConstantsSettings",),
            "_constants.validation": ("FlextConstantsValidation",),
            "_models._context._data": ("FlextModelsContextData",),
            "_models._context._export": ("FlextModelsContextExport",),
            "_models._context._metadata": ("FlextModelsContextMetadata",),
            "_models._context._proxy_var": ("FlextModelsContextProxyVar",),
            "_models._context._scope": ("FlextModelsContextScope",),
            "_models._context._tokens": ("FlextModelsContextTokens",),
            "_models.base": ("FlextModelsBase",),
            "_models.builder": ("FlextModelsBuilder",),
            "_models.collections": ("FlextModelsCollections",),
            "_models.container": ("FlextModelsContainer",),
            "_models.containers": ("FlextModelsContainers",),
            "_models.context": ("FlextModelsContext",),
            "_models.cqrs": ("FlextModelsCqrs",),
            "_models.decorators": ("FlextModelsDecorators",),
            "_models.dispatcher": ("FlextModelsDispatcher",),
            "_models.domain_event": ("FlextModelsDomainEvent",),
            "_models.entity": ("FlextModelsEntity",),
            "_models.errors": ("FlextModelsErrors",),
            "_models.exception_params": ("FlextModelsExceptionParams",),
            "_models.generic": ("FlextGenericModels",),
            "_models.handler": ("FlextModelsHandler",),
            "_models.namespace": ("FlextModelsNamespace",),
            "_models.registry": ("FlextModelsRegistry",),
            "_models.service": ("FlextModelsService",),
            "_models.settings": ("FlextModelsSettings",),
            "_protocols.base": ("FlextProtocolsBase",),
            "_protocols.config": ("FlextProtocolsSettings",),
            "_protocols.container": ("FlextProtocolsContainer",),
            "_protocols.context": ("FlextProtocolsContext",),
            "_protocols.handler": ("FlextProtocolsHandler",),
            "_protocols.logging": ("FlextProtocolsLogging",),
            "_protocols.registry": ("FlextProtocolsRegistry",),
            "_protocols.result": ("FlextProtocolsResult",),
            "_protocols.service": ("FlextProtocolsService",),
            "_typings.annotateds": ("FlextTypesAnnotateds",),
            "_typings.base": ("FlextTypingBase",),
            "_typings.containers": ("FlextTypingContainers",),
            "_typings.core": ("FlextTypesCore",),
            "_typings.services": ("FlextTypesServices",),
            "_typings.typeadapters": ("FlextTypesTypeAdapters",),
            "_typings.validation": ("FlextTypesValidation",),
            "_utilities.args": ("FlextUtilitiesArgs",),
            "_utilities.beartype_conf": ("FlextUtilitiesBeartypeConf",),
            "_utilities.beartype_engine": ("FlextUtilitiesBeartypeEngine",),
            "_utilities.cache": ("FlextUtilitiesCache",),
            "_utilities.checker": ("FlextUtilitiesChecker",),
            "_utilities.collection": ("FlextUtilitiesCollection",),
            "_utilities.configuration": ("FlextUtilitiesConfiguration",),
            "_utilities.context": ("FlextUtilitiesContext",),
            "_utilities.conversion": ("FlextUtilitiesConversion",),
            "_utilities.discovery": ("FlextUtilitiesDiscovery",),
            "_utilities.domain": ("FlextUtilitiesDomain",),
            "_utilities.enforcement": ("FlextUtilitiesEnforcement",),
            "_utilities.enum": ("FlextUtilitiesEnum",),
            "_utilities.file_ops": ("FlextUtilitiesFileOps",),
            "_utilities.generators": ("FlextUtilitiesGenerators",),
            "_utilities.guards": ("FlextUtilitiesGuards",),
            "_utilities.guards_ensure": ("FlextUtilitiesGuardsEnsure",),
            "_utilities.guards_type": ("FlextUtilitiesGuardsType",),
            "_utilities.guards_type_core": ("FlextUtilitiesGuardsTypeCore",),
            "_utilities.guards_type_model": ("FlextUtilitiesGuardsTypeModel",),
            "_utilities.guards_type_protocol": ("FlextUtilitiesGuardsTypeProtocol",),
            "_utilities.mapper": ("FlextUtilitiesMapper",),
            "_utilities.model": ("FlextUtilitiesModel",),
            "_utilities.parser": ("FlextUtilitiesParser",),
            "_utilities.reliability": ("FlextUtilitiesReliability",),
            "_utilities.result_helpers": ("FlextUtilitiesResultHelpers",),
            "_utilities.text": ("FlextUtilitiesText",),
            "pydantic.main": ("BaseModel",),
        },
    ),
    exclude_names=(
        "cleanup_submodule_namespace",
        "install_lazy_exports",
        "lazy_getattr",
        "logger",
        "merge_lazy_imports",
        "output",
        "output_reporting",
    ),
    module_name=__name__,
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)

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
    "FlextModelsSettings",
    "FlextProtocols",
    "FlextProtocolsBase",
    "FlextProtocolsContainer",
    "FlextProtocolsContext",
    "FlextProtocolsHandler",
    "FlextProtocolsLogging",
    "FlextProtocolsRegistry",
    "FlextProtocolsResult",
    "FlextProtocolsService",
    "FlextProtocolsSettings",
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
    "build_lazy_import_map",
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
