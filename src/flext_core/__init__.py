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
    from flext_core._constants.base import FlextConstantsBase
    from flext_core._constants.cqrs import FlextConstantsCqrs
    from flext_core._constants.domain import FlextConstantsDomain
    from flext_core._constants.enforcement import (
        FlextConstantsEnforcement,
        FlextMroViolation,
    )
    from flext_core._constants.errors import FlextConstantsErrors
    from flext_core._constants.infrastructure import FlextConstantsInfrastructure
    from flext_core._constants.mixins import FlextConstantsMixins
    from flext_core._constants.output import FlextConstantsOutput
    from flext_core._constants.platform import FlextConstantsPlatform
    from flext_core._constants.project_metadata import FlextConstantsProjectMetadata
    from flext_core._constants.pydantic import FlextConstantsPydantic
    from flext_core._constants.settings import FlextConstantsSettings
    from flext_core._constants.validation import FlextConstantsValidation
    from flext_core._exceptions.base import FlextExceptionsBase
    from flext_core._exceptions.factories import FlextExceptionsFactories
    from flext_core._exceptions.helpers import FlextExceptionsHelpers
    from flext_core._exceptions.metrics import FlextExceptionsMetrics
    from flext_core._exceptions.template import FlextExceptionsTemplate
    from flext_core._exceptions.types import FlextExceptionsTypes
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
    from flext_core._models.dispatcher import FlextModelsDispatcher
    from flext_core._models.domain_event import FlextModelsDomainEvent
    from flext_core._models.enforcement import FlextModelsEnforcement
    from flext_core._models.entity import FlextModelsEntity
    from flext_core._models.errors import FlextModelsErrors
    from flext_core._models.exception_params import FlextModelsExceptionParams
    from flext_core._models.generic import FlextGenericModels
    from flext_core._models.handler import FlextModelsHandler
    from flext_core._models.namespace import FlextModelsNamespace
    from flext_core._models.project_metadata import FlextModelsProjectMetadata
    from flext_core._models.pydantic import FlextModelsPydantic
    from flext_core._models.registry import FlextModelsRegistry
    from flext_core._models.service import FlextModelsService
    from flext_core._models.settings import FlextModelsSettings
    from flext_core._protocols.base import FlextProtocolsBase
    from flext_core._protocols.container import FlextProtocolsContainer
    from flext_core._protocols.context import FlextProtocolsContext
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._protocols.logging import FlextProtocolsLogging
    from flext_core._protocols.project_metadata import FlextProtocolsProjectMetadata
    from flext_core._protocols.pydantic import FlextProtocolsPydantic
    from flext_core._protocols.registry import FlextProtocolsRegistry
    from flext_core._protocols.result import FlextProtocolsResult
    from flext_core._protocols.service import FlextProtocolsService
    from flext_core._protocols.settings import FlextProtocolsSettings
    from flext_core._typings.annotateds import FlextTypesAnnotateds
    from flext_core._typings.base import FlextTypingBase
    from flext_core._typings.containers import ContainerOrModel, FlextTypingContainers
    from flext_core._typings.core import FlextTypesCore
    from flext_core._typings.project_metadata import FlextTypingProjectMetadata
    from flext_core._typings.pydantic import FlextTypesPydantic
    from flext_core._typings.services import FlextTypesServices
    from flext_core._typings.typeadapters import FlextTypesTypeAdapters
    from flext_core._utilities.args import FlextUtilitiesArgs
    from flext_core._utilities.beartype_conf import FlextUtilitiesBeartypeConf
    from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine
    from flext_core._utilities.checker import FlextUtilitiesChecker
    from flext_core._utilities.collection import FlextUtilitiesCollection
    from flext_core._utilities.context import FlextUtilitiesContext
    from flext_core._utilities.context_crud import FlextUtilitiesContextCrud
    from flext_core._utilities.context_lifecycle import FlextUtilitiesContextLifecycle
    from flext_core._utilities.context_normalization import (
        FlextUtilitiesContextNormalization,
    )
    from flext_core._utilities.context_scope import FlextUtilitiesContextScope
    from flext_core._utilities.context_tracing import FlextUtilitiesContextTracing
    from flext_core._utilities.conversion import FlextUtilitiesConversion
    from flext_core._utilities.discovery import FlextUtilitiesDiscovery
    from flext_core._utilities.domain import FlextUtilitiesDomain
    from flext_core._utilities.enforcement import FlextUtilitiesEnforcement
    from flext_core._utilities.enum import FlextUtilitiesEnum
    from flext_core._utilities.generators import FlextUtilitiesGenerators
    from flext_core._utilities.guards import FlextUtilitiesGuards
    from flext_core._utilities.guards_ensure import FlextUtilitiesGuardsEnsure
    from flext_core._utilities.guards_type import FlextUtilitiesGuardsType
    from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
    from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
    from flext_core._utilities.guards_type_protocol import (
        FlextUtilitiesGuardsTypeProtocol,
    )
    from flext_core._utilities.logging_config import FlextUtilitiesLoggingConfig
    from flext_core._utilities.logging_context import FlextUtilitiesLoggingContext
    from flext_core._utilities.mapper import FlextUtilitiesMapper
    from flext_core._utilities.model import FlextUtilitiesModel
    from flext_core._utilities.parser import FlextUtilitiesParser
    from flext_core._utilities.project_metadata import FlextUtilitiesProjectMetadata
    from flext_core._utilities.pydantic import FlextUtilitiesPydantic
    from flext_core._utilities.reliability import FlextUtilitiesReliability
    from flext_core._utilities.settings import FlextUtilitiesSettings
    from flext_core._utilities.text import FlextUtilitiesText
    from flext_core.constants import FlextConstants, c
    from flext_core.container import FlextContainer
    from flext_core.context import FlextContext
    from flext_core.decorators import FlextDecorators, d
    from flext_core.dispatcher import FlextDispatcher
    from flext_core.exceptions import FlextExceptions, e
    from flext_core.handlers import FlextHandlers, h
    from flext_core.lazy import FlextLazy, build_lazy_import_map, lazy
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
        RootValueT,
        T,
        T_co,
        T_contra,
        T_DomainResult,
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
        "._exceptions",
        "._models",
        "._protocols",
        "._typings",
        "._utilities",
    ),
    build_lazy_import_map(
        {
            ".__version__": (
                "__author__",
                "__author_email__",
                "__description__",
                "__license__",
                "__title__",
                "__url__",
                "__version__",
                "__version_info__",
            ),
            "._constants.base": ("FlextConstantsBase",),
            "._constants.cqrs": ("FlextConstantsCqrs",),
            "._constants.domain": ("FlextConstantsDomain",),
            "._constants.enforcement": (
                "FlextConstantsEnforcement",
                "FlextMroViolation",
            ),
            "._constants.errors": ("FlextConstantsErrors",),
            "._constants.infrastructure": ("FlextConstantsInfrastructure",),
            "._constants.mixins": ("FlextConstantsMixins",),
            "._constants.output": ("FlextConstantsOutput",),
            "._constants.platform": ("FlextConstantsPlatform",),
            "._constants.project_metadata": ("FlextConstantsProjectMetadata",),
            "._constants.pydantic": ("FlextConstantsPydantic",),
            "._constants.settings": ("FlextConstantsSettings",),
            "._constants.validation": ("FlextConstantsValidation",),
            "._exceptions.base": ("FlextExceptionsBase",),
            "._exceptions.factories": ("FlextExceptionsFactories",),
            "._exceptions.helpers": ("FlextExceptionsHelpers",),
            "._exceptions.metrics": ("FlextExceptionsMetrics",),
            "._exceptions.template": ("FlextExceptionsTemplate",),
            "._exceptions.types": ("FlextExceptionsTypes",),
            "._models._context._data": ("FlextModelsContextData",),
            "._models._context._export": ("FlextModelsContextExport",),
            "._models._context._metadata": ("FlextModelsContextMetadata",),
            "._models._context._proxy_var": ("FlextModelsContextProxyVar",),
            "._models._context._scope": ("FlextModelsContextScope",),
            "._models._context._tokens": ("FlextModelsContextTokens",),
            "._models.base": ("FlextModelsBase",),
            "._models.builder": ("FlextModelsBuilder",),
            "._models.collections": ("FlextModelsCollections",),
            "._models.container": ("FlextModelsContainer",),
            "._models.containers": ("FlextModelsContainers",),
            "._models.context": ("FlextModelsContext",),
            "._models.cqrs": ("FlextModelsCqrs",),
            "._models.dispatcher": ("FlextModelsDispatcher",),
            "._models.domain_event": ("FlextModelsDomainEvent",),
            "._models.enforcement": ("FlextModelsEnforcement",),
            "._models.entity": ("FlextModelsEntity",),
            "._models.errors": ("FlextModelsErrors",),
            "._models.exception_params": ("FlextModelsExceptionParams",),
            "._models.generic": ("FlextGenericModels",),
            "._models.handler": ("FlextModelsHandler",),
            "._models.namespace": ("FlextModelsNamespace",),
            "._models.project_metadata": ("FlextModelsProjectMetadata",),
            "._models.pydantic": ("FlextModelsPydantic",),
            "._models.registry": ("FlextModelsRegistry",),
            "._models.service": ("FlextModelsService",),
            "._models.settings": ("FlextModelsSettings",),
            "._protocols.base": ("FlextProtocolsBase",),
            "._protocols.container": ("FlextProtocolsContainer",),
            "._protocols.context": ("FlextProtocolsContext",),
            "._protocols.handler": ("FlextProtocolsHandler",),
            "._protocols.logging": ("FlextProtocolsLogging",),
            "._protocols.project_metadata": ("FlextProtocolsProjectMetadata",),
            "._protocols.pydantic": ("FlextProtocolsPydantic",),
            "._protocols.registry": ("FlextProtocolsRegistry",),
            "._protocols.result": ("FlextProtocolsResult",),
            "._protocols.service": ("FlextProtocolsService",),
            "._protocols.settings": ("FlextProtocolsSettings",),
            "._typings.annotateds": ("FlextTypesAnnotateds",),
            "._typings.base": ("FlextTypingBase",),
            "._typings.containers": (
                "ContainerOrModel",
                "FlextTypingContainers",
            ),
            "._typings.core": ("FlextTypesCore",),
            "._typings.project_metadata": ("FlextTypingProjectMetadata",),
            "._typings.pydantic": ("FlextTypesPydantic",),
            "._typings.services": ("FlextTypesServices",),
            "._typings.typeadapters": ("FlextTypesTypeAdapters",),
            "._utilities.args": ("FlextUtilitiesArgs",),
            "._utilities.beartype_conf": ("FlextUtilitiesBeartypeConf",),
            "._utilities.beartype_engine": ("FlextUtilitiesBeartypeEngine",),
            "._utilities.checker": ("FlextUtilitiesChecker",),
            "._utilities.collection": ("FlextUtilitiesCollection",),
            "._utilities.context": ("FlextUtilitiesContext",),
            "._utilities.context_crud": ("FlextUtilitiesContextCrud",),
            "._utilities.context_lifecycle": ("FlextUtilitiesContextLifecycle",),
            "._utilities.context_normalization": (
                "FlextUtilitiesContextNormalization",
            ),
            "._utilities.context_scope": ("FlextUtilitiesContextScope",),
            "._utilities.context_tracing": ("FlextUtilitiesContextTracing",),
            "._utilities.conversion": ("FlextUtilitiesConversion",),
            "._utilities.discovery": ("FlextUtilitiesDiscovery",),
            "._utilities.domain": ("FlextUtilitiesDomain",),
            "._utilities.enforcement": ("FlextUtilitiesEnforcement",),
            "._utilities.enum": ("FlextUtilitiesEnum",),
            "._utilities.generators": ("FlextUtilitiesGenerators",),
            "._utilities.guards": ("FlextUtilitiesGuards",),
            "._utilities.guards_ensure": ("FlextUtilitiesGuardsEnsure",),
            "._utilities.guards_type": ("FlextUtilitiesGuardsType",),
            "._utilities.guards_type_core": ("FlextUtilitiesGuardsTypeCore",),
            "._utilities.guards_type_model": ("FlextUtilitiesGuardsTypeModel",),
            "._utilities.guards_type_protocol": ("FlextUtilitiesGuardsTypeProtocol",),
            "._utilities.logging_config": ("FlextUtilitiesLoggingConfig",),
            "._utilities.logging_context": ("FlextUtilitiesLoggingContext",),
            "._utilities.mapper": ("FlextUtilitiesMapper",),
            "._utilities.model": ("FlextUtilitiesModel",),
            "._utilities.parser": ("FlextUtilitiesParser",),
            "._utilities.project_metadata": ("FlextUtilitiesProjectMetadata",),
            "._utilities.pydantic": ("FlextUtilitiesPydantic",),
            "._utilities.reliability": ("FlextUtilitiesReliability",),
            "._utilities.settings": ("FlextUtilitiesSettings",),
            "._utilities.text": ("FlextUtilitiesText",),
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
            ".lazy": (
                "FlextLazy",
                "build_lazy_import_map",
                "lazy",
            ),
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
                "RootValueT",
                "T",
                "TRuntime",
                "TV",
                "TV_co",
                "T_DomainResult",
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

__all__: list[str] = [
    "TV",
    "ContainerOrModel",
    "EnumT",
    "FlextConstants",
    "FlextConstantsBase",
    "FlextConstantsCqrs",
    "FlextConstantsDomain",
    "FlextConstantsEnforcement",
    "FlextConstantsErrors",
    "FlextConstantsInfrastructure",
    "FlextConstantsMixins",
    "FlextConstantsOutput",
    "FlextConstantsPlatform",
    "FlextConstantsProjectMetadata",
    "FlextConstantsPydantic",
    "FlextConstantsSettings",
    "FlextConstantsValidation",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextExceptionsBase",
    "FlextExceptionsFactories",
    "FlextExceptionsHelpers",
    "FlextExceptionsMetrics",
    "FlextExceptionsTemplate",
    "FlextExceptionsTypes",
    "FlextGenericModels",
    "FlextHandlers",
    "FlextLazy",
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
    "FlextModelsDispatcher",
    "FlextModelsDomainEvent",
    "FlextModelsEnforcement",
    "FlextModelsEntity",
    "FlextModelsErrors",
    "FlextModelsExceptionParams",
    "FlextModelsHandler",
    "FlextModelsNamespace",
    "FlextModelsProjectMetadata",
    "FlextModelsPydantic",
    "FlextModelsRegistry",
    "FlextModelsService",
    "FlextModelsSettings",
    "FlextMroViolation",
    "FlextProtocols",
    "FlextProtocolsBase",
    "FlextProtocolsContainer",
    "FlextProtocolsContext",
    "FlextProtocolsHandler",
    "FlextProtocolsLogging",
    "FlextProtocolsProjectMetadata",
    "FlextProtocolsPydantic",
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
    "FlextTypesPydantic",
    "FlextTypesServices",
    "FlextTypesTypeAdapters",
    "FlextTypingBase",
    "FlextTypingContainers",
    "FlextTypingProjectMetadata",
    "FlextUtilities",
    "FlextUtilitiesArgs",
    "FlextUtilitiesBeartypeConf",
    "FlextUtilitiesBeartypeEngine",
    "FlextUtilitiesChecker",
    "FlextUtilitiesCollection",
    "FlextUtilitiesContext",
    "FlextUtilitiesContextCrud",
    "FlextUtilitiesContextLifecycle",
    "FlextUtilitiesContextNormalization",
    "FlextUtilitiesContextScope",
    "FlextUtilitiesContextTracing",
    "FlextUtilitiesConversion",
    "FlextUtilitiesDiscovery",
    "FlextUtilitiesDomain",
    "FlextUtilitiesEnforcement",
    "FlextUtilitiesEnum",
    "FlextUtilitiesGenerators",
    "FlextUtilitiesGuards",
    "FlextUtilitiesGuardsEnsure",
    "FlextUtilitiesGuardsType",
    "FlextUtilitiesGuardsTypeCore",
    "FlextUtilitiesGuardsTypeModel",
    "FlextUtilitiesGuardsTypeProtocol",
    "FlextUtilitiesLoggingConfig",
    "FlextUtilitiesLoggingContext",
    "FlextUtilitiesMapper",
    "FlextUtilitiesModel",
    "FlextUtilitiesParser",
    "FlextUtilitiesProjectMetadata",
    "FlextUtilitiesPydantic",
    "FlextUtilitiesReliability",
    "FlextUtilitiesSettings",
    "FlextUtilitiesText",
    "MessageT_contra",
    "P",
    "R",
    "ResultT",
    "RootValueT",
    "T",
    "TRuntime",
    "TV_co",
    "T_DomainResult",
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
    "lazy",
    "m",
    "p",
    "r",
    "s",
    "t",
    "u",
    "x",
]
