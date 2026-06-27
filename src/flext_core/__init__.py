# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core package."""

from __future__ import annotations

import typing as _t

from flext_core.__version__ import (
    __author__,
    __author_email__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
    __version_info__,
)
from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

if _t.TYPE_CHECKING:
    from flext_core._constants.base import FlextConstantsBase as FlextConstantsBase
    from flext_core._constants.cqrs import FlextConstantsCqrs as FlextConstantsCqrs
    from flext_core._constants.enforcement import (
        FlextConstantsEnforcement as FlextConstantsEnforcement,
        FlextMroViolation as FlextMroViolation,
    )
    from flext_core._constants.environment import (
        FlextConstantsEnvironment as FlextConstantsEnvironment,
    )
    from flext_core._constants.errors import (
        FlextConstantsErrors as FlextConstantsErrors,
    )
    from flext_core._constants.file import FlextConstantsFile as FlextConstantsFile
    from flext_core._constants.guards import (
        FlextConstantsGuards as FlextConstantsGuards,
    )
    from flext_core._constants.infrastructure import (
        FlextConstantsInfrastructure as FlextConstantsInfrastructure,
    )
    from flext_core._constants.logging import (
        FlextConstantsLogging as FlextConstantsLogging,
    )
    from flext_core._constants.mixins import (
        FlextConstantsMixins as FlextConstantsMixins,
    )
    from flext_core._constants.project_metadata import (
        FlextConstantsProjectMetadata as FlextConstantsProjectMetadata,
    )
    from flext_core._constants.pydantic import (
        FlextConstantsPydantic as FlextConstantsPydantic,
    )
    from flext_core._constants.regex import FlextConstantsRegex as FlextConstantsRegex
    from flext_core._constants.serialization import (
        FlextConstantsSerialization as FlextConstantsSerialization,
    )
    from flext_core._constants.settings import (
        FlextConstantsSettings as FlextConstantsSettings,
    )
    from flext_core._constants.status import (
        FlextConstantsStatus as FlextConstantsStatus,
    )
    from flext_core._constants.timeout import (
        FlextConstantsTimeout as FlextConstantsTimeout,
    )
    from flext_core._constants.validation import (
        FlextConstantsValidation as FlextConstantsValidation,
    )
    from flext_core._exceptions.base import FlextExceptionsBase as FlextExceptionsBase
    from flext_core._exceptions.factories import (
        FlextExceptionsFactories as FlextExceptionsFactories,
    )
    from flext_core._exceptions.helpers import (
        FlextExceptionsHelpers as FlextExceptionsHelpers,
    )
    from flext_core._exceptions.metrics import (
        FlextExceptionsMetrics as FlextExceptionsMetrics,
    )
    from flext_core._exceptions.template import (
        FlextExceptionsTemplate as FlextExceptionsTemplate,
    )
    from flext_core._exceptions.types import (
        FlextExceptionsTypes as FlextExceptionsTypes,
    )
    from flext_core._models.base import FlextModelsBase as FlextModelsBase
    from flext_core._models.builder import FlextModelsBuilder as FlextModelsBuilder
    from flext_core._models.collections import (
        FlextModelsCollections as FlextModelsCollections,
    )
    from flext_core._models.container import (
        FlextModelsContainer as FlextModelsContainer,
    )
    from flext_core._models.containers import (
        FlextModelsContainers as FlextModelsContainers,
        mc as mc,
    )
    from flext_core._models.context import FlextModelsContext as FlextModelsContext
    from flext_core._models.cqrs import FlextModelsCqrs as FlextModelsCqrs
    from flext_core._models.dispatcher import (
        FlextModelsDispatcher as FlextModelsDispatcher,
    )
    from flext_core._models.domain_event import (
        FlextModelsDomainEvent as FlextModelsDomainEvent,
    )
    from flext_core._models.enforcement import (
        FlextModelsEnforcement as FlextModelsEnforcement,
    )
    from flext_core._models.entity import FlextModelsEntity as FlextModelsEntity
    from flext_core._models.errors import FlextModelsErrors as FlextModelsErrors
    from flext_core._models.exception_params import (
        FlextModelsExceptionParams as FlextModelsExceptionParams,
    )
    from flext_core._models.handler import FlextModelsHandler as FlextModelsHandler
    from flext_core._models.namespace import (
        FlextModelsNamespace as FlextModelsNamespace,
    )
    from flext_core._models.project_metadata import (
        FlextModelsProjectMetadata as FlextModelsProjectMetadata,
    )
    from flext_core._models.pydantic import FlextModelsPydantic as FlextModelsPydantic
    from flext_core._models.registry import FlextModelsRegistry as FlextModelsRegistry
    from flext_core._models.service import FlextModelsService as FlextModelsService
    from flext_core._models.settings import FlextModelsSettings as FlextModelsSettings
    from flext_core._protocols.base import FlextProtocolsBase as FlextProtocolsBase
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
    from flext_core._protocols.pydantic import (
        FlextProtocolsPydantic as FlextProtocolsPydantic,
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
    from flext_core._protocols.settings import (
        FlextProtocolsSettings as FlextProtocolsSettings,
    )
    from flext_core._settings.base import FlextSettingsBase as FlextSettingsBase
    from flext_core._settings.context import (
        FlextSettingsContext as FlextSettingsContext,
    )
    from flext_core._settings.core import FlextSettingsCore as FlextSettingsCore
    from flext_core._settings.database import (
        FlextSettingsDatabase as FlextSettingsDatabase,
    )
    from flext_core._settings.di import FlextSettingsDI as FlextSettingsDI
    from flext_core._settings.dispatcher import (
        FlextSettingsDispatcher as FlextSettingsDispatcher,
    )
    from flext_core._settings.infrastructure import (
        FlextSettingsInfrastructure as FlextSettingsInfrastructure,
    )
    from flext_core._settings.registry import (
        FlextSettingsRegistry as FlextSettingsRegistry,
    )
    from flext_core._typings.annotateds import (
        FlextTypesAnnotateds as FlextTypesAnnotateds,
    )
    from flext_core._typings.base import FlextTypingBase as FlextTypingBase
    from flext_core._typings.containers import (
        FlextTypingContainers as FlextTypingContainers,
    )
    from flext_core._typings.core import FlextTypesCore as FlextTypesCore
    from flext_core._typings.project_metadata import (
        FlextTypingProjectMetadata as FlextTypingProjectMetadata,
    )
    from flext_core._typings.pydantic import FlextTypesPydantic as FlextTypesPydantic
    from flext_core._typings.services import FlextTypesServices as FlextTypesServices
    from flext_core._typings.typeadapters import (
        FlextTypesTypeAdapters as FlextTypesTypeAdapters,
    )
    from flext_core._utilities._beartype.attr_visitor import (
        FlextUtilitiesBeartypeAttrVisitor as FlextUtilitiesBeartypeAttrVisitor,
    )
    from flext_core._utilities._beartype.class_visitor import (
        FlextUtilitiesBeartypeClassVisitor as FlextUtilitiesBeartypeClassVisitor,
    )
    from flext_core._utilities._beartype.deprecated_visitor import (
        FlextUtilitiesBeartypeDeprecatedVisitor as FlextUtilitiesBeartypeDeprecatedVisitor,
    )
    from flext_core._utilities._beartype.field_visitor import (
        FlextUtilitiesBeartypeFieldVisitor as FlextUtilitiesBeartypeFieldVisitor,
    )
    from flext_core._utilities._beartype.helpers import (
        FlextUtilitiesBeartypeHelpers as FlextUtilitiesBeartypeHelpers,
    )
    from flext_core._utilities._beartype.import_visitor import (
        FlextUtilitiesBeartypeImportVisitor as FlextUtilitiesBeartypeImportVisitor,
    )
    from flext_core._utilities._beartype.method_visitor import (
        FlextUtilitiesBeartypeMethodVisitor as FlextUtilitiesBeartypeMethodVisitor,
    )
    from flext_core._utilities._beartype.module_visitor import (
        FlextUtilitiesBeartypeModuleVisitor as FlextUtilitiesBeartypeModuleVisitor,
    )
    from flext_core._utilities.args import FlextUtilitiesArgs as FlextUtilitiesArgs
    from flext_core._utilities.beartype_conf import (
        FlextUtilitiesBeartypeConf as FlextUtilitiesBeartypeConf,
    )
    from flext_core._utilities.beartype_engine import (
        FlextUtilitiesBeartypeEngine as FlextUtilitiesBeartypeEngine,
        ube as ube,
    )
    from flext_core._utilities.beartype_typingext_patch import (
        FlextUtilitiesBeartypeTypingExtPatch as FlextUtilitiesBeartypeTypingExtPatch,
    )
    from flext_core._utilities.checker import (
        FlextUtilitiesChecker as FlextUtilitiesChecker,
    )
    from flext_core._utilities.collection import (
        FlextUtilitiesCollection as FlextUtilitiesCollection,
    )
    from flext_core._utilities.collection_iter import (
        FlextUtilitiesCollectionIter as FlextUtilitiesCollectionIter,
    )
    from flext_core._utilities.collection_merge import (
        FlextUtilitiesCollectionMerge as FlextUtilitiesCollectionMerge,
    )
    from flext_core._utilities.context import (
        FlextUtilitiesContext as FlextUtilitiesContext,
    )
    from flext_core._utilities.context_crud import (
        FlextUtilitiesContextCrud as FlextUtilitiesContextCrud,
    )
    from flext_core._utilities.context_lifecycle import (
        FlextUtilitiesContextLifecycle as FlextUtilitiesContextLifecycle,
    )
    from flext_core._utilities.context_state import (
        FlextUtilitiesContextState as FlextUtilitiesContextState,
    )
    from flext_core._utilities.conversion import (
        FlextUtilitiesConversion as FlextUtilitiesConversion,
    )
    from flext_core._utilities.discovery import (
        FlextUtilitiesDiscovery as FlextUtilitiesDiscovery,
    )
    from flext_core._utilities.dispatcher_execute import (
        execute_dispatcher_handler as execute_dispatcher_handler,
    )
    from flext_core._utilities.domain import (
        FlextUtilitiesDomain as FlextUtilitiesDomain,
    )
    from flext_core._utilities.enforcement import (
        FlextUtilitiesEnforcement as FlextUtilitiesEnforcement,
    )
    from flext_core._utilities.enforcement_collect import (
        FlextUtilitiesEnforcementCollect as FlextUtilitiesEnforcementCollect,
    )
    from flext_core._utilities.enforcement_emit import (
        FlextUtilitiesEnforcementEmit as FlextUtilitiesEnforcementEmit,
    )
    from flext_core._utilities.enum import FlextUtilitiesEnum as FlextUtilitiesEnum
    from flext_core._utilities.generators import (
        FlextUtilitiesGenerators as FlextUtilitiesGenerators,
    )
    from flext_core._utilities.guards import (
        FlextUtilitiesGuards as FlextUtilitiesGuards,
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
    from flext_core._utilities.handler import (
        FlextUtilitiesHandler as FlextUtilitiesHandler,
    )
    from flext_core._utilities.logging_config import (
        FlextUtilitiesLoggingConfig as FlextUtilitiesLoggingConfig,
    )
    from flext_core._utilities.logging_context import (
        FlextUtilitiesLoggingContext as FlextUtilitiesLoggingContext,
    )
    from flext_core._utilities.mapper import (
        FlextUtilitiesMapper as FlextUtilitiesMapper,
    )
    from flext_core._utilities.mapper_access import (
        FlextUtilitiesMapperAccess as FlextUtilitiesMapperAccess,
    )
    from flext_core._utilities.mapper_extract import (
        FlextUtilitiesMapperExtract as FlextUtilitiesMapperExtract,
    )
    from flext_core._utilities.model import FlextUtilitiesModel as FlextUtilitiesModel
    from flext_core._utilities.model_options import (
        FlextUtilitiesModelOptions as FlextUtilitiesModelOptions,
    )
    from flext_core._utilities.model_runtime import (
        FlextUtilitiesModelRuntime as FlextUtilitiesModelRuntime,
    )
    from flext_core._utilities.parser import (
        FlextUtilitiesParser as FlextUtilitiesParser,
    )
    from flext_core._utilities.parser_coerce import (
        FlextUtilitiesParserCoerce as FlextUtilitiesParserCoerce,
    )
    from flext_core._utilities.parser_targets import (
        FlextUtilitiesParserTargets as FlextUtilitiesParserTargets,
    )
    from flext_core._utilities.project_metadata import (
        FlextUtilitiesProjectMetadata as FlextUtilitiesProjectMetadata,
    )
    from flext_core._utilities.pydantic import (
        FlextUtilitiesPydantic as FlextUtilitiesPydantic,
    )
    from flext_core._utilities.reliability import (
        FlextUtilitiesReliability as FlextUtilitiesReliability,
    )
    from flext_core._utilities.runtime_violation_registry import (
        FlextUtilitiesRuntimeViolationRegistry as FlextUtilitiesRuntimeViolationRegistry,
    )
    from flext_core._utilities.settings import (
        FlextUtilitiesSettings as FlextUtilitiesSettings,
    )
    from flext_core._utilities.text import FlextUtilitiesText as FlextUtilitiesText
    from flext_core.constants import FlextConstants as FlextConstants, c as c
    from flext_core.container import FlextContainer as FlextContainer
    from flext_core.context import FlextContext as FlextContext
    from flext_core.decorators import FlextDecorators as FlextDecorators, d as d
    from flext_core.dispatcher import FlextDispatcher as FlextDispatcher
    from flext_core.exceptions import FlextExceptions as FlextExceptions, e as e
    from flext_core.handlers import FlextHandlers as FlextHandlers, h as h
    from flext_core.lazy import (
        FlextLazy as FlextLazy,
        lazy as lazy,
        normalize_lazy_imports as normalize_lazy_imports,
    )
    from flext_core.loggings import FlextLogger as FlextLogger
    from flext_core.mixins import FlextMixins as FlextMixins, x as x
    from flext_core.models import FlextModels as FlextModels, m as m
    from flext_core.protocols import FlextProtocols as FlextProtocols, p as p
    from flext_core.registry import FlextRegistry as FlextRegistry
    from flext_core.result import FlextResult as FlextResult, r as r
    from flext_core.runtime import FlextRuntime as FlextRuntime
    from flext_core.service import FlextService as FlextService, s as s
    from flext_core.settings import FlextSettings as FlextSettings
    from flext_core.typings import FlextTypes as FlextTypes, t as t
    from flext_core.utilities import FlextUtilities as FlextUtilities, u as u
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._constants",
        "._exceptions",
        "._models",
        "._protocols",
        "._settings",
        "._typings",
        "._utilities",
    ),
    build_lazy_import_map(
        {
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
            ".lazy": ("FlextLazy",),
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
                "FlextTypes",
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
        "pytest_addoption",
        "pytest_collect_file",
        "pytest_collection_modifyitems",
        "pytest_configure",
        "pytest_runtest_setup",
        "pytest_runtest_teardown",
        "pytest_sessionfinish",
        "pytest_sessionstart",
        "pytest_terminal_summary",
        "pytest_warning_recorded",
    ),
    module_name=__name__,
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    [
        "__author__",
        "__author_email__",
        "__description__",
        "__license__",
        "__title__",
        "__url__",
        "__version__",
        "__version_info__",
    ],
)

__all__: list[str] = [
    "FlextConstants",
    "FlextContainer",
    "FlextContext",
    "FlextDecorators",
    "FlextDispatcher",
    "FlextExceptions",
    "FlextHandlers",
    "FlextLazy",
    "FlextLogger",
    "FlextMixins",
    "FlextModels",
    "FlextProtocols",
    "FlextRegistry",
    "FlextResult",
    "FlextRuntime",
    "FlextService",
    "FlextSettings",
    "FlextTypes",
    "FlextUtilities",
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
