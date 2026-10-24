# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities import (
        _beartype as _beartype,
        _checker as _checker,
        _enforcement as _enforcement,
        _enforcement_collect as _enforcement_collect,
        _logging_config as _logging_config,
        _logging_context as _logging_context,
        _mapper_access as _mapper_access,
        _mapper_extract as _mapper_extract,
        _parser_targets as _parser_targets,
    )
    from flext_core._utilities._beartype import (
        FlextUtilitiesBeartypeAliasVisitor,
        FlextUtilitiesBeartypeAttrVisitor,
        FlextUtilitiesBeartypeClassVisitor,
        FlextUtilitiesBeartypeDeprecatedVisitor,
        FlextUtilitiesBeartypeFieldUnionScan,
        FlextUtilitiesBeartypeFieldVisitor,
        FlextUtilitiesBeartypeHelpers,
        FlextUtilitiesBeartypeImportBlacklistVisitor,
        FlextUtilitiesBeartypeImportVisitor,
        FlextUtilitiesBeartypeLibraryVisitor,
        FlextUtilitiesBeartypeMethodVisitor,
        FlextUtilitiesBeartypeModuleSource,
        FlextUtilitiesBeartypeModuleSourceDeferred,
        FlextUtilitiesBeartypeModuleSourceProvenance,
        FlextUtilitiesBeartypeModuleSourceScan,
        FlextUtilitiesBeartypeModuleVisitor,
        FlextUtilitiesBeartypeTypeAliases,
        alias_first_violation,
    )
    from flext_core._utilities._guards_type_protocol_specs import (
        FlextUtilitiesGuardsTypeProtocolSpecsMixin,
    )
    from flext_core._utilities._guards_type_protocol_string import (
        FlextUtilitiesGuardsTypeProtocolStringMixin,
    )
    from flext_core._utilities._logging_config import FlextUtilitiesLoggingStream
    from flext_core._utilities.args import FlextUtilitiesArgs
    from flext_core._utilities.base import FlextUtilitiesBase
    from flext_core._utilities.beartype_conf import FlextUtilitiesBeartypeConf
    from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine
    from flext_core._utilities.checker import FlextUtilitiesChecker
    from flext_core._utilities.collection import FlextUtilitiesCollection
    from flext_core._utilities.collection_iter import FlextUtilitiesCollectionIter
    from flext_core._utilities.collection_merge import FlextUtilitiesCollectionMerge
    from flext_core._utilities.config import FlextUtilitiesConfig
    from flext_core._utilities.console import FlextUtilitiesConsole
    from flext_core._utilities.context import FlextUtilitiesContext
    from flext_core._utilities.conversion import FlextUtilitiesConversion
    from flext_core._utilities.discovery import FlextUtilitiesDiscovery
    from flext_core._utilities.dispatcher_execute import FlextUtilitiesDispatcherExecute
    from flext_core._utilities.domain import FlextUtilitiesDomain
    from flext_core._utilities.enforcement import (
        PREDICATE_BINDINGS,
        FlextUtilitiesEnforcement,
    )
    from flext_core._utilities.enforcement_collect import (
        FlextUtilitiesEnforcementCollect,
    )
    from flext_core._utilities.enforcement_emit import FlextUtilitiesEnforcementEmit
    from flext_core._utilities.enums import FlextUtilitiesEnum
    from flext_core._utilities.family_surface import FlextUtilitiesFamilySurface
    from flext_core._utilities.files import FlextUtilitiesFiles
    from flext_core._utilities.generators import FlextUtilitiesGenerators
    from flext_core._utilities.guards import FlextUtilitiesGuards
    from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
    from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
    from flext_core._utilities.guards_type_protocol import (
        FlextUtilitiesGuardsTypeProtocol,
    )
    from flext_core._utilities.handler import FlextUtilitiesHandler
    from flext_core._utilities.logging import FlextUtilitiesLogging
    from flext_core._utilities.logging_config import FlextUtilitiesLoggingConfig
    from flext_core._utilities.logging_context import FlextUtilitiesLoggingContext
    from flext_core._utilities.mapper import FlextUtilitiesMapper
    from flext_core._utilities.mapper_access import FlextUtilitiesMapperAccess
    from flext_core._utilities.mapper_extract import FlextUtilitiesMapperExtract
    from flext_core._utilities.metadata import FlextUtilitiesMetadata
    from flext_core._utilities.model import FlextUtilitiesModel
    from flext_core._utilities.model_options import FlextUtilitiesModelOptions
    from flext_core._utilities.model_runtime import FlextUtilitiesModelRuntime
    from flext_core._utilities.parser import FlextUtilitiesParser
    from flext_core._utilities.parser_coerce import FlextUtilitiesParserCoerce
    from flext_core._utilities.parser_targets import FlextUtilitiesParserTargets
    from flext_core._utilities.project_metadata import FlextUtilitiesProjectMetadata
    from flext_core._utilities.pydantic import FlextUtilitiesPydantic
    from flext_core._utilities.reliability import FlextUtilitiesReliability
    from flext_core._utilities.runtime_violation_registry import (
        FlextUtilitiesRuntimeViolationRegistry,
    )
    from flext_core._utilities.settings import FlextUtilitiesSettings
    from flext_core._utilities.text import FlextUtilitiesText
    from flext_core._utilities.typeadapter_json import FlextUtilitiesTypingAdapterJson
    from flext_core._utilities.typeadapter_scalars import (
        FlextUtilitiesTypingAdapterScalars,
    )


__all__: tuple[str, ...] = (
    "PREDICATE_BINDINGS",
    "FlextUtilitiesArgs",
    "FlextUtilitiesBase",
    "FlextUtilitiesBeartypeAliasVisitor",
    "FlextUtilitiesBeartypeAttrVisitor",
    "FlextUtilitiesBeartypeClassVisitor",
    "FlextUtilitiesBeartypeConf",
    "FlextUtilitiesBeartypeDeprecatedVisitor",
    "FlextUtilitiesBeartypeEngine",
    "FlextUtilitiesBeartypeFieldUnionScan",
    "FlextUtilitiesBeartypeFieldVisitor",
    "FlextUtilitiesBeartypeHelpers",
    "FlextUtilitiesBeartypeImportBlacklistVisitor",
    "FlextUtilitiesBeartypeImportVisitor",
    "FlextUtilitiesBeartypeLibraryVisitor",
    "FlextUtilitiesBeartypeMethodVisitor",
    "FlextUtilitiesBeartypeModuleSource",
    "FlextUtilitiesBeartypeModuleSourceDeferred",
    "FlextUtilitiesBeartypeModuleSourceProvenance",
    "FlextUtilitiesBeartypeModuleSourceScan",
    "FlextUtilitiesBeartypeModuleVisitor",
    "FlextUtilitiesBeartypeTypeAliases",
    "FlextUtilitiesChecker",
    "FlextUtilitiesCollection",
    "FlextUtilitiesCollectionIter",
    "FlextUtilitiesCollectionMerge",
    "FlextUtilitiesConfig",
    "FlextUtilitiesConsole",
    "FlextUtilitiesContext",
    "FlextUtilitiesConversion",
    "FlextUtilitiesDiscovery",
    "FlextUtilitiesDispatcherExecute",
    "FlextUtilitiesDomain",
    "FlextUtilitiesEnforcement",
    "FlextUtilitiesEnforcementCollect",
    "FlextUtilitiesEnforcementEmit",
    "FlextUtilitiesEnum",
    "FlextUtilitiesFamilySurface",
    "FlextUtilitiesFiles",
    "FlextUtilitiesGenerators",
    "FlextUtilitiesGuards",
    "FlextUtilitiesGuardsTypeCore",
    "FlextUtilitiesGuardsTypeModel",
    "FlextUtilitiesGuardsTypeProtocol",
    "FlextUtilitiesGuardsTypeProtocolSpecsMixin",
    "FlextUtilitiesGuardsTypeProtocolStringMixin",
    "FlextUtilitiesHandler",
    "FlextUtilitiesLogging",
    "FlextUtilitiesLoggingConfig",
    "FlextUtilitiesLoggingContext",
    "FlextUtilitiesLoggingStream",
    "FlextUtilitiesMapper",
    "FlextUtilitiesMapperAccess",
    "FlextUtilitiesMapperExtract",
    "FlextUtilitiesMetadata",
    "FlextUtilitiesModel",
    "FlextUtilitiesModelOptions",
    "FlextUtilitiesModelRuntime",
    "FlextUtilitiesParser",
    "FlextUtilitiesParserCoerce",
    "FlextUtilitiesParserTargets",
    "FlextUtilitiesProjectMetadata",
    "FlextUtilitiesPydantic",
    "FlextUtilitiesReliability",
    "FlextUtilitiesRuntimeViolationRegistry",
    "FlextUtilitiesSettings",
    "FlextUtilitiesText",
    "FlextUtilitiesTypingAdapterJson",
    "FlextUtilitiesTypingAdapterScalars",
    "_beartype",
    "_checker",
    "_enforcement",
    "_enforcement_collect",
    "_logging_config",
    "_logging_context",
    "_mapper_access",
    "_mapper_extract",
    "_parser_targets",
    "alias_first_violation",
)

__getattr__ = install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "PREDICATE_BINDINGS": ".enforcement",
        "FlextUtilitiesArgs": ".args",
        "FlextUtilitiesBase": ".base",
        "FlextUtilitiesBeartypeAliasVisitor": "._beartype._alias_visitor",
        "FlextUtilitiesBeartypeAttrVisitor": "._beartype.attr_visitor",
        "FlextUtilitiesBeartypeClassVisitor": "._beartype.class_visitor",
        "FlextUtilitiesBeartypeConf": ".beartype_conf",
        "FlextUtilitiesBeartypeDeprecatedVisitor": "._beartype.deprecated_visitor",
        "FlextUtilitiesBeartypeEngine": ".beartype_engine",
        "FlextUtilitiesBeartypeFieldUnionScan": "._beartype._field_union_scan",
        "FlextUtilitiesBeartypeFieldVisitor": "._beartype.field_visitor",
        "FlextUtilitiesBeartypeHelpers": "._beartype.helpers",
        "FlextUtilitiesBeartypeImportBlacklistVisitor": (
            "._beartype._import_blacklist_visitor"
        ),
        "FlextUtilitiesBeartypeImportVisitor": "._beartype.import_visitor",
        "FlextUtilitiesBeartypeLibraryVisitor": "._beartype._library_visitor",
        "FlextUtilitiesBeartypeMethodVisitor": "._beartype.method_visitor",
        "FlextUtilitiesBeartypeModuleSource": "._beartype.module_source",
        "FlextUtilitiesBeartypeModuleSourceDeferred": (
            "._beartype._module_source_deferred"
        ),
        "FlextUtilitiesBeartypeModuleSourceProvenance": (
            "._beartype._module_source_provenance"
        ),
        "FlextUtilitiesBeartypeModuleSourceScan": "._beartype._module_source_scan",
        "FlextUtilitiesBeartypeModuleVisitor": "._beartype.module_visitor",
        "FlextUtilitiesBeartypeTypeAliases": "._beartype.type_aliases",
        "FlextUtilitiesChecker": ".checker",
        "FlextUtilitiesCollection": ".collection",
        "FlextUtilitiesCollectionIter": ".collection_iter",
        "FlextUtilitiesCollectionMerge": ".collection_merge",
        "FlextUtilitiesConfig": ".config",
        "FlextUtilitiesConsole": ".console",
        "FlextUtilitiesContext": ".context",
        "FlextUtilitiesConversion": ".conversion",
        "FlextUtilitiesDiscovery": ".discovery",
        "FlextUtilitiesDispatcherExecute": ".dispatcher_execute",
        "FlextUtilitiesDomain": ".domain",
        "FlextUtilitiesEnforcement": ".enforcement",
        "FlextUtilitiesEnforcementCollect": ".enforcement_collect",
        "FlextUtilitiesEnforcementEmit": ".enforcement_emit",
        "FlextUtilitiesEnum": ".enums",
        "FlextUtilitiesFamilySurface": ".family_surface",
        "FlextUtilitiesFiles": ".files",
        "FlextUtilitiesGenerators": ".generators",
        "FlextUtilitiesGuards": ".guards",
        "FlextUtilitiesGuardsTypeCore": ".guards_type_core",
        "FlextUtilitiesGuardsTypeModel": ".guards_type_model",
        "FlextUtilitiesGuardsTypeProtocol": ".guards_type_protocol",
        "FlextUtilitiesGuardsTypeProtocolSpecsMixin": "._guards_type_protocol_specs",
        "FlextUtilitiesGuardsTypeProtocolStringMixin": "._guards_type_protocol_string",
        "FlextUtilitiesHandler": ".handler",
        "FlextUtilitiesLogging": ".logging",
        "FlextUtilitiesLoggingConfig": ".logging_config",
        "FlextUtilitiesLoggingContext": ".logging_context",
        "FlextUtilitiesLoggingStream": "._logging_config.logging_stream",
        "FlextUtilitiesMapper": ".mapper",
        "FlextUtilitiesMapperAccess": ".mapper_access",
        "FlextUtilitiesMapperExtract": ".mapper_extract",
        "FlextUtilitiesMetadata": ".metadata",
        "FlextUtilitiesModel": ".model",
        "FlextUtilitiesModelOptions": ".model_options",
        "FlextUtilitiesModelRuntime": ".model_runtime",
        "FlextUtilitiesParser": ".parser",
        "FlextUtilitiesParserCoerce": ".parser_coerce",
        "FlextUtilitiesParserTargets": ".parser_targets",
        "FlextUtilitiesProjectMetadata": ".project_metadata",
        "FlextUtilitiesPydantic": ".pydantic",
        "FlextUtilitiesReliability": ".reliability",
        "FlextUtilitiesRuntimeViolationRegistry": ".runtime_violation_registry",
        "FlextUtilitiesSettings": ".settings",
        "FlextUtilitiesText": ".text",
        "FlextUtilitiesTypingAdapterJson": ".typeadapter_json",
        "FlextUtilitiesTypingAdapterScalars": ".typeadapter_scalars",
        "_beartype": "._beartype",
        "_checker": "._checker",
        "_enforcement_collect": "._enforcement_collect",
        "_enforcement": "._enforcement",
        "_logging_config": "._logging_config",
        "_logging_context": "._logging_context",
        "_mapper_access": "._mapper_access",
        "_mapper_extract": "._mapper_extract",
        "_parser_targets": "._parser_targets",
        "alias_first_violation": ("._beartype._class_visitor.alias_rules"),
    }),
    public_exports=__all__,
)[0]
