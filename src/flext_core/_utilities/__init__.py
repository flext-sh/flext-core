# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype import (
        FlextUtilitiesBeartypeAliasVisitor,
        FlextUtilitiesBeartypeAttrVisitor,
        FlextUtilitiesBeartypeClassVisitor,
        FlextUtilitiesBeartypeClassVisitorPart01,
        FlextUtilitiesBeartypeClassVisitorPart02,
        FlextUtilitiesBeartypeClassVisitorPart03,
        FlextUtilitiesBeartypeDeprecatedVisitor,
        FlextUtilitiesBeartypeFieldUnionScan,
        FlextUtilitiesBeartypeFieldVisitor,
        FlextUtilitiesBeartypeHelpers,
        FlextUtilitiesBeartypeHelpersPart01,
        FlextUtilitiesBeartypeHelpersPart02,
        FlextUtilitiesBeartypeHelpersPart03,
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
    from flext_core._utilities._checker import (
        FlextUtilitiesCheckerPart01,
        FlextUtilitiesCheckerPart02,
        FlextUtilitiesCheckerPart03,
    )
    from flext_core._utilities._enforcement import (
        FlextUtilitiesEnforcementPart02,
        FlextUtilitiesEnforcementPart03,
        FlextUtilitiesEnforcementPart04,
        FlextUtilitiesEnforcementPart05,
    )
    from flext_core._utilities._enforcement_collect import (
        FlextUtilitiesEnforcementCollectPart01,
        FlextUtilitiesEnforcementCollectPart02,
    )
    from flext_core._utilities._guards_type_protocol_specs import (
        FlextUtilitiesGuardsTypeProtocolSpecsMixin,
    )
    from flext_core._utilities._guards_type_protocol_string import (
        FlextUtilitiesGuardsTypeProtocolStringMixin,
    )
    from flext_core._utilities._logging_config import (
        FlextUtilitiesLoggingConfigPart01,
        FlextUtilitiesLoggingConfigPart02,
        FlextUtilitiesLoggingConfigPart03,
        FlextUtilitiesLoggingStream,
    )
    from flext_core._utilities._logging_context import (
        FlextUtilitiesLoggingContextPart01,
        FlextUtilitiesLoggingContextPart02,
    )
    from flext_core._utilities._mapper_access import (
        FlextUtilitiesMapperAccessPart01,
        FlextUtilitiesMapperAccessPart02,
    )
    from flext_core._utilities._mapper_extract import (
        FlextUtilitiesMapperExtractPart01,
        FlextUtilitiesMapperExtractPart02,
    )
    from flext_core._utilities._parser_targets import (
        FlextUtilitiesParserTargetsPart01,
        FlextUtilitiesParserTargetsPart02,
    )
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
    "FlextUtilitiesBeartypeClassVisitorPart01",
    "FlextUtilitiesBeartypeClassVisitorPart02",
    "FlextUtilitiesBeartypeClassVisitorPart03",
    "FlextUtilitiesBeartypeConf",
    "FlextUtilitiesBeartypeDeprecatedVisitor",
    "FlextUtilitiesBeartypeEngine",
    "FlextUtilitiesBeartypeFieldUnionScan",
    "FlextUtilitiesBeartypeFieldVisitor",
    "FlextUtilitiesBeartypeHelpers",
    "FlextUtilitiesBeartypeHelpersPart01",
    "FlextUtilitiesBeartypeHelpersPart02",
    "FlextUtilitiesBeartypeHelpersPart03",
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
    "FlextUtilitiesCheckerPart01",
    "FlextUtilitiesCheckerPart02",
    "FlextUtilitiesCheckerPart03",
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
    "FlextUtilitiesEnforcementCollectPart01",
    "FlextUtilitiesEnforcementCollectPart02",
    "FlextUtilitiesEnforcementEmit",
    "FlextUtilitiesEnforcementPart02",
    "FlextUtilitiesEnforcementPart03",
    "FlextUtilitiesEnforcementPart04",
    "FlextUtilitiesEnforcementPart05",
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
    "FlextUtilitiesLoggingConfigPart01",
    "FlextUtilitiesLoggingConfigPart02",
    "FlextUtilitiesLoggingConfigPart03",
    "FlextUtilitiesLoggingContext",
    "FlextUtilitiesLoggingContextPart01",
    "FlextUtilitiesLoggingContextPart02",
    "FlextUtilitiesLoggingStream",
    "FlextUtilitiesMapper",
    "FlextUtilitiesMapperAccess",
    "FlextUtilitiesMapperAccessPart01",
    "FlextUtilitiesMapperAccessPart02",
    "FlextUtilitiesMapperExtract",
    "FlextUtilitiesMapperExtractPart01",
    "FlextUtilitiesMapperExtractPart02",
    "FlextUtilitiesMetadata",
    "FlextUtilitiesModel",
    "FlextUtilitiesModelOptions",
    "FlextUtilitiesModelRuntime",
    "FlextUtilitiesParser",
    "FlextUtilitiesParserCoerce",
    "FlextUtilitiesParserTargets",
    "FlextUtilitiesParserTargetsPart01",
    "FlextUtilitiesParserTargetsPart02",
    "FlextUtilitiesProjectMetadata",
    "FlextUtilitiesPydantic",
    "FlextUtilitiesReliability",
    "FlextUtilitiesRuntimeViolationRegistry",
    "FlextUtilitiesSettings",
    "FlextUtilitiesText",
    "FlextUtilitiesTypingAdapterJson",
    "FlextUtilitiesTypingAdapterScalars",
    "alias_first_violation",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "PREDICATE_BINDINGS": ".enforcement",
        "FlextUtilitiesArgs": ".args",
        "FlextUtilitiesBase": ".base",
        "FlextUtilitiesBeartypeAliasVisitor": "._beartype._alias_visitor",
        "FlextUtilitiesBeartypeAttrVisitor": "._beartype.attr_visitor",
        "FlextUtilitiesBeartypeClassVisitor": "._beartype.class_visitor",
        "FlextUtilitiesBeartypeClassVisitorPart01": (
            "._beartype._class_visitor.visitor_base"
        ),
        "FlextUtilitiesBeartypeClassVisitorPart02": (
            "._beartype._class_visitor.visitor_checks"
        ),
        "FlextUtilitiesBeartypeClassVisitorPart03": "._beartype._class_visitor.visitor",
        "FlextUtilitiesBeartypeConf": ".beartype_conf",
        "FlextUtilitiesBeartypeDeprecatedVisitor": "._beartype.deprecated_visitor",
        "FlextUtilitiesBeartypeEngine": ".beartype_engine",
        "FlextUtilitiesBeartypeFieldUnionScan": "._beartype._field_union_scan",
        "FlextUtilitiesBeartypeFieldVisitor": "._beartype.field_visitor",
        "FlextUtilitiesBeartypeHelpers": "._beartype.helpers",
        "FlextUtilitiesBeartypeHelpersPart01": "._beartype._helpers.helpers_part_01",
        "FlextUtilitiesBeartypeHelpersPart02": "._beartype._helpers.helpers_part_02",
        "FlextUtilitiesBeartypeHelpersPart03": "._beartype._helpers.helpers_part_03",
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
        "FlextUtilitiesCheckerPart01": "._checker.checker_part_01",
        "FlextUtilitiesCheckerPart02": "._checker.checker_part_02",
        "FlextUtilitiesCheckerPart03": "._checker.checker_part_03",
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
        "FlextUtilitiesEnforcementCollectPart01": (
            "._enforcement_collect.enforcement_collect_part_01"
        ),
        "FlextUtilitiesEnforcementCollectPart02": (
            "._enforcement_collect.enforcement_collect_part_02"
        ),
        "FlextUtilitiesEnforcementEmit": ".enforcement_emit",
        "FlextUtilitiesEnforcementPart02": "._enforcement.enforcement_part_02",
        "FlextUtilitiesEnforcementPart03": "._enforcement.enforcement_part_03",
        "FlextUtilitiesEnforcementPart04": "._enforcement.enforcement_part_04",
        "FlextUtilitiesEnforcementPart05": "._enforcement.enforcement_part_05",
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
        "FlextUtilitiesLoggingConfigPart01": "._logging_config.logging_config_part_01",
        "FlextUtilitiesLoggingConfigPart02": "._logging_config.logging_config_part_02",
        "FlextUtilitiesLoggingConfigPart03": "._logging_config.logging_config_part_03",
        "FlextUtilitiesLoggingContext": ".logging_context",
        "FlextUtilitiesLoggingContextPart01": (
            "._logging_context.logging_context_part_01"
        ),
        "FlextUtilitiesLoggingContextPart02": (
            "._logging_context.logging_context_part_02"
        ),
        "FlextUtilitiesLoggingStream": "._logging_config.logging_stream",
        "FlextUtilitiesMapper": ".mapper",
        "FlextUtilitiesMapperAccess": ".mapper_access",
        "FlextUtilitiesMapperAccessPart01": "._mapper_access.mapper_access_part_01",
        "FlextUtilitiesMapperAccessPart02": "._mapper_access.mapper_access_part_02",
        "FlextUtilitiesMapperExtract": ".mapper_extract",
        "FlextUtilitiesMapperExtractPart01": "._mapper_extract.mapper_extract_part_01",
        "FlextUtilitiesMapperExtractPart02": "._mapper_extract.mapper_extract_part_02",
        "FlextUtilitiesMetadata": ".metadata",
        "FlextUtilitiesModel": ".model",
        "FlextUtilitiesModelOptions": ".model_options",
        "FlextUtilitiesModelRuntime": ".model_runtime",
        "FlextUtilitiesParser": ".parser",
        "FlextUtilitiesParserCoerce": ".parser_coerce",
        "FlextUtilitiesParserTargets": ".parser_targets",
        "FlextUtilitiesParserTargetsPart01": "._parser_targets.parser_targets_part_01",
        "FlextUtilitiesParserTargetsPart02": "._parser_targets.parser_targets_part_02",
        "FlextUtilitiesProjectMetadata": ".project_metadata",
        "FlextUtilitiesPydantic": ".pydantic",
        "FlextUtilitiesReliability": ".reliability",
        "FlextUtilitiesRuntimeViolationRegistry": ".runtime_violation_registry",
        "FlextUtilitiesSettings": ".settings",
        "FlextUtilitiesText": ".text",
        "FlextUtilitiesTypingAdapterJson": ".typeadapter_json",
        "FlextUtilitiesTypingAdapterScalars": ".typeadapter_scalars",
        "alias_first_violation": "._beartype._class_visitor.alias_rules",
    }),
    public_exports=__all__,
)
