# AUTO-GENERATED FILE — Regenerate with: make gen
"""Utilities package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._beartype._alias_visitor import (
        FlextUtilitiesBeartypeAliasVisitor,
    )
    from flext_core._utilities._beartype._class_visitor_parts.class_visitor_part_03 import (
        FlextUtilitiesBeartypeClassVisitor,
    )
    from flext_core._utilities._beartype._library_visitor import (
        FlextUtilitiesBeartypeLibraryVisitor,
    )
    from flext_core._utilities._beartype.attr_visitor import (
        FlextUtilitiesBeartypeAttrVisitor,
    )
    from flext_core._utilities._beartype.deprecated_visitor import (
        FlextUtilitiesBeartypeDeprecatedVisitor,
    )
    from flext_core._utilities._beartype.field_visitor import (
        FlextUtilitiesBeartypeFieldVisitor,
    )
    from flext_core._utilities._beartype.helpers import FlextUtilitiesBeartypeHelpers
    from flext_core._utilities._beartype.import_visitor import (
        FlextUtilitiesBeartypeImportVisitor,
    )
    from flext_core._utilities._beartype.method_visitor import (
        FlextUtilitiesBeartypeMethodVisitor,
    )
    from flext_core._utilities._beartype.module_visitor import (
        FlextUtilitiesBeartypeModuleVisitor,
    )
    from flext_core._utilities._checker_parts.checker_part_03 import (
        FlextUtilitiesChecker,
    )
    from flext_core._utilities._context_crud_set import (
        FlextUtilitiesContextCrudSetMixin,
    )
    from flext_core._utilities._enforcement_collect_parts.enforcement_collect_part_02 import (
        FlextUtilitiesEnforcementCollect,
    )
    from flext_core._utilities._enforcement_parts.enforcement_part_01 import (
        PREDICATE_BINDINGS,
    )
    from flext_core._utilities._enforcement_parts.enforcement_part_05 import (
        FlextUtilitiesEnforcement,
    )
    from flext_core._utilities._enforcement_parts.enforcement_part_06 import (
        EXTENDED_PREDICATE_BINDINGS,
    )
    from flext_core._utilities._generators_parts.generators_part_02 import (
        FlextUtilitiesGenerators,
    )
    from flext_core._utilities._guards_parts.guards_part_02 import FlextUtilitiesGuards
    from flext_core._utilities._guards_type_protocol_specs import (
        FlextUtilitiesGuardsTypeProtocolSpecsMixin,
    )
    from flext_core._utilities._guards_type_protocol_string import (
        FlextUtilitiesGuardsTypeProtocolStringMixin,
    )
    from flext_core._utilities._guards_type_protocol_types import ProtocolGuardInput
    from flext_core._utilities._logging_config_parts.logging_config_part_03 import (
        FlextUtilitiesLoggingConfig,
    )
    from flext_core._utilities._logging_context_parts.logging_context_part_02 import (
        FlextUtilitiesLoggingContext,
    )
    from flext_core._utilities._mapper_access_parts.mapper_access_part_02 import (
        FlextUtilitiesMapperAccess,
    )
    from flext_core._utilities._mapper_extract_parts.mapper_extract_part_02 import (
        FlextUtilitiesMapperExtract,
    )
    from flext_core._utilities._parser_targets_parts.parser_targets_part_02 import (
        FlextUtilitiesParserTargets,
    )
    from flext_core._utilities._project_metadata_parts.project_metadata_part_04 import (
        FlextUtilitiesProjectMetadata,
    )
    from flext_core._utilities.args import FlextUtilitiesArgs
    from flext_core._utilities.beartype_conf import FlextUtilitiesBeartypeConf
    from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine
    from flext_core._utilities.beartype_typingext_patch import (
        FlextUtilitiesBeartypeTypingExtPatch,
    )
    from flext_core._utilities.collection import FlextUtilitiesCollection
    from flext_core._utilities.collection_iter import FlextUtilitiesCollectionIter
    from flext_core._utilities.collection_merge import FlextUtilitiesCollectionMerge
    from flext_core._utilities.context import FlextUtilitiesContext
    from flext_core._utilities.context_crud import FlextUtilitiesContextCrud
    from flext_core._utilities.context_lifecycle import FlextUtilitiesContextLifecycle
    from flext_core._utilities.context_state import FlextUtilitiesContextState
    from flext_core._utilities.conversion import FlextUtilitiesConversion
    from flext_core._utilities.discovery import FlextUtilitiesDiscovery
    from flext_core._utilities.domain import FlextUtilitiesDomain
    from flext_core._utilities.enforcement_emit import FlextUtilitiesEnforcementEmit
    from flext_core._utilities.enum import FlextUtilitiesEnum
    from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
    from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
    from flext_core._utilities.guards_type_protocol import (
        FlextUtilitiesGuardsTypeProtocol,
    )
    from flext_core._utilities.handler import FlextUtilitiesHandler
    from flext_core._utilities.mapper import FlextUtilitiesMapper
    from flext_core._utilities.model import FlextUtilitiesModel
    from flext_core._utilities.model_options import FlextUtilitiesModelOptions
    from flext_core._utilities.model_runtime import FlextUtilitiesModelRuntime
    from flext_core._utilities.parser import FlextUtilitiesParser
    from flext_core._utilities.parser_coerce import FlextUtilitiesParserCoerce
    from flext_core._utilities.pydantic import FlextUtilitiesPydantic
    from flext_core._utilities.reliability import FlextUtilitiesReliability
    from flext_core._utilities.runtime_violation_registry import (
        FlextUtilitiesRuntimeViolationRegistry,
    )
    from flext_core._utilities.settings import FlextUtilitiesSettings
    from flext_core._utilities.text import FlextUtilitiesText
_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._beartype": ("_beartype",),
        "._beartype._alias_visitor": ("FlextUtilitiesBeartypeAliasVisitor",),
        "._beartype._class_visitor_parts.class_visitor_part_03": (
            "FlextUtilitiesBeartypeClassVisitor",
        ),
        "._beartype._library_visitor": ("FlextUtilitiesBeartypeLibraryVisitor",),
        "._beartype.attr_visitor": ("FlextUtilitiesBeartypeAttrVisitor",),
        "._beartype.deprecated_visitor": ("FlextUtilitiesBeartypeDeprecatedVisitor",),
        "._beartype.field_visitor": ("FlextUtilitiesBeartypeFieldVisitor",),
        "._beartype.helpers": ("FlextUtilitiesBeartypeHelpers",),
        "._beartype.import_visitor": ("FlextUtilitiesBeartypeImportVisitor",),
        "._beartype.method_visitor": ("FlextUtilitiesBeartypeMethodVisitor",),
        "._beartype.module_visitor": ("FlextUtilitiesBeartypeModuleVisitor",),
        "._checker_parts": ("_checker_parts",),
        "._checker_parts.checker_part_03": ("FlextUtilitiesChecker",),
        "._context_crud_set": ("FlextUtilitiesContextCrudSetMixin",),
        "._enforcement_collect_parts": ("_enforcement_collect_parts",),
        "._enforcement_collect_parts.enforcement_collect_part_02": (
            "FlextUtilitiesEnforcementCollect",
        ),
        "._enforcement_parts": ("_enforcement_parts",),
        "._enforcement_parts.enforcement_part_01": ("PREDICATE_BINDINGS",),
        "._enforcement_parts.enforcement_part_05": ("FlextUtilitiesEnforcement",),
        "._enforcement_parts.enforcement_part_06": ("EXTENDED_PREDICATE_BINDINGS",),
        "._generators_parts": ("_generators_parts",),
        "._generators_parts.generators_part_02": ("FlextUtilitiesGenerators",),
        "._guards_parts": ("_guards_parts",),
        "._guards_parts.guards_part_02": ("FlextUtilitiesGuards",),
        "._guards_type_protocol_specs": ("FlextUtilitiesGuardsTypeProtocolSpecsMixin",),
        "._guards_type_protocol_string": (
            "FlextUtilitiesGuardsTypeProtocolStringMixin",
        ),
        "._guards_type_protocol_types": ("ProtocolGuardInput",),
        "._logging_config_parts": ("_logging_config_parts",),
        "._logging_config_parts.logging_config_part_03": (
            "FlextUtilitiesLoggingConfig",
        ),
        "._logging_context_parts": ("_logging_context_parts",),
        "._logging_context_parts.logging_context_part_02": (
            "FlextUtilitiesLoggingContext",
        ),
        "._mapper_access_parts": ("_mapper_access_parts",),
        "._mapper_access_parts.mapper_access_part_02": ("FlextUtilitiesMapperAccess",),
        "._mapper_extract_parts": ("_mapper_extract_parts",),
        "._mapper_extract_parts.mapper_extract_part_02": (
            "FlextUtilitiesMapperExtract",
        ),
        "._parser_targets_parts": ("_parser_targets_parts",),
        "._parser_targets_parts.parser_targets_part_02": (
            "FlextUtilitiesParserTargets",
        ),
        "._project_metadata_parts": ("_project_metadata_parts",),
        "._project_metadata_parts.project_metadata_part_04": (
            "FlextUtilitiesProjectMetadata",
        ),
        ".args": ("FlextUtilitiesArgs",),
        ".beartype_conf": ("FlextUtilitiesBeartypeConf",),
        ".beartype_engine": (
            "FlextUtilitiesBeartypeEngine",
            "ube",
        ),
        ".beartype_typingext_patch": ("FlextUtilitiesBeartypeTypingExtPatch",),
        ".collection": ("FlextUtilitiesCollection",),
        ".collection_iter": ("FlextUtilitiesCollectionIter",),
        ".collection_merge": ("FlextUtilitiesCollectionMerge",),
        ".context": ("FlextUtilitiesContext",),
        ".context_crud": ("FlextUtilitiesContextCrud",),
        ".context_lifecycle": ("FlextUtilitiesContextLifecycle",),
        ".context_state": ("FlextUtilitiesContextState",),
        ".conversion": ("FlextUtilitiesConversion",),
        ".discovery": ("FlextUtilitiesDiscovery",),
        ".dispatcher_execute": ("execute_dispatcher_handler",),
        ".domain": ("FlextUtilitiesDomain",),
        ".enforcement_emit": ("FlextUtilitiesEnforcementEmit",),
        ".enum": ("FlextUtilitiesEnum",),
        ".guards_type_core": ("FlextUtilitiesGuardsTypeCore",),
        ".guards_type_model": ("FlextUtilitiesGuardsTypeModel",),
        ".guards_type_protocol": ("FlextUtilitiesGuardsTypeProtocol",),
        ".handler": ("FlextUtilitiesHandler",),
        ".mapper": ("FlextUtilitiesMapper",),
        ".model": ("FlextUtilitiesModel",),
        ".model_options": ("FlextUtilitiesModelOptions",),
        ".model_runtime": ("FlextUtilitiesModelRuntime",),
        ".parser": ("FlextUtilitiesParser",),
        ".parser_coerce": ("FlextUtilitiesParserCoerce",),
        ".pydantic": ("FlextUtilitiesPydantic",),
        ".reliability": ("FlextUtilitiesReliability",),
        ".runtime_violation_registry": ("FlextUtilitiesRuntimeViolationRegistry",),
        ".settings": ("FlextUtilitiesSettings",),
        ".text": ("FlextUtilitiesText",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
