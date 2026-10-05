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
    from flext_core._utilities import (
        _beartype,
        _checker_parts,
        _enforcement_collect_parts,
        _enforcement_parts,
        _logging_config_parts,
        _logging_context_parts,
        _mapper_access_parts,
        _mapper_extract_parts,
        _parser_targets_parts,
    )
    from flext_core._utilities._beartype._alias_visitor import (
        FlextUtilitiesBeartypeAliasVisitor,
    )
    from flext_core._utilities._beartype._class_visitor_parts._parts.class_visitor_part_02_01 import (
        alias_first_violation,
    )
    from flext_core._utilities._beartype._class_visitor_parts._parts.class_visitor_part_02_02 import (
        redundant_inner_violation,
        self_ref_violation,
    )
    from flext_core._utilities._beartype._library_visitor import (
        FlextUtilitiesBeartypeLibraryVisitor,
    )
    from flext_core._utilities._beartype.attr_visitor import (
        FlextUtilitiesBeartypeAttrVisitor,
    )
    from flext_core._utilities._beartype.class_visitor import (
        FlextUtilitiesBeartypeClassVisitor,
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
    from flext_core._utilities._beartype.module_source import (
        FlextUtilitiesBeartypeModuleSource,
    )
    from flext_core._utilities._beartype.module_visitor import (
        FlextUtilitiesBeartypeModuleVisitor,
    )
    from flext_core._utilities._beartype.type_aliases import (
        FlextUtilitiesBeartypeTypeAliases,
    )
    from flext_core._utilities._guards_type_protocol_specs import (
        FlextUtilitiesGuardsTypeProtocolSpecsMixin,
    )
    from flext_core._utilities._guards_type_protocol_string import (
        FlextUtilitiesGuardsTypeProtocolStringMixin,
    )
    from flext_core._utilities._guards_type_protocol_types import ProtocolGuardInput
    from flext_core._utilities.args import FlextUtilitiesArgs
    from flext_core._utilities.base import FlextUtilitiesBase
    from flext_core._utilities.beartype_conf import FlextUtilitiesBeartypeConf
    from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine
    from flext_core._utilities.beartype_typingext_patch import (
        FlextUtilitiesBeartypeTypingExtPatch,
    )
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
    "FlextUtilitiesBeartypeFieldVisitor",
    "FlextUtilitiesBeartypeHelpers",
    "FlextUtilitiesBeartypeImportVisitor",
    "FlextUtilitiesBeartypeLibraryVisitor",
    "FlextUtilitiesBeartypeMethodVisitor",
    "FlextUtilitiesBeartypeModuleSource",
    "FlextUtilitiesBeartypeModuleVisitor",
    "FlextUtilitiesBeartypeTypeAliases",
    "FlextUtilitiesBeartypeTypingExtPatch",
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
    "FlextUtilitiesMapper",
    "FlextUtilitiesMapperAccess",
    "FlextUtilitiesMapperExtract",
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
    "ProtocolGuardInput",
    "_beartype",
    "_checker_parts",
    "_enforcement_collect_parts",
    "_enforcement_parts",
    "_logging_config_parts",
    "_logging_context_parts",
    "_mapper_access_parts",
    "_mapper_extract_parts",
    "_parser_targets_parts",
    "alias_first_violation",
    "redundant_inner_violation",
    "self_ref_violation",
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
        "FlextUtilitiesBeartypeConf": ".beartype_conf",
        "FlextUtilitiesBeartypeDeprecatedVisitor": "._beartype.deprecated_visitor",
        "FlextUtilitiesBeartypeEngine": ".beartype_engine",
        "FlextUtilitiesBeartypeFieldVisitor": "._beartype.field_visitor",
        "FlextUtilitiesBeartypeHelpers": "._beartype.helpers",
        "FlextUtilitiesBeartypeImportVisitor": "._beartype.import_visitor",
        "FlextUtilitiesBeartypeLibraryVisitor": "._beartype._library_visitor",
        "FlextUtilitiesBeartypeMethodVisitor": "._beartype.method_visitor",
        "FlextUtilitiesBeartypeModuleSource": "._beartype.module_source",
        "FlextUtilitiesBeartypeModuleVisitor": "._beartype.module_visitor",
        "FlextUtilitiesBeartypeTypeAliases": "._beartype.type_aliases",
        "FlextUtilitiesBeartypeTypingExtPatch": ".beartype_typingext_patch",
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
        "FlextUtilitiesMapper": ".mapper",
        "FlextUtilitiesMapperAccess": ".mapper_access",
        "FlextUtilitiesMapperExtract": ".mapper_extract",
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
        "ProtocolGuardInput": "._guards_type_protocol_types",
        "_beartype": "._beartype",
        "_checker_parts": "._checker_parts",
        "_enforcement_collect_parts": "._enforcement_collect_parts",
        "_enforcement_parts": "._enforcement_parts",
        "_logging_config_parts": "._logging_config_parts",
        "_logging_context_parts": "._logging_context_parts",
        "_mapper_access_parts": "._mapper_access_parts",
        "_mapper_extract_parts": "._mapper_extract_parts",
        "_parser_targets_parts": "._parser_targets_parts",
        "alias_first_violation": (
            "._beartype._class_visitor_parts._parts.class_visitor_part_02_01"
        ),
        "redundant_inner_violation": (
            "._beartype._class_visitor_parts._parts.class_visitor_part_02_02"
        ),
        "self_ref_violation": (
            "._beartype._class_visitor_parts._parts.class_visitor_part_02_02"
        ),
    }),
    public_exports=__all__,
)
