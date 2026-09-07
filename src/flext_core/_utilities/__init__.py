# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities package."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import (
        _beartype as _beartype,
        _checker_parts as _checker_parts,
        _enforcement_collect_parts as _enforcement_collect_parts,
        _enforcement_parts as _enforcement_parts,
        _logging_config_parts as _logging_config_parts,
        _logging_context_parts as _logging_context_parts,
        _mapper_access_parts as _mapper_access_parts,
        _mapper_extract_parts as _mapper_extract_parts,
        _parser_targets_parts as _parser_targets_parts,
    )
    from ._beartype._alias_visitor import FlextUtilitiesBeartypeAliasVisitor
    from ._beartype._class_visitor_parts._parts.class_visitor_part_02_01 import (
        alias_first_violation,
    )
    from ._beartype._class_visitor_parts._parts.class_visitor_part_02_02 import (
        redundant_inner_violation,
        self_ref_violation,
    )
    from ._beartype._library_visitor import FlextUtilitiesBeartypeLibraryVisitor
    from ._beartype.attr_visitor import FlextUtilitiesBeartypeAttrVisitor
    from ._beartype.class_visitor import FlextUtilitiesBeartypeClassVisitor
    from ._beartype.deprecated_visitor import FlextUtilitiesBeartypeDeprecatedVisitor
    from ._beartype.field_visitor import FlextUtilitiesBeartypeFieldVisitor
    from ._beartype.helpers import FlextUtilitiesBeartypeHelpers
    from ._beartype.import_visitor import FlextUtilitiesBeartypeImportVisitor
    from ._beartype.method_visitor import FlextUtilitiesBeartypeMethodVisitor
    from ._beartype.module_visitor import FlextUtilitiesBeartypeModuleVisitor
    from ._context_crud_set import FlextUtilitiesContextCrudSetMixin
    from ._enforcement_parts.enforcement_part_05 import FlextUtilitiesEnforcement
    from ._enforcement_parts.enforcement_part_06 import EXTENDED_PREDICATE_BINDINGS
    from ._guards_type_protocol_specs import FlextUtilitiesGuardsTypeProtocolSpecsMixin
    from ._guards_type_protocol_string import (
        FlextUtilitiesGuardsTypeProtocolStringMixin,
    )
    from ._guards_type_protocol_types import ProtocolGuardInput
    from .args import FlextUtilitiesArgs
    from .beartype_conf import FlextUtilitiesBeartypeConf
    from .beartype_engine import FlextUtilitiesBeartypeEngine, ube
    from .beartype_typingext_patch import FlextUtilitiesBeartypeTypingExtPatch
    from .checker import FlextUtilitiesChecker
    from .collection import FlextUtilitiesCollection
    from .collection_iter import FlextUtilitiesCollectionIter
    from .collection_merge import FlextUtilitiesCollectionMerge
    from .config import FlextUtilitiesConfig
    from .console import FlextUtilitiesConsole
    from .context import FlextUtilitiesContext
    from .context_crud import FlextUtilitiesContextCrud
    from .context_lifecycle import FlextUtilitiesContextLifecycle
    from .context_state import FlextUtilitiesContextState
    from .conversion import FlextUtilitiesConversion
    from .discovery import FlextUtilitiesDiscovery
    from .dispatcher_execute import execute_dispatcher_handler
    from .domain import FlextUtilitiesDomain
    from .enforcement import PREDICATE_BINDINGS
    from .enforcement_collect import FlextUtilitiesEnforcementCollect
    from .enforcement_emit import FlextUtilitiesEnforcementEmit
    from .enum import FlextUtilitiesEnum
    from .generators import FlextUtilitiesGenerators
    from .guards import FlextUtilitiesGuards
    from .guards_type_core import FlextUtilitiesGuardsTypeCore
    from .guards_type_model import FlextUtilitiesGuardsTypeModel
    from .guards_type_protocol import FlextUtilitiesGuardsTypeProtocol
    from .handler import FlextUtilitiesHandler
    from .logging_config import FlextUtilitiesLoggingConfig
    from .logging_context import FlextUtilitiesLoggingContext
    from .mapper import FlextUtilitiesMapper
    from .mapper_access import FlextUtilitiesMapperAccess
    from .mapper_extract import FlextUtilitiesMapperExtract
    from .model import FlextUtilitiesModel
    from .model_options import FlextUtilitiesModelOptions
    from .model_runtime import FlextUtilitiesModelRuntime
    from .parser import FlextUtilitiesParser
    from .parser_coerce import FlextUtilitiesParserCoerce
    from .parser_targets import FlextUtilitiesParserTargets
    from .project_metadata import FlextUtilitiesProjectMetadata
    from .pydantic import FlextUtilitiesPydantic
    from .reliability import FlextUtilitiesReliability
    from .runtime_violation_registry import FlextUtilitiesRuntimeViolationRegistry
    from .settings import FlextUtilitiesSettings
    from .text import FlextUtilitiesText
__all__: tuple[str, ...] = (
    "EXTENDED_PREDICATE_BINDINGS",
    "PREDICATE_BINDINGS",
    "FlextUtilitiesArgs",
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
    "FlextUtilitiesBeartypeModuleVisitor",
    "FlextUtilitiesBeartypeTypingExtPatch",
    "FlextUtilitiesChecker",
    "FlextUtilitiesCollection",
    "FlextUtilitiesCollectionIter",
    "FlextUtilitiesCollectionMerge",
    "FlextUtilitiesConfig",
    "FlextUtilitiesConsole",
    "FlextUtilitiesContext",
    "FlextUtilitiesContextCrud",
    "FlextUtilitiesContextCrudSetMixin",
    "FlextUtilitiesContextLifecycle",
    "FlextUtilitiesContextState",
    "FlextUtilitiesConversion",
    "FlextUtilitiesDiscovery",
    "FlextUtilitiesDomain",
    "FlextUtilitiesEnforcement",
    "FlextUtilitiesEnforcementCollect",
    "FlextUtilitiesEnforcementEmit",
    "FlextUtilitiesEnum",
    "FlextUtilitiesGenerators",
    "FlextUtilitiesGuards",
    "FlextUtilitiesGuardsTypeCore",
    "FlextUtilitiesGuardsTypeModel",
    "FlextUtilitiesGuardsTypeProtocol",
    "FlextUtilitiesGuardsTypeProtocolSpecsMixin",
    "FlextUtilitiesGuardsTypeProtocolStringMixin",
    "FlextUtilitiesHandler",
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
    "execute_dispatcher_handler",
    "redundant_inner_violation",
    "self_ref_violation",
    "ube",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._beartype": ("_beartype",),
            "._beartype._alias_visitor": ("FlextUtilitiesBeartypeAliasVisitor",),
            "._beartype._class_visitor_parts._parts.class_visitor_part_02_01": (
                "alias_first_violation",
            ),
            "._beartype._class_visitor_parts._parts.class_visitor_part_02_02": (
                "redundant_inner_violation",
                "self_ref_violation",
            ),
            "._beartype._library_visitor": ("FlextUtilitiesBeartypeLibraryVisitor",),
            "._beartype.attr_visitor": ("FlextUtilitiesBeartypeAttrVisitor",),
            "._beartype.class_visitor": ("FlextUtilitiesBeartypeClassVisitor",),
            "._beartype.deprecated_visitor": (
                "FlextUtilitiesBeartypeDeprecatedVisitor",
            ),
            "._beartype.field_visitor": ("FlextUtilitiesBeartypeFieldVisitor",),
            "._beartype.helpers": ("FlextUtilitiesBeartypeHelpers",),
            "._beartype.import_visitor": ("FlextUtilitiesBeartypeImportVisitor",),
            "._beartype.method_visitor": ("FlextUtilitiesBeartypeMethodVisitor",),
            "._beartype.module_visitor": ("FlextUtilitiesBeartypeModuleVisitor",),
            "._checker_parts": ("_checker_parts",),
            "._context_crud_set": ("FlextUtilitiesContextCrudSetMixin",),
            "._enforcement_collect_parts": ("_enforcement_collect_parts",),
            "._enforcement_parts": ("_enforcement_parts",),
            "._enforcement_parts.enforcement_part_05": ("FlextUtilitiesEnforcement",),
            "._enforcement_parts.enforcement_part_06": ("EXTENDED_PREDICATE_BINDINGS",),
            "._guards_type_protocol_specs": (
                "FlextUtilitiesGuardsTypeProtocolSpecsMixin",
            ),
            "._guards_type_protocol_string": (
                "FlextUtilitiesGuardsTypeProtocolStringMixin",
            ),
            "._guards_type_protocol_types": ("ProtocolGuardInput",),
            "._logging_config_parts": ("_logging_config_parts",),
            "._logging_context_parts": ("_logging_context_parts",),
            "._mapper_access_parts": ("_mapper_access_parts",),
            "._mapper_extract_parts": ("_mapper_extract_parts",),
            "._parser_targets_parts": ("_parser_targets_parts",),
            ".args": ("FlextUtilitiesArgs",),
            ".beartype_conf": ("FlextUtilitiesBeartypeConf",),
            ".beartype_engine": ("FlextUtilitiesBeartypeEngine", "ube"),
            ".beartype_typingext_patch": ("FlextUtilitiesBeartypeTypingExtPatch",),
            ".checker": ("FlextUtilitiesChecker",),
            ".collection": ("FlextUtilitiesCollection",),
            ".collection_iter": ("FlextUtilitiesCollectionIter",),
            ".collection_merge": ("FlextUtilitiesCollectionMerge",),
            ".config": ("FlextUtilitiesConfig",),
            ".console": ("FlextUtilitiesConsole",),
            ".context": ("FlextUtilitiesContext",),
            ".context_crud": ("FlextUtilitiesContextCrud",),
            ".context_lifecycle": ("FlextUtilitiesContextLifecycle",),
            ".context_state": ("FlextUtilitiesContextState",),
            ".conversion": ("FlextUtilitiesConversion",),
            ".discovery": ("FlextUtilitiesDiscovery",),
            ".dispatcher_execute": ("execute_dispatcher_handler",),
            ".domain": ("FlextUtilitiesDomain",),
            ".enforcement": ("PREDICATE_BINDINGS",),
            ".enforcement_collect": ("FlextUtilitiesEnforcementCollect",),
            ".enforcement_emit": ("FlextUtilitiesEnforcementEmit",),
            ".enum": ("FlextUtilitiesEnum",),
            ".generators": ("FlextUtilitiesGenerators",),
            ".guards": ("FlextUtilitiesGuards",),
            ".guards_type_core": ("FlextUtilitiesGuardsTypeCore",),
            ".guards_type_model": ("FlextUtilitiesGuardsTypeModel",),
            ".guards_type_protocol": ("FlextUtilitiesGuardsTypeProtocol",),
            ".handler": ("FlextUtilitiesHandler",),
            ".logging_config": ("FlextUtilitiesLoggingConfig",),
            ".logging_context": ("FlextUtilitiesLoggingContext",),
            ".mapper": ("FlextUtilitiesMapper",),
            ".mapper_access": ("FlextUtilitiesMapperAccess",),
            ".mapper_extract": ("FlextUtilitiesMapperExtract",),
            ".model": ("FlextUtilitiesModel",),
            ".model_options": ("FlextUtilitiesModelOptions",),
            ".model_runtime": ("FlextUtilitiesModelRuntime",),
            ".parser": ("FlextUtilitiesParser",),
            ".parser_coerce": ("FlextUtilitiesParserCoerce",),
            ".parser_targets": ("FlextUtilitiesParserTargets",),
            ".project_metadata": ("FlextUtilitiesProjectMetadata",),
            ".pydantic": ("FlextUtilitiesPydantic",),
            ".reliability": ("FlextUtilitiesReliability",),
            ".runtime_violation_registry": ("FlextUtilitiesRuntimeViolationRegistry",),
            ".settings": ("FlextUtilitiesSettings",),
            ".text": ("FlextUtilitiesText",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
