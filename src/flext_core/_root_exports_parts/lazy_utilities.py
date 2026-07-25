"""Generated root lazy export map: lazy_utilities."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

ROOT_LAZY_UTILITIES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "._utilities._beartype.attr_visitor": ("FlextUtilitiesBeartypeAttrVisitor",),
    "._utilities._beartype.class_visitor": ("FlextUtilitiesBeartypeClassVisitor",),
    "._utilities._beartype.deprecated_visitor": (
        "FlextUtilitiesBeartypeDeprecatedVisitor",
    ),
    "._utilities._beartype.field_visitor": ("FlextUtilitiesBeartypeFieldVisitor",),
    "._utilities._beartype.helpers": ("FlextUtilitiesBeartypeHelpers",),
    "._utilities._beartype.import_visitor": ("FlextUtilitiesBeartypeImportVisitor",),
    "._utilities._beartype.method_visitor": ("FlextUtilitiesBeartypeMethodVisitor",),
    "._utilities._beartype.module_visitor": ("FlextUtilitiesBeartypeModuleVisitor",),
    "._utilities.args": ("FlextUtilitiesArgs",),
    "._utilities.beartype_conf": ("FlextUtilitiesBeartypeConf",),
    "._utilities.beartype_engine": ("FlextUtilitiesBeartypeEngine", "ube"),
    "._utilities.beartype_typingext_patch": ("FlextUtilitiesBeartypeTypingExtPatch",),
    "._utilities.checker": ("FlextUtilitiesChecker",),
    "._utilities.collection": ("FlextUtilitiesCollection",),
    "._utilities.collection_iter": ("FlextUtilitiesCollectionIter",),
    "._utilities.collection_merge": ("FlextUtilitiesCollectionMerge",),
    "._utilities.context": ("FlextUtilitiesContext",),
    "._utilities.context_crud": ("FlextUtilitiesContextCrud",),
    "._utilities.context_lifecycle": ("FlextUtilitiesContextLifecycle",),
    "._utilities.context_state": ("FlextUtilitiesContextState",),
    "._utilities.conversion": ("FlextUtilitiesConversion",),
    "._utilities.discovery": ("FlextUtilitiesDiscovery",),
    "._utilities.dispatcher_execute": ("execute_dispatcher_handler",),
    "._utilities.domain": ("FlextUtilitiesDomain",),
    "._utilities.enforcement": ("FlextUtilitiesEnforcement",),
    "._utilities.enforcement_collect": ("FlextUtilitiesEnforcementCollect",),
    "._utilities.enforcement_emit": ("FlextUtilitiesEnforcementEmit",),
    "._utilities.enum": ("FlextUtilitiesEnum",),
    "._utilities.generators": ("FlextUtilitiesGenerators",),
    "._utilities.guards": ("FlextUtilitiesGuards",),
    "._utilities.guards_type_core": ("FlextUtilitiesGuardsTypeCore",),
    "._utilities.guards_type_model": ("FlextUtilitiesGuardsTypeModel",),
    "._utilities.guards_type_protocol": ("FlextUtilitiesGuardsTypeProtocol",),
    "._utilities.handler": ("FlextUtilitiesHandler",),
    "._utilities.logging_config": ("FlextUtilitiesLoggingConfig",),
    "._utilities.logging_context": ("FlextUtilitiesLoggingContext",),
    "._utilities.mapper": ("FlextUtilitiesMapper",),
    "._utilities.mapper_access": ("FlextUtilitiesMapperAccess",),
    "._utilities.mapper_extract": ("FlextUtilitiesMapperExtract",),
    "._utilities.model": ("FlextUtilitiesModel",),
    "._utilities.model_options": ("FlextUtilitiesModelOptions",),
    "._utilities.model_runtime": ("FlextUtilitiesModelRuntime",),
    "._utilities.parser": ("FlextUtilitiesParser",),
    "._utilities.parser_coerce": ("FlextUtilitiesParserCoerce",),
    "._utilities.parser_targets": ("FlextUtilitiesParserTargets",),
    "._utilities.project_metadata": ("FlextUtilitiesProjectMetadata",),
    "._utilities.pydantic": ("FlextUtilitiesPydantic",),
    "._utilities.reliability": ("FlextUtilitiesReliability",),
    "._utilities.runtime_violation_registry": (
        "FlextUtilitiesRuntimeViolationRegistry",
    ),
    "._utilities.settings": ("FlextUtilitiesSettings",),
    "._utilities.text": ("FlextUtilitiesText",),
})

__all__: list[str] = ["ROOT_LAZY_UTILITIES"]
