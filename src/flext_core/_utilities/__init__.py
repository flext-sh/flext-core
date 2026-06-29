# AUTO-GENERATED FILE — Regenerate with: make gen
"""Utilities package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        "._beartype": ("_beartype",),
        "._beartype._class_visitor_parts.class_visitor_part_03": (
            "FlextUtilitiesBeartypeClassVisitor",
        ),
        "._beartype._helpers_parts.helpers_part_03": ("FlextUtilitiesBeartypeHelpers",),
        "._beartype.attr_visitor": ("FlextUtilitiesBeartypeAttrVisitor",),
        "._beartype.deprecated_visitor": ("FlextUtilitiesBeartypeDeprecatedVisitor",),
        "._beartype.field_visitor": ("FlextUtilitiesBeartypeFieldVisitor",),
        "._beartype.import_visitor": ("FlextUtilitiesBeartypeImportVisitor",),
        "._beartype.method_visitor": ("FlextUtilitiesBeartypeMethodVisitor",),
        "._beartype.module_visitor": ("FlextUtilitiesBeartypeModuleVisitor",),
        "._checker_parts": ("_checker_parts",),
        "._checker_parts.checker_part_03": ("FlextUtilitiesChecker",),
        "._enforcement_collect_parts": ("_enforcement_collect_parts",),
        "._enforcement_collect_parts.enforcement_collect_part_02": (
            "FlextUtilitiesEnforcementCollect",
        ),
        "._enforcement_parts": ("_enforcement_parts",),
        "._enforcement_parts.enforcement_part_01": ("PREDICATE_BINDINGS",),
        "._enforcement_parts.enforcement_part_05": ("FlextUtilitiesEnforcement",),
        "._generators_parts": ("_generators_parts",),
        "._generators_parts.generators_part_02": ("FlextUtilitiesGenerators",),
        "._guards_parts": ("_guards_parts",),
        "._guards_parts.guards_part_02": ("FlextUtilitiesGuards",),
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
        "._project_metadata_parts.project_metadata_part_03": (
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
