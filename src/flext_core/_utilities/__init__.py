# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import defer_lazy_bases, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities.args import FlextUtilitiesArgs
    from flext_core._utilities.base import FlextUtilitiesBase
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
    from flext_core._utilities.package_version import FlextUtilitiesPackageVersion
    from flext_core._utilities.parser import FlextUtilitiesParser
    from flext_core._utilities.parser_coerce import FlextUtilitiesParserCoerce
    from flext_core._utilities.parser_targets import FlextUtilitiesParserTargets
    from flext_core._utilities.project_metadata import FlextUtilitiesProjectMetadata
    from flext_core._utilities.pydantic import FlextUtilitiesPydantic
    from flext_core._utilities.reliability import FlextUtilitiesReliability
    from flext_core._utilities.settings import FlextUtilitiesSettings
    from flext_core._utilities.text import FlextUtilitiesText
    from flext_core._utilities.typeadapter_json import FlextUtilitiesTypingAdapterJson
    from flext_core._utilities.typeadapter_scalars import (
        FlextUtilitiesTypingAdapterScalars,
    )


__all__: tuple[str, ...] = (
    "FlextUtilitiesArgs",
    "FlextUtilitiesBase",
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
    "FlextUtilitiesEnum",
    "FlextUtilitiesFamilySurface",
    "FlextUtilitiesFiles",
    "FlextUtilitiesGenerators",
    "FlextUtilitiesGuards",
    "FlextUtilitiesGuardsTypeCore",
    "FlextUtilitiesGuardsTypeModel",
    "FlextUtilitiesGuardsTypeProtocol",
    "FlextUtilitiesHandler",
    "FlextUtilitiesLogging",
    "FlextUtilitiesLoggingConfig",
    "FlextUtilitiesLoggingContext",
    "FlextUtilitiesMapper",
    "FlextUtilitiesMapperAccess",
    "FlextUtilitiesMapperExtract",
    "FlextUtilitiesMetadata",
    "FlextUtilitiesModel",
    "FlextUtilitiesModelOptions",
    "FlextUtilitiesModelRuntime",
    "FlextUtilitiesPackageVersion",
    "FlextUtilitiesParser",
    "FlextUtilitiesParserCoerce",
    "FlextUtilitiesParserTargets",
    "FlextUtilitiesProjectMetadata",
    "FlextUtilitiesPydantic",
    "FlextUtilitiesReliability",
    "FlextUtilitiesSettings",
    "FlextUtilitiesText",
    "FlextUtilitiesTypingAdapterJson",
    "FlextUtilitiesTypingAdapterScalars",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextUtilitiesArgs": ".args",
        "FlextUtilitiesBase": ".base",
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
        "FlextUtilitiesEnum": ".enums",
        "FlextUtilitiesFamilySurface": ".family_surface",
        "FlextUtilitiesFiles": ".files",
        "FlextUtilitiesGenerators": ".generators",
        "FlextUtilitiesGuards": ".guards",
        "FlextUtilitiesGuardsTypeCore": ".guards_type_core",
        "FlextUtilitiesGuardsTypeModel": ".guards_type_model",
        "FlextUtilitiesGuardsTypeProtocol": ".guards_type_protocol",
        "FlextUtilitiesHandler": ".handler",
        "FlextUtilitiesLogging": ".logging",
        "FlextUtilitiesLoggingConfig": ".logging_config",
        "FlextUtilitiesLoggingContext": ".logging_context",
        "FlextUtilitiesMapper": ".mapper",
        "FlextUtilitiesMapperAccess": ".mapper_access",
        "FlextUtilitiesMapperExtract": ".mapper_extract",
        "FlextUtilitiesMetadata": ".metadata",
        "FlextUtilitiesModel": ".model",
        "FlextUtilitiesModelOptions": ".model_options",
        "FlextUtilitiesModelRuntime": ".model_runtime",
        "FlextUtilitiesPackageVersion": ".package_version",
        "FlextUtilitiesParser": ".parser",
        "FlextUtilitiesParserCoerce": ".parser_coerce",
        "FlextUtilitiesParserTargets": ".parser_targets",
        "FlextUtilitiesProjectMetadata": ".project_metadata",
        "FlextUtilitiesPydantic": ".pydantic",
        "FlextUtilitiesReliability": ".reliability",
        "FlextUtilitiesSettings": ".settings",
        "FlextUtilitiesText": ".text",
        "FlextUtilitiesTypingAdapterJson": ".typeadapter_json",
        "FlextUtilitiesTypingAdapterScalars": ".typeadapter_scalars",
    },
    public_exports=__all__,
    deferred_members={
        "FlextUtilitiesArgs": (
            "parse_model",
            "resolve_options",
        ),
        "FlextUtilitiesBase": (
            "type_adapter",
            "create_instance",
        ),
        "FlextUtilitiesChecker": (
            "_is_subclass_of",
            "_is_dict_type",
            "_check_dict_compatibility",
            "_check_object_type_compatibility",
            "_extract_message_type_from_parameter",
            "_get_method_signature",
            "_get_type_hints_safe",
            "_handle_instance_check",
            "_handle_type_or_origin_check",
            "_match_type_or_origin",
            "_evaluate_type_compatibility",
            "_extract_generic_message_types",
            "_extract_message_type_from_handle",
            "can_handle_message_type",
            "compute_accepted_message_types",
            "resolve_message_route",
        ),
        "FlextUtilitiesCollection": (
            "count",
            "find",
            "process",
        ),
        "FlextUtilitiesCollectionIter": (
            "filter",
            "map",
        ),
        "FlextUtilitiesCollectionMerge": (
            "_is_json_mapping",
            "_is_json_list",
            "_merge_deep_single_key",
            "_merge_replace",
            "_merge_filter_none",
            "_merge_filter_empty",
            "_merge_append",
            "_merge_deep",
            "_MERGE_STRATEGIES",
            "merge_mappings",
        ),
        "FlextUtilitiesConfig": (
            "Yaml",
            "_EXPAND_PATTERN",
            "_expand_one",
            "_expand_str",
            "config_load",
            "config_merge",
            "config_env_override",
        ),
        "FlextUtilitiesConsole": (
            "out",
            "err",
        ),
        "FlextUtilitiesContext": (
            "StructlogProxyContextVar",
            "CORRELATION_ID",
            "PARENT_CORRELATION_ID",
            "SERVICE_NAME",
            "SERVICE_VERSION",
            "USER_ID",
            "REQUEST_ID",
            "REQUEST_TIMESTAMP",
            "OPERATION_NAME",
            "OPERATION_START_TIME",
            "OPERATION_METADATA",
            "create_str_proxy",
            "create_datetime_proxy",
            "create_dict_proxy",
            "resolve_correlation_id",
            "new_correlation",
            "apply_correlation_id",
            "ensure_correlation_id",
            "service_context",
            "resolve_operation_name",
            "apply_operation_name",
            "timed_operation",
            "export_full_context",
            "clear_context",
            "create_context_runtime",
            "resolve_context_scope_var",
            "update_context_operation",
        ),
        "FlextUtilitiesConversion": (
            "join",
            "normalize",
            "to_str",
            "to_int",
            "to_float",
            "to_bool",
            "to_positive_int",
            "to_optional_str",
        ),
        "FlextUtilitiesDiscovery": (
            "_factory_config_for",
            "scan_module",
            "service_operations",
            "_service_root",
            "_operation",
            "_validate_operation_shape",
            "_require_operation_docstring",
            "_inherited_handler_metadata",
            "_resolve_operation_request",
            "_error",
        ),
        "FlextUtilitiesDispatcherExecute": (
            "_adapt_dispatcher_output",
            "_normalize_dispatcher_output",
            "_from_result_view",
            "execute_dispatcher_handler",
        ),
        "FlextUtilitiesDomain": (
            "same_type",
            "_iterable_members_equal",
            "_plain_objects_equal",
            "compare_value_objects_by_value",
            "hash_value_object_by_value",
            "add_domain_event",
        ),
        "FlextUtilitiesEnum": (
            "_values_cache",
            "enum_values",
        ),
        "FlextUtilitiesFamilySurface": (
            "distribution_in_family_discovery",
            "_distribution_import_root",
            "family_discovery_candidates",
            "_surface_snapshot",
            "project_alias_owners",
            "_grouped_surface_entries",
            "_canonical_letters",
            "_register_alias_rename",
            "compatibility_alias_renames",
            "_lazy_suffix",
            "lazy_alias_suffixes",
            "runtime_alias_names",
            "facade_module_names",
            "family_facade",
        ),
        "FlextUtilitiesFiles": (
            "append_atomic",
            "write_atomic",
        ),
        "FlextUtilitiesGenerators": (
            "_determine_prefix",
            "_generate_id",
            "_generate_prefixed_id",
            "_build_parts_list",
            "_generate_custom_separator_id",
            "generate",
            "generate_datetime_utc",
            "generate_id",
            "generate_prefixed_id",
            "generate_iso_timestamp",
            "resolve_timezone",
            "configured_timezone",
            "now",
            "now_iso",
            "from_timestamp",
            "from_iso",
        ),
        "FlextUtilitiesGuards": (),
        "FlextUtilitiesGuardsTypeCore": (
            "_object_sequence",
            "_all_container_sequence",
            "all_container_mapping_values",
            "dict_non_empty",
            "empty_value",
            "container_value",
            "list_value",
            "mapping",
            "mapping_view",
            "subclass_view",
            "primitive",
            "scalar",
            "type_name",
            "dict_like",
            "list_like",
            "string_non_empty",
            "instance_of",
            "in_",
        ),
        "FlextUtilitiesGuardsTypeModel": (
            "has_model_dump",
            "model_type",
            "object_tuple",
            "pydantic_model",
        ),
        "FlextUtilitiesGuardsTypeProtocol": ("matches_type",),
        "FlextUtilitiesHandler": (
            "create_runtime_state",
            "start_execution",
            "record_metric",
            "push_context",
            "pop_context",
            "PerformanceTracker",
        ),
        "FlextUtilitiesLogging": (
            "_direct_logger",
            "log",
            "debug",
            "trace",
            "warning",
            "failure",
            "fetch_logger",
            "create_module_logger",
        ),
        "FlextUtilitiesLoggingConfig": (),
        "FlextUtilitiesLoggingContext": (),
        "FlextUtilitiesMapper": (
            "_transform_result",
            "_normalize_step",
            "_remap_step",
            "_strip_step",
            "_is_not_none",
            "_is_not_empty",
            "agg",
            "_deep_eq_values",
            "deep_eq",
            "prop",
            "transform",
        ),
        "FlextUtilitiesMapperAccess": (
            "_normalize_accessible_value",
            "_normalize_payload_value",
            "_container_or_string",
            "_resolve_raw_value",
            "_extract_get_value",
            "_extract_handle_array_index",
            "_normalize_array_index",
            "_get_raw",
        ),
        "FlextUtilitiesMapperExtract": (
            "_extract_fail_or_default",
            "_extract_resolve_result",
            "_extract_resolve_path_part",
            "_extract_seed_current",
            "_extract_path_parts",
            "extract",
        ),
        "FlextUtilitiesMetadata": (
            "redact_metadata",
            "_redact_mapping",
            "_redact_value",
            "_redact_text",
        ),
        "FlextUtilitiesModel": (
            "dump",
            "settings_base",
            "validate_value",
        ),
        "FlextUtilitiesModelOptions": ("resolve_runtime_options",),
        "FlextUtilitiesModelRuntime": (
            "Project",
            "PyprojectDocument",
            "Value",
            "registry_same_value",
            "registry_plugin_key",
            "registry_registration_name",
            "handler_metadata",
            "registry_registration_details",
            "resolve_runtime_settings",
        ),
        "FlextUtilitiesParser": (
            "parse",
            "_dispatch",
        ),
        "FlextUtilitiesParserCoerce": (
            "_CASE_OPS",
            "_parse_normalize_str",
            "_coerce_to_bool",
            "_coerce_to_float",
            "_coerce_to_int",
            "_parse_with_default",
            "norm_str",
        ),
        "FlextUtilitiesParserTargets": (
            "_resolve_opts",
            "_parse_try_direct",
            "_parse_try_enum",
            "_parse_try_model",
            "_parse_try_primitive",
        ),
        "FlextUtilitiesProjectMetadata": (
            "_DISTRIBUTION_SEPARATOR_RE",
            "_REQUIREMENT_NAME_RE",
            "_normalize_distribution_name",
            "read_project_document_cached",
            "build_project_metadata",
            "derive_class_stem",
            "installed_distributions",
            "distribution_requirement_names",
            "project_uses_distribution",
            "package_author",
            "package_homepage",
            "package_version_info",
            "PackageVersion",
        ),
        "FlextUtilitiesPydantic": (
            "validate_call",
            "with_config",
            "from_json",
            "to_json",
            "to_jsonable_python",
            "normalize_to_container",
            "_json_data",
            "normalize_to_json_value",
            "normalize_to_json_mapping",
            "normalize_model_input_mapping",
            "normalize_metadata_input_mapping",
            "validate_metadata_model_input",
            "_empty_collection_seed",
            "empty",
        ),
        "FlextUtilitiesReliability": (
            "_resolve_handled",
            "try_",
            "guard_result",
            "retry",
        ),
        "FlextUtilitiesSettings": ("resolve_process_environment",),
        "FlextUtilitiesText": (
            "normalize_alnum",
            "format_app_id",
            "safe_string",
            "write_file",
        ),
        "FlextUtilitiesTypingAdapterJson": (
            "metadata_map_adapter",
            "json_value_adapter",
            "json_mapping_adapter",
            "strict_json_mapping_adapter",
            "json_dict_adapter",
            "json_dict_sequence_adapter",
            "json_mapping_sequence_adapter",
            "json_mapping_by_str_adapter",
            "json_list_adapter",
            "strict_json_list_adapter",
            "primitives_adapter",
            "container_set_adapter",
            "string_set_adapter",
            "scalar_set_adapter",
            "sortable_dict_adapter",
        ),
        "FlextUtilitiesTypingAdapterScalars": (
            "bool_adapter",
            "int_adapter",
            "scalar_adapter",
            "scalar_mapping_adapter",
            "float_adapter",
            "str_adapter",
            "binary_content_adapter",
            "str_mapping_adapter",
            "header_mapping_adapter",
            "str_dict_adapter",
            "int_dict_adapter",
            "hostname_str_adapter",
            "port_number_adapter",
            "str_sequence_adapter",
            "strict_str_sequence_adapter",
            "strict_str_tuple_adapter",
            "str_or_bytes_adapter",
            "enum_type_adapter",
            "primitive_metadata_mapping_adapter",
        ),
    },
)

defer_lazy_bases(
    __name__,
    {
        "FlextUtilitiesCollection": (
            (".", "FlextUtilitiesCollectionIter"),
            (".", "FlextUtilitiesCollectionMerge"),
        ),
        "FlextUtilitiesCollectionIter": ((".", "FlextUtilitiesCollectionMerge"),),
        "FlextUtilitiesGuards": (
            ("flext_core._utilities._guards", "FlextUtilitiesGuardsTypeProtocolSpecs"),
            ("flext_core._utilities._guards", "FlextUtilitiesGuardsTypeProtocolString"),
        ),
        "FlextUtilitiesGuardsTypeProtocol": (
            ("flext_core._utilities._guards", "FlextUtilitiesGuardsTypeProtocolSpecs"),
            ("flext_core._utilities._guards", "FlextUtilitiesGuardsTypeProtocolString"),
        ),
        "FlextUtilitiesLogging": (
            (".", "FlextUtilitiesLoggingContext"),
            (".", "FlextUtilitiesLoggingConfig"),
        ),
        "FlextUtilitiesLoggingConfig": (
            ("flext_core._utilities._logging", "FlextUtilitiesLoggingComposition"),
        ),
        "FlextUtilitiesLoggingContext": (
            ("flext_core._utilities._logging", "FlextUtilitiesLoggingContextBinding"),
        ),
        "FlextUtilitiesMapper": ((".", "FlextUtilitiesMapperExtract"),),
        "FlextUtilitiesMapperExtract": ((".", "FlextUtilitiesMapperAccess"),),
        "FlextUtilitiesModelRuntime": ((".", "FlextUtilitiesModelOptions"),),
        "FlextUtilitiesParser": ((".", "FlextUtilitiesParserTargets"),),
        "FlextUtilitiesParserTargets": ((".", "FlextUtilitiesParserCoerce"),),
        "FlextUtilitiesPydantic": (
            ("flext_core._models", "FlextModelsPydantic"),
            (".", "FlextUtilitiesTypingAdapterJson"),
            (".", "FlextUtilitiesTypingAdapterScalars"),
        ),
        "FlextUtilitiesTypingAdapterJson": ((".", "FlextUtilitiesBase"),),
        "FlextUtilitiesTypingAdapterScalars": ((".", "FlextUtilitiesBase"),),
    },
)
