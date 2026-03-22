# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Unit package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core import FlextTypes

    from . import contracts as contracts, flext_tests as flext_tests
    from .conftest_infra import (
        infra_git,
        infra_git_repo,
        infra_io,
        infra_path,
        infra_patterns,
        infra_pr_manager,
        infra_pr_workspace_manager,
        infra_reporting,
        infra_safe_command_output,
        infra_selection,
        infra_subprocess,
        infra_templates,
        infra_test_workspace,
        infra_toml,
        infra_workflow_linter,
        infra_workflow_syncer,
    )
    from .contracts.text_contract import TextUtilityContract
    from .flext_tests.test_docker import TestDocker
    from .flext_tests.test_domains import TestFlextTestsDomains
    from .flext_tests.test_files import TestFlextTestsFiles
    from .flext_tests.test_matchers import TestFlextTestsMatchers
    from .flext_tests.test_utilities import TestUtilities
    from .protocols import FlextProtocols, p
    from .test_args_coverage_100 import TestFlextUtilitiesArgs
    from .test_automated_architecture import TestAutomatedArchitecture
    from .test_automated_container import TestAutomatedFlextContainer
    from .test_automated_context import TestAutomatedFlextContext
    from .test_automated_decorators import TestAutomatedFlextDecorators
    from .test_automated_dispatcher import TestAutomatedFlextDispatcher
    from .test_automated_exceptions import EXCEPTION_CLASSES, TestAutomatedExceptions
    from .test_automated_handlers import TestAutomatedFlextHandlers
    from .test_automated_loggings import TestAutomatedFlextLogger
    from .test_automated_mixins import TestAutomatedFlextMixins
    from .test_automated_registry import TestAutomatedFlextRegistry
    from .test_automated_result import TestAutomatedResult
    from .test_automated_runtime import TestAutomatedFlextRuntime
    from .test_automated_service import TestAutomatedFlextService
    from .test_automated_settings import TestAutomatedFlextSettings
    from .test_automated_utilities import TestAutomatedFlextUtilities
    from .test_collection_utilities_coverage_100 import TestCollectionUtilitiesCoverage
    from .test_collections_coverage_100 import TestFlextModelsCollectionsCoverage100
    from .test_config import TestFlextSettings
    from .test_constants import TestConstants
    from .test_constants_full_coverage import (
        test_constants_auto_enum_and_bimapping_paths,
    )
    from .test_container import TestFlextContainer
    from .test_container_full_coverage import TestContainerFullCoverage
    from .test_context import TestFlextContext
    from .test_context_coverage_100 import TestContext100Coverage
    from .test_context_full_coverage import (
        test_clear_keys_values_items_and_validate_branches,
        test_container_and_service_domain_paths,
        test_create_merges_metadata_dict_branch,
        test_create_overloads_and_auto_correlation,
        test_export_paths_with_metadata_and_statistics,
        test_inactive_and_none_value_paths,
        test_narrow_contextvar_exception_branch,
        test_narrow_contextvar_invalid_inputs,
        test_set_set_all_get_validation_and_error_paths,
        test_update_statistics_remove_hook_and_clone_false_result,
    )
    from .test_coverage_76_lines import TestCoverage76Lines
    from .test_coverage_context import TestCoverageContext
    from .test_coverage_exceptions import TestCoverageExceptions
    from .test_coverage_loggings import TestCoverageLoggings
    from .test_coverage_models import TestCoverageModels
    from .test_coverage_utilities import Testu
    from .test_decorators import TestFlextDecorators
    from .test_decorators_discovery_full_coverage import (
        TestDecoratorsDiscoveryFullCoverage,
    )
    from .test_decorators_full_coverage import TestDecoratorsFullCoverage
    from .test_deprecation_warnings import TestDeprecationWarnings
    from .test_di_incremental import Provide, TestDIIncremental, inject
    from .test_di_services_access import TestDiServicesAccess
    from .test_dispatcher_di import TestDispatcherDI
    from .test_dispatcher_full_coverage import TestDispatcherFullCoverage
    from .test_dispatcher_minimal import TestDispatcherMinimal
    from .test_dispatcher_reliability import (
        CircuitBreakerManager,
        RateLimiterManager,
        RetryPolicy,
        test_circuit_breaker_transitions_and_metrics,
        test_rate_limiter_blocks_then_recovers,
        test_rate_limiter_jitter_application,
        test_retry_policy_behavior,
    )
    from .test_dispatcher_reliability_full_coverage import (
        test_dispatcher_reliability_branch_paths,
    )
    from .test_dispatcher_timeout_coverage_100 import (
        TestDispatcherTimeoutCoverage100,
        TimeoutEnforcer,
    )
    from .test_entity_coverage import TestEntityCoverageEdgeCases
    from .test_enum_utilities_coverage_100 import TestEnumUtilitiesCoverage
    from .test_exceptions import Teste
    from .test_exceptions_full_coverage import (
        test_authentication_error_normalizes_extra_kwargs_into_context,
        test_base_error_normalize_metadata_merges_existing_metadata_model,
        test_exceptions_uncovered_metadata_paths,
        test_merge_metadata_context_paths,
        test_not_found_error_correlation_id_selection_and_extra_kwargs,
    )
    from .test_final_75_percent_push import TestFinal75PercentPush
    from .test_handler_decorator_discovery import TestHandlerDecoratorDiscovery
    from .test_handlers import TestFlextHandlers
    from .test_handlers_full_coverage import TestHandlersFullCoverage, handlers_module
    from .test_loggings_error_paths_coverage import TestLoggingsErrorPaths
    from .test_loggings_full_coverage import TestModule
    from .test_loggings_strict_returns import TestLoggingsStrictReturns
    from .test_mixins import TestFlextMixinsNestedClasses
    from .test_mixins_full_coverage import TestMixinsFullCoverage
    from .test_models import TestModels
    from .test_models_79_coverage import TestModels79Coverage
    from .test_models_base_full_coverage import TestModelsBaseFullCoverage
    from .test_models_collections_full_coverage import TestModelsCollectionsFullCoverage
    from .test_models_container import TestFlextModelsContainer
    from .test_models_container_full_coverage import (
        test_container_resource_registration_metadata_normalized,
    )
    from .test_models_context_full_coverage import (
        test_context_data_metadata_normalizer_removed,
        test_context_data_normalize_and_json_checks,
        test_context_data_validate_dict_serializable_error_paths,
        test_context_data_validate_dict_serializable_none_and_mapping,
        test_context_data_validate_dict_serializable_real_dicts,
        test_context_export_serializable_and_validators,
        test_context_export_statistics_validator_and_computed_fields,
        test_context_export_validate_dict_serializable_mapping_and_models,
        test_context_export_validate_dict_serializable_valid,
        test_scope_data_validators_and_errors,
        test_statistics_and_custom_fields_validators,
        test_structlog_proxy_context_var_default_when_key_missing,
        test_structlog_proxy_context_var_get_set_reset_paths,
        test_to_general_value_dict_removed,
    )
    from .test_models_cqrs_full_coverage import (
        test_command_pagination_limit,
        test_cqrs_query_resolve_deeper_and_int_pagination,
        test_flext_message_type_alias_adapter,
        test_handler_builder_fluent_methods,
        test_query_resolve_pagination_wrapper_and_fallback,
        test_query_validate_pagination_dict_and_default,
    )
    from .test_models_entity_full_coverage import (
        test_entity_comparable_map_and_bulk_validation_paths,
    )
    from .test_models_generic_full_coverage import (
        test_canonical_aliases_are_available,
        test_conversion_add_converted_and_error_metadata_append_paths,
        test_conversion_add_skipped_skip_reason_upsert_paths,
        test_conversion_add_warning_metadata_append_paths,
        test_conversion_start_and_complete_methods,
        test_operation_progress_start_operation_sets_runtime_fields,
    )
    from .test_models_handler_full_coverage import (
        test_models_handler_branches,
        test_models_handler_uncovered_mode_and_reset_paths,
    )
    from .test_models_service_full_coverage import (
        test_service_request_timeout_post_validator_messages,
        test_service_request_timeout_validator_branches,
    )
    from .test_models_settings_full_coverage import (
        test_models_settings_branch_paths,
        test_models_settings_context_validator_and_non_standard_status_input,
    )
    from .test_models_validation_full_coverage import (
        test_basic_imports_work,
        test_ensure_utc_datetime_adds_tzinfo_when_naive,
        test_ensure_utc_datetime_preserves_aware,
        test_ensure_utc_datetime_returns_none_on_none,
        test_facade_binding_is_correct,
        test_normalize_to_list_passes_list_through,
        test_normalize_to_list_wraps_int,
        test_normalize_to_list_wraps_scalar,
        test_strip_whitespace_preserves_clean,
        test_strip_whitespace_returns_empty_on_spaces,
        test_strip_whitespace_trims_leading_trailing,
        test_validate_config_dict_normalizes_dict,
        test_validate_tags_list_from_string,
        test_validate_tags_list_normalizes,
    )
    from .test_namespace_validator import TestFlextInfraNamespaceValidator
    from .test_pagination_coverage_100 import TestPaginationCoverage100
    from .test_phase2_coverage_final import TestPhase2CoverageFinal
    from .test_protocols import TestFlextProtocols
    from .test_refactor_cli_models_workflow import (
        test_centralize_pydantic_cli_outputs_extended_metrics,
        test_namespace_enforce_cli_fails_on_manual_protocol_violation,
        test_ultrawork_models_cli_runs_dry_run_copy,
    )
    from .test_refactor_migrate_to_class_mro import (
        test_discover_project_roots_without_nested_git_dirs,
        test_migrate_protocols_rewrites_references_with_p_alias,
        test_migrate_to_mro_inlines_alias_constant_into_constants_class,
        test_migrate_to_mro_moves_constant_and_rewrites_reference,
        test_migrate_to_mro_moves_manual_uppercase_assignment,
        test_migrate_to_mro_normalizes_facade_alias_to_c,
        test_migrate_to_mro_rejects_unknown_target,
        test_migrate_typings_rewrites_references_with_t_alias,
        test_refactor_utilities_iter_python_files_includes_examples_and_scripts,
    )
    from .test_refactor_namespace_enforcer import (
        test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring,
        test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future,
        test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file,
        test_namespace_enforcer_creates_missing_facades_and_rewrites_imports,
        test_namespace_enforcer_detects_cyclic_imports_in_tests_directory,
        test_namespace_enforcer_detects_internal_private_imports,
        test_namespace_enforcer_detects_manual_protocol_outside_canonical_files,
        test_namespace_enforcer_detects_manual_typings_and_compat_aliases,
        test_namespace_enforcer_detects_missing_runtime_alias_outside_src,
        test_namespace_enforcer_does_not_rewrite_indented_import_aliases,
        test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks,
    )
    from .test_refactor_policy_family_rules import (
        test_decorators_family_blocks_dispatcher_target,
        test_dispatcher_family_blocks_models_target,
        test_helper_consolidation_is_prechecked,
        test_models_family_blocks_utilities_target,
        test_runtime_family_blocks_non_runtime_target,
        test_utilities_family_allows_utilities_target,
    )
    from .test_registry import TestFlextRegistry
    from .test_registry_full_coverage import (
        test_create_auto_discover_and_mode_mapping,
        test_execute_and_register_handler_failure_paths,
        test_get_plugin_and_register_metadata_and_list_items_exception,
        test_summary_error_paths_and_bindings_failures,
        test_summary_properties_and_subclass_storage_reset,
    )
    from .test_result import Testr
    from .test_result_additional import (
        test_create_from_callable_and_repr,
        test_flow_through_short_circuits_on_failure,
        test_map_error_identity_and_transform,
        test_ok_accepts_none,
        test_with_resource_cleanup_runs,
    )
    from .test_result_coverage_100 import TestrCoverage
    from .test_result_exception_carrying import TestResultExceptionCarrying
    from .test_result_full_coverage import (
        test_from_validation_and_to_model_paths,
        test_init_fallback_and_lazy_returns_result_property,
        test_lash_runtime_result_paths,
        test_map_flat_map_and_then_paths,
        test_recover_tap_and_tap_error_paths,
        test_type_guards_result,
        test_validation_like_error_structure,
    )
    from .test_runtime import TestFlextRuntime
    from .test_runtime_coverage_100 import TestRuntimeCoverage100
    from .test_runtime_full_coverage import (
        reset_runtime_state,
        runtime_cov_tests,
        runtime_tests,
        test_async_log_writer_paths,
        test_async_log_writer_shutdown_with_full_queue,
        test_config_bridge_and_trace_context_and_http_validation,
        test_configure_structlog_edge_paths,
        test_configure_structlog_print_logger_factory_fallback,
        test_dependency_integration_and_wiring_paths,
        test_dependency_registration_duplicate_guards,
        test_ensure_trace_context_dict_conversion_paths,
        test_get_logger_none_name_paths,
        test_model_helpers_remaining_paths,
        test_model_support_and_hash_compare_paths,
        test_normalization_edge_branches,
        test_normalize_to_container_alias_removal_path,
        test_normalize_to_metadata_alias_removal_path,
        test_reconfigure_and_reset_state_paths,
        test_reuse_existing_runtime_coverage_branches,
        test_reuse_existing_runtime_scenarios,
        test_runtime_create_instance_failure_branch,
        test_runtime_integration_tracking_paths,
        test_runtime_misc_remaining_paths,
        test_runtime_module_accessors_and_metadata,
        test_runtime_result_alias_compatibility,
        test_runtime_result_all_missed_branches,
        test_runtime_result_remaining_paths,
    )
    from .test_service import TestsCore
    from .test_service_additional import (
        RuntimeCloneService,
        test_get_service_info,
        test_is_valid_handles_validation_exception,
        test_result_property_raises_on_failure,
    )
    from .test_service_bootstrap import TestServiceBootstrap
    from .test_service_coverage_100 import TestService100Coverage
    from .test_service_full_coverage import TestServiceFullCoverage
    from .test_settings_full_coverage import (
        test_settings_materialize_and_context_overrides,
    )
    from .test_transformer_class_nesting import (
        test_class_nesting_appends_to_existing_namespace_and_removes_pass,
        test_class_nesting_keeps_unmapped_top_level_classes,
        test_class_nesting_moves_top_level_class_into_new_namespace,
    )
    from .test_transformer_helper_consolidation import (
        TestHelperConsolidationTransformer,
    )
    from .test_transformer_nested_class_propagation import (
        NestedClassPropagationTransformer,
        test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage,
        test_nested_class_propagation_updates_import_annotations_and_calls,
    )
    from .test_typings import TestTypings
    from .test_typings_full_coverage import TestTypingsFullCoverage
    from .test_utilities_args_full_coverage import (
        UnknownHint,
        test_args_get_enum_params_annotated_unwrap_branch,
        test_args_get_enum_params_branches,
    )
    from .test_utilities_cache_coverage_100 import (
        NORMALIZE_COMPONENT_SCENARIOS,
        SORT_KEY_SCENARIOS,
        ClearCacheScenario,
        NormalizeComponentScenario,
        SortKeyScenario,
        TestuCacheClearObjectCache,
        TestuCacheGenerateCacheKey,
        TestuCacheHasCacheAttributes,
        TestuCacheLogger,
        TestuCacheNormalizeComponent,
        TestuCacheSortDictKeys,
        TestuCacheSortKey,
        UtilitiesCacheCoverage100Namespace,
    )
    from .test_utilities_checker_full_coverage import TestUtilitiesCheckerFullCoverage
    from .test_utilities_collection_coverage_100 import TestUtilitiesCollectionCoverage
    from .test_utilities_collection_full_coverage import (
        TestUtilitiesCollectionFullCoverage,
    )
    from .test_utilities_configuration_coverage_100 import (
        TestFlextUtilitiesConfiguration,
    )
    from .test_utilities_configuration_full_coverage import (
        TestUtilitiesConfigurationFullCoverage,
    )
    from .test_utilities_context_full_coverage import TestUtilitiesContextFullCoverage
    from .test_utilities_conversion_full_coverage import (
        test_conversion_string_and_join_paths,
    )
    from .test_utilities_coverage import TestUtilitiesCoverage
    from .test_utilities_data_mapper import TestUtilitiesDataMapper
    from .test_utilities_deprecation_full_coverage import (
        test_deprecated_class_noop_init_branch,
    )
    from .test_utilities_domain import (
        TestuDomain,
        create_compare_entities_cases,
        create_compare_value_objects_cases,
        create_hash_entity_cases,
        create_hash_value_object_cases,
        create_validate_entity_has_id_cases,
        create_validate_value_object_immutable_cases,
    )
    from .test_utilities_domain_full_coverage import TestUtilitiesDomainFullCoverage
    from .test_utilities_enum_full_coverage import TestUtilitiesEnumFullCoverage
    from .test_utilities_generators_full_coverage import (
        TestUtilitiesGeneratorsFullCoverage,
        generators_module,
        runtime_module,
    )
    from .test_utilities_guards_full_coverage import (
        test_aliases_are_available,
        test_chk_exercises_missed_branches,
        test_configuration_mapping_and_dict_negative_branches,
        test_extract_mapping_or_none_branches,
        test_guard_in_has_empty_none_helpers,
        test_guard_instance_attribute_access_warnings,
        test_guards_bool_identity_branch_via_isinstance_fallback,
        test_guards_bool_shortcut_and_issubclass_typeerror,
        test_guards_handler_type_issubclass_typeerror_branch_direct,
        test_guards_issubclass_success_when_callable_is_patched,
        test_guards_issubclass_typeerror_when_class_not_treated_as_callable,
        test_is_container_negative_paths_and_callable,
        test_is_flexible_value_covers_all_branches,
        test_is_handler_type_branches,
        test_is_type_non_empty_unknown_and_tuple_and_fallback,
        test_is_type_protocol_fallback_branches,
        test_non_empty_and_normalize_branches,
        test_protocol_and_simple_guard_helpers,
    )
    from .test_utilities_mapper_coverage_100 import (
        SimpleObj,
        TestuMapperAccessors,
        TestuMapperAdvanced,
        TestuMapperBuild,
        TestuMapperConversions,
        TestuMapperExtract,
        TestuMapperUtils,
        UtilitiesMapperCoverage100Namespace,
    )
    from .test_utilities_mapper_full_coverage import (
        AttrObject,
        BadBool,
        BadMapping,
        BadString,
        ExplodingLenList,
        UtilitiesMapperFullCoverageNamespace,
        mapper,
        test_accessor_take_pick_as_or_flat_and_agg_branches,
        test_at_take_and_as_branches,
        test_bad_string_and_bad_bool_raise_value_error,
        test_build_apply_transform_and_process_error_paths,
        test_construct_transform_and_deep_eq_branches,
        test_conversion_and_extract_success_branches,
        test_convert_default_fallback_matrix,
        test_convert_sequence_branch_returns_tuple,
        test_ensure_and_extract_array_index_helpers,
        test_extract_error_paths_and_prop_accessor,
        test_extract_field_value_and_ensure_variants,
        test_field_and_fields_multi_branches,
        test_filter_map_normalize_convert_helpers,
        test_general_value_helpers_and_logger,
        test_group_sort_unique_slice_chunk_branches,
        test_invert_and_json_conversion_branches,
        test_map_flags_collect_and_invert_branches,
        test_narrow_to_string_keyed_dict_and_mapping_paths,
        test_process_context_data_and_related_convenience,
        test_remaining_build_fields_construct_and_eq_paths,
        test_remaining_uncovered_branches,
        test_small_mapper_convenience_methods,
        test_transform_option_extract_and_step_helpers,
        test_type_guards_and_narrowing_failures,
    )
    from .test_utilities_model_full_coverage import (
        test_merge_defaults_and_dump_paths,
        test_normalize_to_pydantic_dict_and_value_branches,
        test_update_exception_path,
        test_update_success_path_returns_ok_result,
    )
    from .test_utilities_pagination_full_coverage import (
        test_pagination_response_string_fallbacks,
    )
    from .test_utilities_parser_full_coverage import TestUtilitiesParserFullCoverage
    from .test_utilities_reliability import TestFlextUtilitiesReliability
    from .test_utilities_reliability_full_coverage import (
        test_utilities_reliability_branches,
        test_utilities_reliability_compose_returns_non_result_directly,
        test_utilities_reliability_uncovered_retry_compose_and_sequence_paths,
    )
    from .test_utilities_string_parser import (
        TestuStringParser,
        normalized_value_key_cases,
    )
    from .test_utilities_text_full_coverage import TestUtilitiesTextFullCoverage
    from .test_utilities_type_checker_coverage_100 import (
        T,
        TestuTypeChecker,
        TMessage,
        pytestmark,
    )
    from .test_utilities_type_guards_coverage_100 import (
        TestUtilitiesTypeGuardsCoverage100,
    )
    from .test_version import TestFlextVersion

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AttrObject": ("tests.unit.test_utilities_mapper_full_coverage", "AttrObject"),
    "BadBool": ("tests.unit.test_utilities_mapper_full_coverage", "BadBool"),
    "BadMapping": ("tests.unit.test_utilities_mapper_full_coverage", "BadMapping"),
    "BadString": ("tests.unit.test_utilities_mapper_full_coverage", "BadString"),
    "CircuitBreakerManager": (
        "tests.unit.test_dispatcher_reliability",
        "CircuitBreakerManager",
    ),
    "ClearCacheScenario": (
        "tests.unit.test_utilities_cache_coverage_100",
        "ClearCacheScenario",
    ),
    "EXCEPTION_CLASSES": ("tests.unit.test_automated_exceptions", "EXCEPTION_CLASSES"),
    "ExplodingLenList": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "ExplodingLenList",
    ),
    "FlextProtocols": ("tests.unit.protocols", "FlextProtocols"),
    "NORMALIZE_COMPONENT_SCENARIOS": (
        "tests.unit.test_utilities_cache_coverage_100",
        "NORMALIZE_COMPONENT_SCENARIOS",
    ),
    "NestedClassPropagationTransformer": (
        "tests.unit.test_transformer_nested_class_propagation",
        "NestedClassPropagationTransformer",
    ),
    "NormalizeComponentScenario": (
        "tests.unit.test_utilities_cache_coverage_100",
        "NormalizeComponentScenario",
    ),
    "Provide": ("tests.unit.test_di_incremental", "Provide"),
    "RateLimiterManager": (
        "tests.unit.test_dispatcher_reliability",
        "RateLimiterManager",
    ),
    "RetryPolicy": ("tests.unit.test_dispatcher_reliability", "RetryPolicy"),
    "RuntimeCloneService": (
        "tests.unit.test_service_additional",
        "RuntimeCloneService",
    ),
    "SORT_KEY_SCENARIOS": (
        "tests.unit.test_utilities_cache_coverage_100",
        "SORT_KEY_SCENARIOS",
    ),
    "SimpleObj": ("tests.unit.test_utilities_mapper_coverage_100", "SimpleObj"),
    "SortKeyScenario": (
        "tests.unit.test_utilities_cache_coverage_100",
        "SortKeyScenario",
    ),
    "T": ("tests.unit.test_utilities_type_checker_coverage_100", "T"),
    "TMessage": ("tests.unit.test_utilities_type_checker_coverage_100", "TMessage"),
    "TestAutomatedArchitecture": (
        "tests.unit.test_automated_architecture",
        "TestAutomatedArchitecture",
    ),
    "TestAutomatedExceptions": (
        "tests.unit.test_automated_exceptions",
        "TestAutomatedExceptions",
    ),
    "TestAutomatedFlextContainer": (
        "tests.unit.test_automated_container",
        "TestAutomatedFlextContainer",
    ),
    "TestAutomatedFlextContext": (
        "tests.unit.test_automated_context",
        "TestAutomatedFlextContext",
    ),
    "TestAutomatedFlextDecorators": (
        "tests.unit.test_automated_decorators",
        "TestAutomatedFlextDecorators",
    ),
    "TestAutomatedFlextDispatcher": (
        "tests.unit.test_automated_dispatcher",
        "TestAutomatedFlextDispatcher",
    ),
    "TestAutomatedFlextHandlers": (
        "tests.unit.test_automated_handlers",
        "TestAutomatedFlextHandlers",
    ),
    "TestAutomatedFlextLogger": (
        "tests.unit.test_automated_loggings",
        "TestAutomatedFlextLogger",
    ),
    "TestAutomatedFlextMixins": (
        "tests.unit.test_automated_mixins",
        "TestAutomatedFlextMixins",
    ),
    "TestAutomatedFlextRegistry": (
        "tests.unit.test_automated_registry",
        "TestAutomatedFlextRegistry",
    ),
    "TestAutomatedFlextRuntime": (
        "tests.unit.test_automated_runtime",
        "TestAutomatedFlextRuntime",
    ),
    "TestAutomatedFlextService": (
        "tests.unit.test_automated_service",
        "TestAutomatedFlextService",
    ),
    "TestAutomatedFlextSettings": (
        "tests.unit.test_automated_settings",
        "TestAutomatedFlextSettings",
    ),
    "TestAutomatedFlextUtilities": (
        "tests.unit.test_automated_utilities",
        "TestAutomatedFlextUtilities",
    ),
    "TestAutomatedResult": ("tests.unit.test_automated_result", "TestAutomatedResult"),
    "TestCollectionUtilitiesCoverage": (
        "tests.unit.test_collection_utilities_coverage_100",
        "TestCollectionUtilitiesCoverage",
    ),
    "TestConstants": ("tests.unit.test_constants", "TestConstants"),
    "TestContainerFullCoverage": (
        "tests.unit.test_container_full_coverage",
        "TestContainerFullCoverage",
    ),
    "TestContext100Coverage": (
        "tests.unit.test_context_coverage_100",
        "TestContext100Coverage",
    ),
    "TestCoverage76Lines": ("tests.unit.test_coverage_76_lines", "TestCoverage76Lines"),
    "TestCoverageContext": ("tests.unit.test_coverage_context", "TestCoverageContext"),
    "TestCoverageExceptions": (
        "tests.unit.test_coverage_exceptions",
        "TestCoverageExceptions",
    ),
    "TestCoverageLoggings": (
        "tests.unit.test_coverage_loggings",
        "TestCoverageLoggings",
    ),
    "TestCoverageModels": ("tests.unit.test_coverage_models", "TestCoverageModels"),
    "TestDIIncremental": ("tests.unit.test_di_incremental", "TestDIIncremental"),
    "TestDecoratorsDiscoveryFullCoverage": (
        "tests.unit.test_decorators_discovery_full_coverage",
        "TestDecoratorsDiscoveryFullCoverage",
    ),
    "TestDecoratorsFullCoverage": (
        "tests.unit.test_decorators_full_coverage",
        "TestDecoratorsFullCoverage",
    ),
    "TestDeprecationWarnings": (
        "tests.unit.test_deprecation_warnings",
        "TestDeprecationWarnings",
    ),
    "TestDiServicesAccess": (
        "tests.unit.test_di_services_access",
        "TestDiServicesAccess",
    ),
    "TestDispatcherDI": ("tests.unit.test_dispatcher_di", "TestDispatcherDI"),
    "TestDispatcherFullCoverage": (
        "tests.unit.test_dispatcher_full_coverage",
        "TestDispatcherFullCoverage",
    ),
    "TestDispatcherMinimal": (
        "tests.unit.test_dispatcher_minimal",
        "TestDispatcherMinimal",
    ),
    "TestDispatcherTimeoutCoverage100": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TestDispatcherTimeoutCoverage100",
    ),
    "TestDocker": ("tests.unit.flext_tests.test_docker", "TestDocker"),
    "TestEntityCoverageEdgeCases": (
        "tests.unit.test_entity_coverage",
        "TestEntityCoverageEdgeCases",
    ),
    "TestEnumUtilitiesCoverage": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestEnumUtilitiesCoverage",
    ),
    "TestFinal75PercentPush": (
        "tests.unit.test_final_75_percent_push",
        "TestFinal75PercentPush",
    ),
    "TestFlextContainer": ("tests.unit.test_container", "TestFlextContainer"),
    "TestFlextContext": ("tests.unit.test_context", "TestFlextContext"),
    "TestFlextDecorators": ("tests.unit.test_decorators", "TestFlextDecorators"),
    "TestFlextHandlers": ("tests.unit.test_handlers", "TestFlextHandlers"),
    "TestFlextInfraNamespaceValidator": (
        "tests.unit.test_namespace_validator",
        "TestFlextInfraNamespaceValidator",
    ),
    "TestFlextMixinsNestedClasses": (
        "tests.unit.test_mixins",
        "TestFlextMixinsNestedClasses",
    ),
    "TestFlextModelsCollectionsCoverage100": (
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsCoverage100",
    ),
    "TestFlextModelsContainer": (
        "tests.unit.test_models_container",
        "TestFlextModelsContainer",
    ),
    "TestFlextProtocols": ("tests.unit.test_protocols", "TestFlextProtocols"),
    "TestFlextRegistry": ("tests.unit.test_registry", "TestFlextRegistry"),
    "TestFlextRuntime": ("tests.unit.test_runtime", "TestFlextRuntime"),
    "TestFlextSettings": ("tests.unit.test_config", "TestFlextSettings"),
    "TestFlextTestsDomains": (
        "tests.unit.flext_tests.test_domains",
        "TestFlextTestsDomains",
    ),
    "TestFlextTestsFiles": ("tests.unit.flext_tests.test_files", "TestFlextTestsFiles"),
    "TestFlextTestsMatchers": (
        "tests.unit.flext_tests.test_matchers",
        "TestFlextTestsMatchers",
    ),
    "TestFlextUtilitiesArgs": (
        "tests.unit.test_args_coverage_100",
        "TestFlextUtilitiesArgs",
    ),
    "TestFlextUtilitiesConfiguration": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "TestFlextUtilitiesConfiguration",
    ),
    "TestFlextUtilitiesReliability": (
        "tests.unit.test_utilities_reliability",
        "TestFlextUtilitiesReliability",
    ),
    "TestFlextVersion": ("tests.unit.test_version", "TestFlextVersion"),
    "TestHandlerDecoratorDiscovery": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDecoratorDiscovery",
    ),
    "TestHandlersFullCoverage": (
        "tests.unit.test_handlers_full_coverage",
        "TestHandlersFullCoverage",
    ),
    "TestHelperConsolidationTransformer": (
        "tests.unit.test_transformer_helper_consolidation",
        "TestHelperConsolidationTransformer",
    ),
    "TestLoggingsErrorPaths": (
        "tests.unit.test_loggings_error_paths_coverage",
        "TestLoggingsErrorPaths",
    ),
    "TestLoggingsStrictReturns": (
        "tests.unit.test_loggings_strict_returns",
        "TestLoggingsStrictReturns",
    ),
    "TestMixinsFullCoverage": (
        "tests.unit.test_mixins_full_coverage",
        "TestMixinsFullCoverage",
    ),
    "TestModels": ("tests.unit.test_models", "TestModels"),
    "TestModels79Coverage": (
        "tests.unit.test_models_79_coverage",
        "TestModels79Coverage",
    ),
    "TestModelsBaseFullCoverage": (
        "tests.unit.test_models_base_full_coverage",
        "TestModelsBaseFullCoverage",
    ),
    "TestModelsCollectionsFullCoverage": (
        "tests.unit.test_models_collections_full_coverage",
        "TestModelsCollectionsFullCoverage",
    ),
    "TestModule": ("tests.unit.test_loggings_full_coverage", "TestModule"),
    "TestPaginationCoverage100": (
        "tests.unit.test_pagination_coverage_100",
        "TestPaginationCoverage100",
    ),
    "TestPhase2CoverageFinal": (
        "tests.unit.test_phase2_coverage_final",
        "TestPhase2CoverageFinal",
    ),
    "TestResultExceptionCarrying": (
        "tests.unit.test_result_exception_carrying",
        "TestResultExceptionCarrying",
    ),
    "TestRuntimeCoverage100": (
        "tests.unit.test_runtime_coverage_100",
        "TestRuntimeCoverage100",
    ),
    "TestService100Coverage": (
        "tests.unit.test_service_coverage_100",
        "TestService100Coverage",
    ),
    "TestServiceBootstrap": (
        "tests.unit.test_service_bootstrap",
        "TestServiceBootstrap",
    ),
    "TestServiceFullCoverage": (
        "tests.unit.test_service_full_coverage",
        "TestServiceFullCoverage",
    ),
    "TestTypings": ("tests.unit.test_typings", "TestTypings"),
    "TestTypingsFullCoverage": (
        "tests.unit.test_typings_full_coverage",
        "TestTypingsFullCoverage",
    ),
    "TestUtilities": ("tests.unit.flext_tests.test_utilities", "TestUtilities"),
    "TestUtilitiesCheckerFullCoverage": (
        "tests.unit.test_utilities_checker_full_coverage",
        "TestUtilitiesCheckerFullCoverage",
    ),
    "TestUtilitiesCollectionCoverage": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestUtilitiesCollectionCoverage",
    ),
    "TestUtilitiesCollectionFullCoverage": (
        "tests.unit.test_utilities_collection_full_coverage",
        "TestUtilitiesCollectionFullCoverage",
    ),
    "TestUtilitiesConfigurationFullCoverage": (
        "tests.unit.test_utilities_configuration_full_coverage",
        "TestUtilitiesConfigurationFullCoverage",
    ),
    "TestUtilitiesContextFullCoverage": (
        "tests.unit.test_utilities_context_full_coverage",
        "TestUtilitiesContextFullCoverage",
    ),
    "TestUtilitiesCoverage": (
        "tests.unit.test_utilities_coverage",
        "TestUtilitiesCoverage",
    ),
    "TestUtilitiesDataMapper": (
        "tests.unit.test_utilities_data_mapper",
        "TestUtilitiesDataMapper",
    ),
    "TestUtilitiesDomainFullCoverage": (
        "tests.unit.test_utilities_domain_full_coverage",
        "TestUtilitiesDomainFullCoverage",
    ),
    "TestUtilitiesEnumFullCoverage": (
        "tests.unit.test_utilities_enum_full_coverage",
        "TestUtilitiesEnumFullCoverage",
    ),
    "TestUtilitiesGeneratorsFullCoverage": (
        "tests.unit.test_utilities_generators_full_coverage",
        "TestUtilitiesGeneratorsFullCoverage",
    ),
    "TestUtilitiesParserFullCoverage": (
        "tests.unit.test_utilities_parser_full_coverage",
        "TestUtilitiesParserFullCoverage",
    ),
    "TestUtilitiesTextFullCoverage": (
        "tests.unit.test_utilities_text_full_coverage",
        "TestUtilitiesTextFullCoverage",
    ),
    "TestUtilitiesTypeGuardsCoverage100": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TestUtilitiesTypeGuardsCoverage100",
    ),
    "Teste": ("tests.unit.test_exceptions", "Teste"),
    "Testr": ("tests.unit.test_result", "Testr"),
    "TestrCoverage": ("tests.unit.test_result_coverage_100", "TestrCoverage"),
    "TestsCore": ("tests.unit.test_service", "TestsCore"),
    "Testu": ("tests.unit.test_coverage_utilities", "Testu"),
    "TestuCacheClearObjectCache": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheClearObjectCache",
    ),
    "TestuCacheGenerateCacheKey": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheGenerateCacheKey",
    ),
    "TestuCacheHasCacheAttributes": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheHasCacheAttributes",
    ),
    "TestuCacheLogger": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheLogger",
    ),
    "TestuCacheNormalizeComponent": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheNormalizeComponent",
    ),
    "TestuCacheSortDictKeys": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheSortDictKeys",
    ),
    "TestuCacheSortKey": (
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheSortKey",
    ),
    "TestuDomain": ("tests.unit.test_utilities_domain", "TestuDomain"),
    "TestuMapperAccessors": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperAccessors",
    ),
    "TestuMapperAdvanced": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperAdvanced",
    ),
    "TestuMapperBuild": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperBuild",
    ),
    "TestuMapperConversions": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperConversions",
    ),
    "TestuMapperExtract": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperExtract",
    ),
    "TestuMapperUtils": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperUtils",
    ),
    "TestuStringParser": (
        "tests.unit.test_utilities_string_parser",
        "TestuStringParser",
    ),
    "TestuTypeChecker": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "TestuTypeChecker",
    ),
    "TextUtilityContract": (
        "tests.unit.contracts.text_contract",
        "TextUtilityContract",
    ),
    "TimeoutEnforcer": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TimeoutEnforcer",
    ),
    "UnknownHint": ("tests.unit.test_utilities_args_full_coverage", "UnknownHint"),
    "UtilitiesCacheCoverage100Namespace": (
        "tests.unit.test_utilities_cache_coverage_100",
        "UtilitiesCacheCoverage100Namespace",
    ),
    "UtilitiesMapperCoverage100Namespace": (
        "tests.unit.test_utilities_mapper_coverage_100",
        "UtilitiesMapperCoverage100Namespace",
    ),
    "UtilitiesMapperFullCoverageNamespace": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "UtilitiesMapperFullCoverageNamespace",
    ),
    "contracts": ("tests.unit.contracts", ""),
    "create_compare_entities_cases": (
        "tests.unit.test_utilities_domain",
        "create_compare_entities_cases",
    ),
    "create_compare_value_objects_cases": (
        "tests.unit.test_utilities_domain",
        "create_compare_value_objects_cases",
    ),
    "create_hash_entity_cases": (
        "tests.unit.test_utilities_domain",
        "create_hash_entity_cases",
    ),
    "create_hash_value_object_cases": (
        "tests.unit.test_utilities_domain",
        "create_hash_value_object_cases",
    ),
    "create_validate_entity_has_id_cases": (
        "tests.unit.test_utilities_domain",
        "create_validate_entity_has_id_cases",
    ),
    "create_validate_value_object_immutable_cases": (
        "tests.unit.test_utilities_domain",
        "create_validate_value_object_immutable_cases",
    ),
    "flext_tests": ("tests.unit.flext_tests", ""),
    "generators_module": (
        "tests.unit.test_utilities_generators_full_coverage",
        "generators_module",
    ),
    "handlers_module": ("tests.unit.test_handlers_full_coverage", "handlers_module"),
    "infra_git": ("tests.unit.conftest_infra", "infra_git"),
    "infra_git_repo": ("tests.unit.conftest_infra", "infra_git_repo"),
    "infra_io": ("tests.unit.conftest_infra", "infra_io"),
    "infra_path": ("tests.unit.conftest_infra", "infra_path"),
    "infra_patterns": ("tests.unit.conftest_infra", "infra_patterns"),
    "infra_pr_manager": ("tests.unit.conftest_infra", "infra_pr_manager"),
    "infra_pr_workspace_manager": (
        "tests.unit.conftest_infra",
        "infra_pr_workspace_manager",
    ),
    "infra_reporting": ("tests.unit.conftest_infra", "infra_reporting"),
    "infra_safe_command_output": (
        "tests.unit.conftest_infra",
        "infra_safe_command_output",
    ),
    "infra_selection": ("tests.unit.conftest_infra", "infra_selection"),
    "infra_subprocess": ("tests.unit.conftest_infra", "infra_subprocess"),
    "infra_templates": ("tests.unit.conftest_infra", "infra_templates"),
    "infra_test_workspace": ("tests.unit.conftest_infra", "infra_test_workspace"),
    "infra_toml": ("tests.unit.conftest_infra", "infra_toml"),
    "infra_workflow_linter": ("tests.unit.conftest_infra", "infra_workflow_linter"),
    "infra_workflow_syncer": ("tests.unit.conftest_infra", "infra_workflow_syncer"),
    "inject": ("tests.unit.test_di_incremental", "inject"),
    "mapper": ("tests.unit.test_utilities_mapper_full_coverage", "mapper"),
    "normalized_value_key_cases": (
        "tests.unit.test_utilities_string_parser",
        "normalized_value_key_cases",
    ),
    "p": ("tests.unit.protocols", "p"),
    "pytestmark": ("tests.unit.test_utilities_type_checker_coverage_100", "pytestmark"),
    "reset_runtime_state": (
        "tests.unit.test_runtime_full_coverage",
        "reset_runtime_state",
    ),
    "runtime_cov_tests": ("tests.unit.test_runtime_full_coverage", "runtime_cov_tests"),
    "runtime_module": (
        "tests.unit.test_utilities_generators_full_coverage",
        "runtime_module",
    ),
    "runtime_tests": ("tests.unit.test_runtime_full_coverage", "runtime_tests"),
    "test_accessor_take_pick_as_or_flat_and_agg_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_accessor_take_pick_as_or_flat_and_agg_branches",
    ),
    "test_aliases_are_available": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_aliases_are_available",
    ),
    "test_args_get_enum_params_annotated_unwrap_branch": (
        "tests.unit.test_utilities_args_full_coverage",
        "test_args_get_enum_params_annotated_unwrap_branch",
    ),
    "test_args_get_enum_params_branches": (
        "tests.unit.test_utilities_args_full_coverage",
        "test_args_get_enum_params_branches",
    ),
    "test_async_log_writer_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_async_log_writer_paths",
    ),
    "test_async_log_writer_shutdown_with_full_queue": (
        "tests.unit.test_runtime_full_coverage",
        "test_async_log_writer_shutdown_with_full_queue",
    ),
    "test_at_take_and_as_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_at_take_and_as_branches",
    ),
    "test_authentication_error_normalizes_extra_kwargs_into_context": (
        "tests.unit.test_exceptions_full_coverage",
        "test_authentication_error_normalizes_extra_kwargs_into_context",
    ),
    "test_bad_string_and_bad_bool_raise_value_error": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_bad_string_and_bad_bool_raise_value_error",
    ),
    "test_base_error_normalize_metadata_merges_existing_metadata_model": (
        "tests.unit.test_exceptions_full_coverage",
        "test_base_error_normalize_metadata_merges_existing_metadata_model",
    ),
    "test_basic_imports_work": (
        "tests.unit.test_models_validation_full_coverage",
        "test_basic_imports_work",
    ),
    "test_build_apply_transform_and_process_error_paths": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_build_apply_transform_and_process_error_paths",
    ),
    "test_canonical_aliases_are_available": (
        "tests.unit.test_models_generic_full_coverage",
        "test_canonical_aliases_are_available",
    ),
    "test_centralize_pydantic_cli_outputs_extended_metrics": (
        "tests.unit.test_refactor_cli_models_workflow",
        "test_centralize_pydantic_cli_outputs_extended_metrics",
    ),
    "test_chk_exercises_missed_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_chk_exercises_missed_branches",
    ),
    "test_circuit_breaker_transitions_and_metrics": (
        "tests.unit.test_dispatcher_reliability",
        "test_circuit_breaker_transitions_and_metrics",
    ),
    "test_class_nesting_appends_to_existing_namespace_and_removes_pass": (
        "tests.unit.test_transformer_class_nesting",
        "test_class_nesting_appends_to_existing_namespace_and_removes_pass",
    ),
    "test_class_nesting_keeps_unmapped_top_level_classes": (
        "tests.unit.test_transformer_class_nesting",
        "test_class_nesting_keeps_unmapped_top_level_classes",
    ),
    "test_class_nesting_moves_top_level_class_into_new_namespace": (
        "tests.unit.test_transformer_class_nesting",
        "test_class_nesting_moves_top_level_class_into_new_namespace",
    ),
    "test_clear_keys_values_items_and_validate_branches": (
        "tests.unit.test_context_full_coverage",
        "test_clear_keys_values_items_and_validate_branches",
    ),
    "test_command_pagination_limit": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_command_pagination_limit",
    ),
    "test_config_bridge_and_trace_context_and_http_validation": (
        "tests.unit.test_runtime_full_coverage",
        "test_config_bridge_and_trace_context_and_http_validation",
    ),
    "test_configuration_mapping_and_dict_negative_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_configuration_mapping_and_dict_negative_branches",
    ),
    "test_configure_structlog_edge_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_configure_structlog_edge_paths",
    ),
    "test_configure_structlog_print_logger_factory_fallback": (
        "tests.unit.test_runtime_full_coverage",
        "test_configure_structlog_print_logger_factory_fallback",
    ),
    "test_constants_auto_enum_and_bimapping_paths": (
        "tests.unit.test_constants_full_coverage",
        "test_constants_auto_enum_and_bimapping_paths",
    ),
    "test_construct_transform_and_deep_eq_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_construct_transform_and_deep_eq_branches",
    ),
    "test_container_and_service_domain_paths": (
        "tests.unit.test_context_full_coverage",
        "test_container_and_service_domain_paths",
    ),
    "test_container_resource_registration_metadata_normalized": (
        "tests.unit.test_models_container_full_coverage",
        "test_container_resource_registration_metadata_normalized",
    ),
    "test_context_data_metadata_normalizer_removed": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_metadata_normalizer_removed",
    ),
    "test_context_data_normalize_and_json_checks": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_normalize_and_json_checks",
    ),
    "test_context_data_validate_dict_serializable_error_paths": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_error_paths",
    ),
    "test_context_data_validate_dict_serializable_none_and_mapping": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_none_and_mapping",
    ),
    "test_context_data_validate_dict_serializable_real_dicts": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_real_dicts",
    ),
    "test_context_export_serializable_and_validators": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_serializable_and_validators",
    ),
    "test_context_export_statistics_validator_and_computed_fields": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_statistics_validator_and_computed_fields",
    ),
    "test_context_export_validate_dict_serializable_mapping_and_models": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_validate_dict_serializable_mapping_and_models",
    ),
    "test_context_export_validate_dict_serializable_valid": (
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_validate_dict_serializable_valid",
    ),
    "test_conversion_add_converted_and_error_metadata_append_paths": (
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_add_converted_and_error_metadata_append_paths",
    ),
    "test_conversion_add_skipped_skip_reason_upsert_paths": (
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_add_skipped_skip_reason_upsert_paths",
    ),
    "test_conversion_add_warning_metadata_append_paths": (
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_add_warning_metadata_append_paths",
    ),
    "test_conversion_and_extract_success_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_conversion_and_extract_success_branches",
    ),
    "test_conversion_start_and_complete_methods": (
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_start_and_complete_methods",
    ),
    "test_conversion_string_and_join_paths": (
        "tests.unit.test_utilities_conversion_full_coverage",
        "test_conversion_string_and_join_paths",
    ),
    "test_convert_default_fallback_matrix": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_convert_default_fallback_matrix",
    ),
    "test_convert_sequence_branch_returns_tuple": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_convert_sequence_branch_returns_tuple",
    ),
    "test_cqrs_query_resolve_deeper_and_int_pagination": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_cqrs_query_resolve_deeper_and_int_pagination",
    ),
    "test_create_auto_discover_and_mode_mapping": (
        "tests.unit.test_registry_full_coverage",
        "test_create_auto_discover_and_mode_mapping",
    ),
    "test_create_from_callable_and_repr": (
        "tests.unit.test_result_additional",
        "test_create_from_callable_and_repr",
    ),
    "test_create_merges_metadata_dict_branch": (
        "tests.unit.test_context_full_coverage",
        "test_create_merges_metadata_dict_branch",
    ),
    "test_create_overloads_and_auto_correlation": (
        "tests.unit.test_context_full_coverage",
        "test_create_overloads_and_auto_correlation",
    ),
    "test_decorators_family_blocks_dispatcher_target": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_decorators_family_blocks_dispatcher_target",
    ),
    "test_dependency_integration_and_wiring_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_dependency_integration_and_wiring_paths",
    ),
    "test_dependency_registration_duplicate_guards": (
        "tests.unit.test_runtime_full_coverage",
        "test_dependency_registration_duplicate_guards",
    ),
    "test_deprecated_class_noop_init_branch": (
        "tests.unit.test_utilities_deprecation_full_coverage",
        "test_deprecated_class_noop_init_branch",
    ),
    "test_discover_project_roots_without_nested_git_dirs": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_discover_project_roots_without_nested_git_dirs",
    ),
    "test_dispatcher_family_blocks_models_target": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_dispatcher_family_blocks_models_target",
    ),
    "test_dispatcher_reliability_branch_paths": (
        "tests.unit.test_dispatcher_reliability_full_coverage",
        "test_dispatcher_reliability_branch_paths",
    ),
    "test_ensure_and_extract_array_index_helpers": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_ensure_and_extract_array_index_helpers",
    ),
    "test_ensure_trace_context_dict_conversion_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_ensure_trace_context_dict_conversion_paths",
    ),
    "test_ensure_utc_datetime_adds_tzinfo_when_naive": (
        "tests.unit.test_models_validation_full_coverage",
        "test_ensure_utc_datetime_adds_tzinfo_when_naive",
    ),
    "test_ensure_utc_datetime_preserves_aware": (
        "tests.unit.test_models_validation_full_coverage",
        "test_ensure_utc_datetime_preserves_aware",
    ),
    "test_ensure_utc_datetime_returns_none_on_none": (
        "tests.unit.test_models_validation_full_coverage",
        "test_ensure_utc_datetime_returns_none_on_none",
    ),
    "test_entity_comparable_map_and_bulk_validation_paths": (
        "tests.unit.test_models_entity_full_coverage",
        "test_entity_comparable_map_and_bulk_validation_paths",
    ),
    "test_exceptions_uncovered_metadata_paths": (
        "tests.unit.test_exceptions_full_coverage",
        "test_exceptions_uncovered_metadata_paths",
    ),
    "test_execute_and_register_handler_failure_paths": (
        "tests.unit.test_registry_full_coverage",
        "test_execute_and_register_handler_failure_paths",
    ),
    "test_export_paths_with_metadata_and_statistics": (
        "tests.unit.test_context_full_coverage",
        "test_export_paths_with_metadata_and_statistics",
    ),
    "test_extract_error_paths_and_prop_accessor": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_extract_error_paths_and_prop_accessor",
    ),
    "test_extract_field_value_and_ensure_variants": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_extract_field_value_and_ensure_variants",
    ),
    "test_extract_mapping_or_none_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_extract_mapping_or_none_branches",
    ),
    "test_facade_binding_is_correct": (
        "tests.unit.test_models_validation_full_coverage",
        "test_facade_binding_is_correct",
    ),
    "test_field_and_fields_multi_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_field_and_fields_multi_branches",
    ),
    "test_filter_map_normalize_convert_helpers": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_filter_map_normalize_convert_helpers",
    ),
    "test_flext_message_type_alias_adapter": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_flext_message_type_alias_adapter",
    ),
    "test_flow_through_short_circuits_on_failure": (
        "tests.unit.test_result_additional",
        "test_flow_through_short_circuits_on_failure",
    ),
    "test_from_validation_and_to_model_paths": (
        "tests.unit.test_result_full_coverage",
        "test_from_validation_and_to_model_paths",
    ),
    "test_general_value_helpers_and_logger": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_general_value_helpers_and_logger",
    ),
    "test_get_logger_none_name_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_get_logger_none_name_paths",
    ),
    "test_get_plugin_and_register_metadata_and_list_items_exception": (
        "tests.unit.test_registry_full_coverage",
        "test_get_plugin_and_register_metadata_and_list_items_exception",
    ),
    "test_get_service_info": (
        "tests.unit.test_service_additional",
        "test_get_service_info",
    ),
    "test_group_sort_unique_slice_chunk_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_group_sort_unique_slice_chunk_branches",
    ),
    "test_guard_in_has_empty_none_helpers": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guard_in_has_empty_none_helpers",
    ),
    "test_guard_instance_attribute_access_warnings": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guard_instance_attribute_access_warnings",
    ),
    "test_guards_bool_identity_branch_via_isinstance_fallback": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_bool_identity_branch_via_isinstance_fallback",
    ),
    "test_guards_bool_shortcut_and_issubclass_typeerror": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_bool_shortcut_and_issubclass_typeerror",
    ),
    "test_guards_handler_type_issubclass_typeerror_branch_direct": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_handler_type_issubclass_typeerror_branch_direct",
    ),
    "test_guards_issubclass_success_when_callable_is_patched": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_issubclass_success_when_callable_is_patched",
    ),
    "test_guards_issubclass_typeerror_when_class_not_treated_as_callable": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_issubclass_typeerror_when_class_not_treated_as_callable",
    ),
    "test_handler_builder_fluent_methods": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_handler_builder_fluent_methods",
    ),
    "test_helper_consolidation_is_prechecked": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_helper_consolidation_is_prechecked",
    ),
    "test_inactive_and_none_value_paths": (
        "tests.unit.test_context_full_coverage",
        "test_inactive_and_none_value_paths",
    ),
    "test_init_fallback_and_lazy_returns_result_property": (
        "tests.unit.test_result_full_coverage",
        "test_init_fallback_and_lazy_returns_result_property",
    ),
    "test_invert_and_json_conversion_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_invert_and_json_conversion_branches",
    ),
    "test_is_container_negative_paths_and_callable": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_container_negative_paths_and_callable",
    ),
    "test_is_flexible_value_covers_all_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_flexible_value_covers_all_branches",
    ),
    "test_is_handler_type_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_handler_type_branches",
    ),
    "test_is_type_non_empty_unknown_and_tuple_and_fallback": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_type_non_empty_unknown_and_tuple_and_fallback",
    ),
    "test_is_type_protocol_fallback_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_type_protocol_fallback_branches",
    ),
    "test_is_valid_handles_validation_exception": (
        "tests.unit.test_service_additional",
        "test_is_valid_handles_validation_exception",
    ),
    "test_lash_runtime_result_paths": (
        "tests.unit.test_result_full_coverage",
        "test_lash_runtime_result_paths",
    ),
    "test_map_error_identity_and_transform": (
        "tests.unit.test_result_additional",
        "test_map_error_identity_and_transform",
    ),
    "test_map_flags_collect_and_invert_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_map_flags_collect_and_invert_branches",
    ),
    "test_map_flat_map_and_then_paths": (
        "tests.unit.test_result_full_coverage",
        "test_map_flat_map_and_then_paths",
    ),
    "test_merge_defaults_and_dump_paths": (
        "tests.unit.test_utilities_model_full_coverage",
        "test_merge_defaults_and_dump_paths",
    ),
    "test_merge_metadata_context_paths": (
        "tests.unit.test_exceptions_full_coverage",
        "test_merge_metadata_context_paths",
    ),
    "test_migrate_protocols_rewrites_references_with_p_alias": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_protocols_rewrites_references_with_p_alias",
    ),
    "test_migrate_to_mro_inlines_alias_constant_into_constants_class": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_inlines_alias_constant_into_constants_class",
    ),
    "test_migrate_to_mro_moves_constant_and_rewrites_reference": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_moves_constant_and_rewrites_reference",
    ),
    "test_migrate_to_mro_moves_manual_uppercase_assignment": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_moves_manual_uppercase_assignment",
    ),
    "test_migrate_to_mro_normalizes_facade_alias_to_c": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_normalizes_facade_alias_to_c",
    ),
    "test_migrate_to_mro_rejects_unknown_target": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_rejects_unknown_target",
    ),
    "test_migrate_typings_rewrites_references_with_t_alias": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_typings_rewrites_references_with_t_alias",
    ),
    "test_model_helpers_remaining_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_model_helpers_remaining_paths",
    ),
    "test_model_support_and_hash_compare_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_model_support_and_hash_compare_paths",
    ),
    "test_models_family_blocks_utilities_target": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_models_family_blocks_utilities_target",
    ),
    "test_models_handler_branches": (
        "tests.unit.test_models_handler_full_coverage",
        "test_models_handler_branches",
    ),
    "test_models_handler_uncovered_mode_and_reset_paths": (
        "tests.unit.test_models_handler_full_coverage",
        "test_models_handler_uncovered_mode_and_reset_paths",
    ),
    "test_models_settings_branch_paths": (
        "tests.unit.test_models_settings_full_coverage",
        "test_models_settings_branch_paths",
    ),
    "test_models_settings_context_validator_and_non_standard_status_input": (
        "tests.unit.test_models_settings_full_coverage",
        "test_models_settings_context_validator_and_non_standard_status_input",
    ),
    "test_namespace_enforce_cli_fails_on_manual_protocol_violation": (
        "tests.unit.test_refactor_cli_models_workflow",
        "test_namespace_enforce_cli_fails_on_manual_protocol_violation",
    ),
    "test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring",
    ),
    "test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future",
    ),
    "test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file",
    ),
    "test_namespace_enforcer_creates_missing_facades_and_rewrites_imports": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_creates_missing_facades_and_rewrites_imports",
    ),
    "test_namespace_enforcer_detects_cyclic_imports_in_tests_directory": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_cyclic_imports_in_tests_directory",
    ),
    "test_namespace_enforcer_detects_internal_private_imports": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_internal_private_imports",
    ),
    "test_namespace_enforcer_detects_manual_protocol_outside_canonical_files": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_manual_protocol_outside_canonical_files",
    ),
    "test_namespace_enforcer_detects_manual_typings_and_compat_aliases": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_manual_typings_and_compat_aliases",
    ),
    "test_namespace_enforcer_detects_missing_runtime_alias_outside_src": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_missing_runtime_alias_outside_src",
    ),
    "test_namespace_enforcer_does_not_rewrite_indented_import_aliases": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_does_not_rewrite_indented_import_aliases",
    ),
    "test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks": (
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks",
    ),
    "test_narrow_contextvar_exception_branch": (
        "tests.unit.test_context_full_coverage",
        "test_narrow_contextvar_exception_branch",
    ),
    "test_narrow_contextvar_invalid_inputs": (
        "tests.unit.test_context_full_coverage",
        "test_narrow_contextvar_invalid_inputs",
    ),
    "test_narrow_to_string_keyed_dict_and_mapping_paths": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_narrow_to_string_keyed_dict_and_mapping_paths",
    ),
    "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage": (
        "tests.unit.test_transformer_nested_class_propagation",
        "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage",
    ),
    "test_nested_class_propagation_updates_import_annotations_and_calls": (
        "tests.unit.test_transformer_nested_class_propagation",
        "test_nested_class_propagation_updates_import_annotations_and_calls",
    ),
    "test_non_empty_and_normalize_branches": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_non_empty_and_normalize_branches",
    ),
    "test_normalization_edge_branches": (
        "tests.unit.test_runtime_full_coverage",
        "test_normalization_edge_branches",
    ),
    "test_normalize_to_container_alias_removal_path": (
        "tests.unit.test_runtime_full_coverage",
        "test_normalize_to_container_alias_removal_path",
    ),
    "test_normalize_to_list_passes_list_through": (
        "tests.unit.test_models_validation_full_coverage",
        "test_normalize_to_list_passes_list_through",
    ),
    "test_normalize_to_list_wraps_int": (
        "tests.unit.test_models_validation_full_coverage",
        "test_normalize_to_list_wraps_int",
    ),
    "test_normalize_to_list_wraps_scalar": (
        "tests.unit.test_models_validation_full_coverage",
        "test_normalize_to_list_wraps_scalar",
    ),
    "test_normalize_to_metadata_alias_removal_path": (
        "tests.unit.test_runtime_full_coverage",
        "test_normalize_to_metadata_alias_removal_path",
    ),
    "test_normalize_to_pydantic_dict_and_value_branches": (
        "tests.unit.test_utilities_model_full_coverage",
        "test_normalize_to_pydantic_dict_and_value_branches",
    ),
    "test_not_found_error_correlation_id_selection_and_extra_kwargs": (
        "tests.unit.test_exceptions_full_coverage",
        "test_not_found_error_correlation_id_selection_and_extra_kwargs",
    ),
    "test_ok_accepts_none": (
        "tests.unit.test_result_additional",
        "test_ok_accepts_none",
    ),
    "test_operation_progress_start_operation_sets_runtime_fields": (
        "tests.unit.test_models_generic_full_coverage",
        "test_operation_progress_start_operation_sets_runtime_fields",
    ),
    "test_pagination_response_string_fallbacks": (
        "tests.unit.test_utilities_pagination_full_coverage",
        "test_pagination_response_string_fallbacks",
    ),
    "test_process_context_data_and_related_convenience": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_process_context_data_and_related_convenience",
    ),
    "test_protocol_and_simple_guard_helpers": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_protocol_and_simple_guard_helpers",
    ),
    "test_query_resolve_pagination_wrapper_and_fallback": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_query_resolve_pagination_wrapper_and_fallback",
    ),
    "test_query_validate_pagination_dict_and_default": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_query_validate_pagination_dict_and_default",
    ),
    "test_rate_limiter_blocks_then_recovers": (
        "tests.unit.test_dispatcher_reliability",
        "test_rate_limiter_blocks_then_recovers",
    ),
    "test_rate_limiter_jitter_application": (
        "tests.unit.test_dispatcher_reliability",
        "test_rate_limiter_jitter_application",
    ),
    "test_reconfigure_and_reset_state_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_reconfigure_and_reset_state_paths",
    ),
    "test_recover_tap_and_tap_error_paths": (
        "tests.unit.test_result_full_coverage",
        "test_recover_tap_and_tap_error_paths",
    ),
    "test_refactor_utilities_iter_python_files_includes_examples_and_scripts": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_refactor_utilities_iter_python_files_includes_examples_and_scripts",
    ),
    "test_remaining_build_fields_construct_and_eq_paths": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_remaining_build_fields_construct_and_eq_paths",
    ),
    "test_remaining_uncovered_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_remaining_uncovered_branches",
    ),
    "test_result_property_raises_on_failure": (
        "tests.unit.test_service_additional",
        "test_result_property_raises_on_failure",
    ),
    "test_retry_policy_behavior": (
        "tests.unit.test_dispatcher_reliability",
        "test_retry_policy_behavior",
    ),
    "test_reuse_existing_runtime_coverage_branches": (
        "tests.unit.test_runtime_full_coverage",
        "test_reuse_existing_runtime_coverage_branches",
    ),
    "test_reuse_existing_runtime_scenarios": (
        "tests.unit.test_runtime_full_coverage",
        "test_reuse_existing_runtime_scenarios",
    ),
    "test_runtime_create_instance_failure_branch": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_create_instance_failure_branch",
    ),
    "test_runtime_family_blocks_non_runtime_target": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_runtime_family_blocks_non_runtime_target",
    ),
    "test_runtime_integration_tracking_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_integration_tracking_paths",
    ),
    "test_runtime_misc_remaining_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_misc_remaining_paths",
    ),
    "test_runtime_module_accessors_and_metadata": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_module_accessors_and_metadata",
    ),
    "test_runtime_result_alias_compatibility": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_alias_compatibility",
    ),
    "test_runtime_result_all_missed_branches": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_all_missed_branches",
    ),
    "test_runtime_result_remaining_paths": (
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_remaining_paths",
    ),
    "test_scope_data_validators_and_errors": (
        "tests.unit.test_models_context_full_coverage",
        "test_scope_data_validators_and_errors",
    ),
    "test_service_request_timeout_post_validator_messages": (
        "tests.unit.test_models_service_full_coverage",
        "test_service_request_timeout_post_validator_messages",
    ),
    "test_service_request_timeout_validator_branches": (
        "tests.unit.test_models_service_full_coverage",
        "test_service_request_timeout_validator_branches",
    ),
    "test_set_set_all_get_validation_and_error_paths": (
        "tests.unit.test_context_full_coverage",
        "test_set_set_all_get_validation_and_error_paths",
    ),
    "test_settings_materialize_and_context_overrides": (
        "tests.unit.test_settings_full_coverage",
        "test_settings_materialize_and_context_overrides",
    ),
    "test_small_mapper_convenience_methods": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_small_mapper_convenience_methods",
    ),
    "test_statistics_and_custom_fields_validators": (
        "tests.unit.test_models_context_full_coverage",
        "test_statistics_and_custom_fields_validators",
    ),
    "test_strip_whitespace_preserves_clean": (
        "tests.unit.test_models_validation_full_coverage",
        "test_strip_whitespace_preserves_clean",
    ),
    "test_strip_whitespace_returns_empty_on_spaces": (
        "tests.unit.test_models_validation_full_coverage",
        "test_strip_whitespace_returns_empty_on_spaces",
    ),
    "test_strip_whitespace_trims_leading_trailing": (
        "tests.unit.test_models_validation_full_coverage",
        "test_strip_whitespace_trims_leading_trailing",
    ),
    "test_structlog_proxy_context_var_default_when_key_missing": (
        "tests.unit.test_models_context_full_coverage",
        "test_structlog_proxy_context_var_default_when_key_missing",
    ),
    "test_structlog_proxy_context_var_get_set_reset_paths": (
        "tests.unit.test_models_context_full_coverage",
        "test_structlog_proxy_context_var_get_set_reset_paths",
    ),
    "test_summary_error_paths_and_bindings_failures": (
        "tests.unit.test_registry_full_coverage",
        "test_summary_error_paths_and_bindings_failures",
    ),
    "test_summary_properties_and_subclass_storage_reset": (
        "tests.unit.test_registry_full_coverage",
        "test_summary_properties_and_subclass_storage_reset",
    ),
    "test_to_general_value_dict_removed": (
        "tests.unit.test_models_context_full_coverage",
        "test_to_general_value_dict_removed",
    ),
    "test_transform_option_extract_and_step_helpers": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_transform_option_extract_and_step_helpers",
    ),
    "test_type_guards_and_narrowing_failures": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_type_guards_and_narrowing_failures",
    ),
    "test_type_guards_result": (
        "tests.unit.test_result_full_coverage",
        "test_type_guards_result",
    ),
    "test_ultrawork_models_cli_runs_dry_run_copy": (
        "tests.unit.test_refactor_cli_models_workflow",
        "test_ultrawork_models_cli_runs_dry_run_copy",
    ),
    "test_update_exception_path": (
        "tests.unit.test_utilities_model_full_coverage",
        "test_update_exception_path",
    ),
    "test_update_statistics_remove_hook_and_clone_false_result": (
        "tests.unit.test_context_full_coverage",
        "test_update_statistics_remove_hook_and_clone_false_result",
    ),
    "test_update_success_path_returns_ok_result": (
        "tests.unit.test_utilities_model_full_coverage",
        "test_update_success_path_returns_ok_result",
    ),
    "test_utilities_family_allows_utilities_target": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_utilities_family_allows_utilities_target",
    ),
    "test_utilities_reliability_branches": (
        "tests.unit.test_utilities_reliability_full_coverage",
        "test_utilities_reliability_branches",
    ),
    "test_utilities_reliability_compose_returns_non_result_directly": (
        "tests.unit.test_utilities_reliability_full_coverage",
        "test_utilities_reliability_compose_returns_non_result_directly",
    ),
    "test_utilities_reliability_uncovered_retry_compose_and_sequence_paths": (
        "tests.unit.test_utilities_reliability_full_coverage",
        "test_utilities_reliability_uncovered_retry_compose_and_sequence_paths",
    ),
    "test_validate_config_dict_normalizes_dict": (
        "tests.unit.test_models_validation_full_coverage",
        "test_validate_config_dict_normalizes_dict",
    ),
    "test_validate_tags_list_from_string": (
        "tests.unit.test_models_validation_full_coverage",
        "test_validate_tags_list_from_string",
    ),
    "test_validate_tags_list_normalizes": (
        "tests.unit.test_models_validation_full_coverage",
        "test_validate_tags_list_normalizes",
    ),
    "test_validation_like_error_structure": (
        "tests.unit.test_result_full_coverage",
        "test_validation_like_error_structure",
    ),
    "test_with_resource_cleanup_runs": (
        "tests.unit.test_result_additional",
        "test_with_resource_cleanup_runs",
    ),
}

__all__ = [
    "EXCEPTION_CLASSES",
    "NORMALIZE_COMPONENT_SCENARIOS",
    "SORT_KEY_SCENARIOS",
    "AttrObject",
    "BadBool",
    "BadMapping",
    "BadString",
    "CircuitBreakerManager",
    "ClearCacheScenario",
    "ExplodingLenList",
    "FlextProtocols",
    "NestedClassPropagationTransformer",
    "NormalizeComponentScenario",
    "Provide",
    "RateLimiterManager",
    "RetryPolicy",
    "RuntimeCloneService",
    "SimpleObj",
    "SortKeyScenario",
    "T",
    "TMessage",
    "TestAutomatedArchitecture",
    "TestAutomatedExceptions",
    "TestAutomatedFlextContainer",
    "TestAutomatedFlextContext",
    "TestAutomatedFlextDecorators",
    "TestAutomatedFlextDispatcher",
    "TestAutomatedFlextHandlers",
    "TestAutomatedFlextLogger",
    "TestAutomatedFlextMixins",
    "TestAutomatedFlextRegistry",
    "TestAutomatedFlextRuntime",
    "TestAutomatedFlextService",
    "TestAutomatedFlextSettings",
    "TestAutomatedFlextUtilities",
    "TestAutomatedResult",
    "TestCollectionUtilitiesCoverage",
    "TestConstants",
    "TestContainerFullCoverage",
    "TestContext100Coverage",
    "TestCoverage76Lines",
    "TestCoverageContext",
    "TestCoverageExceptions",
    "TestCoverageLoggings",
    "TestCoverageModels",
    "TestDIIncremental",
    "TestDecoratorsDiscoveryFullCoverage",
    "TestDecoratorsFullCoverage",
    "TestDeprecationWarnings",
    "TestDiServicesAccess",
    "TestDispatcherDI",
    "TestDispatcherFullCoverage",
    "TestDispatcherMinimal",
    "TestDispatcherTimeoutCoverage100",
    "TestDocker",
    "TestEntityCoverageEdgeCases",
    "TestEnumUtilitiesCoverage",
    "TestFinal75PercentPush",
    "TestFlextContainer",
    "TestFlextContext",
    "TestFlextDecorators",
    "TestFlextHandlers",
    "TestFlextInfraNamespaceValidator",
    "TestFlextMixinsNestedClasses",
    "TestFlextModelsCollectionsCoverage100",
    "TestFlextModelsContainer",
    "TestFlextProtocols",
    "TestFlextRegistry",
    "TestFlextRuntime",
    "TestFlextSettings",
    "TestFlextTestsDomains",
    "TestFlextTestsFiles",
    "TestFlextTestsMatchers",
    "TestFlextUtilitiesArgs",
    "TestFlextUtilitiesConfiguration",
    "TestFlextUtilitiesReliability",
    "TestFlextVersion",
    "TestHandlerDecoratorDiscovery",
    "TestHandlersFullCoverage",
    "TestHelperConsolidationTransformer",
    "TestLoggingsErrorPaths",
    "TestLoggingsStrictReturns",
    "TestMixinsFullCoverage",
    "TestModels",
    "TestModels79Coverage",
    "TestModelsBaseFullCoverage",
    "TestModelsCollectionsFullCoverage",
    "TestModule",
    "TestPaginationCoverage100",
    "TestPhase2CoverageFinal",
    "TestResultExceptionCarrying",
    "TestRuntimeCoverage100",
    "TestService100Coverage",
    "TestServiceBootstrap",
    "TestServiceFullCoverage",
    "TestTypings",
    "TestTypingsFullCoverage",
    "TestUtilities",
    "TestUtilitiesCheckerFullCoverage",
    "TestUtilitiesCollectionCoverage",
    "TestUtilitiesCollectionFullCoverage",
    "TestUtilitiesConfigurationFullCoverage",
    "TestUtilitiesContextFullCoverage",
    "TestUtilitiesCoverage",
    "TestUtilitiesDataMapper",
    "TestUtilitiesDomainFullCoverage",
    "TestUtilitiesEnumFullCoverage",
    "TestUtilitiesGeneratorsFullCoverage",
    "TestUtilitiesParserFullCoverage",
    "TestUtilitiesTextFullCoverage",
    "TestUtilitiesTypeGuardsCoverage100",
    "Teste",
    "Testr",
    "TestrCoverage",
    "TestsCore",
    "Testu",
    "TestuCacheClearObjectCache",
    "TestuCacheGenerateCacheKey",
    "TestuCacheHasCacheAttributes",
    "TestuCacheLogger",
    "TestuCacheNormalizeComponent",
    "TestuCacheSortDictKeys",
    "TestuCacheSortKey",
    "TestuDomain",
    "TestuMapperAccessors",
    "TestuMapperAdvanced",
    "TestuMapperBuild",
    "TestuMapperConversions",
    "TestuMapperExtract",
    "TestuMapperUtils",
    "TestuStringParser",
    "TestuTypeChecker",
    "TextUtilityContract",
    "TimeoutEnforcer",
    "UnknownHint",
    "UtilitiesCacheCoverage100Namespace",
    "UtilitiesMapperCoverage100Namespace",
    "UtilitiesMapperFullCoverageNamespace",
    "contracts",
    "create_compare_entities_cases",
    "create_compare_value_objects_cases",
    "create_hash_entity_cases",
    "create_hash_value_object_cases",
    "create_validate_entity_has_id_cases",
    "create_validate_value_object_immutable_cases",
    "flext_tests",
    "generators_module",
    "handlers_module",
    "infra_git",
    "infra_git_repo",
    "infra_io",
    "infra_path",
    "infra_patterns",
    "infra_pr_manager",
    "infra_pr_workspace_manager",
    "infra_reporting",
    "infra_safe_command_output",
    "infra_selection",
    "infra_subprocess",
    "infra_templates",
    "infra_test_workspace",
    "infra_toml",
    "infra_workflow_linter",
    "infra_workflow_syncer",
    "inject",
    "mapper",
    "normalized_value_key_cases",
    "p",
    "pytestmark",
    "reset_runtime_state",
    "runtime_cov_tests",
    "runtime_module",
    "runtime_tests",
    "test_accessor_take_pick_as_or_flat_and_agg_branches",
    "test_aliases_are_available",
    "test_args_get_enum_params_annotated_unwrap_branch",
    "test_args_get_enum_params_branches",
    "test_async_log_writer_paths",
    "test_async_log_writer_shutdown_with_full_queue",
    "test_at_take_and_as_branches",
    "test_authentication_error_normalizes_extra_kwargs_into_context",
    "test_bad_string_and_bad_bool_raise_value_error",
    "test_base_error_normalize_metadata_merges_existing_metadata_model",
    "test_basic_imports_work",
    "test_build_apply_transform_and_process_error_paths",
    "test_canonical_aliases_are_available",
    "test_centralize_pydantic_cli_outputs_extended_metrics",
    "test_chk_exercises_missed_branches",
    "test_circuit_breaker_transitions_and_metrics",
    "test_class_nesting_appends_to_existing_namespace_and_removes_pass",
    "test_class_nesting_keeps_unmapped_top_level_classes",
    "test_class_nesting_moves_top_level_class_into_new_namespace",
    "test_clear_keys_values_items_and_validate_branches",
    "test_command_pagination_limit",
    "test_config_bridge_and_trace_context_and_http_validation",
    "test_configuration_mapping_and_dict_negative_branches",
    "test_configure_structlog_edge_paths",
    "test_configure_structlog_print_logger_factory_fallback",
    "test_constants_auto_enum_and_bimapping_paths",
    "test_construct_transform_and_deep_eq_branches",
    "test_container_and_service_domain_paths",
    "test_container_resource_registration_metadata_normalized",
    "test_context_data_metadata_normalizer_removed",
    "test_context_data_normalize_and_json_checks",
    "test_context_data_validate_dict_serializable_error_paths",
    "test_context_data_validate_dict_serializable_none_and_mapping",
    "test_context_data_validate_dict_serializable_real_dicts",
    "test_context_export_serializable_and_validators",
    "test_context_export_statistics_validator_and_computed_fields",
    "test_context_export_validate_dict_serializable_mapping_and_models",
    "test_context_export_validate_dict_serializable_valid",
    "test_conversion_add_converted_and_error_metadata_append_paths",
    "test_conversion_add_skipped_skip_reason_upsert_paths",
    "test_conversion_add_warning_metadata_append_paths",
    "test_conversion_and_extract_success_branches",
    "test_conversion_start_and_complete_methods",
    "test_conversion_string_and_join_paths",
    "test_convert_default_fallback_matrix",
    "test_convert_sequence_branch_returns_tuple",
    "test_cqrs_query_resolve_deeper_and_int_pagination",
    "test_create_auto_discover_and_mode_mapping",
    "test_create_from_callable_and_repr",
    "test_create_merges_metadata_dict_branch",
    "test_create_overloads_and_auto_correlation",
    "test_decorators_family_blocks_dispatcher_target",
    "test_dependency_integration_and_wiring_paths",
    "test_dependency_registration_duplicate_guards",
    "test_deprecated_class_noop_init_branch",
    "test_discover_project_roots_without_nested_git_dirs",
    "test_dispatcher_family_blocks_models_target",
    "test_dispatcher_reliability_branch_paths",
    "test_ensure_and_extract_array_index_helpers",
    "test_ensure_trace_context_dict_conversion_paths",
    "test_ensure_utc_datetime_adds_tzinfo_when_naive",
    "test_ensure_utc_datetime_preserves_aware",
    "test_ensure_utc_datetime_returns_none_on_none",
    "test_entity_comparable_map_and_bulk_validation_paths",
    "test_exceptions_uncovered_metadata_paths",
    "test_execute_and_register_handler_failure_paths",
    "test_export_paths_with_metadata_and_statistics",
    "test_extract_error_paths_and_prop_accessor",
    "test_extract_field_value_and_ensure_variants",
    "test_extract_mapping_or_none_branches",
    "test_facade_binding_is_correct",
    "test_field_and_fields_multi_branches",
    "test_filter_map_normalize_convert_helpers",
    "test_flext_message_type_alias_adapter",
    "test_flow_through_short_circuits_on_failure",
    "test_from_validation_and_to_model_paths",
    "test_general_value_helpers_and_logger",
    "test_get_logger_none_name_paths",
    "test_get_plugin_and_register_metadata_and_list_items_exception",
    "test_get_service_info",
    "test_group_sort_unique_slice_chunk_branches",
    "test_guard_in_has_empty_none_helpers",
    "test_guard_instance_attribute_access_warnings",
    "test_guards_bool_identity_branch_via_isinstance_fallback",
    "test_guards_bool_shortcut_and_issubclass_typeerror",
    "test_guards_handler_type_issubclass_typeerror_branch_direct",
    "test_guards_issubclass_success_when_callable_is_patched",
    "test_guards_issubclass_typeerror_when_class_not_treated_as_callable",
    "test_handler_builder_fluent_methods",
    "test_helper_consolidation_is_prechecked",
    "test_inactive_and_none_value_paths",
    "test_init_fallback_and_lazy_returns_result_property",
    "test_invert_and_json_conversion_branches",
    "test_is_container_negative_paths_and_callable",
    "test_is_flexible_value_covers_all_branches",
    "test_is_handler_type_branches",
    "test_is_type_non_empty_unknown_and_tuple_and_fallback",
    "test_is_type_protocol_fallback_branches",
    "test_is_valid_handles_validation_exception",
    "test_lash_runtime_result_paths",
    "test_map_error_identity_and_transform",
    "test_map_flags_collect_and_invert_branches",
    "test_map_flat_map_and_then_paths",
    "test_merge_defaults_and_dump_paths",
    "test_merge_metadata_context_paths",
    "test_migrate_protocols_rewrites_references_with_p_alias",
    "test_migrate_to_mro_inlines_alias_constant_into_constants_class",
    "test_migrate_to_mro_moves_constant_and_rewrites_reference",
    "test_migrate_to_mro_moves_manual_uppercase_assignment",
    "test_migrate_to_mro_normalizes_facade_alias_to_c",
    "test_migrate_to_mro_rejects_unknown_target",
    "test_migrate_typings_rewrites_references_with_t_alias",
    "test_model_helpers_remaining_paths",
    "test_model_support_and_hash_compare_paths",
    "test_models_family_blocks_utilities_target",
    "test_models_handler_branches",
    "test_models_handler_uncovered_mode_and_reset_paths",
    "test_models_settings_branch_paths",
    "test_models_settings_context_validator_and_non_standard_status_input",
    "test_namespace_enforce_cli_fails_on_manual_protocol_violation",
    "test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring",
    "test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future",
    "test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file",
    "test_namespace_enforcer_creates_missing_facades_and_rewrites_imports",
    "test_namespace_enforcer_detects_cyclic_imports_in_tests_directory",
    "test_namespace_enforcer_detects_internal_private_imports",
    "test_namespace_enforcer_detects_manual_protocol_outside_canonical_files",
    "test_namespace_enforcer_detects_manual_typings_and_compat_aliases",
    "test_namespace_enforcer_detects_missing_runtime_alias_outside_src",
    "test_namespace_enforcer_does_not_rewrite_indented_import_aliases",
    "test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks",
    "test_narrow_contextvar_exception_branch",
    "test_narrow_contextvar_invalid_inputs",
    "test_narrow_to_string_keyed_dict_and_mapping_paths",
    "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage",
    "test_nested_class_propagation_updates_import_annotations_and_calls",
    "test_non_empty_and_normalize_branches",
    "test_normalization_edge_branches",
    "test_normalize_to_container_alias_removal_path",
    "test_normalize_to_list_passes_list_through",
    "test_normalize_to_list_wraps_int",
    "test_normalize_to_list_wraps_scalar",
    "test_normalize_to_metadata_alias_removal_path",
    "test_normalize_to_pydantic_dict_and_value_branches",
    "test_not_found_error_correlation_id_selection_and_extra_kwargs",
    "test_ok_accepts_none",
    "test_operation_progress_start_operation_sets_runtime_fields",
    "test_pagination_response_string_fallbacks",
    "test_process_context_data_and_related_convenience",
    "test_protocol_and_simple_guard_helpers",
    "test_query_resolve_pagination_wrapper_and_fallback",
    "test_query_validate_pagination_dict_and_default",
    "test_rate_limiter_blocks_then_recovers",
    "test_rate_limiter_jitter_application",
    "test_reconfigure_and_reset_state_paths",
    "test_recover_tap_and_tap_error_paths",
    "test_refactor_utilities_iter_python_files_includes_examples_and_scripts",
    "test_remaining_build_fields_construct_and_eq_paths",
    "test_remaining_uncovered_branches",
    "test_result_property_raises_on_failure",
    "test_retry_policy_behavior",
    "test_reuse_existing_runtime_coverage_branches",
    "test_reuse_existing_runtime_scenarios",
    "test_runtime_create_instance_failure_branch",
    "test_runtime_family_blocks_non_runtime_target",
    "test_runtime_integration_tracking_paths",
    "test_runtime_misc_remaining_paths",
    "test_runtime_module_accessors_and_metadata",
    "test_runtime_result_alias_compatibility",
    "test_runtime_result_all_missed_branches",
    "test_runtime_result_remaining_paths",
    "test_scope_data_validators_and_errors",
    "test_service_request_timeout_post_validator_messages",
    "test_service_request_timeout_validator_branches",
    "test_set_set_all_get_validation_and_error_paths",
    "test_settings_materialize_and_context_overrides",
    "test_small_mapper_convenience_methods",
    "test_statistics_and_custom_fields_validators",
    "test_strip_whitespace_preserves_clean",
    "test_strip_whitespace_returns_empty_on_spaces",
    "test_strip_whitespace_trims_leading_trailing",
    "test_structlog_proxy_context_var_default_when_key_missing",
    "test_structlog_proxy_context_var_get_set_reset_paths",
    "test_summary_error_paths_and_bindings_failures",
    "test_summary_properties_and_subclass_storage_reset",
    "test_to_general_value_dict_removed",
    "test_transform_option_extract_and_step_helpers",
    "test_type_guards_and_narrowing_failures",
    "test_type_guards_result",
    "test_ultrawork_models_cli_runs_dry_run_copy",
    "test_update_exception_path",
    "test_update_statistics_remove_hook_and_clone_false_result",
    "test_update_success_path_returns_ok_result",
    "test_utilities_family_allows_utilities_target",
    "test_utilities_reliability_branches",
    "test_utilities_reliability_compose_returns_non_result_directly",
    "test_utilities_reliability_uncovered_retry_compose_and_sequence_paths",
    "test_validate_config_dict_normalizes_dict",
    "test_validate_tags_list_from_string",
    "test_validate_tags_list_normalizes",
    "test_validation_like_error_structure",
    "test_with_resource_cleanup_runs",
]


_LAZY_CACHE: dict[str, FlextTypes.ModuleExport] = {}


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562).

    A local cache ``_LAZY_CACHE`` persists resolved objects across repeated
    accesses during process lifetime.

    Args:
        name: Attribute name requested by dir()/import.

    Returns:
        Lazy-loaded module export type.

    Raises:
        AttributeError: If attribute not registered.

    """
    if name in _LAZY_CACHE:
        return _LAZY_CACHE[name]

    value = lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)
    _LAZY_CACHE[name] = value
    return value


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete.

    Returns:
        List of public names from module exports.

    """
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
