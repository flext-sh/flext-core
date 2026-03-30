# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Unit package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests.unit import (
        conftest_infra as conftest_infra,
        contracts as contracts,
        flext_tests as flext_tests,
        protocols as protocols,
        test_args_coverage_100 as test_args_coverage_100,
        test_collection_utilities_coverage_100 as test_collection_utilities_coverage_100,
        test_collections_coverage_100 as test_collections_coverage_100,
        test_config as test_config,
        test_constants as test_constants,
        test_container as test_container,
        test_container_full_coverage as test_container_full_coverage,
        test_context as test_context,
        test_context_coverage_100 as test_context_coverage_100,
        test_context_full_coverage as test_context_full_coverage,
        test_coverage_context as test_coverage_context,
        test_coverage_exceptions as test_coverage_exceptions,
        test_coverage_loggings as test_coverage_loggings,
        test_coverage_models as test_coverage_models,
        test_coverage_utilities as test_coverage_utilities,
        test_decorators as test_decorators,
        test_decorators_discovery_full_coverage as test_decorators_discovery_full_coverage,
        test_decorators_full_coverage as test_decorators_full_coverage,
        test_deprecation_warnings as test_deprecation_warnings,
        test_di_incremental as test_di_incremental,
        test_di_services_access as test_di_services_access,
        test_dispatcher_di as test_dispatcher_di,
        test_dispatcher_full_coverage as test_dispatcher_full_coverage,
        test_dispatcher_minimal as test_dispatcher_minimal,
        test_dispatcher_reliability as test_dispatcher_reliability,
        test_dispatcher_timeout_coverage_100 as test_dispatcher_timeout_coverage_100,
        test_entity_coverage as test_entity_coverage,
        test_enum_utilities_coverage_100 as test_enum_utilities_coverage_100,
        test_exceptions as test_exceptions,
        test_handler_decorator_discovery as test_handler_decorator_discovery,
        test_handlers as test_handlers,
        test_handlers_full_coverage as test_handlers_full_coverage,
        test_loggings_error_paths_coverage as test_loggings_error_paths_coverage,
        test_loggings_full_coverage as test_loggings_full_coverage,
        test_loggings_strict_returns as test_loggings_strict_returns,
        test_mixins as test_mixins,
        test_mixins_full_coverage as test_mixins_full_coverage,
        test_models as test_models,
        test_models_base_full_coverage as test_models_base_full_coverage,
        test_models_container as test_models_container,
        test_models_context_full_coverage as test_models_context_full_coverage,
        test_models_cqrs_full_coverage as test_models_cqrs_full_coverage,
        test_models_entity_full_coverage as test_models_entity_full_coverage,
        test_models_generic_full_coverage as test_models_generic_full_coverage,
        test_namespace_validator as test_namespace_validator,
        test_pagination_coverage_100 as test_pagination_coverage_100,
        test_protocols as test_protocols,
        test_refactor_cli_models_workflow as test_refactor_cli_models_workflow,
        test_refactor_migrate_to_class_mro as test_refactor_migrate_to_class_mro,
        test_refactor_namespace_enforcer as test_refactor_namespace_enforcer,
        test_refactor_policy_family_rules as test_refactor_policy_family_rules,
        test_registry as test_registry,
        test_registry_full_coverage as test_registry_full_coverage,
        test_result as test_result,
        test_result_additional as test_result_additional,
        test_result_coverage_100 as test_result_coverage_100,
        test_result_exception_carrying as test_result_exception_carrying,
        test_result_full_coverage as test_result_full_coverage,
        test_runtime as test_runtime,
        test_runtime_coverage_100 as test_runtime_coverage_100,
        test_runtime_full_coverage as test_runtime_full_coverage,
        test_service as test_service,
        test_service_additional as test_service_additional,
        test_service_bootstrap as test_service_bootstrap,
        test_service_coverage_100 as test_service_coverage_100,
        test_settings_coverage as test_settings_coverage,
        test_transformer_class_nesting as test_transformer_class_nesting,
        test_transformer_helper_consolidation as test_transformer_helper_consolidation,
        test_transformer_nested_class_propagation as test_transformer_nested_class_propagation,
        test_typings as test_typings,
        test_typings_full_coverage as test_typings_full_coverage,
        test_utilities as test_utilities,
        test_utilities_cache_coverage_100 as test_utilities_cache_coverage_100,
        test_utilities_collection_coverage_100 as test_utilities_collection_coverage_100,
        test_utilities_collection_full_coverage as test_utilities_collection_full_coverage,
        test_utilities_configuration_coverage_100 as test_utilities_configuration_coverage_100,
        test_utilities_configuration_full_coverage as test_utilities_configuration_full_coverage,
        test_utilities_context_full_coverage as test_utilities_context_full_coverage,
        test_utilities_coverage as test_utilities_coverage,
        test_utilities_data_mapper as test_utilities_data_mapper,
        test_utilities_domain as test_utilities_domain,
        test_utilities_domain_full_coverage as test_utilities_domain_full_coverage,
        test_utilities_enum_full_coverage as test_utilities_enum_full_coverage,
        test_utilities_generators_full_coverage as test_utilities_generators_full_coverage,
        test_utilities_guards_full_coverage as test_utilities_guards_full_coverage,
        test_utilities_mapper_coverage_100 as test_utilities_mapper_coverage_100,
        test_utilities_mapper_full_coverage as test_utilities_mapper_full_coverage,
        test_utilities_parser_full_coverage as test_utilities_parser_full_coverage,
        test_utilities_reliability as test_utilities_reliability,
        test_utilities_text_full_coverage as test_utilities_text_full_coverage,
        test_utilities_type_checker_coverage_100 as test_utilities_type_checker_coverage_100,
        test_utilities_type_guards_coverage_100 as test_utilities_type_guards_coverage_100,
        test_version as test_version,
        typings as typings,
    )
    from tests.unit.conftest_infra import (
        infra_git as infra_git,
        infra_git_repo as infra_git_repo,
        infra_io as infra_io,
        infra_path as infra_path,
        infra_patterns as infra_patterns,
        infra_reporting as infra_reporting,
        infra_safe_command_output as infra_safe_command_output,
        infra_selection as infra_selection,
        infra_subprocess as infra_subprocess,
        infra_templates as infra_templates,
        infra_test_workspace as infra_test_workspace,
        infra_toml as infra_toml,
    )
    from tests.unit.contracts import text_contract as text_contract
    from tests.unit.contracts.text_contract import (
        TextUtilityContract as TextUtilityContract,
    )
    from tests.unit.flext_tests import (
        test_docker as test_docker,
        test_domains as test_domains,
        test_files as test_files,
        test_matchers as test_matchers,
    )
    from tests.unit.flext_tests.test_docker import TestDocker as TestDocker
    from tests.unit.flext_tests.test_domains import (
        TestFlextTestsDomains as TestFlextTestsDomains,
    )
    from tests.unit.flext_tests.test_files import (
        TestFlextTestsFiles as TestFlextTestsFiles,
    )
    from tests.unit.flext_tests.test_matchers import (
        TestFlextTestsMatchers as TestFlextTestsMatchers,
    )
    from tests.unit.flext_tests.test_utilities import TestUtilities as TestUtilities
    from tests.unit.protocols import FlextProtocols as FlextProtocols, p as p
    from tests.unit.test_args_coverage_100 import (
        TestFlextUtilitiesArgs as TestFlextUtilitiesArgs,
    )
    from tests.unit.test_collection_utilities_coverage_100 import (
        TestCollectionUtilitiesCoverage as TestCollectionUtilitiesCoverage,
    )
    from tests.unit.test_collections_coverage_100 import (
        TestFlextModelsCollectionsCoverage100 as TestFlextModelsCollectionsCoverage100,
    )
    from tests.unit.test_config import TestFlextSettings as TestFlextSettings
    from tests.unit.test_constants import TestConstants as TestConstants
    from tests.unit.test_container import TestFlextContainer as TestFlextContainer
    from tests.unit.test_container_full_coverage import (
        TestContainerFullCoverage as TestContainerFullCoverage,
    )
    from tests.unit.test_context import TestFlextContext as TestFlextContext
    from tests.unit.test_context_coverage_100 import (
        TestContext100Coverage as TestContext100Coverage,
    )
    from tests.unit.test_context_full_coverage import (
        test_clear_keys_values_items_and_validate_branches as test_clear_keys_values_items_and_validate_branches,
        test_container_and_service_domain_paths as test_container_and_service_domain_paths,
        test_create_merges_metadata_dict_branch as test_create_merges_metadata_dict_branch,
        test_create_overloads_and_auto_correlation as test_create_overloads_and_auto_correlation,
        test_export_paths_with_metadata_and_statistics as test_export_paths_with_metadata_and_statistics,
        test_inactive_and_none_value_paths as test_inactive_and_none_value_paths,
        test_narrow_contextvar_exception_branch as test_narrow_contextvar_exception_branch,
        test_narrow_contextvar_invalid_inputs as test_narrow_contextvar_invalid_inputs,
        test_set_set_all_get_validation_and_error_paths as test_set_set_all_get_validation_and_error_paths,
        test_update_statistics_remove_hook_and_clone_false_result as test_update_statistics_remove_hook_and_clone_false_result,
    )
    from tests.unit.test_coverage_context import (
        TestCoverageContext as TestCoverageContext,
    )
    from tests.unit.test_coverage_exceptions import (
        TestCoverageExceptions as TestCoverageExceptions,
    )
    from tests.unit.test_coverage_loggings import (
        TestCoverageLoggings as TestCoverageLoggings,
    )
    from tests.unit.test_coverage_models import TestCoverageModels as TestCoverageModels
    from tests.unit.test_coverage_utilities import Testu as Testu
    from tests.unit.test_decorators import TestFlextDecorators as TestFlextDecorators
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestDecoratorsDiscoveryFullCoverage as TestDecoratorsDiscoveryFullCoverage,
    )
    from tests.unit.test_decorators_full_coverage import (
        TestDecoratorsFullCoverage as TestDecoratorsFullCoverage,
    )
    from tests.unit.test_deprecation_warnings import (
        TestDeprecationWarnings as TestDeprecationWarnings,
    )
    from tests.unit.test_di_incremental import (
        TestDIIncremental as TestDIIncremental,
        inject as inject,
    )
    from tests.unit.test_di_services_access import (
        TestDiServicesAccess as TestDiServicesAccess,
    )
    from tests.unit.test_dispatcher_di import TestDispatcherDI as TestDispatcherDI
    from tests.unit.test_dispatcher_full_coverage import (
        TestDispatcherFullCoverage as TestDispatcherFullCoverage,
    )
    from tests.unit.test_dispatcher_minimal import (
        TestDispatcherMinimal as TestDispatcherMinimal,
    )
    from tests.unit.test_dispatcher_reliability import (
        test_circuit_breaker_half_open_and_rate_limiter_accessors as test_circuit_breaker_half_open_and_rate_limiter_accessors,
        test_circuit_breaker_transitions_and_metrics as test_circuit_breaker_transitions_and_metrics,
        test_rate_limiter_blocks_then_recovers as test_rate_limiter_blocks_then_recovers,
        test_rate_limiter_jitter_application as test_rate_limiter_jitter_application,
        test_retry_policy_behavior as test_retry_policy_behavior,
    )
    from tests.unit.test_dispatcher_timeout_coverage_100 import (
        TestDispatcherTimeoutCoverage100 as TestDispatcherTimeoutCoverage100,
    )
    from tests.unit.test_entity_coverage import (
        TestEntityCoverageEdgeCases as TestEntityCoverageEdgeCases,
    )
    from tests.unit.test_enum_utilities_coverage_100 import (
        TestEnumUtilitiesCoverage as TestEnumUtilitiesCoverage,
    )
    from tests.unit.test_exceptions import (
        Teste as Teste,
        TestExceptionsHypothesis as TestExceptionsHypothesis,
    )
    from tests.unit.test_handler_decorator_discovery import (
        TestHandlerDecoratorDiscovery as TestHandlerDecoratorDiscovery,
    )
    from tests.unit.test_handlers import TestFlextHandlers as TestFlextHandlers
    from tests.unit.test_handlers_full_coverage import (
        TestHandlersFullCoverage as TestHandlersFullCoverage,
        handlers_module as handlers_module,
    )
    from tests.unit.test_loggings_error_paths_coverage import (
        TestLoggingsErrorPaths as TestLoggingsErrorPaths,
    )
    from tests.unit.test_loggings_full_coverage import TestModule as TestModule
    from tests.unit.test_loggings_strict_returns import (
        TestLoggingsStrictReturns as TestLoggingsStrictReturns,
    )
    from tests.unit.test_mixins import (
        TestFlextMixinsCQRS as TestFlextMixinsCQRS,
        TestFlextMixinsNestedClasses as TestFlextMixinsNestedClasses,
    )
    from tests.unit.test_mixins_full_coverage import (
        TestMixinsFullCoverage as TestMixinsFullCoverage,
    )
    from tests.unit.test_models import TestModels as TestModels
    from tests.unit.test_models_base_full_coverage import (
        TestModelsBaseFullCoverage as TestModelsBaseFullCoverage,
    )
    from tests.unit.test_models_container import (
        TestFlextModelsContainer as TestFlextModelsContainer,
    )
    from tests.unit.test_models_context_full_coverage import (
        test_context_data_metadata_normalizer_removed as test_context_data_metadata_normalizer_removed,
        test_context_data_normalize_and_json_checks as test_context_data_normalize_and_json_checks,
        test_context_data_validate_dict_serializable_error_paths as test_context_data_validate_dict_serializable_error_paths,
        test_context_data_validate_dict_serializable_none_and_mapping as test_context_data_validate_dict_serializable_none_and_mapping,
        test_context_data_validate_dict_serializable_real_dicts as test_context_data_validate_dict_serializable_real_dicts,
        test_context_export_serializable_and_validators as test_context_export_serializable_and_validators,
        test_context_export_statistics_validator_and_computed_fields as test_context_export_statistics_validator_and_computed_fields,
        test_context_export_validate_dict_serializable_mapping_and_models as test_context_export_validate_dict_serializable_mapping_and_models,
        test_context_export_validate_dict_serializable_valid as test_context_export_validate_dict_serializable_valid,
        test_scope_data_validators_and_errors as test_scope_data_validators_and_errors,
        test_statistics_and_custom_fields_validators as test_statistics_and_custom_fields_validators,
        test_structlog_proxy_context_var_default_when_key_missing as test_structlog_proxy_context_var_default_when_key_missing,
        test_structlog_proxy_context_var_get_set_reset_paths as test_structlog_proxy_context_var_get_set_reset_paths,
        test_to_general_value_dict_removed as test_to_general_value_dict_removed,
    )
    from tests.unit.test_models_cqrs_full_coverage import (
        test_command_pagination_limit as test_command_pagination_limit,
        test_cqrs_query_resolve_deeper_and_int_pagination as test_cqrs_query_resolve_deeper_and_int_pagination,
        test_flext_message_type_alias_adapter as test_flext_message_type_alias_adapter,
        test_handler_builder_fluent_methods as test_handler_builder_fluent_methods,
        test_query_resolve_pagination_wrapper_and_fallback as test_query_resolve_pagination_wrapper_and_fallback,
        test_query_validate_pagination_dict_and_default as test_query_validate_pagination_dict_and_default,
    )
    from tests.unit.test_models_entity_full_coverage import (
        test_entity_comparable_map_and_bulk_validation_paths as test_entity_comparable_map_and_bulk_validation_paths,
    )
    from tests.unit.test_models_generic_full_coverage import (
        test_canonical_aliases_are_available as test_canonical_aliases_are_available,
        test_conversion_add_converted_and_error_metadata_append_paths as test_conversion_add_converted_and_error_metadata_append_paths,
        test_conversion_add_skipped_skip_reason_upsert_paths as test_conversion_add_skipped_skip_reason_upsert_paths,
        test_conversion_add_warning_metadata_append_paths as test_conversion_add_warning_metadata_append_paths,
        test_conversion_start_and_complete_methods as test_conversion_start_and_complete_methods,
        test_operation_progress_start_operation_sets_runtime_fields as test_operation_progress_start_operation_sets_runtime_fields,
    )
    from tests.unit.test_namespace_validator import (
        TestFlextInfraNamespaceValidator as TestFlextInfraNamespaceValidator,
    )
    from tests.unit.test_pagination_coverage_100 import (
        TestPaginationCoverage100 as TestPaginationCoverage100,
    )
    from tests.unit.test_protocols import TestFlextProtocols as TestFlextProtocols
    from tests.unit.test_refactor_cli_models_workflow import (
        test_centralize_pydantic_cli_outputs_extended_metrics as test_centralize_pydantic_cli_outputs_extended_metrics,
        test_namespace_enforce_cli_fails_on_manual_protocol_violation as test_namespace_enforce_cli_fails_on_manual_protocol_violation,
        test_ultrawork_models_cli_runs_dry_run_copy as test_ultrawork_models_cli_runs_dry_run_copy,
    )
    from tests.unit.test_refactor_migrate_to_class_mro import (
        test_discover_project_roots_without_nested_git_dirs as test_discover_project_roots_without_nested_git_dirs,
        test_migrate_protocols_rewrites_references_with_p_alias as test_migrate_protocols_rewrites_references_with_p_alias,
        test_migrate_to_mro_inlines_alias_constant_into_constants_class as test_migrate_to_mro_inlines_alias_constant_into_constants_class,
        test_migrate_to_mro_moves_constant_and_rewrites_reference as test_migrate_to_mro_moves_constant_and_rewrites_reference,
        test_migrate_to_mro_moves_manual_uppercase_assignment as test_migrate_to_mro_moves_manual_uppercase_assignment,
        test_migrate_to_mro_normalizes_facade_alias_to_c as test_migrate_to_mro_normalizes_facade_alias_to_c,
        test_migrate_to_mro_rejects_unknown_target as test_migrate_to_mro_rejects_unknown_target,
        test_migrate_typings_rewrites_references_with_t_alias as test_migrate_typings_rewrites_references_with_t_alias,
        test_refactor_utilities_iter_python_files_includes_examples_and_scripts as test_refactor_utilities_iter_python_files_includes_examples_and_scripts,
    )
    from tests.unit.test_refactor_namespace_enforcer import (
        test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring as test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring,
        test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future as test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future,
        test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file as test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file,
        test_namespace_enforcer_creates_missing_facades_and_rewrites_imports as test_namespace_enforcer_creates_missing_facades_and_rewrites_imports,
        test_namespace_enforcer_detects_cyclic_imports_in_tests_directory as test_namespace_enforcer_detects_cyclic_imports_in_tests_directory,
        test_namespace_enforcer_detects_internal_private_imports as test_namespace_enforcer_detects_internal_private_imports,
        test_namespace_enforcer_detects_manual_protocol_outside_canonical_files as test_namespace_enforcer_detects_manual_protocol_outside_canonical_files,
        test_namespace_enforcer_detects_manual_typings_and_compat_aliases as test_namespace_enforcer_detects_manual_typings_and_compat_aliases,
        test_namespace_enforcer_detects_missing_runtime_alias_outside_src as test_namespace_enforcer_detects_missing_runtime_alias_outside_src,
        test_namespace_enforcer_does_not_rewrite_indented_import_aliases as test_namespace_enforcer_does_not_rewrite_indented_import_aliases,
        test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks as test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks,
    )
    from tests.unit.test_refactor_policy_family_rules import (
        test_decorators_family_blocks_dispatcher_target as test_decorators_family_blocks_dispatcher_target,
        test_dispatcher_family_blocks_models_target as test_dispatcher_family_blocks_models_target,
        test_helper_consolidation_is_prechecked as test_helper_consolidation_is_prechecked,
        test_models_family_blocks_utilities_target as test_models_family_blocks_utilities_target,
        test_runtime_family_blocks_non_runtime_target as test_runtime_family_blocks_non_runtime_target,
        test_utilities_family_allows_utilities_target as test_utilities_family_allows_utilities_target,
    )
    from tests.unit.test_registry import TestFlextRegistry as TestFlextRegistry
    from tests.unit.test_registry_full_coverage import (
        test_create_auto_discover_and_mode_mapping as test_create_auto_discover_and_mode_mapping,
        test_execute_and_register_handler_failure_paths as test_execute_and_register_handler_failure_paths,
        test_get_plugin_and_register_metadata_and_list_items_exception as test_get_plugin_and_register_metadata_and_list_items_exception,
        test_summary_error_paths_and_bindings_failures as test_summary_error_paths_and_bindings_failures,
        test_summary_properties_and_subclass_storage_reset as test_summary_properties_and_subclass_storage_reset,
    )
    from tests.unit.test_result import Testr as Testr
    from tests.unit.test_result_additional import (
        test_create_from_callable_and_repr as test_create_from_callable_and_repr,
        test_flow_through_short_circuits_on_failure as test_flow_through_short_circuits_on_failure,
        test_map_error_identity_and_transform as test_map_error_identity_and_transform,
        test_ok_accepts_none as test_ok_accepts_none,
        test_with_resource_cleanup_runs as test_with_resource_cleanup_runs,
    )
    from tests.unit.test_result_coverage_100 import TestrCoverage as TestrCoverage
    from tests.unit.test_result_exception_carrying import (
        TestResultExceptionCarrying as TestResultExceptionCarrying,
    )
    from tests.unit.test_result_full_coverage import (
        test_from_validation_and_to_model_paths as test_from_validation_and_to_model_paths,
        test_init_fallback_and_lazy_returns_result_property as test_init_fallback_and_lazy_returns_result_property,
        test_lash_runtime_result_paths as test_lash_runtime_result_paths,
        test_map_flat_map_and_then_paths as test_map_flat_map_and_then_paths,
        test_recover_tap_and_tap_error_paths as test_recover_tap_and_tap_error_paths,
        test_type_guards_result as test_type_guards_result,
        test_validation_like_error_structure as test_validation_like_error_structure,
    )
    from tests.unit.test_runtime import TestFlextRuntime as TestFlextRuntime
    from tests.unit.test_runtime_coverage_100 import (
        TestRuntimeCoverage100 as TestRuntimeCoverage100,
    )
    from tests.unit.test_runtime_full_coverage import (
        reset_runtime_state as reset_runtime_state,
        runtime_cov_tests as runtime_cov_tests,
        runtime_module as runtime_module,
        runtime_tests as runtime_tests,
        test_async_log_writer_paths as test_async_log_writer_paths,
        test_async_log_writer_shutdown_with_full_queue as test_async_log_writer_shutdown_with_full_queue,
        test_config_bridge_and_trace_context_and_http_validation as test_config_bridge_and_trace_context_and_http_validation,
        test_configure_structlog_edge_paths as test_configure_structlog_edge_paths,
        test_configure_structlog_print_logger_factory_fallback as test_configure_structlog_print_logger_factory_fallback,
        test_dependency_integration_and_wiring_paths as test_dependency_integration_and_wiring_paths,
        test_dependency_registration_duplicate_guards as test_dependency_registration_duplicate_guards,
        test_ensure_trace_context_dict_conversion_paths as test_ensure_trace_context_dict_conversion_paths,
        test_get_logger_none_name_paths as test_get_logger_none_name_paths,
        test_model_helpers_remaining_paths as test_model_helpers_remaining_paths,
        test_model_support_and_hash_compare_paths as test_model_support_and_hash_compare_paths,
        test_normalization_edge_branches as test_normalization_edge_branches,
        test_normalize_to_container_alias_removal_path as test_normalize_to_container_alias_removal_path,
        test_normalize_to_metadata_alias_removal_path as test_normalize_to_metadata_alias_removal_path,
        test_reconfigure_and_reset_state_paths as test_reconfigure_and_reset_state_paths,
        test_reuse_existing_runtime_coverage_branches as test_reuse_existing_runtime_coverage_branches,
        test_reuse_existing_runtime_scenarios as test_reuse_existing_runtime_scenarios,
        test_runtime_create_instance_failure_branch as test_runtime_create_instance_failure_branch,
        test_runtime_integration_tracking_paths as test_runtime_integration_tracking_paths,
        test_runtime_misc_remaining_paths as test_runtime_misc_remaining_paths,
        test_runtime_module_accessors_and_metadata as test_runtime_module_accessors_and_metadata,
        test_runtime_result_alias_compatibility as test_runtime_result_alias_compatibility,
        test_runtime_result_all_missed_branches as test_runtime_result_all_missed_branches,
        test_runtime_result_remaining_paths as test_runtime_result_remaining_paths,
    )
    from tests.unit.test_service import (
        TestsCore as TestsCore,
        TestServiceInternals as TestServiceInternals,
    )
    from tests.unit.test_service_additional import (
        RuntimeCloneService as RuntimeCloneService,
        test_is_valid_handles_validation_exception as test_is_valid_handles_validation_exception,
        test_result_property_raises_on_failure as test_result_property_raises_on_failure,
    )
    from tests.unit.test_service_bootstrap import (
        TestServiceBootstrap as TestServiceBootstrap,
    )
    from tests.unit.test_service_coverage_100 import (
        TestService100Coverage as TestService100Coverage,
    )
    from tests.unit.test_settings_coverage import (
        TestFlextSettingsCoverage as TestFlextSettingsCoverage,
    )
    from tests.unit.test_transformer_class_nesting import (
        test_class_nesting_appends_to_existing_namespace_and_removes_pass as test_class_nesting_appends_to_existing_namespace_and_removes_pass,
        test_class_nesting_keeps_unmapped_top_level_classes as test_class_nesting_keeps_unmapped_top_level_classes,
        test_class_nesting_moves_top_level_class_into_new_namespace as test_class_nesting_moves_top_level_class_into_new_namespace,
    )
    from tests.unit.test_transformer_helper_consolidation import (
        TestHelperConsolidationTransformer as TestHelperConsolidationTransformer,
    )
    from tests.unit.test_transformer_nested_class_propagation import (
        NestedClassPropagationTransformer as NestedClassPropagationTransformer,
        test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage as test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage,
        test_nested_class_propagation_updates_import_annotations_and_calls as test_nested_class_propagation_updates_import_annotations_and_calls,
    )
    from tests.unit.test_typings import TestTypings as TestTypings
    from tests.unit.test_typings_full_coverage import (
        TestTypingsFullCoverage as TestTypingsFullCoverage,
    )
    from tests.unit.test_utilities_cache_coverage_100 import (
        NORMALIZE_COMPONENT_SCENARIOS as NORMALIZE_COMPONENT_SCENARIOS,
        SORT_KEY_SCENARIOS as SORT_KEY_SCENARIOS,
        ClearCacheScenario as ClearCacheScenario,
        NormalizeComponentScenario as NormalizeComponentScenario,
        SortKeyScenario as SortKeyScenario,
        TestuCacheClearObjectCache as TestuCacheClearObjectCache,
        TestuCacheGenerateCacheKey as TestuCacheGenerateCacheKey,
        TestuCacheHasCacheAttributes as TestuCacheHasCacheAttributes,
        TestuCacheLogger as TestuCacheLogger,
        TestuCacheNormalizeComponent as TestuCacheNormalizeComponent,
        TestuCacheSortDictKeys as TestuCacheSortDictKeys,
        TestuCacheSortKey as TestuCacheSortKey,
        UtilitiesCacheCoverage100Namespace as UtilitiesCacheCoverage100Namespace,
    )
    from tests.unit.test_utilities_collection_coverage_100 import (
        TestUtilitiesCollectionCoverage as TestUtilitiesCollectionCoverage,
    )
    from tests.unit.test_utilities_collection_full_coverage import (
        TestUtilitiesCollectionFullCoverage as TestUtilitiesCollectionFullCoverage,
    )
    from tests.unit.test_utilities_configuration_coverage_100 import (
        TestFlextUtilitiesConfiguration as TestFlextUtilitiesConfiguration,
    )
    from tests.unit.test_utilities_configuration_full_coverage import (
        TestUtilitiesConfigurationFullCoverage as TestUtilitiesConfigurationFullCoverage,
    )
    from tests.unit.test_utilities_context_full_coverage import (
        TestUtilitiesContextFullCoverage as TestUtilitiesContextFullCoverage,
    )
    from tests.unit.test_utilities_coverage import (
        TestUtilitiesCoverage as TestUtilitiesCoverage,
    )
    from tests.unit.test_utilities_data_mapper import (
        TestUtilitiesDataMapper as TestUtilitiesDataMapper,
    )
    from tests.unit.test_utilities_domain import (
        TestuDomain as TestuDomain,
        create_compare_entities_cases as create_compare_entities_cases,
        create_compare_value_objects_cases as create_compare_value_objects_cases,
        create_hash_entity_cases as create_hash_entity_cases,
        create_hash_value_object_cases as create_hash_value_object_cases,
    )
    from tests.unit.test_utilities_domain_full_coverage import (
        TestUtilitiesDomainFullCoverage as TestUtilitiesDomainFullCoverage,
    )
    from tests.unit.test_utilities_generators_full_coverage import (
        TestUtilitiesGeneratorsFullCoverage as TestUtilitiesGeneratorsFullCoverage,
        generators_module as generators_module,
    )
    from tests.unit.test_utilities_guards_full_coverage import (
        test_aliases_are_available as test_aliases_are_available,
        test_chk_exercises_missed_branches as test_chk_exercises_missed_branches,
        test_configuration_mapping_and_dict_negative_branches as test_configuration_mapping_and_dict_negative_branches,
        test_extract_mapping_or_none_branches as test_extract_mapping_or_none_branches,
        test_guard_in_has_empty_none_helpers as test_guard_in_has_empty_none_helpers,
        test_guard_instance_attribute_access_warnings as test_guard_instance_attribute_access_warnings,
        test_guards_bool_identity_branch_via_isinstance_fallback as test_guards_bool_identity_branch_via_isinstance_fallback,
        test_guards_bool_shortcut_and_issubclass_typeerror as test_guards_bool_shortcut_and_issubclass_typeerror,
        test_guards_handler_type_issubclass_typeerror_branch_direct as test_guards_handler_type_issubclass_typeerror_branch_direct,
        test_guards_issubclass_success_when_callable_is_patched as test_guards_issubclass_success_when_callable_is_patched,
        test_guards_issubclass_typeerror_when_class_not_treated_as_callable as test_guards_issubclass_typeerror_when_class_not_treated_as_callable,
        test_is_container_negative_paths_and_callable as test_is_container_negative_paths_and_callable,
        test_is_handler_type_branches as test_is_handler_type_branches,
        test_is_type_non_empty_unknown_and_tuple_and_fallback as test_is_type_non_empty_unknown_and_tuple_and_fallback,
        test_is_type_protocol_fallback_branches as test_is_type_protocol_fallback_branches,
        test_non_empty_and_normalize_branches as test_non_empty_and_normalize_branches,
        test_protocol_and_simple_guard_helpers as test_protocol_and_simple_guard_helpers,
    )
    from tests.unit.test_utilities_mapper_coverage_100 import (
        SimpleObj as SimpleObj,
        TestuMapperAccessors as TestuMapperAccessors,
        TestuMapperAdvanced as TestuMapperAdvanced,
        TestuMapperBuild as TestuMapperBuild,
        TestuMapperConversions as TestuMapperConversions,
        TestuMapperExtract as TestuMapperExtract,
        TestuMapperUtils as TestuMapperUtils,
        UtilitiesMapperCoverage100Namespace as UtilitiesMapperCoverage100Namespace,
    )
    from tests.unit.test_utilities_mapper_full_coverage import (
        AttrObject as AttrObject,
        BadBool as BadBool,
        BadMapping as BadMapping,
        BadString as BadString,
        ExplodingLenList as ExplodingLenList,
        UtilitiesMapperFullCoverageNamespace as UtilitiesMapperFullCoverageNamespace,
        mapper as mapper,
        test_at_take_and_as_branches as test_at_take_and_as_branches,
        test_bad_string_and_bad_bool_raise_value_error as test_bad_string_and_bad_bool_raise_value_error,
        test_build_apply_transform_and_process_error_paths as test_build_apply_transform_and_process_error_paths,
        test_construct_transform_and_deep_eq_branches as test_construct_transform_and_deep_eq_branches,
        test_convert_default_fallback_matrix as test_convert_default_fallback_matrix,
        test_convert_sequence_branch_returns_tuple as test_convert_sequence_branch_returns_tuple,
        test_ensure_and_extract_array_index_helpers as test_ensure_and_extract_array_index_helpers,
        test_extract_error_paths_and_prop_accessor as test_extract_error_paths_and_prop_accessor,
        test_extract_field_value_and_ensure_variants as test_extract_field_value_and_ensure_variants,
        test_filter_map_normalize_convert_helpers as test_filter_map_normalize_convert_helpers,
        test_general_value_helpers_and_logger as test_general_value_helpers_and_logger,
        test_group_sort_unique_slice_chunk_branches as test_group_sort_unique_slice_chunk_branches,
        test_narrow_to_string_keyed_dict_and_mapping_paths as test_narrow_to_string_keyed_dict_and_mapping_paths,
        test_transform_option_extract_and_step_helpers as test_transform_option_extract_and_step_helpers,
        test_type_guards_and_narrowing_failures as test_type_guards_and_narrowing_failures,
    )
    from tests.unit.test_utilities_parser_full_coverage import (
        TestUtilitiesParserFullCoverage as TestUtilitiesParserFullCoverage,
    )
    from tests.unit.test_utilities_reliability import (
        TestFlextUtilitiesReliability as TestFlextUtilitiesReliability,
    )
    from tests.unit.test_utilities_text_full_coverage import (
        TestUtilitiesTextFullCoverage as TestUtilitiesTextFullCoverage,
    )
    from tests.unit.test_utilities_type_checker_coverage_100 import (
        T as T,
        TestuTypeChecker as TestuTypeChecker,
        TMessage as TMessage,
        pytestmark as pytestmark,
    )
    from tests.unit.test_utilities_type_guards_coverage_100 import (
        TestUtilitiesTypeGuardsCoverage100 as TestUtilitiesTypeGuardsCoverage100,
    )
    from tests.unit.test_version import TestFlextVersion as TestFlextVersion

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "AttrObject": ["tests.unit.test_utilities_mapper_full_coverage", "AttrObject"],
    "BadBool": ["tests.unit.test_utilities_mapper_full_coverage", "BadBool"],
    "BadMapping": ["tests.unit.test_utilities_mapper_full_coverage", "BadMapping"],
    "BadString": ["tests.unit.test_utilities_mapper_full_coverage", "BadString"],
    "ClearCacheScenario": [
        "tests.unit.test_utilities_cache_coverage_100",
        "ClearCacheScenario",
    ],
    "ExplodingLenList": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "ExplodingLenList",
    ],
    "FlextProtocols": ["tests.unit.protocols", "FlextProtocols"],
    "NORMALIZE_COMPONENT_SCENARIOS": [
        "tests.unit.test_utilities_cache_coverage_100",
        "NORMALIZE_COMPONENT_SCENARIOS",
    ],
    "NestedClassPropagationTransformer": [
        "tests.unit.test_transformer_nested_class_propagation",
        "NestedClassPropagationTransformer",
    ],
    "NormalizeComponentScenario": [
        "tests.unit.test_utilities_cache_coverage_100",
        "NormalizeComponentScenario",
    ],
    "RuntimeCloneService": [
        "tests.unit.test_service_additional",
        "RuntimeCloneService",
    ],
    "SORT_KEY_SCENARIOS": [
        "tests.unit.test_utilities_cache_coverage_100",
        "SORT_KEY_SCENARIOS",
    ],
    "SimpleObj": ["tests.unit.test_utilities_mapper_coverage_100", "SimpleObj"],
    "SortKeyScenario": [
        "tests.unit.test_utilities_cache_coverage_100",
        "SortKeyScenario",
    ],
    "T": ["tests.unit.test_utilities_type_checker_coverage_100", "T"],
    "TMessage": ["tests.unit.test_utilities_type_checker_coverage_100", "TMessage"],
    "TestCollectionUtilitiesCoverage": [
        "tests.unit.test_collection_utilities_coverage_100",
        "TestCollectionUtilitiesCoverage",
    ],
    "TestConstants": ["tests.unit.test_constants", "TestConstants"],
    "TestContainerFullCoverage": [
        "tests.unit.test_container_full_coverage",
        "TestContainerFullCoverage",
    ],
    "TestContext100Coverage": [
        "tests.unit.test_context_coverage_100",
        "TestContext100Coverage",
    ],
    "TestCoverageContext": ["tests.unit.test_coverage_context", "TestCoverageContext"],
    "TestCoverageExceptions": [
        "tests.unit.test_coverage_exceptions",
        "TestCoverageExceptions",
    ],
    "TestCoverageLoggings": [
        "tests.unit.test_coverage_loggings",
        "TestCoverageLoggings",
    ],
    "TestCoverageModels": ["tests.unit.test_coverage_models", "TestCoverageModels"],
    "TestDIIncremental": ["tests.unit.test_di_incremental", "TestDIIncremental"],
    "TestDecoratorsDiscoveryFullCoverage": [
        "tests.unit.test_decorators_discovery_full_coverage",
        "TestDecoratorsDiscoveryFullCoverage",
    ],
    "TestDecoratorsFullCoverage": [
        "tests.unit.test_decorators_full_coverage",
        "TestDecoratorsFullCoverage",
    ],
    "TestDeprecationWarnings": [
        "tests.unit.test_deprecation_warnings",
        "TestDeprecationWarnings",
    ],
    "TestDiServicesAccess": [
        "tests.unit.test_di_services_access",
        "TestDiServicesAccess",
    ],
    "TestDispatcherDI": ["tests.unit.test_dispatcher_di", "TestDispatcherDI"],
    "TestDispatcherFullCoverage": [
        "tests.unit.test_dispatcher_full_coverage",
        "TestDispatcherFullCoverage",
    ],
    "TestDispatcherMinimal": [
        "tests.unit.test_dispatcher_minimal",
        "TestDispatcherMinimal",
    ],
    "TestDispatcherTimeoutCoverage100": [
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TestDispatcherTimeoutCoverage100",
    ],
    "TestDocker": ["tests.unit.flext_tests.test_docker", "TestDocker"],
    "TestEntityCoverageEdgeCases": [
        "tests.unit.test_entity_coverage",
        "TestEntityCoverageEdgeCases",
    ],
    "TestEnumUtilitiesCoverage": [
        "tests.unit.test_enum_utilities_coverage_100",
        "TestEnumUtilitiesCoverage",
    ],
    "TestExceptionsHypothesis": [
        "tests.unit.test_exceptions",
        "TestExceptionsHypothesis",
    ],
    "TestFlextContainer": ["tests.unit.test_container", "TestFlextContainer"],
    "TestFlextContext": ["tests.unit.test_context", "TestFlextContext"],
    "TestFlextDecorators": ["tests.unit.test_decorators", "TestFlextDecorators"],
    "TestFlextHandlers": ["tests.unit.test_handlers", "TestFlextHandlers"],
    "TestFlextInfraNamespaceValidator": [
        "tests.unit.test_namespace_validator",
        "TestFlextInfraNamespaceValidator",
    ],
    "TestFlextMixinsCQRS": ["tests.unit.test_mixins", "TestFlextMixinsCQRS"],
    "TestFlextMixinsNestedClasses": [
        "tests.unit.test_mixins",
        "TestFlextMixinsNestedClasses",
    ],
    "TestFlextModelsCollectionsCoverage100": [
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsCoverage100",
    ],
    "TestFlextModelsContainer": [
        "tests.unit.test_models_container",
        "TestFlextModelsContainer",
    ],
    "TestFlextProtocols": ["tests.unit.test_protocols", "TestFlextProtocols"],
    "TestFlextRegistry": ["tests.unit.test_registry", "TestFlextRegistry"],
    "TestFlextRuntime": ["tests.unit.test_runtime", "TestFlextRuntime"],
    "TestFlextSettings": ["tests.unit.test_config", "TestFlextSettings"],
    "TestFlextSettingsCoverage": [
        "tests.unit.test_settings_coverage",
        "TestFlextSettingsCoverage",
    ],
    "TestFlextTestsDomains": [
        "tests.unit.flext_tests.test_domains",
        "TestFlextTestsDomains",
    ],
    "TestFlextTestsFiles": ["tests.unit.flext_tests.test_files", "TestFlextTestsFiles"],
    "TestFlextTestsMatchers": [
        "tests.unit.flext_tests.test_matchers",
        "TestFlextTestsMatchers",
    ],
    "TestFlextUtilitiesArgs": [
        "tests.unit.test_args_coverage_100",
        "TestFlextUtilitiesArgs",
    ],
    "TestFlextUtilitiesConfiguration": [
        "tests.unit.test_utilities_configuration_coverage_100",
        "TestFlextUtilitiesConfiguration",
    ],
    "TestFlextUtilitiesReliability": [
        "tests.unit.test_utilities_reliability",
        "TestFlextUtilitiesReliability",
    ],
    "TestFlextVersion": ["tests.unit.test_version", "TestFlextVersion"],
    "TestHandlerDecoratorDiscovery": [
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDecoratorDiscovery",
    ],
    "TestHandlersFullCoverage": [
        "tests.unit.test_handlers_full_coverage",
        "TestHandlersFullCoverage",
    ],
    "TestHelperConsolidationTransformer": [
        "tests.unit.test_transformer_helper_consolidation",
        "TestHelperConsolidationTransformer",
    ],
    "TestLoggingsErrorPaths": [
        "tests.unit.test_loggings_error_paths_coverage",
        "TestLoggingsErrorPaths",
    ],
    "TestLoggingsStrictReturns": [
        "tests.unit.test_loggings_strict_returns",
        "TestLoggingsStrictReturns",
    ],
    "TestMixinsFullCoverage": [
        "tests.unit.test_mixins_full_coverage",
        "TestMixinsFullCoverage",
    ],
    "TestModels": ["tests.unit.test_models", "TestModels"],
    "TestModelsBaseFullCoverage": [
        "tests.unit.test_models_base_full_coverage",
        "TestModelsBaseFullCoverage",
    ],
    "TestModule": ["tests.unit.test_loggings_full_coverage", "TestModule"],
    "TestPaginationCoverage100": [
        "tests.unit.test_pagination_coverage_100",
        "TestPaginationCoverage100",
    ],
    "TestResultExceptionCarrying": [
        "tests.unit.test_result_exception_carrying",
        "TestResultExceptionCarrying",
    ],
    "TestRuntimeCoverage100": [
        "tests.unit.test_runtime_coverage_100",
        "TestRuntimeCoverage100",
    ],
    "TestService100Coverage": [
        "tests.unit.test_service_coverage_100",
        "TestService100Coverage",
    ],
    "TestServiceBootstrap": [
        "tests.unit.test_service_bootstrap",
        "TestServiceBootstrap",
    ],
    "TestServiceInternals": ["tests.unit.test_service", "TestServiceInternals"],
    "TestTypings": ["tests.unit.test_typings", "TestTypings"],
    "TestTypingsFullCoverage": [
        "tests.unit.test_typings_full_coverage",
        "TestTypingsFullCoverage",
    ],
    "TestUtilities": ["tests.unit.flext_tests.test_utilities", "TestUtilities"],
    "TestUtilitiesCollectionCoverage": [
        "tests.unit.test_utilities_collection_coverage_100",
        "TestUtilitiesCollectionCoverage",
    ],
    "TestUtilitiesCollectionFullCoverage": [
        "tests.unit.test_utilities_collection_full_coverage",
        "TestUtilitiesCollectionFullCoverage",
    ],
    "TestUtilitiesConfigurationFullCoverage": [
        "tests.unit.test_utilities_configuration_full_coverage",
        "TestUtilitiesConfigurationFullCoverage",
    ],
    "TestUtilitiesContextFullCoverage": [
        "tests.unit.test_utilities_context_full_coverage",
        "TestUtilitiesContextFullCoverage",
    ],
    "TestUtilitiesCoverage": [
        "tests.unit.test_utilities_coverage",
        "TestUtilitiesCoverage",
    ],
    "TestUtilitiesDataMapper": [
        "tests.unit.test_utilities_data_mapper",
        "TestUtilitiesDataMapper",
    ],
    "TestUtilitiesDomainFullCoverage": [
        "tests.unit.test_utilities_domain_full_coverage",
        "TestUtilitiesDomainFullCoverage",
    ],
    "TestUtilitiesGeneratorsFullCoverage": [
        "tests.unit.test_utilities_generators_full_coverage",
        "TestUtilitiesGeneratorsFullCoverage",
    ],
    "TestUtilitiesParserFullCoverage": [
        "tests.unit.test_utilities_parser_full_coverage",
        "TestUtilitiesParserFullCoverage",
    ],
    "TestUtilitiesTextFullCoverage": [
        "tests.unit.test_utilities_text_full_coverage",
        "TestUtilitiesTextFullCoverage",
    ],
    "TestUtilitiesTypeGuardsCoverage100": [
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TestUtilitiesTypeGuardsCoverage100",
    ],
    "Teste": ["tests.unit.test_exceptions", "Teste"],
    "Testr": ["tests.unit.test_result", "Testr"],
    "TestrCoverage": ["tests.unit.test_result_coverage_100", "TestrCoverage"],
    "TestsCore": ["tests.unit.test_service", "TestsCore"],
    "Testu": ["tests.unit.test_coverage_utilities", "Testu"],
    "TestuCacheClearObjectCache": [
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheClearObjectCache",
    ],
    "TestuCacheGenerateCacheKey": [
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheGenerateCacheKey",
    ],
    "TestuCacheHasCacheAttributes": [
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheHasCacheAttributes",
    ],
    "TestuCacheLogger": [
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheLogger",
    ],
    "TestuCacheNormalizeComponent": [
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheNormalizeComponent",
    ],
    "TestuCacheSortDictKeys": [
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheSortDictKeys",
    ],
    "TestuCacheSortKey": [
        "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheSortKey",
    ],
    "TestuDomain": ["tests.unit.test_utilities_domain", "TestuDomain"],
    "TestuMapperAccessors": [
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperAccessors",
    ],
    "TestuMapperAdvanced": [
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperAdvanced",
    ],
    "TestuMapperBuild": [
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperBuild",
    ],
    "TestuMapperConversions": [
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperConversions",
    ],
    "TestuMapperExtract": [
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperExtract",
    ],
    "TestuMapperUtils": [
        "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperUtils",
    ],
    "TestuTypeChecker": [
        "tests.unit.test_utilities_type_checker_coverage_100",
        "TestuTypeChecker",
    ],
    "TextUtilityContract": [
        "tests.unit.contracts.text_contract",
        "TextUtilityContract",
    ],
    "UtilitiesCacheCoverage100Namespace": [
        "tests.unit.test_utilities_cache_coverage_100",
        "UtilitiesCacheCoverage100Namespace",
    ],
    "UtilitiesMapperCoverage100Namespace": [
        "tests.unit.test_utilities_mapper_coverage_100",
        "UtilitiesMapperCoverage100Namespace",
    ],
    "UtilitiesMapperFullCoverageNamespace": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "UtilitiesMapperFullCoverageNamespace",
    ],
    "conftest_infra": ["tests.unit.conftest_infra", ""],
    "contracts": ["tests.unit.contracts", ""],
    "create_compare_entities_cases": [
        "tests.unit.test_utilities_domain",
        "create_compare_entities_cases",
    ],
    "create_compare_value_objects_cases": [
        "tests.unit.test_utilities_domain",
        "create_compare_value_objects_cases",
    ],
    "create_hash_entity_cases": [
        "tests.unit.test_utilities_domain",
        "create_hash_entity_cases",
    ],
    "create_hash_value_object_cases": [
        "tests.unit.test_utilities_domain",
        "create_hash_value_object_cases",
    ],
    "flext_tests": ["tests.unit.flext_tests", ""],
    "generators_module": [
        "tests.unit.test_utilities_generators_full_coverage",
        "generators_module",
    ],
    "handlers_module": ["tests.unit.test_handlers_full_coverage", "handlers_module"],
    "infra_git": ["tests.unit.conftest_infra", "infra_git"],
    "infra_git_repo": ["tests.unit.conftest_infra", "infra_git_repo"],
    "infra_io": ["tests.unit.conftest_infra", "infra_io"],
    "infra_path": ["tests.unit.conftest_infra", "infra_path"],
    "infra_patterns": ["tests.unit.conftest_infra", "infra_patterns"],
    "infra_reporting": ["tests.unit.conftest_infra", "infra_reporting"],
    "infra_safe_command_output": [
        "tests.unit.conftest_infra",
        "infra_safe_command_output",
    ],
    "infra_selection": ["tests.unit.conftest_infra", "infra_selection"],
    "infra_subprocess": ["tests.unit.conftest_infra", "infra_subprocess"],
    "infra_templates": ["tests.unit.conftest_infra", "infra_templates"],
    "infra_test_workspace": ["tests.unit.conftest_infra", "infra_test_workspace"],
    "infra_toml": ["tests.unit.conftest_infra", "infra_toml"],
    "inject": ["tests.unit.test_di_incremental", "inject"],
    "mapper": ["tests.unit.test_utilities_mapper_full_coverage", "mapper"],
    "p": ["tests.unit.protocols", "p"],
    "protocols": ["tests.unit.protocols", ""],
    "pytestmark": ["tests.unit.test_utilities_type_checker_coverage_100", "pytestmark"],
    "reset_runtime_state": [
        "tests.unit.test_runtime_full_coverage",
        "reset_runtime_state",
    ],
    "runtime_cov_tests": ["tests.unit.test_runtime_full_coverage", "runtime_cov_tests"],
    "runtime_module": ["tests.unit.test_runtime_full_coverage", "runtime_module"],
    "runtime_tests": ["tests.unit.test_runtime_full_coverage", "runtime_tests"],
    "test_aliases_are_available": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_aliases_are_available",
    ],
    "test_args_coverage_100": ["tests.unit.test_args_coverage_100", ""],
    "test_async_log_writer_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_async_log_writer_paths",
    ],
    "test_async_log_writer_shutdown_with_full_queue": [
        "tests.unit.test_runtime_full_coverage",
        "test_async_log_writer_shutdown_with_full_queue",
    ],
    "test_at_take_and_as_branches": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_at_take_and_as_branches",
    ],
    "test_bad_string_and_bad_bool_raise_value_error": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_bad_string_and_bad_bool_raise_value_error",
    ],
    "test_build_apply_transform_and_process_error_paths": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_build_apply_transform_and_process_error_paths",
    ],
    "test_canonical_aliases_are_available": [
        "tests.unit.test_models_generic_full_coverage",
        "test_canonical_aliases_are_available",
    ],
    "test_centralize_pydantic_cli_outputs_extended_metrics": [
        "tests.unit.test_refactor_cli_models_workflow",
        "test_centralize_pydantic_cli_outputs_extended_metrics",
    ],
    "test_chk_exercises_missed_branches": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_chk_exercises_missed_branches",
    ],
    "test_circuit_breaker_half_open_and_rate_limiter_accessors": [
        "tests.unit.test_dispatcher_reliability",
        "test_circuit_breaker_half_open_and_rate_limiter_accessors",
    ],
    "test_circuit_breaker_transitions_and_metrics": [
        "tests.unit.test_dispatcher_reliability",
        "test_circuit_breaker_transitions_and_metrics",
    ],
    "test_class_nesting_appends_to_existing_namespace_and_removes_pass": [
        "tests.unit.test_transformer_class_nesting",
        "test_class_nesting_appends_to_existing_namespace_and_removes_pass",
    ],
    "test_class_nesting_keeps_unmapped_top_level_classes": [
        "tests.unit.test_transformer_class_nesting",
        "test_class_nesting_keeps_unmapped_top_level_classes",
    ],
    "test_class_nesting_moves_top_level_class_into_new_namespace": [
        "tests.unit.test_transformer_class_nesting",
        "test_class_nesting_moves_top_level_class_into_new_namespace",
    ],
    "test_clear_keys_values_items_and_validate_branches": [
        "tests.unit.test_context_full_coverage",
        "test_clear_keys_values_items_and_validate_branches",
    ],
    "test_collection_utilities_coverage_100": [
        "tests.unit.test_collection_utilities_coverage_100",
        "",
    ],
    "test_collections_coverage_100": ["tests.unit.test_collections_coverage_100", ""],
    "test_command_pagination_limit": [
        "tests.unit.test_models_cqrs_full_coverage",
        "test_command_pagination_limit",
    ],
    "test_config": ["tests.unit.test_config", ""],
    "test_config_bridge_and_trace_context_and_http_validation": [
        "tests.unit.test_runtime_full_coverage",
        "test_config_bridge_and_trace_context_and_http_validation",
    ],
    "test_configuration_mapping_and_dict_negative_branches": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_configuration_mapping_and_dict_negative_branches",
    ],
    "test_configure_structlog_edge_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_configure_structlog_edge_paths",
    ],
    "test_configure_structlog_print_logger_factory_fallback": [
        "tests.unit.test_runtime_full_coverage",
        "test_configure_structlog_print_logger_factory_fallback",
    ],
    "test_constants": ["tests.unit.test_constants", ""],
    "test_construct_transform_and_deep_eq_branches": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_construct_transform_and_deep_eq_branches",
    ],
    "test_container": ["tests.unit.test_container", ""],
    "test_container_and_service_domain_paths": [
        "tests.unit.test_context_full_coverage",
        "test_container_and_service_domain_paths",
    ],
    "test_container_full_coverage": ["tests.unit.test_container_full_coverage", ""],
    "test_context": ["tests.unit.test_context", ""],
    "test_context_coverage_100": ["tests.unit.test_context_coverage_100", ""],
    "test_context_data_metadata_normalizer_removed": [
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_metadata_normalizer_removed",
    ],
    "test_context_data_normalize_and_json_checks": [
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_normalize_and_json_checks",
    ],
    "test_context_data_validate_dict_serializable_error_paths": [
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_error_paths",
    ],
    "test_context_data_validate_dict_serializable_none_and_mapping": [
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_none_and_mapping",
    ],
    "test_context_data_validate_dict_serializable_real_dicts": [
        "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_real_dicts",
    ],
    "test_context_export_serializable_and_validators": [
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_serializable_and_validators",
    ],
    "test_context_export_statistics_validator_and_computed_fields": [
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_statistics_validator_and_computed_fields",
    ],
    "test_context_export_validate_dict_serializable_mapping_and_models": [
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_validate_dict_serializable_mapping_and_models",
    ],
    "test_context_export_validate_dict_serializable_valid": [
        "tests.unit.test_models_context_full_coverage",
        "test_context_export_validate_dict_serializable_valid",
    ],
    "test_context_full_coverage": ["tests.unit.test_context_full_coverage", ""],
    "test_conversion_add_converted_and_error_metadata_append_paths": [
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_add_converted_and_error_metadata_append_paths",
    ],
    "test_conversion_add_skipped_skip_reason_upsert_paths": [
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_add_skipped_skip_reason_upsert_paths",
    ],
    "test_conversion_add_warning_metadata_append_paths": [
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_add_warning_metadata_append_paths",
    ],
    "test_conversion_start_and_complete_methods": [
        "tests.unit.test_models_generic_full_coverage",
        "test_conversion_start_and_complete_methods",
    ],
    "test_convert_default_fallback_matrix": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_convert_default_fallback_matrix",
    ],
    "test_convert_sequence_branch_returns_tuple": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_convert_sequence_branch_returns_tuple",
    ],
    "test_coverage_context": ["tests.unit.test_coverage_context", ""],
    "test_coverage_exceptions": ["tests.unit.test_coverage_exceptions", ""],
    "test_coverage_loggings": ["tests.unit.test_coverage_loggings", ""],
    "test_coverage_models": ["tests.unit.test_coverage_models", ""],
    "test_coverage_utilities": ["tests.unit.test_coverage_utilities", ""],
    "test_cqrs_query_resolve_deeper_and_int_pagination": [
        "tests.unit.test_models_cqrs_full_coverage",
        "test_cqrs_query_resolve_deeper_and_int_pagination",
    ],
    "test_create_auto_discover_and_mode_mapping": [
        "tests.unit.test_registry_full_coverage",
        "test_create_auto_discover_and_mode_mapping",
    ],
    "test_create_from_callable_and_repr": [
        "tests.unit.test_result_additional",
        "test_create_from_callable_and_repr",
    ],
    "test_create_merges_metadata_dict_branch": [
        "tests.unit.test_context_full_coverage",
        "test_create_merges_metadata_dict_branch",
    ],
    "test_create_overloads_and_auto_correlation": [
        "tests.unit.test_context_full_coverage",
        "test_create_overloads_and_auto_correlation",
    ],
    "test_decorators": ["tests.unit.test_decorators", ""],
    "test_decorators_discovery_full_coverage": [
        "tests.unit.test_decorators_discovery_full_coverage",
        "",
    ],
    "test_decorators_family_blocks_dispatcher_target": [
        "tests.unit.test_refactor_policy_family_rules",
        "test_decorators_family_blocks_dispatcher_target",
    ],
    "test_decorators_full_coverage": ["tests.unit.test_decorators_full_coverage", ""],
    "test_dependency_integration_and_wiring_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_dependency_integration_and_wiring_paths",
    ],
    "test_dependency_registration_duplicate_guards": [
        "tests.unit.test_runtime_full_coverage",
        "test_dependency_registration_duplicate_guards",
    ],
    "test_deprecation_warnings": ["tests.unit.test_deprecation_warnings", ""],
    "test_di_incremental": ["tests.unit.test_di_incremental", ""],
    "test_di_services_access": ["tests.unit.test_di_services_access", ""],
    "test_discover_project_roots_without_nested_git_dirs": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_discover_project_roots_without_nested_git_dirs",
    ],
    "test_dispatcher_di": ["tests.unit.test_dispatcher_di", ""],
    "test_dispatcher_family_blocks_models_target": [
        "tests.unit.test_refactor_policy_family_rules",
        "test_dispatcher_family_blocks_models_target",
    ],
    "test_dispatcher_full_coverage": ["tests.unit.test_dispatcher_full_coverage", ""],
    "test_dispatcher_minimal": ["tests.unit.test_dispatcher_minimal", ""],
    "test_dispatcher_reliability": ["tests.unit.test_dispatcher_reliability", ""],
    "test_dispatcher_timeout_coverage_100": [
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "",
    ],
    "test_docker": ["tests.unit.flext_tests.test_docker", ""],
    "test_domains": ["tests.unit.flext_tests.test_domains", ""],
    "test_ensure_and_extract_array_index_helpers": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_ensure_and_extract_array_index_helpers",
    ],
    "test_ensure_trace_context_dict_conversion_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_ensure_trace_context_dict_conversion_paths",
    ],
    "test_entity_comparable_map_and_bulk_validation_paths": [
        "tests.unit.test_models_entity_full_coverage",
        "test_entity_comparable_map_and_bulk_validation_paths",
    ],
    "test_entity_coverage": ["tests.unit.test_entity_coverage", ""],
    "test_enum_utilities_coverage_100": [
        "tests.unit.test_enum_utilities_coverage_100",
        "",
    ],
    "test_exceptions": ["tests.unit.test_exceptions", ""],
    "test_execute_and_register_handler_failure_paths": [
        "tests.unit.test_registry_full_coverage",
        "test_execute_and_register_handler_failure_paths",
    ],
    "test_export_paths_with_metadata_and_statistics": [
        "tests.unit.test_context_full_coverage",
        "test_export_paths_with_metadata_and_statistics",
    ],
    "test_extract_error_paths_and_prop_accessor": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_extract_error_paths_and_prop_accessor",
    ],
    "test_extract_field_value_and_ensure_variants": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_extract_field_value_and_ensure_variants",
    ],
    "test_extract_mapping_or_none_branches": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_extract_mapping_or_none_branches",
    ],
    "test_files": ["tests.unit.flext_tests.test_files", ""],
    "test_filter_map_normalize_convert_helpers": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_filter_map_normalize_convert_helpers",
    ],
    "test_flext_message_type_alias_adapter": [
        "tests.unit.test_models_cqrs_full_coverage",
        "test_flext_message_type_alias_adapter",
    ],
    "test_flow_through_short_circuits_on_failure": [
        "tests.unit.test_result_additional",
        "test_flow_through_short_circuits_on_failure",
    ],
    "test_from_validation_and_to_model_paths": [
        "tests.unit.test_result_full_coverage",
        "test_from_validation_and_to_model_paths",
    ],
    "test_general_value_helpers_and_logger": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_general_value_helpers_and_logger",
    ],
    "test_get_logger_none_name_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_get_logger_none_name_paths",
    ],
    "test_get_plugin_and_register_metadata_and_list_items_exception": [
        "tests.unit.test_registry_full_coverage",
        "test_get_plugin_and_register_metadata_and_list_items_exception",
    ],
    "test_group_sort_unique_slice_chunk_branches": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_group_sort_unique_slice_chunk_branches",
    ],
    "test_guard_in_has_empty_none_helpers": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guard_in_has_empty_none_helpers",
    ],
    "test_guard_instance_attribute_access_warnings": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guard_instance_attribute_access_warnings",
    ],
    "test_guards_bool_identity_branch_via_isinstance_fallback": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_bool_identity_branch_via_isinstance_fallback",
    ],
    "test_guards_bool_shortcut_and_issubclass_typeerror": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_bool_shortcut_and_issubclass_typeerror",
    ],
    "test_guards_handler_type_issubclass_typeerror_branch_direct": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_handler_type_issubclass_typeerror_branch_direct",
    ],
    "test_guards_issubclass_success_when_callable_is_patched": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_issubclass_success_when_callable_is_patched",
    ],
    "test_guards_issubclass_typeerror_when_class_not_treated_as_callable": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_issubclass_typeerror_when_class_not_treated_as_callable",
    ],
    "test_handler_builder_fluent_methods": [
        "tests.unit.test_models_cqrs_full_coverage",
        "test_handler_builder_fluent_methods",
    ],
    "test_handler_decorator_discovery": [
        "tests.unit.test_handler_decorator_discovery",
        "",
    ],
    "test_handlers": ["tests.unit.test_handlers", ""],
    "test_handlers_full_coverage": ["tests.unit.test_handlers_full_coverage", ""],
    "test_helper_consolidation_is_prechecked": [
        "tests.unit.test_refactor_policy_family_rules",
        "test_helper_consolidation_is_prechecked",
    ],
    "test_inactive_and_none_value_paths": [
        "tests.unit.test_context_full_coverage",
        "test_inactive_and_none_value_paths",
    ],
    "test_init_fallback_and_lazy_returns_result_property": [
        "tests.unit.test_result_full_coverage",
        "test_init_fallback_and_lazy_returns_result_property",
    ],
    "test_is_container_negative_paths_and_callable": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_container_negative_paths_and_callable",
    ],
    "test_is_handler_type_branches": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_handler_type_branches",
    ],
    "test_is_type_non_empty_unknown_and_tuple_and_fallback": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_type_non_empty_unknown_and_tuple_and_fallback",
    ],
    "test_is_type_protocol_fallback_branches": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_type_protocol_fallback_branches",
    ],
    "test_is_valid_handles_validation_exception": [
        "tests.unit.test_service_additional",
        "test_is_valid_handles_validation_exception",
    ],
    "test_lash_runtime_result_paths": [
        "tests.unit.test_result_full_coverage",
        "test_lash_runtime_result_paths",
    ],
    "test_loggings_error_paths_coverage": [
        "tests.unit.test_loggings_error_paths_coverage",
        "",
    ],
    "test_loggings_full_coverage": ["tests.unit.test_loggings_full_coverage", ""],
    "test_loggings_strict_returns": ["tests.unit.test_loggings_strict_returns", ""],
    "test_map_error_identity_and_transform": [
        "tests.unit.test_result_additional",
        "test_map_error_identity_and_transform",
    ],
    "test_map_flat_map_and_then_paths": [
        "tests.unit.test_result_full_coverage",
        "test_map_flat_map_and_then_paths",
    ],
    "test_matchers": ["tests.unit.flext_tests.test_matchers", ""],
    "test_migrate_protocols_rewrites_references_with_p_alias": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_protocols_rewrites_references_with_p_alias",
    ],
    "test_migrate_to_mro_inlines_alias_constant_into_constants_class": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_inlines_alias_constant_into_constants_class",
    ],
    "test_migrate_to_mro_moves_constant_and_rewrites_reference": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_moves_constant_and_rewrites_reference",
    ],
    "test_migrate_to_mro_moves_manual_uppercase_assignment": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_moves_manual_uppercase_assignment",
    ],
    "test_migrate_to_mro_normalizes_facade_alias_to_c": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_normalizes_facade_alias_to_c",
    ],
    "test_migrate_to_mro_rejects_unknown_target": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_rejects_unknown_target",
    ],
    "test_migrate_typings_rewrites_references_with_t_alias": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_typings_rewrites_references_with_t_alias",
    ],
    "test_mixins": ["tests.unit.test_mixins", ""],
    "test_mixins_full_coverage": ["tests.unit.test_mixins_full_coverage", ""],
    "test_model_helpers_remaining_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_model_helpers_remaining_paths",
    ],
    "test_model_support_and_hash_compare_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_model_support_and_hash_compare_paths",
    ],
    "test_models": ["tests.unit.test_models", ""],
    "test_models_base_full_coverage": ["tests.unit.test_models_base_full_coverage", ""],
    "test_models_container": ["tests.unit.test_models_container", ""],
    "test_models_context_full_coverage": [
        "tests.unit.test_models_context_full_coverage",
        "",
    ],
    "test_models_cqrs_full_coverage": ["tests.unit.test_models_cqrs_full_coverage", ""],
    "test_models_entity_full_coverage": [
        "tests.unit.test_models_entity_full_coverage",
        "",
    ],
    "test_models_family_blocks_utilities_target": [
        "tests.unit.test_refactor_policy_family_rules",
        "test_models_family_blocks_utilities_target",
    ],
    "test_models_generic_full_coverage": [
        "tests.unit.test_models_generic_full_coverage",
        "",
    ],
    "test_namespace_enforce_cli_fails_on_manual_protocol_violation": [
        "tests.unit.test_refactor_cli_models_workflow",
        "test_namespace_enforce_cli_fails_on_manual_protocol_violation",
    ],
    "test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring": [
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring",
    ],
    "test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future": [
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future",
    ],
    "test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file": [
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file",
    ],
    "test_namespace_enforcer_creates_missing_facades_and_rewrites_imports": [
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_creates_missing_facades_and_rewrites_imports",
    ],
    "test_namespace_enforcer_detects_cyclic_imports_in_tests_directory": [
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_cyclic_imports_in_tests_directory",
    ],
    "test_namespace_enforcer_detects_internal_private_imports": [
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_internal_private_imports",
    ],
    "test_namespace_enforcer_detects_manual_protocol_outside_canonical_files": [
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_manual_protocol_outside_canonical_files",
    ],
    "test_namespace_enforcer_detects_manual_typings_and_compat_aliases": [
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_manual_typings_and_compat_aliases",
    ],
    "test_namespace_enforcer_detects_missing_runtime_alias_outside_src": [
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_missing_runtime_alias_outside_src",
    ],
    "test_namespace_enforcer_does_not_rewrite_indented_import_aliases": [
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_does_not_rewrite_indented_import_aliases",
    ],
    "test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks": [
        "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks",
    ],
    "test_namespace_validator": ["tests.unit.test_namespace_validator", ""],
    "test_narrow_contextvar_exception_branch": [
        "tests.unit.test_context_full_coverage",
        "test_narrow_contextvar_exception_branch",
    ],
    "test_narrow_contextvar_invalid_inputs": [
        "tests.unit.test_context_full_coverage",
        "test_narrow_contextvar_invalid_inputs",
    ],
    "test_narrow_to_string_keyed_dict_and_mapping_paths": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_narrow_to_string_keyed_dict_and_mapping_paths",
    ],
    "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage": [
        "tests.unit.test_transformer_nested_class_propagation",
        "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage",
    ],
    "test_nested_class_propagation_updates_import_annotations_and_calls": [
        "tests.unit.test_transformer_nested_class_propagation",
        "test_nested_class_propagation_updates_import_annotations_and_calls",
    ],
    "test_non_empty_and_normalize_branches": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_non_empty_and_normalize_branches",
    ],
    "test_normalization_edge_branches": [
        "tests.unit.test_runtime_full_coverage",
        "test_normalization_edge_branches",
    ],
    "test_normalize_to_container_alias_removal_path": [
        "tests.unit.test_runtime_full_coverage",
        "test_normalize_to_container_alias_removal_path",
    ],
    "test_normalize_to_metadata_alias_removal_path": [
        "tests.unit.test_runtime_full_coverage",
        "test_normalize_to_metadata_alias_removal_path",
    ],
    "test_ok_accepts_none": [
        "tests.unit.test_result_additional",
        "test_ok_accepts_none",
    ],
    "test_operation_progress_start_operation_sets_runtime_fields": [
        "tests.unit.test_models_generic_full_coverage",
        "test_operation_progress_start_operation_sets_runtime_fields",
    ],
    "test_pagination_coverage_100": ["tests.unit.test_pagination_coverage_100", ""],
    "test_protocol_and_simple_guard_helpers": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_protocol_and_simple_guard_helpers",
    ],
    "test_protocols": ["tests.unit.test_protocols", ""],
    "test_query_resolve_pagination_wrapper_and_fallback": [
        "tests.unit.test_models_cqrs_full_coverage",
        "test_query_resolve_pagination_wrapper_and_fallback",
    ],
    "test_query_validate_pagination_dict_and_default": [
        "tests.unit.test_models_cqrs_full_coverage",
        "test_query_validate_pagination_dict_and_default",
    ],
    "test_rate_limiter_blocks_then_recovers": [
        "tests.unit.test_dispatcher_reliability",
        "test_rate_limiter_blocks_then_recovers",
    ],
    "test_rate_limiter_jitter_application": [
        "tests.unit.test_dispatcher_reliability",
        "test_rate_limiter_jitter_application",
    ],
    "test_reconfigure_and_reset_state_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_reconfigure_and_reset_state_paths",
    ],
    "test_recover_tap_and_tap_error_paths": [
        "tests.unit.test_result_full_coverage",
        "test_recover_tap_and_tap_error_paths",
    ],
    "test_refactor_cli_models_workflow": [
        "tests.unit.test_refactor_cli_models_workflow",
        "",
    ],
    "test_refactor_migrate_to_class_mro": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "",
    ],
    "test_refactor_namespace_enforcer": [
        "tests.unit.test_refactor_namespace_enforcer",
        "",
    ],
    "test_refactor_policy_family_rules": [
        "tests.unit.test_refactor_policy_family_rules",
        "",
    ],
    "test_refactor_utilities_iter_python_files_includes_examples_and_scripts": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_refactor_utilities_iter_python_files_includes_examples_and_scripts",
    ],
    "test_registry": ["tests.unit.test_registry", ""],
    "test_registry_full_coverage": ["tests.unit.test_registry_full_coverage", ""],
    "test_result": ["tests.unit.test_result", ""],
    "test_result_additional": ["tests.unit.test_result_additional", ""],
    "test_result_coverage_100": ["tests.unit.test_result_coverage_100", ""],
    "test_result_exception_carrying": ["tests.unit.test_result_exception_carrying", ""],
    "test_result_full_coverage": ["tests.unit.test_result_full_coverage", ""],
    "test_result_property_raises_on_failure": [
        "tests.unit.test_service_additional",
        "test_result_property_raises_on_failure",
    ],
    "test_retry_policy_behavior": [
        "tests.unit.test_dispatcher_reliability",
        "test_retry_policy_behavior",
    ],
    "test_reuse_existing_runtime_coverage_branches": [
        "tests.unit.test_runtime_full_coverage",
        "test_reuse_existing_runtime_coverage_branches",
    ],
    "test_reuse_existing_runtime_scenarios": [
        "tests.unit.test_runtime_full_coverage",
        "test_reuse_existing_runtime_scenarios",
    ],
    "test_runtime": ["tests.unit.test_runtime", ""],
    "test_runtime_coverage_100": ["tests.unit.test_runtime_coverage_100", ""],
    "test_runtime_create_instance_failure_branch": [
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_create_instance_failure_branch",
    ],
    "test_runtime_family_blocks_non_runtime_target": [
        "tests.unit.test_refactor_policy_family_rules",
        "test_runtime_family_blocks_non_runtime_target",
    ],
    "test_runtime_full_coverage": ["tests.unit.test_runtime_full_coverage", ""],
    "test_runtime_integration_tracking_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_integration_tracking_paths",
    ],
    "test_runtime_misc_remaining_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_misc_remaining_paths",
    ],
    "test_runtime_module_accessors_and_metadata": [
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_module_accessors_and_metadata",
    ],
    "test_runtime_result_alias_compatibility": [
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_alias_compatibility",
    ],
    "test_runtime_result_all_missed_branches": [
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_all_missed_branches",
    ],
    "test_runtime_result_remaining_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_remaining_paths",
    ],
    "test_scope_data_validators_and_errors": [
        "tests.unit.test_models_context_full_coverage",
        "test_scope_data_validators_and_errors",
    ],
    "test_service": ["tests.unit.test_service", ""],
    "test_service_additional": ["tests.unit.test_service_additional", ""],
    "test_service_bootstrap": ["tests.unit.test_service_bootstrap", ""],
    "test_service_coverage_100": ["tests.unit.test_service_coverage_100", ""],
    "test_set_set_all_get_validation_and_error_paths": [
        "tests.unit.test_context_full_coverage",
        "test_set_set_all_get_validation_and_error_paths",
    ],
    "test_settings_coverage": ["tests.unit.test_settings_coverage", ""],
    "test_statistics_and_custom_fields_validators": [
        "tests.unit.test_models_context_full_coverage",
        "test_statistics_and_custom_fields_validators",
    ],
    "test_structlog_proxy_context_var_default_when_key_missing": [
        "tests.unit.test_models_context_full_coverage",
        "test_structlog_proxy_context_var_default_when_key_missing",
    ],
    "test_structlog_proxy_context_var_get_set_reset_paths": [
        "tests.unit.test_models_context_full_coverage",
        "test_structlog_proxy_context_var_get_set_reset_paths",
    ],
    "test_summary_error_paths_and_bindings_failures": [
        "tests.unit.test_registry_full_coverage",
        "test_summary_error_paths_and_bindings_failures",
    ],
    "test_summary_properties_and_subclass_storage_reset": [
        "tests.unit.test_registry_full_coverage",
        "test_summary_properties_and_subclass_storage_reset",
    ],
    "test_to_general_value_dict_removed": [
        "tests.unit.test_models_context_full_coverage",
        "test_to_general_value_dict_removed",
    ],
    "test_transform_option_extract_and_step_helpers": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_transform_option_extract_and_step_helpers",
    ],
    "test_transformer_class_nesting": ["tests.unit.test_transformer_class_nesting", ""],
    "test_transformer_helper_consolidation": [
        "tests.unit.test_transformer_helper_consolidation",
        "",
    ],
    "test_transformer_nested_class_propagation": [
        "tests.unit.test_transformer_nested_class_propagation",
        "",
    ],
    "test_type_guards_and_narrowing_failures": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_type_guards_and_narrowing_failures",
    ],
    "test_type_guards_result": [
        "tests.unit.test_result_full_coverage",
        "test_type_guards_result",
    ],
    "test_typings": ["tests.unit.test_typings", ""],
    "test_typings_full_coverage": ["tests.unit.test_typings_full_coverage", ""],
    "test_ultrawork_models_cli_runs_dry_run_copy": [
        "tests.unit.test_refactor_cli_models_workflow",
        "test_ultrawork_models_cli_runs_dry_run_copy",
    ],
    "test_update_statistics_remove_hook_and_clone_false_result": [
        "tests.unit.test_context_full_coverage",
        "test_update_statistics_remove_hook_and_clone_false_result",
    ],
    "test_utilities": ["tests.unit.test_utilities", ""],
    "test_utilities_cache_coverage_100": [
        "tests.unit.test_utilities_cache_coverage_100",
        "",
    ],
    "test_utilities_collection_coverage_100": [
        "tests.unit.test_utilities_collection_coverage_100",
        "",
    ],
    "test_utilities_collection_full_coverage": [
        "tests.unit.test_utilities_collection_full_coverage",
        "",
    ],
    "test_utilities_configuration_coverage_100": [
        "tests.unit.test_utilities_configuration_coverage_100",
        "",
    ],
    "test_utilities_configuration_full_coverage": [
        "tests.unit.test_utilities_configuration_full_coverage",
        "",
    ],
    "test_utilities_context_full_coverage": [
        "tests.unit.test_utilities_context_full_coverage",
        "",
    ],
    "test_utilities_coverage": ["tests.unit.test_utilities_coverage", ""],
    "test_utilities_data_mapper": ["tests.unit.test_utilities_data_mapper", ""],
    "test_utilities_domain": ["tests.unit.test_utilities_domain", ""],
    "test_utilities_domain_full_coverage": [
        "tests.unit.test_utilities_domain_full_coverage",
        "",
    ],
    "test_utilities_enum_full_coverage": [
        "tests.unit.test_utilities_enum_full_coverage",
        "",
    ],
    "test_utilities_family_allows_utilities_target": [
        "tests.unit.test_refactor_policy_family_rules",
        "test_utilities_family_allows_utilities_target",
    ],
    "test_utilities_generators_full_coverage": [
        "tests.unit.test_utilities_generators_full_coverage",
        "",
    ],
    "test_utilities_guards_full_coverage": [
        "tests.unit.test_utilities_guards_full_coverage",
        "",
    ],
    "test_utilities_mapper_coverage_100": [
        "tests.unit.test_utilities_mapper_coverage_100",
        "",
    ],
    "test_utilities_mapper_full_coverage": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "",
    ],
    "test_utilities_parser_full_coverage": [
        "tests.unit.test_utilities_parser_full_coverage",
        "",
    ],
    "test_utilities_reliability": ["tests.unit.test_utilities_reliability", ""],
    "test_utilities_text_full_coverage": [
        "tests.unit.test_utilities_text_full_coverage",
        "",
    ],
    "test_utilities_type_checker_coverage_100": [
        "tests.unit.test_utilities_type_checker_coverage_100",
        "",
    ],
    "test_utilities_type_guards_coverage_100": [
        "tests.unit.test_utilities_type_guards_coverage_100",
        "",
    ],
    "test_validation_like_error_structure": [
        "tests.unit.test_result_full_coverage",
        "test_validation_like_error_structure",
    ],
    "test_version": ["tests.unit.test_version", ""],
    "test_with_resource_cleanup_runs": [
        "tests.unit.test_result_additional",
        "test_with_resource_cleanup_runs",
    ],
    "text_contract": ["tests.unit.contracts.text_contract", ""],
    "typings": ["tests.unit.typings", ""],
}

_EXPORTS: Sequence[str] = [
    "AttrObject",
    "BadBool",
    "BadMapping",
    "BadString",
    "ClearCacheScenario",
    "ExplodingLenList",
    "FlextProtocols",
    "NORMALIZE_COMPONENT_SCENARIOS",
    "NestedClassPropagationTransformer",
    "NormalizeComponentScenario",
    "RuntimeCloneService",
    "SORT_KEY_SCENARIOS",
    "SimpleObj",
    "SortKeyScenario",
    "T",
    "TMessage",
    "TestCollectionUtilitiesCoverage",
    "TestConstants",
    "TestContainerFullCoverage",
    "TestContext100Coverage",
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
    "TestExceptionsHypothesis",
    "TestFlextContainer",
    "TestFlextContext",
    "TestFlextDecorators",
    "TestFlextHandlers",
    "TestFlextInfraNamespaceValidator",
    "TestFlextMixinsCQRS",
    "TestFlextMixinsNestedClasses",
    "TestFlextModelsCollectionsCoverage100",
    "TestFlextModelsContainer",
    "TestFlextProtocols",
    "TestFlextRegistry",
    "TestFlextRuntime",
    "TestFlextSettings",
    "TestFlextSettingsCoverage",
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
    "TestModelsBaseFullCoverage",
    "TestModule",
    "TestPaginationCoverage100",
    "TestResultExceptionCarrying",
    "TestRuntimeCoverage100",
    "TestService100Coverage",
    "TestServiceBootstrap",
    "TestServiceInternals",
    "TestTypings",
    "TestTypingsFullCoverage",
    "TestUtilities",
    "TestUtilitiesCollectionCoverage",
    "TestUtilitiesCollectionFullCoverage",
    "TestUtilitiesConfigurationFullCoverage",
    "TestUtilitiesContextFullCoverage",
    "TestUtilitiesCoverage",
    "TestUtilitiesDataMapper",
    "TestUtilitiesDomainFullCoverage",
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
    "TestuTypeChecker",
    "TextUtilityContract",
    "UtilitiesCacheCoverage100Namespace",
    "UtilitiesMapperCoverage100Namespace",
    "UtilitiesMapperFullCoverageNamespace",
    "conftest_infra",
    "contracts",
    "create_compare_entities_cases",
    "create_compare_value_objects_cases",
    "create_hash_entity_cases",
    "create_hash_value_object_cases",
    "flext_tests",
    "generators_module",
    "handlers_module",
    "infra_git",
    "infra_git_repo",
    "infra_io",
    "infra_path",
    "infra_patterns",
    "infra_reporting",
    "infra_safe_command_output",
    "infra_selection",
    "infra_subprocess",
    "infra_templates",
    "infra_test_workspace",
    "infra_toml",
    "inject",
    "mapper",
    "p",
    "protocols",
    "pytestmark",
    "reset_runtime_state",
    "runtime_cov_tests",
    "runtime_module",
    "runtime_tests",
    "test_aliases_are_available",
    "test_args_coverage_100",
    "test_async_log_writer_paths",
    "test_async_log_writer_shutdown_with_full_queue",
    "test_at_take_and_as_branches",
    "test_bad_string_and_bad_bool_raise_value_error",
    "test_build_apply_transform_and_process_error_paths",
    "test_canonical_aliases_are_available",
    "test_centralize_pydantic_cli_outputs_extended_metrics",
    "test_chk_exercises_missed_branches",
    "test_circuit_breaker_half_open_and_rate_limiter_accessors",
    "test_circuit_breaker_transitions_and_metrics",
    "test_class_nesting_appends_to_existing_namespace_and_removes_pass",
    "test_class_nesting_keeps_unmapped_top_level_classes",
    "test_class_nesting_moves_top_level_class_into_new_namespace",
    "test_clear_keys_values_items_and_validate_branches",
    "test_collection_utilities_coverage_100",
    "test_collections_coverage_100",
    "test_command_pagination_limit",
    "test_config",
    "test_config_bridge_and_trace_context_and_http_validation",
    "test_configuration_mapping_and_dict_negative_branches",
    "test_configure_structlog_edge_paths",
    "test_configure_structlog_print_logger_factory_fallback",
    "test_constants",
    "test_construct_transform_and_deep_eq_branches",
    "test_container",
    "test_container_and_service_domain_paths",
    "test_container_full_coverage",
    "test_context",
    "test_context_coverage_100",
    "test_context_data_metadata_normalizer_removed",
    "test_context_data_normalize_and_json_checks",
    "test_context_data_validate_dict_serializable_error_paths",
    "test_context_data_validate_dict_serializable_none_and_mapping",
    "test_context_data_validate_dict_serializable_real_dicts",
    "test_context_export_serializable_and_validators",
    "test_context_export_statistics_validator_and_computed_fields",
    "test_context_export_validate_dict_serializable_mapping_and_models",
    "test_context_export_validate_dict_serializable_valid",
    "test_context_full_coverage",
    "test_conversion_add_converted_and_error_metadata_append_paths",
    "test_conversion_add_skipped_skip_reason_upsert_paths",
    "test_conversion_add_warning_metadata_append_paths",
    "test_conversion_start_and_complete_methods",
    "test_convert_default_fallback_matrix",
    "test_convert_sequence_branch_returns_tuple",
    "test_coverage_context",
    "test_coverage_exceptions",
    "test_coverage_loggings",
    "test_coverage_models",
    "test_coverage_utilities",
    "test_cqrs_query_resolve_deeper_and_int_pagination",
    "test_create_auto_discover_and_mode_mapping",
    "test_create_from_callable_and_repr",
    "test_create_merges_metadata_dict_branch",
    "test_create_overloads_and_auto_correlation",
    "test_decorators",
    "test_decorators_discovery_full_coverage",
    "test_decorators_family_blocks_dispatcher_target",
    "test_decorators_full_coverage",
    "test_dependency_integration_and_wiring_paths",
    "test_dependency_registration_duplicate_guards",
    "test_deprecation_warnings",
    "test_di_incremental",
    "test_di_services_access",
    "test_discover_project_roots_without_nested_git_dirs",
    "test_dispatcher_di",
    "test_dispatcher_family_blocks_models_target",
    "test_dispatcher_full_coverage",
    "test_dispatcher_minimal",
    "test_dispatcher_reliability",
    "test_dispatcher_timeout_coverage_100",
    "test_docker",
    "test_domains",
    "test_ensure_and_extract_array_index_helpers",
    "test_ensure_trace_context_dict_conversion_paths",
    "test_entity_comparable_map_and_bulk_validation_paths",
    "test_entity_coverage",
    "test_enum_utilities_coverage_100",
    "test_exceptions",
    "test_execute_and_register_handler_failure_paths",
    "test_export_paths_with_metadata_and_statistics",
    "test_extract_error_paths_and_prop_accessor",
    "test_extract_field_value_and_ensure_variants",
    "test_extract_mapping_or_none_branches",
    "test_files",
    "test_filter_map_normalize_convert_helpers",
    "test_flext_message_type_alias_adapter",
    "test_flow_through_short_circuits_on_failure",
    "test_from_validation_and_to_model_paths",
    "test_general_value_helpers_and_logger",
    "test_get_logger_none_name_paths",
    "test_get_plugin_and_register_metadata_and_list_items_exception",
    "test_group_sort_unique_slice_chunk_branches",
    "test_guard_in_has_empty_none_helpers",
    "test_guard_instance_attribute_access_warnings",
    "test_guards_bool_identity_branch_via_isinstance_fallback",
    "test_guards_bool_shortcut_and_issubclass_typeerror",
    "test_guards_handler_type_issubclass_typeerror_branch_direct",
    "test_guards_issubclass_success_when_callable_is_patched",
    "test_guards_issubclass_typeerror_when_class_not_treated_as_callable",
    "test_handler_builder_fluent_methods",
    "test_handler_decorator_discovery",
    "test_handlers",
    "test_handlers_full_coverage",
    "test_helper_consolidation_is_prechecked",
    "test_inactive_and_none_value_paths",
    "test_init_fallback_and_lazy_returns_result_property",
    "test_is_container_negative_paths_and_callable",
    "test_is_handler_type_branches",
    "test_is_type_non_empty_unknown_and_tuple_and_fallback",
    "test_is_type_protocol_fallback_branches",
    "test_is_valid_handles_validation_exception",
    "test_lash_runtime_result_paths",
    "test_loggings_error_paths_coverage",
    "test_loggings_full_coverage",
    "test_loggings_strict_returns",
    "test_map_error_identity_and_transform",
    "test_map_flat_map_and_then_paths",
    "test_matchers",
    "test_migrate_protocols_rewrites_references_with_p_alias",
    "test_migrate_to_mro_inlines_alias_constant_into_constants_class",
    "test_migrate_to_mro_moves_constant_and_rewrites_reference",
    "test_migrate_to_mro_moves_manual_uppercase_assignment",
    "test_migrate_to_mro_normalizes_facade_alias_to_c",
    "test_migrate_to_mro_rejects_unknown_target",
    "test_migrate_typings_rewrites_references_with_t_alias",
    "test_mixins",
    "test_mixins_full_coverage",
    "test_model_helpers_remaining_paths",
    "test_model_support_and_hash_compare_paths",
    "test_models",
    "test_models_base_full_coverage",
    "test_models_container",
    "test_models_context_full_coverage",
    "test_models_cqrs_full_coverage",
    "test_models_entity_full_coverage",
    "test_models_family_blocks_utilities_target",
    "test_models_generic_full_coverage",
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
    "test_namespace_validator",
    "test_narrow_contextvar_exception_branch",
    "test_narrow_contextvar_invalid_inputs",
    "test_narrow_to_string_keyed_dict_and_mapping_paths",
    "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage",
    "test_nested_class_propagation_updates_import_annotations_and_calls",
    "test_non_empty_and_normalize_branches",
    "test_normalization_edge_branches",
    "test_normalize_to_container_alias_removal_path",
    "test_normalize_to_metadata_alias_removal_path",
    "test_ok_accepts_none",
    "test_operation_progress_start_operation_sets_runtime_fields",
    "test_pagination_coverage_100",
    "test_protocol_and_simple_guard_helpers",
    "test_protocols",
    "test_query_resolve_pagination_wrapper_and_fallback",
    "test_query_validate_pagination_dict_and_default",
    "test_rate_limiter_blocks_then_recovers",
    "test_rate_limiter_jitter_application",
    "test_reconfigure_and_reset_state_paths",
    "test_recover_tap_and_tap_error_paths",
    "test_refactor_cli_models_workflow",
    "test_refactor_migrate_to_class_mro",
    "test_refactor_namespace_enforcer",
    "test_refactor_policy_family_rules",
    "test_refactor_utilities_iter_python_files_includes_examples_and_scripts",
    "test_registry",
    "test_registry_full_coverage",
    "test_result",
    "test_result_additional",
    "test_result_coverage_100",
    "test_result_exception_carrying",
    "test_result_full_coverage",
    "test_result_property_raises_on_failure",
    "test_retry_policy_behavior",
    "test_reuse_existing_runtime_coverage_branches",
    "test_reuse_existing_runtime_scenarios",
    "test_runtime",
    "test_runtime_coverage_100",
    "test_runtime_create_instance_failure_branch",
    "test_runtime_family_blocks_non_runtime_target",
    "test_runtime_full_coverage",
    "test_runtime_integration_tracking_paths",
    "test_runtime_misc_remaining_paths",
    "test_runtime_module_accessors_and_metadata",
    "test_runtime_result_alias_compatibility",
    "test_runtime_result_all_missed_branches",
    "test_runtime_result_remaining_paths",
    "test_scope_data_validators_and_errors",
    "test_service",
    "test_service_additional",
    "test_service_bootstrap",
    "test_service_coverage_100",
    "test_set_set_all_get_validation_and_error_paths",
    "test_settings_coverage",
    "test_statistics_and_custom_fields_validators",
    "test_structlog_proxy_context_var_default_when_key_missing",
    "test_structlog_proxy_context_var_get_set_reset_paths",
    "test_summary_error_paths_and_bindings_failures",
    "test_summary_properties_and_subclass_storage_reset",
    "test_to_general_value_dict_removed",
    "test_transform_option_extract_and_step_helpers",
    "test_transformer_class_nesting",
    "test_transformer_helper_consolidation",
    "test_transformer_nested_class_propagation",
    "test_type_guards_and_narrowing_failures",
    "test_type_guards_result",
    "test_typings",
    "test_typings_full_coverage",
    "test_ultrawork_models_cli_runs_dry_run_copy",
    "test_update_statistics_remove_hook_and_clone_false_result",
    "test_utilities",
    "test_utilities_cache_coverage_100",
    "test_utilities_collection_coverage_100",
    "test_utilities_collection_full_coverage",
    "test_utilities_configuration_coverage_100",
    "test_utilities_configuration_full_coverage",
    "test_utilities_context_full_coverage",
    "test_utilities_coverage",
    "test_utilities_data_mapper",
    "test_utilities_domain",
    "test_utilities_domain_full_coverage",
    "test_utilities_enum_full_coverage",
    "test_utilities_family_allows_utilities_target",
    "test_utilities_generators_full_coverage",
    "test_utilities_guards_full_coverage",
    "test_utilities_mapper_coverage_100",
    "test_utilities_mapper_full_coverage",
    "test_utilities_parser_full_coverage",
    "test_utilities_reliability",
    "test_utilities_text_full_coverage",
    "test_utilities_type_checker_coverage_100",
    "test_utilities_type_guards_coverage_100",
    "test_validation_like_error_structure",
    "test_version",
    "test_with_resource_cleanup_runs",
    "text_contract",
    "typings",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
