# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Tests package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes

    from . import (
        benchmark as benchmark,
        helpers as helpers,
        integration as integration,
        unit as unit,
    )
    from .base import TestsFlextServiceBase
    from .benchmark.test_container_memory import TestContainerMemory, get_memory_usage
    from .benchmark.test_container_performance import TestContainerPerformance
    from .benchmark.test_refactor_nesting_performance import TestPerformanceBenchmarks
    from .conftest import (
        FunctionalExternalService,
        assert_rejects,
        assert_validates,
        clean_container,
        empty_strings,
        flext_result_failure,
        flext_result_success,
        invalid_hostnames,
        invalid_port_numbers,
        invalid_uris,
        mock_external_service,
        out_of_range,
        parser_scenarios,
        reliability_scenarios,
        reset_global_container,
        sample_data,
        temp_dir,
        temp_directory,
        temp_file,
        test_context,
        valid_hostnames,
        valid_port_numbers,
        valid_ranges,
        valid_strings,
        valid_uris,
        validation_scenarios,
        whitespace_strings,
    )
    from .constants import TestsFlextConstants, c
    from .helpers.factories import (
        FailingService,
        FailingServiceAuto,
        FailingServiceAutoFactory,
        FailingServiceFactory,
        GenericModelFactory,
        GetUserService,
        GetUserService as s,
        GetUserServiceAuto,
        GetUserServiceAutoFactory,
        GetUserServiceFactory,
        ServiceFactoryRegistry,
        ServiceTestCase,
        ServiceTestCaseFactory,
        ServiceTestCases,
        TestDataGenerators,
        User,
        UserFactory,
        ValidatingService,
        ValidatingServiceAuto,
        ValidatingServiceAutoFactory,
        ValidatingServiceFactory,
        reset_all_factories,
    )
    from .helpers.scenarios import TestHelperScenarios
    from .integration import patterns as patterns
    from .integration.patterns.test_advanced_patterns import (
        TestAdvancedPatterns,
        TestFunction,
    )
    from .integration.patterns.test_architectural_patterns import (
        TestArchitecturalPatterns,
    )
    from .integration.patterns.test_patterns_commands import TestPatternsCommands
    from .integration.patterns.test_patterns_logging import TestPatternsLogging
    from .integration.patterns.test_patterns_testing import TestPatternsTesting
    from .integration.test_config_integration import (
        TestFlextSettingsSingletonIntegration,
    )
    from .integration.test_infra_integration import TestInfraIntegration
    from .integration.test_integration import TestLibraryIntegration
    from .integration.test_migration_validation import TestMigrationValidation
    from .integration.test_refactor_nesting_file import (
        pytestmark,
        test_class_nesting_refactor_single_file_end_to_end,
    )
    from .integration.test_refactor_nesting_idempotency import TestIdempotency
    from .integration.test_refactor_nesting_project import TestProjectLevelRefactor
    from .integration.test_refactor_nesting_workspace import TestWorkspaceLevelRefactor
    from .integration.test_refactor_policy_mro import TestRefactorPolicyMRO
    from .integration.test_service import TestService
    from .integration.test_system import TestCompleteFlextSystemIntegration
    from .models import TestsFlextModels, m
    from .protocols import TestsFlextProtocols, p
    from .test_documented_patterns import TestDocumentedPatterns
    from .test_service_result_property import TestServiceResultProperty
    from .test_utils import (
        FlextTestResult,
        FlextTestResultCo,
        TestUtils,
        assertion_helpers,
        fixture_factory,
        test_data_factory,
    )
    from .typings import T, T_co, T_contra, TestsFlextTypes, t
    from .unit import contracts as contracts, flext_tests as flext_tests
    from .unit.conftest_infra import (
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
    from .unit.contracts.text_contract import TextUtilityContract
    from .unit.flext_tests.test_builders import TestFlextTestsBuilders
    from .unit.flext_tests.test_docker import TestDocker
    from .unit.flext_tests.test_domains import TestFlextTestsDomains
    from .unit.flext_tests.test_factories import TestFactoriesHelpers
    from .unit.flext_tests.test_files import TestFlextTestsFiles
    from .unit.flext_tests.test_matchers import TestFlextTestsMatchers
    from .unit.flext_tests.test_utilities import TestUtilities
    from .unit.protocols import FlextProtocols
    from .unit.test_args_coverage_100 import TestFlextUtilitiesArgs
    from .unit.test_automated_architecture import TestAutomatedArchitecture
    from .unit.test_automated_container import TestAutomatedFlextContainer
    from .unit.test_automated_context import TestAutomatedFlextContext
    from .unit.test_automated_decorators import (
        TestAutomatedFlextDecorators,
        TestAutomatedFlextDecorators as d,
    )
    from .unit.test_automated_dispatcher import TestAutomatedFlextDispatcher
    from .unit.test_automated_exceptions import (
        TestAutomatedExceptions,
        TestAutomatedExceptions as e,
    )
    from .unit.test_automated_handlers import (
        TestAutomatedFlextHandlers,
        TestAutomatedFlextHandlers as h,
    )
    from .unit.test_automated_loggings import TestAutomatedFlextLogger
    from .unit.test_automated_mixins import (
        TestAutomatedFlextMixins,
        TestAutomatedFlextMixins as x,
    )
    from .unit.test_automated_registry import TestAutomatedFlextRegistry
    from .unit.test_automated_result import (
        TestAutomatedResult,
        TestAutomatedResult as r,
    )
    from .unit.test_automated_runtime import TestAutomatedFlextRuntime
    from .unit.test_automated_service import TestAutomatedFlextService
    from .unit.test_automated_settings import TestAutomatedFlextSettings
    from .unit.test_automated_utilities import TestAutomatedFlextUtilities
    from .unit.test_collection_utilities_coverage_100 import (
        TestCollectionUtilitiesCoverage,
    )
    from .unit.test_collections_coverage_100 import (
        TestFlextModelsCollectionsCategories,
        TestFlextModelsCollectionsOptions,
        TestFlextModelsCollectionsResults,
        TestFlextModelsCollectionsSettings,
        TestFlextModelsCollectionsStatistics,
    )
    from .unit.test_config import TestFlextSettings
    from .unit.test_constants import TestConstants
    from .unit.test_constants_full_coverage import (
        test_constants_auto_enum_and_bimapping_paths,
    )
    from .unit.test_container import TestFlextContainer
    from .unit.test_container_full_coverage import TestContainerFullCoverage
    from .unit.test_context import TestFlextContext
    from .unit.test_context_coverage_100 import TestContext100Coverage
    from .unit.test_context_full_coverage import (
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
    from .unit.test_coverage_76_lines import TestCoverage76Lines
    from .unit.test_coverage_context import TestCoverageContext
    from .unit.test_coverage_exceptions import (
        TestExceptionContext,
        TestExceptionEdgeCases,
        TestExceptionFactory,
        TestExceptionIntegration,
        TestExceptionLogging,
        TestExceptionMetrics,
        TestExceptionProperties,
        TestExceptionSerialization,
        TestFlextExceptionsHierarchy,
        TestHierarchicalExceptionSystem,
    )
    from .unit.test_coverage_loggings import TestCoverageLoggings
    from .unit.test_coverage_models import (
        TestAggregateRoots,
        TestCommands,
        TestDomainEvents,
        TestEntities,
        TestMetadata,
        TestModelIntegration,
        TestModelSerialization,
        TestModelValidation,
        TestQueries,
        TestValues,
    )
    from .unit.test_decorators import TestFlextDecorators
    from .unit.test_decorators_discovery_full_coverage import (
        TestDecoratorsDiscoveryFullCoverage,
    )
    from .unit.test_decorators_full_coverage import TestDecoratorsFullCoverage
    from .unit.test_deprecation_warnings import TestDeprecationWarnings
    from .unit.test_di_incremental import Provide, TestDIIncremental, inject
    from .unit.test_di_services_access import TestDiServicesAccess
    from .unit.test_dispatcher_di import TestDispatcherDI
    from .unit.test_dispatcher_full_coverage import TestDispatcherFullCoverage
    from .unit.test_dispatcher_minimal import TestDispatcherMinimal
    from .unit.test_dispatcher_reliability import (
        test_circuit_breaker_transitions_and_metrics,
        test_rate_limiter_blocks_then_recovers,
        test_rate_limiter_jitter_application,
        test_retry_policy_behavior,
    )
    from .unit.test_dispatcher_reliability_full_coverage import (
        test_dispatcher_reliability_branch_paths,
    )
    from .unit.test_dispatcher_timeout_coverage_100 import (
        TestDispatcherTimeoutCoverage100,
    )
    from .unit.test_entity_coverage import TestEntityCoverageEdgeCases
    from .unit.test_enum_utilities_coverage_100 import TestEnumUtilitiesCoverage
    from .unit.test_exceptions import Teste
    from .unit.test_exceptions_full_coverage import (
        test_authentication_error_normalizes_extra_kwargs_into_context,
        test_base_error_normalize_metadata_merges_existing_metadata_model,
        test_exceptions_uncovered_metadata_paths,
        test_merge_metadata_context_paths,
        test_not_found_error_correlation_id_selection_and_extra_kwargs,
    )
    from .unit.test_final_75_percent_push import TestFinal75PercentPush
    from .unit.test_handler_decorator_discovery import (
        TestHandlerDecoratorMetadata,
        TestHandlerDiscoveryClass,
        TestHandlerDiscoveryEdgeCases,
        TestHandlerDiscoveryIntegration,
        TestHandlerDiscoveryModule,
        TestHandlerDiscoveryServiceIntegration,
    )
    from .unit.test_handlers import TestFlextHandlers
    from .unit.test_handlers_full_coverage import (
        TestHandlersFullCoverage,
        handlers_module,
    )
    from .unit.test_loggings_error_paths_coverage import TestLoggingsErrorPaths
    from .unit.test_loggings_full_coverage import TestModule
    from .unit.test_loggings_strict_returns import TestLoggingsStrictReturns
    from .unit.test_mixins import TestFlextMixinsNestedClasses
    from .unit.test_mixins_full_coverage import TestMixinsFullCoverage
    from .unit.test_models import TestModels
    from .unit.test_models_79_coverage import (
        TestFlextModelsAggregateRoot,
        TestFlextModelsCommand,
        TestFlextModelsDomainEvent,
        TestFlextModelsEdgeCases,
        TestFlextModelsEntity,
        TestFlextModelsIntegration,
        TestFlextModelsQuery,
        TestFlextModelsValue,
    )
    from .unit.test_models_base_full_coverage import TestModelsBaseFullCoverage
    from .unit.test_models_collections_full_coverage import (
        TestModelsCollectionsFullCoverage,
    )
    from .unit.test_models_container import TestFlextModelsContainer
    from .unit.test_models_container_full_coverage import (
        test_container_resource_registration_metadata_normalized,
    )
    from .unit.test_models_context_full_coverage import (
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
    from .unit.test_models_cqrs_full_coverage import (
        test_command_pagination_limit,
        test_cqrs_query_resolve_deeper_and_int_pagination,
        test_flext_message_type_alias_adapter,
        test_handler_builder_fluent_methods,
        test_query_resolve_pagination_wrapper_and_fallback,
        test_query_validate_pagination_dict_and_default,
    )
    from .unit.test_models_entity_full_coverage import (
        test_entity_comparable_map_and_bulk_validation_paths,
    )
    from .unit.test_models_generic_full_coverage import (
        test_canonical_aliases_are_available,
        test_conversion_add_converted_and_error_metadata_append_paths,
        test_conversion_add_skipped_skip_reason_upsert_paths,
        test_conversion_add_warning_metadata_append_paths,
        test_conversion_start_and_complete_methods,
        test_operation_progress_start_operation_sets_runtime_fields,
    )
    from .unit.test_models_handler_full_coverage import (
        test_models_handler_branches,
        test_models_handler_uncovered_mode_and_reset_paths,
    )
    from .unit.test_models_service_full_coverage import (
        test_service_request_timeout_post_validator_messages,
        test_service_request_timeout_validator_branches,
    )
    from .unit.test_models_settings_full_coverage import (
        test_models_settings_branch_paths,
        test_models_settings_context_validator_and_non_standard_status_input,
    )
    from .unit.test_models_validation_full_coverage import (
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
    from .unit.test_namespace_validator import TestFlextInfraNamespaceValidator
    from .unit.test_pagination_coverage_100 import (
        ExtractPageParamsScenario,
        PaginationScenarios,
        PreparePaginationDataScenario,
        TestPaginationCoverage100,
        ValidatePaginationParamsScenario,
    )
    from .unit.test_phase2_coverage_final import TestPhase2CoverageFinal
    from .unit.test_protocols import TestFlextProtocols
    from .unit.test_refactor_cli_models_workflow import (
        test_centralize_pydantic_cli_outputs_extended_metrics,
        test_namespace_enforce_cli_fails_on_manual_protocol_violation,
        test_ultrawork_models_cli_runs_dry_run_copy,
    )
    from .unit.test_refactor_migrate_to_class_mro import (
        test_discover_project_roots_without_nested_git_dirs,
        test_migrate_protocols_rewrites_references_with_p_alias,
        test_migrate_to_mro_inlines_alias_constant_into_constants_class,
        test_migrate_to_mro_moves_constant_and_rewrites_reference,
        test_migrate_to_mro_moves_manual_uppercase_assignment,
        test_migrate_to_mro_normalizes_facade_alias_to_c,
        test_migrate_to_mro_rejects_unknown_target,
        test_migrate_typings_rewrites_references_with_t_alias,
        test_mro_scanner_includes_constants_variants_in_all_scopes,
        test_refactor_utilities_iter_python_files_includes_examples_and_scripts,
    )
    from .unit.test_refactor_namespace_enforcer import (
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
    from .unit.test_refactor_policy_family_rules import (
        test_decorators_family_blocks_dispatcher_target,
        test_dispatcher_family_blocks_models_target,
        test_helper_consolidation_is_prechecked,
        test_models_family_blocks_utilities_target,
        test_runtime_family_blocks_non_runtime_target,
        test_utilities_family_allows_utilities_target,
    )
    from .unit.test_refactor_pydantic_centralizer import (
        test_centralizer_converts_typed_dict_factory_to_model,
        test_centralizer_does_not_touch_settings_module,
        test_centralizer_moves_dict_alias_in_typings_without_keyword_name,
        test_centralizer_moves_manual_type_aliases_to_models_file,
    )
    from .unit.test_registry import TestFlextRegistry
    from .unit.test_registry_full_coverage import (
        test_create_auto_discover_and_mode_mapping,
        test_execute_and_register_handler_failure_paths,
        test_get_plugin_and_register_metadata_and_list_items_exception,
        test_summary_error_paths_and_bindings_failures,
        test_summary_properties_and_subclass_storage_reset,
    )
    from .unit.test_result import Testr
    from .unit.test_result_additional import (
        test_create_from_callable_and_repr,
        test_flow_through_short_circuits_on_failure,
        test_map_error_identity_and_transform,
        test_ok_accepts_none,
        test_with_resource_cleanup_runs,
    )
    from .unit.test_result_coverage_100 import TestrCoverage
    from .unit.test_result_exception_carrying import (
        TestAltPropagatesException,
        TestCreateFromCallableCarriesException,
        TestErrorOrPatternUnchanged,
        TestExceptionPropertyAccess,
        TestFailNoExceptionBackwardCompat,
        TestFailWithException,
        TestFlatMapPropagatesException,
        TestFromValidationCarriesException,
        TestLashPropagatesException,
        TestMapPropagatesException,
        TestMonadicOperationsUnchanged,
        TestOkNoneGuardStillRaises,
        TestSafeCarriesException,
        TestTraversePropagatesException,
    )
    from .unit.test_result_full_coverage import (
        test_from_validation_and_to_model_paths,
        test_init_fallback_and_lazy_returns_result_property,
        test_lash_runtime_result_paths,
        test_map_flat_map_and_then_paths,
        test_recover_tap_and_tap_error_paths,
        test_type_guards_result,
        test_validation_like_error_structure,
    )
    from .unit.test_runtime import TestFlextRuntime
    from .unit.test_runtime_coverage_100 import TestRuntimeCoverage100
    from .unit.test_runtime_full_coverage import (
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
    from .unit.test_service import TestsCore
    from .unit.test_service_additional import (
        RuntimeCloneService,
        test_get_service_info,
        test_is_valid_handles_validation_exception,
        test_result_property_raises_on_failure,
    )
    from .unit.test_service_bootstrap import TestServiceBootstrap
    from .unit.test_service_coverage_100 import TestService100Coverage
    from .unit.test_service_full_coverage import TestServiceFullCoverage
    from .unit.test_settings_full_coverage import (
        test_settings_materialize_and_context_overrides,
    )
    from .unit.test_transformer_class_nesting import (
        test_class_nesting_appends_to_existing_namespace_and_removes_pass,
        test_class_nesting_keeps_unmapped_top_level_classes,
        test_class_nesting_moves_top_level_class_into_new_namespace,
    )
    from .unit.test_transformer_helper_consolidation import (
        TestHelperConsolidationTransformer,
    )
    from .unit.test_transformer_nested_class_propagation import (
        NestedClassPropagationTransformer,
        test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage,
        test_nested_class_propagation_updates_import_annotations_and_calls,
    )
    from .unit.test_typings import TestTypings
    from .unit.test_typings_full_coverage import TestTypingsFullCoverage
    from .unit.test_utilities import Testu
    from .unit.test_utilities_args_full_coverage import (
        UnknownHint,
        test_args_get_enum_params_annotated_unwrap_branch,
        test_args_get_enum_params_branches,
    )
    from .unit.test_utilities_cache_coverage_100 import (
        CacheScenarios,
        TestuCacheClearObjectCache,
        TestuCacheGenerateCacheKey,
        TestuCacheHasCacheAttributes,
        TestuCacheLogger,
        TestuCacheNormalizeComponent,
        TestuCacheSortDictKeys,
        TestuCacheSortKey,
    )
    from .unit.test_utilities_checker_full_coverage import (
        TestUtilitiesCheckerFullCoverage,
    )
    from .unit.test_utilities_collection_coverage_100 import (
        TestUtilitiesCollectionCoverage,
    )
    from .unit.test_utilities_collection_full_coverage import (
        TestUtilitiesCollectionFullCoverage,
    )
    from .unit.test_utilities_configuration_coverage_100 import (
        BadSingletonForTest,
        ConfigWithoutModelConfigForTest,
        DataclassConfigForTest,
        FailingOptionsForTest,
        OptionsModelForTest,
        SingletonWithoutGetGlobalForTest,
        SingletonWithoutModelDumpForTest,
        StrictOptionsForTest,
        TestConfigConstants,
        TestConfigModels,
        TestFlextUtilitiesConfiguration,
    )
    from .unit.test_utilities_configuration_full_coverage import (
        TestUtilitiesConfigurationFullCoverage,
    )
    from .unit.test_utilities_context_full_coverage import (
        TestUtilitiesContextFullCoverage,
    )
    from .unit.test_utilities_conversion_full_coverage import (
        test_conversion_string_and_join_paths,
    )
    from .unit.test_utilities_coverage import TestUtilitiesCoverage
    from .unit.test_utilities_data_mapper import TestUtilitiesDataMapper
    from .unit.test_utilities_deprecation_full_coverage import (
        test_deprecated_class_noop_init_branch,
    )
    from .unit.test_utilities_domain import (
        TestuDomain,
        create_compare_entities_cases,
        create_compare_value_objects_cases,
        create_hash_entity_cases,
        create_hash_value_object_cases,
        create_validate_entity_has_id_cases,
        create_validate_value_object_immutable_cases,
    )
    from .unit.test_utilities_domain_full_coverage import (
        TestUtilitiesDomainFullCoverage,
    )
    from .unit.test_utilities_enum_full_coverage import TestUtilitiesEnumFullCoverage
    from .unit.test_utilities_generators_full_coverage import (
        TestUtilitiesGeneratorsFullCoverage,
        generators_module,
        runtime_module,
    )
    from .unit.test_utilities_guards_full_coverage import (
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
    from .unit.test_utilities_mapper_coverage_100 import (
        SimpleObj,
        TestuMapperAccessors,
        TestuMapperAdvanced,
        TestuMapperBuild,
        TestuMapperConversions,
        TestuMapperExtract,
        TestuMapperUtils,
    )
    from .unit.test_utilities_mapper_full_coverage import (
        AttrObject,
        BadBool,
        BadMapping,
        BadString,
        ExplodingLenList,
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
    from .unit.test_utilities_model_full_coverage import (
        test_merge_defaults_and_dump_paths,
        test_normalize_to_pydantic_dict_and_value_branches,
        test_update_exception_path,
        test_update_success_path_returns_ok_result,
    )
    from .unit.test_utilities_pagination_full_coverage import (
        test_pagination_response_string_fallbacks,
    )
    from .unit.test_utilities_parser_full_coverage import (
        TestUtilitiesParserFullCoverage,
    )
    from .unit.test_utilities_reliability import TestFlextUtilitiesReliability
    from .unit.test_utilities_reliability_full_coverage import (
        test_utilities_reliability_branches,
        test_utilities_reliability_compose_returns_non_result_directly,
        test_utilities_reliability_uncovered_retry_compose_and_sequence_paths,
    )
    from .unit.test_utilities_string_parser import TestuStringParser
    from .unit.test_utilities_text_full_coverage import TestUtilitiesTextFullCoverage
    from .unit.test_utilities_type_checker_coverage_100 import (
        TestuTypeChecker,
        TMessage,
    )
    from .unit.test_utilities_type_guards_coverage_100 import (
        TestuTypeGuardsIsDictNonEmpty,
        TestuTypeGuardsIsListNonEmpty,
        TestuTypeGuardsIsStringNonEmpty,
        TestuTypeGuardsNormalizeToMetadata,
        TypeGuardsScenarios,
    )
    from .unit.test_version import TestFlextVersion
    from .utilities import TestsFlextUtilities, u

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AttrObject": ("tests.unit.test_utilities_mapper_full_coverage", "AttrObject"),
    "BadBool": ("tests.unit.test_utilities_mapper_full_coverage", "BadBool"),
    "BadMapping": ("tests.unit.test_utilities_mapper_full_coverage", "BadMapping"),
    "BadSingletonForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "BadSingletonForTest",
    ),
    "BadString": ("tests.unit.test_utilities_mapper_full_coverage", "BadString"),
    "CacheScenarios": (
        "tests.unit.test_utilities_cache_coverage_100",
        "CacheScenarios",
    ),
    "ConfigWithoutModelConfigForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "ConfigWithoutModelConfigForTest",
    ),
    "DataclassConfigForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "DataclassConfigForTest",
    ),
    "ExplodingLenList": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "ExplodingLenList",
    ),
    "ExtractPageParamsScenario": (
        "tests.unit.test_pagination_coverage_100",
        "ExtractPageParamsScenario",
    ),
    "FailingOptionsForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "FailingOptionsForTest",
    ),
    "FailingService": ("tests.helpers.factories", "FailingService"),
    "FailingServiceAuto": ("tests.helpers.factories", "FailingServiceAuto"),
    "FailingServiceAutoFactory": (
        "tests.helpers.factories",
        "FailingServiceAutoFactory",
    ),
    "FailingServiceFactory": ("tests.helpers.factories", "FailingServiceFactory"),
    "FlextProtocols": ("tests.unit.protocols", "FlextProtocols"),
    "FlextTestResult": ("tests.test_utils", "FlextTestResult"),
    "FlextTestResultCo": ("tests.test_utils", "FlextTestResultCo"),
    "FunctionalExternalService": ("tests.conftest", "FunctionalExternalService"),
    "GenericModelFactory": ("tests.helpers.factories", "GenericModelFactory"),
    "GetUserService": ("tests.helpers.factories", "GetUserService"),
    "GetUserServiceAuto": ("tests.helpers.factories", "GetUserServiceAuto"),
    "GetUserServiceAutoFactory": (
        "tests.helpers.factories",
        "GetUserServiceAutoFactory",
    ),
    "GetUserServiceFactory": ("tests.helpers.factories", "GetUserServiceFactory"),
    "NestedClassPropagationTransformer": (
        "tests.unit.test_transformer_nested_class_propagation",
        "NestedClassPropagationTransformer",
    ),
    "OptionsModelForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "OptionsModelForTest",
    ),
    "PaginationScenarios": (
        "tests.unit.test_pagination_coverage_100",
        "PaginationScenarios",
    ),
    "PreparePaginationDataScenario": (
        "tests.unit.test_pagination_coverage_100",
        "PreparePaginationDataScenario",
    ),
    "Provide": ("tests.unit.test_di_incremental", "Provide"),
    "RuntimeCloneService": (
        "tests.unit.test_service_additional",
        "RuntimeCloneService",
    ),
    "ServiceFactoryRegistry": ("tests.helpers.factories", "ServiceFactoryRegistry"),
    "ServiceTestCase": ("tests.helpers.factories", "ServiceTestCase"),
    "ServiceTestCaseFactory": ("tests.helpers.factories", "ServiceTestCaseFactory"),
    "ServiceTestCases": ("tests.helpers.factories", "ServiceTestCases"),
    "SimpleObj": ("tests.unit.test_utilities_mapper_coverage_100", "SimpleObj"),
    "SingletonWithoutGetGlobalForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "SingletonWithoutGetGlobalForTest",
    ),
    "SingletonWithoutModelDumpForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "SingletonWithoutModelDumpForTest",
    ),
    "StrictOptionsForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "StrictOptionsForTest",
    ),
    "T": ("tests.typings", "T"),
    "TMessage": ("tests.unit.test_utilities_type_checker_coverage_100", "TMessage"),
    "T_co": ("tests.typings", "T_co"),
    "T_contra": ("tests.typings", "T_contra"),
    "TestAdvancedPatterns": (
        "tests.integration.patterns.test_advanced_patterns",
        "TestAdvancedPatterns",
    ),
    "TestAggregateRoots": ("tests.unit.test_coverage_models", "TestAggregateRoots"),
    "TestAltPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestAltPropagatesException",
    ),
    "TestArchitecturalPatterns": (
        "tests.integration.patterns.test_architectural_patterns",
        "TestArchitecturalPatterns",
    ),
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
    "TestCommands": ("tests.unit.test_coverage_models", "TestCommands"),
    "TestCompleteFlextSystemIntegration": (
        "tests.integration.test_system",
        "TestCompleteFlextSystemIntegration",
    ),
    "TestConfigConstants": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "TestConfigConstants",
    ),
    "TestConfigModels": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "TestConfigModels",
    ),
    "TestConstants": ("tests.unit.test_constants", "TestConstants"),
    "TestContainerFullCoverage": (
        "tests.unit.test_container_full_coverage",
        "TestContainerFullCoverage",
    ),
    "TestContainerMemory": (
        "tests.benchmark.test_container_memory",
        "TestContainerMemory",
    ),
    "TestContainerPerformance": (
        "tests.benchmark.test_container_performance",
        "TestContainerPerformance",
    ),
    "TestContext100Coverage": (
        "tests.unit.test_context_coverage_100",
        "TestContext100Coverage",
    ),
    "TestCoverage76Lines": ("tests.unit.test_coverage_76_lines", "TestCoverage76Lines"),
    "TestCoverageContext": ("tests.unit.test_coverage_context", "TestCoverageContext"),
    "TestCoverageLoggings": (
        "tests.unit.test_coverage_loggings",
        "TestCoverageLoggings",
    ),
    "TestCreateFromCallableCarriesException": (
        "tests.unit.test_result_exception_carrying",
        "TestCreateFromCallableCarriesException",
    ),
    "TestDIIncremental": ("tests.unit.test_di_incremental", "TestDIIncremental"),
    "TestDataGenerators": ("tests.helpers.factories", "TestDataGenerators"),
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
    "TestDocumentedPatterns": (
        "tests.test_documented_patterns",
        "TestDocumentedPatterns",
    ),
    "TestDomainEvents": ("tests.unit.test_coverage_models", "TestDomainEvents"),
    "TestEntities": ("tests.unit.test_coverage_models", "TestEntities"),
    "TestEntityCoverageEdgeCases": (
        "tests.unit.test_entity_coverage",
        "TestEntityCoverageEdgeCases",
    ),
    "TestEnumUtilitiesCoverage": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestEnumUtilitiesCoverage",
    ),
    "TestErrorOrPatternUnchanged": (
        "tests.unit.test_result_exception_carrying",
        "TestErrorOrPatternUnchanged",
    ),
    "TestExceptionContext": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionContext",
    ),
    "TestExceptionEdgeCases": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionEdgeCases",
    ),
    "TestExceptionFactory": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionFactory",
    ),
    "TestExceptionIntegration": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionIntegration",
    ),
    "TestExceptionLogging": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionLogging",
    ),
    "TestExceptionMetrics": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionMetrics",
    ),
    "TestExceptionProperties": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionProperties",
    ),
    "TestExceptionPropertyAccess": (
        "tests.unit.test_result_exception_carrying",
        "TestExceptionPropertyAccess",
    ),
    "TestExceptionSerialization": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionSerialization",
    ),
    "TestFactoriesHelpers": (
        "tests.unit.flext_tests.test_factories",
        "TestFactoriesHelpers",
    ),
    "TestFailNoExceptionBackwardCompat": (
        "tests.unit.test_result_exception_carrying",
        "TestFailNoExceptionBackwardCompat",
    ),
    "TestFailWithException": (
        "tests.unit.test_result_exception_carrying",
        "TestFailWithException",
    ),
    "TestFinal75PercentPush": (
        "tests.unit.test_final_75_percent_push",
        "TestFinal75PercentPush",
    ),
    "TestFlatMapPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestFlatMapPropagatesException",
    ),
    "TestFlextContainer": ("tests.unit.test_container", "TestFlextContainer"),
    "TestFlextContext": ("tests.unit.test_context", "TestFlextContext"),
    "TestFlextDecorators": ("tests.unit.test_decorators", "TestFlextDecorators"),
    "TestFlextExceptionsHierarchy": (
        "tests.unit.test_coverage_exceptions",
        "TestFlextExceptionsHierarchy",
    ),
    "TestFlextHandlers": ("tests.unit.test_handlers", "TestFlextHandlers"),
    "TestFlextInfraNamespaceValidator": (
        "tests.unit.test_namespace_validator",
        "TestFlextInfraNamespaceValidator",
    ),
    "TestFlextMixinsNestedClasses": (
        "tests.unit.test_mixins",
        "TestFlextMixinsNestedClasses",
    ),
    "TestFlextModelsAggregateRoot": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsAggregateRoot",
    ),
    "TestFlextModelsCollectionsCategories": (
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsCategories",
    ),
    "TestFlextModelsCollectionsOptions": (
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsOptions",
    ),
    "TestFlextModelsCollectionsResults": (
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsResults",
    ),
    "TestFlextModelsCollectionsSettings": (
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsSettings",
    ),
    "TestFlextModelsCollectionsStatistics": (
        "tests.unit.test_collections_coverage_100",
        "TestFlextModelsCollectionsStatistics",
    ),
    "TestFlextModelsCommand": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsCommand",
    ),
    "TestFlextModelsContainer": (
        "tests.unit.test_models_container",
        "TestFlextModelsContainer",
    ),
    "TestFlextModelsDomainEvent": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsDomainEvent",
    ),
    "TestFlextModelsEdgeCases": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsEdgeCases",
    ),
    "TestFlextModelsEntity": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsEntity",
    ),
    "TestFlextModelsIntegration": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsIntegration",
    ),
    "TestFlextModelsQuery": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsQuery",
    ),
    "TestFlextModelsValue": (
        "tests.unit.test_models_79_coverage",
        "TestFlextModelsValue",
    ),
    "TestFlextProtocols": ("tests.unit.test_protocols", "TestFlextProtocols"),
    "TestFlextRegistry": ("tests.unit.test_registry", "TestFlextRegistry"),
    "TestFlextRuntime": ("tests.unit.test_runtime", "TestFlextRuntime"),
    "TestFlextSettings": ("tests.unit.test_config", "TestFlextSettings"),
    "TestFlextSettingsSingletonIntegration": (
        "tests.integration.test_config_integration",
        "TestFlextSettingsSingletonIntegration",
    ),
    "TestFlextTestsBuilders": (
        "tests.unit.flext_tests.test_builders",
        "TestFlextTestsBuilders",
    ),
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
    "TestFromValidationCarriesException": (
        "tests.unit.test_result_exception_carrying",
        "TestFromValidationCarriesException",
    ),
    "TestFunction": (
        "tests.integration.patterns.test_advanced_patterns",
        "TestFunction",
    ),
    "TestHandlerDecoratorMetadata": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDecoratorMetadata",
    ),
    "TestHandlerDiscoveryClass": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDiscoveryClass",
    ),
    "TestHandlerDiscoveryEdgeCases": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDiscoveryEdgeCases",
    ),
    "TestHandlerDiscoveryIntegration": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDiscoveryIntegration",
    ),
    "TestHandlerDiscoveryModule": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDiscoveryModule",
    ),
    "TestHandlerDiscoveryServiceIntegration": (
        "tests.unit.test_handler_decorator_discovery",
        "TestHandlerDiscoveryServiceIntegration",
    ),
    "TestHandlersFullCoverage": (
        "tests.unit.test_handlers_full_coverage",
        "TestHandlersFullCoverage",
    ),
    "TestHelperConsolidationTransformer": (
        "tests.unit.test_transformer_helper_consolidation",
        "TestHelperConsolidationTransformer",
    ),
    "TestHelperScenarios": ("tests.helpers.scenarios", "TestHelperScenarios"),
    "TestHierarchicalExceptionSystem": (
        "tests.unit.test_coverage_exceptions",
        "TestHierarchicalExceptionSystem",
    ),
    "TestIdempotency": (
        "tests.integration.test_refactor_nesting_idempotency",
        "TestIdempotency",
    ),
    "TestInfraIntegration": (
        "tests.integration.test_infra_integration",
        "TestInfraIntegration",
    ),
    "TestLashPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestLashPropagatesException",
    ),
    "TestLibraryIntegration": (
        "tests.integration.test_integration",
        "TestLibraryIntegration",
    ),
    "TestLoggingsErrorPaths": (
        "tests.unit.test_loggings_error_paths_coverage",
        "TestLoggingsErrorPaths",
    ),
    "TestLoggingsStrictReturns": (
        "tests.unit.test_loggings_strict_returns",
        "TestLoggingsStrictReturns",
    ),
    "TestMapPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestMapPropagatesException",
    ),
    "TestMetadata": ("tests.unit.test_coverage_models", "TestMetadata"),
    "TestMigrationValidation": (
        "tests.integration.test_migration_validation",
        "TestMigrationValidation",
    ),
    "TestMixinsFullCoverage": (
        "tests.unit.test_mixins_full_coverage",
        "TestMixinsFullCoverage",
    ),
    "TestModelIntegration": ("tests.unit.test_coverage_models", "TestModelIntegration"),
    "TestModelSerialization": (
        "tests.unit.test_coverage_models",
        "TestModelSerialization",
    ),
    "TestModelValidation": ("tests.unit.test_coverage_models", "TestModelValidation"),
    "TestModels": ("tests.unit.test_models", "TestModels"),
    "TestModelsBaseFullCoverage": (
        "tests.unit.test_models_base_full_coverage",
        "TestModelsBaseFullCoverage",
    ),
    "TestModelsCollectionsFullCoverage": (
        "tests.unit.test_models_collections_full_coverage",
        "TestModelsCollectionsFullCoverage",
    ),
    "TestModule": ("tests.unit.test_loggings_full_coverage", "TestModule"),
    "TestMonadicOperationsUnchanged": (
        "tests.unit.test_result_exception_carrying",
        "TestMonadicOperationsUnchanged",
    ),
    "TestOkNoneGuardStillRaises": (
        "tests.unit.test_result_exception_carrying",
        "TestOkNoneGuardStillRaises",
    ),
    "TestPaginationCoverage100": (
        "tests.unit.test_pagination_coverage_100",
        "TestPaginationCoverage100",
    ),
    "TestPatternsCommands": (
        "tests.integration.patterns.test_patterns_commands",
        "TestPatternsCommands",
    ),
    "TestPatternsLogging": (
        "tests.integration.patterns.test_patterns_logging",
        "TestPatternsLogging",
    ),
    "TestPatternsTesting": (
        "tests.integration.patterns.test_patterns_testing",
        "TestPatternsTesting",
    ),
    "TestPerformanceBenchmarks": (
        "tests.benchmark.test_refactor_nesting_performance",
        "TestPerformanceBenchmarks",
    ),
    "TestPhase2CoverageFinal": (
        "tests.unit.test_phase2_coverage_final",
        "TestPhase2CoverageFinal",
    ),
    "TestProjectLevelRefactor": (
        "tests.integration.test_refactor_nesting_project",
        "TestProjectLevelRefactor",
    ),
    "TestQueries": ("tests.unit.test_coverage_models", "TestQueries"),
    "TestRefactorPolicyMRO": (
        "tests.integration.test_refactor_policy_mro",
        "TestRefactorPolicyMRO",
    ),
    "TestRuntimeCoverage100": (
        "tests.unit.test_runtime_coverage_100",
        "TestRuntimeCoverage100",
    ),
    "TestSafeCarriesException": (
        "tests.unit.test_result_exception_carrying",
        "TestSafeCarriesException",
    ),
    "TestService": ("tests.integration.test_service", "TestService"),
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
    "TestServiceResultProperty": (
        "tests.test_service_result_property",
        "TestServiceResultProperty",
    ),
    "TestTraversePropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestTraversePropagatesException",
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
    "TestUtils": ("tests.test_utils", "TestUtils"),
    "TestValues": ("tests.unit.test_coverage_models", "TestValues"),
    "TestWorkspaceLevelRefactor": (
        "tests.integration.test_refactor_nesting_workspace",
        "TestWorkspaceLevelRefactor",
    ),
    "Teste": ("tests.unit.test_exceptions", "Teste"),
    "Testr": ("tests.unit.test_result", "Testr"),
    "TestrCoverage": ("tests.unit.test_result_coverage_100", "TestrCoverage"),
    "TestsCore": ("tests.unit.test_service", "TestsCore"),
    "TestsFlextConstants": ("tests.constants", "TestsFlextConstants"),
    "TestsFlextModels": ("tests.models", "TestsFlextModels"),
    "TestsFlextProtocols": ("tests.protocols", "TestsFlextProtocols"),
    "TestsFlextServiceBase": ("tests.base", "TestsFlextServiceBase"),
    "TestsFlextTypes": ("tests.typings", "TestsFlextTypes"),
    "TestsFlextUtilities": ("tests.utilities", "TestsFlextUtilities"),
    "Testu": ("tests.unit.test_utilities", "Testu"),
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
    "TestuTypeGuardsIsDictNonEmpty": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TestuTypeGuardsIsDictNonEmpty",
    ),
    "TestuTypeGuardsIsListNonEmpty": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TestuTypeGuardsIsListNonEmpty",
    ),
    "TestuTypeGuardsIsStringNonEmpty": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TestuTypeGuardsIsStringNonEmpty",
    ),
    "TestuTypeGuardsNormalizeToMetadata": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TestuTypeGuardsNormalizeToMetadata",
    ),
    "TextUtilityContract": (
        "tests.unit.contracts.text_contract",
        "TextUtilityContract",
    ),
    "TypeGuardsScenarios": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TypeGuardsScenarios",
    ),
    "UnknownHint": ("tests.unit.test_utilities_args_full_coverage", "UnknownHint"),
    "User": ("tests.helpers.factories", "User"),
    "UserFactory": ("tests.helpers.factories", "UserFactory"),
    "ValidatePaginationParamsScenario": (
        "tests.unit.test_pagination_coverage_100",
        "ValidatePaginationParamsScenario",
    ),
    "ValidatingService": ("tests.helpers.factories", "ValidatingService"),
    "ValidatingServiceAuto": ("tests.helpers.factories", "ValidatingServiceAuto"),
    "ValidatingServiceAutoFactory": (
        "tests.helpers.factories",
        "ValidatingServiceAutoFactory",
    ),
    "ValidatingServiceFactory": ("tests.helpers.factories", "ValidatingServiceFactory"),
    "assert_rejects": ("tests.conftest", "assert_rejects"),
    "assert_validates": ("tests.conftest", "assert_validates"),
    "assertion_helpers": ("tests.test_utils", "assertion_helpers"),
    "benchmark": ("tests.benchmark", ""),
    "c": ("tests.constants", "c"),
    "clean_container": ("tests.conftest", "clean_container"),
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
    "d": ("tests.unit.test_automated_decorators", "TestAutomatedFlextDecorators"),
    "e": ("tests.unit.test_automated_exceptions", "TestAutomatedExceptions"),
    "empty_strings": ("tests.conftest", "empty_strings"),
    "fixture_factory": ("tests.test_utils", "fixture_factory"),
    "flext_result_failure": ("tests.conftest", "flext_result_failure"),
    "flext_result_success": ("tests.conftest", "flext_result_success"),
    "flext_tests": ("tests.unit.flext_tests", ""),
    "generators_module": (
        "tests.unit.test_utilities_generators_full_coverage",
        "generators_module",
    ),
    "get_memory_usage": ("tests.benchmark.test_container_memory", "get_memory_usage"),
    "h": ("tests.unit.test_automated_handlers", "TestAutomatedFlextHandlers"),
    "handlers_module": ("tests.unit.test_handlers_full_coverage", "handlers_module"),
    "helpers": ("tests.helpers", ""),
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
    "integration": ("tests.integration", ""),
    "invalid_hostnames": ("tests.conftest", "invalid_hostnames"),
    "invalid_port_numbers": ("tests.conftest", "invalid_port_numbers"),
    "invalid_uris": ("tests.conftest", "invalid_uris"),
    "m": ("tests.models", "m"),
    "mapper": ("tests.unit.test_utilities_mapper_full_coverage", "mapper"),
    "mock_external_service": ("tests.conftest", "mock_external_service"),
    "out_of_range": ("tests.conftest", "out_of_range"),
    "p": ("tests.protocols", "p"),
    "parser_scenarios": ("tests.conftest", "parser_scenarios"),
    "patterns": ("tests.integration.patterns", ""),
    "pytestmark": ("tests.integration.test_refactor_nesting_file", "pytestmark"),
    "r": ("tests.unit.test_automated_result", "TestAutomatedResult"),
    "reliability_scenarios": ("tests.conftest", "reliability_scenarios"),
    "reset_all_factories": ("tests.helpers.factories", "reset_all_factories"),
    "reset_global_container": ("tests.conftest", "reset_global_container"),
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
    "s": ("tests.helpers.factories", "GetUserService"),
    "sample_data": ("tests.conftest", "sample_data"),
    "t": ("tests.typings", "t"),
    "temp_dir": ("tests.conftest", "temp_dir"),
    "temp_directory": ("tests.conftest", "temp_directory"),
    "temp_file": ("tests.conftest", "temp_file"),
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
    "test_centralizer_converts_typed_dict_factory_to_model": (
        "tests.unit.test_refactor_pydantic_centralizer",
        "test_centralizer_converts_typed_dict_factory_to_model",
    ),
    "test_centralizer_does_not_touch_settings_module": (
        "tests.unit.test_refactor_pydantic_centralizer",
        "test_centralizer_does_not_touch_settings_module",
    ),
    "test_centralizer_moves_dict_alias_in_typings_without_keyword_name": (
        "tests.unit.test_refactor_pydantic_centralizer",
        "test_centralizer_moves_dict_alias_in_typings_without_keyword_name",
    ),
    "test_centralizer_moves_manual_type_aliases_to_models_file": (
        "tests.unit.test_refactor_pydantic_centralizer",
        "test_centralizer_moves_manual_type_aliases_to_models_file",
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
    "test_class_nesting_refactor_single_file_end_to_end": (
        "tests.integration.test_refactor_nesting_file",
        "test_class_nesting_refactor_single_file_end_to_end",
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
    "test_context": ("tests.conftest", "test_context"),
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
    "test_data_factory": ("tests.test_utils", "test_data_factory"),
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
    "test_mro_scanner_includes_constants_variants_in_all_scopes": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_mro_scanner_includes_constants_variants_in_all_scopes",
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
    "u": ("tests.utilities", "u"),
    "unit": ("tests.unit", ""),
    "valid_hostnames": ("tests.conftest", "valid_hostnames"),
    "valid_port_numbers": ("tests.conftest", "valid_port_numbers"),
    "valid_ranges": ("tests.conftest", "valid_ranges"),
    "valid_strings": ("tests.conftest", "valid_strings"),
    "valid_uris": ("tests.conftest", "valid_uris"),
    "validation_scenarios": ("tests.conftest", "validation_scenarios"),
    "whitespace_strings": ("tests.conftest", "whitespace_strings"),
    "x": ("tests.unit.test_automated_mixins", "TestAutomatedFlextMixins"),
}

__all__ = [
    "AttrObject",
    "BadBool",
    "BadMapping",
    "BadSingletonForTest",
    "BadString",
    "CacheScenarios",
    "ConfigWithoutModelConfigForTest",
    "DataclassConfigForTest",
    "ExplodingLenList",
    "ExtractPageParamsScenario",
    "FailingOptionsForTest",
    "FailingService",
    "FailingServiceAuto",
    "FailingServiceAutoFactory",
    "FailingServiceFactory",
    "FlextProtocols",
    "FlextTestResult",
    "FlextTestResultCo",
    "FunctionalExternalService",
    "GenericModelFactory",
    "GetUserService",
    "GetUserServiceAuto",
    "GetUserServiceAutoFactory",
    "GetUserServiceFactory",
    "NestedClassPropagationTransformer",
    "OptionsModelForTest",
    "PaginationScenarios",
    "PreparePaginationDataScenario",
    "Provide",
    "RuntimeCloneService",
    "ServiceFactoryRegistry",
    "ServiceTestCase",
    "ServiceTestCaseFactory",
    "ServiceTestCases",
    "SimpleObj",
    "SingletonWithoutGetGlobalForTest",
    "SingletonWithoutModelDumpForTest",
    "StrictOptionsForTest",
    "T",
    "TMessage",
    "T_co",
    "T_contra",
    "TestAdvancedPatterns",
    "TestAggregateRoots",
    "TestAltPropagatesException",
    "TestArchitecturalPatterns",
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
    "TestCommands",
    "TestCompleteFlextSystemIntegration",
    "TestConfigConstants",
    "TestConfigModels",
    "TestConstants",
    "TestContainerFullCoverage",
    "TestContainerMemory",
    "TestContainerPerformance",
    "TestContext100Coverage",
    "TestCoverage76Lines",
    "TestCoverageContext",
    "TestCoverageLoggings",
    "TestCreateFromCallableCarriesException",
    "TestDIIncremental",
    "TestDataGenerators",
    "TestDecoratorsDiscoveryFullCoverage",
    "TestDecoratorsFullCoverage",
    "TestDeprecationWarnings",
    "TestDiServicesAccess",
    "TestDispatcherDI",
    "TestDispatcherFullCoverage",
    "TestDispatcherMinimal",
    "TestDispatcherTimeoutCoverage100",
    "TestDocker",
    "TestDocumentedPatterns",
    "TestDomainEvents",
    "TestEntities",
    "TestEntityCoverageEdgeCases",
    "TestEnumUtilitiesCoverage",
    "TestErrorOrPatternUnchanged",
    "TestExceptionContext",
    "TestExceptionEdgeCases",
    "TestExceptionFactory",
    "TestExceptionIntegration",
    "TestExceptionLogging",
    "TestExceptionMetrics",
    "TestExceptionProperties",
    "TestExceptionPropertyAccess",
    "TestExceptionSerialization",
    "TestFactoriesHelpers",
    "TestFailNoExceptionBackwardCompat",
    "TestFailWithException",
    "TestFinal75PercentPush",
    "TestFlatMapPropagatesException",
    "TestFlextContainer",
    "TestFlextContext",
    "TestFlextDecorators",
    "TestFlextExceptionsHierarchy",
    "TestFlextHandlers",
    "TestFlextInfraNamespaceValidator",
    "TestFlextMixinsNestedClasses",
    "TestFlextModelsAggregateRoot",
    "TestFlextModelsCollectionsCategories",
    "TestFlextModelsCollectionsOptions",
    "TestFlextModelsCollectionsResults",
    "TestFlextModelsCollectionsSettings",
    "TestFlextModelsCollectionsStatistics",
    "TestFlextModelsCommand",
    "TestFlextModelsContainer",
    "TestFlextModelsDomainEvent",
    "TestFlextModelsEdgeCases",
    "TestFlextModelsEntity",
    "TestFlextModelsIntegration",
    "TestFlextModelsQuery",
    "TestFlextModelsValue",
    "TestFlextProtocols",
    "TestFlextRegistry",
    "TestFlextRuntime",
    "TestFlextSettings",
    "TestFlextSettingsSingletonIntegration",
    "TestFlextTestsBuilders",
    "TestFlextTestsDomains",
    "TestFlextTestsFiles",
    "TestFlextTestsMatchers",
    "TestFlextUtilitiesArgs",
    "TestFlextUtilitiesConfiguration",
    "TestFlextUtilitiesReliability",
    "TestFlextVersion",
    "TestFromValidationCarriesException",
    "TestFunction",
    "TestHandlerDecoratorMetadata",
    "TestHandlerDiscoveryClass",
    "TestHandlerDiscoveryEdgeCases",
    "TestHandlerDiscoveryIntegration",
    "TestHandlerDiscoveryModule",
    "TestHandlerDiscoveryServiceIntegration",
    "TestHandlersFullCoverage",
    "TestHelperConsolidationTransformer",
    "TestHelperScenarios",
    "TestHierarchicalExceptionSystem",
    "TestIdempotency",
    "TestInfraIntegration",
    "TestLashPropagatesException",
    "TestLibraryIntegration",
    "TestLoggingsErrorPaths",
    "TestLoggingsStrictReturns",
    "TestMapPropagatesException",
    "TestMetadata",
    "TestMigrationValidation",
    "TestMixinsFullCoverage",
    "TestModelIntegration",
    "TestModelSerialization",
    "TestModelValidation",
    "TestModels",
    "TestModelsBaseFullCoverage",
    "TestModelsCollectionsFullCoverage",
    "TestModule",
    "TestMonadicOperationsUnchanged",
    "TestOkNoneGuardStillRaises",
    "TestPaginationCoverage100",
    "TestPatternsCommands",
    "TestPatternsLogging",
    "TestPatternsTesting",
    "TestPerformanceBenchmarks",
    "TestPhase2CoverageFinal",
    "TestProjectLevelRefactor",
    "TestQueries",
    "TestRefactorPolicyMRO",
    "TestRuntimeCoverage100",
    "TestSafeCarriesException",
    "TestService",
    "TestService100Coverage",
    "TestServiceBootstrap",
    "TestServiceFullCoverage",
    "TestServiceResultProperty",
    "TestTraversePropagatesException",
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
    "TestUtils",
    "TestValues",
    "TestWorkspaceLevelRefactor",
    "Teste",
    "Testr",
    "TestrCoverage",
    "TestsCore",
    "TestsFlextConstants",
    "TestsFlextModels",
    "TestsFlextProtocols",
    "TestsFlextServiceBase",
    "TestsFlextTypes",
    "TestsFlextUtilities",
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
    "TestuTypeGuardsIsDictNonEmpty",
    "TestuTypeGuardsIsListNonEmpty",
    "TestuTypeGuardsIsStringNonEmpty",
    "TestuTypeGuardsNormalizeToMetadata",
    "TextUtilityContract",
    "TypeGuardsScenarios",
    "UnknownHint",
    "User",
    "UserFactory",
    "ValidatePaginationParamsScenario",
    "ValidatingService",
    "ValidatingServiceAuto",
    "ValidatingServiceAutoFactory",
    "ValidatingServiceFactory",
    "assert_rejects",
    "assert_validates",
    "assertion_helpers",
    "benchmark",
    "c",
    "clean_container",
    "contracts",
    "create_compare_entities_cases",
    "create_compare_value_objects_cases",
    "create_hash_entity_cases",
    "create_hash_value_object_cases",
    "create_validate_entity_has_id_cases",
    "create_validate_value_object_immutable_cases",
    "d",
    "e",
    "empty_strings",
    "fixture_factory",
    "flext_result_failure",
    "flext_result_success",
    "flext_tests",
    "generators_module",
    "get_memory_usage",
    "h",
    "handlers_module",
    "helpers",
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
    "integration",
    "invalid_hostnames",
    "invalid_port_numbers",
    "invalid_uris",
    "m",
    "mapper",
    "mock_external_service",
    "out_of_range",
    "p",
    "parser_scenarios",
    "patterns",
    "pytestmark",
    "r",
    "reliability_scenarios",
    "reset_all_factories",
    "reset_global_container",
    "reset_runtime_state",
    "runtime_cov_tests",
    "runtime_module",
    "runtime_tests",
    "s",
    "sample_data",
    "t",
    "temp_dir",
    "temp_directory",
    "temp_file",
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
    "test_centralizer_converts_typed_dict_factory_to_model",
    "test_centralizer_does_not_touch_settings_module",
    "test_centralizer_moves_dict_alias_in_typings_without_keyword_name",
    "test_centralizer_moves_manual_type_aliases_to_models_file",
    "test_chk_exercises_missed_branches",
    "test_circuit_breaker_transitions_and_metrics",
    "test_class_nesting_appends_to_existing_namespace_and_removes_pass",
    "test_class_nesting_keeps_unmapped_top_level_classes",
    "test_class_nesting_moves_top_level_class_into_new_namespace",
    "test_class_nesting_refactor_single_file_end_to_end",
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
    "test_context",
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
    "test_data_factory",
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
    "test_mro_scanner_includes_constants_variants_in_all_scopes",
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
    "u",
    "unit",
    "valid_hostnames",
    "valid_port_numbers",
    "valid_ranges",
    "valid_strings",
    "valid_uris",
    "validation_scenarios",
    "whitespace_strings",
    "x",
]


def __getattr__(name: str) -> FlextTypes.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
