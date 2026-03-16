# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Tests package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_core.typings import FlextTypes
    from tests import benchmark, helpers, integration, unit
    from tests.base import TestsFlextServiceBase
    from tests.benchmark.test_container_memory import (
        TestContainerMemory,
        get_memory_usage,
    )
    from tests.benchmark.test_container_performance import (
        PerformanceBenchmark,
        TestContainerPerformance,
    )
    from tests.benchmark.test_refactor_nesting_performance import (
        TestPerformanceBenchmarks,
    )
    from tests.conftest import (
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
    from tests.constants import TestsFlextConstants, c
    from tests.helpers.factories import (
        FailingService,
        FailingServiceAuto,
        FailingServiceAutoFactory,
        FailingServiceFactory,
        GenericModelFactory,
        GetUserService as s,
        GetUserServiceAuto,
        GetUserServiceAutoFactory,
        GetUserServiceFactory,
        ServiceFactoryRegistry,
        ServiceTestCaseFactory,
        ServiceTestCases,
        TestDataGenerators,
        UserFactory,
        ValidatingService,
        ValidatingServiceAuto,
        ValidatingServiceAutoFactory,
        ValidatingServiceFactory,
        reset_all_factories,
    )
    from tests.helpers.scenarios import (
        ParserScenario,
        ParserScenarios,
        ReliabilityScenario,
        ReliabilityScenarios,
        ValidationScenario,
        ValidationScenarios,
    )
    from tests.integration import patterns
    from tests.integration.patterns.test_advanced_patterns import TestFunction
    from tests.integration.patterns.test_architectural_patterns import (
        TestEnterprisePatterns,
        TestEventDrivenPatterns,
    )
    from tests.integration.patterns.test_patterns_commands import (
        CreateUserCommand,
        CreateUserCommandHandler,
        FailingCommand,
        FailingCommandHandler,
        FlextCommandId,
        FlextCommandType,
        TestFlextCommand,
        TestFlextCommandHandler,
        TestFlextCommandResults,
        UpdateUserCommand,
        UpdateUserCommandHandler,
    )
    from tests.integration.patterns.test_patterns_logging import (
        TestFlextContext,
        TestFlextLogger,
        TestFlextLoggerIntegration,
        TestFlextLoggerUsage,
        TestFlextLogLevel,
        assert_result_success,
        make_result_logger,
    )
    from tests.integration.patterns.test_patterns_testing import (
        AssertionBuilder,
        FixtureBuilder,
        FlextTestBuilder,
        GivenWhenThenBuilder,
        MockScenario,
        ParameterizedTestBuilder,
        SuiteBuilder,
        TestAdvancedPatterns,
        TestComprehensiveIntegration,
        TestPerformanceAnalysis,
        TestPropertyBasedPatterns,
        TestRealWorldScenarios,
        arrange_act_assert,
        mark_test_pattern,
    )
    from tests.integration.test_config_integration import (
        ConfigTestCase,
        ConfigTestFactories,
        TestFlextSettingsSingletonIntegration,
        ThreadSafetyTest,
    )
    from tests.integration.test_infra_integration import (
        TestBaseMkGenerationFlow,
        TestContainerIntegration,
        TestCrossModuleIntegration,
        TestIntegrationWithRealCommandServices,
        TestOutputSingletonConsistency,
        TestPathResolverDiscoveryFlow,
        TestServicerChaining,
        TestWorkspaceDetectionOrchestrationFlow,
    )
    from tests.integration.test_integration import TestLibraryIntegration
    from tests.integration.test_migration_validation import (
        TestBackwardCompatibility,
        TestMigrationComplexity,
        TestMigrationScenario1,
        TestMigrationScenario2,
        TestMigrationScenario4,
        TestMigrationScenario5,
    )
    from tests.integration.test_refactor_nesting_file import (
        test_class_nesting_refactor_single_file_end_to_end,
    )
    from tests.integration.test_refactor_nesting_idempotency import TestIdempotency
    from tests.integration.test_refactor_nesting_project import TestProjectLevelRefactor
    from tests.integration.test_refactor_nesting_workspace import (
        TestWorkspaceLevelRefactor,
    )
    from tests.integration.test_refactor_policy_mro import (
        AlgarOudMigConstants,
        AlgarOudMigModels,
        AlgarOudMigProtocols,
        AlgarOudMigTypes,
        AlgarOudMigUtilities,
        FlextCliConstants,
        FlextCliModels,
        FlextCliProtocols,
        FlextCliTypes,
        FlextCliUtilities,
        FlextLdapConstants,
        FlextLdapModels,
        FlextLdapProtocols,
        FlextLdapTypes,
        FlextLdapUtilities,
        test_mro_resolver_accepts_expected_order,
        test_mro_resolver_rejects_wrong_order,
    )
    from tests.integration.test_service import (
        LifecycleService,
        NotificationService,
        ServiceConfig,
        TestFlextServiceIntegration,
        UserQueryService,
        UserServiceEntity,
        pytestmark,
    )
    from tests.integration.test_system import TestCompleteFlextSystemIntegration
    from tests.models import TestsFlextModels, m
    from tests.protocols import TestsFlextProtocols, p
    from tests.test_documented_patterns import (
        GetUserService,
        MultiOperationService,
        RailwayTestCase,
        SendEmailService,
        ServiceTestCase,
        TestAllPatternsIntegration,
        TestFactories,
        TestPattern1V1Explicit,
        TestPattern2V2Property,
        TestPattern3RailwayV1,
        TestPattern4RailwayV2Property,
        TestPattern5MonadicComposition,
        TestPattern6ErrorHandling,
        TestPattern7AutomaticInfrastructure,
        TestPattern8MultipleOperations,
        User,
        ValidationService,
    )
    from tests.test_service_result_property import TestServiceResultProperty
    from tests.test_utils import (
        AssertionHelpers,
        FlextTestResult,
        FlextTestResultCo,
        TestDataFactory,
        TestFixtureFactory,
        assertion_helpers,
        fixture_factory,
        test_data_factory,
    )
    from tests.typings import T, T_co, T_contra, TestsFlextTypes, t
    from tests.unit import contracts, flext_tests
    from tests.unit.conftest_infra import (
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
    from tests.unit.contracts.text_contract import TextUtilityContract
    from tests.unit.flext_tests.test_builders import TestFlextTestsBuilders
    from tests.unit.flext_tests.test_docker import (
        TestContainerInfo,
        TestContainerStatus,
        TestFlextTestsDocker,
        TestFlextTestsDockerWorkerId,
        TestFlextTestsDockerWorkspaceRoot,
    )
    from tests.unit.flext_tests.test_domains import TestFlextTestsDomains
    from tests.unit.flext_tests.test_factories import (
        TestConfig,
        TestFactoriesHelpers,
        TestFlextTestsFactoriesModernAPI,
        TestsFlextTestsFactoriesDict,
        TestsFlextTestsFactoriesGeneric,
        TestsFlextTestsFactoriesList,
        TestsFlextTestsFactoriesModel,
        TestsFlextTestsFactoriesRes,
        TestUser,
    )
    from tests.unit.flext_tests.test_files import (
        TestAssertExists,
        TestBatchOperations,
        TestCreateInStatic,
        TestFileInfo,
        TestFileInfoFromModels,
        TestFlextTestsFiles,
        TestFlextTestsFilesNewApi,
        TestInfoWithContentMeta,
        TestShortAlias,
    )
    from tests.unit.flext_tests.test_matchers import TestFlextTestsMatchers
    from tests.unit.flext_tests.test_utilities import (
        TestFlextTestsUtilitiesFactory,
        TestFlextTestsUtilitiesResult,
        TestFlextTestsUtilitiesResult as r,
        TestFlextTestsUtilitiesResultCompat,
        TestFlextTestsUtilitiesTestContext,
    )
    from tests.unit.test_args_coverage_100 import TestFlextUtilitiesArgs
    from tests.unit.test_automated_architecture import TestAutomatedArchitecture
    from tests.unit.test_automated_container import TestAutomatedFlextContainer
    from tests.unit.test_automated_context import TestAutomatedFlextContext
    from tests.unit.test_automated_decorators import (
        TestAutomatedFlextDecorators,
        TestAutomatedFlextDecorators as d,
    )
    from tests.unit.test_automated_dispatcher import TestAutomatedFlextDispatcher
    from tests.unit.test_automated_exceptions import (
        TestAutomatedExceptions,
        TestAutomatedExceptions as e,
    )
    from tests.unit.test_automated_handlers import (
        TestAutomatedFlextHandlers,
        TestAutomatedFlextHandlers as h,
    )
    from tests.unit.test_automated_loggings import TestAutomatedFlextLogger
    from tests.unit.test_automated_mixins import (
        TestAutomatedFlextMixins,
        TestAutomatedFlextMixins as x,
    )
    from tests.unit.test_automated_registry import TestAutomatedFlextRegistry
    from tests.unit.test_automated_result import TestAutomatedResult
    from tests.unit.test_automated_runtime import TestAutomatedFlextRuntime
    from tests.unit.test_automated_service import TestAutomatedFlextService
    from tests.unit.test_automated_settings import TestAutomatedFlextSettings
    from tests.unit.test_automated_utilities import TestAutomatedFlextUtilities
    from tests.unit.test_collection_utilities_coverage_100 import (
        CoerceListValidatorScenario,
        CollectionScenarios,
        ParseMappingScenario,
        ParseSequenceScenario,
    )
    from tests.unit.test_collections_coverage_100 import (
        TestFlextModelsCollectionsCategories,
        TestFlextModelsCollectionsOptions,
        TestFlextModelsCollectionsResults,
        TestFlextModelsCollectionsSettings,
        TestFlextModelsCollectionsStatistics,
    )
    from tests.unit.test_config import TestFlextSettings
    from tests.unit.test_constants import TestFlextConstants
    from tests.unit.test_constants_full_coverage import (
        test_constants_auto_enum_and_bimapping_paths,
    )
    from tests.unit.test_container import TestFlextContainer
    from tests.unit.test_container_full_coverage import (
        test_additional_container_branches_cover_fluent_and_lookup_paths,
        test_additional_register_factory_and_unregister_paths,
        test_builder,
        test_config_context_properties_and_defaults,
        test_configure_with_resource_register_and_factory_error_paths,
        test_container_remaining_branch_paths_in_sync_factory_and_getters,
        test_create_auto_register_factories_path,
        test_create_auto_register_factory_wrapper_callable_and_non_callable,
        test_create_scoped_instance_and_scoped_additional_branches,
        test_get_and_get_typed_resource_factory_paths,
        test_initialize_di_components_error_paths,
        test_initialize_di_components_second_type_error_branch,
        test_misc_unregistration_clear_and_reset,
        test_provide_property_paths,
        test_register_existing_providers_full_paths_and_misc_methods,
        test_register_existing_providers_skips_and_register_core_fallback,
        test_scoped_config_context_branches,
        test_sync_config_namespace_paths,
        test_sync_config_registers_namespace_factories_and_fallbacks,
    )
    from tests.unit.test_context_coverage_100 import TestContext100Coverage
    from tests.unit.test_context_full_coverage import (
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
    from tests.unit.test_coverage_76_lines import (
        TestResultBasics,
        TestResultTransformations,
    )
    from tests.unit.test_coverage_context import (
        TestContextDataModel,
        TestCorrelationDomain,
        TestPerformanceDomain,
        TestServiceDomain,
        TestUtilitiesDomain,
    )
    from tests.unit.test_coverage_exceptions import (
        TestExceptionContext,
        TestExceptionEdgeCases,
        TestExceptionFactory,
        TestExceptionIntegration,
        TestExceptionMetrics,
        TestExceptionProperties,
        TestExceptionSerialization,
        TestFlextExceptionsHierarchy,
        TestHierarchicalExceptionSystem,
    )
    from tests.unit.test_coverage_loggings import (
        TestEdgeCases,
        TestExceptionLogging,
        TestFactoryPatterns,
        TestGlobalContextManagement,
        TestInstanceCreation,
        TestLevelBasedContextManagement,
        TestLoggingIntegration,
        TestLoggingMethods,
        TestScopedContextManagement,
    )
    from tests.unit.test_coverage_models import (
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
    from tests.unit.test_decorators import TestFlextDecorators
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestFactoryDecoratorsDiscoveryHasFactories,
        TestFactoryDecoratorsDiscoveryScanModule,
    )
    from tests.unit.test_decorators_full_coverage import (
        test_bind_operation_context_without_ensure_correlation_and_bind_failure,
        test_clear_operation_scope_and_handle_log_result_paths,
        test_combined_with_and_without_railway_uses_injection,
        test_deprecated_wrapper_emits_warning_and_returns_value,
        test_execute_retry_exponential_and_handle_exhaustion_raise_last_exception,
        test_execute_retry_loop_covers_default_linear_and_never_ran,
        test_handle_log_result_without_fallback_logger_and_non_dict_like_extra,
        test_handle_retry_exhaustion_falsey_exception_reaches_timeout_error,
        test_inject_sets_missing_dependency_from_container,
        test_log_operation_track_perf_exception_adds_duration,
        test_railway_and_retry_additional_paths,
        test_resolve_logger_prefers_logger_attribute,
        test_retry_unreachable_timeouterror_path,
        test_timeout_additional_success_and_reraise_timeout_paths,
        test_timeout_covers_exception_timeout_branch,
        test_timeout_raises_when_successful_call_exceeds_limit,
        test_timeout_reraises_original_exception_when_within_limit,
        test_track_performance_success_and_failure_paths,
        test_with_correlation_with_context_track_operation_and_factory,
    )
    from tests.unit.test_deprecation_warnings import (
        TestFacadeDeprecatedAliases,
        TestGuardsDeprecatedMethods,
        TestMapperDeprecatedMethods,
        TestRuntimeDeprecatedNormalizeMethods,
        TestStrictContainerNormalization,
    )
    from tests.unit.test_di_incremental import (
        TestContainerDIRealExecution,
        TestDependencyIntegrationRealExecution,
        TestDIBridgeRealExecution,
        TestRealWiringScenarios,
        TestServiceBootstrapWithDI,
    )
    from tests.unit.test_di_services_access import (
        TestConfigServiceViaDI,
        TestContextServiceViaDI,
        TestLoggerServiceViaDI,
        TestServicesIntegrationViaDI,
    )
    from tests.unit.test_dispatcher_di import TestDispatcherDI
    from tests.unit.test_dispatcher_full_coverage import (
        EventHandler,
        QueryHandler,
        SampleCommand,
        SampleEvent,
        SampleHandler,
        SampleQuery,
        UnregisteredCommand,
        dispatcher,
        test_callable_registration_with_attribute,
        test_dispatch_invalid_input_types,
        test_event_publishing_strict,
        test_exception_handling_in_dispatch,
        test_handler_attribute_discovery,
        test_invalid_registration_attempts,
        test_strict_registration_and_dispatch,
    )
    from tests.unit.test_dispatcher_minimal import (
        AutoCommand,
        AutoDiscoveryHandler,
        EchoHandler,
        EventSubscriber,
        ExplodingHandler,
        test_dispatch_after_handler_removal_fails,
        test_dispatch_auto_discovery_handler,
        test_dispatch_command_success,
        test_dispatch_handler_exception_returns_failure,
        test_dispatch_no_handler_fails,
        test_publish_event_to_subscriber,
        test_publish_no_subscribers_succeeds,
        test_register_handler_as_event_subscriber,
        test_register_handler_with_can_handle,
        test_register_handler_with_message_type,
        test_register_handler_without_route_fails,
    )
    from tests.unit.test_dispatcher_reliability import (
        test_circuit_breaker_transitions_and_metrics,
        test_rate_limiter_blocks_then_recovers,
        test_rate_limiter_jitter_application,
        test_retry_policy_behavior,
    )
    from tests.unit.test_dispatcher_reliability_full_coverage import (
        test_dispatcher_reliability_branch_paths,
    )
    from tests.unit.test_dispatcher_timeout_coverage_100 import (
        TestTimeoutEnforcerCleanup,
        TestTimeoutEnforcerEdgeCases,
        TestTimeoutEnforcerExecutorManagement,
        TestTimeoutEnforcerInitialization,
        TimeoutEnforcerScenarios,
    )
    from tests.unit.test_entity_coverage import TestEntityCoverageEdgeCases
    from tests.unit.test_enum_utilities_coverage_100 import (
        CoerceValidatorScenario,
        EnumScenarios,
        IsMemberScenario,
        IsSubsetScenario,
        ParseOrDefaultScenario,
        ParseScenario,
        TestuEnumCoerceByNameValidator,
        TestuEnumCoerceValidator,
        TestuEnumIsMember,
        TestuEnumIsSubset,
        TestuEnumMetadata,
        TestuEnumParse,
        TestuEnumParseOrDefault,
    )
    from tests.unit.test_exceptions import Teste
    from tests.unit.test_exceptions_full_coverage import (
        test_authentication_error_normalizes_extra_kwargs_into_context,
        test_base_error_normalize_metadata_merges_existing_metadata_model,
        test_exceptions_uncovered_metadata_paths,
        test_merge_metadata_context_paths,
        test_not_found_error_correlation_id_selection_and_extra_kwargs,
    )
    from tests.unit.test_final_75_percent_push import TestCoveragePush75Percent
    from tests.unit.test_handler_decorator_discovery import (
        TestHandlerDecoratorMetadata,
        TestHandlerDiscoveryClass,
        TestHandlerDiscoveryEdgeCases,
        TestHandlerDiscoveryIntegration,
        TestHandlerDiscoveryModule,
        TestHandlerDiscoveryServiceIntegration,
    )
    from tests.unit.test_handlers import TestFlextHandlers
    from tests.unit.test_handlers_full_coverage import (
        handlers_module,
        test_create_from_callable_branches,
        test_discovery_narrowed_function_paths,
        test_handler_type_literal_and_invalid,
        test_invalid_handler_mode_init_raises,
        test_run_pipeline_query_and_event_paths,
    )
    from tests.unit.test_loggings_error_paths_coverage import TestLoggingsErrorPaths
    from tests.unit.test_loggings_full_coverage import (
        test_loggings_bind_clear_level_error_paths,
        test_loggings_context_and_factory_paths,
        test_loggings_exception_and_adapter_paths,
        test_loggings_instance_and_message_format_paths,
        test_loggings_remaining_branch_paths,
        test_loggings_source_and_log_error_paths,
        test_loggings_uncovered_level_trace_path_and_exception_guards,
    )
    from tests.unit.test_loggings_strict_returns import (
        TestBackwardCompatDiscardReturnValue,
        TestCriticalReturnsResultBool,
        TestDebugReturnsResultBool,
        TestErrorReturnsResultBool,
        TestExceptionReturnsResultBool,
        TestInfoReturnsResultBool,
        TestLogReturnsResultBool,
        TestProtocolComplianceStructlogLogger,
        TestTraceReturnsResultBool,
        TestWarningReturnsResultBool,
    )
    from tests.unit.test_mixins import TestFlextMixinsNestedClasses
    from tests.unit.test_mixins_full_coverage import (
        test_mixins_container_registration_and_logger_paths,
        test_mixins_context_logging_and_cqrs_paths,
        test_mixins_context_stack_pop_initializes_missing_stack_attr,
        test_mixins_remaining_branch_paths,
        test_mixins_result_and_model_conversion_paths,
        test_mixins_runtime_bootstrap_and_track_paths,
        test_mixins_validation_and_protocol_paths,
    )
    from tests.unit.test_models import TestFlextModels
    from tests.unit.test_models_79_coverage import (
        TestFlextModelsAggregateRoot,
        TestFlextModelsCommand,
        TestFlextModelsDomainEvent,
        TestFlextModelsEdgeCases,
        TestFlextModelsEntity,
        TestFlextModelsIntegration,
        TestFlextModelsQuery,
        TestFlextModelsValue,
    )
    from tests.unit.test_models_base_full_coverage import (
        test_frozen_value_model_equality_and_hash,
        test_identifiable_unique_id_empty_rejected,
        test_metadata_attributes_accepts_basemodel_mapping,
        test_metadata_attributes_accepts_none,
        test_metadata_attributes_accepts_t_dict_and_mapping,
        test_metadata_attributes_rejects_basemodel_non_mapping_dump,
        test_metadata_attributes_rejects_non_mapping,
        test_timestampable_timestamp_conversion_and_json_serializer,
        test_timestamped_model_and_alias_and_canonical_symbols,
    )
    from tests.unit.test_models_collections_full_coverage import (
        test_categories_clear_and_symbols_are_available,
        test_config_hash_from_mapping_and_non_hashable,
        test_options_merge_conflict_paths_and_empty_merge_options,
        test_results_internal_conflict_paths_and_combine,
        test_rules_merge_combines_model_dump_values,
        test_statistics_from_dict_and_none_conflict_resolution,
    )
    from tests.unit.test_models_container import (
        ContainerModelsScenarios,
        TestFlextModelsContainer,
        TestFlextUtilitiesModelNormalizeToMetadata,
    )
    from tests.unit.test_models_container_full_coverage import (
        test_container_resource_registration_metadata_normalized,
    )
    from tests.unit.test_models_context_full_coverage import (
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
    from tests.unit.test_models_cqrs_full_coverage import (
        test_command_pagination_limit,
        test_cqrs_query_resolve_deeper_and_int_pagination,
        test_flext_message_type_alias_adapter,
        test_handler_builder_fluent_methods,
        test_query_resolve_pagination_wrapper_and_fallback,
        test_query_validate_pagination_dict_and_default,
    )
    from tests.unit.test_models_entity_full_coverage import (
        test_entity_comparable_map_and_bulk_validation_paths,
    )
    from tests.unit.test_models_generic_full_coverage import (
        test_canonical_aliases_are_available,
        test_conversion_add_converted_and_error_metadata_append_paths,
        test_conversion_add_skipped_skip_reason_upsert_paths,
        test_conversion_add_warning_metadata_append_paths,
        test_conversion_start_and_complete_methods,
        test_operation_progress_start_operation_sets_runtime_fields,
    )
    from tests.unit.test_models_handler_full_coverage import (
        test_models_handler_branches,
        test_models_handler_uncovered_mode_and_reset_paths,
    )
    from tests.unit.test_models_service_full_coverage import (
        test_service_request_timeout_post_validator_messages,
        test_service_request_timeout_validator_branches,
    )
    from tests.unit.test_models_settings_full_coverage import (
        test_models_settings_branch_paths,
        test_models_settings_context_validator_and_non_standard_status_input,
    )
    from tests.unit.test_models_validation_full_coverage import (
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
    from tests.unit.test_namespace_validator import TestFlextInfraNamespaceValidator
    from tests.unit.test_pagination_coverage_100 import (
        ExtractPageParamsScenario,
        PaginationScenarios,
        PreparePaginationDataScenario,
        TestuPaginationBuildPaginationResponse,
        TestuPaginationExtractPageParams,
        TestuPaginationExtractPaginationConfig,
        TestuPaginationPreparePaginationData,
        TestuPaginationValidatePaginationParams,
        ValidatePaginationParamsScenario,
    )
    from tests.unit.test_phase2_coverage_final import TestPhase2FinalCoveragePush
    from tests.unit.test_protocols import TestFlextProtocols
    from tests.unit.test_refactor_cli_models_workflow import (
        test_centralize_pydantic_cli_outputs_extended_metrics,
        test_namespace_enforce_cli_fails_on_manual_protocol_violation,
        test_ultrawork_models_cli_runs_dry_run_copy,
    )
    from tests.unit.test_refactor_migrate_to_class_mro import (
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
    from tests.unit.test_refactor_namespace_enforcer import (
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
    from tests.unit.test_refactor_policy_family_rules import (
        test_decorators_family_blocks_dispatcher_target,
        test_dispatcher_family_blocks_models_target,
        test_helper_consolidation_is_prechecked,
        test_models_family_blocks_utilities_target,
        test_runtime_family_blocks_non_runtime_target,
        test_utilities_family_allows_utilities_target,
    )
    from tests.unit.test_refactor_pydantic_centralizer import (
        test_centralizer_converts_typed_dict_factory_to_model,
        test_centralizer_does_not_touch_settings_module,
        test_centralizer_moves_dict_alias_in_typings_without_keyword_name,
        test_centralizer_moves_manual_type_aliases_to_models_file,
    )
    from tests.unit.test_registry import ConcreteTestHandler, TestFlextRegistry
    from tests.unit.test_registry_full_coverage import (
        test_create_auto_discover_and_mode_mapping,
        test_execute_and_register_handler_failure_paths,
        test_get_plugin_and_register_metadata_and_list_items_exception,
        test_summary_error_paths_and_bindings_failures,
        test_summary_properties_and_subclass_storage_reset,
    )
    from tests.unit.test_result import Testr
    from tests.unit.test_result_additional import (
        test_create_from_callable_and_repr,
        test_flow_through_short_circuits_on_failure,
        test_map_error_identity_and_transform,
        test_ok_accepts_none,
        test_with_resource_cleanup_runs,
    )
    from tests.unit.test_result_coverage_100 import TestrCoverage
    from tests.unit.test_result_exception_carrying import (
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
    from tests.unit.test_result_full_coverage import (
        test_from_validation_and_to_model_paths,
        test_init_fallback_and_lazy_returns_result_property,
        test_lash_runtime_result_paths,
        test_map_flat_map_and_then_paths,
        test_recover_tap_and_tap_error_paths,
        test_type_guards_result,
        test_validation_like_error_structure,
    )
    from tests.unit.test_runtime import TestFlextRuntime
    from tests.unit.test_runtime_coverage_100 import (
        TestRuntimeDictLike,
        TestRuntimeTypeChecking,
    )
    from tests.unit.test_runtime_full_coverage import (
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
        test_deprecated_normalize_to_general_value_warns,
        test_deprecated_normalize_to_metadata_value_warns,
        test_ensure_trace_context_dict_conversion_paths,
        test_get_logger_none_name_paths,
        test_model_helpers_remaining_paths,
        test_model_support_and_hash_compare_paths,
        test_normalization_edge_branches,
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
    from tests.unit.test_service import TestsCore
    from tests.unit.test_service_additional import (
        RuntimeCloneService,
        test_get_service_info,
        test_is_valid_handles_validation_exception,
        test_result_property_raises_on_failure,
    )
    from tests.unit.test_service_bootstrap import (
        ConcreteTestService,
        TestServiceBootstrap,
    )
    from tests.unit.test_service_coverage_100 import (
        TestService,
        TestService100Coverage,
        TestServiceWithValidation,
    )
    from tests.unit.test_service_full_coverage import (
        test_service_create_initial_runtime_prefers_custom_config_type_and_context_property,
        test_service_create_runtime_container_overrides_branch,
        test_service_init_type_guards_and_properties,
    )
    from tests.unit.test_settings_full_coverage import (
        test_settings_materialize_and_context_overrides,
    )
    from tests.unit.test_transformer_class_nesting import (
        test_class_nesting_appends_to_existing_namespace_and_removes_pass,
        test_class_nesting_keeps_unmapped_top_level_classes,
        test_class_nesting_moves_top_level_class_into_new_namespace,
    )
    from tests.unit.test_transformer_helper_consolidation import (
        TestHelperConsolidationTransformer,
    )
    from tests.unit.test_transformer_nested_class_propagation import (
        NestedClassPropagationTransformer,
        test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage,
        test_nested_class_propagation_updates_import_annotations_and_calls,
    )
    from tests.unit.test_typings import TestFlextTypings
    from tests.unit.test_typings_full_coverage import (
        TestConfigMapDictOps,
        TestDictMixinOperations,
        TestValidatorCallable,
        TestValidatorMapMixin,
    )
    from tests.unit.test_utilities import Testu
    from tests.unit.test_utilities_args_full_coverage import (
        UnknownHint,
        test_args_get_enum_params_annotated_unwrap_branch,
        test_args_get_enum_params_branches,
    )
    from tests.unit.test_utilities_cache_coverage_100 import (
        CacheScenarios,
        TestuCacheClearObjectCache,
        TestuCacheGenerateCacheKey,
        TestuCacheHasCacheAttributes,
        TestuCacheLogger,
        TestuCacheNormalizeComponent,
        TestuCacheSortDictKeys,
        TestuCacheSortKey,
    )
    from tests.unit.test_utilities_checker_full_coverage import (
        MissingType,
        test_checker_logger_and_safe_type_hints_fallback,
        test_extract_message_type_annotation_and_dict_subclass_paths,
        test_extract_message_type_from_handle_with_only_self,
        test_extract_message_type_from_parameter_branches,
        test_object_dict_and_type_error_fallback_paths,
    )
    from tests.unit.test_utilities_collection_coverage_100 import (
        TestuCollectionBatch,
        TestuCollectionChunk,
        TestuCollectionCoerceDictValidator,
        TestuCollectionCoerceListValidator,
        TestuCollectionCount,
        TestuCollectionFilter,
        TestuCollectionFind,
        TestuCollectionGroup,
        TestuCollectionMap,
        TestuCollectionMerge,
        TestuCollectionParseMapping,
        TestuCollectionParseSequence,
        TestuCollectionProcess,
    )
    from tests.unit.test_utilities_collection_full_coverage import (
        test_batch_fail_collect_flatten_and_progress,
        test_collection_batch_failure_error_capture_and_parse_sequence_outer_error,
        test_find_mapping_no_match_and_merge_error_paths,
        test_is_general_value_list_accepts_list_subclass,
        test_parse_mapping_outer_exception,
        test_process_outer_exception_and_coercion_branches,
    )
    from tests.unit.test_utilities_configuration_coverage_100 import (
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
    from tests.unit.test_utilities_configuration_full_coverage import (
        test_build_options_invalid_only_kwargs_returns_base,
        test_private_getters_exception_paths,
        test_register_singleton_register_factory_and_bulk_register_paths,
        test_resolve_env_file_and_log_level,
    )
    from tests.unit.test_utilities_context_full_coverage import (
        TestCloneContainer,
        TestCloneRuntime,
        TestCreateDatetimeProxy,
        TestCreateDictProxy,
        TestCreateStrProxy,
    )
    from tests.unit.test_utilities_conversion_full_coverage import (
        test_conversion_string_and_join_paths,
    )
    from tests.unit.test_utilities_coverage import TestUtilitiesCoverage
    from tests.unit.test_utilities_data_mapper import (
        TestMapperBuildFlagsDict,
        TestMapperCollectActiveKeys,
        TestMapperFilterDict,
        TestMapperInvertDict,
        TestMapperMapDictKeys,
        TestMapperTransformValues,
    )
    from tests.unit.test_utilities_deprecation_full_coverage import (
        test_deprecated_class_noop_init_branch,
    )
    from tests.unit.test_utilities_domain import (
        TestuDomain,
        create_compare_entities_cases,
        create_compare_value_objects_cases,
        create_hash_entity_cases,
        create_hash_value_object_cases,
        create_validate_entity_has_id_cases,
        create_validate_value_object_immutable_cases,
    )
    from tests.unit.test_utilities_domain_full_coverage import (
        TestDomainHashValue,
        TestDomainLogger,
        TestValidateValueImmutable,
        test_validate_value_object_immutable_exception_and_no_setattr_branch,
    )
    from tests.unit.test_utilities_enum_full_coverage import (
        Priority,
        Status,
        TextLike,
        test_auto_value_lowercases_input,
        test_bi_map_returns_forward_copy_and_inverse,
        test_create_discriminated_union_multiple_enums,
        test_create_enum_executes_factory_path,
        test_dispatch_coerce_mode_with_enum_string_and_other_object,
        test_dispatch_is_member_by_name_and_by_value,
        test_dispatch_is_name_mode,
        test_dispatch_parse_mode_with_enum_string_and_other_object,
        test_dispatch_unknown_mode_raises,
        test_get_enum_values_returns_immutable_sequence,
        test_members_uses_cache_on_second_call,
        test_names_uses_cache_on_second_call,
        test_private_coerce_with_enum_and_string,
        test_private_is_member_by_name,
        test_private_is_member_by_value,
        test_private_parse_success_and_failure,
        test_shortcuts_delegate_to_primary_methods,
    )
    from tests.unit.test_utilities_generators_full_coverage import (
        generators_module,
        runtime_module,
        test_enrich_and_ensure_trace_context_branches,
        test_ensure_dict_branches,
        test_generate_special_paths_and_dynamic_subclass,
        test_generators_additional_missed_paths,
        test_generators_mapping_non_dict_normalization_path,
        test_normalize_context_to_dict_error_paths,
    )
    from tests.unit.test_utilities_guards_full_coverage import (
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
    from tests.unit.test_utilities_mapper_coverage_100 import (
        SimpleObj,
        TestuMapperAccessors,
        TestuMapperAdvanced,
        TestuMapperBuild,
        TestuMapperConversions,
        TestuMapperExtract,
        TestuMapperUtils,
    )
    from tests.unit.test_utilities_mapper_full_coverage import (
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
    from tests.unit.test_utilities_model_full_coverage import (
        test_merge_defaults_and_dump_paths,
        test_normalize_to_pydantic_dict_and_value_branches,
        test_update_exception_path,
        test_update_success_path_returns_ok_result,
    )
    from tests.unit.test_utilities_pagination_full_coverage import (
        test_pagination_response_string_fallbacks,
    )
    from tests.unit.test_utilities_parser_full_coverage import (
        test_parser_convert_and_norm_branches,
        test_parser_internal_helpers_additional_coverage,
        test_parser_parse_helpers_and_primitive_coercion_branches,
        test_parser_pipeline_and_pattern_branches,
        test_parser_remaining_branch_paths,
        test_parser_safe_length_and_parse_delimited_error_paths,
        test_parser_split_and_normalize_exception_paths,
        test_parser_success_and_edge_paths_cover_major_branches,
    )
    from tests.unit.test_utilities_reliability import TestFlextUtilitiesReliability
    from tests.unit.test_utilities_reliability_full_coverage import (
        test_utilities_reliability_branches,
        test_utilities_reliability_compose_returns_non_result_directly,
        test_utilities_reliability_uncovered_retry_compose_and_sequence_paths,
    )
    from tests.unit.test_utilities_string_parser import (
        StringParserTestFactory,
        TestuStringParser,
    )
    from tests.unit.test_utilities_text_full_coverage import (
        TestCleanText,
        TestFormatAppId,
        TestSafeString,
        TestTextLogger,
    )
    from tests.unit.test_utilities_type_checker_coverage_100 import (
        DictHandler,
        ExplicitTypeHandler,
        GenericHandler,
        IntHandler,
        NoHandleMethod,
        NonCallableHandle,
        ObjectHandler,
        StringHandler,
        TestuTypeChecker,
        TMessage,
    )
    from tests.unit.test_utilities_type_guards_coverage_100 import (
        TestuTypeGuardsIsDictNonEmpty,
        TestuTypeGuardsIsListNonEmpty,
        TestuTypeGuardsIsStringNonEmpty,
        TestuTypeGuardsNormalizeToMetadata,
        TypeGuardsScenarios,
    )
    from tests.unit.test_version import TestFlextVersion
    from tests.utilities import TestsFlextUtilities, u

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AlgarOudMigConstants": (
        "tests.integration.test_refactor_policy_mro",
        "AlgarOudMigConstants",
    ),
    "AlgarOudMigModels": (
        "tests.integration.test_refactor_policy_mro",
        "AlgarOudMigModels",
    ),
    "AlgarOudMigProtocols": (
        "tests.integration.test_refactor_policy_mro",
        "AlgarOudMigProtocols",
    ),
    "AlgarOudMigTypes": (
        "tests.integration.test_refactor_policy_mro",
        "AlgarOudMigTypes",
    ),
    "AlgarOudMigUtilities": (
        "tests.integration.test_refactor_policy_mro",
        "AlgarOudMigUtilities",
    ),
    "AssertionBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "AssertionBuilder",
    ),
    "AssertionHelpers": ("tests.test_utils", "AssertionHelpers"),
    "AttrObject": ("tests.unit.test_utilities_mapper_full_coverage", "AttrObject"),
    "AutoCommand": ("tests.unit.test_dispatcher_minimal", "AutoCommand"),
    "AutoDiscoveryHandler": (
        "tests.unit.test_dispatcher_minimal",
        "AutoDiscoveryHandler",
    ),
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
    "CoerceListValidatorScenario": (
        "tests.unit.test_collection_utilities_coverage_100",
        "CoerceListValidatorScenario",
    ),
    "CoerceValidatorScenario": (
        "tests.unit.test_enum_utilities_coverage_100",
        "CoerceValidatorScenario",
    ),
    "CollectionScenarios": (
        "tests.unit.test_collection_utilities_coverage_100",
        "CollectionScenarios",
    ),
    "ConcreteTestHandler": ("tests.unit.test_registry", "ConcreteTestHandler"),
    "ConcreteTestService": ("tests.unit.test_service_bootstrap", "ConcreteTestService"),
    "ConfigTestCase": ("tests.integration.test_config_integration", "ConfigTestCase"),
    "ConfigTestFactories": (
        "tests.integration.test_config_integration",
        "ConfigTestFactories",
    ),
    "ConfigWithoutModelConfigForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "ConfigWithoutModelConfigForTest",
    ),
    "ContainerModelsScenarios": (
        "tests.unit.test_models_container",
        "ContainerModelsScenarios",
    ),
    "CreateUserCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "CreateUserCommand",
    ),
    "CreateUserCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "CreateUserCommandHandler",
    ),
    "DataclassConfigForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "DataclassConfigForTest",
    ),
    "DictHandler": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "DictHandler",
    ),
    "EchoHandler": ("tests.unit.test_dispatcher_minimal", "EchoHandler"),
    "EnumScenarios": ("tests.unit.test_enum_utilities_coverage_100", "EnumScenarios"),
    "EventHandler": ("tests.unit.test_dispatcher_full_coverage", "EventHandler"),
    "EventSubscriber": ("tests.unit.test_dispatcher_minimal", "EventSubscriber"),
    "ExplicitTypeHandler": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "ExplicitTypeHandler",
    ),
    "ExplodingHandler": ("tests.unit.test_dispatcher_minimal", "ExplodingHandler"),
    "ExplodingLenList": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "ExplodingLenList",
    ),
    "ExtractPageParamsScenario": (
        "tests.unit.test_pagination_coverage_100",
        "ExtractPageParamsScenario",
    ),
    "FailingCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "FailingCommand",
    ),
    "FailingCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "FailingCommandHandler",
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
    "FixtureBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "FixtureBuilder",
    ),
    "FlextCliConstants": (
        "tests.integration.test_refactor_policy_mro",
        "FlextCliConstants",
    ),
    "FlextCliModels": ("tests.integration.test_refactor_policy_mro", "FlextCliModels"),
    "FlextCliProtocols": (
        "tests.integration.test_refactor_policy_mro",
        "FlextCliProtocols",
    ),
    "FlextCliTypes": ("tests.integration.test_refactor_policy_mro", "FlextCliTypes"),
    "FlextCliUtilities": (
        "tests.integration.test_refactor_policy_mro",
        "FlextCliUtilities",
    ),
    "FlextCommandId": (
        "tests.integration.patterns.test_patterns_commands",
        "FlextCommandId",
    ),
    "FlextCommandType": (
        "tests.integration.patterns.test_patterns_commands",
        "FlextCommandType",
    ),
    "FlextLdapConstants": (
        "tests.integration.test_refactor_policy_mro",
        "FlextLdapConstants",
    ),
    "FlextLdapModels": (
        "tests.integration.test_refactor_policy_mro",
        "FlextLdapModels",
    ),
    "FlextLdapProtocols": (
        "tests.integration.test_refactor_policy_mro",
        "FlextLdapProtocols",
    ),
    "FlextLdapTypes": ("tests.integration.test_refactor_policy_mro", "FlextLdapTypes"),
    "FlextLdapUtilities": (
        "tests.integration.test_refactor_policy_mro",
        "FlextLdapUtilities",
    ),
    "FlextTestBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "FlextTestBuilder",
    ),
    "FlextTestResult": ("tests.test_utils", "FlextTestResult"),
    "FlextTestResultCo": ("tests.test_utils", "FlextTestResultCo"),
    "FunctionalExternalService": ("tests.conftest", "FunctionalExternalService"),
    "GenericHandler": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "GenericHandler",
    ),
    "GenericModelFactory": ("tests.helpers.factories", "GenericModelFactory"),
    "GetUserService": ("tests.test_documented_patterns", "GetUserService"),
    "GetUserServiceAuto": ("tests.helpers.factories", "GetUserServiceAuto"),
    "GetUserServiceAutoFactory": (
        "tests.helpers.factories",
        "GetUserServiceAutoFactory",
    ),
    "GetUserServiceFactory": ("tests.helpers.factories", "GetUserServiceFactory"),
    "GivenWhenThenBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "GivenWhenThenBuilder",
    ),
    "IntHandler": ("tests.unit.test_utilities_type_checker_coverage_100", "IntHandler"),
    "IsMemberScenario": (
        "tests.unit.test_enum_utilities_coverage_100",
        "IsMemberScenario",
    ),
    "IsSubsetScenario": (
        "tests.unit.test_enum_utilities_coverage_100",
        "IsSubsetScenario",
    ),
    "LifecycleService": ("tests.integration.test_service", "LifecycleService"),
    "MissingType": ("tests.unit.test_utilities_checker_full_coverage", "MissingType"),
    "MockScenario": (
        "tests.integration.patterns.test_patterns_testing",
        "MockScenario",
    ),
    "MultiOperationService": (
        "tests.test_documented_patterns",
        "MultiOperationService",
    ),
    "NestedClassPropagationTransformer": (
        "tests.unit.test_transformer_nested_class_propagation",
        "NestedClassPropagationTransformer",
    ),
    "NoHandleMethod": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "NoHandleMethod",
    ),
    "NonCallableHandle": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "NonCallableHandle",
    ),
    "NotificationService": ("tests.integration.test_service", "NotificationService"),
    "ObjectHandler": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "ObjectHandler",
    ),
    "OptionsModelForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "OptionsModelForTest",
    ),
    "PaginationScenarios": (
        "tests.unit.test_pagination_coverage_100",
        "PaginationScenarios",
    ),
    "ParameterizedTestBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "ParameterizedTestBuilder",
    ),
    "ParseMappingScenario": (
        "tests.unit.test_collection_utilities_coverage_100",
        "ParseMappingScenario",
    ),
    "ParseOrDefaultScenario": (
        "tests.unit.test_enum_utilities_coverage_100",
        "ParseOrDefaultScenario",
    ),
    "ParseScenario": ("tests.unit.test_enum_utilities_coverage_100", "ParseScenario"),
    "ParseSequenceScenario": (
        "tests.unit.test_collection_utilities_coverage_100",
        "ParseSequenceScenario",
    ),
    "ParserScenario": ("tests.helpers.scenarios", "ParserScenario"),
    "ParserScenarios": ("tests.helpers.scenarios", "ParserScenarios"),
    "PerformanceBenchmark": (
        "tests.benchmark.test_container_performance",
        "PerformanceBenchmark",
    ),
    "PreparePaginationDataScenario": (
        "tests.unit.test_pagination_coverage_100",
        "PreparePaginationDataScenario",
    ),
    "Priority": ("tests.unit.test_utilities_enum_full_coverage", "Priority"),
    "QueryHandler": ("tests.unit.test_dispatcher_full_coverage", "QueryHandler"),
    "RailwayTestCase": ("tests.test_documented_patterns", "RailwayTestCase"),
    "ReliabilityScenario": ("tests.helpers.scenarios", "ReliabilityScenario"),
    "ReliabilityScenarios": ("tests.helpers.scenarios", "ReliabilityScenarios"),
    "RuntimeCloneService": (
        "tests.unit.test_service_additional",
        "RuntimeCloneService",
    ),
    "SampleCommand": ("tests.unit.test_dispatcher_full_coverage", "SampleCommand"),
    "SampleEvent": ("tests.unit.test_dispatcher_full_coverage", "SampleEvent"),
    "SampleHandler": ("tests.unit.test_dispatcher_full_coverage", "SampleHandler"),
    "SampleQuery": ("tests.unit.test_dispatcher_full_coverage", "SampleQuery"),
    "SendEmailService": ("tests.test_documented_patterns", "SendEmailService"),
    "ServiceConfig": ("tests.integration.test_service", "ServiceConfig"),
    "ServiceFactoryRegistry": ("tests.helpers.factories", "ServiceFactoryRegistry"),
    "ServiceTestCase": ("tests.test_documented_patterns", "ServiceTestCase"),
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
    "Status": ("tests.unit.test_utilities_enum_full_coverage", "Status"),
    "StrictOptionsForTest": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "StrictOptionsForTest",
    ),
    "StringHandler": (
        "tests.unit.test_utilities_type_checker_coverage_100",
        "StringHandler",
    ),
    "StringParserTestFactory": (
        "tests.unit.test_utilities_string_parser",
        "StringParserTestFactory",
    ),
    "SuiteBuilder": (
        "tests.integration.patterns.test_patterns_testing",
        "SuiteBuilder",
    ),
    "T": ("tests.typings", "T"),
    "TMessage": ("tests.unit.test_utilities_type_checker_coverage_100", "TMessage"),
    "T_co": ("tests.typings", "T_co"),
    "T_contra": ("tests.typings", "T_contra"),
    "TestAdvancedPatterns": (
        "tests.integration.patterns.test_patterns_testing",
        "TestAdvancedPatterns",
    ),
    "TestAggregateRoots": ("tests.unit.test_coverage_models", "TestAggregateRoots"),
    "TestAllPatternsIntegration": (
        "tests.test_documented_patterns",
        "TestAllPatternsIntegration",
    ),
    "TestAltPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestAltPropagatesException",
    ),
    "TestAssertExists": ("tests.unit.flext_tests.test_files", "TestAssertExists"),
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
    "TestBackwardCompatDiscardReturnValue": (
        "tests.unit.test_loggings_strict_returns",
        "TestBackwardCompatDiscardReturnValue",
    ),
    "TestBackwardCompatibility": (
        "tests.integration.test_migration_validation",
        "TestBackwardCompatibility",
    ),
    "TestBaseMkGenerationFlow": (
        "tests.integration.test_infra_integration",
        "TestBaseMkGenerationFlow",
    ),
    "TestBatchOperations": ("tests.unit.flext_tests.test_files", "TestBatchOperations"),
    "TestCleanText": ("tests.unit.test_utilities_text_full_coverage", "TestCleanText"),
    "TestCloneContainer": (
        "tests.unit.test_utilities_context_full_coverage",
        "TestCloneContainer",
    ),
    "TestCloneRuntime": (
        "tests.unit.test_utilities_context_full_coverage",
        "TestCloneRuntime",
    ),
    "TestCommands": ("tests.unit.test_coverage_models", "TestCommands"),
    "TestCompleteFlextSystemIntegration": (
        "tests.integration.test_system",
        "TestCompleteFlextSystemIntegration",
    ),
    "TestComprehensiveIntegration": (
        "tests.integration.patterns.test_patterns_testing",
        "TestComprehensiveIntegration",
    ),
    "TestConfig": ("tests.unit.flext_tests.test_factories", "TestConfig"),
    "TestConfigConstants": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "TestConfigConstants",
    ),
    "TestConfigMapDictOps": (
        "tests.unit.test_typings_full_coverage",
        "TestConfigMapDictOps",
    ),
    "TestConfigModels": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "TestConfigModels",
    ),
    "TestConfigServiceViaDI": (
        "tests.unit.test_di_services_access",
        "TestConfigServiceViaDI",
    ),
    "TestContainerDIRealExecution": (
        "tests.unit.test_di_incremental",
        "TestContainerDIRealExecution",
    ),
    "TestContainerInfo": ("tests.unit.flext_tests.test_docker", "TestContainerInfo"),
    "TestContainerIntegration": (
        "tests.integration.test_infra_integration",
        "TestContainerIntegration",
    ),
    "TestContainerMemory": (
        "tests.benchmark.test_container_memory",
        "TestContainerMemory",
    ),
    "TestContainerPerformance": (
        "tests.benchmark.test_container_performance",
        "TestContainerPerformance",
    ),
    "TestContainerStatus": (
        "tests.unit.flext_tests.test_docker",
        "TestContainerStatus",
    ),
    "TestContext100Coverage": (
        "tests.unit.test_context_coverage_100",
        "TestContext100Coverage",
    ),
    "TestContextDataModel": (
        "tests.unit.test_coverage_context",
        "TestContextDataModel",
    ),
    "TestContextServiceViaDI": (
        "tests.unit.test_di_services_access",
        "TestContextServiceViaDI",
    ),
    "TestCorrelationDomain": (
        "tests.unit.test_coverage_context",
        "TestCorrelationDomain",
    ),
    "TestCoveragePush75Percent": (
        "tests.unit.test_final_75_percent_push",
        "TestCoveragePush75Percent",
    ),
    "TestCreateDatetimeProxy": (
        "tests.unit.test_utilities_context_full_coverage",
        "TestCreateDatetimeProxy",
    ),
    "TestCreateDictProxy": (
        "tests.unit.test_utilities_context_full_coverage",
        "TestCreateDictProxy",
    ),
    "TestCreateFromCallableCarriesException": (
        "tests.unit.test_result_exception_carrying",
        "TestCreateFromCallableCarriesException",
    ),
    "TestCreateInStatic": ("tests.unit.flext_tests.test_files", "TestCreateInStatic"),
    "TestCreateStrProxy": (
        "tests.unit.test_utilities_context_full_coverage",
        "TestCreateStrProxy",
    ),
    "TestCriticalReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestCriticalReturnsResultBool",
    ),
    "TestCrossModuleIntegration": (
        "tests.integration.test_infra_integration",
        "TestCrossModuleIntegration",
    ),
    "TestDIBridgeRealExecution": (
        "tests.unit.test_di_incremental",
        "TestDIBridgeRealExecution",
    ),
    "TestDataFactory": ("tests.test_utils", "TestDataFactory"),
    "TestDataGenerators": ("tests.helpers.factories", "TestDataGenerators"),
    "TestDebugReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestDebugReturnsResultBool",
    ),
    "TestDependencyIntegrationRealExecution": (
        "tests.unit.test_di_incremental",
        "TestDependencyIntegrationRealExecution",
    ),
    "TestDictMixinOperations": (
        "tests.unit.test_typings_full_coverage",
        "TestDictMixinOperations",
    ),
    "TestDispatcherDI": ("tests.unit.test_dispatcher_di", "TestDispatcherDI"),
    "TestDomainEvents": ("tests.unit.test_coverage_models", "TestDomainEvents"),
    "TestDomainHashValue": (
        "tests.unit.test_utilities_domain_full_coverage",
        "TestDomainHashValue",
    ),
    "TestDomainLogger": (
        "tests.unit.test_utilities_domain_full_coverage",
        "TestDomainLogger",
    ),
    "TestEdgeCases": ("tests.unit.test_coverage_loggings", "TestEdgeCases"),
    "TestEnterprisePatterns": (
        "tests.integration.patterns.test_architectural_patterns",
        "TestEnterprisePatterns",
    ),
    "TestEntities": ("tests.unit.test_coverage_models", "TestEntities"),
    "TestEntityCoverageEdgeCases": (
        "tests.unit.test_entity_coverage",
        "TestEntityCoverageEdgeCases",
    ),
    "TestErrorOrPatternUnchanged": (
        "tests.unit.test_result_exception_carrying",
        "TestErrorOrPatternUnchanged",
    ),
    "TestErrorReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestErrorReturnsResultBool",
    ),
    "TestEventDrivenPatterns": (
        "tests.integration.patterns.test_architectural_patterns",
        "TestEventDrivenPatterns",
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
        "tests.unit.test_coverage_loggings",
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
    "TestExceptionReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestExceptionReturnsResultBool",
    ),
    "TestExceptionSerialization": (
        "tests.unit.test_coverage_exceptions",
        "TestExceptionSerialization",
    ),
    "TestFacadeDeprecatedAliases": (
        "tests.unit.test_deprecation_warnings",
        "TestFacadeDeprecatedAliases",
    ),
    "TestFactories": ("tests.test_documented_patterns", "TestFactories"),
    "TestFactoriesHelpers": (
        "tests.unit.flext_tests.test_factories",
        "TestFactoriesHelpers",
    ),
    "TestFactoryDecoratorsDiscoveryHasFactories": (
        "tests.unit.test_decorators_discovery_full_coverage",
        "TestFactoryDecoratorsDiscoveryHasFactories",
    ),
    "TestFactoryDecoratorsDiscoveryScanModule": (
        "tests.unit.test_decorators_discovery_full_coverage",
        "TestFactoryDecoratorsDiscoveryScanModule",
    ),
    "TestFactoryPatterns": ("tests.unit.test_coverage_loggings", "TestFactoryPatterns"),
    "TestFailNoExceptionBackwardCompat": (
        "tests.unit.test_result_exception_carrying",
        "TestFailNoExceptionBackwardCompat",
    ),
    "TestFailWithException": (
        "tests.unit.test_result_exception_carrying",
        "TestFailWithException",
    ),
    "TestFileInfo": ("tests.unit.flext_tests.test_files", "TestFileInfo"),
    "TestFileInfoFromModels": (
        "tests.unit.flext_tests.test_files",
        "TestFileInfoFromModels",
    ),
    "TestFixtureFactory": ("tests.test_utils", "TestFixtureFactory"),
    "TestFlatMapPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestFlatMapPropagatesException",
    ),
    "TestFlextCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "TestFlextCommand",
    ),
    "TestFlextCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "TestFlextCommandHandler",
    ),
    "TestFlextCommandResults": (
        "tests.integration.patterns.test_patterns_commands",
        "TestFlextCommandResults",
    ),
    "TestFlextConstants": ("tests.unit.test_constants", "TestFlextConstants"),
    "TestFlextContainer": ("tests.unit.test_container", "TestFlextContainer"),
    "TestFlextContext": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextContext",
    ),
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
    "TestFlextLogLevel": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLogLevel",
    ),
    "TestFlextLogger": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLogger",
    ),
    "TestFlextLoggerIntegration": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLoggerIntegration",
    ),
    "TestFlextLoggerUsage": (
        "tests.integration.patterns.test_patterns_logging",
        "TestFlextLoggerUsage",
    ),
    "TestFlextMixinsNestedClasses": (
        "tests.unit.test_mixins",
        "TestFlextMixinsNestedClasses",
    ),
    "TestFlextModels": ("tests.unit.test_models", "TestFlextModels"),
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
    "TestFlextServiceIntegration": (
        "tests.integration.test_service",
        "TestFlextServiceIntegration",
    ),
    "TestFlextSettings": ("tests.unit.test_config", "TestFlextSettings"),
    "TestFlextSettingsSingletonIntegration": (
        "tests.integration.test_config_integration",
        "TestFlextSettingsSingletonIntegration",
    ),
    "TestFlextTestsBuilders": (
        "tests.unit.flext_tests.test_builders",
        "TestFlextTestsBuilders",
    ),
    "TestFlextTestsDocker": (
        "tests.unit.flext_tests.test_docker",
        "TestFlextTestsDocker",
    ),
    "TestFlextTestsDockerWorkerId": (
        "tests.unit.flext_tests.test_docker",
        "TestFlextTestsDockerWorkerId",
    ),
    "TestFlextTestsDockerWorkspaceRoot": (
        "tests.unit.flext_tests.test_docker",
        "TestFlextTestsDockerWorkspaceRoot",
    ),
    "TestFlextTestsDomains": (
        "tests.unit.flext_tests.test_domains",
        "TestFlextTestsDomains",
    ),
    "TestFlextTestsFactoriesModernAPI": (
        "tests.unit.flext_tests.test_factories",
        "TestFlextTestsFactoriesModernAPI",
    ),
    "TestFlextTestsFiles": ("tests.unit.flext_tests.test_files", "TestFlextTestsFiles"),
    "TestFlextTestsFilesNewApi": (
        "tests.unit.flext_tests.test_files",
        "TestFlextTestsFilesNewApi",
    ),
    "TestFlextTestsMatchers": (
        "tests.unit.flext_tests.test_matchers",
        "TestFlextTestsMatchers",
    ),
    "TestFlextTestsUtilitiesFactory": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesFactory",
    ),
    "TestFlextTestsUtilitiesResult": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesResult",
    ),
    "TestFlextTestsUtilitiesResultCompat": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesResultCompat",
    ),
    "TestFlextTestsUtilitiesTestContext": (
        "tests.unit.flext_tests.test_utilities",
        "TestFlextTestsUtilitiesTestContext",
    ),
    "TestFlextTypings": ("tests.unit.test_typings", "TestFlextTypings"),
    "TestFlextUtilitiesArgs": (
        "tests.unit.test_args_coverage_100",
        "TestFlextUtilitiesArgs",
    ),
    "TestFlextUtilitiesConfiguration": (
        "tests.unit.test_utilities_configuration_coverage_100",
        "TestFlextUtilitiesConfiguration",
    ),
    "TestFlextUtilitiesModelNormalizeToMetadata": (
        "tests.unit.test_models_container",
        "TestFlextUtilitiesModelNormalizeToMetadata",
    ),
    "TestFlextUtilitiesReliability": (
        "tests.unit.test_utilities_reliability",
        "TestFlextUtilitiesReliability",
    ),
    "TestFlextVersion": ("tests.unit.test_version", "TestFlextVersion"),
    "TestFormatAppId": (
        "tests.unit.test_utilities_text_full_coverage",
        "TestFormatAppId",
    ),
    "TestFromValidationCarriesException": (
        "tests.unit.test_result_exception_carrying",
        "TestFromValidationCarriesException",
    ),
    "TestFunction": (
        "tests.integration.patterns.test_advanced_patterns",
        "TestFunction",
    ),
    "TestGlobalContextManagement": (
        "tests.unit.test_coverage_loggings",
        "TestGlobalContextManagement",
    ),
    "TestGuardsDeprecatedMethods": (
        "tests.unit.test_deprecation_warnings",
        "TestGuardsDeprecatedMethods",
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
    "TestHelperConsolidationTransformer": (
        "tests.unit.test_transformer_helper_consolidation",
        "TestHelperConsolidationTransformer",
    ),
    "TestHierarchicalExceptionSystem": (
        "tests.unit.test_coverage_exceptions",
        "TestHierarchicalExceptionSystem",
    ),
    "TestIdempotency": (
        "tests.integration.test_refactor_nesting_idempotency",
        "TestIdempotency",
    ),
    "TestInfoReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestInfoReturnsResultBool",
    ),
    "TestInfoWithContentMeta": (
        "tests.unit.flext_tests.test_files",
        "TestInfoWithContentMeta",
    ),
    "TestInstanceCreation": (
        "tests.unit.test_coverage_loggings",
        "TestInstanceCreation",
    ),
    "TestIntegrationWithRealCommandServices": (
        "tests.integration.test_infra_integration",
        "TestIntegrationWithRealCommandServices",
    ),
    "TestLashPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestLashPropagatesException",
    ),
    "TestLevelBasedContextManagement": (
        "tests.unit.test_coverage_loggings",
        "TestLevelBasedContextManagement",
    ),
    "TestLibraryIntegration": (
        "tests.integration.test_integration",
        "TestLibraryIntegration",
    ),
    "TestLogReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestLogReturnsResultBool",
    ),
    "TestLoggerServiceViaDI": (
        "tests.unit.test_di_services_access",
        "TestLoggerServiceViaDI",
    ),
    "TestLoggingIntegration": (
        "tests.unit.test_coverage_loggings",
        "TestLoggingIntegration",
    ),
    "TestLoggingMethods": ("tests.unit.test_coverage_loggings", "TestLoggingMethods"),
    "TestLoggingsErrorPaths": (
        "tests.unit.test_loggings_error_paths_coverage",
        "TestLoggingsErrorPaths",
    ),
    "TestMapPropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestMapPropagatesException",
    ),
    "TestMapperBuildFlagsDict": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperBuildFlagsDict",
    ),
    "TestMapperCollectActiveKeys": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperCollectActiveKeys",
    ),
    "TestMapperDeprecatedMethods": (
        "tests.unit.test_deprecation_warnings",
        "TestMapperDeprecatedMethods",
    ),
    "TestMapperFilterDict": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperFilterDict",
    ),
    "TestMapperInvertDict": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperInvertDict",
    ),
    "TestMapperMapDictKeys": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperMapDictKeys",
    ),
    "TestMapperTransformValues": (
        "tests.unit.test_utilities_data_mapper",
        "TestMapperTransformValues",
    ),
    "TestMetadata": ("tests.unit.test_coverage_models", "TestMetadata"),
    "TestMigrationComplexity": (
        "tests.integration.test_migration_validation",
        "TestMigrationComplexity",
    ),
    "TestMigrationScenario1": (
        "tests.integration.test_migration_validation",
        "TestMigrationScenario1",
    ),
    "TestMigrationScenario2": (
        "tests.integration.test_migration_validation",
        "TestMigrationScenario2",
    ),
    "TestMigrationScenario4": (
        "tests.integration.test_migration_validation",
        "TestMigrationScenario4",
    ),
    "TestMigrationScenario5": (
        "tests.integration.test_migration_validation",
        "TestMigrationScenario5",
    ),
    "TestModelIntegration": ("tests.unit.test_coverage_models", "TestModelIntegration"),
    "TestModelSerialization": (
        "tests.unit.test_coverage_models",
        "TestModelSerialization",
    ),
    "TestModelValidation": ("tests.unit.test_coverage_models", "TestModelValidation"),
    "TestMonadicOperationsUnchanged": (
        "tests.unit.test_result_exception_carrying",
        "TestMonadicOperationsUnchanged",
    ),
    "TestOkNoneGuardStillRaises": (
        "tests.unit.test_result_exception_carrying",
        "TestOkNoneGuardStillRaises",
    ),
    "TestOutputSingletonConsistency": (
        "tests.integration.test_infra_integration",
        "TestOutputSingletonConsistency",
    ),
    "TestPathResolverDiscoveryFlow": (
        "tests.integration.test_infra_integration",
        "TestPathResolverDiscoveryFlow",
    ),
    "TestPattern1V1Explicit": (
        "tests.test_documented_patterns",
        "TestPattern1V1Explicit",
    ),
    "TestPattern2V2Property": (
        "tests.test_documented_patterns",
        "TestPattern2V2Property",
    ),
    "TestPattern3RailwayV1": (
        "tests.test_documented_patterns",
        "TestPattern3RailwayV1",
    ),
    "TestPattern4RailwayV2Property": (
        "tests.test_documented_patterns",
        "TestPattern4RailwayV2Property",
    ),
    "TestPattern5MonadicComposition": (
        "tests.test_documented_patterns",
        "TestPattern5MonadicComposition",
    ),
    "TestPattern6ErrorHandling": (
        "tests.test_documented_patterns",
        "TestPattern6ErrorHandling",
    ),
    "TestPattern7AutomaticInfrastructure": (
        "tests.test_documented_patterns",
        "TestPattern7AutomaticInfrastructure",
    ),
    "TestPattern8MultipleOperations": (
        "tests.test_documented_patterns",
        "TestPattern8MultipleOperations",
    ),
    "TestPerformanceAnalysis": (
        "tests.integration.patterns.test_patterns_testing",
        "TestPerformanceAnalysis",
    ),
    "TestPerformanceBenchmarks": (
        "tests.benchmark.test_refactor_nesting_performance",
        "TestPerformanceBenchmarks",
    ),
    "TestPerformanceDomain": (
        "tests.unit.test_coverage_context",
        "TestPerformanceDomain",
    ),
    "TestPhase2FinalCoveragePush": (
        "tests.unit.test_phase2_coverage_final",
        "TestPhase2FinalCoveragePush",
    ),
    "TestProjectLevelRefactor": (
        "tests.integration.test_refactor_nesting_project",
        "TestProjectLevelRefactor",
    ),
    "TestPropertyBasedPatterns": (
        "tests.integration.patterns.test_patterns_testing",
        "TestPropertyBasedPatterns",
    ),
    "TestProtocolComplianceStructlogLogger": (
        "tests.unit.test_loggings_strict_returns",
        "TestProtocolComplianceStructlogLogger",
    ),
    "TestQueries": ("tests.unit.test_coverage_models", "TestQueries"),
    "TestRealWiringScenarios": (
        "tests.unit.test_di_incremental",
        "TestRealWiringScenarios",
    ),
    "TestRealWorldScenarios": (
        "tests.integration.patterns.test_patterns_testing",
        "TestRealWorldScenarios",
    ),
    "TestResultBasics": ("tests.unit.test_coverage_76_lines", "TestResultBasics"),
    "TestResultTransformations": (
        "tests.unit.test_coverage_76_lines",
        "TestResultTransformations",
    ),
    "TestRuntimeDeprecatedNormalizeMethods": (
        "tests.unit.test_deprecation_warnings",
        "TestRuntimeDeprecatedNormalizeMethods",
    ),
    "TestRuntimeDictLike": (
        "tests.unit.test_runtime_coverage_100",
        "TestRuntimeDictLike",
    ),
    "TestRuntimeTypeChecking": (
        "tests.unit.test_runtime_coverage_100",
        "TestRuntimeTypeChecking",
    ),
    "TestSafeCarriesException": (
        "tests.unit.test_result_exception_carrying",
        "TestSafeCarriesException",
    ),
    "TestSafeString": (
        "tests.unit.test_utilities_text_full_coverage",
        "TestSafeString",
    ),
    "TestScopedContextManagement": (
        "tests.unit.test_coverage_loggings",
        "TestScopedContextManagement",
    ),
    "TestService": ("tests.unit.test_service_coverage_100", "TestService"),
    "TestService100Coverage": (
        "tests.unit.test_service_coverage_100",
        "TestService100Coverage",
    ),
    "TestServiceBootstrap": (
        "tests.unit.test_service_bootstrap",
        "TestServiceBootstrap",
    ),
    "TestServiceBootstrapWithDI": (
        "tests.unit.test_di_incremental",
        "TestServiceBootstrapWithDI",
    ),
    "TestServiceDomain": ("tests.unit.test_coverage_context", "TestServiceDomain"),
    "TestServiceResultProperty": (
        "tests.test_service_result_property",
        "TestServiceResultProperty",
    ),
    "TestServiceWithValidation": (
        "tests.unit.test_service_coverage_100",
        "TestServiceWithValidation",
    ),
    "TestServicerChaining": (
        "tests.integration.test_infra_integration",
        "TestServicerChaining",
    ),
    "TestServicesIntegrationViaDI": (
        "tests.unit.test_di_services_access",
        "TestServicesIntegrationViaDI",
    ),
    "TestShortAlias": ("tests.unit.flext_tests.test_files", "TestShortAlias"),
    "TestStrictContainerNormalization": (
        "tests.unit.test_deprecation_warnings",
        "TestStrictContainerNormalization",
    ),
    "TestTextLogger": (
        "tests.unit.test_utilities_text_full_coverage",
        "TestTextLogger",
    ),
    "TestTimeoutEnforcerCleanup": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TestTimeoutEnforcerCleanup",
    ),
    "TestTimeoutEnforcerEdgeCases": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TestTimeoutEnforcerEdgeCases",
    ),
    "TestTimeoutEnforcerExecutorManagement": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TestTimeoutEnforcerExecutorManagement",
    ),
    "TestTimeoutEnforcerInitialization": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TestTimeoutEnforcerInitialization",
    ),
    "TestTraceReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestTraceReturnsResultBool",
    ),
    "TestTraversePropagatesException": (
        "tests.unit.test_result_exception_carrying",
        "TestTraversePropagatesException",
    ),
    "TestUser": ("tests.unit.flext_tests.test_factories", "TestUser"),
    "TestUtilitiesCoverage": (
        "tests.unit.test_utilities_coverage",
        "TestUtilitiesCoverage",
    ),
    "TestUtilitiesDomain": ("tests.unit.test_coverage_context", "TestUtilitiesDomain"),
    "TestValidateValueImmutable": (
        "tests.unit.test_utilities_domain_full_coverage",
        "TestValidateValueImmutable",
    ),
    "TestValidatorCallable": (
        "tests.unit.test_typings_full_coverage",
        "TestValidatorCallable",
    ),
    "TestValidatorMapMixin": (
        "tests.unit.test_typings_full_coverage",
        "TestValidatorMapMixin",
    ),
    "TestValues": ("tests.unit.test_coverage_models", "TestValues"),
    "TestWarningReturnsResultBool": (
        "tests.unit.test_loggings_strict_returns",
        "TestWarningReturnsResultBool",
    ),
    "TestWorkspaceDetectionOrchestrationFlow": (
        "tests.integration.test_infra_integration",
        "TestWorkspaceDetectionOrchestrationFlow",
    ),
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
    "TestsFlextTestsFactoriesDict": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesDict",
    ),
    "TestsFlextTestsFactoriesGeneric": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesGeneric",
    ),
    "TestsFlextTestsFactoriesList": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesList",
    ),
    "TestsFlextTestsFactoriesModel": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesModel",
    ),
    "TestsFlextTestsFactoriesRes": (
        "tests.unit.flext_tests.test_factories",
        "TestsFlextTestsFactoriesRes",
    ),
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
    "TestuCollectionBatch": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionBatch",
    ),
    "TestuCollectionChunk": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionChunk",
    ),
    "TestuCollectionCoerceDictValidator": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionCoerceDictValidator",
    ),
    "TestuCollectionCoerceListValidator": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionCoerceListValidator",
    ),
    "TestuCollectionCount": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionCount",
    ),
    "TestuCollectionFilter": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionFilter",
    ),
    "TestuCollectionFind": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionFind",
    ),
    "TestuCollectionGroup": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionGroup",
    ),
    "TestuCollectionMap": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionMap",
    ),
    "TestuCollectionMerge": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionMerge",
    ),
    "TestuCollectionParseMapping": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionParseMapping",
    ),
    "TestuCollectionParseSequence": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionParseSequence",
    ),
    "TestuCollectionProcess": (
        "tests.unit.test_utilities_collection_coverage_100",
        "TestuCollectionProcess",
    ),
    "TestuDomain": ("tests.unit.test_utilities_domain", "TestuDomain"),
    "TestuEnumCoerceByNameValidator": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumCoerceByNameValidator",
    ),
    "TestuEnumCoerceValidator": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumCoerceValidator",
    ),
    "TestuEnumIsMember": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumIsMember",
    ),
    "TestuEnumIsSubset": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumIsSubset",
    ),
    "TestuEnumMetadata": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumMetadata",
    ),
    "TestuEnumParse": ("tests.unit.test_enum_utilities_coverage_100", "TestuEnumParse"),
    "TestuEnumParseOrDefault": (
        "tests.unit.test_enum_utilities_coverage_100",
        "TestuEnumParseOrDefault",
    ),
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
    "TestuPaginationBuildPaginationResponse": (
        "tests.unit.test_pagination_coverage_100",
        "TestuPaginationBuildPaginationResponse",
    ),
    "TestuPaginationExtractPageParams": (
        "tests.unit.test_pagination_coverage_100",
        "TestuPaginationExtractPageParams",
    ),
    "TestuPaginationExtractPaginationConfig": (
        "tests.unit.test_pagination_coverage_100",
        "TestuPaginationExtractPaginationConfig",
    ),
    "TestuPaginationPreparePaginationData": (
        "tests.unit.test_pagination_coverage_100",
        "TestuPaginationPreparePaginationData",
    ),
    "TestuPaginationValidatePaginationParams": (
        "tests.unit.test_pagination_coverage_100",
        "TestuPaginationValidatePaginationParams",
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
    "TextLike": ("tests.unit.test_utilities_enum_full_coverage", "TextLike"),
    "TextUtilityContract": (
        "tests.unit.contracts.text_contract",
        "TextUtilityContract",
    ),
    "ThreadSafetyTest": (
        "tests.integration.test_config_integration",
        "ThreadSafetyTest",
    ),
    "TimeoutEnforcerScenarios": (
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TimeoutEnforcerScenarios",
    ),
    "TypeGuardsScenarios": (
        "tests.unit.test_utilities_type_guards_coverage_100",
        "TypeGuardsScenarios",
    ),
    "UnknownHint": ("tests.unit.test_utilities_args_full_coverage", "UnknownHint"),
    "UnregisteredCommand": (
        "tests.unit.test_dispatcher_full_coverage",
        "UnregisteredCommand",
    ),
    "UpdateUserCommand": (
        "tests.integration.patterns.test_patterns_commands",
        "UpdateUserCommand",
    ),
    "UpdateUserCommandHandler": (
        "tests.integration.patterns.test_patterns_commands",
        "UpdateUserCommandHandler",
    ),
    "User": ("tests.test_documented_patterns", "User"),
    "UserFactory": ("tests.helpers.factories", "UserFactory"),
    "UserQueryService": ("tests.integration.test_service", "UserQueryService"),
    "UserServiceEntity": ("tests.integration.test_service", "UserServiceEntity"),
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
    "ValidationScenario": ("tests.helpers.scenarios", "ValidationScenario"),
    "ValidationScenarios": ("tests.helpers.scenarios", "ValidationScenarios"),
    "ValidationService": ("tests.test_documented_patterns", "ValidationService"),
    "arrange_act_assert": (
        "tests.integration.patterns.test_patterns_testing",
        "arrange_act_assert",
    ),
    "assert_rejects": ("tests.conftest", "assert_rejects"),
    "assert_result_success": (
        "tests.integration.patterns.test_patterns_logging",
        "assert_result_success",
    ),
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
    "dispatcher": ("tests.unit.test_dispatcher_full_coverage", "dispatcher"),
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
    "integration": ("tests.integration", ""),
    "invalid_hostnames": ("tests.conftest", "invalid_hostnames"),
    "invalid_port_numbers": ("tests.conftest", "invalid_port_numbers"),
    "invalid_uris": ("tests.conftest", "invalid_uris"),
    "m": ("tests.models", "m"),
    "make_result_logger": (
        "tests.integration.patterns.test_patterns_logging",
        "make_result_logger",
    ),
    "mapper": ("tests.unit.test_utilities_mapper_full_coverage", "mapper"),
    "mark_test_pattern": (
        "tests.integration.patterns.test_patterns_testing",
        "mark_test_pattern",
    ),
    "mock_external_service": ("tests.conftest", "mock_external_service"),
    "out_of_range": ("tests.conftest", "out_of_range"),
    "p": ("tests.protocols", "p"),
    "parser_scenarios": ("tests.conftest", "parser_scenarios"),
    "patterns": ("tests.integration.patterns", ""),
    "pytestmark": ("tests.integration.test_service", "pytestmark"),
    "r": ("tests.unit.flext_tests.test_utilities", "TestFlextTestsUtilitiesResult"),
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
    "test_additional_container_branches_cover_fluent_and_lookup_paths": (
        "tests.unit.test_container_full_coverage",
        "test_additional_container_branches_cover_fluent_and_lookup_paths",
    ),
    "test_additional_register_factory_and_unregister_paths": (
        "tests.unit.test_container_full_coverage",
        "test_additional_register_factory_and_unregister_paths",
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
    "test_auto_value_lowercases_input": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_auto_value_lowercases_input",
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
    "test_batch_fail_collect_flatten_and_progress": (
        "tests.unit.test_utilities_collection_full_coverage",
        "test_batch_fail_collect_flatten_and_progress",
    ),
    "test_bi_map_returns_forward_copy_and_inverse": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_bi_map_returns_forward_copy_and_inverse",
    ),
    "test_bind_operation_context_without_ensure_correlation_and_bind_failure": (
        "tests.unit.test_decorators_full_coverage",
        "test_bind_operation_context_without_ensure_correlation_and_bind_failure",
    ),
    "test_build_apply_transform_and_process_error_paths": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_build_apply_transform_and_process_error_paths",
    ),
    "test_build_options_invalid_only_kwargs_returns_base": (
        "tests.unit.test_utilities_configuration_full_coverage",
        "test_build_options_invalid_only_kwargs_returns_base",
    ),
    "test_builder": ("tests.unit.test_container_full_coverage", "test_builder"),
    "test_callable_registration_with_attribute": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_callable_registration_with_attribute",
    ),
    "test_canonical_aliases_are_available": (
        "tests.unit.test_models_generic_full_coverage",
        "test_canonical_aliases_are_available",
    ),
    "test_categories_clear_and_symbols_are_available": (
        "tests.unit.test_models_collections_full_coverage",
        "test_categories_clear_and_symbols_are_available",
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
    "test_checker_logger_and_safe_type_hints_fallback": (
        "tests.unit.test_utilities_checker_full_coverage",
        "test_checker_logger_and_safe_type_hints_fallback",
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
    "test_clear_operation_scope_and_handle_log_result_paths": (
        "tests.unit.test_decorators_full_coverage",
        "test_clear_operation_scope_and_handle_log_result_paths",
    ),
    "test_collection_batch_failure_error_capture_and_parse_sequence_outer_error": (
        "tests.unit.test_utilities_collection_full_coverage",
        "test_collection_batch_failure_error_capture_and_parse_sequence_outer_error",
    ),
    "test_combined_with_and_without_railway_uses_injection": (
        "tests.unit.test_decorators_full_coverage",
        "test_combined_with_and_without_railway_uses_injection",
    ),
    "test_command_pagination_limit": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_command_pagination_limit",
    ),
    "test_config_bridge_and_trace_context_and_http_validation": (
        "tests.unit.test_runtime_full_coverage",
        "test_config_bridge_and_trace_context_and_http_validation",
    ),
    "test_config_context_properties_and_defaults": (
        "tests.unit.test_container_full_coverage",
        "test_config_context_properties_and_defaults",
    ),
    "test_config_hash_from_mapping_and_non_hashable": (
        "tests.unit.test_models_collections_full_coverage",
        "test_config_hash_from_mapping_and_non_hashable",
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
    "test_configure_with_resource_register_and_factory_error_paths": (
        "tests.unit.test_container_full_coverage",
        "test_configure_with_resource_register_and_factory_error_paths",
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
    "test_container_remaining_branch_paths_in_sync_factory_and_getters": (
        "tests.unit.test_container_full_coverage",
        "test_container_remaining_branch_paths_in_sync_factory_and_getters",
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
    "test_create_auto_register_factories_path": (
        "tests.unit.test_container_full_coverage",
        "test_create_auto_register_factories_path",
    ),
    "test_create_auto_register_factory_wrapper_callable_and_non_callable": (
        "tests.unit.test_container_full_coverage",
        "test_create_auto_register_factory_wrapper_callable_and_non_callable",
    ),
    "test_create_discriminated_union_multiple_enums": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_create_discriminated_union_multiple_enums",
    ),
    "test_create_enum_executes_factory_path": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_create_enum_executes_factory_path",
    ),
    "test_create_from_callable_and_repr": (
        "tests.unit.test_result_additional",
        "test_create_from_callable_and_repr",
    ),
    "test_create_from_callable_branches": (
        "tests.unit.test_handlers_full_coverage",
        "test_create_from_callable_branches",
    ),
    "test_create_merges_metadata_dict_branch": (
        "tests.unit.test_context_full_coverage",
        "test_create_merges_metadata_dict_branch",
    ),
    "test_create_overloads_and_auto_correlation": (
        "tests.unit.test_context_full_coverage",
        "test_create_overloads_and_auto_correlation",
    ),
    "test_create_scoped_instance_and_scoped_additional_branches": (
        "tests.unit.test_container_full_coverage",
        "test_create_scoped_instance_and_scoped_additional_branches",
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
    "test_deprecated_normalize_to_general_value_warns": (
        "tests.unit.test_runtime_full_coverage",
        "test_deprecated_normalize_to_general_value_warns",
    ),
    "test_deprecated_normalize_to_metadata_value_warns": (
        "tests.unit.test_runtime_full_coverage",
        "test_deprecated_normalize_to_metadata_value_warns",
    ),
    "test_deprecated_wrapper_emits_warning_and_returns_value": (
        "tests.unit.test_decorators_full_coverage",
        "test_deprecated_wrapper_emits_warning_and_returns_value",
    ),
    "test_discover_project_roots_without_nested_git_dirs": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_discover_project_roots_without_nested_git_dirs",
    ),
    "test_discovery_narrowed_function_paths": (
        "tests.unit.test_handlers_full_coverage",
        "test_discovery_narrowed_function_paths",
    ),
    "test_dispatch_after_handler_removal_fails": (
        "tests.unit.test_dispatcher_minimal",
        "test_dispatch_after_handler_removal_fails",
    ),
    "test_dispatch_auto_discovery_handler": (
        "tests.unit.test_dispatcher_minimal",
        "test_dispatch_auto_discovery_handler",
    ),
    "test_dispatch_coerce_mode_with_enum_string_and_other_object": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_dispatch_coerce_mode_with_enum_string_and_other_object",
    ),
    "test_dispatch_command_success": (
        "tests.unit.test_dispatcher_minimal",
        "test_dispatch_command_success",
    ),
    "test_dispatch_handler_exception_returns_failure": (
        "tests.unit.test_dispatcher_minimal",
        "test_dispatch_handler_exception_returns_failure",
    ),
    "test_dispatch_invalid_input_types": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_dispatch_invalid_input_types",
    ),
    "test_dispatch_is_member_by_name_and_by_value": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_dispatch_is_member_by_name_and_by_value",
    ),
    "test_dispatch_is_name_mode": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_dispatch_is_name_mode",
    ),
    "test_dispatch_no_handler_fails": (
        "tests.unit.test_dispatcher_minimal",
        "test_dispatch_no_handler_fails",
    ),
    "test_dispatch_parse_mode_with_enum_string_and_other_object": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_dispatch_parse_mode_with_enum_string_and_other_object",
    ),
    "test_dispatch_unknown_mode_raises": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_dispatch_unknown_mode_raises",
    ),
    "test_dispatcher_family_blocks_models_target": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_dispatcher_family_blocks_models_target",
    ),
    "test_dispatcher_reliability_branch_paths": (
        "tests.unit.test_dispatcher_reliability_full_coverage",
        "test_dispatcher_reliability_branch_paths",
    ),
    "test_enrich_and_ensure_trace_context_branches": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_enrich_and_ensure_trace_context_branches",
    ),
    "test_ensure_and_extract_array_index_helpers": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_ensure_and_extract_array_index_helpers",
    ),
    "test_ensure_dict_branches": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_ensure_dict_branches",
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
    "test_event_publishing_strict": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_event_publishing_strict",
    ),
    "test_exception_handling_in_dispatch": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_exception_handling_in_dispatch",
    ),
    "test_exceptions_uncovered_metadata_paths": (
        "tests.unit.test_exceptions_full_coverage",
        "test_exceptions_uncovered_metadata_paths",
    ),
    "test_execute_and_register_handler_failure_paths": (
        "tests.unit.test_registry_full_coverage",
        "test_execute_and_register_handler_failure_paths",
    ),
    "test_execute_retry_exponential_and_handle_exhaustion_raise_last_exception": (
        "tests.unit.test_decorators_full_coverage",
        "test_execute_retry_exponential_and_handle_exhaustion_raise_last_exception",
    ),
    "test_execute_retry_loop_covers_default_linear_and_never_ran": (
        "tests.unit.test_decorators_full_coverage",
        "test_execute_retry_loop_covers_default_linear_and_never_ran",
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
    "test_extract_message_type_annotation_and_dict_subclass_paths": (
        "tests.unit.test_utilities_checker_full_coverage",
        "test_extract_message_type_annotation_and_dict_subclass_paths",
    ),
    "test_extract_message_type_from_handle_with_only_self": (
        "tests.unit.test_utilities_checker_full_coverage",
        "test_extract_message_type_from_handle_with_only_self",
    ),
    "test_extract_message_type_from_parameter_branches": (
        "tests.unit.test_utilities_checker_full_coverage",
        "test_extract_message_type_from_parameter_branches",
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
    "test_find_mapping_no_match_and_merge_error_paths": (
        "tests.unit.test_utilities_collection_full_coverage",
        "test_find_mapping_no_match_and_merge_error_paths",
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
    "test_frozen_value_model_equality_and_hash": (
        "tests.unit.test_models_base_full_coverage",
        "test_frozen_value_model_equality_and_hash",
    ),
    "test_general_value_helpers_and_logger": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_general_value_helpers_and_logger",
    ),
    "test_generate_special_paths_and_dynamic_subclass": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_generate_special_paths_and_dynamic_subclass",
    ),
    "test_generators_additional_missed_paths": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_generators_additional_missed_paths",
    ),
    "test_generators_mapping_non_dict_normalization_path": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_generators_mapping_non_dict_normalization_path",
    ),
    "test_get_and_get_typed_resource_factory_paths": (
        "tests.unit.test_container_full_coverage",
        "test_get_and_get_typed_resource_factory_paths",
    ),
    "test_get_enum_values_returns_immutable_sequence": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_get_enum_values_returns_immutable_sequence",
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
    "test_handle_log_result_without_fallback_logger_and_non_dict_like_extra": (
        "tests.unit.test_decorators_full_coverage",
        "test_handle_log_result_without_fallback_logger_and_non_dict_like_extra",
    ),
    "test_handle_retry_exhaustion_falsey_exception_reaches_timeout_error": (
        "tests.unit.test_decorators_full_coverage",
        "test_handle_retry_exhaustion_falsey_exception_reaches_timeout_error",
    ),
    "test_handler_attribute_discovery": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_handler_attribute_discovery",
    ),
    "test_handler_builder_fluent_methods": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_handler_builder_fluent_methods",
    ),
    "test_handler_type_literal_and_invalid": (
        "tests.unit.test_handlers_full_coverage",
        "test_handler_type_literal_and_invalid",
    ),
    "test_helper_consolidation_is_prechecked": (
        "tests.unit.test_refactor_policy_family_rules",
        "test_helper_consolidation_is_prechecked",
    ),
    "test_identifiable_unique_id_empty_rejected": (
        "tests.unit.test_models_base_full_coverage",
        "test_identifiable_unique_id_empty_rejected",
    ),
    "test_inactive_and_none_value_paths": (
        "tests.unit.test_context_full_coverage",
        "test_inactive_and_none_value_paths",
    ),
    "test_init_fallback_and_lazy_returns_result_property": (
        "tests.unit.test_result_full_coverage",
        "test_init_fallback_and_lazy_returns_result_property",
    ),
    "test_initialize_di_components_error_paths": (
        "tests.unit.test_container_full_coverage",
        "test_initialize_di_components_error_paths",
    ),
    "test_initialize_di_components_second_type_error_branch": (
        "tests.unit.test_container_full_coverage",
        "test_initialize_di_components_second_type_error_branch",
    ),
    "test_inject_sets_missing_dependency_from_container": (
        "tests.unit.test_decorators_full_coverage",
        "test_inject_sets_missing_dependency_from_container",
    ),
    "test_invalid_handler_mode_init_raises": (
        "tests.unit.test_handlers_full_coverage",
        "test_invalid_handler_mode_init_raises",
    ),
    "test_invalid_registration_attempts": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_invalid_registration_attempts",
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
    "test_is_general_value_list_accepts_list_subclass": (
        "tests.unit.test_utilities_collection_full_coverage",
        "test_is_general_value_list_accepts_list_subclass",
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
    "test_log_operation_track_perf_exception_adds_duration": (
        "tests.unit.test_decorators_full_coverage",
        "test_log_operation_track_perf_exception_adds_duration",
    ),
    "test_loggings_bind_clear_level_error_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_bind_clear_level_error_paths",
    ),
    "test_loggings_context_and_factory_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_context_and_factory_paths",
    ),
    "test_loggings_exception_and_adapter_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_exception_and_adapter_paths",
    ),
    "test_loggings_instance_and_message_format_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_instance_and_message_format_paths",
    ),
    "test_loggings_remaining_branch_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_remaining_branch_paths",
    ),
    "test_loggings_source_and_log_error_paths": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_source_and_log_error_paths",
    ),
    "test_loggings_uncovered_level_trace_path_and_exception_guards": (
        "tests.unit.test_loggings_full_coverage",
        "test_loggings_uncovered_level_trace_path_and_exception_guards",
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
    "test_members_uses_cache_on_second_call": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_members_uses_cache_on_second_call",
    ),
    "test_merge_defaults_and_dump_paths": (
        "tests.unit.test_utilities_model_full_coverage",
        "test_merge_defaults_and_dump_paths",
    ),
    "test_merge_metadata_context_paths": (
        "tests.unit.test_exceptions_full_coverage",
        "test_merge_metadata_context_paths",
    ),
    "test_metadata_attributes_accepts_basemodel_mapping": (
        "tests.unit.test_models_base_full_coverage",
        "test_metadata_attributes_accepts_basemodel_mapping",
    ),
    "test_metadata_attributes_accepts_none": (
        "tests.unit.test_models_base_full_coverage",
        "test_metadata_attributes_accepts_none",
    ),
    "test_metadata_attributes_accepts_t_dict_and_mapping": (
        "tests.unit.test_models_base_full_coverage",
        "test_metadata_attributes_accepts_t_dict_and_mapping",
    ),
    "test_metadata_attributes_rejects_basemodel_non_mapping_dump": (
        "tests.unit.test_models_base_full_coverage",
        "test_metadata_attributes_rejects_basemodel_non_mapping_dump",
    ),
    "test_metadata_attributes_rejects_non_mapping": (
        "tests.unit.test_models_base_full_coverage",
        "test_metadata_attributes_rejects_non_mapping",
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
    "test_misc_unregistration_clear_and_reset": (
        "tests.unit.test_container_full_coverage",
        "test_misc_unregistration_clear_and_reset",
    ),
    "test_mixins_container_registration_and_logger_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_container_registration_and_logger_paths",
    ),
    "test_mixins_context_logging_and_cqrs_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_context_logging_and_cqrs_paths",
    ),
    "test_mixins_context_stack_pop_initializes_missing_stack_attr": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_context_stack_pop_initializes_missing_stack_attr",
    ),
    "test_mixins_remaining_branch_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_remaining_branch_paths",
    ),
    "test_mixins_result_and_model_conversion_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_result_and_model_conversion_paths",
    ),
    "test_mixins_runtime_bootstrap_and_track_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_runtime_bootstrap_and_track_paths",
    ),
    "test_mixins_validation_and_protocol_paths": (
        "tests.unit.test_mixins_full_coverage",
        "test_mixins_validation_and_protocol_paths",
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
    "test_mro_resolver_accepts_expected_order": (
        "tests.integration.test_refactor_policy_mro",
        "test_mro_resolver_accepts_expected_order",
    ),
    "test_mro_resolver_rejects_wrong_order": (
        "tests.integration.test_refactor_policy_mro",
        "test_mro_resolver_rejects_wrong_order",
    ),
    "test_mro_scanner_includes_constants_variants_in_all_scopes": (
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_mro_scanner_includes_constants_variants_in_all_scopes",
    ),
    "test_names_uses_cache_on_second_call": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_names_uses_cache_on_second_call",
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
    "test_normalize_context_to_dict_error_paths": (
        "tests.unit.test_utilities_generators_full_coverage",
        "test_normalize_context_to_dict_error_paths",
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
    "test_normalize_to_pydantic_dict_and_value_branches": (
        "tests.unit.test_utilities_model_full_coverage",
        "test_normalize_to_pydantic_dict_and_value_branches",
    ),
    "test_not_found_error_correlation_id_selection_and_extra_kwargs": (
        "tests.unit.test_exceptions_full_coverage",
        "test_not_found_error_correlation_id_selection_and_extra_kwargs",
    ),
    "test_object_dict_and_type_error_fallback_paths": (
        "tests.unit.test_utilities_checker_full_coverage",
        "test_object_dict_and_type_error_fallback_paths",
    ),
    "test_ok_accepts_none": (
        "tests.unit.test_result_additional",
        "test_ok_accepts_none",
    ),
    "test_operation_progress_start_operation_sets_runtime_fields": (
        "tests.unit.test_models_generic_full_coverage",
        "test_operation_progress_start_operation_sets_runtime_fields",
    ),
    "test_options_merge_conflict_paths_and_empty_merge_options": (
        "tests.unit.test_models_collections_full_coverage",
        "test_options_merge_conflict_paths_and_empty_merge_options",
    ),
    "test_pagination_response_string_fallbacks": (
        "tests.unit.test_utilities_pagination_full_coverage",
        "test_pagination_response_string_fallbacks",
    ),
    "test_parse_mapping_outer_exception": (
        "tests.unit.test_utilities_collection_full_coverage",
        "test_parse_mapping_outer_exception",
    ),
    "test_parser_convert_and_norm_branches": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_convert_and_norm_branches",
    ),
    "test_parser_internal_helpers_additional_coverage": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_internal_helpers_additional_coverage",
    ),
    "test_parser_parse_helpers_and_primitive_coercion_branches": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_parse_helpers_and_primitive_coercion_branches",
    ),
    "test_parser_pipeline_and_pattern_branches": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_pipeline_and_pattern_branches",
    ),
    "test_parser_remaining_branch_paths": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_remaining_branch_paths",
    ),
    "test_parser_safe_length_and_parse_delimited_error_paths": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_safe_length_and_parse_delimited_error_paths",
    ),
    "test_parser_split_and_normalize_exception_paths": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_split_and_normalize_exception_paths",
    ),
    "test_parser_success_and_edge_paths_cover_major_branches": (
        "tests.unit.test_utilities_parser_full_coverage",
        "test_parser_success_and_edge_paths_cover_major_branches",
    ),
    "test_private_coerce_with_enum_and_string": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_private_coerce_with_enum_and_string",
    ),
    "test_private_getters_exception_paths": (
        "tests.unit.test_utilities_configuration_full_coverage",
        "test_private_getters_exception_paths",
    ),
    "test_private_is_member_by_name": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_private_is_member_by_name",
    ),
    "test_private_is_member_by_value": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_private_is_member_by_value",
    ),
    "test_private_parse_success_and_failure": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_private_parse_success_and_failure",
    ),
    "test_process_context_data_and_related_convenience": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_process_context_data_and_related_convenience",
    ),
    "test_process_outer_exception_and_coercion_branches": (
        "tests.unit.test_utilities_collection_full_coverage",
        "test_process_outer_exception_and_coercion_branches",
    ),
    "test_protocol_and_simple_guard_helpers": (
        "tests.unit.test_utilities_guards_full_coverage",
        "test_protocol_and_simple_guard_helpers",
    ),
    "test_provide_property_paths": (
        "tests.unit.test_container_full_coverage",
        "test_provide_property_paths",
    ),
    "test_publish_event_to_subscriber": (
        "tests.unit.test_dispatcher_minimal",
        "test_publish_event_to_subscriber",
    ),
    "test_publish_no_subscribers_succeeds": (
        "tests.unit.test_dispatcher_minimal",
        "test_publish_no_subscribers_succeeds",
    ),
    "test_query_resolve_pagination_wrapper_and_fallback": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_query_resolve_pagination_wrapper_and_fallback",
    ),
    "test_query_validate_pagination_dict_and_default": (
        "tests.unit.test_models_cqrs_full_coverage",
        "test_query_validate_pagination_dict_and_default",
    ),
    "test_railway_and_retry_additional_paths": (
        "tests.unit.test_decorators_full_coverage",
        "test_railway_and_retry_additional_paths",
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
    "test_register_existing_providers_full_paths_and_misc_methods": (
        "tests.unit.test_container_full_coverage",
        "test_register_existing_providers_full_paths_and_misc_methods",
    ),
    "test_register_existing_providers_skips_and_register_core_fallback": (
        "tests.unit.test_container_full_coverage",
        "test_register_existing_providers_skips_and_register_core_fallback",
    ),
    "test_register_handler_as_event_subscriber": (
        "tests.unit.test_dispatcher_minimal",
        "test_register_handler_as_event_subscriber",
    ),
    "test_register_handler_with_can_handle": (
        "tests.unit.test_dispatcher_minimal",
        "test_register_handler_with_can_handle",
    ),
    "test_register_handler_with_message_type": (
        "tests.unit.test_dispatcher_minimal",
        "test_register_handler_with_message_type",
    ),
    "test_register_handler_without_route_fails": (
        "tests.unit.test_dispatcher_minimal",
        "test_register_handler_without_route_fails",
    ),
    "test_register_singleton_register_factory_and_bulk_register_paths": (
        "tests.unit.test_utilities_configuration_full_coverage",
        "test_register_singleton_register_factory_and_bulk_register_paths",
    ),
    "test_remaining_build_fields_construct_and_eq_paths": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_remaining_build_fields_construct_and_eq_paths",
    ),
    "test_remaining_uncovered_branches": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_remaining_uncovered_branches",
    ),
    "test_resolve_env_file_and_log_level": (
        "tests.unit.test_utilities_configuration_full_coverage",
        "test_resolve_env_file_and_log_level",
    ),
    "test_resolve_logger_prefers_logger_attribute": (
        "tests.unit.test_decorators_full_coverage",
        "test_resolve_logger_prefers_logger_attribute",
    ),
    "test_result_property_raises_on_failure": (
        "tests.unit.test_service_additional",
        "test_result_property_raises_on_failure",
    ),
    "test_results_internal_conflict_paths_and_combine": (
        "tests.unit.test_models_collections_full_coverage",
        "test_results_internal_conflict_paths_and_combine",
    ),
    "test_retry_policy_behavior": (
        "tests.unit.test_dispatcher_reliability",
        "test_retry_policy_behavior",
    ),
    "test_retry_unreachable_timeouterror_path": (
        "tests.unit.test_decorators_full_coverage",
        "test_retry_unreachable_timeouterror_path",
    ),
    "test_reuse_existing_runtime_coverage_branches": (
        "tests.unit.test_runtime_full_coverage",
        "test_reuse_existing_runtime_coverage_branches",
    ),
    "test_reuse_existing_runtime_scenarios": (
        "tests.unit.test_runtime_full_coverage",
        "test_reuse_existing_runtime_scenarios",
    ),
    "test_rules_merge_combines_model_dump_values": (
        "tests.unit.test_models_collections_full_coverage",
        "test_rules_merge_combines_model_dump_values",
    ),
    "test_run_pipeline_query_and_event_paths": (
        "tests.unit.test_handlers_full_coverage",
        "test_run_pipeline_query_and_event_paths",
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
    "test_scoped_config_context_branches": (
        "tests.unit.test_container_full_coverage",
        "test_scoped_config_context_branches",
    ),
    "test_service_create_initial_runtime_prefers_custom_config_type_and_context_property": (
        "tests.unit.test_service_full_coverage",
        "test_service_create_initial_runtime_prefers_custom_config_type_and_context_property",
    ),
    "test_service_create_runtime_container_overrides_branch": (
        "tests.unit.test_service_full_coverage",
        "test_service_create_runtime_container_overrides_branch",
    ),
    "test_service_init_type_guards_and_properties": (
        "tests.unit.test_service_full_coverage",
        "test_service_init_type_guards_and_properties",
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
    "test_shortcuts_delegate_to_primary_methods": (
        "tests.unit.test_utilities_enum_full_coverage",
        "test_shortcuts_delegate_to_primary_methods",
    ),
    "test_small_mapper_convenience_methods": (
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_small_mapper_convenience_methods",
    ),
    "test_statistics_and_custom_fields_validators": (
        "tests.unit.test_models_context_full_coverage",
        "test_statistics_and_custom_fields_validators",
    ),
    "test_statistics_from_dict_and_none_conflict_resolution": (
        "tests.unit.test_models_collections_full_coverage",
        "test_statistics_from_dict_and_none_conflict_resolution",
    ),
    "test_strict_registration_and_dispatch": (
        "tests.unit.test_dispatcher_full_coverage",
        "test_strict_registration_and_dispatch",
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
    "test_sync_config_namespace_paths": (
        "tests.unit.test_container_full_coverage",
        "test_sync_config_namespace_paths",
    ),
    "test_sync_config_registers_namespace_factories_and_fallbacks": (
        "tests.unit.test_container_full_coverage",
        "test_sync_config_registers_namespace_factories_and_fallbacks",
    ),
    "test_timeout_additional_success_and_reraise_timeout_paths": (
        "tests.unit.test_decorators_full_coverage",
        "test_timeout_additional_success_and_reraise_timeout_paths",
    ),
    "test_timeout_covers_exception_timeout_branch": (
        "tests.unit.test_decorators_full_coverage",
        "test_timeout_covers_exception_timeout_branch",
    ),
    "test_timeout_raises_when_successful_call_exceeds_limit": (
        "tests.unit.test_decorators_full_coverage",
        "test_timeout_raises_when_successful_call_exceeds_limit",
    ),
    "test_timeout_reraises_original_exception_when_within_limit": (
        "tests.unit.test_decorators_full_coverage",
        "test_timeout_reraises_original_exception_when_within_limit",
    ),
    "test_timestampable_timestamp_conversion_and_json_serializer": (
        "tests.unit.test_models_base_full_coverage",
        "test_timestampable_timestamp_conversion_and_json_serializer",
    ),
    "test_timestamped_model_and_alias_and_canonical_symbols": (
        "tests.unit.test_models_base_full_coverage",
        "test_timestamped_model_and_alias_and_canonical_symbols",
    ),
    "test_to_general_value_dict_removed": (
        "tests.unit.test_models_context_full_coverage",
        "test_to_general_value_dict_removed",
    ),
    "test_track_performance_success_and_failure_paths": (
        "tests.unit.test_decorators_full_coverage",
        "test_track_performance_success_and_failure_paths",
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
    "test_validate_value_object_immutable_exception_and_no_setattr_branch": (
        "tests.unit.test_utilities_domain_full_coverage",
        "test_validate_value_object_immutable_exception_and_no_setattr_branch",
    ),
    "test_validation_like_error_structure": (
        "tests.unit.test_result_full_coverage",
        "test_validation_like_error_structure",
    ),
    "test_with_correlation_with_context_track_operation_and_factory": (
        "tests.unit.test_decorators_full_coverage",
        "test_with_correlation_with_context_track_operation_and_factory",
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
    "AlgarOudMigConstants",
    "AlgarOudMigModels",
    "AlgarOudMigProtocols",
    "AlgarOudMigTypes",
    "AlgarOudMigUtilities",
    "AssertionBuilder",
    "AssertionHelpers",
    "AttrObject",
    "AutoCommand",
    "AutoDiscoveryHandler",
    "BadBool",
    "BadMapping",
    "BadSingletonForTest",
    "BadString",
    "CacheScenarios",
    "CoerceListValidatorScenario",
    "CoerceValidatorScenario",
    "CollectionScenarios",
    "ConcreteTestHandler",
    "ConcreteTestService",
    "ConfigTestCase",
    "ConfigTestFactories",
    "ConfigWithoutModelConfigForTest",
    "ContainerModelsScenarios",
    "CreateUserCommand",
    "CreateUserCommandHandler",
    "DataclassConfigForTest",
    "DictHandler",
    "EchoHandler",
    "EnumScenarios",
    "EventHandler",
    "EventSubscriber",
    "ExplicitTypeHandler",
    "ExplodingHandler",
    "ExplodingLenList",
    "ExtractPageParamsScenario",
    "FailingCommand",
    "FailingCommandHandler",
    "FailingOptionsForTest",
    "FailingService",
    "FailingServiceAuto",
    "FailingServiceAutoFactory",
    "FailingServiceFactory",
    "FixtureBuilder",
    "FlextCliConstants",
    "FlextCliModels",
    "FlextCliProtocols",
    "FlextCliTypes",
    "FlextCliUtilities",
    "FlextCommandId",
    "FlextCommandType",
    "FlextLdapConstants",
    "FlextLdapModels",
    "FlextLdapProtocols",
    "FlextLdapTypes",
    "FlextLdapUtilities",
    "FlextTestBuilder",
    "FlextTestResult",
    "FlextTestResultCo",
    "FunctionalExternalService",
    "GenericHandler",
    "GenericModelFactory",
    "GetUserService",
    "GetUserServiceAuto",
    "GetUserServiceAutoFactory",
    "GetUserServiceFactory",
    "GivenWhenThenBuilder",
    "IntHandler",
    "IsMemberScenario",
    "IsSubsetScenario",
    "LifecycleService",
    "MissingType",
    "MockScenario",
    "MultiOperationService",
    "NestedClassPropagationTransformer",
    "NoHandleMethod",
    "NonCallableHandle",
    "NotificationService",
    "ObjectHandler",
    "OptionsModelForTest",
    "PaginationScenarios",
    "ParameterizedTestBuilder",
    "ParseMappingScenario",
    "ParseOrDefaultScenario",
    "ParseScenario",
    "ParseSequenceScenario",
    "ParserScenario",
    "ParserScenarios",
    "PerformanceBenchmark",
    "PreparePaginationDataScenario",
    "Priority",
    "QueryHandler",
    "RailwayTestCase",
    "ReliabilityScenario",
    "ReliabilityScenarios",
    "RuntimeCloneService",
    "SampleCommand",
    "SampleEvent",
    "SampleHandler",
    "SampleQuery",
    "SendEmailService",
    "ServiceConfig",
    "ServiceFactoryRegistry",
    "ServiceTestCase",
    "ServiceTestCaseFactory",
    "ServiceTestCases",
    "SimpleObj",
    "SingletonWithoutGetGlobalForTest",
    "SingletonWithoutModelDumpForTest",
    "Status",
    "StrictOptionsForTest",
    "StringHandler",
    "StringParserTestFactory",
    "SuiteBuilder",
    "T",
    "TMessage",
    "T_co",
    "T_contra",
    "TestAdvancedPatterns",
    "TestAggregateRoots",
    "TestAllPatternsIntegration",
    "TestAltPropagatesException",
    "TestAssertExists",
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
    "TestBackwardCompatDiscardReturnValue",
    "TestBackwardCompatibility",
    "TestBaseMkGenerationFlow",
    "TestBatchOperations",
    "TestCleanText",
    "TestCloneContainer",
    "TestCloneRuntime",
    "TestCommands",
    "TestCompleteFlextSystemIntegration",
    "TestComprehensiveIntegration",
    "TestConfig",
    "TestConfigConstants",
    "TestConfigMapDictOps",
    "TestConfigModels",
    "TestConfigServiceViaDI",
    "TestContainerDIRealExecution",
    "TestContainerInfo",
    "TestContainerIntegration",
    "TestContainerMemory",
    "TestContainerPerformance",
    "TestContainerStatus",
    "TestContext100Coverage",
    "TestContextDataModel",
    "TestContextServiceViaDI",
    "TestCorrelationDomain",
    "TestCoveragePush75Percent",
    "TestCreateDatetimeProxy",
    "TestCreateDictProxy",
    "TestCreateFromCallableCarriesException",
    "TestCreateInStatic",
    "TestCreateStrProxy",
    "TestCriticalReturnsResultBool",
    "TestCrossModuleIntegration",
    "TestDIBridgeRealExecution",
    "TestDataFactory",
    "TestDataGenerators",
    "TestDebugReturnsResultBool",
    "TestDependencyIntegrationRealExecution",
    "TestDictMixinOperations",
    "TestDispatcherDI",
    "TestDomainEvents",
    "TestDomainHashValue",
    "TestDomainLogger",
    "TestEdgeCases",
    "TestEnterprisePatterns",
    "TestEntities",
    "TestEntityCoverageEdgeCases",
    "TestErrorOrPatternUnchanged",
    "TestErrorReturnsResultBool",
    "TestEventDrivenPatterns",
    "TestExceptionContext",
    "TestExceptionEdgeCases",
    "TestExceptionFactory",
    "TestExceptionIntegration",
    "TestExceptionLogging",
    "TestExceptionMetrics",
    "TestExceptionProperties",
    "TestExceptionPropertyAccess",
    "TestExceptionReturnsResultBool",
    "TestExceptionSerialization",
    "TestFacadeDeprecatedAliases",
    "TestFactories",
    "TestFactoriesHelpers",
    "TestFactoryDecoratorsDiscoveryHasFactories",
    "TestFactoryDecoratorsDiscoveryScanModule",
    "TestFactoryPatterns",
    "TestFailNoExceptionBackwardCompat",
    "TestFailWithException",
    "TestFileInfo",
    "TestFileInfoFromModels",
    "TestFixtureFactory",
    "TestFlatMapPropagatesException",
    "TestFlextCommand",
    "TestFlextCommandHandler",
    "TestFlextCommandResults",
    "TestFlextConstants",
    "TestFlextContainer",
    "TestFlextContext",
    "TestFlextDecorators",
    "TestFlextExceptionsHierarchy",
    "TestFlextHandlers",
    "TestFlextInfraNamespaceValidator",
    "TestFlextLogLevel",
    "TestFlextLogger",
    "TestFlextLoggerIntegration",
    "TestFlextLoggerUsage",
    "TestFlextMixinsNestedClasses",
    "TestFlextModels",
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
    "TestFlextServiceIntegration",
    "TestFlextSettings",
    "TestFlextSettingsSingletonIntegration",
    "TestFlextTestsBuilders",
    "TestFlextTestsDocker",
    "TestFlextTestsDockerWorkerId",
    "TestFlextTestsDockerWorkspaceRoot",
    "TestFlextTestsDomains",
    "TestFlextTestsFactoriesModernAPI",
    "TestFlextTestsFiles",
    "TestFlextTestsFilesNewApi",
    "TestFlextTestsMatchers",
    "TestFlextTestsUtilitiesFactory",
    "TestFlextTestsUtilitiesResult",
    "TestFlextTestsUtilitiesResultCompat",
    "TestFlextTestsUtilitiesTestContext",
    "TestFlextTypings",
    "TestFlextUtilitiesArgs",
    "TestFlextUtilitiesConfiguration",
    "TestFlextUtilitiesModelNormalizeToMetadata",
    "TestFlextUtilitiesReliability",
    "TestFlextVersion",
    "TestFormatAppId",
    "TestFromValidationCarriesException",
    "TestFunction",
    "TestGlobalContextManagement",
    "TestGuardsDeprecatedMethods",
    "TestHandlerDecoratorMetadata",
    "TestHandlerDiscoveryClass",
    "TestHandlerDiscoveryEdgeCases",
    "TestHandlerDiscoveryIntegration",
    "TestHandlerDiscoveryModule",
    "TestHandlerDiscoveryServiceIntegration",
    "TestHelperConsolidationTransformer",
    "TestHierarchicalExceptionSystem",
    "TestIdempotency",
    "TestInfoReturnsResultBool",
    "TestInfoWithContentMeta",
    "TestInstanceCreation",
    "TestIntegrationWithRealCommandServices",
    "TestLashPropagatesException",
    "TestLevelBasedContextManagement",
    "TestLibraryIntegration",
    "TestLogReturnsResultBool",
    "TestLoggerServiceViaDI",
    "TestLoggingIntegration",
    "TestLoggingMethods",
    "TestLoggingsErrorPaths",
    "TestMapPropagatesException",
    "TestMapperBuildFlagsDict",
    "TestMapperCollectActiveKeys",
    "TestMapperDeprecatedMethods",
    "TestMapperFilterDict",
    "TestMapperInvertDict",
    "TestMapperMapDictKeys",
    "TestMapperTransformValues",
    "TestMetadata",
    "TestMigrationComplexity",
    "TestMigrationScenario1",
    "TestMigrationScenario2",
    "TestMigrationScenario4",
    "TestMigrationScenario5",
    "TestModelIntegration",
    "TestModelSerialization",
    "TestModelValidation",
    "TestMonadicOperationsUnchanged",
    "TestOkNoneGuardStillRaises",
    "TestOutputSingletonConsistency",
    "TestPathResolverDiscoveryFlow",
    "TestPattern1V1Explicit",
    "TestPattern2V2Property",
    "TestPattern3RailwayV1",
    "TestPattern4RailwayV2Property",
    "TestPattern5MonadicComposition",
    "TestPattern6ErrorHandling",
    "TestPattern7AutomaticInfrastructure",
    "TestPattern8MultipleOperations",
    "TestPerformanceAnalysis",
    "TestPerformanceBenchmarks",
    "TestPerformanceDomain",
    "TestPhase2FinalCoveragePush",
    "TestProjectLevelRefactor",
    "TestPropertyBasedPatterns",
    "TestProtocolComplianceStructlogLogger",
    "TestQueries",
    "TestRealWiringScenarios",
    "TestRealWorldScenarios",
    "TestResultBasics",
    "TestResultTransformations",
    "TestRuntimeDeprecatedNormalizeMethods",
    "TestRuntimeDictLike",
    "TestRuntimeTypeChecking",
    "TestSafeCarriesException",
    "TestSafeString",
    "TestScopedContextManagement",
    "TestService",
    "TestService100Coverage",
    "TestServiceBootstrap",
    "TestServiceBootstrapWithDI",
    "TestServiceDomain",
    "TestServiceResultProperty",
    "TestServiceWithValidation",
    "TestServicerChaining",
    "TestServicesIntegrationViaDI",
    "TestShortAlias",
    "TestStrictContainerNormalization",
    "TestTextLogger",
    "TestTimeoutEnforcerCleanup",
    "TestTimeoutEnforcerEdgeCases",
    "TestTimeoutEnforcerExecutorManagement",
    "TestTimeoutEnforcerInitialization",
    "TestTraceReturnsResultBool",
    "TestTraversePropagatesException",
    "TestUser",
    "TestUtilitiesCoverage",
    "TestUtilitiesDomain",
    "TestValidateValueImmutable",
    "TestValidatorCallable",
    "TestValidatorMapMixin",
    "TestValues",
    "TestWarningReturnsResultBool",
    "TestWorkspaceDetectionOrchestrationFlow",
    "TestWorkspaceLevelRefactor",
    "Teste",
    "Testr",
    "TestrCoverage",
    "TestsCore",
    "TestsFlextConstants",
    "TestsFlextModels",
    "TestsFlextProtocols",
    "TestsFlextServiceBase",
    "TestsFlextTestsFactoriesDict",
    "TestsFlextTestsFactoriesGeneric",
    "TestsFlextTestsFactoriesList",
    "TestsFlextTestsFactoriesModel",
    "TestsFlextTestsFactoriesRes",
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
    "TestuCollectionBatch",
    "TestuCollectionChunk",
    "TestuCollectionCoerceDictValidator",
    "TestuCollectionCoerceListValidator",
    "TestuCollectionCount",
    "TestuCollectionFilter",
    "TestuCollectionFind",
    "TestuCollectionGroup",
    "TestuCollectionMap",
    "TestuCollectionMerge",
    "TestuCollectionParseMapping",
    "TestuCollectionParseSequence",
    "TestuCollectionProcess",
    "TestuDomain",
    "TestuEnumCoerceByNameValidator",
    "TestuEnumCoerceValidator",
    "TestuEnumIsMember",
    "TestuEnumIsSubset",
    "TestuEnumMetadata",
    "TestuEnumParse",
    "TestuEnumParseOrDefault",
    "TestuMapperAccessors",
    "TestuMapperAdvanced",
    "TestuMapperBuild",
    "TestuMapperConversions",
    "TestuMapperExtract",
    "TestuMapperUtils",
    "TestuPaginationBuildPaginationResponse",
    "TestuPaginationExtractPageParams",
    "TestuPaginationExtractPaginationConfig",
    "TestuPaginationPreparePaginationData",
    "TestuPaginationValidatePaginationParams",
    "TestuStringParser",
    "TestuTypeChecker",
    "TestuTypeGuardsIsDictNonEmpty",
    "TestuTypeGuardsIsListNonEmpty",
    "TestuTypeGuardsIsStringNonEmpty",
    "TestuTypeGuardsNormalizeToMetadata",
    "TextLike",
    "TextUtilityContract",
    "ThreadSafetyTest",
    "TimeoutEnforcerScenarios",
    "TypeGuardsScenarios",
    "UnknownHint",
    "UnregisteredCommand",
    "UpdateUserCommand",
    "UpdateUserCommandHandler",
    "User",
    "UserFactory",
    "UserQueryService",
    "UserServiceEntity",
    "ValidatePaginationParamsScenario",
    "ValidatingService",
    "ValidatingServiceAuto",
    "ValidatingServiceAutoFactory",
    "ValidatingServiceFactory",
    "ValidationScenario",
    "ValidationScenarios",
    "ValidationService",
    "arrange_act_assert",
    "assert_rejects",
    "assert_result_success",
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
    "dispatcher",
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
    "integration",
    "invalid_hostnames",
    "invalid_port_numbers",
    "invalid_uris",
    "m",
    "make_result_logger",
    "mapper",
    "mark_test_pattern",
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
    "test_additional_container_branches_cover_fluent_and_lookup_paths",
    "test_additional_register_factory_and_unregister_paths",
    "test_aliases_are_available",
    "test_args_get_enum_params_annotated_unwrap_branch",
    "test_args_get_enum_params_branches",
    "test_async_log_writer_paths",
    "test_async_log_writer_shutdown_with_full_queue",
    "test_at_take_and_as_branches",
    "test_authentication_error_normalizes_extra_kwargs_into_context",
    "test_auto_value_lowercases_input",
    "test_bad_string_and_bad_bool_raise_value_error",
    "test_base_error_normalize_metadata_merges_existing_metadata_model",
    "test_basic_imports_work",
    "test_batch_fail_collect_flatten_and_progress",
    "test_bi_map_returns_forward_copy_and_inverse",
    "test_bind_operation_context_without_ensure_correlation_and_bind_failure",
    "test_build_apply_transform_and_process_error_paths",
    "test_build_options_invalid_only_kwargs_returns_base",
    "test_builder",
    "test_callable_registration_with_attribute",
    "test_canonical_aliases_are_available",
    "test_categories_clear_and_symbols_are_available",
    "test_centralize_pydantic_cli_outputs_extended_metrics",
    "test_centralizer_converts_typed_dict_factory_to_model",
    "test_centralizer_does_not_touch_settings_module",
    "test_centralizer_moves_dict_alias_in_typings_without_keyword_name",
    "test_centralizer_moves_manual_type_aliases_to_models_file",
    "test_checker_logger_and_safe_type_hints_fallback",
    "test_chk_exercises_missed_branches",
    "test_circuit_breaker_transitions_and_metrics",
    "test_class_nesting_appends_to_existing_namespace_and_removes_pass",
    "test_class_nesting_keeps_unmapped_top_level_classes",
    "test_class_nesting_moves_top_level_class_into_new_namespace",
    "test_class_nesting_refactor_single_file_end_to_end",
    "test_clear_keys_values_items_and_validate_branches",
    "test_clear_operation_scope_and_handle_log_result_paths",
    "test_collection_batch_failure_error_capture_and_parse_sequence_outer_error",
    "test_combined_with_and_without_railway_uses_injection",
    "test_command_pagination_limit",
    "test_config_bridge_and_trace_context_and_http_validation",
    "test_config_context_properties_and_defaults",
    "test_config_hash_from_mapping_and_non_hashable",
    "test_configuration_mapping_and_dict_negative_branches",
    "test_configure_structlog_edge_paths",
    "test_configure_structlog_print_logger_factory_fallback",
    "test_configure_with_resource_register_and_factory_error_paths",
    "test_constants_auto_enum_and_bimapping_paths",
    "test_construct_transform_and_deep_eq_branches",
    "test_container_and_service_domain_paths",
    "test_container_remaining_branch_paths_in_sync_factory_and_getters",
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
    "test_create_auto_register_factories_path",
    "test_create_auto_register_factory_wrapper_callable_and_non_callable",
    "test_create_discriminated_union_multiple_enums",
    "test_create_enum_executes_factory_path",
    "test_create_from_callable_and_repr",
    "test_create_from_callable_branches",
    "test_create_merges_metadata_dict_branch",
    "test_create_overloads_and_auto_correlation",
    "test_create_scoped_instance_and_scoped_additional_branches",
    "test_data_factory",
    "test_decorators_family_blocks_dispatcher_target",
    "test_dependency_integration_and_wiring_paths",
    "test_dependency_registration_duplicate_guards",
    "test_deprecated_class_noop_init_branch",
    "test_deprecated_normalize_to_general_value_warns",
    "test_deprecated_normalize_to_metadata_value_warns",
    "test_deprecated_wrapper_emits_warning_and_returns_value",
    "test_discover_project_roots_without_nested_git_dirs",
    "test_discovery_narrowed_function_paths",
    "test_dispatch_after_handler_removal_fails",
    "test_dispatch_auto_discovery_handler",
    "test_dispatch_coerce_mode_with_enum_string_and_other_object",
    "test_dispatch_command_success",
    "test_dispatch_handler_exception_returns_failure",
    "test_dispatch_invalid_input_types",
    "test_dispatch_is_member_by_name_and_by_value",
    "test_dispatch_is_name_mode",
    "test_dispatch_no_handler_fails",
    "test_dispatch_parse_mode_with_enum_string_and_other_object",
    "test_dispatch_unknown_mode_raises",
    "test_dispatcher_family_blocks_models_target",
    "test_dispatcher_reliability_branch_paths",
    "test_enrich_and_ensure_trace_context_branches",
    "test_ensure_and_extract_array_index_helpers",
    "test_ensure_dict_branches",
    "test_ensure_trace_context_dict_conversion_paths",
    "test_ensure_utc_datetime_adds_tzinfo_when_naive",
    "test_ensure_utc_datetime_preserves_aware",
    "test_ensure_utc_datetime_returns_none_on_none",
    "test_entity_comparable_map_and_bulk_validation_paths",
    "test_event_publishing_strict",
    "test_exception_handling_in_dispatch",
    "test_exceptions_uncovered_metadata_paths",
    "test_execute_and_register_handler_failure_paths",
    "test_execute_retry_exponential_and_handle_exhaustion_raise_last_exception",
    "test_execute_retry_loop_covers_default_linear_and_never_ran",
    "test_export_paths_with_metadata_and_statistics",
    "test_extract_error_paths_and_prop_accessor",
    "test_extract_field_value_and_ensure_variants",
    "test_extract_mapping_or_none_branches",
    "test_extract_message_type_annotation_and_dict_subclass_paths",
    "test_extract_message_type_from_handle_with_only_self",
    "test_extract_message_type_from_parameter_branches",
    "test_facade_binding_is_correct",
    "test_field_and_fields_multi_branches",
    "test_filter_map_normalize_convert_helpers",
    "test_find_mapping_no_match_and_merge_error_paths",
    "test_flext_message_type_alias_adapter",
    "test_flow_through_short_circuits_on_failure",
    "test_from_validation_and_to_model_paths",
    "test_frozen_value_model_equality_and_hash",
    "test_general_value_helpers_and_logger",
    "test_generate_special_paths_and_dynamic_subclass",
    "test_generators_additional_missed_paths",
    "test_generators_mapping_non_dict_normalization_path",
    "test_get_and_get_typed_resource_factory_paths",
    "test_get_enum_values_returns_immutable_sequence",
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
    "test_handle_log_result_without_fallback_logger_and_non_dict_like_extra",
    "test_handle_retry_exhaustion_falsey_exception_reaches_timeout_error",
    "test_handler_attribute_discovery",
    "test_handler_builder_fluent_methods",
    "test_handler_type_literal_and_invalid",
    "test_helper_consolidation_is_prechecked",
    "test_identifiable_unique_id_empty_rejected",
    "test_inactive_and_none_value_paths",
    "test_init_fallback_and_lazy_returns_result_property",
    "test_initialize_di_components_error_paths",
    "test_initialize_di_components_second_type_error_branch",
    "test_inject_sets_missing_dependency_from_container",
    "test_invalid_handler_mode_init_raises",
    "test_invalid_registration_attempts",
    "test_invert_and_json_conversion_branches",
    "test_is_container_negative_paths_and_callable",
    "test_is_flexible_value_covers_all_branches",
    "test_is_general_value_list_accepts_list_subclass",
    "test_is_handler_type_branches",
    "test_is_type_non_empty_unknown_and_tuple_and_fallback",
    "test_is_type_protocol_fallback_branches",
    "test_is_valid_handles_validation_exception",
    "test_lash_runtime_result_paths",
    "test_log_operation_track_perf_exception_adds_duration",
    "test_loggings_bind_clear_level_error_paths",
    "test_loggings_context_and_factory_paths",
    "test_loggings_exception_and_adapter_paths",
    "test_loggings_instance_and_message_format_paths",
    "test_loggings_remaining_branch_paths",
    "test_loggings_source_and_log_error_paths",
    "test_loggings_uncovered_level_trace_path_and_exception_guards",
    "test_map_error_identity_and_transform",
    "test_map_flags_collect_and_invert_branches",
    "test_map_flat_map_and_then_paths",
    "test_members_uses_cache_on_second_call",
    "test_merge_defaults_and_dump_paths",
    "test_merge_metadata_context_paths",
    "test_metadata_attributes_accepts_basemodel_mapping",
    "test_metadata_attributes_accepts_none",
    "test_metadata_attributes_accepts_t_dict_and_mapping",
    "test_metadata_attributes_rejects_basemodel_non_mapping_dump",
    "test_metadata_attributes_rejects_non_mapping",
    "test_migrate_protocols_rewrites_references_with_p_alias",
    "test_migrate_to_mro_inlines_alias_constant_into_constants_class",
    "test_migrate_to_mro_moves_constant_and_rewrites_reference",
    "test_migrate_to_mro_moves_manual_uppercase_assignment",
    "test_migrate_to_mro_normalizes_facade_alias_to_c",
    "test_migrate_to_mro_rejects_unknown_target",
    "test_migrate_typings_rewrites_references_with_t_alias",
    "test_misc_unregistration_clear_and_reset",
    "test_mixins_container_registration_and_logger_paths",
    "test_mixins_context_logging_and_cqrs_paths",
    "test_mixins_context_stack_pop_initializes_missing_stack_attr",
    "test_mixins_remaining_branch_paths",
    "test_mixins_result_and_model_conversion_paths",
    "test_mixins_runtime_bootstrap_and_track_paths",
    "test_mixins_validation_and_protocol_paths",
    "test_model_helpers_remaining_paths",
    "test_model_support_and_hash_compare_paths",
    "test_models_family_blocks_utilities_target",
    "test_models_handler_branches",
    "test_models_handler_uncovered_mode_and_reset_paths",
    "test_models_settings_branch_paths",
    "test_models_settings_context_validator_and_non_standard_status_input",
    "test_mro_resolver_accepts_expected_order",
    "test_mro_resolver_rejects_wrong_order",
    "test_mro_scanner_includes_constants_variants_in_all_scopes",
    "test_names_uses_cache_on_second_call",
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
    "test_normalize_context_to_dict_error_paths",
    "test_normalize_to_list_passes_list_through",
    "test_normalize_to_list_wraps_int",
    "test_normalize_to_list_wraps_scalar",
    "test_normalize_to_pydantic_dict_and_value_branches",
    "test_not_found_error_correlation_id_selection_and_extra_kwargs",
    "test_object_dict_and_type_error_fallback_paths",
    "test_ok_accepts_none",
    "test_operation_progress_start_operation_sets_runtime_fields",
    "test_options_merge_conflict_paths_and_empty_merge_options",
    "test_pagination_response_string_fallbacks",
    "test_parse_mapping_outer_exception",
    "test_parser_convert_and_norm_branches",
    "test_parser_internal_helpers_additional_coverage",
    "test_parser_parse_helpers_and_primitive_coercion_branches",
    "test_parser_pipeline_and_pattern_branches",
    "test_parser_remaining_branch_paths",
    "test_parser_safe_length_and_parse_delimited_error_paths",
    "test_parser_split_and_normalize_exception_paths",
    "test_parser_success_and_edge_paths_cover_major_branches",
    "test_private_coerce_with_enum_and_string",
    "test_private_getters_exception_paths",
    "test_private_is_member_by_name",
    "test_private_is_member_by_value",
    "test_private_parse_success_and_failure",
    "test_process_context_data_and_related_convenience",
    "test_process_outer_exception_and_coercion_branches",
    "test_protocol_and_simple_guard_helpers",
    "test_provide_property_paths",
    "test_publish_event_to_subscriber",
    "test_publish_no_subscribers_succeeds",
    "test_query_resolve_pagination_wrapper_and_fallback",
    "test_query_validate_pagination_dict_and_default",
    "test_railway_and_retry_additional_paths",
    "test_rate_limiter_blocks_then_recovers",
    "test_rate_limiter_jitter_application",
    "test_reconfigure_and_reset_state_paths",
    "test_recover_tap_and_tap_error_paths",
    "test_refactor_utilities_iter_python_files_includes_examples_and_scripts",
    "test_register_existing_providers_full_paths_and_misc_methods",
    "test_register_existing_providers_skips_and_register_core_fallback",
    "test_register_handler_as_event_subscriber",
    "test_register_handler_with_can_handle",
    "test_register_handler_with_message_type",
    "test_register_handler_without_route_fails",
    "test_register_singleton_register_factory_and_bulk_register_paths",
    "test_remaining_build_fields_construct_and_eq_paths",
    "test_remaining_uncovered_branches",
    "test_resolve_env_file_and_log_level",
    "test_resolve_logger_prefers_logger_attribute",
    "test_result_property_raises_on_failure",
    "test_results_internal_conflict_paths_and_combine",
    "test_retry_policy_behavior",
    "test_retry_unreachable_timeouterror_path",
    "test_reuse_existing_runtime_coverage_branches",
    "test_reuse_existing_runtime_scenarios",
    "test_rules_merge_combines_model_dump_values",
    "test_run_pipeline_query_and_event_paths",
    "test_runtime_create_instance_failure_branch",
    "test_runtime_family_blocks_non_runtime_target",
    "test_runtime_integration_tracking_paths",
    "test_runtime_misc_remaining_paths",
    "test_runtime_module_accessors_and_metadata",
    "test_runtime_result_alias_compatibility",
    "test_runtime_result_all_missed_branches",
    "test_runtime_result_remaining_paths",
    "test_scope_data_validators_and_errors",
    "test_scoped_config_context_branches",
    "test_service_create_initial_runtime_prefers_custom_config_type_and_context_property",
    "test_service_create_runtime_container_overrides_branch",
    "test_service_init_type_guards_and_properties",
    "test_service_request_timeout_post_validator_messages",
    "test_service_request_timeout_validator_branches",
    "test_set_set_all_get_validation_and_error_paths",
    "test_settings_materialize_and_context_overrides",
    "test_shortcuts_delegate_to_primary_methods",
    "test_small_mapper_convenience_methods",
    "test_statistics_and_custom_fields_validators",
    "test_statistics_from_dict_and_none_conflict_resolution",
    "test_strict_registration_and_dispatch",
    "test_strip_whitespace_preserves_clean",
    "test_strip_whitespace_returns_empty_on_spaces",
    "test_strip_whitespace_trims_leading_trailing",
    "test_structlog_proxy_context_var_default_when_key_missing",
    "test_structlog_proxy_context_var_get_set_reset_paths",
    "test_summary_error_paths_and_bindings_failures",
    "test_summary_properties_and_subclass_storage_reset",
    "test_sync_config_namespace_paths",
    "test_sync_config_registers_namespace_factories_and_fallbacks",
    "test_timeout_additional_success_and_reraise_timeout_paths",
    "test_timeout_covers_exception_timeout_branch",
    "test_timeout_raises_when_successful_call_exceeds_limit",
    "test_timeout_reraises_original_exception_when_within_limit",
    "test_timestampable_timestamp_conversion_and_json_serializer",
    "test_timestamped_model_and_alias_and_canonical_symbols",
    "test_to_general_value_dict_removed",
    "test_track_performance_success_and_failure_paths",
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
    "test_validate_value_object_immutable_exception_and_no_setattr_branch",
    "test_validation_like_error_structure",
    "test_with_correlation_with_context_track_operation_and_factory",
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
