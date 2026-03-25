# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Tests package."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from flext_tests import d, e, h, r, s, x

    from flext_core import FlextTypes
    from tests import benchmark, helpers, integration, unit
    from tests.base import TestsFlextServiceBase
    from tests.benchmark.test_container_memory import (
        TestContainerMemory,
        get_memory_usage,
    )
    from tests.benchmark.test_container_performance import TestContainerPerformance
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
    from tests.constants import FlextCoreTestConstants, FlextCoreTestConstants as c
    from tests.fixtures.namespace_validator.rule0_loose_items import (
        Rule0LooseItemsFixture,
    )
    from tests.fixtures.namespace_validator.rule0_multiple_classes import (
        FlextTestConstants,
        Rule0MultipleClassesFixture,
    )
    from tests.fixtures.namespace_validator.rule0_no_class import MAX_VALUE, helper
    from tests.fixtures.namespace_validator.rule0_wrong_prefix import RandomConstants
    from tests.fixtures.namespace_validator.rule1_loose_constant import (
        DEFAULT_TIMEOUT,
        MAX_RETRIES,
    )
    from tests.fixtures.namespace_validator.rule1_loose_enum import (
        FlextTestModels,
        Rule1LooseEnumFixture,
        Status,
    )
    from tests.fixtures.namespace_validator.rule1_magic_number import FlextTestUtilities
    from tests.fixtures.namespace_validator.rule2_protocol_in_types import (
        FlextTestTypes,
    )
    from tests.fixtures.namespace_validator.typings import LooseTypeAlias
    from tests.helpers._scenarios_impl import (
        ParserScenario,
        ParserScenarios,
        ReliabilityScenario,
        ReliabilityScenarios,
        ValidationScenario,
        ValidationScenarios,
    )
    from tests.helpers.factories import TestHelperFactories
    from tests.helpers.factories_impl import (
        FailingService,
        FailingServiceAuto,
        FailingServiceAutoFactory,
        FailingServiceFactory,
        GenericModelFactory,
        GetUserService,
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
    from tests.helpers.scenarios import TestHelperScenarios
    from tests.integration import patterns
    from tests.integration.patterns.test_advanced_patterns import (
        TestAdvancedPatterns,
        TestFunction,
    )
    from tests.integration.patterns.test_architectural_patterns import (
        TestArchitecturalPatterns,
    )
    from tests.integration.patterns.test_patterns_commands import TestPatternsCommands
    from tests.integration.patterns.test_patterns_logging import (
        EXPECTED_BULK_SIZE,
        TestPatternsLogging,
    )
    from tests.integration.patterns.test_patterns_testing import TestPatternsTesting
    from tests.integration.test_architecture import TestAutomatedArchitecture
    from tests.integration.test_config_integration import (
        TestFlextSettingsSingletonIntegration,
    )
    from tests.integration.test_infra_integration import TestInfraIntegration
    from tests.integration.test_integration import TestLibraryIntegration
    from tests.integration.test_migration_validation import TestMigrationValidation
    from tests.integration.test_refactor_nesting_file import (
        test_class_nesting_refactor_single_file_end_to_end,
    )
    from tests.integration.test_refactor_nesting_idempotency import TestIdempotency
    from tests.integration.test_refactor_nesting_project import TestProjectLevelRefactor
    from tests.integration.test_refactor_nesting_workspace import (
        TestWorkspaceLevelRefactor,
    )
    from tests.integration.test_refactor_policy_mro import TestRefactorPolicyMRO
    from tests.integration.test_service import TestService
    from tests.integration.test_system import TestCompleteFlextSystemIntegration
    from tests.models import FlextCoreTestModels, FlextCoreTestModels as m
    from tests.protocols import FlextCoreTestProtocols, FlextCoreTestProtocols as p
    from tests.test_documented_patterns import TestDocumentedPatterns
    from tests.test_service_result_property import TestServiceResultProperty
    from tests.test_utils import (
        FlextTestResult,
        FlextTestResultCo,
        TestUtils,
        assertion_helpers,
        fixture_factory,
        test_data_factory,
    )
    from tests.typings import (
        FlextCoreTestTypes,
        FlextCoreTestTypes as t,
        T_co,
        T_contra,
    )
    from tests.unit import contracts, flext_tests
    from tests.unit._models import TestUnitModels
    from tests.unit._models_impl import (
        BadConfigForTest,
        CacheTestModel,
        ComplexModel,
        ConfigModelForTest,
        InputPayloadMap,
        InvalidModelForTest,
        NestedModel,
        SampleModel,
        SingletonClassForTest,
        TestCaseMap,
        _BadCopyModel,
        _BrokenDumpModel,
        _Cfg,
        _DumpErrorModel,
        _ErrorsModel,
        _FakeConfig,
        _FrozenEntity,
        _GoodModel,
        _Model,
        _MsgWithCommandId,
        _MsgWithMessageId,
        _Opts,
        _PlainErrorModel,
        _SampleEntity,
        _SvcModel,
        _TargetModel,
        _ValidationLikeError,
    )
    from tests.unit.conftest_infra import (
        infra_git,
        infra_git_repo,
        infra_io,
        infra_path,
        infra_patterns,
        infra_reporting,
        infra_safe_command_output,
        infra_selection,
        infra_subprocess,
        infra_templates,
        infra_test_workspace,
        infra_toml,
    )
    from tests.unit.contracts.text_contract import TextUtilityContract
    from tests.unit.flext_tests.test_docker import TestDocker
    from tests.unit.flext_tests.test_domains import TestFlextTestsDomains
    from tests.unit.flext_tests.test_files import TestFlextTestsFiles
    from tests.unit.flext_tests.test_matchers import TestFlextTestsMatchers
    from tests.unit.flext_tests.test_utilities import TestUtilities
    from tests.unit.protocols import FlextProtocols
    from tests.unit.test_args_coverage_100 import TestFlextUtilitiesArgs
    from tests.unit.test_collection_utilities_coverage_100 import (
        TestCollectionUtilitiesCoverage,
    )
    from tests.unit.test_collections_coverage_100 import (
        TestFlextModelsCollectionsCoverage100,
    )
    from tests.unit.test_config import TestFlextSettings
    from tests.unit.test_constants import TestConstants
    from tests.unit.test_container import TestFlextContainer
    from tests.unit.test_container_full_coverage import TestContainerFullCoverage
    from tests.unit.test_context import TestFlextContext
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
    from tests.unit.test_coverage_context import TestCoverageContext
    from tests.unit.test_coverage_exceptions import TestCoverageExceptions
    from tests.unit.test_coverage_loggings import TestCoverageLoggings
    from tests.unit.test_coverage_models import TestCoverageModels
    from tests.unit.test_coverage_utilities import Testu
    from tests.unit.test_decorators import TestFlextDecorators
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestDecoratorsDiscoveryFullCoverage,
    )
    from tests.unit.test_decorators_full_coverage import TestDecoratorsFullCoverage
    from tests.unit.test_deprecation_warnings import TestDeprecationWarnings
    from tests.unit.test_di_incremental import TestDIIncremental, inject
    from tests.unit.test_di_services_access import TestDiServicesAccess
    from tests.unit.test_dispatcher_di import TestDispatcherDI
    from tests.unit.test_dispatcher_full_coverage import TestDispatcherFullCoverage
    from tests.unit.test_dispatcher_minimal import TestDispatcherMinimal
    from tests.unit.test_dispatcher_reliability import (
        test_circuit_breaker_half_open_and_rate_limiter_accessors,
        test_circuit_breaker_transitions_and_metrics,
        test_rate_limiter_blocks_then_recovers,
        test_rate_limiter_jitter_application,
        test_retry_policy_behavior,
    )
    from tests.unit.test_dispatcher_timeout_coverage_100 import (
        TestDispatcherTimeoutCoverage100,
    )
    from tests.unit.test_entity_coverage import TestEntityCoverageEdgeCases
    from tests.unit.test_enum_utilities_coverage_100 import TestEnumUtilitiesCoverage
    from tests.unit.test_exceptions import Teste, TestExceptionsHypothesis
    from tests.unit.test_handler_decorator_discovery import (
        TestHandlerDecoratorDiscovery,
    )
    from tests.unit.test_handlers import TestFlextHandlers
    from tests.unit.test_handlers_full_coverage import (
        TestHandlersFullCoverage,
        handlers_module,
    )
    from tests.unit.test_loggings_error_paths_coverage import TestLoggingsErrorPaths
    from tests.unit.test_loggings_full_coverage import TestModule
    from tests.unit.test_loggings_strict_returns import TestLoggingsStrictReturns
    from tests.unit.test_mixins import TestFlextMixinsCQRS, TestFlextMixinsNestedClasses
    from tests.unit.test_mixins_full_coverage import TestMixinsFullCoverage
    from tests.unit.test_models import TestModels
    from tests.unit.test_models_base_full_coverage import TestModelsBaseFullCoverage
    from tests.unit.test_models_container import TestFlextModelsContainer
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
    from tests.unit.test_namespace_validator import TestFlextInfraNamespaceValidator
    from tests.unit.test_pagination_coverage_100 import TestPaginationCoverage100
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
    from tests.unit.test_registry import TestFlextRegistry
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
    from tests.unit.test_result_exception_carrying import TestResultExceptionCarrying
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
    from tests.unit.test_runtime_coverage_100 import TestRuntimeCoverage100
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
    from tests.unit.test_service import TestsCore, TestServiceInternals
    from tests.unit.test_service_additional import (
        RuntimeCloneService,
        test_get_service_info,
        test_is_valid_handles_validation_exception,
        test_result_property_raises_on_failure,
    )
    from tests.unit.test_service_bootstrap import TestServiceBootstrap
    from tests.unit.test_service_coverage_100 import TestService100Coverage
    from tests.unit.test_settings_coverage import TestFlextSettingsCoverage
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
    from tests.unit.test_typings import TestTypings
    from tests.unit.test_typings_full_coverage import TestTypingsFullCoverage
    from tests.unit.test_utilities_cache_coverage_100 import (
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
    from tests.unit.test_utilities_collection_coverage_100 import (
        TestUtilitiesCollectionCoverage,
    )
    from tests.unit.test_utilities_collection_full_coverage import (
        TestUtilitiesCollectionFullCoverage,
    )
    from tests.unit.test_utilities_configuration_coverage_100 import (
        TestFlextUtilitiesConfiguration,
    )
    from tests.unit.test_utilities_configuration_full_coverage import (
        TestUtilitiesConfigurationFullCoverage,
    )
    from tests.unit.test_utilities_context_full_coverage import (
        TestUtilitiesContextFullCoverage,
    )
    from tests.unit.test_utilities_coverage import TestUtilitiesCoverage
    from tests.unit.test_utilities_data_mapper import TestUtilitiesDataMapper
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
        TestUtilitiesDomainFullCoverage,
    )
    from tests.unit.test_utilities_enum_full_coverage import (
        TestUtilitiesEnumFullCoverage,
    )
    from tests.unit.test_utilities_generators_full_coverage import (
        TestUtilitiesGeneratorsFullCoverage,
        generators_module,
        runtime_module,
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
        UtilitiesMapperCoverage100Namespace,
    )
    from tests.unit.test_utilities_mapper_full_coverage import (
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
    from tests.unit.test_utilities_parser_full_coverage import (
        TestUtilitiesParserFullCoverage,
    )
    from tests.unit.test_utilities_reliability import TestFlextUtilitiesReliability
    from tests.unit.test_utilities_string_parser import (
        TestuStringParser,
        normalized_value_key_cases,
    )
    from tests.unit.test_utilities_text_full_coverage import (
        TestUtilitiesTextFullCoverage,
    )
    from tests.unit.test_utilities_type_checker_coverage_100 import (
        T,
        TestuTypeChecker,
        TMessage,
        pytestmark,
    )
    from tests.unit.test_utilities_type_guards_coverage_100 import (
        TestUtilitiesTypeGuardsCoverage100,
    )
    from tests.unit.test_version import TestFlextVersion
    from tests.utilities import FlextCoreTestUtilities, FlextCoreTestUtilities as u

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "AttrObject": ["tests.unit.test_utilities_mapper_full_coverage", "AttrObject"],
    "BadBool": ["tests.unit.test_utilities_mapper_full_coverage", "BadBool"],
    "BadConfigForTest": ["tests.unit._models_impl", "BadConfigForTest"],
    "BadMapping": ["tests.unit.test_utilities_mapper_full_coverage", "BadMapping"],
    "BadString": ["tests.unit.test_utilities_mapper_full_coverage", "BadString"],
    "CacheTestModel": ["tests.unit._models_impl", "CacheTestModel"],
    "CircuitBreakerManager": [
        "tests.unit.test_dispatcher_reliability",
        "CircuitBreakerManager",
    ],
    "ClearCacheScenario": [
        "tests.unit.test_utilities_cache_coverage_100",
        "ClearCacheScenario",
    ],
    "ComplexModel": ["tests.unit._models_impl", "ComplexModel"],
    "ConfigModelForTest": ["tests.unit._models_impl", "ConfigModelForTest"],
    "DEFAULT_TIMEOUT": [
        "tests.fixtures.namespace_validator.rule1_loose_constant",
        "DEFAULT_TIMEOUT",
    ],
    "EXPECTED_BULK_SIZE": [
        "tests.integration.patterns.test_patterns_logging",
        "EXPECTED_BULK_SIZE",
    ],
    "ExplodingLenList": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "ExplodingLenList",
    ],
    "FailingService": ["tests.helpers.factories_impl", "FailingService"],
    "FailingServiceAuto": ["tests.helpers.factories_impl", "FailingServiceAuto"],
    "FailingServiceAutoFactory": [
        "tests.helpers.factories_impl",
        "FailingServiceAutoFactory",
    ],
    "FailingServiceFactory": ["tests.helpers.factories_impl", "FailingServiceFactory"],
    "FlextCoreTestConstants": ["tests.constants", "FlextCoreTestConstants"],
    "FlextCoreTestModels": ["tests.models", "FlextCoreTestModels"],
    "FlextCoreTestProtocols": ["tests.protocols", "FlextCoreTestProtocols"],
    "FlextCoreTestTypes": ["tests.typings", "FlextCoreTestTypes"],
    "FlextCoreTestUtilities": ["tests.utilities", "FlextCoreTestUtilities"],
    "FlextProtocols": ["tests.unit.protocols", "FlextProtocols"],
    "FlextTestConstants": [
        "tests.fixtures.namespace_validator.rule0_multiple_classes",
        "FlextTestConstants",
    ],
    "FlextTestModels": [
        "tests.fixtures.namespace_validator.rule1_loose_enum",
        "FlextTestModels",
    ],
    "FlextTestResult": ["tests.test_utils", "FlextTestResult"],
    "FlextTestResultCo": ["tests.test_utils", "FlextTestResultCo"],
    "FlextTestTypes": [
        "tests.fixtures.namespace_validator.rule2_protocol_in_types",
        "FlextTestTypes",
    ],
    "FlextTestUtilities": [
        "tests.fixtures.namespace_validator.rule1_magic_number",
        "FlextTestUtilities",
    ],
    "FunctionalExternalService": ["tests.conftest", "FunctionalExternalService"],
    "GenericModelFactory": ["tests.helpers.factories_impl", "GenericModelFactory"],
    "GetUserService": ["tests.helpers.factories_impl", "GetUserService"],
    "GetUserServiceAuto": ["tests.helpers.factories_impl", "GetUserServiceAuto"],
    "GetUserServiceAutoFactory": [
        "tests.helpers.factories_impl",
        "GetUserServiceAutoFactory",
    ],
    "GetUserServiceFactory": ["tests.helpers.factories_impl", "GetUserServiceFactory"],
    "InputPayloadMap": ["tests.unit._models_impl", "InputPayloadMap"],
    "InvalidModelForTest": ["tests.unit._models_impl", "InvalidModelForTest"],
    "LooseTypeAlias": ["tests.fixtures.namespace_validator.typings", "LooseTypeAlias"],
    "MAX_RETRIES": [
        "tests.fixtures.namespace_validator.rule1_loose_constant",
        "MAX_RETRIES",
    ],
    "MAX_VALUE": ["tests.fixtures.namespace_validator.rule0_no_class", "MAX_VALUE"],
    "NORMALIZE_COMPONENT_SCENARIOS": [
        "tests.unit.test_utilities_cache_coverage_100",
        "NORMALIZE_COMPONENT_SCENARIOS",
    ],
    "NestedClassPropagationTransformer": [
        "tests.unit.test_transformer_nested_class_propagation",
        "NestedClassPropagationTransformer",
    ],
    "NestedModel": ["tests.unit._models_impl", "NestedModel"],
    "NormalizeComponentScenario": [
        "tests.unit.test_utilities_cache_coverage_100",
        "NormalizeComponentScenario",
    ],
    "ParserScenario": ["tests.helpers._scenarios_impl", "ParserScenario"],
    "ParserScenarios": ["tests.helpers._scenarios_impl", "ParserScenarios"],
    "Provide": ["tests.unit.test_di_incremental", "Provide"],
    "RandomConstants": [
        "tests.fixtures.namespace_validator.rule0_wrong_prefix",
        "RandomConstants",
    ],
    "RateLimiterManager": [
        "tests.unit.test_dispatcher_reliability",
        "RateLimiterManager",
    ],
    "ReliabilityScenario": ["tests.helpers._scenarios_impl", "ReliabilityScenario"],
    "ReliabilityScenarios": ["tests.helpers._scenarios_impl", "ReliabilityScenarios"],
    "RetryPolicy": ["tests.unit.test_dispatcher_reliability", "RetryPolicy"],
    "Rule0LooseItemsFixture": [
        "tests.fixtures.namespace_validator.rule0_loose_items",
        "Rule0LooseItemsFixture",
    ],
    "Rule0MultipleClassesFixture": [
        "tests.fixtures.namespace_validator.rule0_multiple_classes",
        "Rule0MultipleClassesFixture",
    ],
    "Rule1LooseEnumFixture": [
        "tests.fixtures.namespace_validator.rule1_loose_enum",
        "Rule1LooseEnumFixture",
    ],
    "RuntimeCloneService": [
        "tests.unit.test_service_additional",
        "RuntimeCloneService",
    ],
    "SORT_KEY_SCENARIOS": [
        "tests.unit.test_utilities_cache_coverage_100",
        "SORT_KEY_SCENARIOS",
    ],
    "SampleModel": ["tests.unit._models_impl", "SampleModel"],
    "ServiceFactoryRegistry": [
        "tests.helpers.factories_impl",
        "ServiceFactoryRegistry",
    ],
    "ServiceTestCase": ["tests.helpers.factories_impl", "ServiceTestCase"],
    "ServiceTestCaseFactory": [
        "tests.helpers.factories_impl",
        "ServiceTestCaseFactory",
    ],
    "ServiceTestCases": ["tests.helpers.factories_impl", "ServiceTestCases"],
    "SimpleObj": ["tests.unit.test_utilities_mapper_coverage_100", "SimpleObj"],
    "SingletonClassForTest": ["tests.unit._models_impl", "SingletonClassForTest"],
    "SortKeyScenario": [
        "tests.unit.test_utilities_cache_coverage_100",
        "SortKeyScenario",
    ],
    "Status": ["tests.fixtures.namespace_validator.rule1_loose_enum", "Status"],
    "T": ["tests.unit.test_utilities_type_checker_coverage_100", "T"],
    "TMessage": ["tests.unit.test_utilities_type_checker_coverage_100", "TMessage"],
    "T_co": ["tests.typings", "T_co"],
    "T_contra": ["tests.typings", "T_contra"],
    "TestAdvancedPatterns": [
        "tests.integration.patterns.test_advanced_patterns",
        "TestAdvancedPatterns",
    ],
    "TestArchitecturalPatterns": [
        "tests.integration.patterns.test_architectural_patterns",
        "TestArchitecturalPatterns",
    ],
    "TestAutomatedArchitecture": [
        "tests.integration.test_architecture",
        "TestAutomatedArchitecture",
    ],
    "TestCaseMap": ["tests.unit._models_impl", "TestCaseMap"],
    "TestCollectionUtilitiesCoverage": [
        "tests.unit.test_collection_utilities_coverage_100",
        "TestCollectionUtilitiesCoverage",
    ],
    "TestCompleteFlextSystemIntegration": [
        "tests.integration.test_system",
        "TestCompleteFlextSystemIntegration",
    ],
    "TestConstants": ["tests.unit.test_constants", "TestConstants"],
    "TestContainerFullCoverage": [
        "tests.unit.test_container_full_coverage",
        "TestContainerFullCoverage",
    ],
    "TestContainerMemory": [
        "tests.benchmark.test_container_memory",
        "TestContainerMemory",
    ],
    "TestContainerPerformance": [
        "tests.benchmark.test_container_performance",
        "TestContainerPerformance",
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
    "TestDataGenerators": ["tests.helpers.factories_impl", "TestDataGenerators"],
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
    "TestDocumentedPatterns": [
        "tests.test_documented_patterns",
        "TestDocumentedPatterns",
    ],
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
    "TestFlextSettingsSingletonIntegration": [
        "tests.integration.test_config_integration",
        "TestFlextSettingsSingletonIntegration",
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
    "TestFunction": [
        "tests.integration.patterns.test_advanced_patterns",
        "TestFunction",
    ],
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
    "TestHelperFactories": ["tests.helpers.factories", "TestHelperFactories"],
    "TestHelperScenarios": ["tests.helpers.scenarios", "TestHelperScenarios"],
    "TestIdempotency": [
        "tests.integration.test_refactor_nesting_idempotency",
        "TestIdempotency",
    ],
    "TestInfraIntegration": [
        "tests.integration.test_infra_integration",
        "TestInfraIntegration",
    ],
    "TestLibraryIntegration": [
        "tests.integration.test_integration",
        "TestLibraryIntegration",
    ],
    "TestLoggingsErrorPaths": [
        "tests.unit.test_loggings_error_paths_coverage",
        "TestLoggingsErrorPaths",
    ],
    "TestLoggingsStrictReturns": [
        "tests.unit.test_loggings_strict_returns",
        "TestLoggingsStrictReturns",
    ],
    "TestMigrationValidation": [
        "tests.integration.test_migration_validation",
        "TestMigrationValidation",
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
    "TestPatternsCommands": [
        "tests.integration.patterns.test_patterns_commands",
        "TestPatternsCommands",
    ],
    "TestPatternsLogging": [
        "tests.integration.patterns.test_patterns_logging",
        "TestPatternsLogging",
    ],
    "TestPatternsTesting": [
        "tests.integration.patterns.test_patterns_testing",
        "TestPatternsTesting",
    ],
    "TestPerformanceBenchmarks": [
        "tests.benchmark.test_refactor_nesting_performance",
        "TestPerformanceBenchmarks",
    ],
    "TestProjectLevelRefactor": [
        "tests.integration.test_refactor_nesting_project",
        "TestProjectLevelRefactor",
    ],
    "TestRefactorPolicyMRO": [
        "tests.integration.test_refactor_policy_mro",
        "TestRefactorPolicyMRO",
    ],
    "TestResultExceptionCarrying": [
        "tests.unit.test_result_exception_carrying",
        "TestResultExceptionCarrying",
    ],
    "TestRuntimeCoverage100": [
        "tests.unit.test_runtime_coverage_100",
        "TestRuntimeCoverage100",
    ],
    "TestService": ["tests.integration.test_service", "TestService"],
    "TestService100Coverage": [
        "tests.unit.test_service_coverage_100",
        "TestService100Coverage",
    ],
    "TestServiceBootstrap": [
        "tests.unit.test_service_bootstrap",
        "TestServiceBootstrap",
    ],
    "TestServiceInternals": ["tests.unit.test_service", "TestServiceInternals"],
    "TestServiceResultProperty": [
        "tests.test_service_result_property",
        "TestServiceResultProperty",
    ],
    "TestTypings": ["tests.unit.test_typings", "TestTypings"],
    "TestTypingsFullCoverage": [
        "tests.unit.test_typings_full_coverage",
        "TestTypingsFullCoverage",
    ],
    "TestUnitModels": ["tests.unit._models", "TestUnitModels"],
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
    "TestUtilitiesEnumFullCoverage": [
        "tests.unit.test_utilities_enum_full_coverage",
        "TestUtilitiesEnumFullCoverage",
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
    "TestUtils": ["tests.test_utils", "TestUtils"],
    "TestWorkspaceLevelRefactor": [
        "tests.integration.test_refactor_nesting_workspace",
        "TestWorkspaceLevelRefactor",
    ],
    "Teste": ["tests.unit.test_exceptions", "Teste"],
    "Testr": ["tests.unit.test_result", "Testr"],
    "TestrCoverage": ["tests.unit.test_result_coverage_100", "TestrCoverage"],
    "TestsCore": ["tests.unit.test_service", "TestsCore"],
    "TestsFlextServiceBase": ["tests.base", "TestsFlextServiceBase"],
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
    "TestuStringParser": [
        "tests.unit.test_utilities_string_parser",
        "TestuStringParser",
    ],
    "TestuTypeChecker": [
        "tests.unit.test_utilities_type_checker_coverage_100",
        "TestuTypeChecker",
    ],
    "TextUtilityContract": [
        "tests.unit.contracts.text_contract",
        "TextUtilityContract",
    ],
    "TimeoutEnforcer": [
        "tests.unit.test_dispatcher_timeout_coverage_100",
        "TimeoutEnforcer",
    ],
    "User": ["tests.helpers.factories_impl", "User"],
    "UserFactory": ["tests.helpers.factories_impl", "UserFactory"],
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
    "ValidatingService": ["tests.helpers.factories_impl", "ValidatingService"],
    "ValidatingServiceAuto": ["tests.helpers.factories_impl", "ValidatingServiceAuto"],
    "ValidatingServiceAutoFactory": [
        "tests.helpers.factories_impl",
        "ValidatingServiceAutoFactory",
    ],
    "ValidatingServiceFactory": [
        "tests.helpers.factories_impl",
        "ValidatingServiceFactory",
    ],
    "ValidationScenario": ["tests.helpers._scenarios_impl", "ValidationScenario"],
    "ValidationScenarios": ["tests.helpers._scenarios_impl", "ValidationScenarios"],
    "_BadCopyModel": ["tests.unit._models_impl", "_BadCopyModel"],
    "_BrokenDumpModel": ["tests.unit._models_impl", "_BrokenDumpModel"],
    "_Cfg": ["tests.unit._models_impl", "_Cfg"],
    "_DumpErrorModel": ["tests.unit._models_impl", "_DumpErrorModel"],
    "_ErrorsModel": ["tests.unit._models_impl", "_ErrorsModel"],
    "_FakeConfig": ["tests.unit._models_impl", "_FakeConfig"],
    "_FrozenEntity": ["tests.unit._models_impl", "_FrozenEntity"],
    "_GoodModel": ["tests.unit._models_impl", "_GoodModel"],
    "_Model": ["tests.unit._models_impl", "_Model"],
    "_MsgWithCommandId": ["tests.unit._models_impl", "_MsgWithCommandId"],
    "_MsgWithMessageId": ["tests.unit._models_impl", "_MsgWithMessageId"],
    "_Opts": ["tests.unit._models_impl", "_Opts"],
    "_PlainErrorModel": ["tests.unit._models_impl", "_PlainErrorModel"],
    "_SampleEntity": ["tests.unit._models_impl", "_SampleEntity"],
    "_SvcModel": ["tests.unit._models_impl", "_SvcModel"],
    "_TargetModel": ["tests.unit._models_impl", "_TargetModel"],
    "_ValidationLikeError": ["tests.unit._models_impl", "_ValidationLikeError"],
    "assert_rejects": ["tests.conftest", "assert_rejects"],
    "assert_validates": ["tests.conftest", "assert_validates"],
    "assertion_helpers": ["tests.test_utils", "assertion_helpers"],
    "benchmark": ["tests.benchmark", ""],
    "c": ["tests.constants", "FlextCoreTestConstants"],
    "clean_container": ["tests.conftest", "clean_container"],
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
    "create_validate_entity_has_id_cases": [
        "tests.unit.test_utilities_domain",
        "create_validate_entity_has_id_cases",
    ],
    "create_validate_value_object_immutable_cases": [
        "tests.unit.test_utilities_domain",
        "create_validate_value_object_immutable_cases",
    ],
    "d": ["flext_tests", "d"],
    "e": ["flext_tests", "e"],
    "empty_strings": ["tests.conftest", "empty_strings"],
    "fixture_factory": ["tests.test_utils", "fixture_factory"],
    "flext_result_failure": ["tests.conftest", "flext_result_failure"],
    "flext_result_success": ["tests.conftest", "flext_result_success"],
    "flext_tests": ["tests.unit.flext_tests", ""],
    "generators_module": [
        "tests.unit.test_utilities_generators_full_coverage",
        "generators_module",
    ],
    "get_memory_usage": ["tests.benchmark.test_container_memory", "get_memory_usage"],
    "h": ["flext_tests", "h"],
    "handlers_module": ["tests.unit.test_handlers_full_coverage", "handlers_module"],
    "helper": ["tests.fixtures.namespace_validator.rule0_no_class", "helper"],
    "helpers": ["tests.helpers", ""],
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
    "integration": ["tests.integration", ""],
    "invalid_hostnames": ["tests.conftest", "invalid_hostnames"],
    "invalid_port_numbers": ["tests.conftest", "invalid_port_numbers"],
    "invalid_uris": ["tests.conftest", "invalid_uris"],
    "m": ["tests.models", "FlextCoreTestModels"],
    "mapper": ["tests.unit.test_utilities_mapper_full_coverage", "mapper"],
    "mock_external_service": ["tests.conftest", "mock_external_service"],
    "normalized_value_key_cases": [
        "tests.unit.test_utilities_string_parser",
        "normalized_value_key_cases",
    ],
    "out_of_range": ["tests.conftest", "out_of_range"],
    "p": ["tests.protocols", "FlextCoreTestProtocols"],
    "parser_scenarios": ["tests.conftest", "parser_scenarios"],
    "patterns": ["tests.integration.patterns", ""],
    "pytestmark": ["tests.unit.test_utilities_type_checker_coverage_100", "pytestmark"],
    "r": ["flext_tests", "r"],
    "reliability_scenarios": ["tests.conftest", "reliability_scenarios"],
    "reset_all_factories": ["tests.helpers.factories_impl", "reset_all_factories"],
    "reset_global_container": ["tests.conftest", "reset_global_container"],
    "reset_runtime_state": [
        "tests.unit.test_runtime_full_coverage",
        "reset_runtime_state",
    ],
    "runtime_cov_tests": ["tests.unit.test_runtime_full_coverage", "runtime_cov_tests"],
    "runtime_module": [
        "tests.unit.test_utilities_generators_full_coverage",
        "runtime_module",
    ],
    "runtime_tests": ["tests.unit.test_runtime_full_coverage", "runtime_tests"],
    "s": ["flext_tests", "s"],
    "sample_data": ["tests.conftest", "sample_data"],
    "t": ["tests.typings", "FlextCoreTestTypes"],
    "temp_dir": ["tests.conftest", "temp_dir"],
    "temp_directory": ["tests.conftest", "temp_directory"],
    "temp_file": ["tests.conftest", "temp_file"],
    "test_accessor_take_pick_as_or_flat_and_agg_branches": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_accessor_take_pick_as_or_flat_and_agg_branches",
    ],
    "test_aliases_are_available": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_aliases_are_available",
    ],
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
    "test_class_nesting_refactor_single_file_end_to_end": [
        "tests.integration.test_refactor_nesting_file",
        "test_class_nesting_refactor_single_file_end_to_end",
    ],
    "test_clear_keys_values_items_and_validate_branches": [
        "tests.unit.test_context_full_coverage",
        "test_clear_keys_values_items_and_validate_branches",
    ],
    "test_command_pagination_limit": [
        "tests.unit.test_models_cqrs_full_coverage",
        "test_command_pagination_limit",
    ],
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
    "test_construct_transform_and_deep_eq_branches": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_construct_transform_and_deep_eq_branches",
    ],
    "test_container_and_service_domain_paths": [
        "tests.unit.test_context_full_coverage",
        "test_container_and_service_domain_paths",
    ],
    "test_context": ["tests.conftest", "test_context"],
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
    "test_conversion_and_extract_success_branches": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_conversion_and_extract_success_branches",
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
    "test_data_factory": ["tests.test_utils", "test_data_factory"],
    "test_decorators_family_blocks_dispatcher_target": [
        "tests.unit.test_refactor_policy_family_rules",
        "test_decorators_family_blocks_dispatcher_target",
    ],
    "test_dependency_integration_and_wiring_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_dependency_integration_and_wiring_paths",
    ],
    "test_dependency_registration_duplicate_guards": [
        "tests.unit.test_runtime_full_coverage",
        "test_dependency_registration_duplicate_guards",
    ],
    "test_discover_project_roots_without_nested_git_dirs": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_discover_project_roots_without_nested_git_dirs",
    ],
    "test_dispatcher_family_blocks_models_target": [
        "tests.unit.test_refactor_policy_family_rules",
        "test_dispatcher_family_blocks_models_target",
    ],
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
    "test_field_and_fields_multi_branches": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_field_and_fields_multi_branches",
    ],
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
    "test_get_service_info": [
        "tests.unit.test_service_additional",
        "test_get_service_info",
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
    "test_invert_and_json_conversion_branches": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_invert_and_json_conversion_branches",
    ],
    "test_is_container_negative_paths_and_callable": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_container_negative_paths_and_callable",
    ],
    "test_is_flexible_value_covers_all_branches": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_is_flexible_value_covers_all_branches",
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
    "test_map_error_identity_and_transform": [
        "tests.unit.test_result_additional",
        "test_map_error_identity_and_transform",
    ],
    "test_map_flags_collect_and_invert_branches": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_map_flags_collect_and_invert_branches",
    ],
    "test_map_flat_map_and_then_paths": [
        "tests.unit.test_result_full_coverage",
        "test_map_flat_map_and_then_paths",
    ],
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
    "test_model_helpers_remaining_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_model_helpers_remaining_paths",
    ],
    "test_model_support_and_hash_compare_paths": [
        "tests.unit.test_runtime_full_coverage",
        "test_model_support_and_hash_compare_paths",
    ],
    "test_models_family_blocks_utilities_target": [
        "tests.unit.test_refactor_policy_family_rules",
        "test_models_family_blocks_utilities_target",
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
    "test_process_context_data_and_related_convenience": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_process_context_data_and_related_convenience",
    ],
    "test_protocol_and_simple_guard_helpers": [
        "tests.unit.test_utilities_guards_full_coverage",
        "test_protocol_and_simple_guard_helpers",
    ],
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
    "test_refactor_utilities_iter_python_files_includes_examples_and_scripts": [
        "tests.unit.test_refactor_migrate_to_class_mro",
        "test_refactor_utilities_iter_python_files_includes_examples_and_scripts",
    ],
    "test_remaining_build_fields_construct_and_eq_paths": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_remaining_build_fields_construct_and_eq_paths",
    ],
    "test_remaining_uncovered_branches": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_remaining_uncovered_branches",
    ],
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
    "test_runtime_create_instance_failure_branch": [
        "tests.unit.test_runtime_full_coverage",
        "test_runtime_create_instance_failure_branch",
    ],
    "test_runtime_family_blocks_non_runtime_target": [
        "tests.unit.test_refactor_policy_family_rules",
        "test_runtime_family_blocks_non_runtime_target",
    ],
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
    "test_set_set_all_get_validation_and_error_paths": [
        "tests.unit.test_context_full_coverage",
        "test_set_set_all_get_validation_and_error_paths",
    ],
    "test_small_mapper_convenience_methods": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_small_mapper_convenience_methods",
    ],
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
    "test_type_guards_and_narrowing_failures": [
        "tests.unit.test_utilities_mapper_full_coverage",
        "test_type_guards_and_narrowing_failures",
    ],
    "test_type_guards_result": [
        "tests.unit.test_result_full_coverage",
        "test_type_guards_result",
    ],
    "test_ultrawork_models_cli_runs_dry_run_copy": [
        "tests.unit.test_refactor_cli_models_workflow",
        "test_ultrawork_models_cli_runs_dry_run_copy",
    ],
    "test_update_statistics_remove_hook_and_clone_false_result": [
        "tests.unit.test_context_full_coverage",
        "test_update_statistics_remove_hook_and_clone_false_result",
    ],
    "test_utilities_family_allows_utilities_target": [
        "tests.unit.test_refactor_policy_family_rules",
        "test_utilities_family_allows_utilities_target",
    ],
    "test_validation_like_error_structure": [
        "tests.unit.test_result_full_coverage",
        "test_validation_like_error_structure",
    ],
    "test_with_resource_cleanup_runs": [
        "tests.unit.test_result_additional",
        "test_with_resource_cleanup_runs",
    ],
    "u": ["tests.utilities", "FlextCoreTestUtilities"],
    "unit": ["tests.unit", ""],
    "valid_hostnames": ["tests.conftest", "valid_hostnames"],
    "valid_port_numbers": ["tests.conftest", "valid_port_numbers"],
    "valid_ranges": ["tests.conftest", "valid_ranges"],
    "valid_strings": ["tests.conftest", "valid_strings"],
    "valid_uris": ["tests.conftest", "valid_uris"],
    "validation_scenarios": ["tests.conftest", "validation_scenarios"],
    "whitespace_strings": ["tests.conftest", "whitespace_strings"],
    "x": ["flext_tests", "x"],
}

__all__ = [
    "DEFAULT_TIMEOUT",
    "EXPECTED_BULK_SIZE",
    "MAX_RETRIES",
    "MAX_VALUE",
    "NORMALIZE_COMPONENT_SCENARIOS",
    "SORT_KEY_SCENARIOS",
    "AttrObject",
    "BadBool",
    "BadConfigForTest",
    "BadMapping",
    "BadString",
    "CacheTestModel",
    "CircuitBreakerManager",
    "ClearCacheScenario",
    "ComplexModel",
    "ConfigModelForTest",
    "ExplodingLenList",
    "FailingService",
    "FailingServiceAuto",
    "FailingServiceAutoFactory",
    "FailingServiceFactory",
    "FlextCoreTestConstants",
    "FlextCoreTestModels",
    "FlextCoreTestProtocols",
    "FlextCoreTestTypes",
    "FlextCoreTestUtilities",
    "FlextProtocols",
    "FlextTestConstants",
    "FlextTestModels",
    "FlextTestResult",
    "FlextTestResultCo",
    "FlextTestTypes",
    "FlextTestUtilities",
    "FunctionalExternalService",
    "GenericModelFactory",
    "GetUserService",
    "GetUserServiceAuto",
    "GetUserServiceAutoFactory",
    "GetUserServiceFactory",
    "InputPayloadMap",
    "InvalidModelForTest",
    "LooseTypeAlias",
    "NestedClassPropagationTransformer",
    "NestedModel",
    "NormalizeComponentScenario",
    "ParserScenario",
    "ParserScenarios",
    "Provide",
    "RandomConstants",
    "RateLimiterManager",
    "ReliabilityScenario",
    "ReliabilityScenarios",
    "RetryPolicy",
    "Rule0LooseItemsFixture",
    "Rule0MultipleClassesFixture",
    "Rule1LooseEnumFixture",
    "RuntimeCloneService",
    "SampleModel",
    "ServiceFactoryRegistry",
    "ServiceTestCase",
    "ServiceTestCaseFactory",
    "ServiceTestCases",
    "SimpleObj",
    "SingletonClassForTest",
    "SortKeyScenario",
    "Status",
    "T",
    "TMessage",
    "T_co",
    "T_contra",
    "TestAdvancedPatterns",
    "TestArchitecturalPatterns",
    "TestAutomatedArchitecture",
    "TestCaseMap",
    "TestCollectionUtilitiesCoverage",
    "TestCompleteFlextSystemIntegration",
    "TestConstants",
    "TestContainerFullCoverage",
    "TestContainerMemory",
    "TestContainerPerformance",
    "TestContext100Coverage",
    "TestCoverageContext",
    "TestCoverageExceptions",
    "TestCoverageLoggings",
    "TestCoverageModels",
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
    "TestFlextSettingsSingletonIntegration",
    "TestFlextTestsDomains",
    "TestFlextTestsFiles",
    "TestFlextTestsMatchers",
    "TestFlextUtilitiesArgs",
    "TestFlextUtilitiesConfiguration",
    "TestFlextUtilitiesReliability",
    "TestFlextVersion",
    "TestFunction",
    "TestHandlerDecoratorDiscovery",
    "TestHandlersFullCoverage",
    "TestHelperConsolidationTransformer",
    "TestHelperFactories",
    "TestHelperScenarios",
    "TestIdempotency",
    "TestInfraIntegration",
    "TestLibraryIntegration",
    "TestLoggingsErrorPaths",
    "TestLoggingsStrictReturns",
    "TestMigrationValidation",
    "TestMixinsFullCoverage",
    "TestModels",
    "TestModelsBaseFullCoverage",
    "TestModule",
    "TestPaginationCoverage100",
    "TestPatternsCommands",
    "TestPatternsLogging",
    "TestPatternsTesting",
    "TestPerformanceBenchmarks",
    "TestProjectLevelRefactor",
    "TestRefactorPolicyMRO",
    "TestResultExceptionCarrying",
    "TestRuntimeCoverage100",
    "TestService",
    "TestService100Coverage",
    "TestServiceBootstrap",
    "TestServiceInternals",
    "TestServiceResultProperty",
    "TestTypings",
    "TestTypingsFullCoverage",
    "TestUnitModels",
    "TestUtilities",
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
    "TestUtils",
    "TestWorkspaceLevelRefactor",
    "Teste",
    "Testr",
    "TestrCoverage",
    "TestsCore",
    "TestsFlextServiceBase",
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
    "User",
    "UserFactory",
    "UtilitiesCacheCoverage100Namespace",
    "UtilitiesMapperCoverage100Namespace",
    "UtilitiesMapperFullCoverageNamespace",
    "ValidatingService",
    "ValidatingServiceAuto",
    "ValidatingServiceAutoFactory",
    "ValidatingServiceFactory",
    "ValidationScenario",
    "ValidationScenarios",
    "_BadCopyModel",
    "_BrokenDumpModel",
    "_Cfg",
    "_DumpErrorModel",
    "_ErrorsModel",
    "_FakeConfig",
    "_FrozenEntity",
    "_GoodModel",
    "_Model",
    "_MsgWithCommandId",
    "_MsgWithMessageId",
    "_Opts",
    "_PlainErrorModel",
    "_SampleEntity",
    "_SvcModel",
    "_TargetModel",
    "_ValidationLikeError",
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
    "helper",
    "helpers",
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
    "integration",
    "invalid_hostnames",
    "invalid_port_numbers",
    "invalid_uris",
    "m",
    "mapper",
    "mock_external_service",
    "normalized_value_key_cases",
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
    "test_class_nesting_refactor_single_file_end_to_end",
    "test_clear_keys_values_items_and_validate_branches",
    "test_command_pagination_limit",
    "test_config_bridge_and_trace_context_and_http_validation",
    "test_configuration_mapping_and_dict_negative_branches",
    "test_configure_structlog_edge_paths",
    "test_configure_structlog_print_logger_factory_fallback",
    "test_construct_transform_and_deep_eq_branches",
    "test_container_and_service_domain_paths",
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
    "test_discover_project_roots_without_nested_git_dirs",
    "test_dispatcher_family_blocks_models_target",
    "test_ensure_and_extract_array_index_helpers",
    "test_ensure_trace_context_dict_conversion_paths",
    "test_entity_comparable_map_and_bulk_validation_paths",
    "test_execute_and_register_handler_failure_paths",
    "test_export_paths_with_metadata_and_statistics",
    "test_extract_error_paths_and_prop_accessor",
    "test_extract_field_value_and_ensure_variants",
    "test_extract_mapping_or_none_branches",
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
    "test_normalize_to_metadata_alias_removal_path",
    "test_ok_accepts_none",
    "test_operation_progress_start_operation_sets_runtime_fields",
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
    "test_set_set_all_get_validation_and_error_paths",
    "test_small_mapper_convenience_methods",
    "test_statistics_and_custom_fields_validators",
    "test_structlog_proxy_context_var_default_when_key_missing",
    "test_structlog_proxy_context_var_get_set_reset_paths",
    "test_summary_error_paths_and_bindings_failures",
    "test_summary_properties_and_subclass_storage_reset",
    "test_to_general_value_dict_removed",
    "test_transform_option_extract_and_step_helpers",
    "test_type_guards_and_narrowing_failures",
    "test_type_guards_result",
    "test_ultrawork_models_cli_runs_dry_run_copy",
    "test_update_statistics_remove_hook_and_clone_false_result",
    "test_utilities_family_allows_utilities_target",
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


_LAZY_CACHE: MutableMapping[str, FlextTypes.ModuleExport] = {}


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


def __dir__() -> Sequence[str]:
    """Return list of available attributes for dir() and autocomplete.

    Returns:
        List of public names from module exports.

    """
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
