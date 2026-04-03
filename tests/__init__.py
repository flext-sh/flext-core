# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Tests package."""

from __future__ import annotations

import typing as _t

from flext_core.decorators import FlextDecorators as d
from flext_core.exceptions import FlextExceptions as e
from flext_core.handlers import FlextHandlers as h
from flext_core.lazy import install_lazy_exports, merge_lazy_imports
from flext_core.mixins import FlextMixins as x
from flext_core.result import FlextResult as r
from flext_core.service import FlextService as s
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
from tests.helpers.scenarios import TestHelperScenarios
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
from tests.integration.patterns.test_patterns_testing import (
    P,
    R,
    TestPatternsTesting,
)
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
    T,
    T_co,
    T_contra,
)
from tests.unit._models.test_base import TestFlextModelsBase
from tests.unit._models.test_cqrs import TestFlextModelsCqrs
from tests.unit._models.test_entity import TestFlextModelsEntity
from tests.unit._models.test_errors import TestFlextModelsErrors
from tests.unit._models.test_exception_params import TestFlextModelsExceptionParams
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
from tests.unit._utilities.test_guards import TestFlextUtilitiesGuards
from tests.unit._utilities.test_mapper import TestFlextUtilitiesMapper
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
from tests.unit.protocols import FlextUnitTestProtocols
from tests.unit.test_args_coverage_100 import TestFlextUtilitiesArgs
from tests.unit.test_collection_utilities_coverage_100 import (
    TestCollectionUtilitiesCoverage,
)
from tests.unit.test_collections_coverage_100 import (
    TestFlextModelsCollectionsCoverage100,
)
from tests.unit.test_config import TestFlextSettings
from tests.unit.test_constants_new import TestFlextConstants
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
from tests.unit.test_handlers_full_coverage import TestHandlersFullCoverage
from tests.unit.test_loggings_error_paths_coverage import TestLoggingsErrorPaths
from tests.unit.test_loggings_full_coverage import TestModule
from tests.unit.test_loggings_strict_returns import TestLoggingsStrictReturns
from tests.unit.test_mixins import TestFlextMixinsNestedClasses
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
from tests.unit.test_protocols_new import TestFlextProtocols
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
    runtime_module,
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
    test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage,
    test_nested_class_propagation_updates_import_annotations_and_calls,
)
from tests.unit.test_typings_full_coverage import TestTypingsFullCoverage
from tests.unit.test_typings_new import TestFlextTypes
from tests.unit.test_utilities_cache_coverage_100 import (
    NORMALIZE_COMPONENT_SCENARIOS,
    NormalizeComponentScenario,
    TestuCacheLogger,
    TestuCacheNormalizeComponent,
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
)
from tests.unit.test_utilities_domain_full_coverage import (
    TestUtilitiesDomainFullCoverage,
)
from tests.unit.test_utilities_generators_full_coverage import (
    TestUtilitiesGeneratorsFullCoverage,
    generators_module,
)
from tests.unit.test_utilities_guards_full_coverage import (
    test_aliases_are_available,
    test_chk_exercises_missed_branches,
    test_configuration_mapping_and_dict_negative_branches,
    test_guard_in_has_empty_none_helpers,
    test_guard_instance_attribute_access_warnings,
    test_guards_bool_identity_branch_via_isinstance_fallback,
    test_guards_bool_shortcut_and_issubclass_typeerror,
    test_guards_handler_type_issubclass_typeerror_branch_direct,
    test_guards_issubclass_success_when_callable_is_patched,
    test_guards_issubclass_typeerror_when_class_not_treated_as_callable,
    test_is_container_negative_paths_and_callable,
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
    test_bad_string_and_bad_bool_raise_value_error,
    test_build_apply_transform_and_process_error_paths,
    test_convert_default_fallback_matrix,
    test_convert_sequence_branch_returns_tuple,
    test_extract_array_index_helpers,
    test_extract_error_paths_and_prop_accessor,
    test_extract_field_value_and_ensure_variants,
    test_filter_map_normalize_convert_helpers,
    test_general_value_helpers_and_logger,
    test_group_sort_unique_slice_chunk_branches,
    test_narrow_to_string_keyed_dict_and_mapping_paths,
    test_take_and_as_branches,
    test_transform_and_deep_eq_branches,
    test_transform_option_extract_and_step_helpers,
    test_type_guards_and_narrowing_failures,
)
from tests.unit.test_utilities_parser_full_coverage import (
    TestUtilitiesParserFullCoverage,
)
from tests.unit.test_utilities_reliability import TestFlextUtilitiesReliability
from tests.unit.test_utilities_text_full_coverage import (
    TestUtilitiesTextFullCoverage,
)
from tests.unit.test_utilities_type_checker_coverage_100 import (
    TestuTypeChecker,
    TMessage,
    pytestmark,
)
from tests.unit.test_utilities_type_guards_coverage_100 import (
    TestUtilitiesTypeGuardsCoverage100,
)
from tests.unit.test_version import TestFlextVersion
from tests.utilities import FlextCoreTestUtilities, FlextCoreTestUtilities as u

if _t.TYPE_CHECKING:
    import tests.base as _tests_base

    base = _tests_base
    import tests.benchmark as _tests_benchmark

    benchmark = _tests_benchmark
    import tests.benchmark.test_container_memory as _tests_benchmark_test_container_memory

    test_container_memory = _tests_benchmark_test_container_memory
    import tests.benchmark.test_container_performance as _tests_benchmark_test_container_performance

    test_container_performance = _tests_benchmark_test_container_performance
    import tests.benchmark.test_refactor_nesting_performance as _tests_benchmark_test_refactor_nesting_performance

    test_refactor_nesting_performance = (
        _tests_benchmark_test_refactor_nesting_performance
    )
    import tests.conftest as _tests_conftest

    conftest = _tests_conftest
    import tests.constants as _tests_constants

    constants = _tests_constants
    import tests.helpers as _tests_helpers

    helpers = _tests_helpers
    import tests.helpers.factories as _tests_helpers_factories

    factories = _tests_helpers_factories
    import tests.helpers.factories_impl as _tests_helpers_factories_impl

    factories_impl = _tests_helpers_factories_impl
    import tests.helpers.scenarios as _tests_helpers_scenarios

    scenarios = _tests_helpers_scenarios
    import tests.integration as _tests_integration

    integration = _tests_integration
    import tests.integration.patterns as _tests_integration_patterns

    patterns = _tests_integration_patterns
    import tests.integration.patterns.test_advanced_patterns as _tests_integration_patterns_test_advanced_patterns

    test_advanced_patterns = _tests_integration_patterns_test_advanced_patterns
    import tests.integration.patterns.test_architectural_patterns as _tests_integration_patterns_test_architectural_patterns

    test_architectural_patterns = (
        _tests_integration_patterns_test_architectural_patterns
    )
    import tests.integration.patterns.test_patterns_commands as _tests_integration_patterns_test_patterns_commands

    test_patterns_commands = _tests_integration_patterns_test_patterns_commands
    import tests.integration.patterns.test_patterns_logging as _tests_integration_patterns_test_patterns_logging

    test_patterns_logging = _tests_integration_patterns_test_patterns_logging
    import tests.integration.patterns.test_patterns_testing as _tests_integration_patterns_test_patterns_testing

    test_patterns_testing = _tests_integration_patterns_test_patterns_testing
    import tests.integration.test_architecture as _tests_integration_test_architecture

    test_architecture = _tests_integration_test_architecture
    import tests.integration.test_config_integration as _tests_integration_test_config_integration

    test_config_integration = _tests_integration_test_config_integration
    import tests.integration.test_infra_integration as _tests_integration_test_infra_integration

    test_infra_integration = _tests_integration_test_infra_integration
    import tests.integration.test_integration as _tests_integration_test_integration

    test_integration = _tests_integration_test_integration
    import tests.integration.test_migration_validation as _tests_integration_test_migration_validation

    test_migration_validation = _tests_integration_test_migration_validation
    import tests.integration.test_refactor_nesting_file as _tests_integration_test_refactor_nesting_file

    test_refactor_nesting_file = _tests_integration_test_refactor_nesting_file
    import tests.integration.test_refactor_nesting_idempotency as _tests_integration_test_refactor_nesting_idempotency

    test_refactor_nesting_idempotency = (
        _tests_integration_test_refactor_nesting_idempotency
    )
    import tests.integration.test_refactor_nesting_project as _tests_integration_test_refactor_nesting_project

    test_refactor_nesting_project = _tests_integration_test_refactor_nesting_project
    import tests.integration.test_refactor_nesting_workspace as _tests_integration_test_refactor_nesting_workspace

    test_refactor_nesting_workspace = _tests_integration_test_refactor_nesting_workspace
    import tests.integration.test_refactor_policy_mro as _tests_integration_test_refactor_policy_mro

    test_refactor_policy_mro = _tests_integration_test_refactor_policy_mro
    import tests.integration.test_service as _tests_integration_test_service

    test_service = _tests_integration_test_service
    import tests.integration.test_system as _tests_integration_test_system

    test_system = _tests_integration_test_system
    import tests.models as _tests_models

    models = _tests_models
    import tests.protocols as _tests_protocols

    protocols = _tests_protocols
    import tests.test_documented_patterns as _tests_test_documented_patterns

    test_documented_patterns = _tests_test_documented_patterns
    import tests.test_service_result_property as _tests_test_service_result_property

    test_service_result_property = _tests_test_service_result_property
    import tests.test_utils as _tests_test_utils

    test_utils = _tests_test_utils
    import tests.typings as _tests_typings

    typings = _tests_typings
    import tests.unit as _tests_unit

    unit = _tests_unit
    import tests.unit._models.test_base as _tests_unit__models_test_base

    test_base = _tests_unit__models_test_base
    import tests.unit._models.test_cqrs as _tests_unit__models_test_cqrs

    test_cqrs = _tests_unit__models_test_cqrs
    import tests.unit._models.test_entity as _tests_unit__models_test_entity

    test_entity = _tests_unit__models_test_entity
    import tests.unit._models.test_errors as _tests_unit__models_test_errors

    test_errors = _tests_unit__models_test_errors
    import tests.unit._models.test_exception_params as _tests_unit__models_test_exception_params

    test_exception_params = _tests_unit__models_test_exception_params
    import tests.unit._utilities.test_guards as _tests_unit__utilities_test_guards

    test_guards = _tests_unit__utilities_test_guards
    import tests.unit._utilities.test_mapper as _tests_unit__utilities_test_mapper

    test_mapper = _tests_unit__utilities_test_mapper
    import tests.unit.conftest_infra as _tests_unit_conftest_infra

    conftest_infra = _tests_unit_conftest_infra
    import tests.unit.contracts as _tests_unit_contracts

    contracts = _tests_unit_contracts
    import tests.unit.contracts.text_contract as _tests_unit_contracts_text_contract

    text_contract = _tests_unit_contracts_text_contract
    import tests.unit.flext_tests as _tests_unit_flext_tests

    flext_tests = _tests_unit_flext_tests
    import tests.unit.flext_tests.test_docker as _tests_unit_flext_tests_test_docker

    test_docker = _tests_unit_flext_tests_test_docker
    import tests.unit.flext_tests.test_domains as _tests_unit_flext_tests_test_domains

    test_domains = _tests_unit_flext_tests_test_domains
    import tests.unit.flext_tests.test_files as _tests_unit_flext_tests_test_files

    test_files = _tests_unit_flext_tests_test_files
    import tests.unit.flext_tests.test_matchers as _tests_unit_flext_tests_test_matchers

    test_matchers = _tests_unit_flext_tests_test_matchers
    import tests.unit.test_args_coverage_100 as _tests_unit_test_args_coverage_100

    test_args_coverage_100 = _tests_unit_test_args_coverage_100
    import tests.unit.test_collection_utilities_coverage_100 as _tests_unit_test_collection_utilities_coverage_100

    test_collection_utilities_coverage_100 = (
        _tests_unit_test_collection_utilities_coverage_100
    )
    import tests.unit.test_collections_coverage_100 as _tests_unit_test_collections_coverage_100

    test_collections_coverage_100 = _tests_unit_test_collections_coverage_100
    import tests.unit.test_config as _tests_unit_test_config

    test_config = _tests_unit_test_config
    import tests.unit.test_constants_new as _tests_unit_test_constants_new

    test_constants_new = _tests_unit_test_constants_new
    import tests.unit.test_container as _tests_unit_test_container

    test_container = _tests_unit_test_container
    import tests.unit.test_container_full_coverage as _tests_unit_test_container_full_coverage

    test_container_full_coverage = _tests_unit_test_container_full_coverage
    import tests.unit.test_context_coverage_100 as _tests_unit_test_context_coverage_100

    test_context_coverage_100 = _tests_unit_test_context_coverage_100
    import tests.unit.test_context_full_coverage as _tests_unit_test_context_full_coverage

    test_context_full_coverage = _tests_unit_test_context_full_coverage
    import tests.unit.test_coverage_context as _tests_unit_test_coverage_context

    test_coverage_context = _tests_unit_test_coverage_context
    import tests.unit.test_coverage_exceptions as _tests_unit_test_coverage_exceptions

    test_coverage_exceptions = _tests_unit_test_coverage_exceptions
    import tests.unit.test_coverage_loggings as _tests_unit_test_coverage_loggings

    test_coverage_loggings = _tests_unit_test_coverage_loggings
    import tests.unit.test_coverage_models as _tests_unit_test_coverage_models

    test_coverage_models = _tests_unit_test_coverage_models
    import tests.unit.test_coverage_utilities as _tests_unit_test_coverage_utilities

    test_coverage_utilities = _tests_unit_test_coverage_utilities
    import tests.unit.test_decorators as _tests_unit_test_decorators

    test_decorators = _tests_unit_test_decorators
    import tests.unit.test_decorators_discovery_full_coverage as _tests_unit_test_decorators_discovery_full_coverage

    test_decorators_discovery_full_coverage = (
        _tests_unit_test_decorators_discovery_full_coverage
    )
    import tests.unit.test_decorators_full_coverage as _tests_unit_test_decorators_full_coverage

    test_decorators_full_coverage = _tests_unit_test_decorators_full_coverage
    import tests.unit.test_deprecation_warnings as _tests_unit_test_deprecation_warnings

    test_deprecation_warnings = _tests_unit_test_deprecation_warnings
    import tests.unit.test_di_incremental as _tests_unit_test_di_incremental

    test_di_incremental = _tests_unit_test_di_incremental
    import tests.unit.test_di_services_access as _tests_unit_test_di_services_access

    test_di_services_access = _tests_unit_test_di_services_access
    import tests.unit.test_dispatcher_di as _tests_unit_test_dispatcher_di

    test_dispatcher_di = _tests_unit_test_dispatcher_di
    import tests.unit.test_dispatcher_full_coverage as _tests_unit_test_dispatcher_full_coverage

    test_dispatcher_full_coverage = _tests_unit_test_dispatcher_full_coverage
    import tests.unit.test_dispatcher_minimal as _tests_unit_test_dispatcher_minimal

    test_dispatcher_minimal = _tests_unit_test_dispatcher_minimal
    import tests.unit.test_dispatcher_reliability as _tests_unit_test_dispatcher_reliability

    test_dispatcher_reliability = _tests_unit_test_dispatcher_reliability
    import tests.unit.test_dispatcher_timeout_coverage_100 as _tests_unit_test_dispatcher_timeout_coverage_100

    test_dispatcher_timeout_coverage_100 = (
        _tests_unit_test_dispatcher_timeout_coverage_100
    )
    import tests.unit.test_entity_coverage as _tests_unit_test_entity_coverage

    test_entity_coverage = _tests_unit_test_entity_coverage
    import tests.unit.test_enum_utilities_coverage_100 as _tests_unit_test_enum_utilities_coverage_100

    test_enum_utilities_coverage_100 = _tests_unit_test_enum_utilities_coverage_100
    import tests.unit.test_exceptions as _tests_unit_test_exceptions

    test_exceptions = _tests_unit_test_exceptions
    import tests.unit.test_handler_decorator_discovery as _tests_unit_test_handler_decorator_discovery

    test_handler_decorator_discovery = _tests_unit_test_handler_decorator_discovery
    import tests.unit.test_handlers as _tests_unit_test_handlers

    test_handlers = _tests_unit_test_handlers
    import tests.unit.test_handlers_full_coverage as _tests_unit_test_handlers_full_coverage

    test_handlers_full_coverage = _tests_unit_test_handlers_full_coverage
    import tests.unit.test_loggings_error_paths_coverage as _tests_unit_test_loggings_error_paths_coverage

    test_loggings_error_paths_coverage = _tests_unit_test_loggings_error_paths_coverage
    import tests.unit.test_loggings_full_coverage as _tests_unit_test_loggings_full_coverage

    test_loggings_full_coverage = _tests_unit_test_loggings_full_coverage
    import tests.unit.test_loggings_strict_returns as _tests_unit_test_loggings_strict_returns

    test_loggings_strict_returns = _tests_unit_test_loggings_strict_returns
    import tests.unit.test_mixins as _tests_unit_test_mixins

    test_mixins = _tests_unit_test_mixins
    import tests.unit.test_mixins_full_coverage as _tests_unit_test_mixins_full_coverage

    test_mixins_full_coverage = _tests_unit_test_mixins_full_coverage
    import tests.unit.test_models as _tests_unit_test_models

    test_models = _tests_unit_test_models
    import tests.unit.test_models_base_full_coverage as _tests_unit_test_models_base_full_coverage

    test_models_base_full_coverage = _tests_unit_test_models_base_full_coverage
    import tests.unit.test_models_container as _tests_unit_test_models_container

    test_models_container = _tests_unit_test_models_container
    import tests.unit.test_models_context_full_coverage as _tests_unit_test_models_context_full_coverage

    test_models_context_full_coverage = _tests_unit_test_models_context_full_coverage
    import tests.unit.test_models_cqrs_full_coverage as _tests_unit_test_models_cqrs_full_coverage

    test_models_cqrs_full_coverage = _tests_unit_test_models_cqrs_full_coverage
    import tests.unit.test_models_entity_full_coverage as _tests_unit_test_models_entity_full_coverage

    test_models_entity_full_coverage = _tests_unit_test_models_entity_full_coverage
    import tests.unit.test_models_generic_full_coverage as _tests_unit_test_models_generic_full_coverage

    test_models_generic_full_coverage = _tests_unit_test_models_generic_full_coverage
    import tests.unit.test_namespace_validator as _tests_unit_test_namespace_validator

    test_namespace_validator = _tests_unit_test_namespace_validator
    import tests.unit.test_protocols_new as _tests_unit_test_protocols_new

    test_protocols_new = _tests_unit_test_protocols_new
    import tests.unit.test_refactor_cli_models_workflow as _tests_unit_test_refactor_cli_models_workflow

    test_refactor_cli_models_workflow = _tests_unit_test_refactor_cli_models_workflow
    import tests.unit.test_refactor_migrate_to_class_mro as _tests_unit_test_refactor_migrate_to_class_mro

    test_refactor_migrate_to_class_mro = _tests_unit_test_refactor_migrate_to_class_mro
    import tests.unit.test_refactor_namespace_enforcer as _tests_unit_test_refactor_namespace_enforcer

    test_refactor_namespace_enforcer = _tests_unit_test_refactor_namespace_enforcer
    import tests.unit.test_refactor_policy_family_rules as _tests_unit_test_refactor_policy_family_rules

    test_refactor_policy_family_rules = _tests_unit_test_refactor_policy_family_rules
    import tests.unit.test_registry as _tests_unit_test_registry

    test_registry = _tests_unit_test_registry
    import tests.unit.test_registry_full_coverage as _tests_unit_test_registry_full_coverage

    test_registry_full_coverage = _tests_unit_test_registry_full_coverage
    import tests.unit.test_result as _tests_unit_test_result

    test_result = _tests_unit_test_result
    import tests.unit.test_result_additional as _tests_unit_test_result_additional

    test_result_additional = _tests_unit_test_result_additional
    import tests.unit.test_result_coverage_100 as _tests_unit_test_result_coverage_100

    test_result_coverage_100 = _tests_unit_test_result_coverage_100
    import tests.unit.test_result_exception_carrying as _tests_unit_test_result_exception_carrying

    test_result_exception_carrying = _tests_unit_test_result_exception_carrying
    import tests.unit.test_result_full_coverage as _tests_unit_test_result_full_coverage

    test_result_full_coverage = _tests_unit_test_result_full_coverage
    import tests.unit.test_runtime as _tests_unit_test_runtime

    test_runtime = _tests_unit_test_runtime
    import tests.unit.test_runtime_coverage_100 as _tests_unit_test_runtime_coverage_100

    test_runtime_coverage_100 = _tests_unit_test_runtime_coverage_100
    import tests.unit.test_runtime_full_coverage as _tests_unit_test_runtime_full_coverage

    test_runtime_full_coverage = _tests_unit_test_runtime_full_coverage
    import tests.unit.test_service_additional as _tests_unit_test_service_additional

    test_service_additional = _tests_unit_test_service_additional
    import tests.unit.test_service_bootstrap as _tests_unit_test_service_bootstrap

    test_service_bootstrap = _tests_unit_test_service_bootstrap
    import tests.unit.test_service_coverage_100 as _tests_unit_test_service_coverage_100

    test_service_coverage_100 = _tests_unit_test_service_coverage_100
    import tests.unit.test_settings_coverage as _tests_unit_test_settings_coverage

    test_settings_coverage = _tests_unit_test_settings_coverage
    import tests.unit.test_transformer_class_nesting as _tests_unit_test_transformer_class_nesting

    test_transformer_class_nesting = _tests_unit_test_transformer_class_nesting
    import tests.unit.test_transformer_helper_consolidation as _tests_unit_test_transformer_helper_consolidation

    test_transformer_helper_consolidation = (
        _tests_unit_test_transformer_helper_consolidation
    )
    import tests.unit.test_transformer_nested_class_propagation as _tests_unit_test_transformer_nested_class_propagation

    test_transformer_nested_class_propagation = (
        _tests_unit_test_transformer_nested_class_propagation
    )
    import tests.unit.test_typings_full_coverage as _tests_unit_test_typings_full_coverage

    test_typings_full_coverage = _tests_unit_test_typings_full_coverage
    import tests.unit.test_typings_new as _tests_unit_test_typings_new

    test_typings_new = _tests_unit_test_typings_new
    import tests.unit.test_utilities as _tests_unit_test_utilities

    test_utilities = _tests_unit_test_utilities
    import tests.unit.test_utilities_cache_coverage_100 as _tests_unit_test_utilities_cache_coverage_100

    test_utilities_cache_coverage_100 = _tests_unit_test_utilities_cache_coverage_100
    import tests.unit.test_utilities_collection_coverage_100 as _tests_unit_test_utilities_collection_coverage_100

    test_utilities_collection_coverage_100 = (
        _tests_unit_test_utilities_collection_coverage_100
    )
    import tests.unit.test_utilities_collection_full_coverage as _tests_unit_test_utilities_collection_full_coverage

    test_utilities_collection_full_coverage = (
        _tests_unit_test_utilities_collection_full_coverage
    )
    import tests.unit.test_utilities_configuration_coverage_100 as _tests_unit_test_utilities_configuration_coverage_100

    test_utilities_configuration_coverage_100 = (
        _tests_unit_test_utilities_configuration_coverage_100
    )
    import tests.unit.test_utilities_configuration_full_coverage as _tests_unit_test_utilities_configuration_full_coverage

    test_utilities_configuration_full_coverage = (
        _tests_unit_test_utilities_configuration_full_coverage
    )
    import tests.unit.test_utilities_context_full_coverage as _tests_unit_test_utilities_context_full_coverage

    test_utilities_context_full_coverage = (
        _tests_unit_test_utilities_context_full_coverage
    )
    import tests.unit.test_utilities_coverage as _tests_unit_test_utilities_coverage

    test_utilities_coverage = _tests_unit_test_utilities_coverage
    import tests.unit.test_utilities_data_mapper as _tests_unit_test_utilities_data_mapper

    test_utilities_data_mapper = _tests_unit_test_utilities_data_mapper
    import tests.unit.test_utilities_domain as _tests_unit_test_utilities_domain

    test_utilities_domain = _tests_unit_test_utilities_domain
    import tests.unit.test_utilities_domain_full_coverage as _tests_unit_test_utilities_domain_full_coverage

    test_utilities_domain_full_coverage = (
        _tests_unit_test_utilities_domain_full_coverage
    )
    import tests.unit.test_utilities_enum_full_coverage as _tests_unit_test_utilities_enum_full_coverage

    test_utilities_enum_full_coverage = _tests_unit_test_utilities_enum_full_coverage
    import tests.unit.test_utilities_generators_full_coverage as _tests_unit_test_utilities_generators_full_coverage

    test_utilities_generators_full_coverage = (
        _tests_unit_test_utilities_generators_full_coverage
    )
    import tests.unit.test_utilities_guards_full_coverage as _tests_unit_test_utilities_guards_full_coverage

    test_utilities_guards_full_coverage = (
        _tests_unit_test_utilities_guards_full_coverage
    )
    import tests.unit.test_utilities_mapper_coverage_100 as _tests_unit_test_utilities_mapper_coverage_100

    test_utilities_mapper_coverage_100 = _tests_unit_test_utilities_mapper_coverage_100
    import tests.unit.test_utilities_mapper_full_coverage as _tests_unit_test_utilities_mapper_full_coverage

    test_utilities_mapper_full_coverage = (
        _tests_unit_test_utilities_mapper_full_coverage
    )
    import tests.unit.test_utilities_parser_full_coverage as _tests_unit_test_utilities_parser_full_coverage

    test_utilities_parser_full_coverage = (
        _tests_unit_test_utilities_parser_full_coverage
    )
    import tests.unit.test_utilities_reliability as _tests_unit_test_utilities_reliability

    test_utilities_reliability = _tests_unit_test_utilities_reliability
    import tests.unit.test_utilities_text_full_coverage as _tests_unit_test_utilities_text_full_coverage

    test_utilities_text_full_coverage = _tests_unit_test_utilities_text_full_coverage
    import tests.unit.test_utilities_type_checker_coverage_100 as _tests_unit_test_utilities_type_checker_coverage_100

    test_utilities_type_checker_coverage_100 = (
        _tests_unit_test_utilities_type_checker_coverage_100
    )
    import tests.unit.test_utilities_type_guards_coverage_100 as _tests_unit_test_utilities_type_guards_coverage_100

    test_utilities_type_guards_coverage_100 = (
        _tests_unit_test_utilities_type_guards_coverage_100
    )
    import tests.unit.test_version as _tests_unit_test_version

    test_version = _tests_unit_test_version
    import tests.utilities as _tests_utilities

    utilities = _tests_utilities

    _ = (
        AttrObject,
        BadBool,
        BadConfigForTest,
        BadMapping,
        BadString,
        CacheTestModel,
        ComplexModel,
        ConfigModelForTest,
        DEFAULT_TIMEOUT,
        EXPECTED_BULK_SIZE,
        ExplodingLenList,
        FailingService,
        FailingServiceAuto,
        FailingServiceAutoFactory,
        FailingServiceFactory,
        FlextCoreTestConstants,
        FlextCoreTestModels,
        FlextCoreTestProtocols,
        FlextCoreTestTypes,
        FlextCoreTestUtilities,
        FlextTestConstants,
        FlextTestModels,
        FlextTestResult,
        FlextTestResultCo,
        FlextTestTypes,
        FlextTestUtilities,
        FlextUnitTestProtocols,
        FunctionalExternalService,
        GenericModelFactory,
        GetUserService,
        GetUserServiceAuto,
        GetUserServiceAutoFactory,
        GetUserServiceFactory,
        InputPayloadMap,
        InvalidModelForTest,
        LooseTypeAlias,
        MAX_RETRIES,
        MAX_VALUE,
        NORMALIZE_COMPONENT_SCENARIOS,
        NestedModel,
        NormalizeComponentScenario,
        P,
        ParserScenario,
        ParserScenarios,
        R,
        RandomConstants,
        ReliabilityScenario,
        ReliabilityScenarios,
        Rule0LooseItemsFixture,
        Rule0MultipleClassesFixture,
        Rule1LooseEnumFixture,
        RuntimeCloneService,
        SampleModel,
        ServiceFactoryRegistry,
        ServiceTestCaseFactory,
        ServiceTestCases,
        SimpleObj,
        SingletonClassForTest,
        Status,
        T,
        TMessage,
        T_co,
        T_contra,
        TestAdvancedPatterns,
        TestArchitecturalPatterns,
        TestAutomatedArchitecture,
        TestCaseMap,
        TestCollectionUtilitiesCoverage,
        TestCompleteFlextSystemIntegration,
        TestContainerFullCoverage,
        TestContainerMemory,
        TestContainerPerformance,
        TestContext100Coverage,
        TestCoverageContext,
        TestCoverageExceptions,
        TestCoverageLoggings,
        TestCoverageModels,
        TestDIIncremental,
        TestDataGenerators,
        TestDecoratorsDiscoveryFullCoverage,
        TestDecoratorsFullCoverage,
        TestDeprecationWarnings,
        TestDiServicesAccess,
        TestDispatcherDI,
        TestDispatcherFullCoverage,
        TestDispatcherMinimal,
        TestDispatcherTimeoutCoverage100,
        TestDocker,
        TestDocumentedPatterns,
        TestEntityCoverageEdgeCases,
        TestEnumUtilitiesCoverage,
        TestExceptionsHypothesis,
        TestFlextConstants,
        TestFlextContainer,
        TestFlextContext,
        TestFlextDecorators,
        TestFlextHandlers,
        TestFlextInfraNamespaceValidator,
        TestFlextMixinsNestedClasses,
        TestFlextModelsBase,
        TestFlextModelsCollectionsCoverage100,
        TestFlextModelsContainer,
        TestFlextModelsCqrs,
        TestFlextModelsEntity,
        TestFlextModelsErrors,
        TestFlextModelsExceptionParams,
        TestFlextProtocols,
        TestFlextRegistry,
        TestFlextRuntime,
        TestFlextSettings,
        TestFlextSettingsCoverage,
        TestFlextSettingsSingletonIntegration,
        TestFlextTestsDomains,
        TestFlextTestsFiles,
        TestFlextTestsMatchers,
        TestFlextTypes,
        TestFlextUtilitiesArgs,
        TestFlextUtilitiesConfiguration,
        TestFlextUtilitiesGuards,
        TestFlextUtilitiesMapper,
        TestFlextUtilitiesReliability,
        TestFlextVersion,
        TestFunction,
        TestHandlerDecoratorDiscovery,
        TestHandlersFullCoverage,
        TestHelperConsolidationTransformer,
        TestHelperFactories,
        TestHelperScenarios,
        TestIdempotency,
        TestInfraIntegration,
        TestLibraryIntegration,
        TestLoggingsErrorPaths,
        TestLoggingsStrictReturns,
        TestMigrationValidation,
        TestMixinsFullCoverage,
        TestModels,
        TestModelsBaseFullCoverage,
        TestModule,
        TestPatternsCommands,
        TestPatternsLogging,
        TestPatternsTesting,
        TestPerformanceBenchmarks,
        TestProjectLevelRefactor,
        TestRefactorPolicyMRO,
        TestResultExceptionCarrying,
        TestRuntimeCoverage100,
        TestService,
        TestService100Coverage,
        TestServiceBootstrap,
        TestServiceInternals,
        TestServiceResultProperty,
        TestTypingsFullCoverage,
        TestUtilities,
        TestUtilitiesCollectionCoverage,
        TestUtilitiesCollectionFullCoverage,
        TestUtilitiesConfigurationFullCoverage,
        TestUtilitiesContextFullCoverage,
        TestUtilitiesCoverage,
        TestUtilitiesDataMapper,
        TestUtilitiesDomainFullCoverage,
        TestUtilitiesGeneratorsFullCoverage,
        TestUtilitiesParserFullCoverage,
        TestUtilitiesTextFullCoverage,
        TestUtilitiesTypeGuardsCoverage100,
        TestUtils,
        TestWorkspaceLevelRefactor,
        Teste,
        Testr,
        TestrCoverage,
        TestsCore,
        TestsFlextServiceBase,
        Testu,
        TestuCacheLogger,
        TestuCacheNormalizeComponent,
        TestuDomain,
        TestuMapperAccessors,
        TestuMapperAdvanced,
        TestuMapperBuild,
        TestuMapperConversions,
        TestuMapperExtract,
        TestuMapperUtils,
        TestuTypeChecker,
        TextUtilityContract,
        UserFactory,
        UtilitiesCacheCoverage100Namespace,
        UtilitiesMapperCoverage100Namespace,
        UtilitiesMapperFullCoverageNamespace,
        ValidatingService,
        ValidatingServiceAuto,
        ValidatingServiceAutoFactory,
        ValidatingServiceFactory,
        ValidationScenario,
        ValidationScenarios,
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
        assert_rejects,
        assert_validates,
        assertion_helpers,
        base,
        benchmark,
        c,
        clean_container,
        conftest,
        conftest_infra,
        constants,
        contracts,
        create_compare_entities_cases,
        create_compare_value_objects_cases,
        create_hash_entity_cases,
        create_hash_value_object_cases,
        d,
        e,
        empty_strings,
        factories,
        factories_impl,
        fixture_factory,
        flext_result_failure,
        flext_result_success,
        flext_tests,
        generators_module,
        get_memory_usage,
        h,
        helper,
        helpers,
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
        inject,
        integration,
        invalid_hostnames,
        invalid_port_numbers,
        invalid_uris,
        m,
        mapper,
        mock_external_service,
        models,
        out_of_range,
        p,
        parser_scenarios,
        patterns,
        protocols,
        pytestmark,
        r,
        reliability_scenarios,
        reset_all_factories,
        reset_global_container,
        reset_runtime_state,
        runtime_module,
        s,
        sample_data,
        scenarios,
        t,
        temp_dir,
        temp_directory,
        temp_file,
        test_advanced_patterns,
        test_aliases_are_available,
        test_architectural_patterns,
        test_architecture,
        test_args_coverage_100,
        test_async_log_writer_paths,
        test_async_log_writer_shutdown_with_full_queue,
        test_bad_string_and_bad_bool_raise_value_error,
        test_base,
        test_build_apply_transform_and_process_error_paths,
        test_canonical_aliases_are_available,
        test_centralize_pydantic_cli_outputs_extended_metrics,
        test_chk_exercises_missed_branches,
        test_circuit_breaker_half_open_and_rate_limiter_accessors,
        test_circuit_breaker_transitions_and_metrics,
        test_class_nesting_appends_to_existing_namespace_and_removes_pass,
        test_class_nesting_keeps_unmapped_top_level_classes,
        test_class_nesting_moves_top_level_class_into_new_namespace,
        test_class_nesting_refactor_single_file_end_to_end,
        test_clear_keys_values_items_and_validate_branches,
        test_collection_utilities_coverage_100,
        test_collections_coverage_100,
        test_command_pagination_limit,
        test_config,
        test_config_bridge_and_trace_context_and_http_validation,
        test_config_integration,
        test_configuration_mapping_and_dict_negative_branches,
        test_configure_structlog_edge_paths,
        test_configure_structlog_print_logger_factory_fallback,
        test_constants_new,
        test_container,
        test_container_and_service_domain_paths,
        test_container_full_coverage,
        test_container_memory,
        test_container_performance,
        test_context,
        test_context_coverage_100,
        test_context_data_metadata_normalizer_removed,
        test_context_data_normalize_and_json_checks,
        test_context_data_validate_dict_serializable_error_paths,
        test_context_data_validate_dict_serializable_none_and_mapping,
        test_context_data_validate_dict_serializable_real_dicts,
        test_context_export_serializable_and_validators,
        test_context_export_statistics_validator_and_computed_fields,
        test_context_export_validate_dict_serializable_mapping_and_models,
        test_context_export_validate_dict_serializable_valid,
        test_context_full_coverage,
        test_conversion_add_converted_and_error_metadata_append_paths,
        test_conversion_add_skipped_skip_reason_upsert_paths,
        test_conversion_add_warning_metadata_append_paths,
        test_conversion_start_and_complete_methods,
        test_convert_default_fallback_matrix,
        test_convert_sequence_branch_returns_tuple,
        test_coverage_context,
        test_coverage_exceptions,
        test_coverage_loggings,
        test_coverage_models,
        test_coverage_utilities,
        test_cqrs,
        test_cqrs_query_resolve_deeper_and_int_pagination,
        test_create_auto_discover_and_mode_mapping,
        test_create_from_callable_and_repr,
        test_create_merges_metadata_dict_branch,
        test_create_overloads_and_auto_correlation,
        test_data_factory,
        test_decorators,
        test_decorators_discovery_full_coverage,
        test_decorators_family_blocks_dispatcher_target,
        test_decorators_full_coverage,
        test_dependency_integration_and_wiring_paths,
        test_dependency_registration_duplicate_guards,
        test_deprecation_warnings,
        test_di_incremental,
        test_di_services_access,
        test_discover_project_roots_without_nested_git_dirs,
        test_dispatcher_di,
        test_dispatcher_family_blocks_models_target,
        test_dispatcher_full_coverage,
        test_dispatcher_minimal,
        test_dispatcher_reliability,
        test_dispatcher_timeout_coverage_100,
        test_docker,
        test_documented_patterns,
        test_domains,
        test_ensure_trace_context_dict_conversion_paths,
        test_entity,
        test_entity_comparable_map_and_bulk_validation_paths,
        test_entity_coverage,
        test_enum_utilities_coverage_100,
        test_errors,
        test_exception_params,
        test_exceptions,
        test_execute_and_register_handler_failure_paths,
        test_export_paths_with_metadata_and_statistics,
        test_extract_array_index_helpers,
        test_extract_error_paths_and_prop_accessor,
        test_extract_field_value_and_ensure_variants,
        test_files,
        test_filter_map_normalize_convert_helpers,
        test_flext_message_type_alias_adapter,
        test_flow_through_short_circuits_on_failure,
        test_from_validation_and_to_model_paths,
        test_general_value_helpers_and_logger,
        test_get_logger_none_name_paths,
        test_get_plugin_and_register_metadata_and_list_items_exception,
        test_group_sort_unique_slice_chunk_branches,
        test_guard_in_has_empty_none_helpers,
        test_guard_instance_attribute_access_warnings,
        test_guards,
        test_guards_bool_identity_branch_via_isinstance_fallback,
        test_guards_bool_shortcut_and_issubclass_typeerror,
        test_guards_handler_type_issubclass_typeerror_branch_direct,
        test_guards_issubclass_success_when_callable_is_patched,
        test_guards_issubclass_typeerror_when_class_not_treated_as_callable,
        test_handler_builder_fluent_methods,
        test_handler_decorator_discovery,
        test_handlers,
        test_handlers_full_coverage,
        test_helper_consolidation_is_prechecked,
        test_inactive_and_none_value_paths,
        test_infra_integration,
        test_init_fallback_and_lazy_returns_result_property,
        test_integration,
        test_is_container_negative_paths_and_callable,
        test_is_type_non_empty_unknown_and_tuple_and_fallback,
        test_is_type_protocol_fallback_branches,
        test_is_valid_handles_validation_exception,
        test_lash_runtime_result_paths,
        test_loggings_error_paths_coverage,
        test_loggings_full_coverage,
        test_loggings_strict_returns,
        test_map_error_identity_and_transform,
        test_map_flat_map_and_then_paths,
        test_mapper,
        test_matchers,
        test_migrate_protocols_rewrites_references_with_p_alias,
        test_migrate_to_mro_inlines_alias_constant_into_constants_class,
        test_migrate_to_mro_moves_constant_and_rewrites_reference,
        test_migrate_to_mro_moves_manual_uppercase_assignment,
        test_migrate_to_mro_normalizes_facade_alias_to_c,
        test_migrate_to_mro_rejects_unknown_target,
        test_migrate_typings_rewrites_references_with_t_alias,
        test_migration_validation,
        test_mixins,
        test_mixins_full_coverage,
        test_model_helpers_remaining_paths,
        test_model_support_and_hash_compare_paths,
        test_models,
        test_models_base_full_coverage,
        test_models_container,
        test_models_context_full_coverage,
        test_models_cqrs_full_coverage,
        test_models_entity_full_coverage,
        test_models_family_blocks_utilities_target,
        test_models_generic_full_coverage,
        test_namespace_enforce_cli_fails_on_manual_protocol_violation,
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
        test_namespace_validator,
        test_narrow_contextvar_exception_branch,
        test_narrow_contextvar_invalid_inputs,
        test_narrow_to_string_keyed_dict_and_mapping_paths,
        test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage,
        test_nested_class_propagation_updates_import_annotations_and_calls,
        test_non_empty_and_normalize_branches,
        test_normalization_edge_branches,
        test_normalize_to_container_alias_removal_path,
        test_normalize_to_metadata_alias_removal_path,
        test_ok_accepts_none,
        test_operation_progress_start_operation_sets_runtime_fields,
        test_patterns_commands,
        test_patterns_logging,
        test_patterns_testing,
        test_protocol_and_simple_guard_helpers,
        test_protocols_new,
        test_query_resolve_pagination_wrapper_and_fallback,
        test_query_validate_pagination_dict_and_default,
        test_rate_limiter_blocks_then_recovers,
        test_rate_limiter_jitter_application,
        test_reconfigure_and_reset_state_paths,
        test_recover_tap_and_tap_error_paths,
        test_refactor_cli_models_workflow,
        test_refactor_migrate_to_class_mro,
        test_refactor_namespace_enforcer,
        test_refactor_nesting_file,
        test_refactor_nesting_idempotency,
        test_refactor_nesting_performance,
        test_refactor_nesting_project,
        test_refactor_nesting_workspace,
        test_refactor_policy_family_rules,
        test_refactor_policy_mro,
        test_refactor_utilities_iter_python_files_includes_examples_and_scripts,
        test_registry,
        test_registry_full_coverage,
        test_result,
        test_result_additional,
        test_result_coverage_100,
        test_result_exception_carrying,
        test_result_full_coverage,
        test_result_property_raises_on_failure,
        test_retry_policy_behavior,
        test_reuse_existing_runtime_coverage_branches,
        test_reuse_existing_runtime_scenarios,
        test_runtime,
        test_runtime_coverage_100,
        test_runtime_create_instance_failure_branch,
        test_runtime_family_blocks_non_runtime_target,
        test_runtime_full_coverage,
        test_runtime_integration_tracking_paths,
        test_runtime_misc_remaining_paths,
        test_runtime_module_accessors_and_metadata,
        test_runtime_result_alias_compatibility,
        test_runtime_result_all_missed_branches,
        test_runtime_result_remaining_paths,
        test_scope_data_validators_and_errors,
        test_service,
        test_service_additional,
        test_service_bootstrap,
        test_service_coverage_100,
        test_service_result_property,
        test_set_set_all_get_validation_and_error_paths,
        test_settings_coverage,
        test_statistics_and_custom_fields_validators,
        test_structlog_proxy_context_var_default_when_key_missing,
        test_structlog_proxy_context_var_get_set_reset_paths,
        test_summary_error_paths_and_bindings_failures,
        test_summary_properties_and_subclass_storage_reset,
        test_system,
        test_take_and_as_branches,
        test_to_general_value_dict_removed,
        test_transform_and_deep_eq_branches,
        test_transform_option_extract_and_step_helpers,
        test_transformer_class_nesting,
        test_transformer_helper_consolidation,
        test_transformer_nested_class_propagation,
        test_type_guards_and_narrowing_failures,
        test_type_guards_result,
        test_typings_full_coverage,
        test_typings_new,
        test_ultrawork_models_cli_runs_dry_run_copy,
        test_update_statistics_remove_hook_and_clone_false_result,
        test_utilities,
        test_utilities_cache_coverage_100,
        test_utilities_collection_coverage_100,
        test_utilities_collection_full_coverage,
        test_utilities_configuration_coverage_100,
        test_utilities_configuration_full_coverage,
        test_utilities_context_full_coverage,
        test_utilities_coverage,
        test_utilities_data_mapper,
        test_utilities_domain,
        test_utilities_domain_full_coverage,
        test_utilities_enum_full_coverage,
        test_utilities_family_allows_utilities_target,
        test_utilities_generators_full_coverage,
        test_utilities_guards_full_coverage,
        test_utilities_mapper_coverage_100,
        test_utilities_mapper_full_coverage,
        test_utilities_parser_full_coverage,
        test_utilities_reliability,
        test_utilities_text_full_coverage,
        test_utilities_type_checker_coverage_100,
        test_utilities_type_guards_coverage_100,
        test_utils,
        test_validation_like_error_structure,
        test_version,
        test_with_resource_cleanup_runs,
        text_contract,
        typings,
        u,
        unit,
        utilities,
        valid_hostnames,
        valid_port_numbers,
        valid_ranges,
        valid_strings,
        valid_uris,
        validation_scenarios,
        whitespace_strings,
        x,
    )
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "tests.benchmark",
        "tests.helpers",
        "tests.integration",
        "tests.unit",
    ),
    {
        "DEFAULT_TIMEOUT": "tests.fixtures.namespace_validator.rule1_loose_constant",
        "FlextCoreTestConstants": "tests.constants",
        "FlextCoreTestModels": "tests.models",
        "FlextCoreTestProtocols": "tests.protocols",
        "FlextCoreTestTypes": "tests.typings",
        "FlextCoreTestUtilities": "tests.utilities",
        "FlextTestConstants": "tests.fixtures.namespace_validator.rule0_multiple_classes",
        "FlextTestModels": "tests.fixtures.namespace_validator.rule1_loose_enum",
        "FlextTestResult": "tests.test_utils",
        "FlextTestResultCo": "tests.test_utils",
        "FlextTestTypes": "tests.fixtures.namespace_validator.rule2_protocol_in_types",
        "FlextTestUtilities": "tests.fixtures.namespace_validator.rule1_magic_number",
        "FunctionalExternalService": "tests.conftest",
        "LooseTypeAlias": "tests.fixtures.namespace_validator.typings",
        "MAX_RETRIES": "tests.fixtures.namespace_validator.rule1_loose_constant",
        "MAX_VALUE": "tests.fixtures.namespace_validator.rule0_no_class",
        "RandomConstants": "tests.fixtures.namespace_validator.rule0_wrong_prefix",
        "Rule0LooseItemsFixture": "tests.fixtures.namespace_validator.rule0_loose_items",
        "Rule0MultipleClassesFixture": "tests.fixtures.namespace_validator.rule0_multiple_classes",
        "Rule1LooseEnumFixture": "tests.fixtures.namespace_validator.rule1_loose_enum",
        "Status": "tests.fixtures.namespace_validator.rule1_loose_enum",
        "T": "tests.typings",
        "T_co": "tests.typings",
        "T_contra": "tests.typings",
        "TestDocumentedPatterns": "tests.test_documented_patterns",
        "TestServiceResultProperty": "tests.test_service_result_property",
        "TestUtils": "tests.test_utils",
        "TestsFlextServiceBase": "tests.base",
        "assert_rejects": "tests.conftest",
        "assert_validates": "tests.conftest",
        "assertion_helpers": "tests.test_utils",
        "base": "tests.base",
        "benchmark": "tests.benchmark",
        "c": ("tests.constants", "FlextCoreTestConstants"),
        "clean_container": "tests.conftest",
        "conftest": "tests.conftest",
        "constants": "tests.constants",
        "d": ("flext_core.decorators", "FlextDecorators"),
        "e": ("flext_core.exceptions", "FlextExceptions"),
        "empty_strings": "tests.conftest",
        "fixture_factory": "tests.test_utils",
        "flext_result_failure": "tests.conftest",
        "flext_result_success": "tests.conftest",
        "h": ("flext_core.handlers", "FlextHandlers"),
        "helper": "tests.fixtures.namespace_validator.rule0_no_class",
        "helpers": "tests.helpers",
        "integration": "tests.integration",
        "invalid_hostnames": "tests.conftest",
        "invalid_port_numbers": "tests.conftest",
        "invalid_uris": "tests.conftest",
        "m": ("tests.models", "FlextCoreTestModels"),
        "mock_external_service": "tests.conftest",
        "models": "tests.models",
        "out_of_range": "tests.conftest",
        "p": ("tests.protocols", "FlextCoreTestProtocols"),
        "parser_scenarios": "tests.conftest",
        "protocols": "tests.protocols",
        "r": ("flext_core.result", "FlextResult"),
        "reliability_scenarios": "tests.conftest",
        "reset_global_container": "tests.conftest",
        "s": ("flext_core.service", "FlextService"),
        "sample_data": "tests.conftest",
        "t": ("tests.typings", "FlextCoreTestTypes"),
        "temp_dir": "tests.conftest",
        "temp_directory": "tests.conftest",
        "temp_file": "tests.conftest",
        "test_context": "tests.conftest",
        "test_data_factory": "tests.test_utils",
        "test_documented_patterns": "tests.test_documented_patterns",
        "test_service_result_property": "tests.test_service_result_property",
        "test_utils": "tests.test_utils",
        "typings": "tests.typings",
        "u": ("tests.utilities", "FlextCoreTestUtilities"),
        "unit": "tests.unit",
        "utilities": "tests.utilities",
        "valid_hostnames": "tests.conftest",
        "valid_port_numbers": "tests.conftest",
        "valid_ranges": "tests.conftest",
        "valid_strings": "tests.conftest",
        "valid_uris": "tests.conftest",
        "validation_scenarios": "tests.conftest",
        "whitespace_strings": "tests.conftest",
        "x": ("flext_core.mixins", "FlextMixins"),
    },
)

__all__ = [
    "DEFAULT_TIMEOUT",
    "EXPECTED_BULK_SIZE",
    "MAX_RETRIES",
    "MAX_VALUE",
    "NORMALIZE_COMPONENT_SCENARIOS",
    "AttrObject",
    "BadBool",
    "BadConfigForTest",
    "BadMapping",
    "BadString",
    "CacheTestModel",
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
    "FlextTestConstants",
    "FlextTestModels",
    "FlextTestResult",
    "FlextTestResultCo",
    "FlextTestTypes",
    "FlextTestUtilities",
    "FlextUnitTestProtocols",
    "FunctionalExternalService",
    "GenericModelFactory",
    "GetUserService",
    "GetUserServiceAuto",
    "GetUserServiceAutoFactory",
    "GetUserServiceFactory",
    "InputPayloadMap",
    "InvalidModelForTest",
    "LooseTypeAlias",
    "NestedModel",
    "NormalizeComponentScenario",
    "P",
    "ParserScenario",
    "ParserScenarios",
    "R",
    "RandomConstants",
    "ReliabilityScenario",
    "ReliabilityScenarios",
    "Rule0LooseItemsFixture",
    "Rule0MultipleClassesFixture",
    "Rule1LooseEnumFixture",
    "RuntimeCloneService",
    "SampleModel",
    "ServiceFactoryRegistry",
    "ServiceTestCaseFactory",
    "ServiceTestCases",
    "SimpleObj",
    "SingletonClassForTest",
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
    "TestFlextConstants",
    "TestFlextContainer",
    "TestFlextContext",
    "TestFlextDecorators",
    "TestFlextHandlers",
    "TestFlextInfraNamespaceValidator",
    "TestFlextMixinsNestedClasses",
    "TestFlextModelsBase",
    "TestFlextModelsCollectionsCoverage100",
    "TestFlextModelsContainer",
    "TestFlextModelsCqrs",
    "TestFlextModelsEntity",
    "TestFlextModelsErrors",
    "TestFlextModelsExceptionParams",
    "TestFlextProtocols",
    "TestFlextRegistry",
    "TestFlextRuntime",
    "TestFlextSettings",
    "TestFlextSettingsCoverage",
    "TestFlextSettingsSingletonIntegration",
    "TestFlextTestsDomains",
    "TestFlextTestsFiles",
    "TestFlextTestsMatchers",
    "TestFlextTypes",
    "TestFlextUtilitiesArgs",
    "TestFlextUtilitiesConfiguration",
    "TestFlextUtilitiesGuards",
    "TestFlextUtilitiesMapper",
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
    "TestUtils",
    "TestWorkspaceLevelRefactor",
    "Teste",
    "Testr",
    "TestrCoverage",
    "TestsCore",
    "TestsFlextServiceBase",
    "Testu",
    "TestuCacheLogger",
    "TestuCacheNormalizeComponent",
    "TestuDomain",
    "TestuMapperAccessors",
    "TestuMapperAdvanced",
    "TestuMapperBuild",
    "TestuMapperConversions",
    "TestuMapperExtract",
    "TestuMapperUtils",
    "TestuTypeChecker",
    "TextUtilityContract",
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
    "base",
    "benchmark",
    "c",
    "clean_container",
    "conftest",
    "conftest_infra",
    "constants",
    "contracts",
    "create_compare_entities_cases",
    "create_compare_value_objects_cases",
    "create_hash_entity_cases",
    "create_hash_value_object_cases",
    "d",
    "e",
    "empty_strings",
    "factories",
    "factories_impl",
    "fixture_factory",
    "flext_result_failure",
    "flext_result_success",
    "flext_tests",
    "generators_module",
    "get_memory_usage",
    "h",
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
    "models",
    "out_of_range",
    "p",
    "parser_scenarios",
    "patterns",
    "protocols",
    "pytestmark",
    "r",
    "reliability_scenarios",
    "reset_all_factories",
    "reset_global_container",
    "reset_runtime_state",
    "runtime_module",
    "s",
    "sample_data",
    "scenarios",
    "t",
    "temp_dir",
    "temp_directory",
    "temp_file",
    "test_advanced_patterns",
    "test_aliases_are_available",
    "test_architectural_patterns",
    "test_architecture",
    "test_args_coverage_100",
    "test_async_log_writer_paths",
    "test_async_log_writer_shutdown_with_full_queue",
    "test_bad_string_and_bad_bool_raise_value_error",
    "test_base",
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
    "test_collection_utilities_coverage_100",
    "test_collections_coverage_100",
    "test_command_pagination_limit",
    "test_config",
    "test_config_bridge_and_trace_context_and_http_validation",
    "test_config_integration",
    "test_configuration_mapping_and_dict_negative_branches",
    "test_configure_structlog_edge_paths",
    "test_configure_structlog_print_logger_factory_fallback",
    "test_constants_new",
    "test_container",
    "test_container_and_service_domain_paths",
    "test_container_full_coverage",
    "test_container_memory",
    "test_container_performance",
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
    "test_cqrs",
    "test_cqrs_query_resolve_deeper_and_int_pagination",
    "test_create_auto_discover_and_mode_mapping",
    "test_create_from_callable_and_repr",
    "test_create_merges_metadata_dict_branch",
    "test_create_overloads_and_auto_correlation",
    "test_data_factory",
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
    "test_documented_patterns",
    "test_domains",
    "test_ensure_trace_context_dict_conversion_paths",
    "test_entity",
    "test_entity_comparable_map_and_bulk_validation_paths",
    "test_entity_coverage",
    "test_enum_utilities_coverage_100",
    "test_errors",
    "test_exception_params",
    "test_exceptions",
    "test_execute_and_register_handler_failure_paths",
    "test_export_paths_with_metadata_and_statistics",
    "test_extract_array_index_helpers",
    "test_extract_error_paths_and_prop_accessor",
    "test_extract_field_value_and_ensure_variants",
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
    "test_guards",
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
    "test_infra_integration",
    "test_init_fallback_and_lazy_returns_result_property",
    "test_integration",
    "test_is_container_negative_paths_and_callable",
    "test_is_type_non_empty_unknown_and_tuple_and_fallback",
    "test_is_type_protocol_fallback_branches",
    "test_is_valid_handles_validation_exception",
    "test_lash_runtime_result_paths",
    "test_loggings_error_paths_coverage",
    "test_loggings_full_coverage",
    "test_loggings_strict_returns",
    "test_map_error_identity_and_transform",
    "test_map_flat_map_and_then_paths",
    "test_mapper",
    "test_matchers",
    "test_migrate_protocols_rewrites_references_with_p_alias",
    "test_migrate_to_mro_inlines_alias_constant_into_constants_class",
    "test_migrate_to_mro_moves_constant_and_rewrites_reference",
    "test_migrate_to_mro_moves_manual_uppercase_assignment",
    "test_migrate_to_mro_normalizes_facade_alias_to_c",
    "test_migrate_to_mro_rejects_unknown_target",
    "test_migrate_typings_rewrites_references_with_t_alias",
    "test_migration_validation",
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
    "test_patterns_commands",
    "test_patterns_logging",
    "test_patterns_testing",
    "test_protocol_and_simple_guard_helpers",
    "test_protocols_new",
    "test_query_resolve_pagination_wrapper_and_fallback",
    "test_query_validate_pagination_dict_and_default",
    "test_rate_limiter_blocks_then_recovers",
    "test_rate_limiter_jitter_application",
    "test_reconfigure_and_reset_state_paths",
    "test_recover_tap_and_tap_error_paths",
    "test_refactor_cli_models_workflow",
    "test_refactor_migrate_to_class_mro",
    "test_refactor_namespace_enforcer",
    "test_refactor_nesting_file",
    "test_refactor_nesting_idempotency",
    "test_refactor_nesting_performance",
    "test_refactor_nesting_project",
    "test_refactor_nesting_workspace",
    "test_refactor_policy_family_rules",
    "test_refactor_policy_mro",
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
    "test_service_result_property",
    "test_set_set_all_get_validation_and_error_paths",
    "test_settings_coverage",
    "test_statistics_and_custom_fields_validators",
    "test_structlog_proxy_context_var_default_when_key_missing",
    "test_structlog_proxy_context_var_get_set_reset_paths",
    "test_summary_error_paths_and_bindings_failures",
    "test_summary_properties_and_subclass_storage_reset",
    "test_system",
    "test_take_and_as_branches",
    "test_to_general_value_dict_removed",
    "test_transform_and_deep_eq_branches",
    "test_transform_option_extract_and_step_helpers",
    "test_transformer_class_nesting",
    "test_transformer_helper_consolidation",
    "test_transformer_nested_class_propagation",
    "test_type_guards_and_narrowing_failures",
    "test_type_guards_result",
    "test_typings_full_coverage",
    "test_typings_new",
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
    "test_utils",
    "test_validation_like_error_structure",
    "test_version",
    "test_with_resource_cleanup_runs",
    "text_contract",
    "typings",
    "u",
    "unit",
    "utilities",
    "valid_hostnames",
    "valid_port_numbers",
    "valid_ranges",
    "valid_strings",
    "valid_uris",
    "validation_scenarios",
    "whitespace_strings",
    "x",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
