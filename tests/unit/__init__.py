# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Unit package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports, merge_lazy_imports

if _t.TYPE_CHECKING:
    import tests.unit._models as _tests_unit__models

    _models = _tests_unit__models
    import tests.unit._models_impl as _tests_unit__models_impl
    from tests.unit._models import (
        TestFlextModelsBase,
        TestFlextModelsCqrs,
        TestFlextModelsEntity,
        TestFlextModelsExceptionParams,
        test_base,
        test_cqrs,
        test_entity,
        test_exception_params,
    )

    _models_impl = _tests_unit__models_impl
    import tests.unit._utilities as _tests_unit__utilities
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

    _utilities = _tests_unit__utilities
    import tests.unit.conftest_infra as _tests_unit_conftest_infra
    from tests.unit._utilities import (
        TestFlextUtilitiesGuards,
        TestFlextUtilitiesMapper,
        test_guards,
        test_mapper,
    )

    conftest_infra = _tests_unit_conftest_infra
    import tests.unit.contracts as _tests_unit_contracts
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

    contracts = _tests_unit_contracts
    import tests.unit.flext_tests as _tests_unit_flext_tests
    from tests.unit.contracts import TextUtilityContract, text_contract

    flext_tests = _tests_unit_flext_tests
    import tests.unit.protocols as _tests_unit_protocols
    from tests.unit.flext_tests import (
        TestDocker,
        TestFlextTestsDomains,
        TestFlextTestsFiles,
        TestFlextTestsMatchers,
        TestUtilities,
        test_docker,
        test_domains,
        test_files,
        test_matchers,
    )

    protocols = _tests_unit_protocols
    import tests.unit.test_args_coverage_100 as _tests_unit_test_args_coverage_100
    from tests.unit.protocols import FlextUnitTestProtocols, FlextUnitTestProtocols as p

    test_args_coverage_100 = _tests_unit_test_args_coverage_100
    import tests.unit.test_beartype_engine as _tests_unit_test_beartype_engine
    from tests.unit.test_args_coverage_100 import TestFlextUtilitiesArgs

    test_beartype_engine = _tests_unit_test_beartype_engine
    import tests.unit.test_collection_utilities_coverage_100 as _tests_unit_test_collection_utilities_coverage_100
    from tests.unit.test_beartype_engine import (
        TestAliasContainsAny,
        TestBeartypeClawCompatibility,
        TestBeartypeConf,
        TestContainsAny,
        TestCountUnionMembers,
        TestFacadeAccessibility,
        TestForbiddenCollectionOrigin,
        TestIsStrNoneUnion,
    )

    test_collection_utilities_coverage_100 = (
        _tests_unit_test_collection_utilities_coverage_100
    )
    import tests.unit.test_collections_coverage_100 as _tests_unit_test_collections_coverage_100
    from tests.unit.test_collection_utilities_coverage_100 import (
        TestCollectionUtilitiesCoverage,
    )

    test_collections_coverage_100 = _tests_unit_test_collections_coverage_100
    import tests.unit.test_config as _tests_unit_test_config
    from tests.unit.test_collections_coverage_100 import (
        TestFlextModelsCollectionsCoverage100,
    )

    test_config = _tests_unit_test_config
    import tests.unit.test_constants_new as _tests_unit_test_constants_new
    from tests.unit.test_config import TestFlextSettings

    test_constants_new = _tests_unit_test_constants_new
    import tests.unit.test_container as _tests_unit_test_container
    from tests.unit.test_constants_new import TestFlextConstants

    test_container = _tests_unit_test_container
    import tests.unit.test_container_full_coverage as _tests_unit_test_container_full_coverage
    from tests.unit.test_container import TestFlextContainer

    test_container_full_coverage = _tests_unit_test_container_full_coverage
    import tests.unit.test_context as _tests_unit_test_context
    from tests.unit.test_container_full_coverage import TestContainerFullCoverage

    test_context = _tests_unit_test_context
    import tests.unit.test_context_coverage_100 as _tests_unit_test_context_coverage_100
    from tests.unit.test_context import TestFlextContext

    test_context_coverage_100 = _tests_unit_test_context_coverage_100
    import tests.unit.test_context_full_coverage as _tests_unit_test_context_full_coverage
    from tests.unit.test_context_coverage_100 import TestContext100Coverage

    test_context_full_coverage = _tests_unit_test_context_full_coverage
    import tests.unit.test_coverage_context as _tests_unit_test_coverage_context
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

    test_coverage_context = _tests_unit_test_coverage_context
    import tests.unit.test_coverage_exceptions as _tests_unit_test_coverage_exceptions
    from tests.unit.test_coverage_context import TestCoverageContext

    test_coverage_exceptions = _tests_unit_test_coverage_exceptions
    import tests.unit.test_coverage_loggings as _tests_unit_test_coverage_loggings
    from tests.unit.test_coverage_exceptions import TestCoverageExceptions

    test_coverage_loggings = _tests_unit_test_coverage_loggings
    import tests.unit.test_coverage_models as _tests_unit_test_coverage_models
    from tests.unit.test_coverage_loggings import TestCoverageLoggings

    test_coverage_models = _tests_unit_test_coverage_models
    import tests.unit.test_coverage_utilities as _tests_unit_test_coverage_utilities
    from tests.unit.test_coverage_models import TestCoverageModels

    test_coverage_utilities = _tests_unit_test_coverage_utilities
    import tests.unit.test_decorators as _tests_unit_test_decorators
    from tests.unit.test_coverage_utilities import Testu

    test_decorators = _tests_unit_test_decorators
    import tests.unit.test_decorators_discovery_full_coverage as _tests_unit_test_decorators_discovery_full_coverage
    from tests.unit.test_decorators import TestFlextDecorators

    test_decorators_discovery_full_coverage = (
        _tests_unit_test_decorators_discovery_full_coverage
    )
    import tests.unit.test_decorators_full_coverage as _tests_unit_test_decorators_full_coverage
    from tests.unit.test_decorators_discovery_full_coverage import (
        TestDecoratorsDiscoveryFullCoverage,
    )

    test_decorators_full_coverage = _tests_unit_test_decorators_full_coverage
    import tests.unit.test_deprecation_warnings as _tests_unit_test_deprecation_warnings
    from tests.unit.test_decorators_full_coverage import TestDecoratorsFullCoverage

    test_deprecation_warnings = _tests_unit_test_deprecation_warnings
    import tests.unit.test_di_incremental as _tests_unit_test_di_incremental
    from tests.unit.test_deprecation_warnings import TestDeprecationWarnings

    test_di_incremental = _tests_unit_test_di_incremental
    import tests.unit.test_di_services_access as _tests_unit_test_di_services_access
    from tests.unit.test_di_incremental import TestDIIncremental, inject

    test_di_services_access = _tests_unit_test_di_services_access
    import tests.unit.test_dispatcher_di as _tests_unit_test_dispatcher_di
    from tests.unit.test_di_services_access import TestDiServicesAccess

    test_dispatcher_di = _tests_unit_test_dispatcher_di
    import tests.unit.test_dispatcher_full_coverage as _tests_unit_test_dispatcher_full_coverage
    from tests.unit.test_dispatcher_di import TestDispatcherDI

    test_dispatcher_full_coverage = _tests_unit_test_dispatcher_full_coverage
    import tests.unit.test_dispatcher_minimal as _tests_unit_test_dispatcher_minimal
    from tests.unit.test_dispatcher_full_coverage import TestDispatcherFullCoverage

    test_dispatcher_minimal = _tests_unit_test_dispatcher_minimal
    import tests.unit.test_dispatcher_reliability as _tests_unit_test_dispatcher_reliability
    from tests.unit.test_dispatcher_minimal import TestDispatcherMinimal

    test_dispatcher_reliability = _tests_unit_test_dispatcher_reliability
    import tests.unit.test_dispatcher_timeout_coverage_100 as _tests_unit_test_dispatcher_timeout_coverage_100
    from tests.unit.test_dispatcher_reliability import (
        test_circuit_breaker_half_open_and_rate_limiter_accessors,
        test_circuit_breaker_transitions_and_metrics,
        test_rate_limiter_blocks_then_recovers,
        test_rate_limiter_jitter_application,
    )

    test_dispatcher_timeout_coverage_100 = (
        _tests_unit_test_dispatcher_timeout_coverage_100
    )
    import tests.unit.test_enforcement as _tests_unit_test_enforcement
    from tests.unit.test_dispatcher_timeout_coverage_100 import (
        TestDispatcherTimeoutCoverage100,
    )

    test_enforcement = _tests_unit_test_enforcement
    import tests.unit.test_entity_coverage as _tests_unit_test_entity_coverage
    from tests.unit.test_enforcement import (
        TestAllLayerIntegration,
        TestBaseModelCoverage,
        TestCheckExtraPolicy,
        TestCheckFieldDescriptions,
        TestCheckNoAny,
        TestCheckNoBareCollections,
        TestCheckNoV1Patterns,
        TestConstantsEnforcement,
        TestEnforcementMode,
        TestExemptions,
        TestNamespacePrefixDerivation,
        TestProtocolsEnforcement,
        TestTypesEnforcement,
        TestUtilitiesEnforcement,
    )

    test_entity_coverage = _tests_unit_test_entity_coverage
    import tests.unit.test_enum_utilities_coverage_100 as _tests_unit_test_enum_utilities_coverage_100
    from tests.unit.test_entity_coverage import TestEntityCoverageEdgeCases

    test_enum_utilities_coverage_100 = _tests_unit_test_enum_utilities_coverage_100
    import tests.unit.test_exceptions as _tests_unit_test_exceptions
    from tests.unit.test_enum_utilities_coverage_100 import TestEnumUtilitiesCoverage

    test_exceptions = _tests_unit_test_exceptions
    import tests.unit.test_handler_decorator_discovery as _tests_unit_test_handler_decorator_discovery
    from tests.unit.test_exceptions import Teste, TestExceptionsHypothesis

    test_handler_decorator_discovery = _tests_unit_test_handler_decorator_discovery
    import tests.unit.test_handlers as _tests_unit_test_handlers
    from tests.unit.test_handler_decorator_discovery import (
        TestHandlerDecoratorDiscovery,
    )

    test_handlers = _tests_unit_test_handlers
    import tests.unit.test_handlers_full_coverage as _tests_unit_test_handlers_full_coverage
    from tests.unit.test_handlers import TestFlextHandlers

    test_handlers_full_coverage = _tests_unit_test_handlers_full_coverage
    import tests.unit.test_loggings_error_paths_coverage as _tests_unit_test_loggings_error_paths_coverage
    from tests.unit.test_handlers_full_coverage import TestHandlersFullCoverage

    test_loggings_error_paths_coverage = _tests_unit_test_loggings_error_paths_coverage
    import tests.unit.test_loggings_full_coverage as _tests_unit_test_loggings_full_coverage
    from tests.unit.test_loggings_error_paths_coverage import TestLoggingsErrorPaths

    test_loggings_full_coverage = _tests_unit_test_loggings_full_coverage
    import tests.unit.test_loggings_strict_returns as _tests_unit_test_loggings_strict_returns
    from tests.unit.test_loggings_full_coverage import TestModule

    test_loggings_strict_returns = _tests_unit_test_loggings_strict_returns
    import tests.unit.test_mixins as _tests_unit_test_mixins
    from tests.unit.test_loggings_strict_returns import TestLoggingsStrictReturns

    test_mixins = _tests_unit_test_mixins
    import tests.unit.test_mixins_full_coverage as _tests_unit_test_mixins_full_coverage
    from tests.unit.test_mixins import TestFlextMixinsNestedClasses

    test_mixins_full_coverage = _tests_unit_test_mixins_full_coverage
    import tests.unit.test_models as _tests_unit_test_models
    from tests.unit.test_mixins_full_coverage import TestMixinsFullCoverage

    test_models = _tests_unit_test_models
    import tests.unit.test_models_base_full_coverage as _tests_unit_test_models_base_full_coverage
    from tests.unit.test_models import TestModels

    test_models_base_full_coverage = _tests_unit_test_models_base_full_coverage
    import tests.unit.test_models_container as _tests_unit_test_models_container
    from tests.unit.test_models_base_full_coverage import TestModelsBaseFullCoverage

    test_models_container = _tests_unit_test_models_container
    import tests.unit.test_models_context_full_coverage as _tests_unit_test_models_context_full_coverage
    from tests.unit.test_models_container import TestFlextModelsContainer

    test_models_context_full_coverage = _tests_unit_test_models_context_full_coverage
    import tests.unit.test_models_cqrs_full_coverage as _tests_unit_test_models_cqrs_full_coverage
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

    test_models_cqrs_full_coverage = _tests_unit_test_models_cqrs_full_coverage
    import tests.unit.test_models_entity_full_coverage as _tests_unit_test_models_entity_full_coverage
    from tests.unit.test_models_cqrs_full_coverage import (
        test_command_pagination_limit,
        test_cqrs_query_resolve_deeper_and_int_pagination,
        test_flext_message_type_alias_adapter,
        test_handler_builder_fluent_methods,
        test_query_resolve_pagination_wrapper_and_fallback,
        test_query_validate_pagination_dict_and_default,
    )

    test_models_entity_full_coverage = _tests_unit_test_models_entity_full_coverage
    import tests.unit.test_models_generic_full_coverage as _tests_unit_test_models_generic_full_coverage
    from tests.unit.test_models_entity_full_coverage import (
        test_entity_comparable_map_and_bulk_validation_paths,
    )

    test_models_generic_full_coverage = _tests_unit_test_models_generic_full_coverage
    import tests.unit.test_namespace_validator as _tests_unit_test_namespace_validator
    from tests.unit.test_models_generic_full_coverage import (
        test_canonical_aliases_are_available,
        test_conversion_add_converted_and_error_metadata_append_paths,
        test_conversion_add_skipped_skip_reason_upsert_paths,
        test_conversion_add_warning_metadata_append_paths,
        test_conversion_start_and_complete_methods,
        test_operation_progress_start_operation_sets_runtime_fields,
    )

    test_namespace_validator = _tests_unit_test_namespace_validator
    import tests.unit.test_protocols_new as _tests_unit_test_protocols_new
    from tests.unit.test_namespace_validator import TestFlextInfraNamespaceValidator

    test_protocols_new = _tests_unit_test_protocols_new
    import tests.unit.test_refactor_cli_models_workflow as _tests_unit_test_refactor_cli_models_workflow
    from tests.unit.test_protocols_new import TestFlextProtocols

    test_refactor_cli_models_workflow = _tests_unit_test_refactor_cli_models_workflow
    import tests.unit.test_refactor_migrate_to_class_mro as _tests_unit_test_refactor_migrate_to_class_mro
    from tests.unit.test_refactor_cli_models_workflow import (
        test_centralize_pydantic_cli_outputs_extended_metrics,
        test_namespace_enforce_cli_fails_on_manual_protocol_violation,
        test_ultrawork_models_cli_runs_dry_run_copy,
    )

    test_refactor_migrate_to_class_mro = _tests_unit_test_refactor_migrate_to_class_mro
    import tests.unit.test_refactor_namespace_enforcer as _tests_unit_test_refactor_namespace_enforcer
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

    test_refactor_namespace_enforcer = _tests_unit_test_refactor_namespace_enforcer
    import tests.unit.test_refactor_policy_family_rules as _tests_unit_test_refactor_policy_family_rules
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

    test_refactor_policy_family_rules = _tests_unit_test_refactor_policy_family_rules
    import tests.unit.test_registry as _tests_unit_test_registry
    from tests.unit.test_refactor_policy_family_rules import (
        test_decorators_family_blocks_dispatcher_target,
        test_dispatcher_family_blocks_models_target,
        test_helper_consolidation_is_prechecked,
        test_models_family_blocks_utilities_target,
        test_runtime_family_blocks_non_runtime_target,
        test_utilities_family_allows_utilities_target,
    )

    test_registry = _tests_unit_test_registry
    import tests.unit.test_registry_full_coverage as _tests_unit_test_registry_full_coverage
    from tests.unit.test_registry import TestFlextRegistry

    test_registry_full_coverage = _tests_unit_test_registry_full_coverage
    import tests.unit.test_result as _tests_unit_test_result
    from tests.unit.test_registry_full_coverage import (
        test_create_auto_discover_and_mode_mapping,
        test_execute_and_register_handler_failure_paths,
        test_get_plugin_and_register_metadata_and_list_items_exception,
        test_summary_error_paths_and_bindings_failures,
        test_summary_properties_and_subclass_storage_reset,
    )

    test_result = _tests_unit_test_result
    import tests.unit.test_result_additional as _tests_unit_test_result_additional
    from tests.unit.test_result import Testr

    test_result_additional = _tests_unit_test_result_additional
    import tests.unit.test_result_coverage_100 as _tests_unit_test_result_coverage_100
    from tests.unit.test_result_additional import (
        test_create_from_callable_and_repr,
        test_flow_through_short_circuits_on_failure,
        test_map_error_identity_and_transform,
        test_ok_accepts_none,
        test_with_resource_cleanup_runs,
    )

    test_result_coverage_100 = _tests_unit_test_result_coverage_100
    import tests.unit.test_result_exception_carrying as _tests_unit_test_result_exception_carrying
    from tests.unit.test_result_coverage_100 import TestrCoverage

    test_result_exception_carrying = _tests_unit_test_result_exception_carrying
    import tests.unit.test_result_full_coverage as _tests_unit_test_result_full_coverage
    from tests.unit.test_result_exception_carrying import TestResultExceptionCarrying

    test_result_full_coverage = _tests_unit_test_result_full_coverage
    import tests.unit.test_runtime as _tests_unit_test_runtime
    from tests.unit.test_result_full_coverage import (
        test_from_validation_and_to_model_paths,
        test_init_fallback_and_lazy_returns_result_property,
        test_lash_runtime_result_paths,
        test_map_flat_map_and_then_paths,
        test_recover_tap_and_tap_error_paths,
        test_type_guards_result,
        test_validation_like_error_structure,
    )

    test_runtime = _tests_unit_test_runtime
    import tests.unit.test_runtime_coverage_100 as _tests_unit_test_runtime_coverage_100
    from tests.unit.test_runtime import TestFlextRuntime

    test_runtime_coverage_100 = _tests_unit_test_runtime_coverage_100
    import tests.unit.test_runtime_full_coverage as _tests_unit_test_runtime_full_coverage
    from tests.unit.test_runtime_coverage_100 import TestRuntimeCoverage100

    test_runtime_full_coverage = _tests_unit_test_runtime_full_coverage
    import tests.unit.test_service as _tests_unit_test_service
    from tests.unit.test_runtime_full_coverage import (
        reset_runtime_state,
        runtime_module,
        test_async_log_writer_paths,
        test_async_log_writer_shutdown_with_full_queue,
        test_config_bridge_and_trace_context_and_http_validation,
        test_configure_structlog_async_logging_uses_print_logger_factory,
        test_configure_structlog_edge_paths,
        test_configure_structlog_print_logger_factory_fallback,
        test_dependency_integration_and_wiring_paths,
        test_dependency_registration_duplicate_guards,
        test_ensure_trace_context_dict_conversion_paths,
        test_get_logger_auto_configures_structlog,
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

    test_service = _tests_unit_test_service
    import tests.unit.test_service_additional as _tests_unit_test_service_additional
    from tests.unit.test_service import TestsCore, TestServiceInternals

    test_service_additional = _tests_unit_test_service_additional
    import tests.unit.test_service_bootstrap as _tests_unit_test_service_bootstrap
    from tests.unit.test_service_additional import (
        RuntimeCloneService,
        test_is_valid_handles_validation_exception,
        test_result_property_raises_on_failure,
    )

    test_service_bootstrap = _tests_unit_test_service_bootstrap
    import tests.unit.test_service_coverage_100 as _tests_unit_test_service_coverage_100
    from tests.unit.test_service_bootstrap import TestServiceBootstrap

    test_service_coverage_100 = _tests_unit_test_service_coverage_100
    import tests.unit.test_settings_coverage as _tests_unit_test_settings_coverage
    from tests.unit.test_service_coverage_100 import TestService100Coverage

    test_settings_coverage = _tests_unit_test_settings_coverage
    import tests.unit.test_transformer_class_nesting as _tests_unit_test_transformer_class_nesting
    from tests.unit.test_settings_coverage import TestFlextSettingsCoverage

    test_transformer_class_nesting = _tests_unit_test_transformer_class_nesting
    import tests.unit.test_transformer_helper_consolidation as _tests_unit_test_transformer_helper_consolidation
    from tests.unit.test_transformer_class_nesting import (
        test_class_nesting_appends_to_existing_namespace_and_removes_pass,
        test_class_nesting_keeps_unmapped_top_level_classes,
        test_class_nesting_moves_top_level_class_into_new_namespace,
    )

    test_transformer_helper_consolidation = (
        _tests_unit_test_transformer_helper_consolidation
    )
    import tests.unit.test_transformer_nested_class_propagation as _tests_unit_test_transformer_nested_class_propagation
    from tests.unit.test_transformer_helper_consolidation import (
        TestHelperConsolidationTransformer,
    )

    test_transformer_nested_class_propagation = (
        _tests_unit_test_transformer_nested_class_propagation
    )
    import tests.unit.test_typings_full_coverage as _tests_unit_test_typings_full_coverage
    from tests.unit.test_transformer_nested_class_propagation import (
        test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage,
        test_nested_class_propagation_updates_import_annotations_and_calls,
    )

    test_typings_full_coverage = _tests_unit_test_typings_full_coverage
    import tests.unit.test_typings_new as _tests_unit_test_typings_new
    from tests.unit.test_typings_full_coverage import TestTypingsFullCoverage

    test_typings_new = _tests_unit_test_typings_new
    import tests.unit.test_utilities as _tests_unit_test_utilities
    from tests.unit.test_typings_new import TestFlextTypes

    test_utilities = _tests_unit_test_utilities
    import tests.unit.test_utilities_cache_coverage_100 as _tests_unit_test_utilities_cache_coverage_100

    test_utilities_cache_coverage_100 = _tests_unit_test_utilities_cache_coverage_100
    import tests.unit.test_utilities_collection_coverage_100 as _tests_unit_test_utilities_collection_coverage_100
    from tests.unit.test_utilities_cache_coverage_100 import (
        NORMALIZE_COMPONENT_SCENARIOS,
        NormalizeComponentScenario,
        TestuCacheLogger,
        TestuCacheNormalizeComponent,
        UtilitiesCacheCoverage100Namespace,
    )

    test_utilities_collection_coverage_100 = (
        _tests_unit_test_utilities_collection_coverage_100
    )
    import tests.unit.test_utilities_collection_full_coverage as _tests_unit_test_utilities_collection_full_coverage
    from tests.unit.test_utilities_collection_coverage_100 import (
        TestUtilitiesCollectionCoverage,
    )

    test_utilities_collection_full_coverage = (
        _tests_unit_test_utilities_collection_full_coverage
    )
    import tests.unit.test_utilities_configuration_coverage_100 as _tests_unit_test_utilities_configuration_coverage_100
    from tests.unit.test_utilities_collection_full_coverage import (
        TestUtilitiesCollectionFullCoverage,
    )

    test_utilities_configuration_coverage_100 = (
        _tests_unit_test_utilities_configuration_coverage_100
    )
    import tests.unit.test_utilities_configuration_full_coverage as _tests_unit_test_utilities_configuration_full_coverage
    from tests.unit.test_utilities_configuration_coverage_100 import (
        TestFlextUtilitiesConfiguration,
    )

    test_utilities_configuration_full_coverage = (
        _tests_unit_test_utilities_configuration_full_coverage
    )
    import tests.unit.test_utilities_context_full_coverage as _tests_unit_test_utilities_context_full_coverage
    from tests.unit.test_utilities_configuration_full_coverage import (
        TestUtilitiesConfigurationFullCoverage,
    )

    test_utilities_context_full_coverage = (
        _tests_unit_test_utilities_context_full_coverage
    )
    import tests.unit.test_utilities_coverage as _tests_unit_test_utilities_coverage
    from tests.unit.test_utilities_context_full_coverage import (
        TestUtilitiesContextFullCoverage,
    )

    test_utilities_coverage = _tests_unit_test_utilities_coverage
    import tests.unit.test_utilities_data_mapper as _tests_unit_test_utilities_data_mapper
    from tests.unit.test_utilities_coverage import TestUtilitiesCoverage

    test_utilities_data_mapper = _tests_unit_test_utilities_data_mapper
    import tests.unit.test_utilities_domain as _tests_unit_test_utilities_domain
    from tests.unit.test_utilities_data_mapper import TestUtilitiesDataMapper

    test_utilities_domain = _tests_unit_test_utilities_domain
    import tests.unit.test_utilities_domain_full_coverage as _tests_unit_test_utilities_domain_full_coverage
    from tests.unit.test_utilities_domain import (
        TestuDomain,
        create_compare_entities_cases,
        create_compare_value_objects_cases,
        create_hash_entity_cases,
        create_hash_value_object_cases,
    )

    test_utilities_domain_full_coverage = (
        _tests_unit_test_utilities_domain_full_coverage
    )
    import tests.unit.test_utilities_enum_full_coverage as _tests_unit_test_utilities_enum_full_coverage
    from tests.unit.test_utilities_domain_full_coverage import (
        TestUtilitiesDomainFullCoverage,
    )

    test_utilities_enum_full_coverage = _tests_unit_test_utilities_enum_full_coverage
    import tests.unit.test_utilities_generators_full_coverage as _tests_unit_test_utilities_generators_full_coverage

    test_utilities_generators_full_coverage = (
        _tests_unit_test_utilities_generators_full_coverage
    )
    import tests.unit.test_utilities_guards_full_coverage as _tests_unit_test_utilities_guards_full_coverage
    from tests.unit.test_utilities_generators_full_coverage import (
        TestUtilitiesGeneratorsFullCoverage,
    )

    test_utilities_guards_full_coverage = (
        _tests_unit_test_utilities_guards_full_coverage
    )
    import tests.unit.test_utilities_mapper_coverage_100 as _tests_unit_test_utilities_mapper_coverage_100
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

    test_utilities_mapper_coverage_100 = _tests_unit_test_utilities_mapper_coverage_100
    import tests.unit.test_utilities_mapper_full_coverage as _tests_unit_test_utilities_mapper_full_coverage
    from tests.unit.test_utilities_mapper_coverage_100 import (
        SimpleObj,
        TestuMapperAccessors,
        TestuMapperAdvanced,
        TestuMapperConversions,
        TestuMapperExtract,
        TestuMapperUtils,
        UtilitiesMapperCoverage100Namespace,
    )

    test_utilities_mapper_full_coverage = (
        _tests_unit_test_utilities_mapper_full_coverage
    )
    import tests.unit.test_utilities_parser_full_coverage as _tests_unit_test_utilities_parser_full_coverage
    from tests.unit.test_utilities_mapper_full_coverage import (
        AttrObject,
        BadBool,
        BadMapping,
        BadString,
        ExplodingLenList,
        mapper,
        test_bad_string_and_bad_bool_raise_value_error,
        test_extract_array_index_helpers,
        test_extract_error_paths_and_prop_accessor,
    )

    test_utilities_parser_full_coverage = (
        _tests_unit_test_utilities_parser_full_coverage
    )
    import tests.unit.test_utilities_reliability as _tests_unit_test_utilities_reliability
    from tests.unit.test_utilities_parser_full_coverage import (
        TestUtilitiesParserFullCoverage,
    )

    test_utilities_reliability = _tests_unit_test_utilities_reliability
    import tests.unit.test_utilities_text_full_coverage as _tests_unit_test_utilities_text_full_coverage
    from tests.unit.test_utilities_reliability import TestFlextUtilitiesReliability

    test_utilities_text_full_coverage = _tests_unit_test_utilities_text_full_coverage
    import tests.unit.test_utilities_type_checker_coverage_100 as _tests_unit_test_utilities_type_checker_coverage_100
    from tests.unit.test_utilities_text_full_coverage import (
        TestUtilitiesTextFullCoverage,
    )

    test_utilities_type_checker_coverage_100 = (
        _tests_unit_test_utilities_type_checker_coverage_100
    )
    import tests.unit.test_utilities_type_guards_coverage_100 as _tests_unit_test_utilities_type_guards_coverage_100
    from tests.unit.test_utilities_type_checker_coverage_100 import (
        TestuTypeChecker,
        TMessage,
        pytestmark,
    )

    test_utilities_type_guards_coverage_100 = (
        _tests_unit_test_utilities_type_guards_coverage_100
    )
    import tests.unit.test_version as _tests_unit_test_version
    from tests.unit.test_utilities_type_guards_coverage_100 import (
        TestUtilitiesTypeGuardsCoverage100,
    )

    test_version = _tests_unit_test_version
    import tests.unit.typings as _tests_unit_typings
    from tests.unit.test_version import TestFlextVersion

    typings = _tests_unit_typings
    from flext_core.constants import FlextConstants as c
    from flext_core.decorators import FlextDecorators as d
    from flext_core.exceptions import FlextExceptions as e
    from flext_core.handlers import FlextHandlers as h
    from flext_core.mixins import FlextMixins as x
    from flext_core.models import FlextModels as m
    from flext_core.result import FlextResult as r
    from flext_core.service import FlextService as s
    from flext_core.typings import FlextTypes as t
    from flext_core.utilities import FlextUtilities as u
_LAZY_IMPORTS = merge_lazy_imports(
    (
        "tests.unit._models",
        "tests.unit._utilities",
        "tests.unit.contracts",
        "tests.unit.flext_tests",
    ),
    {
        "AttrObject": "tests.unit.test_utilities_mapper_full_coverage",
        "BadBool": "tests.unit.test_utilities_mapper_full_coverage",
        "BadConfigForTest": "tests.unit._models_impl",
        "BadMapping": "tests.unit.test_utilities_mapper_full_coverage",
        "BadString": "tests.unit.test_utilities_mapper_full_coverage",
        "CacheTestModel": "tests.unit._models_impl",
        "ComplexModel": "tests.unit._models_impl",
        "ConfigModelForTest": "tests.unit._models_impl",
        "ExplodingLenList": "tests.unit.test_utilities_mapper_full_coverage",
        "FlextUnitTestProtocols": "tests.unit.protocols",
        "InputPayloadMap": "tests.unit._models_impl",
        "InvalidModelForTest": "tests.unit._models_impl",
        "NORMALIZE_COMPONENT_SCENARIOS": "tests.unit.test_utilities_cache_coverage_100",
        "NestedModel": "tests.unit._models_impl",
        "NormalizeComponentScenario": "tests.unit.test_utilities_cache_coverage_100",
        "RuntimeCloneService": "tests.unit.test_service_additional",
        "SampleModel": "tests.unit._models_impl",
        "SimpleObj": "tests.unit.test_utilities_mapper_coverage_100",
        "SingletonClassForTest": "tests.unit._models_impl",
        "TMessage": "tests.unit.test_utilities_type_checker_coverage_100",
        "TestAliasContainsAny": "tests.unit.test_beartype_engine",
        "TestAllLayerIntegration": "tests.unit.test_enforcement",
        "TestBaseModelCoverage": "tests.unit.test_enforcement",
        "TestBeartypeClawCompatibility": "tests.unit.test_beartype_engine",
        "TestBeartypeConf": "tests.unit.test_beartype_engine",
        "TestCaseMap": "tests.unit._models_impl",
        "TestCheckExtraPolicy": "tests.unit.test_enforcement",
        "TestCheckFieldDescriptions": "tests.unit.test_enforcement",
        "TestCheckNoAny": "tests.unit.test_enforcement",
        "TestCheckNoBareCollections": "tests.unit.test_enforcement",
        "TestCheckNoV1Patterns": "tests.unit.test_enforcement",
        "TestCollectionUtilitiesCoverage": "tests.unit.test_collection_utilities_coverage_100",
        "TestConstantsEnforcement": "tests.unit.test_enforcement",
        "TestContainerFullCoverage": "tests.unit.test_container_full_coverage",
        "TestContainsAny": "tests.unit.test_beartype_engine",
        "TestContext100Coverage": "tests.unit.test_context_coverage_100",
        "TestCountUnionMembers": "tests.unit.test_beartype_engine",
        "TestCoverageContext": "tests.unit.test_coverage_context",
        "TestCoverageExceptions": "tests.unit.test_coverage_exceptions",
        "TestCoverageLoggings": "tests.unit.test_coverage_loggings",
        "TestCoverageModels": "tests.unit.test_coverage_models",
        "TestDIIncremental": "tests.unit.test_di_incremental",
        "TestDecoratorsDiscoveryFullCoverage": "tests.unit.test_decorators_discovery_full_coverage",
        "TestDecoratorsFullCoverage": "tests.unit.test_decorators_full_coverage",
        "TestDeprecationWarnings": "tests.unit.test_deprecation_warnings",
        "TestDiServicesAccess": "tests.unit.test_di_services_access",
        "TestDispatcherDI": "tests.unit.test_dispatcher_di",
        "TestDispatcherFullCoverage": "tests.unit.test_dispatcher_full_coverage",
        "TestDispatcherMinimal": "tests.unit.test_dispatcher_minimal",
        "TestDispatcherTimeoutCoverage100": "tests.unit.test_dispatcher_timeout_coverage_100",
        "TestEnforcementMode": "tests.unit.test_enforcement",
        "TestEntityCoverageEdgeCases": "tests.unit.test_entity_coverage",
        "TestEnumUtilitiesCoverage": "tests.unit.test_enum_utilities_coverage_100",
        "TestExceptionsHypothesis": "tests.unit.test_exceptions",
        "TestExemptions": "tests.unit.test_enforcement",
        "TestFacadeAccessibility": "tests.unit.test_beartype_engine",
        "TestFlextConstants": "tests.unit.test_constants_new",
        "TestFlextContainer": "tests.unit.test_container",
        "TestFlextContext": "tests.unit.test_context",
        "TestFlextDecorators": "tests.unit.test_decorators",
        "TestFlextHandlers": "tests.unit.test_handlers",
        "TestFlextInfraNamespaceValidator": "tests.unit.test_namespace_validator",
        "TestFlextMixinsNestedClasses": "tests.unit.test_mixins",
        "TestFlextModelsCollectionsCoverage100": "tests.unit.test_collections_coverage_100",
        "TestFlextModelsContainer": "tests.unit.test_models_container",
        "TestFlextProtocols": "tests.unit.test_protocols_new",
        "TestFlextRegistry": "tests.unit.test_registry",
        "TestFlextRuntime": "tests.unit.test_runtime",
        "TestFlextSettings": "tests.unit.test_config",
        "TestFlextSettingsCoverage": "tests.unit.test_settings_coverage",
        "TestFlextTypes": "tests.unit.test_typings_new",
        "TestFlextUtilitiesArgs": "tests.unit.test_args_coverage_100",
        "TestFlextUtilitiesConfiguration": "tests.unit.test_utilities_configuration_coverage_100",
        "TestFlextUtilitiesReliability": "tests.unit.test_utilities_reliability",
        "TestFlextVersion": "tests.unit.test_version",
        "TestForbiddenCollectionOrigin": "tests.unit.test_beartype_engine",
        "TestHandlerDecoratorDiscovery": "tests.unit.test_handler_decorator_discovery",
        "TestHandlersFullCoverage": "tests.unit.test_handlers_full_coverage",
        "TestHelperConsolidationTransformer": "tests.unit.test_transformer_helper_consolidation",
        "TestIsStrNoneUnion": "tests.unit.test_beartype_engine",
        "TestLoggingsErrorPaths": "tests.unit.test_loggings_error_paths_coverage",
        "TestLoggingsStrictReturns": "tests.unit.test_loggings_strict_returns",
        "TestMixinsFullCoverage": "tests.unit.test_mixins_full_coverage",
        "TestModels": "tests.unit.test_models",
        "TestModelsBaseFullCoverage": "tests.unit.test_models_base_full_coverage",
        "TestModule": "tests.unit.test_loggings_full_coverage",
        "TestNamespacePrefixDerivation": "tests.unit.test_enforcement",
        "TestProtocolsEnforcement": "tests.unit.test_enforcement",
        "TestResultExceptionCarrying": "tests.unit.test_result_exception_carrying",
        "TestRuntimeCoverage100": "tests.unit.test_runtime_coverage_100",
        "TestService100Coverage": "tests.unit.test_service_coverage_100",
        "TestServiceBootstrap": "tests.unit.test_service_bootstrap",
        "TestServiceInternals": "tests.unit.test_service",
        "TestTypesEnforcement": "tests.unit.test_enforcement",
        "TestTypingsFullCoverage": "tests.unit.test_typings_full_coverage",
        "TestUtilitiesCollectionCoverage": "tests.unit.test_utilities_collection_coverage_100",
        "TestUtilitiesCollectionFullCoverage": "tests.unit.test_utilities_collection_full_coverage",
        "TestUtilitiesConfigurationFullCoverage": "tests.unit.test_utilities_configuration_full_coverage",
        "TestUtilitiesContextFullCoverage": "tests.unit.test_utilities_context_full_coverage",
        "TestUtilitiesCoverage": "tests.unit.test_utilities_coverage",
        "TestUtilitiesDataMapper": "tests.unit.test_utilities_data_mapper",
        "TestUtilitiesDomainFullCoverage": "tests.unit.test_utilities_domain_full_coverage",
        "TestUtilitiesEnforcement": "tests.unit.test_enforcement",
        "TestUtilitiesGeneratorsFullCoverage": "tests.unit.test_utilities_generators_full_coverage",
        "TestUtilitiesParserFullCoverage": "tests.unit.test_utilities_parser_full_coverage",
        "TestUtilitiesTextFullCoverage": "tests.unit.test_utilities_text_full_coverage",
        "TestUtilitiesTypeGuardsCoverage100": "tests.unit.test_utilities_type_guards_coverage_100",
        "Teste": "tests.unit.test_exceptions",
        "Testr": "tests.unit.test_result",
        "TestrCoverage": "tests.unit.test_result_coverage_100",
        "TestsCore": "tests.unit.test_service",
        "Testu": "tests.unit.test_coverage_utilities",
        "TestuCacheLogger": "tests.unit.test_utilities_cache_coverage_100",
        "TestuCacheNormalizeComponent": "tests.unit.test_utilities_cache_coverage_100",
        "TestuDomain": "tests.unit.test_utilities_domain",
        "TestuMapperAccessors": "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperAdvanced": "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperConversions": "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperExtract": "tests.unit.test_utilities_mapper_coverage_100",
        "TestuMapperUtils": "tests.unit.test_utilities_mapper_coverage_100",
        "TestuTypeChecker": "tests.unit.test_utilities_type_checker_coverage_100",
        "UtilitiesCacheCoverage100Namespace": "tests.unit.test_utilities_cache_coverage_100",
        "UtilitiesMapperCoverage100Namespace": "tests.unit.test_utilities_mapper_coverage_100",
        "_BadCopyModel": "tests.unit._models_impl",
        "_BrokenDumpModel": "tests.unit._models_impl",
        "_Cfg": "tests.unit._models_impl",
        "_DumpErrorModel": "tests.unit._models_impl",
        "_ErrorsModel": "tests.unit._models_impl",
        "_FakeConfig": "tests.unit._models_impl",
        "_FrozenEntity": "tests.unit._models_impl",
        "_GoodModel": "tests.unit._models_impl",
        "_Model": "tests.unit._models_impl",
        "_MsgWithCommandId": "tests.unit._models_impl",
        "_MsgWithMessageId": "tests.unit._models_impl",
        "_Opts": "tests.unit._models_impl",
        "_PlainErrorModel": "tests.unit._models_impl",
        "_SampleEntity": "tests.unit._models_impl",
        "_SvcModel": "tests.unit._models_impl",
        "_TargetModel": "tests.unit._models_impl",
        "_ValidationLikeError": "tests.unit._models_impl",
        "_models": "tests.unit._models",
        "_models_impl": "tests.unit._models_impl",
        "_utilities": "tests.unit._utilities",
        "c": ("flext_core.constants", "FlextConstants"),
        "conftest_infra": "tests.unit.conftest_infra",
        "contracts": "tests.unit.contracts",
        "create_compare_entities_cases": "tests.unit.test_utilities_domain",
        "create_compare_value_objects_cases": "tests.unit.test_utilities_domain",
        "create_hash_entity_cases": "tests.unit.test_utilities_domain",
        "create_hash_value_object_cases": "tests.unit.test_utilities_domain",
        "d": ("flext_core.decorators", "FlextDecorators"),
        "e": ("flext_core.exceptions", "FlextExceptions"),
        "flext_tests": "tests.unit.flext_tests",
        "h": ("flext_core.handlers", "FlextHandlers"),
        "infra_git": "tests.unit.conftest_infra",
        "infra_git_repo": "tests.unit.conftest_infra",
        "infra_io": "tests.unit.conftest_infra",
        "infra_path": "tests.unit.conftest_infra",
        "infra_patterns": "tests.unit.conftest_infra",
        "infra_reporting": "tests.unit.conftest_infra",
        "infra_safe_command_output": "tests.unit.conftest_infra",
        "infra_selection": "tests.unit.conftest_infra",
        "infra_subprocess": "tests.unit.conftest_infra",
        "infra_templates": "tests.unit.conftest_infra",
        "infra_test_workspace": "tests.unit.conftest_infra",
        "infra_toml": "tests.unit.conftest_infra",
        "inject": "tests.unit.test_di_incremental",
        "m": ("flext_core.models", "FlextModels"),
        "mapper": "tests.unit.test_utilities_mapper_full_coverage",
        "p": ("tests.unit.protocols", "FlextUnitTestProtocols"),
        "protocols": "tests.unit.protocols",
        "pytestmark": "tests.unit.test_utilities_type_checker_coverage_100",
        "r": ("flext_core.result", "FlextResult"),
        "reset_runtime_state": "tests.unit.test_runtime_full_coverage",
        "runtime_module": "tests.unit.test_runtime_full_coverage",
        "s": ("flext_core.service", "FlextService"),
        "t": ("flext_core.typings", "FlextTypes"),
        "test_aliases_are_available": "tests.unit.test_utilities_guards_full_coverage",
        "test_args_coverage_100": "tests.unit.test_args_coverage_100",
        "test_async_log_writer_paths": "tests.unit.test_runtime_full_coverage",
        "test_async_log_writer_shutdown_with_full_queue": "tests.unit.test_runtime_full_coverage",
        "test_bad_string_and_bad_bool_raise_value_error": "tests.unit.test_utilities_mapper_full_coverage",
        "test_beartype_engine": "tests.unit.test_beartype_engine",
        "test_canonical_aliases_are_available": "tests.unit.test_models_generic_full_coverage",
        "test_centralize_pydantic_cli_outputs_extended_metrics": "tests.unit.test_refactor_cli_models_workflow",
        "test_chk_exercises_missed_branches": "tests.unit.test_utilities_guards_full_coverage",
        "test_circuit_breaker_half_open_and_rate_limiter_accessors": "tests.unit.test_dispatcher_reliability",
        "test_circuit_breaker_transitions_and_metrics": "tests.unit.test_dispatcher_reliability",
        "test_class_nesting_appends_to_existing_namespace_and_removes_pass": "tests.unit.test_transformer_class_nesting",
        "test_class_nesting_keeps_unmapped_top_level_classes": "tests.unit.test_transformer_class_nesting",
        "test_class_nesting_moves_top_level_class_into_new_namespace": "tests.unit.test_transformer_class_nesting",
        "test_clear_keys_values_items_and_validate_branches": "tests.unit.test_context_full_coverage",
        "test_collection_utilities_coverage_100": "tests.unit.test_collection_utilities_coverage_100",
        "test_collections_coverage_100": "tests.unit.test_collections_coverage_100",
        "test_command_pagination_limit": "tests.unit.test_models_cqrs_full_coverage",
        "test_config": "tests.unit.test_config",
        "test_config_bridge_and_trace_context_and_http_validation": "tests.unit.test_runtime_full_coverage",
        "test_configuration_mapping_and_dict_negative_branches": "tests.unit.test_utilities_guards_full_coverage",
        "test_configure_structlog_async_logging_uses_print_logger_factory": "tests.unit.test_runtime_full_coverage",
        "test_configure_structlog_edge_paths": "tests.unit.test_runtime_full_coverage",
        "test_configure_structlog_print_logger_factory_fallback": "tests.unit.test_runtime_full_coverage",
        "test_constants_new": "tests.unit.test_constants_new",
        "test_container": "tests.unit.test_container",
        "test_container_and_service_domain_paths": "tests.unit.test_context_full_coverage",
        "test_container_full_coverage": "tests.unit.test_container_full_coverage",
        "test_context": "tests.unit.test_context",
        "test_context_coverage_100": "tests.unit.test_context_coverage_100",
        "test_context_data_metadata_normalizer_removed": "tests.unit.test_models_context_full_coverage",
        "test_context_data_normalize_and_json_checks": "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_error_paths": "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_none_and_mapping": "tests.unit.test_models_context_full_coverage",
        "test_context_data_validate_dict_serializable_real_dicts": "tests.unit.test_models_context_full_coverage",
        "test_context_export_serializable_and_validators": "tests.unit.test_models_context_full_coverage",
        "test_context_export_statistics_validator_and_computed_fields": "tests.unit.test_models_context_full_coverage",
        "test_context_export_validate_dict_serializable_mapping_and_models": "tests.unit.test_models_context_full_coverage",
        "test_context_export_validate_dict_serializable_valid": "tests.unit.test_models_context_full_coverage",
        "test_context_full_coverage": "tests.unit.test_context_full_coverage",
        "test_conversion_add_converted_and_error_metadata_append_paths": "tests.unit.test_models_generic_full_coverage",
        "test_conversion_add_skipped_skip_reason_upsert_paths": "tests.unit.test_models_generic_full_coverage",
        "test_conversion_add_warning_metadata_append_paths": "tests.unit.test_models_generic_full_coverage",
        "test_conversion_start_and_complete_methods": "tests.unit.test_models_generic_full_coverage",
        "test_coverage_context": "tests.unit.test_coverage_context",
        "test_coverage_exceptions": "tests.unit.test_coverage_exceptions",
        "test_coverage_loggings": "tests.unit.test_coverage_loggings",
        "test_coverage_models": "tests.unit.test_coverage_models",
        "test_coverage_utilities": "tests.unit.test_coverage_utilities",
        "test_cqrs_query_resolve_deeper_and_int_pagination": "tests.unit.test_models_cqrs_full_coverage",
        "test_create_auto_discover_and_mode_mapping": "tests.unit.test_registry_full_coverage",
        "test_create_from_callable_and_repr": "tests.unit.test_result_additional",
        "test_create_merges_metadata_dict_branch": "tests.unit.test_context_full_coverage",
        "test_create_overloads_and_auto_correlation": "tests.unit.test_context_full_coverage",
        "test_decorators": "tests.unit.test_decorators",
        "test_decorators_discovery_full_coverage": "tests.unit.test_decorators_discovery_full_coverage",
        "test_decorators_family_blocks_dispatcher_target": "tests.unit.test_refactor_policy_family_rules",
        "test_decorators_full_coverage": "tests.unit.test_decorators_full_coverage",
        "test_dependency_integration_and_wiring_paths": "tests.unit.test_runtime_full_coverage",
        "test_dependency_registration_duplicate_guards": "tests.unit.test_runtime_full_coverage",
        "test_deprecation_warnings": "tests.unit.test_deprecation_warnings",
        "test_di_incremental": "tests.unit.test_di_incremental",
        "test_di_services_access": "tests.unit.test_di_services_access",
        "test_discover_project_roots_without_nested_git_dirs": "tests.unit.test_refactor_migrate_to_class_mro",
        "test_dispatcher_di": "tests.unit.test_dispatcher_di",
        "test_dispatcher_family_blocks_models_target": "tests.unit.test_refactor_policy_family_rules",
        "test_dispatcher_full_coverage": "tests.unit.test_dispatcher_full_coverage",
        "test_dispatcher_minimal": "tests.unit.test_dispatcher_minimal",
        "test_dispatcher_reliability": "tests.unit.test_dispatcher_reliability",
        "test_dispatcher_timeout_coverage_100": "tests.unit.test_dispatcher_timeout_coverage_100",
        "test_enforcement": "tests.unit.test_enforcement",
        "test_ensure_trace_context_dict_conversion_paths": "tests.unit.test_runtime_full_coverage",
        "test_entity_comparable_map_and_bulk_validation_paths": "tests.unit.test_models_entity_full_coverage",
        "test_entity_coverage": "tests.unit.test_entity_coverage",
        "test_enum_utilities_coverage_100": "tests.unit.test_enum_utilities_coverage_100",
        "test_exceptions": "tests.unit.test_exceptions",
        "test_execute_and_register_handler_failure_paths": "tests.unit.test_registry_full_coverage",
        "test_export_paths_with_metadata_and_statistics": "tests.unit.test_context_full_coverage",
        "test_extract_array_index_helpers": "tests.unit.test_utilities_mapper_full_coverage",
        "test_extract_error_paths_and_prop_accessor": "tests.unit.test_utilities_mapper_full_coverage",
        "test_flext_message_type_alias_adapter": "tests.unit.test_models_cqrs_full_coverage",
        "test_flow_through_short_circuits_on_failure": "tests.unit.test_result_additional",
        "test_from_validation_and_to_model_paths": "tests.unit.test_result_full_coverage",
        "test_get_logger_auto_configures_structlog": "tests.unit.test_runtime_full_coverage",
        "test_get_logger_none_name_paths": "tests.unit.test_runtime_full_coverage",
        "test_get_plugin_and_register_metadata_and_list_items_exception": "tests.unit.test_registry_full_coverage",
        "test_guard_in_has_empty_none_helpers": "tests.unit.test_utilities_guards_full_coverage",
        "test_guard_instance_attribute_access_warnings": "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_bool_identity_branch_via_isinstance_fallback": "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_bool_shortcut_and_issubclass_typeerror": "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_handler_type_issubclass_typeerror_branch_direct": "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_issubclass_success_when_callable_is_patched": "tests.unit.test_utilities_guards_full_coverage",
        "test_guards_issubclass_typeerror_when_class_not_treated_as_callable": "tests.unit.test_utilities_guards_full_coverage",
        "test_handler_builder_fluent_methods": "tests.unit.test_models_cqrs_full_coverage",
        "test_handler_decorator_discovery": "tests.unit.test_handler_decorator_discovery",
        "test_handlers": "tests.unit.test_handlers",
        "test_handlers_full_coverage": "tests.unit.test_handlers_full_coverage",
        "test_helper_consolidation_is_prechecked": "tests.unit.test_refactor_policy_family_rules",
        "test_inactive_and_none_value_paths": "tests.unit.test_context_full_coverage",
        "test_init_fallback_and_lazy_returns_result_property": "tests.unit.test_result_full_coverage",
        "test_is_container_negative_paths_and_callable": "tests.unit.test_utilities_guards_full_coverage",
        "test_is_type_non_empty_unknown_and_tuple_and_fallback": "tests.unit.test_utilities_guards_full_coverage",
        "test_is_type_protocol_fallback_branches": "tests.unit.test_utilities_guards_full_coverage",
        "test_is_valid_handles_validation_exception": "tests.unit.test_service_additional",
        "test_lash_runtime_result_paths": "tests.unit.test_result_full_coverage",
        "test_loggings_error_paths_coverage": "tests.unit.test_loggings_error_paths_coverage",
        "test_loggings_full_coverage": "tests.unit.test_loggings_full_coverage",
        "test_loggings_strict_returns": "tests.unit.test_loggings_strict_returns",
        "test_map_error_identity_and_transform": "tests.unit.test_result_additional",
        "test_map_flat_map_and_then_paths": "tests.unit.test_result_full_coverage",
        "test_migrate_protocols_rewrites_references_with_p_alias": "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_inlines_alias_constant_into_constants_class": "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_moves_constant_and_rewrites_reference": "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_moves_manual_uppercase_assignment": "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_normalizes_facade_alias_to_c": "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_to_mro_rejects_unknown_target": "tests.unit.test_refactor_migrate_to_class_mro",
        "test_migrate_typings_rewrites_references_with_t_alias": "tests.unit.test_refactor_migrate_to_class_mro",
        "test_mixins": "tests.unit.test_mixins",
        "test_mixins_full_coverage": "tests.unit.test_mixins_full_coverage",
        "test_model_helpers_remaining_paths": "tests.unit.test_runtime_full_coverage",
        "test_model_support_and_hash_compare_paths": "tests.unit.test_runtime_full_coverage",
        "test_models": "tests.unit.test_models",
        "test_models_base_full_coverage": "tests.unit.test_models_base_full_coverage",
        "test_models_container": "tests.unit.test_models_container",
        "test_models_context_full_coverage": "tests.unit.test_models_context_full_coverage",
        "test_models_cqrs_full_coverage": "tests.unit.test_models_cqrs_full_coverage",
        "test_models_entity_full_coverage": "tests.unit.test_models_entity_full_coverage",
        "test_models_family_blocks_utilities_target": "tests.unit.test_refactor_policy_family_rules",
        "test_models_generic_full_coverage": "tests.unit.test_models_generic_full_coverage",
        "test_namespace_enforce_cli_fails_on_manual_protocol_violation": "tests.unit.test_refactor_cli_models_workflow",
        "test_namespace_enforcer_apply_inserts_future_after_single_line_module_docstring": "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_apply_keeps_script_shebang_when_adding_future": "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_apply_moves_manual_protocol_to_protocols_file": "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_creates_missing_facades_and_rewrites_imports": "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_cyclic_imports_in_tests_directory": "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_internal_private_imports": "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_manual_protocol_outside_canonical_files": "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_manual_typings_and_compat_aliases": "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_detects_missing_runtime_alias_outside_src": "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_does_not_rewrite_indented_import_aliases": "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_enforcer_does_not_rewrite_multiline_import_alias_blocks": "tests.unit.test_refactor_namespace_enforcer",
        "test_namespace_validator": "tests.unit.test_namespace_validator",
        "test_narrow_contextvar_exception_branch": "tests.unit.test_context_full_coverage",
        "test_narrow_contextvar_invalid_inputs": "tests.unit.test_context_full_coverage",
        "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage": "tests.unit.test_transformer_nested_class_propagation",
        "test_nested_class_propagation_updates_import_annotations_and_calls": "tests.unit.test_transformer_nested_class_propagation",
        "test_non_empty_and_normalize_branches": "tests.unit.test_utilities_guards_full_coverage",
        "test_normalization_edge_branches": "tests.unit.test_runtime_full_coverage",
        "test_normalize_to_container_alias_removal_path": "tests.unit.test_runtime_full_coverage",
        "test_normalize_to_metadata_alias_removal_path": "tests.unit.test_runtime_full_coverage",
        "test_ok_accepts_none": "tests.unit.test_result_additional",
        "test_operation_progress_start_operation_sets_runtime_fields": "tests.unit.test_models_generic_full_coverage",
        "test_protocol_and_simple_guard_helpers": "tests.unit.test_utilities_guards_full_coverage",
        "test_protocols_new": "tests.unit.test_protocols_new",
        "test_query_resolve_pagination_wrapper_and_fallback": "tests.unit.test_models_cqrs_full_coverage",
        "test_query_validate_pagination_dict_and_default": "tests.unit.test_models_cqrs_full_coverage",
        "test_rate_limiter_blocks_then_recovers": "tests.unit.test_dispatcher_reliability",
        "test_rate_limiter_jitter_application": "tests.unit.test_dispatcher_reliability",
        "test_reconfigure_and_reset_state_paths": "tests.unit.test_runtime_full_coverage",
        "test_recover_tap_and_tap_error_paths": "tests.unit.test_result_full_coverage",
        "test_refactor_cli_models_workflow": "tests.unit.test_refactor_cli_models_workflow",
        "test_refactor_migrate_to_class_mro": "tests.unit.test_refactor_migrate_to_class_mro",
        "test_refactor_namespace_enforcer": "tests.unit.test_refactor_namespace_enforcer",
        "test_refactor_policy_family_rules": "tests.unit.test_refactor_policy_family_rules",
        "test_refactor_utilities_iter_python_files_includes_examples_and_scripts": "tests.unit.test_refactor_migrate_to_class_mro",
        "test_registry": "tests.unit.test_registry",
        "test_registry_full_coverage": "tests.unit.test_registry_full_coverage",
        "test_result": "tests.unit.test_result",
        "test_result_additional": "tests.unit.test_result_additional",
        "test_result_coverage_100": "tests.unit.test_result_coverage_100",
        "test_result_exception_carrying": "tests.unit.test_result_exception_carrying",
        "test_result_full_coverage": "tests.unit.test_result_full_coverage",
        "test_result_property_raises_on_failure": "tests.unit.test_service_additional",
        "test_reuse_existing_runtime_coverage_branches": "tests.unit.test_runtime_full_coverage",
        "test_reuse_existing_runtime_scenarios": "tests.unit.test_runtime_full_coverage",
        "test_runtime": "tests.unit.test_runtime",
        "test_runtime_coverage_100": "tests.unit.test_runtime_coverage_100",
        "test_runtime_create_instance_failure_branch": "tests.unit.test_runtime_full_coverage",
        "test_runtime_family_blocks_non_runtime_target": "tests.unit.test_refactor_policy_family_rules",
        "test_runtime_full_coverage": "tests.unit.test_runtime_full_coverage",
        "test_runtime_integration_tracking_paths": "tests.unit.test_runtime_full_coverage",
        "test_runtime_misc_remaining_paths": "tests.unit.test_runtime_full_coverage",
        "test_runtime_module_accessors_and_metadata": "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_alias_compatibility": "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_all_missed_branches": "tests.unit.test_runtime_full_coverage",
        "test_runtime_result_remaining_paths": "tests.unit.test_runtime_full_coverage",
        "test_scope_data_validators_and_errors": "tests.unit.test_models_context_full_coverage",
        "test_service": "tests.unit.test_service",
        "test_service_additional": "tests.unit.test_service_additional",
        "test_service_bootstrap": "tests.unit.test_service_bootstrap",
        "test_service_coverage_100": "tests.unit.test_service_coverage_100",
        "test_set_set_all_get_validation_and_error_paths": "tests.unit.test_context_full_coverage",
        "test_settings_coverage": "tests.unit.test_settings_coverage",
        "test_statistics_and_custom_fields_validators": "tests.unit.test_models_context_full_coverage",
        "test_structlog_proxy_context_var_default_when_key_missing": "tests.unit.test_models_context_full_coverage",
        "test_structlog_proxy_context_var_get_set_reset_paths": "tests.unit.test_models_context_full_coverage",
        "test_summary_error_paths_and_bindings_failures": "tests.unit.test_registry_full_coverage",
        "test_summary_properties_and_subclass_storage_reset": "tests.unit.test_registry_full_coverage",
        "test_to_general_value_dict_removed": "tests.unit.test_models_context_full_coverage",
        "test_transformer_class_nesting": "tests.unit.test_transformer_class_nesting",
        "test_transformer_helper_consolidation": "tests.unit.test_transformer_helper_consolidation",
        "test_transformer_nested_class_propagation": "tests.unit.test_transformer_nested_class_propagation",
        "test_type_guards_result": "tests.unit.test_result_full_coverage",
        "test_typings_full_coverage": "tests.unit.test_typings_full_coverage",
        "test_typings_new": "tests.unit.test_typings_new",
        "test_ultrawork_models_cli_runs_dry_run_copy": "tests.unit.test_refactor_cli_models_workflow",
        "test_update_statistics_remove_hook_and_clone_false_result": "tests.unit.test_context_full_coverage",
        "test_utilities": "tests.unit.test_utilities",
        "test_utilities_cache_coverage_100": "tests.unit.test_utilities_cache_coverage_100",
        "test_utilities_collection_coverage_100": "tests.unit.test_utilities_collection_coverage_100",
        "test_utilities_collection_full_coverage": "tests.unit.test_utilities_collection_full_coverage",
        "test_utilities_configuration_coverage_100": "tests.unit.test_utilities_configuration_coverage_100",
        "test_utilities_configuration_full_coverage": "tests.unit.test_utilities_configuration_full_coverage",
        "test_utilities_context_full_coverage": "tests.unit.test_utilities_context_full_coverage",
        "test_utilities_coverage": "tests.unit.test_utilities_coverage",
        "test_utilities_data_mapper": "tests.unit.test_utilities_data_mapper",
        "test_utilities_domain": "tests.unit.test_utilities_domain",
        "test_utilities_domain_full_coverage": "tests.unit.test_utilities_domain_full_coverage",
        "test_utilities_enum_full_coverage": "tests.unit.test_utilities_enum_full_coverage",
        "test_utilities_family_allows_utilities_target": "tests.unit.test_refactor_policy_family_rules",
        "test_utilities_generators_full_coverage": "tests.unit.test_utilities_generators_full_coverage",
        "test_utilities_guards_full_coverage": "tests.unit.test_utilities_guards_full_coverage",
        "test_utilities_mapper_coverage_100": "tests.unit.test_utilities_mapper_coverage_100",
        "test_utilities_mapper_full_coverage": "tests.unit.test_utilities_mapper_full_coverage",
        "test_utilities_parser_full_coverage": "tests.unit.test_utilities_parser_full_coverage",
        "test_utilities_reliability": "tests.unit.test_utilities_reliability",
        "test_utilities_text_full_coverage": "tests.unit.test_utilities_text_full_coverage",
        "test_utilities_type_checker_coverage_100": "tests.unit.test_utilities_type_checker_coverage_100",
        "test_utilities_type_guards_coverage_100": "tests.unit.test_utilities_type_guards_coverage_100",
        "test_validation_like_error_structure": "tests.unit.test_result_full_coverage",
        "test_version": "tests.unit.test_version",
        "test_with_resource_cleanup_runs": "tests.unit.test_result_additional",
        "typings": "tests.unit.typings",
        "u": ("flext_core.utilities", "FlextUtilities"),
        "x": ("flext_core.mixins", "FlextMixins"),
    },
)

__all__ = [
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
    "FlextUnitTestProtocols",
    "InputPayloadMap",
    "InvalidModelForTest",
    "NestedModel",
    "NormalizeComponentScenario",
    "RuntimeCloneService",
    "SampleModel",
    "SimpleObj",
    "SingletonClassForTest",
    "TMessage",
    "TestAliasContainsAny",
    "TestAllLayerIntegration",
    "TestBaseModelCoverage",
    "TestBeartypeClawCompatibility",
    "TestBeartypeConf",
    "TestCaseMap",
    "TestCheckExtraPolicy",
    "TestCheckFieldDescriptions",
    "TestCheckNoAny",
    "TestCheckNoBareCollections",
    "TestCheckNoV1Patterns",
    "TestCollectionUtilitiesCoverage",
    "TestConstantsEnforcement",
    "TestContainerFullCoverage",
    "TestContainsAny",
    "TestContext100Coverage",
    "TestCountUnionMembers",
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
    "TestEnforcementMode",
    "TestEntityCoverageEdgeCases",
    "TestEnumUtilitiesCoverage",
    "TestExceptionsHypothesis",
    "TestExemptions",
    "TestFacadeAccessibility",
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
    "TestFlextModelsExceptionParams",
    "TestFlextProtocols",
    "TestFlextRegistry",
    "TestFlextRuntime",
    "TestFlextSettings",
    "TestFlextSettingsCoverage",
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
    "TestForbiddenCollectionOrigin",
    "TestHandlerDecoratorDiscovery",
    "TestHandlersFullCoverage",
    "TestHelperConsolidationTransformer",
    "TestIsStrNoneUnion",
    "TestLoggingsErrorPaths",
    "TestLoggingsStrictReturns",
    "TestMixinsFullCoverage",
    "TestModels",
    "TestModelsBaseFullCoverage",
    "TestModule",
    "TestNamespacePrefixDerivation",
    "TestProtocolsEnforcement",
    "TestResultExceptionCarrying",
    "TestRuntimeCoverage100",
    "TestService100Coverage",
    "TestServiceBootstrap",
    "TestServiceInternals",
    "TestTypesEnforcement",
    "TestTypingsFullCoverage",
    "TestUtilities",
    "TestUtilitiesCollectionCoverage",
    "TestUtilitiesCollectionFullCoverage",
    "TestUtilitiesConfigurationFullCoverage",
    "TestUtilitiesContextFullCoverage",
    "TestUtilitiesCoverage",
    "TestUtilitiesDataMapper",
    "TestUtilitiesDomainFullCoverage",
    "TestUtilitiesEnforcement",
    "TestUtilitiesGeneratorsFullCoverage",
    "TestUtilitiesParserFullCoverage",
    "TestUtilitiesTextFullCoverage",
    "TestUtilitiesTypeGuardsCoverage100",
    "Teste",
    "Testr",
    "TestrCoverage",
    "TestsCore",
    "Testu",
    "TestuCacheLogger",
    "TestuCacheNormalizeComponent",
    "TestuDomain",
    "TestuMapperAccessors",
    "TestuMapperAdvanced",
    "TestuMapperConversions",
    "TestuMapperExtract",
    "TestuMapperUtils",
    "TestuTypeChecker",
    "TextUtilityContract",
    "UtilitiesCacheCoverage100Namespace",
    "UtilitiesMapperCoverage100Namespace",
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
    "_models",
    "_models_impl",
    "_utilities",
    "c",
    "conftest_infra",
    "contracts",
    "create_compare_entities_cases",
    "create_compare_value_objects_cases",
    "create_hash_entity_cases",
    "create_hash_value_object_cases",
    "d",
    "e",
    "flext_tests",
    "h",
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
    "m",
    "mapper",
    "p",
    "protocols",
    "pytestmark",
    "r",
    "reset_runtime_state",
    "runtime_module",
    "s",
    "t",
    "test_aliases_are_available",
    "test_args_coverage_100",
    "test_async_log_writer_paths",
    "test_async_log_writer_shutdown_with_full_queue",
    "test_bad_string_and_bad_bool_raise_value_error",
    "test_base",
    "test_beartype_engine",
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
    "test_configure_structlog_async_logging_uses_print_logger_factory",
    "test_configure_structlog_edge_paths",
    "test_configure_structlog_print_logger_factory_fallback",
    "test_constants_new",
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
    "test_enforcement",
    "test_ensure_trace_context_dict_conversion_paths",
    "test_entity",
    "test_entity_comparable_map_and_bulk_validation_paths",
    "test_entity_coverage",
    "test_enum_utilities_coverage_100",
    "test_exception_params",
    "test_exceptions",
    "test_execute_and_register_handler_failure_paths",
    "test_export_paths_with_metadata_and_statistics",
    "test_extract_array_index_helpers",
    "test_extract_error_paths_and_prop_accessor",
    "test_files",
    "test_flext_message_type_alias_adapter",
    "test_flow_through_short_circuits_on_failure",
    "test_from_validation_and_to_model_paths",
    "test_get_logger_auto_configures_structlog",
    "test_get_logger_none_name_paths",
    "test_get_plugin_and_register_metadata_and_list_items_exception",
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
    "test_init_fallback_and_lazy_returns_result_property",
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
    "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage",
    "test_nested_class_propagation_updates_import_annotations_and_calls",
    "test_non_empty_and_normalize_branches",
    "test_normalization_edge_branches",
    "test_normalize_to_container_alias_removal_path",
    "test_normalize_to_metadata_alias_removal_path",
    "test_ok_accepts_none",
    "test_operation_progress_start_operation_sets_runtime_fields",
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
    "test_transformer_class_nesting",
    "test_transformer_helper_consolidation",
    "test_transformer_nested_class_propagation",
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
    "test_validation_like_error_structure",
    "test_version",
    "test_with_resource_cleanup_runs",
    "text_contract",
    "typings",
    "u",
    "x",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
