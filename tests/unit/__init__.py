# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Unit package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports
from tests.unit.contracts import _LAZY_IMPORTS as _CHILD_LAZY_0
from tests.unit.flext_tests import _LAZY_IMPORTS as _CHILD_LAZY_1

if TYPE_CHECKING:
    from tests.unit._models import *
    from tests.unit._models_impl import *
    from tests.unit.conftest_infra import *
    from tests.unit.contracts import *
    from tests.unit.flext_tests import *
    from tests.unit.protocols import *
    from tests.unit.test_args_coverage_100 import *
    from tests.unit.test_collection_utilities_coverage_100 import *
    from tests.unit.test_collections_coverage_100 import *
    from tests.unit.test_config import *
    from tests.unit.test_constants import *
    from tests.unit.test_container import *
    from tests.unit.test_container_full_coverage import *
    from tests.unit.test_context import *
    from tests.unit.test_context_coverage_100 import *
    from tests.unit.test_context_full_coverage import *
    from tests.unit.test_coverage_context import *
    from tests.unit.test_coverage_exceptions import *
    from tests.unit.test_coverage_loggings import *
    from tests.unit.test_coverage_models import *
    from tests.unit.test_coverage_utilities import *
    from tests.unit.test_decorators import *
    from tests.unit.test_decorators_discovery_full_coverage import *
    from tests.unit.test_decorators_full_coverage import *
    from tests.unit.test_deprecation_warnings import *
    from tests.unit.test_di_incremental import *
    from tests.unit.test_di_services_access import *
    from tests.unit.test_dispatcher_di import *
    from tests.unit.test_dispatcher_full_coverage import *
    from tests.unit.test_dispatcher_minimal import *
    from tests.unit.test_dispatcher_reliability import *
    from tests.unit.test_dispatcher_timeout_coverage_100 import *
    from tests.unit.test_entity_coverage import *
    from tests.unit.test_enum_utilities_coverage_100 import *
    from tests.unit.test_exceptions import *
    from tests.unit.test_handler_decorator_discovery import *
    from tests.unit.test_handlers import *
    from tests.unit.test_handlers_full_coverage import *
    from tests.unit.test_loggings_error_paths_coverage import *
    from tests.unit.test_loggings_full_coverage import *
    from tests.unit.test_loggings_strict_returns import *
    from tests.unit.test_mixins import *
    from tests.unit.test_mixins_full_coverage import *
    from tests.unit.test_models import *
    from tests.unit.test_models_base_full_coverage import *
    from tests.unit.test_models_container import *
    from tests.unit.test_models_context_full_coverage import *
    from tests.unit.test_models_cqrs_full_coverage import *
    from tests.unit.test_models_entity_full_coverage import *
    from tests.unit.test_models_generic_full_coverage import *
    from tests.unit.test_namespace_validator import *
    from tests.unit.test_pagination_coverage_100 import *
    from tests.unit.test_protocols import *
    from tests.unit.test_refactor_cli_models_workflow import *
    from tests.unit.test_refactor_migrate_to_class_mro import *
    from tests.unit.test_refactor_namespace_enforcer import *
    from tests.unit.test_refactor_policy_family_rules import *
    from tests.unit.test_registry import *
    from tests.unit.test_registry_full_coverage import *
    from tests.unit.test_result import *
    from tests.unit.test_result_additional import *
    from tests.unit.test_result_coverage_100 import *
    from tests.unit.test_result_exception_carrying import *
    from tests.unit.test_result_full_coverage import *
    from tests.unit.test_runtime import *
    from tests.unit.test_runtime_coverage_100 import *
    from tests.unit.test_runtime_full_coverage import *
    from tests.unit.test_service import *
    from tests.unit.test_service_additional import *
    from tests.unit.test_service_bootstrap import *
    from tests.unit.test_service_coverage_100 import *
    from tests.unit.test_settings_coverage import *
    from tests.unit.test_transformer_class_nesting import *
    from tests.unit.test_transformer_helper_consolidation import *
    from tests.unit.test_transformer_nested_class_propagation import *
    from tests.unit.test_typings import *
    from tests.unit.test_typings_full_coverage import *
    from tests.unit.test_utilities_cache_coverage_100 import *
    from tests.unit.test_utilities_collection_coverage_100 import *
    from tests.unit.test_utilities_collection_full_coverage import *
    from tests.unit.test_utilities_configuration_coverage_100 import *
    from tests.unit.test_utilities_configuration_full_coverage import *
    from tests.unit.test_utilities_context_full_coverage import *
    from tests.unit.test_utilities_coverage import *
    from tests.unit.test_utilities_data_mapper import *
    from tests.unit.test_utilities_domain import *
    from tests.unit.test_utilities_domain_full_coverage import *
    from tests.unit.test_utilities_generators_full_coverage import *
    from tests.unit.test_utilities_guards_full_coverage import *
    from tests.unit.test_utilities_mapper_coverage_100 import *
    from tests.unit.test_utilities_mapper_full_coverage import *
    from tests.unit.test_utilities_parser_full_coverage import *
    from tests.unit.test_utilities_reliability import *
    from tests.unit.test_utilities_text_full_coverage import *
    from tests.unit.test_utilities_type_checker_coverage_100 import *
    from tests.unit.test_utilities_type_guards_coverage_100 import *
    from tests.unit.test_version import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    **_CHILD_LAZY_0,
    **_CHILD_LAZY_1,
    "AttrObject": "tests.unit.test_utilities_mapper_full_coverage",
    "BadBool": "tests.unit.test_utilities_mapper_full_coverage",
    "BadConfigForTest": "tests.unit._models_impl",
    "BadMapping": "tests.unit.test_utilities_mapper_full_coverage",
    "BadString": "tests.unit.test_utilities_mapper_full_coverage",
    "CacheTestModel": "tests.unit._models_impl",
    "ClearCacheScenario": "tests.unit.test_utilities_cache_coverage_100",
    "ComplexModel": "tests.unit._models_impl",
    "ConfigModelForTest": "tests.unit._models_impl",
    "ExplodingLenList": "tests.unit.test_utilities_mapper_full_coverage",
    "FlextProtocols": "tests.unit.protocols",
    "InputPayloadMap": "tests.unit._models_impl",
    "InvalidModelForTest": "tests.unit._models_impl",
    "NORMALIZE_COMPONENT_SCENARIOS": "tests.unit.test_utilities_cache_coverage_100",
    "NestedClassPropagationTransformer": "tests.unit.test_transformer_nested_class_propagation",
    "NestedModel": "tests.unit._models_impl",
    "NormalizeComponentScenario": "tests.unit.test_utilities_cache_coverage_100",
    "RuntimeCloneService": "tests.unit.test_service_additional",
    "SORT_KEY_SCENARIOS": "tests.unit.test_utilities_cache_coverage_100",
    "SampleModel": "tests.unit._models_impl",
    "SimpleObj": "tests.unit.test_utilities_mapper_coverage_100",
    "SingletonClassForTest": "tests.unit._models_impl",
    "SortKeyScenario": "tests.unit.test_utilities_cache_coverage_100",
    "T": "tests.unit.test_utilities_type_checker_coverage_100",
    "TMessage": "tests.unit.test_utilities_type_checker_coverage_100",
    "TestCaseMap": "tests.unit._models_impl",
    "TestCollectionUtilitiesCoverage": "tests.unit.test_collection_utilities_coverage_100",
    "TestConstants": "tests.unit.test_constants",
    "TestContainerFullCoverage": "tests.unit.test_container_full_coverage",
    "TestContext100Coverage": "tests.unit.test_context_coverage_100",
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
    "TestEntityCoverageEdgeCases": "tests.unit.test_entity_coverage",
    "TestEnumUtilitiesCoverage": "tests.unit.test_enum_utilities_coverage_100",
    "TestExceptionsHypothesis": "tests.unit.test_exceptions",
    "TestFlextContainer": "tests.unit.test_container",
    "TestFlextContext": "tests.unit.test_context",
    "TestFlextDecorators": "tests.unit.test_decorators",
    "TestFlextHandlers": "tests.unit.test_handlers",
    "TestFlextInfraNamespaceValidator": "tests.unit.test_namespace_validator",
    "TestFlextMixinsCQRS": "tests.unit.test_mixins",
    "TestFlextMixinsNestedClasses": "tests.unit.test_mixins",
    "TestFlextModelsCollectionsCoverage100": "tests.unit.test_collections_coverage_100",
    "TestFlextModelsContainer": "tests.unit.test_models_container",
    "TestFlextProtocols": "tests.unit.test_protocols",
    "TestFlextRegistry": "tests.unit.test_registry",
    "TestFlextRuntime": "tests.unit.test_runtime",
    "TestFlextSettings": "tests.unit.test_config",
    "TestFlextSettingsCoverage": "tests.unit.test_settings_coverage",
    "TestFlextUtilitiesArgs": "tests.unit.test_args_coverage_100",
    "TestFlextUtilitiesConfiguration": "tests.unit.test_utilities_configuration_coverage_100",
    "TestFlextUtilitiesReliability": "tests.unit.test_utilities_reliability",
    "TestFlextVersion": "tests.unit.test_version",
    "TestHandlerDecoratorDiscovery": "tests.unit.test_handler_decorator_discovery",
    "TestHandlersFullCoverage": "tests.unit.test_handlers_full_coverage",
    "TestHelperConsolidationTransformer": "tests.unit.test_transformer_helper_consolidation",
    "TestLoggingsErrorPaths": "tests.unit.test_loggings_error_paths_coverage",
    "TestLoggingsStrictReturns": "tests.unit.test_loggings_strict_returns",
    "TestMixinsFullCoverage": "tests.unit.test_mixins_full_coverage",
    "TestModels": "tests.unit.test_models",
    "TestModelsBaseFullCoverage": "tests.unit.test_models_base_full_coverage",
    "TestModule": "tests.unit.test_loggings_full_coverage",
    "TestPaginationCoverage100": "tests.unit.test_pagination_coverage_100",
    "TestResultExceptionCarrying": "tests.unit.test_result_exception_carrying",
    "TestRuntimeCoverage100": "tests.unit.test_runtime_coverage_100",
    "TestService100Coverage": "tests.unit.test_service_coverage_100",
    "TestServiceBootstrap": "tests.unit.test_service_bootstrap",
    "TestServiceInternals": "tests.unit.test_service",
    "TestTypings": "tests.unit.test_typings",
    "TestTypingsFullCoverage": "tests.unit.test_typings_full_coverage",
    "TestUnitModels": "tests.unit._models",
    "TestUtilitiesCollectionCoverage": "tests.unit.test_utilities_collection_coverage_100",
    "TestUtilitiesCollectionFullCoverage": "tests.unit.test_utilities_collection_full_coverage",
    "TestUtilitiesConfigurationFullCoverage": "tests.unit.test_utilities_configuration_full_coverage",
    "TestUtilitiesContextFullCoverage": "tests.unit.test_utilities_context_full_coverage",
    "TestUtilitiesCoverage": "tests.unit.test_utilities_coverage",
    "TestUtilitiesDataMapper": "tests.unit.test_utilities_data_mapper",
    "TestUtilitiesDomainFullCoverage": "tests.unit.test_utilities_domain_full_coverage",
    "TestUtilitiesGeneratorsFullCoverage": "tests.unit.test_utilities_generators_full_coverage",
    "TestUtilitiesParserFullCoverage": "tests.unit.test_utilities_parser_full_coverage",
    "TestUtilitiesTextFullCoverage": "tests.unit.test_utilities_text_full_coverage",
    "TestUtilitiesTypeGuardsCoverage100": "tests.unit.test_utilities_type_guards_coverage_100",
    "Teste": "tests.unit.test_exceptions",
    "Testr": "tests.unit.test_result",
    "TestrCoverage": "tests.unit.test_result_coverage_100",
    "TestsCore": "tests.unit.test_service",
    "Testu": "tests.unit.test_coverage_utilities",
    "TestuCacheClearObjectCache": "tests.unit.test_utilities_cache_coverage_100",
    "TestuCacheGenerateCacheKey": "tests.unit.test_utilities_cache_coverage_100",
    "TestuCacheHasCacheAttributes": "tests.unit.test_utilities_cache_coverage_100",
    "TestuCacheLogger": "tests.unit.test_utilities_cache_coverage_100",
    "TestuCacheNormalizeComponent": "tests.unit.test_utilities_cache_coverage_100",
    "TestuCacheSortDictKeys": "tests.unit.test_utilities_cache_coverage_100",
    "TestuCacheSortKey": "tests.unit.test_utilities_cache_coverage_100",
    "TestuDomain": "tests.unit.test_utilities_domain",
    "TestuMapperAccessors": "tests.unit.test_utilities_mapper_coverage_100",
    "TestuMapperAdvanced": "tests.unit.test_utilities_mapper_coverage_100",
    "TestuMapperBuild": "tests.unit.test_utilities_mapper_coverage_100",
    "TestuMapperConversions": "tests.unit.test_utilities_mapper_coverage_100",
    "TestuMapperExtract": "tests.unit.test_utilities_mapper_coverage_100",
    "TestuMapperUtils": "tests.unit.test_utilities_mapper_coverage_100",
    "TestuTypeChecker": "tests.unit.test_utilities_type_checker_coverage_100",
    "UtilitiesCacheCoverage100Namespace": "tests.unit.test_utilities_cache_coverage_100",
    "UtilitiesMapperCoverage100Namespace": "tests.unit.test_utilities_mapper_coverage_100",
    "UtilitiesMapperFullCoverageNamespace": "tests.unit.test_utilities_mapper_full_coverage",
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
    "conftest_infra": "tests.unit.conftest_infra",
    "contracts": "tests.unit.contracts",
    "create_compare_entities_cases": "tests.unit.test_utilities_domain",
    "create_compare_value_objects_cases": "tests.unit.test_utilities_domain",
    "create_hash_entity_cases": "tests.unit.test_utilities_domain",
    "create_hash_value_object_cases": "tests.unit.test_utilities_domain",
    "flext_tests": "tests.unit.flext_tests",
    "generators_module": "tests.unit.test_utilities_generators_full_coverage",
    "handlers_module": "tests.unit.test_handlers_full_coverage",
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
    "mapper": "tests.unit.test_utilities_mapper_full_coverage",
    "p": "tests.unit.protocols",
    "protocols": "tests.unit.protocols",
    "pytestmark": "tests.unit.test_utilities_type_checker_coverage_100",
    "reset_runtime_state": "tests.unit.test_runtime_full_coverage",
    "runtime_cov_tests": "tests.unit.test_runtime_full_coverage",
    "runtime_module": "tests.unit.test_runtime_full_coverage",
    "runtime_tests": "tests.unit.test_runtime_full_coverage",
    "test_aliases_are_available": "tests.unit.test_utilities_guards_full_coverage",
    "test_args_coverage_100": "tests.unit.test_args_coverage_100",
    "test_async_log_writer_paths": "tests.unit.test_runtime_full_coverage",
    "test_async_log_writer_shutdown_with_full_queue": "tests.unit.test_runtime_full_coverage",
    "test_bad_string_and_bad_bool_raise_value_error": "tests.unit.test_utilities_mapper_full_coverage",
    "test_build_apply_transform_and_process_error_paths": "tests.unit.test_utilities_mapper_full_coverage",
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
    "test_configure_structlog_edge_paths": "tests.unit.test_runtime_full_coverage",
    "test_configure_structlog_print_logger_factory_fallback": "tests.unit.test_runtime_full_coverage",
    "test_constants": "tests.unit.test_constants",
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
    "test_convert_default_fallback_matrix": "tests.unit.test_utilities_mapper_full_coverage",
    "test_convert_sequence_branch_returns_tuple": "tests.unit.test_utilities_mapper_full_coverage",
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
    "test_ensure_trace_context_dict_conversion_paths": "tests.unit.test_runtime_full_coverage",
    "test_entity_comparable_map_and_bulk_validation_paths": "tests.unit.test_models_entity_full_coverage",
    "test_entity_coverage": "tests.unit.test_entity_coverage",
    "test_enum_utilities_coverage_100": "tests.unit.test_enum_utilities_coverage_100",
    "test_exceptions": "tests.unit.test_exceptions",
    "test_execute_and_register_handler_failure_paths": "tests.unit.test_registry_full_coverage",
    "test_export_paths_with_metadata_and_statistics": "tests.unit.test_context_full_coverage",
    "test_extract_array_index_helpers": "tests.unit.test_utilities_mapper_full_coverage",
    "test_extract_error_paths_and_prop_accessor": "tests.unit.test_utilities_mapper_full_coverage",
    "test_extract_field_value_and_ensure_variants": "tests.unit.test_utilities_mapper_full_coverage",
    "test_filter_map_normalize_convert_helpers": "tests.unit.test_utilities_mapper_full_coverage",
    "test_flext_message_type_alias_adapter": "tests.unit.test_models_cqrs_full_coverage",
    "test_flow_through_short_circuits_on_failure": "tests.unit.test_result_additional",
    "test_from_validation_and_to_model_paths": "tests.unit.test_result_full_coverage",
    "test_general_value_helpers_and_logger": "tests.unit.test_utilities_mapper_full_coverage",
    "test_get_logger_none_name_paths": "tests.unit.test_runtime_full_coverage",
    "test_get_plugin_and_register_metadata_and_list_items_exception": "tests.unit.test_registry_full_coverage",
    "test_group_sort_unique_slice_chunk_branches": "tests.unit.test_utilities_mapper_full_coverage",
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
    "test_narrow_to_string_keyed_dict_and_mapping_paths": "tests.unit.test_utilities_mapper_full_coverage",
    "test_nested_class_propagation_preserves_asname_and_rewrites_alias_usage": "tests.unit.test_transformer_nested_class_propagation",
    "test_nested_class_propagation_updates_import_annotations_and_calls": "tests.unit.test_transformer_nested_class_propagation",
    "test_non_empty_and_normalize_branches": "tests.unit.test_utilities_guards_full_coverage",
    "test_normalization_edge_branches": "tests.unit.test_runtime_full_coverage",
    "test_normalize_to_container_alias_removal_path": "tests.unit.test_runtime_full_coverage",
    "test_normalize_to_metadata_alias_removal_path": "tests.unit.test_runtime_full_coverage",
    "test_ok_accepts_none": "tests.unit.test_result_additional",
    "test_operation_progress_start_operation_sets_runtime_fields": "tests.unit.test_models_generic_full_coverage",
    "test_pagination_coverage_100": "tests.unit.test_pagination_coverage_100",
    "test_protocol_and_simple_guard_helpers": "tests.unit.test_utilities_guards_full_coverage",
    "test_protocols": "tests.unit.test_protocols",
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
    "test_retry_policy_behavior": "tests.unit.test_dispatcher_reliability",
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
    "test_take_and_as_branches": "tests.unit.test_utilities_mapper_full_coverage",
    "test_to_general_value_dict_removed": "tests.unit.test_models_context_full_coverage",
    "test_transform_and_deep_eq_branches": "tests.unit.test_utilities_mapper_full_coverage",
    "test_transform_option_extract_and_step_helpers": "tests.unit.test_utilities_mapper_full_coverage",
    "test_transformer_class_nesting": "tests.unit.test_transformer_class_nesting",
    "test_transformer_helper_consolidation": "tests.unit.test_transformer_helper_consolidation",
    "test_transformer_nested_class_propagation": "tests.unit.test_transformer_nested_class_propagation",
    "test_type_guards_and_narrowing_failures": "tests.unit.test_utilities_mapper_full_coverage",
    "test_type_guards_result": "tests.unit.test_result_full_coverage",
    "test_typings": "tests.unit.test_typings",
    "test_typings_full_coverage": "tests.unit.test_typings_full_coverage",
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
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
