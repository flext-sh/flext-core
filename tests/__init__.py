# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

_LAZY_IMPORTS = merge_lazy_imports(
    (
        "._constants",
        "._models",
        "._utilities",
        ".benchmark",
        ".fixtures",
        ".integration",
        ".unit",
    ),
    build_lazy_import_map(
        {
            "._constants": ("_constants",),
            "._constants.domain": ("TestsFlextConstantsDomain",),
            "._constants.errors": ("TestsFlextConstantsErrors",),
            "._constants.fixtures": ("TestsFlextConstantsFixtures",),
            "._constants.loggings": ("TestsFlextConstantsLoggings",),
            "._constants.other": ("TestsFlextConstantsOther",),
            "._constants.result": ("TestsFlextConstantsResult",),
            "._constants.services": ("TestsFlextConstantsServices",),
            "._constants.settings": ("TestsFlextConstantsSettings",),
            "._models": ("_models",),
            "._models._mixins.container": ("TestsFlextModelsContainerMixin",),
            "._models._mixins.core": ("TestsFlextModelsCoreMixin",),
            "._models._mixins.core_errors": ("TestsFlextModelsCoreErrorsMixin",),
            "._models._mixins.core_public": ("TestsFlextModelsCorePublicMixin",),
            "._models._mixins.core_state": ("TestsFlextModelsCoreStateMixin",),
            "._models._mixins.domain": ("TestsFlextModelsDomainMixin",),
            "._models._mixins.fixture_payloads": (
                "TestsFlextModelsFixturePayloadsMixin",
            ),
            "._models._mixins.fixture_suite": ("TestsFlextModelsFixtureSuiteMixin",),
            "._models._mixins.fixtures": ("TestsFlextModelsFixtureDictsMixin",),
            "._models._mixins.guards_mapper": ("TestsFlextModelsGuardsMapperMixin",),
            "._models._mixins.service_case_core": (
                "TestsFlextModelsServiceCaseCoreMixin",
            ),
            "._models._mixins.service_case_reliability": (
                "TestsFlextModelsServiceCaseReliabilityMixin",
            ),
            "._models._mixins.service_case_validation": (
                "TestsFlextModelsServiceCaseValidationMixin",
            ),
            "._models._mixins.service_cases": ("TestsFlextModelsServiceCasesMixin",),
            "._models._mixins.test_data": ("TestsFlextModelsTestDataMixin",),
            "._models._mixins.test_data_identity": (
                "TestsFlextModelsTestDataIdentityMixin",
            ),
            "._models._mixins.test_data_values": (
                "TestsFlextModelsTestDataValuesMixin",
            ),
            "._models.mixins": ("TestsFlextModelsMixins",),
            "._utilities": ("_utilities",),
            "._utilities.case_factories": ("TestsFlextUtilitiesCaseFactoriesMixin",),
            "._utilities.case_generators": ("TestsFlextUtilitiesCaseGeneratorsMixin",),
            "._utilities.case_service_factories": (
                "TestsFlextUtilitiesCaseServiceFactoriesMixin",
            ),
            "._utilities.contracts": ("TestsFlextUtilitiesContractsMixin",),
            "._utilities.dispatch": ("TestsFlextUtilitiesDispatchMixin",),
            "._utilities.parser_reliability": (
                "TestsFlextUtilitiesParserReliabilityMixin",
            ),
            "._utilities.parser_scenarios": (
                "TestsFlextUtilitiesParserScenariosMixin",
            ),
            "._utilities.railway": ("TestsFlextUtilitiesRailwayMixin",),
            "._utilities.railway_cases": ("TestsFlextUtilitiesRailwayCasesMixin",),
            "._utilities.railway_pipelines": (
                "TestsFlextUtilitiesRailwayPipelinesMixin",
            ),
            "._utilities.railway_services": (
                "TestsFlextUtilitiesRailwayServicesMixin",
            ),
            "._utilities.reliability_scenarios": (
                "TestsFlextUtilitiesReliabilityScenariosMixin",
            ),
            "._utilities.service_factories": (
                "TestsFlextUtilitiesServiceFactoriesMixin",
            ),
            "._utilities.services": ("TestsFlextUtilitiesServicesMixin",),
            "._utilities.user_factories": ("TestsFlextUtilitiesUserFactoriesMixin",),
            "._utilities.validation_factories": (
                "TestsFlextUtilitiesValidationFactoriesMixin",
            ),
            "._utilities.validation_network": (
                "TestsFlextUtilitiesValidationNetworkScenarios",
            ),
            "._utilities.validation_numeric": (
                "TestsFlextUtilitiesValidationNumericScenarios",
            ),
            "._utilities.validation_pattern": (
                "TestsFlextUtilitiesValidationPatternScenarios",
            ),
            "._utilities.validation_scenarios": (
                "TestsFlextUtilitiesValidationScenariosMixin",
            ),
            "._utilities.validation_string": (
                "TestsFlextUtilitiesValidationStringScenarios",
            ),
            "._utilities.validation_uri": (
                "TestsFlextUtilitiesValidationUriScenarios",
            ),
            ".base": (
                "TestsFlextServiceBase",
                "s",
            ),
            ".benchmark": ("benchmark",),
            ".benchmark.test_container_memory": ("TestsFlextContainerMemory",),
            ".benchmark.test_container_performance": (
                "TestsFlextContainerPerformance",
            ),
            ".benchmark.test_lazy_performance": ("TestsFlextLazyPerformance",),
            ".conftest": ("conftest",),
            ".constants": (
                "TestsFlextConstants",
                "c",
            ),
            ".fixtures": ("fixtures",),
            ".fixtures.bad_module": (
                "TestsFlextBadAccessors",
                "TestsFlextBadAnyField",
                "TestsFlextBadBareCollection",
                "TestsFlextBadConstants",
                "TestsFlextBadFrozen",
                "TestsFlextBadInlineUnion",
                "TestsFlextBadMissingDesc",
                "TestsFlextBadMutableDefault",
                "TestsFlextBadWorkerSettings",
            ),
            ".fixtures.clean_module": (
                "TestsFlextCleanConstants",
                "TestsFlextCleanModels",
                "TestsFlextCleanProtocols",
                "TestsFlextCleanServiceBase",
            ),
            ".integration": ("integration",),
            ".integration.migration_validation_cases": (
                "FlextMigrationApplicationCase",
            ),
            ".integration.service_fixtures": (
                "FlextServiceFixtures",
                "LifecycleService",
                "NotificationService",
                "ServiceConfig",
                "UserQueryService",
                "UserServiceEntity",
            ),
            ".integration.service_lifecycle_cases": ("FlextServiceLifecycleCases",),
            ".integration.settings_integration_factories": (
                "FlextSettingsFactories",
                "SettingsConfigTestCase",
                "SettingsConfigTestFactories",
                "SettingsThreadSafetyTest",
            ),
            ".integration.settings_integration_precedence": (
                "FlextSettingsPrecedenceCase",
            ),
            ".integration.system_integration_cases": ("FlextSystemWorkflowCases",),
            ".integration.test_architecture": ("TestsFlextAutomatedArchitecture",),
            ".integration.test_documented_patterns": ("TestsFlextDocumentedPatterns",),
            ".integration.test_examples_execution": ("TestsFlextExamplesExecution",),
            ".integration.test_integration": ("TestsFlextLibraryIntegration",),
            ".integration.test_migration_validation": (
                "TestsFlextMigrationValidation",
            ),
            ".integration.test_service": ("TestsFlextServiceIntegration",),
            ".integration.test_settings_integration": (
                "TestsFlextSettingsIntegration",
            ),
            ".integration.test_system": ("TestsFlextSystemIntegration",),
            ".models": (
                "TestsFlextModels",
                "m",
            ),
            ".protocols": (
                "TestsFlextProtocols",
                "p",
            ),
            ".typings": (
                "TestsFlextTypes",
                "t",
            ),
            ".unit": ("unit",),
            ".unit._models.test_base": ("TestsFlextModelsBase",),
            ".unit._models.test_cqrs": ("TestsFlextModelsCQRS",),
            ".unit._models.test_enforcement_sources": (
                "TestsFlextModelsEnforcementSources",
            ),
            ".unit._models.test_entity": ("TestsFlextModelsEntity",),
            ".unit._models.test_exception_params_core": (
                "TestsFlextModelsExceptionParamsCore",
            ),
            ".unit._models.test_exception_params_operations": (
                "TestsFlextModelsExceptionParamsOperations",
            ),
            ".unit._models.test_exception_params_resources": (
                "TestsFlextModelsExceptionParamsResources",
            ),
            ".unit._utilities.test_guards": ("TestsFlextUtilitiesGuards",),
            ".unit._utilities.test_mapper": ("TestsFlextUtilitiesMapper",),
            ".unit.test_beartype_engine_annotations": (
                "TestsFlextBeartypeEngineAnnotations",
            ),
            ".unit.test_beartype_engine_claw_packages": (
                "TestsFlextBeartypeEngineClawPackages",
            ),
            ".unit.test_beartype_engine_config": ("TestsFlextBeartypeEngineConfig",),
            ".unit.test_beartype_engine_import_hooks": (
                "TestsFlextBeartypeEngineImportHooks",
            ),
            ".unit.test_beartype_engine_namespace_hooks": (
                "TestsFlextBeartypeEngineNamespaceHooks",
            ),
            ".unit.test_beartype_engine_runtime": ("TestsFlextBeartypeEngineRuntime",),
            ".unit.test_constants_new": ("TestsFlextConstantsNew",),
            ".unit.test_constants_project_metadata": (
                "TestsFlextConstantsProjectMetadata",
            ),
            ".unit.test_container_config": ("TestsFlextContainerConfig",),
            ".unit.test_container_lifecycle": ("TestsFlextContainerLifecycle",),
            ".unit.test_container_properties": ("TestsFlextContainerProperties",),
            ".unit.test_container_registration": ("TestsFlextContainerRegistration",),
            ".unit.test_container_resolution": ("TestsFlextContainerResolution",),
            ".unit.test_context": ("TestsFlextContext",),
            ".unit.test_coverage_loggings": ("TestsFlextCoverageLoggings",),
            ".unit.test_decorators_combined": ("TestsFlextDecoratorsCombined",),
            ".unit.test_decorators_discovery_full_coverage": (
                "TestsFlextDecoratorsDiscovery",
            ),
            ".unit.test_decorators_full_coverage": ("TestsFlextDecorators",),
            ".unit.test_decorators_injection_logging": (
                "TestsFlextDecoratorsInjectionLogging",
            ),
            ".unit.test_decorators_railway_retry": (
                "TestsFlextDecoratorsRailwayRetry",
            ),
            ".unit.test_deprecation_warnings": ("TestsFlextDeprecationWarnings",),
            ".unit.test_dispatcher": ("TestsFlextDispatcher",),
            ".unit.test_enforcement_accessors": ("TestsFlextEnforcementAccessors",),
            ".unit.test_enforcement_apt_hooks": ("TestsFlextEnforcementAptHooks",),
            ".unit.test_enforcement_catalog": ("TestsFlextEnforcementCatalog",),
            ".unit.test_enforcement_integration": ("TestsFlextEnforcementIntegration",),
            ".unit.test_enforcement_layers": ("TestsFlextEnforcementLayers",),
            ".unit.test_enforcement_models": ("TestsFlextEnforcementModels",),
            ".unit.test_enforcement_namespace": ("TestsFlextEnforcementNamespace",),
            ".unit.test_enforcement_reports": ("TestsFlextEnforcementReports",),
            ".unit.test_enum_utilities_coverage_100": ("TestsFlextEnumUtilities",),
            ".unit.test_exceptions_base": ("TestsFlextExceptionsBase",),
            ".unit.test_exceptions_public_metrics": (
                "TestsFlextCoverageExceptionMetrics",
            ),
            ".unit.test_exceptions_structured_contracts": (
                "TestsFlextCoverageExceptionContracts",
            ),
            ".unit.test_exceptions_typed_metrics": (
                "TestsFlextExceptionsTypedMetrics",
            ),
            ".unit.test_handler_decorator_edges": ("TestsFlextHandlerDecoratorEdges",),
            ".unit.test_handler_decorator_metadata": (
                "TestsFlextHandlerDecoratorMetadata",
            ),
            ".unit.test_handler_discovery_class": ("TestsFlextHandlerDiscoveryClass",),
            ".unit.test_handler_discovery_module": (
                "TestsFlextHandlerDiscoveryModule",
            ),
            ".unit.test_handlers_dispatch": ("TestsFlextHandlersDispatch",),
            ".unit.test_handlers_factory": ("TestsFlextHandlersFactory",),
            ".unit.test_handlers_lifecycle": ("TestsFlextHandlersLifecycle",),
            ".unit.test_handlers_properties": ("TestsFlextHandlersProperties",),
            ".unit.test_handlers_validation_context": (
                "TestsFlextHandlersValidationContext",
            ),
            ".unit.test_lazy_exports": ("TestsFlextLazy",),
            ".unit.test_lazy_exports_merge": ("TestsFlextLazyMerge",),
            ".unit.test_loggings_full_coverage": ("TestsFlextLoggings",),
            ".unit.test_mixins": ("TestsFlextMixins",),
            ".unit.test_models": ("TestsFlextModelsUnit",),
            ".unit.test_models_base_full_coverage": (
                "TestsFlextModelsBaseFullCoverage",
            ),
            ".unit.test_models_container": ("TestsFlextModelsContainer",),
            ".unit.test_models_cqrs_full_coverage": ("TestsFlextModelsCqrs",),
            ".unit.test_models_project_metadata": ("TestsFlextModelsProjectMetadata",),
            ".unit.test_project_metadata_facade_access": (
                "TestsFlextFacadeFlatSsotAccess",
            ),
            ".unit.test_public_api_contract": ("TestsFlextCorePublicApiContract",),
            ".unit.test_registry": ("TestsFlextRegistry",),
            ".unit.test_result_callables_fold": ("TestsFlextResultCallablesFold",),
            ".unit.test_result_chain_helpers": ("TestsFlextResultChainHelpers",),
            ".unit.test_result_exception_failures": (
                "TestsFlextResultExceptionFailures",
            ),
            ".unit.test_result_exception_mapping": (
                "TestsFlextResultExceptionMapping",
            ),
            ".unit.test_result_exception_safe_callable": (
                "TestsFlextResultExceptionSafeCallable",
            ),
            ".unit.test_result_exception_traverse_validation": (
                "TestsFlextResultExceptionTraverseValidation",
            ),
            ".unit.test_result_laws": ("TestsFlextResultLaws",),
            ".unit.test_result_operations": ("TestsFlextResultOperations",),
            ".unit.test_result_recent_behaviors": ("TestsFlextResultRecentBehaviors",),
            ".unit.test_result_transforms": ("TestsFlextResultTransforms",),
            ".unit.test_result_traverse_resource": (
                "TestsFlextResultTraverseResource",
            ),
            ".unit.test_runtime": ("TestsFlextRuntime",),
            ".unit.test_service": ("TestsFlextService",),
            ".unit.test_service_bootstrap": ("TestsFlextServiceBootstrap",),
            ".unit.test_settings": ("TestsFlextSettings",),
            ".unit.test_settings_validation_alias": (
                "TestUpdateGlobalWithValidationAlias",
            ),
            ".unit.test_typings_aliases": ("TestsFlextTypesAliases",),
            ".unit.test_typings_containers": ("TestsFlextTypesContainers",),
            ".unit.test_typings_validation_numbers": (
                "TestsFlextTypesValidationNumbers",
            ),
            ".unit.test_typings_validation_scalars": (
                "TestsFlextTypesValidationScalars",
            ),
            ".unit.test_utilities": ("TestsFlextUtilitiesSmoke",),
            ".unit.test_utilities_collection_coverage_100": (
                "TestsFlextUtilitiesCollection",
            ),
            ".unit.test_utilities_coverage": ("TestsFlextUtilitiesCoverage",),
            ".unit.test_utilities_domain": ("TestsFlextUtilitiesDomain",),
            ".unit.test_utilities_generators_full_coverage": (
                "TestsFlextUtilitiesGenerators",
            ),
            ".unit.test_utilities_project_metadata_config": (
                "TestsFlextUtilitiesProjectMetadataConfig",
            ),
            ".unit.test_utilities_project_metadata_read": (
                "TestsFlextUtilitiesProjectMetadataRead",
            ),
            ".unit.test_utilities_pydantic_coverage_100": (
                "TestsFlextUtilitiesPydantic",
            ),
            ".unit.test_utilities_reliability": ("TestsFlextUtilitiesReliability",),
            ".unit.test_utilities_runtime_violation_registry_coverage_100": (
                "TestsFlextRuntimeViolationRegistry",
            ),
            ".unit.test_utilities_settings_coverage_100": (
                "TestsFlextUtilitiesSettings",
                "TestsFlextUtilitiesSettingsEnvFile",
                "TestsFlextUtilitiesSettingsRegisterFactory",
            ),
            ".unit.test_utilities_text_full_coverage": ("TestsFlextUtilitiesText",),
            ".unit.test_utilities_type_guards_coverage_100": (
                "TestsFlextUtilitiesTypeGuards",
            ),
            ".unit.test_version": ("TestsFlextVersion",),
            ".utilities": (
                "TestsFlextUtilities",
                "u",
            ),
            "flext_tests": (
                "d",
                "e",
                "h",
                "r",
                "td",
                "tf",
                "tk",
                "tm",
                "tv",
                "x",
            ),
        },
    ),
    exclude_names=(
        "cleanup_submodule_namespace",
        "install_lazy_exports",
        "lazy_getattr",
        "logger",
        "merge_lazy_imports",
        "output",
        "output_reporting",
        "pytest_addoption",
        "pytest_collect_file",
        "pytest_collection_modifyitems",
        "pytest_configure",
        "pytest_runtest_setup",
        "pytest_runtest_teardown",
        "pytest_sessionfinish",
        "pytest_sessionstart",
        "pytest_terminal_summary",
        "pytest_warning_recorded",
    ),
    module_name=__name__,
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
