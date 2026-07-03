"""Lazy export map part 02."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map

TESTS_FLEXT_CORE_LAZY_IMPORTS_PART_02 = build_lazy_import_map({
    "._utilities.railway_services": ("TestsFlextUtilitiesRailwayServicesMixin",),
    "._utilities.reliability_scenarios": (
        "TestsFlextUtilitiesReliabilityScenariosMixin",
    ),
    "._utilities.service_factories": ("TestsFlextUtilitiesServiceFactoriesMixin",),
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
    "._utilities.validation_string": ("TestsFlextUtilitiesValidationStringScenarios",),
    "._utilities.validation_uri": ("TestsFlextUtilitiesValidationUriScenarios",),
    ".base": (
        "TestsFlextServiceBase",
        "s",
    ),
    ".benchmark.test_container_memory": ("TestsFlextContainerMemory",),
    ".benchmark.test_container_performance": ("TestsFlextContainerPerformance",),
    ".benchmark.test_lazy_performance": ("TestsFlextLazyPerformance",),
    ".constants": (
        "TestsFlextConstants",
        "c",
    ),
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
    ".integration.migration_validation_cases": ("FlextMigrationApplicationCase",),
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
    ".integration.settings_integration_precedence": ("FlextSettingsPrecedenceCase",),
    ".integration.system_integration_cases": ("FlextSystemWorkflowCases",),
    ".integration.test_architecture": ("TestsFlextAutomatedArchitecture",),
    ".integration.test_documented_patterns": ("TestsFlextDocumentedPatterns",),
    ".integration.test_examples_execution": ("TestsFlextExamplesExecution",),
    ".integration.test_integration": ("TestsFlextLibraryIntegration",),
    ".integration.test_migration_validation": ("TestsFlextMigrationValidation",),
    ".integration.test_service": ("TestsFlextServiceIntegration",),
    ".integration.test_settings_integration": ("TestsFlextSettingsIntegration",),
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
})

__all__: list[str] = ["TESTS_FLEXT_CORE_LAZY_IMPORTS_PART_02"]
