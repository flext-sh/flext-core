# AUTO-GENERATED FILE — Regenerate with: make gen
"""Integration package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map({
    ".migration_validation_cases": ("TestsFlextFlextMigrationApplicationCase",),
    ".service_fixtures": (
        "TestsFlextFlextServiceFixtures",
        "TestsFlextLifecycleService",
        "TestsFlextNotificationService",
        "TestsFlextServiceConfig",
        "TestsFlextUserQueryService",
        "TestsFlextUserServiceEntity",
    ),
    ".service_lifecycle_cases": ("TestsFlextFlextServiceLifecycleCases",),
    ".settings_integration_factories": (
        "TestsFlextFlextSettingsFactories",
        "TestsFlextSettingsConfigTestCase",
        "TestsFlextSettingsConfigTestFactories",
        "TestsFlextSettingsThreadSafetyTest",
    ),
    ".settings_integration_precedence": ("TestsFlextFlextSettingsPrecedenceCase",),
    ".system_integration_cases": ("TestsFlextFlextSystemWorkflowCases",),
    ".test_architecture": ("TestsFlextCoreArchitecture",),
    ".test_documented_patterns": ("TestsFlextCoreDocumentedPatterns",),
    ".test_examples_execution": ("TestsFlextExamplesExecution",),
    ".test_integration": ("TestsFlextCoreIntegration",),
    ".test_migration_validation": ("TestsFlextCoreMigrationValidation",),
    ".test_service": ("TestsFlextCoreService",),
    ".test_settings_integration": ("TestsFlextSettingsIntegration",),
    ".test_system": ("TestsFlextCoreSystem",),
    "flext_tests": (
        "c",
        "d",
        "e",
        "h",
        "m",
        "p",
        "r",
        "s",
        "t",
        "td",
        "tf",
        "tk",
        "tm",
        "tv",
        "u",
        "x",
    ),
})


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
