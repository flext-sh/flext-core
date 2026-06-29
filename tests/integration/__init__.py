# AUTO-GENERATED FILE — Regenerate with: make gen
"""Integration package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".migration_validation_cases": ("FlextMigrationApplicationCase",),
        ".service_fixtures": (
            "FlextServiceFixtures",
            "LifecycleService",
            "NotificationService",
            "ServiceConfig",
            "UserQueryService",
            "UserServiceEntity",
        ),
        ".service_lifecycle_cases": ("FlextServiceLifecycleCases",),
        ".settings_integration_factories": (
            "FlextSettingsFactories",
            "SettingsConfigTestCase",
            "SettingsConfigTestFactories",
            "SettingsThreadSafetyTest",
        ),
        ".settings_integration_precedence": ("FlextSettingsPrecedenceCase",),
        ".system_integration_cases": ("FlextSystemWorkflowCases",),
        ".test_architecture": ("TestsFlextAutomatedArchitecture",),
        ".test_documented_patterns": ("TestsFlextDocumentedPatterns",),
        ".test_examples_execution": ("TestsFlextExamplesExecution",),
        ".test_integration": ("TestsFlextLibraryIntegration",),
        ".test_migration_validation": ("TestsFlextMigrationValidation",),
        ".test_service": ("TestsFlextServiceIntegration",),
        ".test_settings_integration": ("TestsFlextSettingsIntegration",),
        ".test_system": ("TestsFlextSystemIntegration",),
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
    },
)


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, publish_all=False)
