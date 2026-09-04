# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests.integration package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_tests import c, d, e, h, m, p, r, s, t, td, tf, tk, tm, tv, u, x

    from .migration_validation_cases import (
        TestsFlextFlextMigrationApplicationCase,
        capture_stdout,
    )
    from .service_fixtures import (
        TestsFlextFlextServiceFixtures,
        TestsFlextLifecycleService,
        TestsFlextNotificationService,
        TestsFlextServiceConfig,
        TestsFlextUserQueryService,
        TestsFlextUserServiceEntity,
    )
    from .service_lifecycle_cases import TestsFlextFlextServiceLifecycleCases
    from .settings_integration_factories import (
        TestsFlextFlextSettingsFactories,
        TestsFlextSettingsConfigTestCase,
        TestsFlextSettingsConfigTestFactories,
        TestsFlextSettingsThreadSafetyTest,
    )
    from .settings_integration_precedence import TestsFlextFlextSettingsPrecedenceCase
    from .system_integration_cases import TestsFlextFlextSystemWorkflowCases
    from .test_architecture import TestsFlextCoreArchitecture
    from .test_documented_patterns import TestsFlextCoreDocumentedPatterns
    from .test_examples_execution import TestsFlextExamplesExecution
    from .test_integration import TestsFlextCoreIntegration
    from .test_migration_validation import TestsFlextCoreMigrationValidation
    from .test_service import TestsFlextCoreService
    from .test_settings_integration import TestsFlextSettingsIntegration
    from .test_system import TestsFlextCoreSystem
__all__: tuple[str, ...] = (
    "TestsFlextCoreArchitecture",
    "TestsFlextCoreDocumentedPatterns",
    "TestsFlextCoreIntegration",
    "TestsFlextCoreMigrationValidation",
    "TestsFlextCoreService",
    "TestsFlextCoreSystem",
    "TestsFlextExamplesExecution",
    "TestsFlextFlextMigrationApplicationCase",
    "TestsFlextFlextServiceFixtures",
    "TestsFlextFlextServiceLifecycleCases",
    "TestsFlextFlextSettingsFactories",
    "TestsFlextFlextSettingsPrecedenceCase",
    "TestsFlextFlextSystemWorkflowCases",
    "TestsFlextLifecycleService",
    "TestsFlextNotificationService",
    "TestsFlextServiceConfig",
    "TestsFlextSettingsConfigTestCase",
    "TestsFlextSettingsConfigTestFactories",
    "TestsFlextSettingsIntegration",
    "TestsFlextSettingsThreadSafetyTest",
    "TestsFlextUserQueryService",
    "TestsFlextUserServiceEntity",
    "c",
    "capture_stdout",
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
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".migration_validation_cases": (
                "TestsFlextFlextMigrationApplicationCase",
                "capture_stdout",
            ),
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
            ".settings_integration_precedence": (
                "TestsFlextFlextSettingsPrecedenceCase",
            ),
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
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
