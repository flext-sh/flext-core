# AUTO-GENERATED FILE — Regenerate with: make gen
"""Integration package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_tests import (
        c as c,
        d as d,
        e as e,
        h as h,
        m as m,
        p as p,
        r as r,
        s as s,
        t as t,
        td as td,
        tf as tf,
        tk as tk,
        tm as tm,
        tv as tv,
        u as u,
        x as x,
    )

    from tests.integration.migration_validation_cases import (
        FlextMigrationApplicationCase as FlextMigrationApplicationCase,
    )
    from tests.integration.service_fixtures import (
        FlextServiceFixtures as FlextServiceFixtures,
        LifecycleService as LifecycleService,
        NotificationService as NotificationService,
        ServiceConfig as ServiceConfig,
        UserQueryService as UserQueryService,
        UserServiceEntity as UserServiceEntity,
    )
    from tests.integration.service_lifecycle_cases import (
        FlextServiceLifecycleCases as FlextServiceLifecycleCases,
    )
    from tests.integration.settings_integration_factories import (
        FlextSettingsFactories as FlextSettingsFactories,
        SettingsConfigTestCase as SettingsConfigTestCase,
        SettingsConfigTestFactories as SettingsConfigTestFactories,
        SettingsThreadSafetyTest as SettingsThreadSafetyTest,
    )
    from tests.integration.settings_integration_precedence import (
        FlextSettingsPrecedenceCase as FlextSettingsPrecedenceCase,
    )
    from tests.integration.system_integration_cases import (
        FlextSystemWorkflowCases as FlextSystemWorkflowCases,
    )
    from tests.integration.test_architecture import (
        TestsFlextAutomatedArchitecture as TestsFlextAutomatedArchitecture,
    )
    from tests.integration.test_documented_patterns import (
        TestsFlextDocumentedPatterns as TestsFlextDocumentedPatterns,
    )
    from tests.integration.test_examples_execution import (
        TestsFlextExamplesExecution as TestsFlextExamplesExecution,
    )
    from tests.integration.test_integration import (
        TestsFlextLibraryIntegration as TestsFlextLibraryIntegration,
    )
    from tests.integration.test_migration_validation import (
        TestsFlextMigrationValidation as TestsFlextMigrationValidation,
    )
    from tests.integration.test_service import (
        TestsFlextServiceIntegration as TestsFlextServiceIntegration,
    )
    from tests.integration.test_settings_integration import (
        TestsFlextSettingsIntegration as TestsFlextSettingsIntegration,
    )
    from tests.integration.test_system import (
        TestsFlextSystemIntegration as TestsFlextSystemIntegration,
    )
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


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
