# AUTO-GENERATED FILE — Regenerate with: make gen
"""Tests package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import benchmark as benchmark
    from . import fixtures as fixtures
    from . import integration as integration
    from . import unit as unit
    from flext_core import FlextConstants
    from flext_tests import FlextTestsConstants

    from .base import s
    from .integration.migration_validation_cases import (
        TestsFlextFlextMigrationApplicationCase,
        capture_stdout,
    )
    from .integration.service_fixtures import (
        TestsFlextFlextServiceFixtures,
        TestsFlextLifecycleService,
        TestsFlextNotificationService,
        TestsFlextServiceConfig,
        TestsFlextUserQueryService,
        TestsFlextUserServiceEntity,
    )
    from .integration.service_lifecycle_cases import (
        TestsFlextFlextServiceLifecycleCases,
    )
    from .integration.settings_integration_factories import (
        TestsFlextFlextSettingsFactories,
        TestsFlextSettingsConfigTestCase,
        TestsFlextSettingsConfigTestFactories,
        TestsFlextSettingsThreadSafetyTest,
    )
    from .integration.settings_integration_precedence import (
        TestsFlextFlextSettingsPrecedenceCase,
    )
    from .integration.system_integration_cases import TestsFlextFlextSystemWorkflowCases
    from .models import m
    from .protocols import p
    from .typings import t
    from .utilities import u
__all__: tuple[str, ...] = (
    "FlextConstants",
    "FlextTestsConstants",
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
    "TestsFlextSettingsThreadSafetyTest",
    "TestsFlextUserQueryService",
    "TestsFlextUserServiceEntity",
    "benchmark",
    "capture_stdout",
    "fixtures",
    "integration",
    "m",
    "p",
    "s",
    "t",
    "u",
    "unit",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            ".base": ("s",),
            ".benchmark": ("benchmark",),
            ".fixtures": ("fixtures",),
            ".integration": ("integration",),
            ".integration.migration_validation_cases": (
                "TestsFlextFlextMigrationApplicationCase",
                "capture_stdout",
            ),
            ".integration.service_fixtures": (
                "TestsFlextFlextServiceFixtures",
                "TestsFlextLifecycleService",
                "TestsFlextNotificationService",
                "TestsFlextServiceConfig",
                "TestsFlextUserQueryService",
                "TestsFlextUserServiceEntity",
            ),
            ".integration.service_lifecycle_cases": (
                "TestsFlextFlextServiceLifecycleCases",
            ),
            ".integration.settings_integration_factories": (
                "TestsFlextFlextSettingsFactories",
                "TestsFlextSettingsConfigTestCase",
                "TestsFlextSettingsConfigTestFactories",
                "TestsFlextSettingsThreadSafetyTest",
            ),
            ".integration.settings_integration_precedence": (
                "TestsFlextFlextSettingsPrecedenceCase",
            ),
            ".integration.system_integration_cases": (
                "TestsFlextFlextSystemWorkflowCases",
            ),
            ".models": ("m",),
            ".protocols": ("p",),
            ".typings": ("t",),
            ".unit": ("unit",),
            ".utilities": ("u",),
            "flext_core": ("FlextConstants",),
            "flext_tests": ("FlextTestsConstants",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
