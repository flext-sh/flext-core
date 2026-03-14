# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make codegen
#
"""Test helpers for flext-core - service factories only.

This directory contains ONLY flext-core-specific service factories.
All generic test utilities come from flext_tests directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import cleanup_submodule_namespace, lazy_getattr

if TYPE_CHECKING:
    from tests.helpers.factories import (
        FailingService,
        FailingServiceAuto,
        FailingServiceAutoFactory,
        FailingServiceFactory,
        GenericModelFactory,
        GetUserService,
        GetUserService as s,
        GetUserServiceAuto,
        GetUserServiceAutoFactory,
        GetUserServiceFactory,
        ServiceFactoryRegistry,
        ServiceTestCase,
        ServiceTestCaseFactory,
        ServiceTestCases,
        TestDataGenerators,
        User,
        UserFactory,
        ValidatingService,
        ValidatingServiceAuto,
        ValidatingServiceAutoFactory,
        ValidatingServiceFactory,
        reset_all_factories,
    )
    from tests.helpers.scenarios import (
        ParserScenario,
        ParserScenarios,
        ReliabilityScenario,
        ReliabilityScenarios,
        ValidationScenario,
        ValidationScenarios,
    )

# Lazy import mapping: export_name -> (module_path, attr_name)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "FailingService": ("tests.helpers.factories", "FailingService"),
    "FailingServiceAuto": ("tests.helpers.factories", "FailingServiceAuto"),
    "FailingServiceAutoFactory": (
        "tests.helpers.factories",
        "FailingServiceAutoFactory",
    ),
    "FailingServiceFactory": ("tests.helpers.factories", "FailingServiceFactory"),
    "GenericModelFactory": ("tests.helpers.factories", "GenericModelFactory"),
    "GetUserService": ("tests.helpers.factories", "GetUserService"),
    "GetUserServiceAuto": ("tests.helpers.factories", "GetUserServiceAuto"),
    "GetUserServiceAutoFactory": (
        "tests.helpers.factories",
        "GetUserServiceAutoFactory",
    ),
    "GetUserServiceFactory": ("tests.helpers.factories", "GetUserServiceFactory"),
    "ParserScenario": ("tests.helpers.scenarios", "ParserScenario"),
    "ParserScenarios": ("tests.helpers.scenarios", "ParserScenarios"),
    "ReliabilityScenario": ("tests.helpers.scenarios", "ReliabilityScenario"),
    "ReliabilityScenarios": ("tests.helpers.scenarios", "ReliabilityScenarios"),
    "ServiceFactoryRegistry": ("tests.helpers.factories", "ServiceFactoryRegistry"),
    "ServiceTestCase": ("tests.helpers.factories", "ServiceTestCase"),
    "ServiceTestCaseFactory": ("tests.helpers.factories", "ServiceTestCaseFactory"),
    "ServiceTestCases": ("tests.helpers.factories", "ServiceTestCases"),
    "TestDataGenerators": ("tests.helpers.factories", "TestDataGenerators"),
    "User": ("tests.helpers.factories", "User"),
    "UserFactory": ("tests.helpers.factories", "UserFactory"),
    "ValidatingService": ("tests.helpers.factories", "ValidatingService"),
    "ValidatingServiceAuto": ("tests.helpers.factories", "ValidatingServiceAuto"),
    "ValidatingServiceAutoFactory": (
        "tests.helpers.factories",
        "ValidatingServiceAutoFactory",
    ),
    "ValidatingServiceFactory": ("tests.helpers.factories", "ValidatingServiceFactory"),
    "ValidationScenario": ("tests.helpers.scenarios", "ValidationScenario"),
    "ValidationScenarios": ("tests.helpers.scenarios", "ValidationScenarios"),
    "reset_all_factories": ("tests.helpers.factories", "reset_all_factories"),
    "s": ("tests.helpers.factories", "GetUserService"),
}

__all__ = [
    "FailingService",
    "FailingServiceAuto",
    "FailingServiceAutoFactory",
    "FailingServiceFactory",
    "GenericModelFactory",
    "GetUserService",
    "GetUserServiceAuto",
    "GetUserServiceAutoFactory",
    "GetUserServiceFactory",
    "ParserScenario",
    "ParserScenarios",
    "ReliabilityScenario",
    "ReliabilityScenarios",
    "ServiceFactoryRegistry",
    "ServiceTestCase",
    "ServiceTestCaseFactory",
    "ServiceTestCases",
    "TestDataGenerators",
    "User",
    "UserFactory",
    "ValidatingService",
    "ValidatingServiceAuto",
    "ValidatingServiceAutoFactory",
    "ValidatingServiceFactory",
    "ValidationScenario",
    "ValidationScenarios",
    "reset_all_factories",
    "s",
]


def __getattr__(name: str) -> t.ModuleExport:
    """Lazy-load module attributes on first access (PEP 562)."""
    return lazy_getattr(name, _LAZY_IMPORTS, globals(), __name__)


def __dir__() -> list[str]:
    """Return list of available attributes for dir() and autocomplete."""
    return sorted(__all__)


cleanup_submodule_namespace(__name__, _LAZY_IMPORTS)
