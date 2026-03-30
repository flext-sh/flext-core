# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Test helpers for flext-core - service factories only.

This directory contains ONLY flext-core-specific service factories.
All generic test utilities come from flext_tests directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from tests.helpers import (
        factories as factories,
        factories_impl as factories_impl,
        scenarios as scenarios,
    )
    from tests.helpers.factories import TestHelperFactories as TestHelperFactories
    from tests.helpers.factories_impl import (
        FailingService as FailingService,
        FailingServiceAuto as FailingServiceAuto,
        FailingServiceAutoFactory as FailingServiceAutoFactory,
        FailingServiceFactory as FailingServiceFactory,
        GenericModelFactory as GenericModelFactory,
        GetUserService as GetUserService,
        GetUserServiceAuto as GetUserServiceAuto,
        GetUserServiceAutoFactory as GetUserServiceAutoFactory,
        GetUserServiceFactory as GetUserServiceFactory,
        ServiceFactoryRegistry as ServiceFactoryRegistry,
        ServiceTestCase as ServiceTestCase,
        ServiceTestCaseFactory as ServiceTestCaseFactory,
        ServiceTestCases as ServiceTestCases,
        TestDataGenerators as TestDataGenerators,
        User as User,
        UserFactory as UserFactory,
        ValidatingService as ValidatingService,
        ValidatingServiceAuto as ValidatingServiceAuto,
        ValidatingServiceAutoFactory as ValidatingServiceAutoFactory,
        ValidatingServiceFactory as ValidatingServiceFactory,
        reset_all_factories as reset_all_factories,
    )
    from tests.helpers.scenarios import TestHelperScenarios as TestHelperScenarios

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "FailingService": ["tests.helpers.factories_impl", "FailingService"],
    "FailingServiceAuto": ["tests.helpers.factories_impl", "FailingServiceAuto"],
    "FailingServiceAutoFactory": [
        "tests.helpers.factories_impl",
        "FailingServiceAutoFactory",
    ],
    "FailingServiceFactory": ["tests.helpers.factories_impl", "FailingServiceFactory"],
    "GenericModelFactory": ["tests.helpers.factories_impl", "GenericModelFactory"],
    "GetUserService": ["tests.helpers.factories_impl", "GetUserService"],
    "GetUserServiceAuto": ["tests.helpers.factories_impl", "GetUserServiceAuto"],
    "GetUserServiceAutoFactory": [
        "tests.helpers.factories_impl",
        "GetUserServiceAutoFactory",
    ],
    "GetUserServiceFactory": ["tests.helpers.factories_impl", "GetUserServiceFactory"],
    "ServiceFactoryRegistry": [
        "tests.helpers.factories_impl",
        "ServiceFactoryRegistry",
    ],
    "ServiceTestCase": ["tests.helpers.factories_impl", "ServiceTestCase"],
    "ServiceTestCaseFactory": [
        "tests.helpers.factories_impl",
        "ServiceTestCaseFactory",
    ],
    "ServiceTestCases": ["tests.helpers.factories_impl", "ServiceTestCases"],
    "TestDataGenerators": ["tests.helpers.factories_impl", "TestDataGenerators"],
    "TestHelperFactories": ["tests.helpers.factories", "TestHelperFactories"],
    "TestHelperScenarios": ["tests.helpers.scenarios", "TestHelperScenarios"],
    "User": ["tests.helpers.factories_impl", "User"],
    "UserFactory": ["tests.helpers.factories_impl", "UserFactory"],
    "ValidatingService": ["tests.helpers.factories_impl", "ValidatingService"],
    "ValidatingServiceAuto": ["tests.helpers.factories_impl", "ValidatingServiceAuto"],
    "ValidatingServiceAutoFactory": [
        "tests.helpers.factories_impl",
        "ValidatingServiceAutoFactory",
    ],
    "ValidatingServiceFactory": [
        "tests.helpers.factories_impl",
        "ValidatingServiceFactory",
    ],
    "factories": ["tests.helpers.factories", ""],
    "factories_impl": ["tests.helpers.factories_impl", ""],
    "reset_all_factories": ["tests.helpers.factories_impl", "reset_all_factories"],
    "scenarios": ["tests.helpers.scenarios", ""],
}

_EXPORTS: Sequence[str] = [
    "FailingService",
    "FailingServiceAuto",
    "FailingServiceAutoFactory",
    "FailingServiceFactory",
    "GenericModelFactory",
    "GetUserService",
    "GetUserServiceAuto",
    "GetUserServiceAutoFactory",
    "GetUserServiceFactory",
    "ServiceFactoryRegistry",
    "ServiceTestCase",
    "ServiceTestCaseFactory",
    "ServiceTestCases",
    "TestDataGenerators",
    "TestHelperFactories",
    "TestHelperScenarios",
    "User",
    "UserFactory",
    "ValidatingService",
    "ValidatingServiceAuto",
    "ValidatingServiceAutoFactory",
    "ValidatingServiceFactory",
    "factories",
    "factories_impl",
    "reset_all_factories",
    "scenarios",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
