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
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
    from tests.helpers._scenarios_impl import *
    from tests.helpers.factories import *
    from tests.helpers.factories_impl import *
    from tests.helpers.scenarios import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
    "FailingService": "tests.helpers.factories_impl",
    "FailingServiceAuto": "tests.helpers.factories_impl",
    "FailingServiceAutoFactory": "tests.helpers.factories_impl",
    "FailingServiceFactory": "tests.helpers.factories_impl",
    "GenericModelFactory": "tests.helpers.factories_impl",
    "GetUserService": "tests.helpers.factories_impl",
    "GetUserServiceAuto": "tests.helpers.factories_impl",
    "GetUserServiceAutoFactory": "tests.helpers.factories_impl",
    "GetUserServiceFactory": "tests.helpers.factories_impl",
    "ParserScenario": "tests.helpers._scenarios_impl",
    "ParserScenarios": "tests.helpers._scenarios_impl",
    "ReliabilityScenario": "tests.helpers._scenarios_impl",
    "ReliabilityScenarios": "tests.helpers._scenarios_impl",
    "ServiceFactoryRegistry": "tests.helpers.factories_impl",
    "ServiceTestCaseFactory": "tests.helpers.factories_impl",
    "ServiceTestCases": "tests.helpers.factories_impl",
    "TestDataGenerators": "tests.helpers.factories_impl",
    "TestHelperFactories": "tests.helpers.factories",
    "TestHelperScenarios": "tests.helpers.scenarios",
    "UserFactory": "tests.helpers.factories_impl",
    "ValidatingService": "tests.helpers.factories_impl",
    "ValidatingServiceAuto": "tests.helpers.factories_impl",
    "ValidatingServiceAutoFactory": "tests.helpers.factories_impl",
    "ValidatingServiceFactory": "tests.helpers.factories_impl",
    "ValidationScenario": "tests.helpers._scenarios_impl",
    "ValidationScenarios": "tests.helpers._scenarios_impl",
    "_scenarios_impl": "tests.helpers._scenarios_impl",
    "factories": "tests.helpers.factories",
    "factories_impl": "tests.helpers.factories_impl",
    "reset_all_factories": "tests.helpers.factories_impl",
    "scenarios": "tests.helpers.scenarios",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
