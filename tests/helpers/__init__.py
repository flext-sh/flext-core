"""Test helpers for flext-core - service factories only.

This directory contains ONLY flext-core-specific service factories.
All generic test utilities come from flext_tests directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from tests.helpers.factories import (
    FailingService,
    FailingServiceAuto,
    FailingServiceAutoFactory,
    FailingServiceFactory,
    GetUserService,
    GetUserServiceAuto,
    GetUserServiceAutoFactory,
    GetUserServiceFactory,
    ServiceFactoryRegistry,
    ServiceTestCase,
    ServiceTestCaseFactory,
    ServiceTestCases,
    ServiceTestType,
    TestDataGenerators,
    User,
    UserFactory,
    ValidatingService,
    ValidatingServiceAuto,
    ValidatingServiceAutoFactory,
    ValidatingServiceFactory,
    reset_all_factories,
)

__all__ = [
    "FailingService",
    "FailingServiceAuto",
    "FailingServiceAutoFactory",
    "FailingServiceFactory",
    "GetUserService",
    "GetUserServiceAuto",
    "GetUserServiceAutoFactory",
    "GetUserServiceFactory",
    "ServiceFactoryRegistry",
    "ServiceTestCase",
    "ServiceTestCaseFactory",
    "ServiceTestCases",
    "ServiceTestType",
    "TestDataGenerators",
    "User",
    "UserFactory",
    "ValidatingService",
    "ValidatingServiceAuto",
    "ValidatingServiceAutoFactory",
    "ValidatingServiceFactory",
    "reset_all_factories",
]
