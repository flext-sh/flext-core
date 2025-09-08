"""FLEXT Core Test Support - Comprehensive testing utilities and fixtures.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from .asyncs import AsyncTestUtils, FlextTestsAsync
from .builders import FlextTestsBuilders, TestBuilders
from .domains import (
    FlextTestsDomains,
    PayloadDataFactory,
    ServiceDataFactory,
    SimpleConfigurationFactory,
    UserDataFactory,
)
from .factories import (
    AdminUserFactory,
    BaseTestEntity,
    BaseTestValueObject,
    BatchFactories,
    BooleanFieldFactory,
    ConfigFactory,
    EdgeCaseGenerators,
    FlextResultFactory,
    FloatFieldFactory,
    InactiveUserFactory,
    IntegerFieldFactory,
    ProductionConfigFactory,
    RepositoryError,
    SequenceGenerators,
    StringFieldFactory,
    TestConfig,
    TestEntityFactory,
    TestField,
    TestUser,
    TestValueObjectFactory,
    UserFactory,
    create_validation_test_cases,
    failure_result,
    success_result,
    validation_failure,
)
from .fixtures import (
    BenchmarkFixture,
    FailingUserRepository,
    FlextTestsFixtures,
    InMemoryUserRepository,
    RealAuditService,
    RealEmailService,
)
from .http_support import APITestClient, FlextTestsHttp, HTTPTestUtils
from .hypothesis import (
    CompositeStrategies,
    EdgeCaseStrategies,
    FlextStrategies,
    FlextTestsHypothesis,
    PerformanceStrategies,
    PropertyTestHelpers,
)
from .matchers import FlextMatchers, FlextTestsMatchers
from .performance import (
    BenchmarkProtocol,
    BenchmarkUtils,
    ComplexityAnalyzer,
    FlextTestsPerformance,
    MemoryProfiler,
    PerformanceProfiler,
    StressTestRunner,
)
from .utilities import FlextTestUtilities, FlextTestsUtilities

# =============================================================================
# EXPORTS - Explicit list of all exported items
# =============================================================================

__all__ = [
    # Main module classes
    "FlextTestsAsync",
    "FlextTestsBuilders",
    "FlextTestsDomains",
    "FlextTestsFixtures",
    "FlextTestsHttp",
    "FlextTestsHypothesis",
    "CompositeStrategies",
    "EdgeCaseStrategies",
    "FlextStrategies",
    "PerformanceStrategies",
    "PropertyTestHelpers",
    "FlextTestsMatchers",
    "FlextTestsPerformance",
    "FlextTestsUtilities",
    "BenchmarkProtocol",
    "ComplexityAnalyzer",
    "FlextTestUtilities",
    "BenchmarkFixture",
    # Factory classes
    "AdminUserFactory",
    "UserFactory",
    "InactiveUserFactory",
    "ConfigFactory",
    "ProductionConfigFactory",
    "StringFieldFactory",
    "IntegerFieldFactory",
    "BooleanFieldFactory",
    "FloatFieldFactory",
    "TestEntityFactory",
    "TestValueObjectFactory",
    "FlextResultFactory",
    # Model classes
    "BaseTestEntity",
    "BaseTestValueObject",
    "TestConfig",
    "TestField",
    "TestUser",
    # Utility classes
    "APITestClient",
    "AsyncTestUtils",
    "BatchFactories",
    "BenchmarkUtils",
    "EdgeCaseGenerators",
    "FlextMatchers",
    "HTTPTestUtils",
    "MemoryProfiler",
    "PayloadDataFactory",
    "PerformanceProfiler",
    "SequenceGenerators",
    "ServiceDataFactory",
    "SimpleConfigurationFactory",
    "StressTestRunner",
    "TestBuilders",
    "UserDataFactory",
    # Repository classes
    "FailingUserRepository",
    "InMemoryUserRepository",
    "RealAuditService",
    "RealEmailService",
    # Exception classes
    "RepositoryError",
    # Functions
    "create_validation_test_cases",
    "failure_result",
    "success_result",
    "validation_failure",
]
