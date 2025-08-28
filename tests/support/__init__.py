"""Unified test support libraries for flext-core testing.

This package provides centralized test utilities, fixtures, and patterns
following SOLID principles and modern Python testing practices.
"""

from __future__ import annotations

from tests.support.async_utils import (
    AsyncConcurrencyTesting,
    AsyncContextManagers,
    AsyncFixtureUtils,
    AsyncMarkers,
    AsyncMockUtils,
    AsyncTestUtils,
)
from tests.support.builders import TestBuilders

from tests.support.domain_factories import (
    ConfigurationFactory,
    FlextResultFactory,
    PayloadDataFactory,
    ServiceDataFactory,
    UserDataFactory,
)
# factory_boy_factories temporarily disabled due to missing dependency
# from tests.support.factory_boy_factories import (
#     AdminUserFactory,
#     ConfigFactory,
#     InactiveUserFactory,
#     ProductionConfigFactory,
#     UserFactory,
# )
from tests.support.fixtures import FlextTestFixtures
from tests.support.http_utils import (
    APITestClient,
    HTTPScenarioBuilder,
    HTTPTestUtils,
    WebhookTestUtils,
)
from tests.support.matchers import FlextMatchers
from tests.support.performance_utils import (
    AsyncBenchmark,
    BenchmarkUtils,
    MemoryProfiler,
    PerformanceMarkers,
    PerformanceProfiler,
)

__all__ = [
    # Domain factories
    "FlextResultFactory",
    "UserDataFactory",
    "ConfigurationFactory",
    "PayloadDataFactory",
    "ServiceDataFactory",
    # Factory Boy factories
    # Factory_boy factories disabled due to missing dependency
    # "UserFactory",
    # "AdminUserFactory",
    # "InactiveUserFactory", 
    # "ConfigFactory",
    # "ProductionConfigFactory",
    # Core utilities
    "FlextTestFixtures",
    "FlextMatchers",
    "TestBuilders",
    # Performance testing
    "PerformanceProfiler",
    "BenchmarkUtils",
    "MemoryProfiler",
    "AsyncBenchmark",
    "PerformanceMarkers",
    # Async testing
    "AsyncTestUtils",
    "AsyncContextManagers",
    "AsyncMockUtils",
    "AsyncFixtureUtils",
    "AsyncConcurrencyTesting",
    "AsyncMarkers",
    # HTTP testing
    "HTTPTestUtils",
    "APITestClient",
    "HTTPScenarioBuilder",
    "WebhookTestUtils",
]
