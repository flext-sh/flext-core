"""Unified test support libraries for flext-core testing.

This package provides centralized test utilities, fixtures, and patterns
following SOLID principles and modern Python testing practices.
"""

from __future__ import annotations

from .asyncs import (
    AsyncConcurrencyTesting,
    AsyncContextManagers,
    AsyncFixtureUtils,
    AsyncMarkers,
    AsyncMockUtils,
    AsyncTestUtils,
)
from .builders import (
    TestBuilders,
    build_failure_result,
    build_string_field,
    build_success_result,
    build_test_container,
)

from .domains import (
    ConfigurationFactory,
    FlextResultFactory,
    PayloadDataFactory,
    ServiceDataFactory,
    UserDataFactory,
)

# HTTP testing utilities (optional - requires pytest-httpx)
try:
    from .http import (
        APITestClient,
        HTTPScenarioBuilder,
        HTTPTestUtils,
        WebhookTestUtils,
    )
except ImportError:
    # Create dummy classes when pytest-httpx is not available
    class APITestClient:
        """Dummy APITestClient when pytest-httpx not available."""
        pass
    
    class HTTPScenarioBuilder:
        """Dummy HTTPScenarioBuilder when pytest-httpx not available."""
        pass
    
    class HTTPTestUtils:
        """Dummy HTTPTestUtils when pytest-httpx not available."""
        pass
    
    class WebhookTestUtils:
        """Dummy WebhookTestUtils when pytest-httpx not available."""
        pass
from .matchers import FlextMatchers
from .performance import (
    AsyncBenchmark,
    BenchmarkUtils,
    MemoryProfiler,
    PerformanceMarkers,
    PerformanceProfiler,
)

# Build __all__ dynamically based on available imports
_all = [
    # Domain factories
    "FlextResultFactory",
    "UserDataFactory",
    "ConfigurationFactory", 
    "PayloadDataFactory",
    "ServiceDataFactory",
    # Core utilities
    "FlextMatchers",
    "TestBuilders",
    # Builder convenience functions
    "build_failure_result",
    "build_string_field",
    "build_success_result",
    "build_test_container",
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
]

# Add HTTP testing if available
try:
    # Check if we have real HTTP classes (not dummies)
    if hasattr(HTTPTestUtils, '__module__') and HTTPTestUtils.__module__ == 'tests.support.http':
        _all.extend([
            "HTTPTestUtils",
            "APITestClient", 
            "HTTPScenarioBuilder",
            "WebhookTestUtils",
        ])
    else:
        # Add dummy classes to __all__ for consistency
        _all.extend([
            "HTTPTestUtils",
            "APITestClient",
            "HTTPScenarioBuilder", 
            "WebhookTestUtils",
        ])
except AttributeError:
    # Add dummy classes to __all__ for consistency
    _all.extend([
        "HTTPTestUtils",
        "APITestClient",
        "HTTPScenarioBuilder",
        "WebhookTestUtils",
    ])

__all__ = _all
