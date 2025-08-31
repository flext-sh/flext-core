"""FLEXT Core Test Support - Comprehensive testing utilities and fixtures.

This module provides the test support foundation for the FLEXT ecosystem with test utilities,
fixtures, builders, matchers, performance testing, and domain-specific test helpers following
modern testing patterns and SOLID principles.

Architecture:
    Foundation Layer: Utilities, matchers, builders
    Domain Layer: Domain factories, test models, test entities
    Application Layer: Fixtures, hypothesis testing, async utilities
    Infrastructure Layer: HTTP testing, performance testing
    Support Layer: Legacy factories, advanced patterns

Core Components:
    AsyncTestUtils: Async testing utilities with context managers and mocking
    TestBuilders: Builder patterns for test data creation and validation
    ConfigurationFactory: Test configuration objects with environment overrides
    FlextMatchers: Custom pytest matchers for FlextResult and domain objects
    HTTPTestUtils: HTTP/API testing utilities with scenario builders
    HypothesisStrategies: Property-based testing strategies for domain objects
    PerformanceProfiler: Performance testing and benchmarking utilities
    TestFactories: Factory patterns for creating test fixtures and data
    TestUtilities: General testing helpers and validation utilities

Examples:
    Test data builders:
    >>> user = TestBuilders.user().with_email("test@example.com").build()
    >>> result = TestBuilders.success_result(user).build()

    Custom matchers:
    >>> assert result | should | be_successful
    >>> assert result | should | contain_value(user)

    Performance testing:
    >>> with PerformanceProfiler() as profiler:
    ...     expensive_operation()
    >>> assert profiler.duration < 1.0

    HTTP testing:
    >>> async with APITestClient() as client:
    ...     response = await client.post("/api/users", json=user_data)
    ...     assert response.status_code == 201

Notes:
    - All test utilities should follow SOLID principles and be composable
    - Use factories for creating consistent test data across test suites
    - Leverage hypothesis for property-based testing of domain logic
    - HTTP utilities require pytest-httpx for full functionality
    - Performance utilities integrate with pytest benchmarking plugins

"""

from __future__ import annotations


from .asyncs import *
from .builders import *
from .domains import *
from .factories import *
from .http_support import *
from .hypothesis import *
from .matchers import *
from .performance import *
from .utilities import *

# =============================================================================
# CONSOLIDATED EXPORTS - Combine all __all__ from modules
# =============================================================================

# Import modules for __all__ collection
from . import asyncs as _asyncs
from . import builders as _builders
from . import domains as _domains
from . import factories as _factories
from . import http_support as _http
from . import hypothesis as _hypothesis
from . import matchers as _matchers
from . import performance as _performance
from . import utilities as _utilities


# Collect all __all__ exports from imported modules
_temp_exports: list[str] = []

_modules_to_check = [
    _asyncs,
    _builders,
    _domains,
    _factories,
    _http,
    _hypothesis,
    _matchers,
    _performance,
    _utilities,
]


for module in _modules_to_check:
    if hasattr(module, "__all__"):
        _temp_exports.extend(module.__all__)

# Remove duplicates and sort for consistent exports - build complete list first
_seen: set[str] = set()
_final_exports: list[str] = []
for item in _temp_exports:
    if item not in _seen:
        _seen.add(item)
        _final_exports.append(item)
_final_exports.sort()

# Define __all__ as literal list for linter compatibility
# This dynamic assignment is necessary for aggregating module exports
__all__: list[str] = _final_exports  # pyright: ignore[reportUnsupportedDunderAll] # noqa: PLE0605
