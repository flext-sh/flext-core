"""Comprehensive test configuration and utilities for flext-core.

Provides highly automated testing infrastructure following strict
type-system-architecture.md rules with real functionality testing.
"""

from __future__ import annotations

import pytest

from tests import u

# NOTE (multi-agent): Shared fixtures come only from flext-tests' pytest11 plugin.

collect_ignore_glob = [
    "**/__init__.py",
]


@pytest.fixture
def mock_external_service() -> u.Tests.FunctionalExternalService:
    """Provide mock external service for integration tests."""
    return u.Tests.FunctionalExternalService()
