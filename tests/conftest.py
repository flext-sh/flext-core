"""Comprehensive test configuration and utilities for flext-core.

Provides highly automated testing infrastructure following strict
type-system-architecture.md rules with real functionality testing.
"""

from __future__ import annotations

import pytest
from flext_tests import (
    clean_container as _shared_clean_container,
    reset_settings as _shared_reset_settings,
    sample_data as _shared_sample_data,
    settings as _shared_settings,
    settings_factory as _shared_settings_factory,
    temp_dir as _shared_temp_dir,
    temp_file as _shared_temp_file,
    test_context as _shared_test_context,
    test_runtime as _shared_test_runtime,
)

from tests.utilities import u

clean_container = _shared_clean_container
reset_settings = _shared_reset_settings
sample_data = _shared_sample_data
settings = _shared_settings
settings_factory = _shared_settings_factory
temp_dir = _shared_temp_dir
temp_file = _shared_temp_file
test_context = _shared_test_context
test_runtime = _shared_test_runtime

collect_ignore_glob = [
    "**/__init__.py",
]


@pytest.fixture
def mock_external_service() -> u.Tests.FunctionalExternalService:
    """Provide mock external service for integration tests."""
    return u.Tests.FunctionalExternalService()
