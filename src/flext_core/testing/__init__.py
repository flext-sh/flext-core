"""Testing utilities for FLEXT framework.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides common testing utilities and fixtures that can be shared
across all FLEXT projects to maintain consistency and reduce duplication.
"""

from __future__ import annotations

from .base import AsyncTestCase
from .base import BaseTestCase
from .fixtures import DatabaseFixtures
from .fixtures import MemoryFixtures
from .fixtures import TestFixtures
from .fixtures import (
    TestFixtures as FlextTestFixtures,  # Alias for backward compatibility
)
from .fixtures import get_project_root_fixture
from .fixtures import get_test_environment_fixture
from .fixtures import setup_flext_test_environment
from .mocks import MockConfig
from .mocks import MockLogger
from .mocks import MockRepository

__all__ = [
    "AsyncTestCase",
    "BaseTestCase",
    "DatabaseFixtures",
    "FlextTestFixtures",  # Alias for backward compatibility
    "MemoryFixtures",
    "MockConfig",
    "MockLogger",
    "MockRepository",
    "TestFixtures",
    # Centralized test environment setup - eliminates conftest.py duplication
    "get_project_root_fixture",
    "get_test_environment_fixture",
    "setup_flext_test_environment",
]
