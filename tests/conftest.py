"""Comprehensive test configuration and utilities for flext-core.

Provides centralized fixtures following FLEXT architecture with c,t,p,m,u pattern.
Extends root conftest.py with flext-core specific fixtures.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

import pytest
from flext_tests import (
    FlextTestsConstants,
    FlextTestsModels,
    FlextTestsProtocols,
    FlextTestsTypes,
    FlextTestsUtilities,
)

from flext_core import FlextSettings


@pytest.fixture(scope="session")
def flext_core_fixtures_dir() -> Path:
    """Path to flext-core test fixtures directory.

    Returns:
        Path to flext-core fixtures directory.

    """
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def flext_settings() -> FlextSettings:
    """Default FlextSettings for testing.

    Returns:
        FlextSettings instance with test defaults.

    """
    return FlextSettings(async_logging=False, debug=False, trace=False)


@pytest.fixture(scope="session")
def flext_constants() -> FlextTestsConstants:
    """flext-core constants facade.

    Returns:
        The FlextTestsConstants facade.

    """
    return FlextTestsConstants


@pytest.fixture(scope="session")
def flext_types() -> FlextTestsTypes:
    """flext-core types facade.

    Returns:
        The FlextTestsTypes facade.

    """
    return FlextTestsTypes


@pytest.fixture(scope="session")
def flext_protocols() -> FlextTestsProtocols:
    """flext-core protocols facade.

    Returns:
        The FlextTestsProtocols facade.

    """
    return FlextTestsProtocols


@pytest.fixture(scope="session")
def flext_models() -> FlextTestsModels:
    """flext-core models facade.

    Returns:
        The FlextTestsModels facade.

    """
    return FlextTestsModels


@pytest.fixture(scope="session")
def flext_utilities() -> FlextTestsUtilities:
    """flext-core utilities facade.

    Returns:
        The FlextTestsUtilities facade.

    """
    return FlextTestsUtilities


collect_ignore_glob = ["**/__init__.py"]
