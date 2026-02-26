"""Coverage stubs for utilities methods tested elsewhere.

Tested modules: Various flext_core._utilities modules
Test scope: Placeholder tests for utilities coverage where methods are tested
in specialized coverage modules (e.g., _coverage_100.py files).

This module provides placeholder tests to ensure coverage tools recognize
the utilities module structure while actual comprehensive tests are in
dedicated coverage files.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core.models import FlextModels as m
from flext_core.utilities import FlextUtilities


class TestUtilitiesCoverage:
    """Placeholder tests for utilities coverage - methods tested in dedicated coverage modules."""

    def test_coverage_placeholder(self) -> None:
        """Placeholder test to ensure module is recognized by coverage tools."""
        assert True


__all__ = ["TestUtilitiesCoverage"]


def test_utilities_get_method_coverage() -> None:
    """Test FlextUtilities.get() method for line 401 coverage."""
    u = FlextUtilities
    test_data = m.ConfigMap(root={"key": "value", "other": 456})

    # Test direct key access
    result = u.get(test_data.root, "key")
    assert result == "value"

    # Test missing key with default
    result = u.get(test_data.root, "missing", default="fallback")
    assert result == "fallback"
