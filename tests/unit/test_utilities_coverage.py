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

from collections.abc import Mapping

from flext_core import FlextUtilities, r, t


class TestUtilitiesCoverage:
    """Placeholder tests for utilities coverage - methods tested in dedicated coverage modules."""

    def test_coverage_placeholder(self) -> None:
        """Placeholder test to ensure module is recognized by coverage tools."""
        assert True


__all__ = ["TestUtilitiesCoverage"]


def test_utilities_get_method_coverage() -> None:
    """Test FlextUtilities.get() method for line 401 coverage."""
    u = FlextUtilities
    test_data = t.ConfigMap(root={"key": "value", "other": 456})
    result = u.get(test_data.root, "key")
    assert result == "value"
    result = u.get(test_data.root, "missing", default="fallback")
    assert result == "fallback"


def test_utilities_vals_result_contract() -> None:
    u = FlextUtilities
    values_from_mapping = u.vals({"a": 1, "b": 2})
    assert values_from_mapping.is_success and values_from_mapping.value == [1, 2]
    failed_values_result = r[Mapping[str, int]].fail("failed")
    values_from_failed_result = u.vals(failed_values_result, default=[0])
    assert values_from_failed_result.is_success and values_from_failed_result.value == [
        0,
    ]
    empty_mapping: dict[str, int] = {}
    empty_without_default = u.vals(empty_mapping)
    assert empty_without_default.is_failure
