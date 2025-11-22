"""Comprehensive tests for FlextUtilities.DataMapper - 100% coverage target.

This module provides real tests (no mocks) for all data mapping functions
in FlextUtilities.DataMapper to achieve 100% code coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

from flext_core import FlextUtilities

# ============================================================================
# Test Map Dict Keys
# ============================================================================


class TestFlextUtilitiesDataMapperMapDictKeys:
    """Test map_dict_keys method."""

    def test_map_dict_keys_basic(self) -> None:
        """Test basic dict key mapping."""
        source: dict[str, object] = {"old_key": "value1", "foo": "value2"}
        mapping: dict[str, str] = {"old_key": "new_key", "foo": "bar"}
        result = FlextUtilities.DataMapper.map_dict_keys(source, mapping)
        assert result.is_success
        assert result.unwrap() == {"new_key": "value1", "bar": "value2"}

    def test_map_dict_keys_keep_unmapped(self) -> None:
        """Test mapping with keep_unmapped=True."""
        source: dict[str, object] = {"old_key": "value1", "unmapped": "value2"}
        mapping: dict[str, str] = {"old_key": "new_key"}
        result = FlextUtilities.DataMapper.map_dict_keys(
            source,
            mapping,
            keep_unmapped=True,
        )
        assert result.is_success
        assert result.unwrap() == {"new_key": "value1", "unmapped": "value2"}

    def test_map_dict_keys_no_keep_unmapped(self) -> None:
        """Test mapping with keep_unmapped=False."""
        source: dict[str, object] = {"old_key": "value1", "unmapped": "value2"}
        mapping: dict[str, str] = {"old_key": "new_key"}
        result = FlextUtilities.DataMapper.map_dict_keys(
            source,
            mapping,
            keep_unmapped=False,
        )
        assert result.is_success
        assert result.unwrap() == {"new_key": "value1"}

    def test_map_dict_keys_exception_handling(self) -> None:
        """Test mapping exception handling."""

        # Create dict that will fail on items()
        class BadDict:
            def items(self) -> list[tuple[str, object]]:
                msg = "Items failed"
                raise RuntimeError(msg)

        bad = BadDict()
        result = FlextUtilities.DataMapper.map_dict_keys(
            cast("dict[str, object]", bad), {}
        )
        assert result.is_failure
        assert result.error is not None
        assert "Failed to map" in result.error


# ============================================================================
# Test Build Flags Dict
# ============================================================================


class TestFlextUtilitiesDataMapperBuildFlagsDict:
    """Test build_flags_dict method."""

    def test_build_flags_dict_basic(self) -> None:
        """Test basic flags dict building."""
        flags = ["read", "write"]
        mapping = {"read": "can_read", "write": "can_write", "delete": "can_delete"}
        result = FlextUtilities.DataMapper.build_flags_dict(flags, mapping)
        assert result.is_success
        flags_dict = result.unwrap()
        assert flags_dict["can_read"] is True
        assert flags_dict["can_write"] is True
        assert flags_dict["can_delete"] is False

    def test_build_flags_dict_custom_default(self) -> None:
        """Test flags dict with custom default value."""
        flags = ["read"]
        mapping = {"read": "can_read", "write": "can_write"}
        result = FlextUtilities.DataMapper.build_flags_dict(
            flags,
            mapping,
            default_value=True,
        )
        assert result.is_success
        flags_dict = result.unwrap()
        assert flags_dict["can_read"] is True
        assert flags_dict["can_write"] is True  # Default is True

    def test_build_flags_dict_exception_handling(self) -> None:
        """Test flags dict exception handling."""

        # Create list that will fail on iteration
        class BadList:
            def __iter__(self) -> object:
                msg = "Iteration failed"
                raise RuntimeError(msg)

        bad = BadList()
        result = FlextUtilities.DataMapper.build_flags_dict(cast("list[str]", bad), {})
        assert result.is_failure
        assert result.error is not None
        assert "Failed to build" in result.error


# ============================================================================
# Test Collect Active Keys
# ============================================================================


class TestFlextUtilitiesDataMapperCollectActiveKeys:
    """Test collect_active_keys method."""

    def test_collect_active_keys_basic(self) -> None:
        """Test basic active keys collection."""
        source = {"read": True, "write": True, "delete": False}
        mapping = {"read": "r", "write": "w", "delete": "d"}
        result = FlextUtilities.DataMapper.collect_active_keys(source, mapping)
        assert result.is_success
        assert result.unwrap() == ["r", "w"]

    def test_collect_active_keys_none_active(self) -> None:
        """Test collection with no active keys."""
        source = {"read": False, "write": False}
        mapping = {"read": "r", "write": "w"}
        result = FlextUtilities.DataMapper.collect_active_keys(source, mapping)
        assert result.is_success
        assert result.unwrap() == []

    def test_collect_active_keys_exception_handling(self) -> None:
        """Test collection exception handling."""

        # Create dict that will fail on get()
        class BadDict:
            def get(self, key: str) -> bool:
                msg = "Get failed"
                raise RuntimeError(msg)

        bad = BadDict()
        result = FlextUtilities.DataMapper.collect_active_keys(
            cast("dict[str, bool]", bad), {"key": "output"}
        )
        assert result.is_failure
        assert result.error is not None
        assert "Failed to collect" in result.error


# ============================================================================
# Test Transform Values
# ============================================================================


class TestFlextUtilitiesDataMapperTransformValues:
    """Test transform_values method."""

    def test_transform_values_basic(self) -> None:
        """Test basic value transformation."""
        source: dict[str, object] = {"a": "hello", "b": "world"}
        result = FlextUtilities.DataMapper.transform_values(
            source,
            lambda v: str(v).upper(),
        )
        assert result == {"a": "HELLO", "b": "WORLD"}

    def test_transform_values_numbers(self) -> None:
        """Test transforming numeric values."""
        source: dict[str, object] = {"a": 1, "b": 2, "c": 3}
        # Transform function needs to handle object type

        def transform_func(v: object) -> object:
            if isinstance(v, int):
                return v * 2
            return v

        result = FlextUtilities.DataMapper.transform_values(source, transform_func)
        assert result == {"a": 2, "b": 4, "c": 6}


# ============================================================================
# Test Filter Dict
# ============================================================================


class TestFlextUtilitiesDataMapperFilterDict:
    """Test filter_dict method."""

    def test_filter_dict_basic(self) -> None:
        """Test basic dict filtering."""
        source: dict[str, object] = {"a": 1, "b": 2, "c": 3}
        # Filter function needs to handle object type

        def filter_func(k: str, v: object) -> bool:
            if isinstance(v, int):
                return v > 1
            return False

        result = FlextUtilities.DataMapper.filter_dict(source, filter_func)
        assert result == {"b": 2, "c": 3}

    def test_filter_dict_by_key(self) -> None:
        """Test filtering by key."""
        source: dict[str, object] = {"a": 1, "b": 2, "c": 3}
        result = FlextUtilities.DataMapper.filter_dict(source, lambda k, v: k != "b")
        assert result == {"a": 1, "c": 3}

    def test_filter_dict_all_filtered(self) -> None:
        """Test filtering that removes all items."""
        source: dict[str, object] = {"a": 1, "b": 2}
        result = FlextUtilities.DataMapper.filter_dict(source, lambda k, v: False)
        assert result == {}


# ============================================================================
# Test Invert Dict
# ============================================================================


class TestFlextUtilitiesDataMapperInvertDict:
    """Test invert_dict method."""

    def test_invert_dict_basic(self) -> None:
        """Test basic dict inversion."""
        source = {"a": "x", "b": "y"}
        result = FlextUtilities.DataMapper.invert_dict(source)
        assert result == {"x": "a", "y": "b"}

    def test_invert_dict_collisions_last(self) -> None:
        """Test inversion with collisions, keep last."""
        source = {"a": "x", "b": "y", "c": "x"}
        result = FlextUtilities.DataMapper.invert_dict(source, handle_collisions="last")
        assert result == {"x": "c", "y": "b"}  # Last "x" kept

    def test_invert_dict_collisions_first(self) -> None:
        """Test inversion with collisions, keep first."""
        source = {"a": "x", "b": "y", "c": "x"}
        result = FlextUtilities.DataMapper.invert_dict(
            source,
            handle_collisions="first",
        )
        assert result == {"x": "a", "y": "b"}  # First "x" kept
