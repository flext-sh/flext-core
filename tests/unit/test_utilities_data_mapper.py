"""Comprehensive tests for FlextUtilities.Mapper - 100% coverage target.

Tested modules: flext_core._utilities.mapper
Test scope: Data mapping utilities for dict key mapping, flags building, active keys
collection, value transformation, dict filtering, and inversion with full edge case coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import Iterator

from flext_core.typings import FlextTypes
from flext_core.utilities import FlextUtilities
from flext_tests import tm
from tests.constants import TestsFlextConstants


class TestMapperMapDictKeys:
    """Tests for FlextUtilities.Mapper.map_dict_keys."""

    def test_basic_key_mapping(self) -> None:
        """Test basic key mapping."""
        mc = TestsFlextConstants.Mapper
        source = {mc.OLD_KEY: mc.VALUE1, mc.FOO: mc.VALUE2}
        mapping = {mc.OLD_KEY: mc.NEW_KEY, mc.FOO: mc.BAR}

        result = FlextUtilities.Mapper.map_dict_keys(source, mapping)

        tm.ok(result, eq={mc.NEW_KEY: mc.VALUE1, mc.BAR: mc.VALUE2})

    def test_keep_unmapped_true(self) -> None:
        """Test keeping unmapped keys."""
        mc = TestsFlextConstants.Mapper
        source = {mc.OLD_KEY: mc.VALUE1, mc.UNMAPPED: mc.VALUE2}
        mapping = {mc.OLD_KEY: mc.NEW_KEY}

        result = FlextUtilities.Mapper.map_dict_keys(
            source,
            mapping,
            keep_unmapped=True,
        )

        tm.ok(result, eq={mc.NEW_KEY: mc.VALUE1, mc.UNMAPPED: mc.VALUE2})

    def test_keep_unmapped_false(self) -> None:
        """Test discarding unmapped keys."""
        mc = TestsFlextConstants.Mapper
        source = {mc.OLD_KEY: mc.VALUE1, mc.UNMAPPED: mc.VALUE2}
        mapping = {mc.OLD_KEY: mc.NEW_KEY}

        result = FlextUtilities.Mapper.map_dict_keys(
            source,
            mapping,
            keep_unmapped=False,
        )

        tm.ok(result, eq={mc.NEW_KEY: mc.VALUE1})

    def test_exception_handling(self) -> None:
        """Test exception handling with bad dict."""

        class BadDict(UserDict[str, FlextTypes.GeneralValueType]):
            """Dict that raises on items()."""

            def items(self) -> FlextTypes.GeneralValueType:
                """Raise error on items attempt."""
                msg = "Bad dict items"
                raise RuntimeError(msg)

        result = FlextUtilities.Mapper.map_dict_keys(BadDict(), {})

        tm.fail(result, contains="Failed to map dict keys")


class TestMapperBuildFlagsDict:
    """Tests for FlextUtilities.Mapper.build_flags_dict."""

    def test_basic_flags_building(self) -> None:
        """Test basic flags dict building."""
        mc = TestsFlextConstants.Mapper
        flags = [mc.FLAGS_READ, mc.FLAGS_WRITE]
        mapping = {
            mc.FLAGS_READ: mc.CAN_READ,
            mc.FLAGS_WRITE: mc.CAN_WRITE,
            mc.FLAGS_DELETE: mc.CAN_DELETE,
        }

        result = FlextUtilities.Mapper.build_flags_dict(flags, mapping)

        assert result.is_success
        assert result.value == {
            mc.CAN_READ: True,
            mc.CAN_WRITE: True,
            mc.CAN_DELETE: False,
        }

    def test_custom_default_value(self) -> None:
        """Test custom default value for inactive flags."""
        mc = TestsFlextConstants.Mapper
        flags = [mc.FLAGS_READ]
        mapping = {mc.FLAGS_READ: mc.CAN_READ, mc.FLAGS_WRITE: mc.CAN_WRITE}

        result = FlextUtilities.Mapper.build_flags_dict(
            flags,
            mapping,
            default_value=True,
        )

        assert result.is_success
        # When default_value=True, unset flags start True and active flags become True
        assert result.value == {mc.CAN_READ: True, mc.CAN_WRITE: True}

    def test_exception_handling(self) -> None:
        """Test exception handling with bad list."""

        class BadList(UserList[str]):
            """List that raises on iteration."""

            def __iter__(self) -> Iterator[str]:
                """Raise error on iteration."""
                msg = "Bad list iteration"
                raise RuntimeError(msg)

        result = FlextUtilities.Mapper.build_flags_dict(BadList(), {})

        assert result.is_failure
        assert "Failed to build flags dict" in str(result.error)


class TestMapperCollectActiveKeys:
    """Tests for FlextUtilities.Mapper.collect_active_keys."""

    def test_basic_active_keys(self) -> None:
        """Test collecting active keys."""
        mc = TestsFlextConstants.Mapper
        source = {
            mc.FLAGS_READ: True,
            mc.FLAGS_WRITE: True,
            mc.FLAGS_DELETE: False,
        }
        mapping = {
            mc.FLAGS_READ: "r",
            mc.FLAGS_WRITE: "w",
            mc.FLAGS_DELETE: "d",
        }

        result = FlextUtilities.Mapper.collect_active_keys(source, mapping)

        assert result.is_success
        assert set(result.value) == {"r", "w"}

    def test_none_active(self) -> None:
        """Test when no keys are active."""
        mc = TestsFlextConstants.Mapper
        source = {mc.FLAGS_READ: False, mc.FLAGS_WRITE: False}
        mapping = {mc.FLAGS_READ: "r", mc.FLAGS_WRITE: "w"}

        result = FlextUtilities.Mapper.collect_active_keys(source, mapping)

        assert result.is_success
        assert result.value == []

    def test_exception_handling(self) -> None:
        """Test exception handling with bad dict."""

        class BadDictGet(UserDict[str, bool]):
            """Dict that raises on get()."""

            def get(self, key: str, default: bool | None = None) -> bool:
                """Raise error on get attempt."""
                msg = "Bad dict get"
                raise RuntimeError(msg)

        result = FlextUtilities.Mapper.collect_active_keys(
            BadDictGet(),
            {"key": "output"},
        )

        assert result.is_failure
        assert "Failed to collect active keys" in str(result.error)


class TestMapperTransformValues:
    """Tests for FlextUtilities.Mapper.transform_values."""

    def test_basic_transform(self) -> None:
        """Test basic value transformation."""
        mc = TestsFlextConstants.Mapper
        source = {mc.A: mc.HELLO, mc.B: mc.WORLD}

        result = FlextUtilities.Mapper.transform_values(
            source,
            lambda v: str(v).upper(),
        )

        assert result == {mc.A: mc.HELLO_UPPER, mc.B: mc.WORLD_UPPER}

    def test_numeric_transform(self) -> None:
        """Test numeric value transformation."""
        mc = TestsFlextConstants.Mapper
        source = {mc.A: mc.NUM_1, mc.B: mc.NUM_2, mc.C: mc.NUM_3}

        result = FlextUtilities.Mapper.transform_values(
            source,
            lambda v: v * 2 if isinstance(v, int) else v,
        )

        assert result == {mc.A: 2, mc.B: 4, mc.C: 6}


class TestMapperFilterDict:
    """Tests for FlextUtilities.Mapper.filter_dict."""

    def test_basic_filter(self) -> None:
        """Test basic dict filtering."""
        mc = TestsFlextConstants.Mapper
        source = {mc.A: mc.NUM_1, mc.B: mc.NUM_2, mc.C: mc.NUM_3}

        result = FlextUtilities.Mapper.filter_dict(
            source,
            lambda k, v: isinstance(v, int) and v > mc.NUM_1,
        )

        assert result == {mc.B: mc.NUM_2, mc.C: mc.NUM_3}


class TestMapperInvertDict:
    """Tests for FlextUtilities.Mapper.invert_dict."""

    def test_basic_invert(self) -> None:
        """Test basic dict inversion."""
        mc = TestsFlextConstants.Mapper
        source = {mc.X: mc.Y, mc.A: mc.B}

        result = FlextUtilities.Mapper.invert_dict(source)

        assert result == {mc.Y: mc.X, mc.B: mc.A}

    def test_collision_handling_last(self) -> None:
        """Test collision handling with 'last' strategy."""
        mc = TestsFlextConstants.Mapper
        source = {mc.A: mc.B, mc.X: mc.B}

        result = FlextUtilities.Mapper.invert_dict(source, handle_collisions="last")

        assert result == {mc.B: mc.X}
