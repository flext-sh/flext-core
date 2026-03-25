"""Comprehensive tests for FlextUtilities - 100% coverage target.

Tested modules: flext_core._utilities.mapper
Test scope: Data mapping utilities for dict key mapping, flags building, active keys
collection, value transformation, dict filtering, and inversion with full edge case coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict, UserList
from collections.abc import ItemsView, Iterator, Mapping
from typing import cast, override

from flext_core import FlextTypes, FlextUtilities
from flext_tests import tm
from tests import (
    assertion_helpers,
    c,
    t,
)


class TestUtilitiesDataMapper:
    class _BadDict(UserDict[str, FlextTypes.Container]):
        @override
        def items(self) -> ItemsView[str, FlextTypes.Container]:
            msg = "Bad dict items"
            raise RuntimeError(msg)

    class _BadList(UserList[str]):
        @override
        def __iter__(self) -> Iterator[str]:
            msg = "Bad list iteration"
            raise RuntimeError(msg)

    class _BadDictGet:
        def get(self, key: str, default: bool | None = None) -> bool:
            _ = (key, default)
            msg = "Bad dict get"
            raise RuntimeError(msg)

    def test_basic_key_mapping(self) -> None:
        mc = c.Core.Mapper
        source_raw = {mc.OLD_KEY: mc.VALUE1, mc.FOO: mc.VALUE2}
        mapping = {mc.OLD_KEY: mc.NEW_KEY, mc.FOO: mc.BAR}
        result = FlextUtilities.map_dict_keys(source_raw, mapping)
        mapped = assertion_helpers.assert_flext_result_success(result)
        assert mapped == {mc.NEW_KEY: mc.VALUE1, mc.BAR: mc.VALUE2}

    def test_keep_unmapped_true(self) -> None:
        mc = c.Core.Mapper
        source_raw = {mc.OLD_KEY: mc.VALUE1, mc.UNMAPPED: mc.VALUE2}
        mapping = {mc.OLD_KEY: mc.NEW_KEY}
        result = FlextUtilities.map_dict_keys(
            source_raw,
            mapping,
            keep_unmapped=True,
        )
        mapped = assertion_helpers.assert_flext_result_success(result)
        assert mapped == {mc.NEW_KEY: mc.VALUE1, mc.UNMAPPED: mc.VALUE2}

    def test_keep_unmapped_false(self) -> None:
        mc = c.Core.Mapper
        source_raw = {mc.OLD_KEY: mc.VALUE1, mc.UNMAPPED: mc.VALUE2}
        mapping = {mc.OLD_KEY: mc.NEW_KEY}
        result = FlextUtilities.map_dict_keys(
            source_raw,
            mapping,
            keep_unmapped=False,
        )
        mapped = assertion_helpers.assert_flext_result_success(result)
        assert mapped == {mc.NEW_KEY: mc.VALUE1}

    def test_map_dict_keys_exception_handling(self) -> None:
        bad_dict_instance = self._BadDict()
        result = FlextUtilities.map_dict_keys(bad_dict_instance, {})
        tm.fail(result, contains="Failed to map dict keys")

    def test_basic_flags_building(self) -> None:
        mc = c.Core.Mapper
        flags = [mc.FLAGS_READ, mc.FLAGS_WRITE]
        mapping = {
            mc.FLAGS_READ: mc.CAN_READ,
            mc.FLAGS_WRITE: mc.CAN_WRITE,
            mc.FLAGS_DELETE: mc.CAN_DELETE,
        }
        result = FlextUtilities.build_flags_dict(flags, mapping)
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == {
            mc.CAN_READ: True,
            mc.CAN_WRITE: True,
            mc.CAN_DELETE: False,
        }

    def test_custom_default_value(self) -> None:
        mc = c.Core.Mapper
        flags = [mc.FLAGS_READ]
        mapping = {mc.FLAGS_READ: mc.CAN_READ, mc.FLAGS_WRITE: mc.CAN_WRITE}
        result = FlextUtilities.build_flags_dict(
            flags,
            mapping,
            default_value=True,
        )
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == {mc.CAN_READ: True, mc.CAN_WRITE: True}

    def test_build_flags_dict_exception_handling(self) -> None:
        bad_list_instance = self._BadList()
        bad_list_typed: t.StrSequence = cast("t.StrSequence", bad_list_instance)
        result = FlextUtilities.build_flags_dict(bad_list_typed, {})
        _ = assertion_helpers.assert_flext_result_failure(result)
        assert "Failed to build flags dict" in str(result.error)

    def test_basic_active_keys(self) -> None:
        mc = c.Core.Mapper
        source = {mc.FLAGS_READ: True, mc.FLAGS_WRITE: True, mc.FLAGS_DELETE: False}
        mapping = {mc.FLAGS_READ: "r", mc.FLAGS_WRITE: "w", mc.FLAGS_DELETE: "d"}
        result = FlextUtilities.collect_active_keys(source, mapping)
        _ = assertion_helpers.assert_flext_result_success(result)
        assert set(result.value) == {"r", "w"}

    def test_none_active(self) -> None:
        mc = c.Core.Mapper
        source = {mc.FLAGS_READ: False, mc.FLAGS_WRITE: False}
        mapping = {mc.FLAGS_READ: "r", mc.FLAGS_WRITE: "w"}
        result = FlextUtilities.collect_active_keys(source, mapping)
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == []

    def test_collect_active_keys_exception_handling(self) -> None:
        result = FlextUtilities.collect_active_keys(
            cast("Mapping[str, bool]", self._BadDictGet()),
            {"key": "output"},
        )
        _ = assertion_helpers.assert_flext_result_failure(result)
        assert "Failed to collect active keys" in str(result.error)

    def test_basic_transform(self) -> None:
        mc = c.Core.Mapper
        source_raw = {mc.A: mc.HELLO, mc.B: mc.WORLD}
        result = FlextUtilities.transform_values(
            source_raw,
            lambda v: str(v).upper(),
        )
        assert result == {mc.A: mc.HELLO_UPPER, mc.B: mc.WORLD_UPPER}

    def test_numeric_transform(self) -> None:
        mc = c.Core.Mapper
        source_raw = {mc.A: mc.NUM_1, mc.B: mc.NUM_2, mc.C: mc.NUM_3}
        result = FlextUtilities.transform_values(
            source_raw,
            lambda v: v * 2 if isinstance(v, int) else v,
        )
        assert result == {mc.A: 2, mc.B: 4, mc.C: 6}

    def test_basic_filter(self) -> None:
        mc = c.Core.Mapper
        source_raw = {mc.A: mc.NUM_1, mc.B: mc.NUM_2, mc.C: mc.NUM_3}
        result = FlextUtilities.filter_dict(
            source_raw,
            lambda k, v: isinstance(v, int) and v > mc.NUM_1,
        )
        assert result == {mc.B: mc.NUM_2, mc.C: mc.NUM_3}

    def test_basic_invert(self) -> None:
        mc = c.Core.Mapper
        source = {mc.X: mc.Y, mc.A: mc.B}
        result = FlextUtilities.invert_dict(source)
        assert result == {mc.Y: mc.X, mc.B: mc.A}

    def test_collision_handling_last(self) -> None:
        mc = c.Core.Mapper
        source = {mc.A: mc.B, mc.X: mc.B}
        result = FlextUtilities.invert_dict(source, handle_collisions="last")
        assert result == {mc.B: mc.X}
