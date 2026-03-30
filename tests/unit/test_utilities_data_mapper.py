"""Comprehensive tests for FlextUtilities - 100% coverage target.

Tested modules: flext_core._utilities.mapper
Test scope: Data mapping utilities for dict key mapping, flags building, active keys
collection, value transformation, dict filtering, and inversion with full edge case coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict
from collections.abc import ItemsView
from typing import override

from flext_tests import tm

from tests import (
    assertion_helpers,
    c,
    t,
    u,
)


class TestUtilitiesDataMapper:
    class _BadDict(UserDict[str, t.Container]):
        @override
        def items(self) -> ItemsView[str, t.Container]:
            msg = "Bad dict items"
            raise RuntimeError(msg)

    def test_basic_key_mapping(self) -> None:
        mc = c.Core.Mapper
        source_raw = {mc.OLD_KEY: mc.VALUE1, mc.FOO: mc.VALUE2}
        mapping = {mc.OLD_KEY: mc.NEW_KEY, mc.FOO: mc.BAR}
        result = u.map_dict_keys(source_raw, mapping)
        mapped = assertion_helpers.assert_flext_result_success(result)
        assert mapped == {mc.NEW_KEY: mc.VALUE1, mc.BAR: mc.VALUE2}

    def test_keep_unmapped_true(self) -> None:
        mc = c.Core.Mapper
        source_raw = {mc.OLD_KEY: mc.VALUE1, mc.UNMAPPED: mc.VALUE2}
        mapping = {mc.OLD_KEY: mc.NEW_KEY}
        result = u.map_dict_keys(
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
        result = u.map_dict_keys(
            source_raw,
            mapping,
            keep_unmapped=False,
        )
        mapped = assertion_helpers.assert_flext_result_success(result)
        assert mapped == {mc.NEW_KEY: mc.VALUE1}

    def test_map_dict_keys_exception_handling(self) -> None:
        bad_dict_instance = self._BadDict()
        result = u.map_dict_keys(bad_dict_instance, {})
        tm.fail(result, contains="Failed to map dict keys")

    def test_basic_filter(self) -> None:
        mc = c.Core.Mapper
        source_raw = {mc.A: mc.NUM_1, mc.B: mc.NUM_2, mc.C: mc.NUM_3}
        result = u.filter_dict(
            source_raw,
            lambda k, v: isinstance(v, int) and v > mc.NUM_1,
        )
        assert result == {mc.B: mc.NUM_2, mc.C: mc.NUM_3}
