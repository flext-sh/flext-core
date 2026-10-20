"""Behavioral tests for the beartype conf factory and engine predicates.

Asserts the public contract of ``FlextUtilitiesBeartypeConf`` and the
annotation-inspection predicates exposed on the ``u`` facade — return
values only, never internal wiring.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAliasType

import pytest
from beartype import BeartypeConf, BeartypeStrategy

from flext_core.utilities import FlextUtilitiesBeartypeConf
from tests import c, t, tm, u
from tests.unit._enforcement_support import invalid_annotations

_FORBIDDEN: frozenset[str] = frozenset({"dict", "list", "set"})


class TestsFlextCoreBeartypeEngineConfig:
    """Public contract of the beartype conf factory + engine predicates."""

    @staticmethod
    def test_build_conf_returns_beartype_conf_instance() -> None:
        """The factory yields a real ``BeartypeConf`` callers can pass to beartype."""
        conf = FlextUtilitiesBeartypeConf.build_beartype_conf()
        tm.that(conf, is_=BeartypeConf)

    @staticmethod
    def test_default_mode_is_off() -> None:
        """flext_core ships with enforcement disabled by default."""
        tm.that(c.BEARTYPE_MODE is c.EnforcementMode.OFF, truth=True)

    @staticmethod
    def test_disabled_mode_yields_no_op_strategy() -> None:
        """With mode OFF the conf uses the O0 (no-check) strategy."""
        conf = FlextUtilitiesBeartypeConf.build_beartype_conf()
        tm.that(conf.strategy is BeartypeStrategy.O0, truth=True)

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (str | int, False),
            (str, False),
            (None, False),
        ],
    )
    @staticmethod
    def test_contains_any_detects_unrestricted_hints(
        hint: t.TypeHintSpecifier | None,
        *,
        expected: bool,
    ) -> None:
        """``contains_any`` is True exactly when a hint admits any value."""
        tm.that(u.contains_any(hint) is expected, truth=True)

    @pytest.mark.parametrize("alias_name", ["ObjectAlias", "NestedObjectAlias"])
    @staticmethod
    def test_contains_any_detects_unrestricted_source_hints(
        tmp_path: Path, alias_name: str
    ) -> None:
        """Unrestricted native hints enter only through explicit source fixtures."""
        with invalid_annotations(tmp_path) as source:
            alias = getattr(source.InvalidTypes, alias_name)
            if not isinstance(alias, TypeAliasType):
                pytest.fail("Invalid source fixture is not a native alias")
            tm.ok(u.contains_any(alias.__value__))

    @pytest.mark.parametrize(
        ("alias_name", "expected"),
        [("AnyAlias", True), ("NestedAnyAlias", True), ("CleanAlias", False)],
    )
    @staticmethod
    def test_alias_contains_any_unwraps_type_alias(
        tmp_path: Path,
        alias_name: str,
        *,
        expected: bool,
    ) -> None:
        """``alias_contains_any`` follows a PEP 695 alias to its underlying value."""
        with invalid_annotations(tmp_path) as source:
            alias = getattr(source.InvalidTypes, alias_name)
            if not isinstance(alias, TypeAliasType):
                pytest.fail("Source fixture did not declare a native type alias")
            tm.that(u.alias_contains_any(alias), eq=expected)

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (dict[str, int], (True, "dict")),
            (list[int], (True, "list")),
            (t.VariadicTuple[int], (False, "")),
            (str, (False, "")),
            (None, (False, "")),
        ],
    )
    @staticmethod
    def test_has_forbidden_collection_origin_reports_name(
        hint: t.TypeHintSpecifier | None,
        expected: tuple[bool, str],
    ) -> None:
        """Parametrized collection origins map to the forbidden flag + name."""
        tm.that(u.has_forbidden_collection_origin(hint, _FORBIDDEN), eq=expected)

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [(str | int | None, 2), (str | None, 1), (str | int, 2), (str, 0), (None, 0)],
    )
    @staticmethod
    def test_count_union_members_excludes_none(
        hint: t.TypeHintSpecifier | None,
        expected: int,
    ) -> None:
        """Union member count ignores ``NoneType`` and non-unions score zero."""
        tm.that(u.count_union_members(hint), eq=expected)

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (str | None, True),
            (int | None, False),
            (str | int, False),
            (str, False),
            (None, False),
        ],
    )
    @staticmethod
    def test_matches_str_none_union_is_exact(
        hint: t.TypeHintSpecifier | None,
        *,
        expected: bool,
    ) -> None:
        """Only the ``str | None`` shape matches; other unions do not."""
        tm.that(u.matches_str_none_union(hint) is expected, truth=True)
