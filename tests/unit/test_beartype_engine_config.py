"""Behavioral tests for the beartype conf factory and engine predicates.

Asserts the public contract of ``FlextUtilitiesBeartypeConf`` and the
annotation-inspection predicates exposed on the ``u`` facade — return
values only, never internal wiring.
"""

from __future__ import annotations

import pytest
from beartype import BeartypeConf, BeartypeStrategy

from flext_core.utilities import FlextUtilitiesBeartypeConf
from tests.constants import c
from tests.typings import t
from tests.unit._beartype_engine_support import AnyAlias, CleanAlias, NestedAnyAlias
from tests.utilities import u

_FORBIDDEN: frozenset[str] = frozenset({"dict", "list", "set"})


class TestsFlextCoreBeartypeEngineConfig:
    """Public contract of the beartype conf factory + engine predicates."""

    def test_build_conf_returns_beartype_conf_instance(self) -> None:
        """The factory yields a real ``BeartypeConf`` callers can pass to beartype."""
        conf = FlextUtilitiesBeartypeConf.build_beartype_conf()
        assert isinstance(conf, BeartypeConf)

    def test_default_mode_is_off(self) -> None:
        """flext_core ships with enforcement disabled by default."""
        assert c.BEARTYPE_MODE is c.EnforcementMode.OFF

    def test_disabled_mode_yields_no_op_strategy(self) -> None:
        """With mode OFF the conf uses the O0 (no-check) strategy."""
        conf = FlextUtilitiesBeartypeConf.build_beartype_conf()
        assert conf.strategy is BeartypeStrategy.O0

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (object, True),
            (list[object], True),
            (str | int, False),
            (str, False),
            (None, False),
        ],
    )
    def test_contains_any_detects_unrestricted_hints(
        self, hint: t.TypeHintSpecifier | None, *, expected: bool
    ) -> None:
        """``contains_any`` is True exactly when a hint admits any value."""
        assert u.contains_any(hint) is expected

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [(AnyAlias, True), (NestedAnyAlias, True), (CleanAlias, False)],
    )
    def test_alias_contains_any_unwraps_type_alias(
        self, alias: t.TypeHintSpecifier | None, *, expected: bool
    ) -> None:
        """``alias_contains_any`` follows a PEP 695 alias to its underlying value."""
        assert u.alias_contains_any(alias) is expected

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [
            (dict[str, int], (True, "dict")),
            (list[int], (True, "list")),
            (tuple[int, ...], (False, "")),
            (str, (False, "")),
            (None, (False, "")),
        ],
    )
    def test_has_forbidden_collection_origin_reports_name(
        self, hint: t.TypeHintSpecifier | None, expected: tuple[bool, str]
    ) -> None:
        """Parametrized collection origins map to the forbidden flag + name."""
        assert u.has_forbidden_collection_origin(hint, _FORBIDDEN) == expected

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [(str | int | None, 2), (str | None, 1), (str | int, 2), (str, 0), (None, 0)],
    )
    def test_count_union_members_excludes_none(
        self, hint: t.TypeHintSpecifier | None, expected: int
    ) -> None:
        """Union member count ignores ``NoneType`` and non-unions score zero."""
        assert u.count_union_members(hint) == expected

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
    def test_matches_str_none_union_is_exact(
        self, hint: t.TypeHintSpecifier | None, *, expected: bool
    ) -> None:
        """Only the ``str | None`` shape matches; other unions do not."""
        assert u.matches_str_none_union(hint) is expected
