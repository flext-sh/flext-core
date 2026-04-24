"""Utilities settings smoke tests for stable public helpers."""

from __future__ import annotations

from tests import c, u


class TestsFlextCoreUtilitiesSettings:
    def test_resolve_effective_log_level_prioritizes_trace(self) -> None:
        assert (
            u.resolve_effective_log_level(
                trace=True,
                debug=False,
                log_level=c.LogLevel.WARNING,
            )
            == c.LogLevel.DEBUG
        )

    def test_resolve_effective_log_level_promotes_debug(self) -> None:
        assert (
            u.resolve_effective_log_level(
                trace=False,
                debug=True,
                log_level=c.LogLevel.ERROR,
            )
            == c.LogLevel.INFO
        )

    def test_resolve_effective_log_level_keeps_explicit_level(self) -> None:
        assert (
            u.resolve_effective_log_level(
                trace=False,
                debug=False,
                log_level=c.LogLevel.ERROR,
            )
            == c.LogLevel.ERROR
        )
