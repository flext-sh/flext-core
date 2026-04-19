"""Utilities settings full coverage smoke tests."""

from __future__ import annotations

from tests import c, u


class TestUtilitiesSettingsFullCoverage:
    def test_effective_log_level_trace(self) -> None:
        assert (
            u.resolve_effective_log_level(
                trace=True,
                debug=False,
                log_level=c.LogLevel.ERROR,
            )
            == c.LogLevel.DEBUG
        )

    def test_effective_log_level_debug(self) -> None:
        assert (
            u.resolve_effective_log_level(
                trace=False,
                debug=True,
                log_level=c.LogLevel.ERROR,
            )
            == c.LogLevel.INFO
        )
