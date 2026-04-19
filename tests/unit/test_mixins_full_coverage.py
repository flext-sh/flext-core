"""Mixins smoke coverage aligned to stable result contract."""

from __future__ import annotations

from tests import r


class TestMixinsFullCoverage:
    def test_success_result_contract(self) -> None:
        result = r[str].ok("ok")
        assert result.success

    def test_failure_result_contract(self) -> None:
        result = r[str].fail("error")
        assert result.failure
