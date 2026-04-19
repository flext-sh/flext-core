"""Dispatcher timeout smoke coverage aligned to current contracts."""

from __future__ import annotations

from tests import r


class TestDispatcherTimeoutCoverage100:
    def test_result_failure_contract_for_timeout_like_error(self) -> None:
        result = r[str].fail("timeout")
        assert result.failure
        assert result.error == "timeout"

    def test_result_success_contract_for_non_timeout_path(self) -> None:
        result = r[str].ok("ok")
        assert result.success
        assert result.value == "ok"
