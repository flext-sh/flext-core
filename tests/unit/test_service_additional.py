"""Additional service tests aligned with the current slim API."""

from __future__ import annotations

from typing import override

from tests import p, r, s, t


class TestsFlextCoreServiceAdditionalRuntimeCloneService(s[str]):
    """Service used in additional execution-path tests."""

    __test__ = False

    should_fail: bool = False

    @override
    def execute(self) -> p.Result[str]:
        if self.should_fail:
            return r[str].fail("fail_exec")
        return r[str].ok("run")


class TestServiceAdditional:
    """Additional coverage tests for service execute flows."""

    def test_execute_success_path(self) -> None:
        result = TestsFlextCoreServiceAdditionalRuntimeCloneService().execute()
        assert result.success
        assert result.value == "run"

    def test_execute_failure_path(self) -> None:
        result = TestsFlextCoreServiceAdditionalRuntimeCloneService(
            should_fail=True,
        ).execute()
        assert result.failure
        assert result.error == "fail_exec"


__all__: t.MutableSequenceOf[str] = ["TestServiceAdditional"]
