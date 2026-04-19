"""Additional coverage for flext_core using real executions."""

from __future__ import annotations

from typing import override

import pytest

from tests import e, p, r, s


class TestsFlextCoreServiceAdditionalRuntimeCloneService(s[str]):
    """Service exposing runtime cloning for testing."""

    __test__ = False

    flag: bool = True

    @override
    def execute(self) -> p.Result[str]:
        return r[str].ok("run")

    @override
    def validate_business_rules(self) -> p.Result[bool]:
        if not self.flag:
            return r[bool].fail("bad flag")
        return r[bool].ok(True)


class TestServiceAdditional:
    """Additional coverage tests for flext_core service flows."""

    def test_valid_handles_validation_exception(self) -> None:
        """Valid should return False when validation raises exceptions."""

        class TestsFlextCoreServiceAdditionalRaisingValidationService(s[str]):
            @override
            def validate_business_rules(self) -> p.Result[bool]:
                msg = "boom"
                raise RuntimeError(msg)

            @override
            def execute(self) -> p.Result[str]:
                return r[str].ok("x")

        service = TestsFlextCoreServiceAdditionalRaisingValidationService()
        assert service.valid() is False

    def test_result_property_raises_on_failure(self) -> None:
        """Result property should raise BaseError on failed execution."""

        class FailingOnResultService(s[str]):
            @override
            def execute(self) -> p.Result[str]:
                return r[str].fail("fail_exec")

        service = FailingOnResultService()
        with pytest.raises(e.BaseError, match="fail_exec"):
            _ = service.result


__all__: list[str] = ["TestServiceAdditional"]
