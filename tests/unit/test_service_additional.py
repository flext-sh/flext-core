"""Additional coverage for flext_core using real executions."""

from __future__ import annotations

from typing import override

import pytest

from tests import e, r, s


class RuntimeCloneService(s[str]):
    """Service exposing runtime cloning for testing."""

    flag: bool = True

    @override
    def execute(self) -> r[str]:
        return r[str].ok("run")

    @override
    def validate_business_rules(self) -> r[bool]:
        if not self.flag:
            return r[bool].fail("bad flag")
        return r[bool].ok(True)


def test_valid_handles_validation_exception() -> None:
    """Valid should return False when validation raises exceptions."""

    class RaisingValidationService(s[str]):
        @override
        def validate_business_rules(self) -> r[bool]:
            msg = "boom"
            raise RuntimeError(msg)

        @override
        def execute(self) -> r[str]:
            return r[str].ok("x")

    service = RaisingValidationService()
    assert service.valid() is False


def test_result_property_raises_on_failure() -> None:
    """Result property should raise BaseError on failed execution."""

    class FailingOnResultService(s[str]):
        @override
        def execute(self) -> r[str]:
            return r[str].fail("fail_exec")

    service = FailingOnResultService()
    with pytest.raises(e.BaseError, match="fail_exec"):
        _ = service.result
