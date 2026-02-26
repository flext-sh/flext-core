"""Additional coverage for flext_core.service using real executions."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from flext_core import e, r, s, t


class RuntimeCloneService(s[str]):
    """Service exposing runtime cloning for testing."""

    flag: bool = True

    def execute(self) -> r[str]:
        return r[str].ok("run")

    def validate_business_rules(self) -> r[bool]:
        if not self.flag:
            return r[bool].fail("bad flag")
        return r[bool].ok(True)


def test_is_valid_handles_validation_exception() -> None:
    """is_valid should return False when validation raises exceptions."""

    class RaisingValidationService(s[str]):
        def validate_business_rules(self) -> r[bool]:
            msg = "boom"
            raise RuntimeError(msg)

        def execute(self) -> r[str]:
            return r[str].ok("x")

    service = RaisingValidationService()
    assert service.is_valid() is False


def test_result_property_raises_on_failure() -> None:
    """Result property should raise BaseError on failed execution."""

    class FailingOnResultService(s[str]):
        def execute(self) -> r[str]:
            return r[str].fail("fail_exec")

    service = FailingOnResultService()
    with pytest.raises(e.BaseError, match="fail_exec"):
        _ = service.result


def test_get_service_info() -> None:
    """Service should return basic service info."""
    service = RuntimeCloneService()
    info: Mapping[str, t.FlexibleValue] = service.get_service_info()
    assert info["service_type"] == "RuntimeCloneService"
