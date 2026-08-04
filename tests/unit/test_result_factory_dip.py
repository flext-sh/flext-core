"""Direct factory and protocol contracts for the Result DIP surface.

Pins ``from_result`` / ``from_failure`` / ``copy_from_result``, runtime
``p.Result`` conformance, and ``flow_through`` normalization when a step returns
a foreign result-like value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from flext_core import FlextResult
from flext_tests import r, tm
from tests.protocols import p


@dataclass(frozen=True, slots=True)
class _ForeignOk:
    """Minimal success-shaped result returned by a flow_through step."""

    value: int
    success: bool = True
    failure: bool = False
    error: str | None = None
    error_code: str | None = None
    error_data: dict[str, object] | None = None
    exception: BaseException | None = None

    def unwrap(self) -> int:
        return self.value


@dataclass(frozen=True, slots=True)
class _ForeignFail:
    """Minimal failure-shaped result returned by a flow_through step."""

    error: str
    success: bool = False
    failure: bool = True
    error_code: str | None = "E_FOREIGN"
    error_data: dict[str, object] | None = None
    exception: BaseException | None = None
    value: int | None = None


class TestsFlextCoreResultFactoryDip:
    """Public factory and protocol contracts after the p.Result DIP refactor."""

    def test_from_result_copies_success_payload(self) -> None:
        source: p.Result[str] = r[str].ok("payload")
        copied: p.Result[str] = r.from_result(source)
        tm.ok(copied, eq="payload")
        assert isinstance(copied, FlextResult)

    def test_from_result_copies_failure_metadata(self) -> None:
        cause = ValueError("root")
        source: p.Result[str] = r[str].fail(
            "broken",
            error_code="E_BROKEN",
            error_data={"k": "v"},
            exception=cause,
        )
        copied: p.Result[str] = r.from_result(source)
        tm.fail(copied, has="broken")
        tm.that(copied.error_code, eq="E_BROKEN")
        tm.that(copied.error_data, eq={"k": "v"})
        assert copied.exception is cause

    def test_from_failure_rebuilds_failed_result(self) -> None:
        cause = RuntimeError("x")
        source: p.Result[int] = r[int].fail(
            "nope", error_code="E_NOPE", error_data={"a": 1}, exception=cause
        )
        rebuilt: p.Result[int] = r[int].from_failure(source)
        tm.fail(rebuilt, has="nope")
        tm.that(rebuilt.error_code, eq="E_NOPE")
        tm.that(rebuilt.error_data, eq={"a": 1})
        assert rebuilt.exception is cause

    def test_from_failure_rejects_successful_source(self) -> None:
        with pytest.raises(ValueError, match="successful result"):
            _ = r[int].from_failure(r[int].ok(1))

    def test_copy_from_result_preserves_success(self) -> None:
        source: p.Result[int] = r[int].ok(9)
        copied: p.Result[int] = r.copy_from_result(source)
        tm.ok(copied, eq=9)

    def test_copy_from_result_preserves_failure(self) -> None:
        source: p.Result[int] = r[int].fail("copy-fail", error_code="E_COPY")
        copied: p.Result[int] = r.copy_from_result(source)
        tm.fail(copied, has="copy-fail")
        tm.that(copied.error_code, eq="E_COPY")

    def test_concrete_results_satisfy_result_protocol_at_runtime(self) -> None:
        ok_result: p.Result[str] = r[str].ok("value")
        fail_result: p.Result[str] = r[str].fail("boom")
        assert isinstance(ok_result, p.Result)
        assert isinstance(fail_result, p.Result)
        assert isinstance(ok_result, p.SuccessCheckable)
        assert isinstance(fail_result, p.SuccessCheckable)

    def test_flow_through_normalizes_foreign_success_onto_facade(self) -> None:
        def foreign_step(value: int) -> p.Result[int]:
            return cast("p.Result[int]", _ForeignOk(value=value + 1))

        def facade_step(value: int) -> p.Result[int]:
            return r[int].ok(value * 10)

        final: p.Result[int] = r[int].ok(3).flow_through(foreign_step, facade_step)
        tm.ok(final, eq=40)
        assert isinstance(final, FlextResult)

    def test_flow_through_normalizes_foreign_failure_onto_facade(self) -> None:
        def foreign_fail(_value: int) -> p.Result[int]:
            return cast("p.Result[int]", _ForeignFail(error="foreign-stop"))

        def unreachable(_value: int) -> p.Result[int]:
            return r[int].ok(999)

        final: p.Result[int] = r[int].ok(1).flow_through(foreign_fail, unreachable)
        tm.fail(final, has="foreign-stop")
        tm.that(final.error_code, eq="E_FOREIGN")
        assert isinstance(final, FlextResult)
