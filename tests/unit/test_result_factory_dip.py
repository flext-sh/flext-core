"""Direct factory and protocol contracts for the Result DIP surface.

Pins ``from_result`` / ``from_failure`` / ``copy_from_result``, runtime
``p.Result`` conformance, and ``flow_through`` normalization when a step returns
a foreign result-like value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import pytest

from flext_core import FlextResult, e, m
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
    error_data: Mapping[str, str | int | bool | None] | None = None
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
    error_data: Mapping[str, str | int | bool | None] | None = None
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
            "broken", error_code="E_BROKEN", error_data={"k": "v"}, exception=cause
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

    def test_empty_fail_combinators_stay_failed_results(self) -> None:
        for empty in (None, ""):
            source: p.Result[int] = r[int].fail(empty)
            mapped: p.Result[int] = source.map(lambda value: value + 1)
            flat: p.Result[int] = source.flat_map(lambda value: r[int].ok(value))
            copied: p.Result[int] = r.from_result(source)
            tm.fail(mapped, has="")
            tm.that(mapped.error, eq="")
            tm.fail(flat, has="")
            tm.that(flat.error, eq="")
            tm.fail(copied, has="")
            tm.that(copied.error, eq="")
            assert isinstance(mapped, FlextResult)
            assert isinstance(flat, FlextResult)
            assert isinstance(copied, FlextResult)

    def test_from_failure_rebuilds_foreign_failure_like(self) -> None:
        foreign = _ForeignFail(
            error="foreign-fail", error_code="E_FOREIGN", error_data={"k": 1}
        )
        rebuilt: p.Result[int] = r[int].from_failure(foreign)
        tm.fail(rebuilt, has="foreign-fail")
        tm.that(rebuilt.error_code, eq="E_FOREIGN")
        tm.that(rebuilt.error_data, eq={"k": 1})
        assert isinstance(rebuilt, FlextResult)

    def test_copy_from_result_preserves_exception_identity_on_flext_result(
        self,
    ) -> None:
        cause = RuntimeError("root-cause")
        source: p.Result[int] = r[int].fail(
            "copy-exc", error_code="E_EXC", exception=cause
        )
        copied: p.Result[int] = r.copy_from_result(source)
        tm.fail(copied, has="copy-exc")
        tm.that(copied.error_code, eq="E_EXC")
        assert copied.exception is cause
        assert isinstance(copied, FlextResult)

    def test_fail_from_exception_redacts_sensitive_error_data_keys(self) -> None:
        exc = e.OperationError(
            "denied",
            context={"password": "s3cret", "host": "db.example", "token": "t0k"},
        )
        result: p.Result[int] = r[int].fail(None, exception=exc)
        tm.fail(result, has="")
        assert result.error_data is not None
        tm.that(result.error_data.get("host"), eq="db.example")
        assert "password" not in result.error_data
        assert "token" not in result.error_data
        assert "s3cret" not in str(result.error_data)
        assert "t0k" not in str(result.error_data)

    def test_fail_op_wraps_exception_and_stays_concrete_result(self) -> None:
        cause = RuntimeError("db down")
        failed: p.Result[int] = r[int].fail_op("connect", cause)
        tm.fail(failed, has="connect failed")
        tm.fail(failed, has="db down")
        assert failed.exception is cause
        assert isinstance(failed, FlextResult)
        assert r.failed_result(failed)
        assert not r.successful_result(failed)

    def test_fail_op_with_string_reason_has_no_exception(self) -> None:
        failed: p.Result[str] = r[str].fail_op("parse", "bad token")
        tm.fail(failed, has="parse failed: bad token")
        tm.that(failed.exception, eq=None)
        assert isinstance(failed, FlextResult)

    def test_from_validation_ok_and_fail_paths(self) -> None:
        class _Sample(m.StrictModel):
            name: str

        ok_result: p.Result[_Sample] = r.from_validation({"name": "ada"}, _Sample)
        tm.ok(ok_result)
        tm.that(ok_result.unwrap().name, eq="ada")
        assert isinstance(ok_result, FlextResult)
        assert r.successful_result(ok_result)

        bad: p.Result[_Sample] = r.from_validation({"name": 1}, _Sample)
        tm.fail(bad)
        assert bad.exception is not None
        assert isinstance(bad, FlextResult)
        assert r.failed_result(bad)
