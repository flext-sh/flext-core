"""Direct factory and protocol contracts for the Result DIP surface.

Pins ``from_result`` / ``from_failure``, runtime
``p.Result`` conformance, and ``flow_through`` normalization when a step returns
a foreign result-like value.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import FlextResult
from tests import e, m, p, r, tm, u


class TestsFlextCoreResultFactoryDip:
    """Public factory and protocol contracts after the p.Result DIP refactor."""

    @staticmethod
    def test_from_result_copies_success_payload() -> None:
        """Test from result copies success payload."""
        source: p.Result[str] = r[str].ok("payload")
        copied: p.Result[str] = r.from_result(source)
        tm.ok(copied, eq="payload")
        tm.that(isinstance(copied, FlextResult), truth=True)

    @staticmethod
    def test_from_result_copies_failure_metadata() -> None:
        """Test from result copies failure metadata."""
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
        tm.that(copied.exception is cause, truth=True)

    @staticmethod
    def test_from_failure_rebuilds_failed_result() -> None:
        """Test from failure rebuilds failed result."""
        cause = RuntimeError("x")
        source: p.Result[int] = r[int].fail(
            "nope",
            error_code="E_NOPE",
            error_data={"a": 1},
            exception=cause,
        )
        rebuilt: p.Result[int] = r[int].from_failure(source)
        tm.fail(rebuilt, has="nope")
        tm.that(rebuilt.error_code, eq="E_NOPE")
        tm.that(rebuilt.error_data, eq={"a": 1})
        tm.that(rebuilt.exception is cause, truth=True)

    @staticmethod
    def test_from_failure_rejects_successful_source() -> None:
        """Test from failure rejects successful source."""
        with pytest.raises(ValueError, match="successful result"):
            _ = r[int].from_failure(r[int].ok(1))

    @staticmethod
    def test_from_result_preserves_success() -> None:
        """Test from result preserves success."""
        source: p.Result[int] = r[int].ok(9)
        copied: p.Result[int] = r.from_result(source)
        tm.ok(copied, eq=9)

    @staticmethod
    def test_from_result_preserves_failure() -> None:
        """Test from result preserves failure."""
        source: p.Result[int] = r[int].fail("copy-fail", error_code="E_COPY")
        copied: p.Result[int] = r.from_result(source)
        tm.fail(copied, has="copy-fail")
        tm.that(copied.error_code, eq="E_COPY")

    @staticmethod
    def test_concrete_results_satisfy_result_protocol_at_runtime() -> None:
        """Test concrete results satisfy result protocol at runtime."""
        ok_result: p.Result[str] = r[str].ok("value")
        fail_result: p.Result[str] = r[str].fail("boom")
        tm.that(ok_result, is_=p.Result)
        tm.that(fail_result, is_=p.Result)
        tm.that(ok_result, is_=p.SuccessCheckable)
        tm.that(fail_result, is_=p.SuccessCheckable)

    @staticmethod
    def test_flow_through_normalizes_returned_success_onto_facade() -> None:
        """Normalize actual Result callbacks without claiming a partial interface."""

        def first_step(value: int) -> p.Result[int]:
            return r[int].ok(value + 1)

        def facade_step(value: int) -> p.Result[int]:
            return r[int].ok(value * 10)

        final: p.Result[int] = r[int].ok(3).flow_through(first_step, facade_step)
        tm.ok(final, eq=40)
        tm.ok(isinstance(final, FlextResult))

    @staticmethod
    def test_flow_through_normalizes_returned_failure_onto_facade() -> None:
        """Copy failure metadata and stop before invoking another actual step."""

        def first_fail(_value: int) -> p.Result[int]:
            return r[int].fail("step-stop", error_code="E_STEP")

        def unreachable(_value: int) -> p.Result[int]:
            pytest.fail("A failed flow must not invoke another step")

        final: p.Result[int] = r[int].ok(1).flow_through(first_fail, unreachable)
        tm.fail(final, has="step-stop")
        tm.that(final.error_code, eq="E_STEP")
        tm.ok(isinstance(final, FlextResult))

    @staticmethod
    @pytest.mark.parametrize("failed", [False, True])
    def test_empty_flow_preserves_original_identity(failed: bool) -> None:
        """A zero-step flow neither reconstructs state nor invokes a callback."""
        source: p.Result[int] = r[int].fail("stopped") if failed else r[int].ok(3)
        result: p.Result[int] = source.flow_through()
        tm.ok(result is source)

    @staticmethod
    def test_union_aware_flow_can_transform_the_payload() -> None:
        """Every step accepts both the initial and subsequent payload domain."""

        def to_text(value: int | str) -> p.Result[str]:
            return r[str].ok(str(value))

        def decorate(value: int | str) -> p.Result[str]:
            return r[str].ok(f"[{value}]")

        result: p.Result[int | str] = r[int].ok(3).flow_through(to_text, decorate)
        tm.ok(result, eq="[3]")

    @staticmethod
    def test_widening_recovery_preserves_success_identity() -> None:
        """Union returns need no reconstruction or cast of successful state."""

        def recover(_message: str) -> str:
            pytest.fail("A successful result must not run its recovery callback")

        def lash(_message: str) -> p.Result[str]:
            pytest.fail("A successful result must not run its lash callback")

        source: p.Result[int] = r[int].ok(3)
        recovered: p.Result[int | str] = source.recover(recover)
        lashed: p.Result[int | str] = source.lash(lash)
        tm.ok(recovered is source)
        tm.ok(lashed is source)

    @staticmethod
    def test_widening_lash_keeps_callback_normalization_and_failure_metadata() -> None:
        """Changing the observed domain preserves the existing normalization."""
        cause = ValueError("original")
        callback_result: p.Result[str] = r[str].fail(
            "mapped",
            error_code="E_MAPPED",
            error_data={"kind": "step"},
            exception=cause,
        )

        def callback(_message: str) -> p.Result[str]:
            return callback_result

        result: p.Result[int | str] = r[int].fail("input").lash(callback)
        tm.fail(result, has="mapped")
        tm.ok(result is not callback_result)
        tm.that(result.error_code, eq=callback_result.error_code)
        tm.that(result.error_data, eq=callback_result.error_data)
        tm.ok(result.exception is cause)

    @staticmethod
    def test_empty_fail_combinators_stay_failed_results() -> None:
        """Test empty fail combinators stay failed results."""
        for empty in (None, ""):
            source: p.Result[int] = r[int].fail(empty)
            mapped: p.Result[int] = source.map(lambda value: value + 1)
            flat: p.Result[int] = source.flat_map(r[int].ok)
            copied: p.Result[int] = r.from_result(source)
            tm.fail(mapped, has="")
            tm.that(mapped.error, eq="")
            tm.fail(flat, has="")
            tm.that(flat.error, eq="")
            tm.fail(copied, has="")
            tm.that(copied.error, eq="")
            tm.that(isinstance(mapped, FlextResult), truth=True)
            tm.that(isinstance(flat, FlextResult), truth=True)
            tm.that(isinstance(copied, FlextResult), truth=True)

    @staticmethod
    def test_from_failure_rebuilds_foreign_failure_like() -> None:
        """Test from failure rebuilds foreign failure like."""
        foreign = m.Tests.FailureEnvelope(
            error="foreign-fail",
            error_code="E_FOREIGN",
            error_data={"k": 1},
        )
        rebuilt: p.Result[int] = r[int].from_failure(foreign)
        tm.fail(rebuilt, has="foreign-fail")
        tm.that(rebuilt.error_code, eq="E_FOREIGN")
        tm.that(rebuilt.error_data, eq={"k": 1})
        tm.that(isinstance(rebuilt, FlextResult), truth=True)

    @staticmethod
    def test_from_result_preserves_exception_identity_on_flext_result() -> None:
        """Test from result preserves exception identity on flext result."""
        cause = RuntimeError("root-cause")
        source: p.Result[int] = r[int].fail(
            "copy-exc",
            error_code="E_EXC",
            exception=cause,
        )
        copied: p.Result[int] = r.from_result(source)
        tm.fail(copied, has="copy-exc")
        tm.that(copied.error_code, eq="E_EXC")
        tm.that(copied.exception is cause, truth=True)
        tm.that(isinstance(copied, FlextResult), truth=True)

    @staticmethod
    def test_fail_from_exception_redacts_sensitive_error_data_keys() -> None:
        """Test fail from exception redacts sensitive error data keys."""
        password, token = u.generate_id(), u.generate_id()
        exc = e.OperationError(
            "denied",
            context={"password": password, "host": "db.example", "token": token},
        )
        result: p.Result[int] = r[int].fail(None, exception=exc)
        tm.fail(result, has="")
        error_data = tm.not_none(result.error_data)
        tm.that(error_data.get("host"), eq="db.example")
        _assert_redacted(error_data, password, token)

    @staticmethod
    def test_fail_op_wraps_exception_and_stays_concrete_result() -> None:
        """Test fail op wraps exception and stays concrete result."""
        cause = RuntimeError("db down")
        failed: p.Result[int] = r[int].fail_op("connect", cause)
        tm.fail(failed, has="connect failed")
        tm.fail(failed, has="db down")
        tm.that(failed.exception is cause, truth=True)
        tm.that(r.failed_result(failed), truth=True)
        tm.that(r.successful_result(failed), truth=False)
        tm.that(isinstance(failed, FlextResult), truth=True)

    @staticmethod
    def test_fail_op_with_string_reason_has_no_exception() -> None:
        """Test fail op with string reason has no exception."""
        failed: p.Result[str] = r[str].fail_op("parse", "bad token")
        tm.fail(failed, has="parse failed: bad token")
        tm.that(failed.exception, eq=None)
        tm.that(isinstance(failed, FlextResult), truth=True)

    @staticmethod
    def test_from_validation_ok_and_fail_paths() -> None:
        """Test from validation ok and fail paths."""

        class _Sample(m.StrictModel):
            name: str

        ok_result: p.Result[_Sample] = r.from_validation({"name": "ada"}, _Sample)
        tm.ok(ok_result)
        tm.that(ok_result.unwrap().name, eq="ada")
        tm.that(r.successful_result(ok_result), truth=True)
        tm.that(isinstance(ok_result, FlextResult), truth=True)

        bad: p.Result[_Sample] = r.from_validation({"name": 1}, _Sample)
        tm.fail(bad)
        tm.that(bad.exception, none=False)
        tm.that(r.failed_result(bad), truth=True)
        tm.that(isinstance(bad, FlextResult), truth=True)

    @staticmethod
    def test_fail_explicit_error_data_redacts_sensitive_keys() -> None:
        """Test fail explicit error data redacts sensitive keys."""
        password, token = u.generate_id(), u.generate_id()
        result: p.Result[int] = r[int].fail(
            "denied",
            error_data={"password": password, "host": "db.example", "token": token},
        )
        tm.fail(result, has="denied")
        error_data = tm.not_none(result.error_data)
        tm.that(error_data.get("host"), eq="db.example")
        _assert_redacted(error_data, password, token)

    @staticmethod
    def test_fail_explicit_error_data_wins_but_still_redacts_with_exception() -> None:
        """Test fail explicit error data wins but still redacts with exception."""
        exception_password, password, api_key = (
            u.generate_id(),
            u.generate_id(),
            u.generate_id(),
        )
        exc = e.OperationError(
            "x", context={"password": exception_password, "host": "h"}
        )
        result: p.Result[int] = r[int].fail(
            "denied",
            error_data={"password": password, "host": "kept", "api_key": api_key},
            exception=exc,
        )
        tm.fail(result, has="denied")
        error_data = tm.not_none(result.error_data)
        tm.that(error_data.get("host"), eq="kept")
        tm.that("password" not in str(result.error_data), truth=True)
        tm.that("api_key" not in str(result.error_data), truth=True)
        for secret in (exception_password, password, api_key):
            tm.that(secret not in str(result.error_data), truth=True)


def _assert_redacted(error_data: object, password: str, token: str) -> None:
    text = str(error_data)
    tm.that(text, lacks="password")
    tm.that(text, lacks="token")
    tm.that(text, lacks=password)
    tm.that(text, lacks=token)
