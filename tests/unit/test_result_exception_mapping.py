"""Result exception propagation mapping tests."""

from __future__ import annotations

from flext_tests import r, tm

from tests.protocols import p
from tests.unit._result_exception_support import TestsFlextResultExceptionCarrying


class TestsFlextResultExceptionMapping(TestsFlextResultExceptionCarrying):
    def test_map_propagates_exception_on_failure(self) -> None:
        exc = ValueError("original error")
        result: p.Result[int] = r[int].fail("error", exception=exc)
        mapped: p.Result[int] = result.map(lambda value: value * 2)
        tm.that(mapped.failure, eq=True)
        tm.that(mapped.exception is exc, eq=True)

    def test_map_success_no_exception(self) -> None:
        result: p.Result[int] = r[int].ok(5)
        mapped: p.Result[int] = result.map(lambda value: value * 2)
        tm.that(mapped.success, eq=True)
        tm.that(mapped.value, eq=10)
        tm.that(mapped.exception, none=True)

    def test_map_chain_preserves_exception(self) -> None:
        exc = RuntimeError("chain error")
        result: p.Result[int] = r[int].fail("error", exception=exc)
        mapped: p.Result[int] = result.map(lambda value: value + 1).map(
            lambda value: value * 2,
        )
        tm.that(mapped.failure, eq=True)
        tm.that(mapped.exception is exc, eq=True)

    def test_flat_map_propagates_exception_on_failure(self) -> None:
        exc = TypeError("type error")
        result: p.Result[int] = r[int].fail("error", exception=exc)
        flat_mapped: p.Result[str] = result.flat_map(
            lambda value: r[str].ok(str(value))
        )
        tm.that(flat_mapped.failure, eq=True)
        tm.that(flat_mapped.exception is exc, eq=True)

    def test_flat_map_success_no_exception(self) -> None:
        result: p.Result[int] = r[int].ok(5)
        flat_mapped: p.Result[str] = result.flat_map(
            lambda value: r[str].ok(str(value))
        )
        tm.that(flat_mapped.success, eq=True)
        tm.that(flat_mapped.value, eq="5")
        tm.that(flat_mapped.exception, none=True)

    def test_flat_map_callback_exception_returns_failure(self) -> None:
        result: p.Result[int] = r[int].ok(5)
        exc = RuntimeError("flat-map callback failed")

        def callback(_value: int) -> p.Result[str]:
            raise exc

        flat_mapped = result.flat_map(callback)

        tm.that(flat_mapped.failure, eq=True)
        tm.that(flat_mapped.exception is exc, eq=True)
        tm.that(flat_mapped.error, eq=str(exc))

    def test_flat_map_chain_preserves_exception(self) -> None:
        exc = KeyError("missing key")
        result: p.Result[int] = r[int].fail("error", exception=exc)
        flat_mapped: p.Result[str] = result.flat_map(
            lambda value: r[int].ok(value + 1),
        ).flat_map(
            lambda value: r[str].ok(str(value)),
        )
        tm.that(flat_mapped.failure, eq=True)
        tm.that(flat_mapped.exception is exc, eq=True)

    def test_alt_propagates_exception(self) -> None:
        exc = ValueError("original")
        result: p.Result[int] = r[int].fail("error", exception=exc)
        altered: p.Result[int] = result.map_error(lambda error: f"transformed: {error}")
        tm.that(altered.failure, eq=True)
        tm.that(altered.exception is exc, eq=True)
        tm.that(altered.error is not None and "transformed" in altered.error, eq=True)

    def test_alt_success_no_exception(self) -> None:
        result: p.Result[int] = r[int].ok(42)
        altered: p.Result[int] = result.map_error(lambda error: f"error: {error}")
        tm.that(altered.success, eq=True)
        tm.that(altered.value, eq=42)
        tm.that(altered.exception, none=True)

    def test_lash_propagates_exception(self) -> None:
        exc = RuntimeError("recovery needed")
        result: p.Result[int] = r[int].fail("error", exception=exc)
        recovered = result.lash(lambda _: r[int].ok(0))
        tm.that(recovered.success, eq=True)
        tm.that(recovered.value, eq=0)

    def test_lash_preserves_exception_on_recovery_failure(self) -> None:
        exc = ValueError("original error")
        result: p.Result[int] = r[int].fail("error", exception=exc)
        recovery_exc = RuntimeError("recovery failed")
        recovered: p.Result[int] = result.lash(
            lambda error: r[int].fail(
                f"recovery failed: {error}",
                exception=recovery_exc,
            ),
        )
        tm.that(recovered.failure, eq=True)
        tm.that(recovered.exception is recovery_exc, eq=True)

    def test_lash_callback_exception_returns_failure(self) -> None:
        result: p.Result[int] = r[int].fail("error")
        exc = RuntimeError("lash callback failed")

        def callback(_error: str) -> p.Result[int]:
            raise exc

        recovered = result.lash(callback)

        tm.that(recovered.failure, eq=True)
        tm.that(recovered.exception is exc, eq=True)
        tm.that(recovered.error, eq=str(exc))
