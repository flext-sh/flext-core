"""Behavioral tests for r[T] combinator exception propagation.

Every test asserts the public contract of the result combinators
(``map`` / ``flat_map`` / ``flow_through`` / ``map_error`` / ``lash`` /
``recover`` / ``filter`` / ``tap`` / ``tap_error``) through their observable
outcome: the success/failure flag, the carried value, the carried error
message, and the carried exception object. No private attribute is touched
and no collaborator is mocked or patched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import r, tm

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Never

    from tests.protocols import p


def _raiser(exc: Exception) -> Callable[[object], Never]:
    """Return a one-argument callback that always raises ``exc``."""

    def _callback(_arg: object) -> Never:
        raise exc

    return _callback


class TestsFlextCoreResultExceptionMapping:
    """Contract tests for exception carrying across result combinators."""

    def test_map_on_failure_preserves_carried_exception(self) -> None:
        exc = ValueError("original error")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        mapped: p.Result[int] = result.map(lambda value: value * 2)

        tm.that(mapped.failure, eq=True)
        tm.that(mapped.exception is exc, eq=True)

    def test_map_on_success_transforms_value_without_exception(self) -> None:
        result: p.Result[int] = r[int].ok(5)

        mapped: p.Result[int] = result.map(lambda value: value * 2)

        tm.that(mapped.success, eq=True)
        tm.that(mapped.value, eq=10)
        tm.that(mapped.exception, none=True)

    def test_map_chain_keeps_original_exception(self) -> None:
        exc = RuntimeError("chain error")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        mapped: p.Result[int] = result.map(lambda value: value + 1).map(
            lambda value: value * 2,
        )

        tm.that(mapped.failure, eq=True)
        tm.that(mapped.exception is exc, eq=True)

    def test_flat_map_on_failure_preserves_carried_exception(self) -> None:
        exc = TypeError("type error")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        flat_mapped: p.Result[str] = result.flat_map(
            lambda value: r[str].ok(str(value)),
        )

        tm.that(flat_mapped.failure, eq=True)
        tm.that(flat_mapped.exception is exc, eq=True)

    def test_flat_map_on_success_binds_next_result(self) -> None:
        result: p.Result[int] = r[int].ok(5)

        flat_mapped: p.Result[str] = result.flat_map(
            lambda value: r[str].ok(str(value)),
        )

        tm.that(flat_mapped.success, eq=True)
        tm.that(flat_mapped.value, eq="5")
        tm.that(flat_mapped.exception, none=True)

    def test_flat_map_chain_keeps_original_exception(self) -> None:
        exc = KeyError("missing key")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        flat_mapped: p.Result[str] = result.flat_map(
            lambda value: r[int].ok(value + 1),
        ).flat_map(
            lambda value: r[str].ok(str(value)),
        )

        tm.that(flat_mapped.failure, eq=True)
        tm.that(flat_mapped.exception is exc, eq=True)

    def test_map_error_on_failure_transforms_message_keeping_exception(self) -> None:
        exc = ValueError("original")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        altered: p.Result[int] = result.map_error(
            lambda error: f"transformed: {error}",
        )

        tm.that(altered.failure, eq=True)
        tm.that(altered.exception is exc, eq=True)
        tm.that(altered.error is not None and "transformed" in altered.error, eq=True)

    def test_map_error_on_success_is_identity(self) -> None:
        result: p.Result[int] = r[int].ok(42)

        altered: p.Result[int] = result.map_error(lambda error: f"error: {error}")

        tm.that(altered.success, eq=True)
        tm.that(altered.value, eq=42)
        tm.that(altered.exception, none=True)

    def test_lash_recovers_failure_into_success_value(self) -> None:
        exc = RuntimeError("recovery needed")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        recovered: p.Result[int] = result.lash(lambda _: r[int].ok(0))

        tm.that(recovered.success, eq=True)
        tm.that(recovered.value, eq=0)

    def test_lash_recovery_failure_carries_new_exception(self) -> None:
        original_exc = ValueError("original error")
        result: p.Result[int] = r[int].fail("error", exception=original_exc)
        recovery_exc = RuntimeError("recovery failed")

        recovered: p.Result[int] = result.lash(
            lambda error: r[int].fail(
                f"recovery failed: {error}",
                exception=recovery_exc,
            ),
        )

        tm.that(recovered.failure, eq=True)
        tm.that(recovered.exception is recovery_exc, eq=True)

    @pytest.mark.parametrize(
        "invoke",
        [
            pytest.param(lambda exc: r[int].ok(5).map(_raiser(exc)), id="map"),
            pytest.param(
                lambda exc: r[int].ok(5).flat_map(_raiser(exc)),
                id="flat_map",
            ),
            pytest.param(
                lambda exc: r[int].ok(5).flow_through(_raiser(exc)),
                id="flow_through",
            ),
            pytest.param(
                lambda exc: r[int].fail("error").map_error(_raiser(exc)),
                id="map_error",
            ),
            pytest.param(
                lambda exc: r[int].fail("error").recover(_raiser(exc)),
                id="recover",
            ),
            pytest.param(
                lambda exc: r[int].fail("error").lash(_raiser(exc)),
                id="lash",
            ),
            pytest.param(
                lambda exc: r[int].ok(5).filter(_raiser(exc)),
                id="filter",
            ),
            pytest.param(lambda exc: r[int].ok(5).tap(_raiser(exc)), id="tap"),
            pytest.param(
                lambda exc: r[int].fail("error").tap_error(_raiser(exc)),
                id="tap_error",
            ),
        ],
    )
    def test_callback_exception_becomes_carried_failure(
        self,
        invoke: Callable[[Exception], p.Result[int]],
    ) -> None:
        exc = RuntimeError("callback failed")

        result: p.Result[int] = invoke(exc)

        tm.that(result.failure, eq=True)
        tm.that(result.exception is exc, eq=True)
        tm.that(result.error, eq=str(exc))
