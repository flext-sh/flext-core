"""Behavioral tests for r[T] combinator exception propagation.

Every test asserts the public contract of the result combinators
(``map`` / ``flat_map`` / ``flow_through`` / ``map_error`` / ``lash`` /
``recover`` / ``filter`` / ``tap`` / ``tap_error``) through their observable
outcome: the success/failure flag, the carried value, the carried error
message, and the carried exception object. No private attribute is touched
and no collaborator is mocked or patched.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests import r, tm

if TYPE_CHECKING:
    from typing import Literal, Never

    from tests import p


class TestsFlextCoreResultExceptionMapping:
    """Contract tests for exception carrying across result combinators."""

    @staticmethod
    def test_map_on_failure_preserves_carried_exception() -> None:
        """Test map on failure preserves carried exception."""
        exc = ValueError("original error")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        mapped: p.Result[int] = result.map(lambda value: value * 2)

        tm.that(mapped.failure, eq=True)
        tm.that(mapped.exception is exc, eq=True)

    @staticmethod
    def test_map_on_success_transforms_value_without_exception() -> None:
        """Test map on success transforms value without exception."""
        result: p.Result[int] = r[int].ok(5)

        mapped: p.Result[int] = result.map(lambda value: value * 2)

        tm.that(mapped.success, eq=True)
        tm.that(mapped.value, eq=10)
        tm.that(mapped.exception, none=True)

    @staticmethod
    def test_map_chain_keeps_original_exception() -> None:
        """Test map chain keeps original exception."""
        exc = RuntimeError("chain error")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        mapped: p.Result[int] = result.map(lambda value: value + 1).map(
            lambda value: value * 2,
        )

        tm.that(mapped.failure, eq=True)
        tm.that(mapped.exception is exc, eq=True)

    @staticmethod
    def test_flat_map_on_failure_preserves_carried_exception() -> None:
        """Test flat map on failure preserves carried exception."""
        exc = TypeError("type error")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        flat_mapped: p.Result[str] = result.flat_map(
            lambda value: r[str].ok(str(value)),
        )

        tm.that(flat_mapped.failure, eq=True)
        tm.that(flat_mapped.exception is exc, eq=True)

    @staticmethod
    def test_flat_map_on_success_binds_next_result() -> None:
        """Test flat map on success binds next result."""
        result: p.Result[int] = r[int].ok(5)

        flat_mapped: p.Result[str] = result.flat_map(
            lambda value: r[str].ok(str(value)),
        )

        tm.that(flat_mapped.success, eq=True)
        tm.that(flat_mapped.value, eq="5")
        tm.that(flat_mapped.exception, none=True)

    @staticmethod
    def test_flat_map_chain_keeps_original_exception() -> None:
        """Test flat map chain keeps original exception."""
        exc = KeyError("missing key")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        flat_mapped: p.Result[str] = result.flat_map(
            lambda value: r[int].ok(value + 1),
        ).flat_map(lambda value: r[str].ok(str(value)))

        tm.that(flat_mapped.failure, eq=True)
        tm.that(flat_mapped.exception is exc, eq=True)

    @staticmethod
    def test_map_error_on_failure_transforms_message_keeping_exception() -> None:
        """Test map error on failure transforms message keeping exception."""
        exc = ValueError("original")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        altered: p.Result[int] = result.map_error(lambda error: f"transformed: {error}")

        tm.that(altered.failure, eq=True)
        tm.that(altered.exception is exc, eq=True)
        tm.that(altered.error is not None and "transformed" in altered.error, eq=True)

    @staticmethod
    def test_map_error_on_success_is_identity() -> None:
        """Test map error on success is identity."""
        result: p.Result[int] = r[int].ok(42)

        altered: p.Result[int] = result.map_error(lambda error: f"error: {error}")

        tm.that(altered.success, eq=True)
        tm.that(altered.value, eq=42)
        tm.that(altered.exception, none=True)

    @staticmethod
    def test_lash_recovers_failure_into_success_value() -> None:
        """Test lash recovers failure into success value."""
        exc = RuntimeError("recovery needed")
        result: p.Result[int] = r[int].fail("error", exception=exc)

        recovered: p.Result[int] = result.lash(lambda _: r[int].ok(0))

        tm.that(recovered.success, eq=True)
        tm.that(recovered.value, eq=0)

    @staticmethod
    def test_lash_recovery_failure_carries_new_exception() -> None:
        """Test lash recovery failure carries new exception."""
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
        "combinator",
        (
            "map",
            "flat_map",
            "flow_through",
            "map_error",
            "recover",
            "lash",
            "filter",
            "tap",
            "tap_error",
        ),
    )
    @staticmethod
    def test_callback_exception_becomes_carried_failure(
        combinator: Literal[
            "map", "flat_map", "flow_through", "map_error", "recover",
            "lash", "filter", "tap", "tap_error",
        ],
    ) -> None:
        """Test callback exception becomes carried failure."""
        exc = RuntimeError("callback failed")

        def reject_value(_value: int) -> Never:
            raise exc

        def reject_result(_value: int) -> p.Result[int]:
            raise exc

        def reject_error(_error: str) -> Never:
            raise exc

        def reject_recovery(_error: str) -> p.Result[int]:
            raise exc

        match combinator:
            case "map":
                result = r[int].ok(5).map(reject_value)
            case "flat_map":
                result = r[int].ok(5).flat_map(reject_result)
            case "flow_through":
                result = r[int].ok(5).flow_through(reject_result)
            case "map_error":
                result = r[int].fail("error").map_error(reject_error)
            case "recover":
                result = r[int].fail("error").recover(reject_error)
            case "lash":
                result = r[int].fail("error").lash(reject_recovery)
            case "filter":
                result = r[int].ok(5).filter(reject_value)
            case "tap":
                result = r[int].ok(5).tap(reject_value)
            case "tap_error":
                result = r[int].fail("error").tap_error(reject_error)

        tm.that(result.failure, eq=True)
        tm.that(result.exception is exc, eq=True)
        tm.that(result.error, eq=str(exc))
