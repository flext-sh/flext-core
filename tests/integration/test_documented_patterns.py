"""Behavioral integration tests for documented flext-core public patterns.

Every assertion targets the observable public contract of ``r`` (FlextResult),
``e`` (FlextExceptions) and ``d`` (FlextDecorators): return values, the ``r[T]``
success/failure outcome, structured error payloads and raised exception fields.
No private attribute, internal collaborator or implementation detail is inspected.
"""

from __future__ import annotations

import pytest
from flext_tests import tm

from collections.abc import Callable

from tests.protocols import p, d, e, r

_COMBINED_SUM = 6
_FLAT_MAP_VALUE = 20
_FLOW_VALUE = 2
_INCREMENTED_VALUE = 4
_RECOVERED_VALUE = 99


class TestsFlextCoreDocumentedPatterns:
    """Public-contract behavior of the documented result/exception/decorator DSL."""

    @pytest.mark.parametrize(
        ("seed", "transform", "expected"),
        [
            (1, lambda value: value + 1, 2),
            (10, lambda value: value * 3, 30),
            (-5, lambda value: abs(value), 5),
        ],
    )
    def test_map_transforms_success_value(
        self, seed: int, transform: Callable[[int], int], expected: int
    ) -> None:
        """Map transforms the public success value with the supplied callable."""
        # Arrange / Act
        result = r[int].ok(seed).map(transform)

        # Assert
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=expected)

    def test_map_leaves_failure_untransformed(self) -> None:
        """Map preserves a failure and its original error message."""
        # Arrange
        failure = r[int].fail("boom")

        # Act
        result = failure.map(lambda value: value + 1)

        # Assert
        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="boom")

    def test_flat_map_chains_fallible_steps(self) -> None:
        """Flat map composes successive successful result-producing steps."""
        # Act
        result = (
            r[int]
            .ok(1)
            .flat_map(lambda value: r[int].ok(value + 1))
            .flat_map(lambda value: r[int].ok(value * 10))
        )

        # Assert
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=_FLAT_MAP_VALUE)

    def test_flat_map_short_circuits_on_first_failure(self) -> None:
        """Flat map preserves the first failure without executing later steps."""
        # Act
        result = (
            r[int]
            .ok(1)
            .flat_map(lambda _: r[int].fail("stopped"))
            .flat_map(lambda value: r[int].ok(value + 100))
        )

        # Assert
        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="stopped")

    @pytest.mark.parametrize(
        ("result", "expected"), [(r[str].ok("flext"), 5), (r[str].fail("missing"), 0)]
    )
    def test_map_or_returns_default_on_failure(
        self, result: p.Result[str], expected: int
    ) -> None:
        """Map or returns the mapped length or its declared failure default."""
        # Act / Assert
        tm.that(result.map_or(0, len), eq=expected)

    def test_recover_replaces_failure_with_value(self) -> None:
        """Recover converts a failure into the value returned by its handler."""
        # Act
        result = r[int].fail("boom").recover(lambda _error: _RECOVERED_VALUE)

        # Assert
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=_RECOVERED_VALUE)

    def test_map_error_rewrites_error_message(self) -> None:
        """Map error transforms only the public failure message."""
        # Act
        result = r[int].fail("boom").map_error(lambda message: message.upper())

        # Assert
        tm.that(result.failure, eq=True)
        tm.that(result.error, eq="BOOM")

    def test_value_access_on_failure_raises(self) -> None:
        """Reading the value channel of a failed result raises RuntimeError."""
        # Arrange
        failure = r[int].fail("no value here")

        # Act / Assert
        with pytest.raises(RuntimeError):
            _ = failure.value

    def test_flow_through_pipes_success_into_next_step(self) -> None:
        """Flow through sends a success value into the next result step."""

        def step(value: int) -> p.Result[int]:
            return r[int].ok(value + 1)

        # Act
        result = r[int].ok(1).flow_through(step)

        # Assert
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=_FLOW_VALUE)

    def test_not_found_factory_carries_structured_payload(self) -> None:
        """The not-found factory preserves its code, message, and resource data."""

        def fetch_profile_name(user_id: str) -> p.Result[str]:
            if user_id != "u-1":
                return e.fail_not_found("user", user_id)
            return r[str].ok("Ada")

        # Act
        found = fetch_profile_name("u-1")
        missing = fetch_profile_name("u-2")

        # Assert — success path
        tm.that(found.success, eq=True)
        tm.that(found.value, eq="Ada")

        # Assert — failure path exposes typed code + structured data publicly
        tm.that(missing.failure, eq=True)
        tm.that(missing.error_code, eq="NOT_FOUND_ERROR")
        tm.that(missing.error, eq="User 'u-2' not found")
        error_data = tm.not_none(missing.error_data)
        tm.that(error_data["resource_type"], eq="user")
        tm.that(error_data["resource_id"], eq="u-2")

    @pytest.mark.parametrize(
        ("raw_email", "expect_success", "expected_value"),
        [
            (" Ada@example.com ", True, "ada@example.com"),
            ("USER@Example.COM", True, "user@example.com"),
            (None, False, None),
            ("   ", False, None),
        ],
    )
    def test_validation_normalizes_or_fails(
        self, raw_email: str | None, expected_value: str | None, *, expect_success: bool
    ) -> None:
        """Validation normalizes meaningful email input and rejects empty input."""

        def require_email(candidate: str | None) -> p.Result[str]:
            if candidate is None:
                return e.fail_validation("email", error="cannot be None")
            normalized = candidate.strip().lower()
            if not normalized:
                return e.fail_validation("email", error="cannot be blank")
            return r[str].ok(normalized)

        # Act
        result = require_email(raw_email)

        # Assert
        tm.that(result.success, eq=expect_success)
        if expect_success:
            tm.that(result.value, eq=expected_value)
        else:
            tm.that(result.error_code, eq="VALIDATION_ERROR")
            tm.that(tm.not_none(result.error_data)["field"], eq="email")

    def test_timeout_exception_preserves_cause_and_context(self) -> None:
        """Timeout errors retain operation context, cause, and correlation id."""

        def fetch_remote_profile() -> str:
            socket_message = "socket stalled"
            timeout_message = "Remote profile lookup timed out"
            try:
                raise RuntimeError(socket_message)
            except RuntimeError as exc:
                raise e.TimeoutError(
                    timeout_message,
                    operation="fetch profile",
                    timeout_seconds=2.0,
                    auto_correlation=True,
                    context={"service": "profile-api"},
                ) from exc

        # Act / Assert
        with pytest.raises(e.TimeoutError) as raised:
            fetch_remote_profile()

        error = raised.value
        tm.that(error.operation, eq="fetch profile")
        tm.that(tm.not_none(error.__cause__), is_=RuntimeError)
        tm.that(error.correlation_id, none=False)

    def test_railway_decorator_wraps_return_in_success_result(self) -> None:
        """The railway decorator exposes a successful result for a plain return."""

        @d.railway()
        def increment(value: int) -> int:
            return value + 1

        # Act
        result = increment(3)

        # Assert
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=_INCREMENTED_VALUE)

    def test_combined_decorator_wraps_return_in_success_result(self) -> None:
        """The combined decorator exposes a successful result for a plain return."""

        @d.combined(operation_name="sum_values", railway_enabled=True, track_perf=False)
        def sum_values(values: list[int]) -> int:
            return sum(values)

        # Act
        result = sum_values([1, 2, 3])

        # Assert
        tm.that(result.success, eq=True)
        tm.that(result.value, eq=_COMBINED_SUM)
