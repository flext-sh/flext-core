"""Behavioral integration tests for documented flext-core public patterns.

Every assertion targets the observable public contract of ``r`` (FlextResult),
``e`` (FlextExceptions) and ``d`` (FlextDecorators): return values, the ``r[T]``
success/failure outcome, structured error payloads and raised exception fields.
No private attribute, internal collaborator or implementation detail is inspected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from flext_tests import d, e, r

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.protocols import p


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
        self,
        seed: int,
        transform: Callable[[int], int],
        expected: int,
    ) -> None:
        # Arrange / Act
        result = r[int].ok(seed).map(transform)

        # Assert
        assert result.success
        assert result.value == expected

    def test_map_leaves_failure_untransformed(self) -> None:
        # Arrange
        failure = r[int].fail("boom")

        # Act
        result = failure.map(lambda value: value + 1)

        # Assert
        assert result.failure
        assert result.error == "boom"

    def test_flat_map_chains_fallible_steps(self) -> None:
        # Act
        result = (
            r[int]
            .ok(1)
            .flat_map(lambda value: r[int].ok(value + 1))
            .flat_map(lambda value: r[int].ok(value * 10))
        )

        # Assert
        assert result.success
        assert result.value == 20

    def test_flat_map_short_circuits_on_first_failure(self) -> None:
        # Act
        result = (
            r[int]
            .ok(1)
            .flat_map(lambda _: r[int].fail("stopped"))
            .flat_map(lambda value: r[int].ok(value + 100))
        )

        # Assert
        assert result.failure
        assert result.error == "stopped"

    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (r[str].ok("flext"), 5),
            (r[str].fail("missing"), 0),
        ],
    )
    def test_map_or_returns_default_on_failure(
        self,
        result: p.Result[str],
        expected: int,
    ) -> None:
        # Act / Assert
        assert result.map_or(0, len) == expected

    def test_recover_replaces_failure_with_value(self) -> None:
        # Act
        result = r[int].fail("boom").recover(lambda _error: 99)

        # Assert
        assert result.success
        assert result.value == 99

    def test_map_error_rewrites_error_message(self) -> None:
        # Act
        result = r[int].fail("boom").map_error(lambda message: message.upper())

        # Assert
        assert result.failure
        assert result.error == "BOOM"

    def test_value_access_on_failure_raises(self) -> None:
        # Arrange
        failure = r[int].fail("no value here")

        # Act / Assert
        with pytest.raises(RuntimeError):
            _ = failure.value

    def test_flow_through_pipes_success_into_next_step(self) -> None:
        def step(value: int) -> p.Result[int]:
            return r[int].ok(value + 1)

        # Act
        result = r[int].ok(1).flow_through(step)

        # Assert
        assert result.success
        assert result.value == 2

    def test_not_found_factory_carries_structured_payload(self) -> None:
        def fetch_profile_name(user_id: str) -> p.Result[str]:
            if user_id != "u-1":
                return e.fail_not_found("user", user_id)
            return r[str].ok("Ada")

        # Act
        found = fetch_profile_name("u-1")
        missing = fetch_profile_name("u-2")

        # Assert — success path
        assert found.success
        assert found.value == "Ada"

        # Assert — failure path exposes typed code + structured data publicly
        assert missing.failure
        assert missing.error_code == "NOT_FOUND_ERROR"
        assert missing.error == "User 'u-2' not found"
        assert missing.error_data is not None
        assert missing.error_data["resource_type"] == "user"
        assert missing.error_data["resource_id"] == "u-2"

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
        self,
        raw_email: str | None,
        expect_success: bool,
        expected_value: str | None,
    ) -> None:
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
        assert result.success is expect_success
        if expect_success:
            assert result.value == expected_value
        else:
            assert result.error_code == "VALIDATION_ERROR"
            assert result.error_data is not None
            assert result.error_data["field"] == "email"

    def test_timeout_exception_preserves_cause_and_context(self) -> None:
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
        assert error.operation == "fetch profile"
        assert error.__cause__ is not None
        assert isinstance(error.__cause__, RuntimeError)
        assert error.correlation_id is not None

    def test_railway_decorator_wraps_return_in_success_result(self) -> None:
        @d.railway()
        def increment(value: int) -> int:
            return value + 1

        # Act
        result = increment(3)

        # Assert
        assert result.success
        assert result.value == 4

    def test_combined_decorator_wraps_return_in_success_result(self) -> None:
        @d.combined(
            operation_name="sum_values",
            railway_enabled=True,
            track_perf=False,
        )
        def sum_values(values: list[int]) -> int:
            return sum(values)

        # Act
        result = sum_values([1, 2, 3])

        # Assert
        assert result.success
        assert result.value == 6
