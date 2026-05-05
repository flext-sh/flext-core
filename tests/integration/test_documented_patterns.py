"""Documented pattern integration smoke tests."""

from __future__ import annotations

import pytest

from tests import d, e, p, r


class TestsFlextDocumentedPatterns:
    def test_result_map_pattern(self) -> None:
        result = r[int].ok(1).map(lambda value: value + 1)
        assert result.success
        assert result.value == 2

    def test_result_map_or_pattern(self) -> None:
        success_length = r[str].ok("flext").map_or(0, len)
        failure_length = r[str].fail("missing").map_or(0, len)

        assert success_length == 5
        assert failure_length == 0

    def test_result_flow_through_pattern(self) -> None:
        def step(value: int) -> p.Result[int]:
            return r[int].ok(value + 1)

        result = r[int].ok(1).flow_through(step)
        assert result.success
        assert result.value == 2

    def test_exception_factory_boundary_pattern(self) -> None:
        def fetch_profile_name(user_id: str) -> p.Result[str]:
            if user_id != "u-1":
                return e.fail_not_found("user", user_id)
            return r[str].ok("Ada")

        missing = fetch_profile_name("u-2")

        assert missing.failure
        assert missing.error_data is not None
        assert missing.error_data["resource_id"] == "u-2"

    def test_none_handling_pattern(self) -> None:
        def require_email(raw_email: str | None) -> p.Result[str]:
            if raw_email is None:
                return e.fail_validation("email", error="cannot be None")
            normalized = raw_email.strip().lower()
            if not normalized:
                return e.fail_validation("email", error="cannot be blank")
            return r[str].ok(normalized)

        normalized = require_email(" Ada@example.com ")
        missing = require_email(None)
        blank = require_email("  ")

        assert normalized.success
        assert normalized.value == "ada@example.com"
        assert missing.failure
        assert blank.failure

    def test_exception_propagation_pattern(self) -> None:
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

        with pytest.raises(e.TimeoutError) as raised:
            fetch_remote_profile()

        error = raised.value
        assert error.operation == "fetch profile"
        assert error.__cause__ is not None
        assert error.correlation_id is not None

    def test_combined_decorator_pattern(self) -> None:
        @d.combined(
            operation_name="sum_values",
            railway_enabled=True,
            track_perf=False,
        )
        def sum_values(values: list[int]) -> int:
            return sum(values)

        result = sum_values([1, 2, 3])

        assert result.success
        assert result.value == 6
