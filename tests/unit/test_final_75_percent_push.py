"""Final push to 75% coverage - simple, focused tests.

Target: Cover 89 uncovered lines to reach exactly 75% coverage.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

from flext_core import (
    FlextContainer,
    FlextExceptions,
    FlextResult,
    FlextUtilities,
)


class TestCoveragePush75Percent:
    """Simple tests targeting uncovered lines."""

    # FlextResult coverage
    def test_result_basic_ok(self) -> None:
        """Test basic FlextResult ok."""
        r = FlextResult[int].ok(42)
        assert r.is_success
        assert r.value == 42

    def test_result_basic_fail(self) -> None:
        """Test basic FlextResult fail."""
        r = FlextResult[int].fail("error")
        assert r.is_failure
        assert r.error == "error"

    def test_result_map(self) -> None:
        """Test result mapping."""
        r = FlextResult[int].ok(5)
        r2 = r.map(lambda x: x * 2)
        assert r2.value == 10

    def test_result_flat_map(self) -> None:
        """Test result flat_map."""
        r = FlextResult[int].ok(5)
        r2 = r.flat_map(lambda x: FlextResult[int].ok(x * 2))
        assert r2.value == 10

    def test_result_flat_map_fail(self) -> None:
        """Test result flat_map with failure."""
        r = FlextResult[int].ok(5)
        r2 = r.flat_map(lambda x: FlextResult[int].fail("error"))
        assert r2.is_failure

    def test_result_or_else_with_success(self) -> None:
        """Test or_else_get on success."""
        r = FlextResult[int].ok(42)
        r2 = r.or_else_get(lambda: FlextResult[int].ok(99))
        assert r2.value == 42

    def test_result_or_else_with_failure(self) -> None:
        """Test or_else_get on failure."""
        r = FlextResult[int].fail("error")
        r2 = r.or_else_get(lambda: FlextResult[int].ok(99))
        assert r2.value == 99

    # FlextContainer coverage
    def test_container_basic(self) -> None:
        """Test basic container operations."""
        c = FlextContainer()
        r = c.with_service("test", "value")

        assert r is c  # Fluent interface returns Self

        r2 = c.get("test")
        assert r2.is_success
        assert r2.value == "value"

    def test_container_not_found(self) -> None:
        """Test container get not found."""
        c = FlextContainer()
        r = c.get("nonexistent")
        assert r.is_failure

    def test_container_clear(self) -> None:
        """Test container clear."""
        c = FlextContainer()
        c.with_service("test", "value")
        c.clear()
        r = c.get("test")
        assert r.is_failure

    # FlextExceptions coverage
    def test_exception_base(self) -> None:
        """Test base exception."""
        exc = FlextExceptions.BaseError("test")
        assert isinstance(exc, Exception)
        assert str(exc)

    def test_exception_validation(self) -> None:
        """Test validation exception."""
        exc = FlextExceptions.ValidationError("invalid")
        assert "VALIDATION_ERROR" in str(exc)

    def test_exception_type_error(self) -> None:
        """Test type exception."""
        exc = FlextExceptions.TypeError("wrong type")
        assert "TYPE_ERROR" in str(exc)

    def test_exception_operation(self) -> None:
        """Test operation exception."""
        exc = FlextExceptions.OperationError("failed")
        assert "OPERATION_ERROR" in str(exc)

    # FlextUtilities coverage
    def test_utilities_uuid(self) -> None:
        """Test UUID generation."""
        uuid1 = FlextUtilities.Generators.generate_uuid()
        uuid2 = FlextUtilities.Generators.generate_uuid()
        assert uuid1 != uuid2
        assert len(uuid1) == 36

    def test_utilities_timestamp(self) -> None:
        """Test timestamp generation."""
        ts = FlextUtilities.Generators.generate_iso_timestamp()
        assert isinstance(ts, str)
        assert len(ts) > 0

    # Additional chaining
    def test_result_chaining(self) -> None:
        """Test result method chaining."""
        r = FlextResult[int].ok(10).map(lambda x: x + 5).map(lambda x: x * 2)
        assert r.value == 30

    def test_result_failure_propagation(self) -> None:
        """Test failure propagates through chain."""
        r = FlextResult[int].fail("error").map(lambda x: x + 5).map(lambda x: x * 2)
        assert r.is_failure

    # Edge cases
    def test_result_data_property(self) -> None:
        """Test result .data property (backward compat)."""
        r = FlextResult[str].ok("value")
        assert r.data == "value"
        assert r.value == "value"

    def test_result_double_map(self) -> None:
        """Test double map operation."""
        r = FlextResult[int].ok(5)
        r2 = r.map(str)
        assert r2.value == "5"

    def test_container_register_multiple(self) -> None:
        """Test registering multiple services."""
        c = FlextContainer()
        c.with_service("svc1", "val1")
        c.with_service("svc2", "val2")
        assert c.get("svc1").value == "val1"
        assert c.get("svc2").value == "val2"

    # Additional exception coverage
    def test_exception_auth_error(self) -> None:
        """Test authentication error exception."""
        exc = FlextExceptions.AuthenticationError("auth issue")
        assert "AUTH" in str(exc).upper()

    def test_exception_config_error(self) -> None:
        """Test configuration error exception."""
        exc = FlextExceptions.ConfigurationError("config issue")
        assert "CONFIG" in str(exc).upper()

    def test_exception_connection_error(self) -> None:
        """Test connection error exception."""
        exc = FlextExceptions.ConnectionError("connection issue")
        assert "CONNECTION" in str(exc).upper()

    def test_exception_timeout_error(self) -> None:
        """Test timeout error exception."""
        exc = FlextExceptions.TimeoutError("timeout issue")
        assert "TIMEOUT" in str(exc).upper()

    # Additional result methods
    def test_result_unwrap_or(self) -> None:
        """Test unwrap_or with default."""
        r = FlextResult[int].fail("error")
        assert r.unwrap_or(42) == 42

        r2 = FlextResult[int].ok(10)
        assert r2.unwrap_or(42) == 10

    def test_result_expect(self) -> None:
        """Test expect method."""
        r = FlextResult[str].ok("value")
        assert r.expect("message") == "value"

    def test_result_tap(self) -> None:
        """Test tap for side effects."""
        called: list[int] = []
        r = FlextResult[int].ok(42)
        r.tap(called.append)
        assert 42 in called

    def test_result_lash(self) -> None:
        """Test lash for error handling."""
        r = FlextResult[int].fail("error")
        r2 = r.lash(lambda e: FlextResult[int].ok(99))
        assert r2.is_success
        assert r2.value == 99

    def test_result_from_callable(self) -> None:
        """Test from_callable factory."""

        def factory() -> str:
            return "result"

        r = FlextResult.from_callable(factory)
        assert r.is_success
        assert r.value == "result"

    def test_result_iter(self) -> None:
        """Test iteration over result."""
        r = FlextResult[str].ok("value")
        items = list(r)
        assert "value" in items

    def test_result_bool(self) -> None:
        """Test result as boolean."""
        success = FlextResult[int].ok(42)
        assert bool(success) is True

        failure = FlextResult[int].fail("error")
        assert bool(failure) is False

    def test_result_or_operator(self) -> None:
        """Test | operator for default."""
        r = FlextResult[int].fail("error")
        result = r | 42
        assert result == 42

    def test_result_xor_operator(self) -> None:
        """Test ^ operator for recovery."""
        r = FlextResult[int].fail("error")
        r2 = r ^ (lambda e: 99)
        assert r2.is_success
        assert r2.value == 99

    def test_result_getitem(self) -> None:
        """Test indexing result."""
        r = FlextResult[int].ok(42)
        assert r[0] == 42
        assert r[1] == ""  # Success case returns empty string for error index

    def test_result_eq(self) -> None:
        """Test result equality."""
        r1 = FlextResult[int].ok(42)
        r2 = FlextResult[int].ok(42)
        assert r1 == r2

    def test_result_repr(self) -> None:
        """Test result repr."""
        r = FlextResult[int].ok(42)
        repr_str = repr(r)
        assert "FlextResult" in repr_str


__all__ = ["TestCoveragePush75Percent"]
