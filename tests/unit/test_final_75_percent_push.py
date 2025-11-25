"""Final push to 75% coverage - simple, focused tests.

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
        r2 = r.flat_map(lambda _: FlextResult[int].fail("error"))
        assert r2.is_failure

    def test_result_lash_on_success(self) -> None:
        """Test lash passes through success."""
        r = FlextResult[int].ok(42)
        r2 = r.lash(lambda _: FlextResult[int].ok(99))
        assert r2.value == 42

    def test_result_lash_on_failure(self) -> None:
        """Test lash recovers failure."""
        r = FlextResult[int].fail("error")
        r2 = r.lash(lambda _: FlextResult[int].ok(99))
        assert r2.is_success
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

    def test_container_clear_all(self) -> None:
        """Test container clear_all."""
        c = FlextContainer()
        c.with_service("test", "value")
        c.clear_all()
        r = c.get("test")
        assert r.is_failure

    def test_container_unregister(self) -> None:
        """Test container unregister."""
        c = FlextContainer()
        c.with_service("test", "value")
        c.unregister("test")
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
    def test_utilities_id(self) -> None:
        """Test ID generation."""
        id1 = FlextUtilities.Generators.generate_id()
        id2 = FlextUtilities.Generators.generate_id()
        assert id1 != id2
        assert len(id1) == 36

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

    def test_result_repr(self) -> None:
        """Test result repr."""
        r = FlextResult[int].ok(42)
        repr_str = repr(r)
        assert "FlextResult" in repr_str

    def test_result_filter(self) -> None:
        """Test filter method."""
        r = FlextResult[int].ok(42)
        r2 = r.filter(lambda x: x > 0)
        assert r2.is_success
        assert r2.value == 42

        r3 = r.filter(lambda x: x > 100)
        assert r3.is_failure

    def test_result_safe_factory(self) -> None:
        """Test safe factory method."""

        @FlextResult.safe
        def divide(a: int, b: int) -> int:
            return a // b

        r = divide(10, 2)
        assert r.is_success
        assert r.value == 5

        r2 = divide(10, 0)
        assert r2.is_failure


__all__ = ["TestCoveragePush75Percent"]
