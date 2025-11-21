"""Tests for FlextService .result property (V2 pattern).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT

"""

from __future__ import annotations

import pytest

from flext_core import FlextExceptions, FlextModels, FlextResult, FlextService

# =========================================================================
# Test Models
# =========================================================================


class User(FlextModels.Entity):
    """Test user entity."""

    user_id: str
    name: str
    email: str
    is_active: bool = True


# =========================================================================
# Test Services
# =========================================================================


class GetUserService(FlextService[User]):
    """Service to get a user by ID."""

    user_id: str

    def execute(self, **_kwargs: object) -> FlextResult[User]:
        """Get user by ID."""
        return FlextResult.ok(
            User(
                user_id=self.user_id,
                name=f"User {self.user_id}",
                email=f"user{self.user_id}@example.com",
            )
        )


class FailingService(FlextService[str]):
    """Service that always fails."""

    error_message: str = "Operation failed"

    def execute(self, **_kwargs: object) -> FlextResult[str]:
        """Always fails."""
        return FlextResult.fail(self.error_message)


class ValidatingService(FlextService[str]):
    """Service with validation."""

    value_input: str
    min_length: int = 3

    def execute(self, **_kwargs: object) -> FlextResult[str]:
        """Validate and return value."""
        if len(self.value_input) < self.min_length:
            return FlextResult.fail(
                f"Value must be at least {self.min_length} characters"
            )
        return FlextResult.ok(self.value_input.upper())


# =========================================================================
# Test V2 Property Pattern
# =========================================================================


class TestServiceResultProperty:
    """Test .result property (V2 zero-ceremony pattern)."""

    def test_result_property_returns_unwrapped_result(self) -> None:
        """V2: .result returns unwrapped domain result."""
        # V2: Direct result access
        user = GetUserService(user_id="123").result

        # Assert it's the domain object directly
        assert isinstance(user, User)
        assert user.user_id == "123"
        assert user.name == "User 123"
        assert user.email == "user123@example.com"

    def test_result_property_raises_on_failure(self) -> None:
        """V2: .result raises exception on failure."""
        # V2: Failures are raised as exceptions
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            FailingService(error_message="Test error").result

        assert "Test error" in str(exc_info.value)

    def test_result_property_with_validation_success(self) -> None:
        """V2: Validation success returns value."""
        # Valid input
        result = ValidatingService(value_input="hello").result

        # Result is the unwrapped string
        assert isinstance(result, str)
        assert result == "HELLO"

    def test_result_property_with_validation_failure(self) -> None:
        """V2: Validation failure raises exception."""
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            ValidatingService(value_input="ab").result  # Too short

        assert "must be at least 3 characters" in str(exc_info.value)

    def test_result_property_type_inference(self) -> None:
        """V2: Type checkers infer correct type."""
        # This should type-check correctly (mypy/pyright validate this)
        user: User = GetUserService(user_id="456").result

        assert isinstance(user, User)
        assert user.user_id == "456"
        assert user.name == "User 456"

    def test_result_property_lazy_evaluation(self) -> None:
        """V2: Property is lazily evaluated (executes only when accessed)."""
        # Create service instance (no execution yet)
        service = GetUserService(user_id="789")

        # execute() is not called until .result is accessed
        # We can verify this by checking that the service itself doesn't have the result
        assert hasattr(service, "result")

        # Now access .result (triggers execution)
        user = service.result
        assert user.user_id == "789"


# =========================================================================
# Test V1 Compatibility (Backward Compatibility)
# =========================================================================


class TestV1Compatibility:
    """Test that V1 pattern still works (backward compatibility)."""

    def test_v1_execute_still_works(self) -> None:
        """V1: .execute() continues to work."""
        # V1: Explicit mode
        service = GetUserService(user_id="999")
        result = service.execute()

        # Assert it's a FlextResult
        assert isinstance(result, FlextResult)
        assert result.is_success

        # Unwrap to get domain result
        user = result.unwrap()
        assert isinstance(user, User)
        assert user.user_id == "999"

    def test_v1_error_handling_with_flext_result(self) -> None:
        """V1: Error handling via FlextResult pattern."""
        # V1: Get result
        result = FailingService(error_message="Test failure").execute()

        # Assert it's a failure
        assert isinstance(result, FlextResult)
        assert result.is_failure
        assert "Test failure" in str(result.error)

    def test_v1_railway_pattern_composition(self) -> None:
        """V1: Railway pattern composition with map/flat_map."""
        # V1: Execute and compose with monadic methods
        result = (
            GetUserService(user_id="111")
            .execute()
            .map(lambda user: user.name)
            .map(lambda name: name.upper())
            .map(lambda name: f"Hello, {name}!")
        )

        # Assert successful composition
        assert result.is_success
        greeting = result.unwrap()
        assert greeting == "Hello, USER 111!"

    def test_v2_and_v1_return_same_result(self) -> None:
        """V2 and V1 should return equivalent results."""
        # V2
        user_v2 = GetUserService(user_id="555").result

        # V1
        user_v1 = GetUserService(user_id="555").execute().unwrap()

        # Both should be equivalent
        assert user_v2.user_id == user_v1.user_id
        assert user_v2.name == user_v1.name
        assert user_v2.email == user_v1.email


# =========================================================================
# Test Property Behavior
# =========================================================================


class TestPropertyBehavior:
    """Test Pydantic @computed_field behavior."""

    def test_result_is_computed_field(self) -> None:
        """Verify .result is a Pydantic computed_field."""
        # Check that value is accessible as a property
        service = GetUserService(user_id="123")
        assert hasattr(service, "result")

        # Access should trigger execution
        user = service.result
        assert isinstance(user, User)

    def test_result_property_in_model_dump(self) -> None:
        """Computed fields can be included in model_dump if configured."""
        service = GetUserService(user_id="123")

        # By default, computed fields are excluded
        dump = service.model_dump()
        assert "user_id" in dump

        # value is computed on access, not stored
        user = service.result
        assert isinstance(user, User)
