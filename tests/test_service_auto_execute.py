"""Tests for FlextService auto_execute feature (V2 Zero Ceremony)."""

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
    email: str | None = None

    @property
    def id(self) -> str:
        """Entity ID."""
        return self.user_id


# =========================================================================
# Test Services
# =========================================================================


class ManualUserService(FlextService[User]):
    """Manual execution service (default behavior)."""

    user_id: str

    def execute(self) -> FlextResult[User]:
        """Get user by ID."""
        return FlextResult.ok(
            User(
                user_id=self.user_id,
                name=f"User {self.user_id}",
                email=f"user{self.user_id}@example.com",
            )
        )


class AutoUserService(FlextService[User]):
    """Auto-execution service (zero ceremony)."""

    auto_execute = True  # Enable auto-execution
    user_id: str

    def execute(self) -> FlextResult[User]:
        """Get user by ID."""
        return FlextResult.ok(
            User(
                user_id=self.user_id,
                name=f"User {self.user_id}",
                email=f"user{self.user_id}@example.com",
            )
        )


class FailingAutoService(FlextService[str]):
    """Auto-execution service that fails."""

    auto_execute = True
    error_message: str = "Test error"

    def execute(self) -> FlextResult[str]:
        """Always fails."""
        return FlextResult.fail(self.error_message)


# =========================================================================
# Test Auto-Execution Pattern
# =========================================================================


class TestAutoExecution:
    """Test auto_execute class attribute."""

    def test_manual_service_returns_instance(self) -> None:
        """Default: Returns service instance, not result."""
        service = ManualUserService(user_id="123")

        # Should be service instance
        assert isinstance(service, ManualUserService)
        assert isinstance(service, FlextService)

        # Can call execute() manually
        result = service.execute()
        assert result.is_success
        user = result.unwrap()
        assert isinstance(user, User)
        assert user.user_id == "123"

    def test_auto_service_returns_result_directly(self) -> None:
        """auto_execute=True: Returns unwrapped result directly."""
        # This returns User directly, not service instance!
        user = AutoUserService(user_id="456")

        # Should be User, not service
        assert isinstance(user, User)
        assert not isinstance(user, AutoUserService)
        assert user.user_id == "456"
        assert user.name == "User 456"

    def test_auto_service_raises_on_failure(self) -> None:
        """auto_execute=True: Failures raise exception."""
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            FailingAutoService(error_message="Custom error")

        assert "Custom error" in str(exc_info.value)

    def test_manual_service_can_use_result_property(self) -> None:
        """Manual service can use .result property."""
        service = ManualUserService(user_id="789")

        # Access .result property
        user = service.result

        assert isinstance(user, User)
        assert user.user_id == "789"

    def test_auto_vs_manual_same_domain_result(self) -> None:
        """Auto and manual services produce same domain result."""
        # Manual service
        manual_user = ManualUserService(user_id="999").result

        # Auto service (returns User directly due to auto_execute=True)
        auto_user = AutoUserService(user_id="999")

        # Same domain result
        assert isinstance(auto_user, User)
        assert manual_user.user_id == auto_user.user_id
        assert manual_user.name == auto_user.name
        assert manual_user.email == auto_user.email


# =========================================================================
# Test Backward Compatibility
# =========================================================================


class TestBackwardCompatibility:
    """Ensure auto_execute doesn't break existing code."""

    def test_default_auto_execute_is_false(self) -> None:
        """Default auto_execute is False for backward compatibility."""
        assert ManualUserService.auto_execute is False

    def test_existing_services_unchanged(self) -> None:
        """Existing services without auto_execute still work."""
        # This should return service instance (default behavior)
        service = ManualUserService(user_id="123")
        assert isinstance(service, ManualUserService)

        # Can still use V1 pattern
        result = service.execute()
        assert result.is_success

        # Can still use V2 property pattern
        user = service.result
        assert isinstance(user, User)
