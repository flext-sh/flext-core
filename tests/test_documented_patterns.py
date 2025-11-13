"""Test all patterns documented in FLEXT_SERVICE_ARCHITECTURE.md V6.1.

This module validates ALL patterns documented in the architecture guide:
- Pattern 1: V1 Explícito (.execute().unwrap())
- Pattern 2: V2 Property (.result)
- Pattern 3: V2 Auto (auto_execute = True)
- Pattern 4: Railway Pattern em V1
- Pattern 5: Railway Pattern em V2 Property
- Pattern 6: Railway Pattern em V2 Auto (auto_execute = False)
- Pattern 7: Composição Monadic (map, and_then, filter, tap)
- Pattern 8: Error Handling Pythonic (try/except)
- Pattern 9: Infraestrutura Automática (config, logger, container)
- Pattern 10: Múltiplas Operações (operation field)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from typing import Any

import pytest
from pydantic import BaseModel

from flext_core import (
    FlextConfig,
    FlextContainer,
    FlextExceptions,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextService,
)

# ============================================================================
# Test Models
# ============================================================================


class User(FlextModels.Entity):
    """User domain model."""

    name: str
    email: str


class EmailResponse(BaseModel):
    """Email response model."""

    status: str
    message_id: str


# ============================================================================
# Test Services
# ============================================================================


class GetUserService(FlextService[User]):
    """Service to get user - V1/V2 compatible."""

    user_id: str

    def execute(self) -> FlextResult[User]:
        """Get user by ID."""
        if self.user_id == "invalid":
            return FlextResult.fail("User not found")

        return FlextResult.ok(
            User(
                unique_id=self.user_id,
                name=f"User {self.user_id}",
                email=f"user{self.user_id}@example.com",
            )
        )


class AutoGetUserService(FlextService[User]):
    """Service with auto_execute enabled - V2 Auto pattern."""

    auto_execute = True  # Enable V2 Auto pattern
    user_id: str

    def execute(self) -> FlextResult[User]:
        """Get user by ID."""
        if self.user_id == "invalid":
            return FlextResult.fail("User not found")

        return FlextResult.ok(
            User(
                unique_id=self.user_id,
                name=f"User {self.user_id}",
                email=f"user{self.user_id}@example.com",
            )
        )


class SendEmailService(FlextService[EmailResponse]):
    """Service to send email."""

    to: str
    subject: str

    def execute(self) -> FlextResult[EmailResponse]:
        """Send email."""
        if "@" not in self.to:
            return FlextResult.fail("Invalid email address")

        return FlextResult.ok(EmailResponse(status="sent", message_id=f"msg-{self.to}"))


class ValidationService(FlextService[dict[str, Any]]):
    """Service that validates input."""

    value: int

    def execute(self) -> FlextResult[dict[str, Any]]:
        """Validate value."""
        if self.value < 0:
            return FlextResult.fail("Value must be positive")

        if self.value > 100:
            return FlextResult.fail("Value must be <= 100")

        return FlextResult.ok({"valid": True, "value": self.value})


class MultiOperationService(FlextService[dict[str, Any]]):
    """Service with multiple operations."""

    operation: str
    value: int

    def execute(self) -> FlextResult[dict[str, Any]]:
        """Execute based on operation."""
        if self.operation == "double":
            return FlextResult.ok({"operation": "double", "result": self.value * 2})
        if self.operation == "square":
            return FlextResult.ok({"operation": "square", "result": self.value**2})
        if self.operation == "negate":
            return FlextResult.ok({"operation": "negate", "result": -self.value})

        return FlextResult.fail(f"Unknown operation: {self.operation}")


# ============================================================================
# Pattern 1: V1 Explícito (.execute().unwrap())
# ============================================================================


class TestPattern1V1Explicit:
    """Test Pattern 1: V1 Explícito (.execute().unwrap())."""

    def test_v1_explicit_success(self) -> None:
        """V1: Execute and unwrap on success."""
        # V1 Pattern: .execute().unwrap()
        result = GetUserService(user_id="123").execute()
        assert result.is_success
        user = result.unwrap()

        assert isinstance(user, User)
        assert user.entity_id == "123"
        assert user.name == "User 123"

    def test_v1_explicit_failure(self) -> None:
        """V1: Execute and check failure."""
        # V1 Pattern: .execute() returns FlextResult
        result = GetUserService(user_id="invalid").execute()
        assert result.is_failure
        assert result.error is not None
        assert "not found" in result.error.lower()

    def test_v1_explicit_with_if_check(self) -> None:
        """V1: Explicit error checking with if."""
        result = GetUserService(user_id="456").execute()

        if result.is_success:
            user = result.unwrap()
            assert user.entity_id == "456"
        else:
            pytest.fail("Should succeed")


# ============================================================================
# Pattern 2: V2 Property (.result)
# ============================================================================


class TestPattern2V2Property:
    """Test Pattern 2: V2 Property (.result)."""

    def test_v2_property_success(self) -> None:
        """V2 Property: Use .result for happy path."""
        # V2 Property Pattern: .result (68% less code)
        user = GetUserService(user_id="789").result

        assert isinstance(user, User)
        assert user.entity_id == "789"
        assert user.name == "User 789"

    def test_v2_property_failure_raises(self) -> None:
        """V2 Property: .result raises exception on failure."""
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            GetUserService(user_id="invalid").result

        assert "not found" in str(exc_info.value).lower()

    def test_v2_property_execute_still_available(self) -> None:
        """V2 Property: .execute() still works for railway pattern."""
        # V2 Property maintains backward compatibility
        result = GetUserService(user_id="123").execute()

        assert result.is_success
        user = result.unwrap()
        assert user.entity_id == "123"


# ============================================================================
# Pattern 3: V2 Auto (auto_execute = True)
# ============================================================================


class TestPattern3V2Auto:
    """Test Pattern 3: V2 Auto (auto_execute = True)."""

    def test_v2_auto_returns_value_directly(self) -> None:
        """V2 Auto: Instantiation returns unwrapped value (95% less code)."""
        # V2 Auto Pattern: Just instantiate (4 chars!)
        user = AutoGetUserService(user_id="999")

        # Returns User directly, not service instance
        assert isinstance(user, User)
        assert not isinstance(user, AutoGetUserService)
        assert user.entity_id == "999"

    def test_v2_auto_failure_raises(self) -> None:
        """V2 Auto: Failure raises exception."""
        with pytest.raises(FlextExceptions.BaseError):
            AutoGetUserService(user_id="invalid")

    def test_v2_auto_manual_service_returns_instance(self) -> None:
        """V2 Auto: Default (auto_execute=False) returns service instance."""
        # Regular service (auto_execute = False) returns instance
        service = GetUserService(user_id="123")

        assert isinstance(service, GetUserService)
        assert not isinstance(service, User)


# ============================================================================
# Pattern 4: Railway Pattern em V1
# ============================================================================


class TestPattern4RailwayV1:
    """Test Pattern 4: Railway Pattern em V1."""

    def test_v1_railway_map(self) -> None:
        """V1 Railway: map transformation."""
        result = GetUserService(user_id="123").execute().map(lambda user: user.email)

        assert result.is_success
        assert result.unwrap() == "user123@example.com"

    def test_v1_railway_flat_map(self) -> None:
        """V1 Railway: flat_map chaining (also known as and_then)."""
        pipeline = (
            GetUserService(user_id="123")
            .execute()
            .flat_map(
                lambda user: SendEmailService(to=user.email, subject="Hello").execute()
            )
        )

        assert pipeline.is_success
        email_response = pipeline.unwrap()
        assert email_response.status == "sent"

    def test_v1_railway_filter(self) -> None:
        """V1 Railway: filter with predicate."""
        result = (
            GetUserService(user_id="123")
            .execute()
            .filter(lambda user: user.entity_id == "123", "User ID mismatch")
        )

        assert result.is_success

        # Test filter failure
        filtered_fail = (
            GetUserService(user_id="123")
            .execute()
            .filter(lambda user: user.entity_id == "999", "ID does not match")
        )

        assert filtered_fail.is_failure
        assert "ID does not match" in filtered_fail.error

    def test_v1_railway_composition(self) -> None:
        """V1 Railway: Full composition pipeline."""
        pipeline = (
            GetUserService(user_id="123")
            .execute()
            .map(lambda user: user.email)
            .filter(lambda email: "@" in email, "Invalid email")
            .flat_map(
                lambda email: SendEmailService(to=email, subject="Test").execute()
            )
            .map(lambda response: response.status)
        )

        assert pipeline.is_success
        assert pipeline.unwrap() == "sent"


# ============================================================================
# Pattern 5: Railway Pattern em V2 Property
# ============================================================================


class TestPattern5RailwayV2Property:
    """Test Pattern 5: Railway Pattern em V2 Property."""

    def test_v2_property_can_use_execute_for_railway(self) -> None:
        """V2 Property: .execute() available for railway pattern."""
        # V2 Property: Use .result for happy path
        user_happy = GetUserService(user_id="123").result
        assert user_happy.entity_id == "123"

        # V2 Property: Use .execute() for railway pattern
        result = GetUserService(user_id="123").execute().map(lambda u: u.email)

        assert result.is_success
        assert result.unwrap() == "user123@example.com"

    def test_v2_property_railway_chaining(self) -> None:
        """V2 Property: Full railway chaining."""
        pipeline = (
            GetUserService(user_id="456")
            .execute()
            .flat_map(
                lambda user: SendEmailService(to=user.email, subject="Hello").execute()
            )
            .map(lambda response: response.message_id)
        )

        assert pipeline.is_success
        message_id = pipeline.unwrap()
        assert message_id.startswith("msg-")


# ============================================================================
# Pattern 6: Railway Pattern em V2 Auto (auto_execute = False)
# ============================================================================


class TestPattern6RailwayV2Auto:
    """Test Pattern 6: Railway Pattern em V2 Auto (auto_execute = False)."""

    def test_v2_auto_with_manual_mode_supports_railway(self) -> None:
        """V2 Auto: Manual mode (auto_execute=False) supports railway."""

        class ManualService(FlextService[User]):
            """Service with auto_execute = False for railway."""

            auto_execute = False  # Manual mode for railway
            user_id: str

            def execute(self) -> FlextResult[User]:
                """Get user."""
                return FlextResult.ok(
                    User(unique_id=self.user_id, name="Test", email="test@example.com")
                )

        # With auto_execute = False, returns service instance
        service = ManualService(user_id="789")
        assert isinstance(service, ManualService)

        # Can use railway pattern
        result = service.execute().map(lambda u: u.email)

        assert result.is_success
        assert result.unwrap() == "test@example.com"


# ============================================================================
# Pattern 7: Composição Monadic (map, and_then, filter, tap)
# ============================================================================


class TestPattern7MonadicComposition:
    """Test Pattern 7: Composição Monadic."""

    def test_monadic_map(self) -> None:
        """Monadic: map transforms value."""
        result = (
            GetUserService(user_id="123").execute().map(lambda user: user.name.upper())
        )

        assert result.unwrap() == "USER 123"

    def test_monadic_flat_map(self) -> None:
        """Monadic: flat_map chains operations (also known as and_then)."""
        pipeline = (
            GetUserService(user_id="123")
            .execute()
            .flat_map(lambda user: FlextResult.ok(user.email))
            .flat_map(
                lambda email: SendEmailService(to=email, subject="Test").execute()
            )
        )

        assert pipeline.is_success

    def test_monadic_filter(self) -> None:
        """Monadic: filter validates predicate."""
        result = (
            ValidationService(value=50)
            .execute()
            .filter(lambda data: data["value"] < 100, "Value too high")
        )

        assert result.is_success

    def test_monadic_tap(self) -> None:
        """Monadic: tap for side effects."""
        called = []

        result = (
            GetUserService(user_id="123")
            .execute()
            .tap(lambda user: called.append(user.entity_id))
            .map(lambda user: user.name)
        )

        assert result.is_success
        assert "123" in called

    def test_monadic_recover(self) -> None:
        """Monadic: recover provides default on failure."""
        result = (
            GetUserService(user_id="invalid")
            .execute()
            .recover(
                lambda _: User(
                    unique_id="default", name="Default", email="default@example.com"
                )
            )
        )

        assert result.is_success
        user = result.unwrap()
        assert user.entity_id == "default"

    def test_monadic_complex_pipeline(self) -> None:
        """Monadic: Complex pipeline with multiple operations."""
        executed_steps = []

        pipeline = (
            GetUserService(user_id="123")
            .execute()
            .tap(lambda _: executed_steps.append("1-got-user"))
            .map(lambda user: user.email)
            .tap(lambda _: executed_steps.append("2-extracted-email"))
            .filter(lambda email: "@" in email, "Invalid email")
            .tap(lambda _: executed_steps.append("3-validated-email"))
            .flat_map(
                lambda email: SendEmailService(to=email, subject="Test").execute()
            )
            .tap(lambda _: executed_steps.append("4-sent-email"))
            .map(lambda response: response.status)
        )

        assert pipeline.is_success
        assert pipeline.unwrap() == "sent"
        assert len(executed_steps) == 4


# ============================================================================
# Pattern 8: Error Handling Pythonic (try/except)
# ============================================================================


class TestPattern8ErrorHandling:
    """Test Pattern 8: Error Handling Pythonic."""

    def test_error_handling_try_except_v2_property(self) -> None:
        """Error Handling: try/except with V2 Property."""
        try:
            user = GetUserService(user_id="123").result
            assert user.entity_id == "123"
        except FlextExceptions.BaseError:
            pytest.fail("Should not raise")

    def test_error_handling_try_except_v2_property_failure(self) -> None:
        """Error Handling: try/except catches failure."""
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            GetUserService(user_id="invalid").result
        assert "not found" in str(exc_info.value).lower()

    def test_error_handling_try_except_v2_auto(self) -> None:
        """Error Handling: try/except with V2 Auto."""
        from typing import cast
        try:
            user = cast("User", AutoGetUserService(user_id="789"))
            assert user.entity_id == "789"
        except FlextExceptions.BaseError:
            pytest.fail("Should not raise")

    def test_error_handling_graceful_degradation(self) -> None:
        """Error Handling: Graceful degradation pattern."""
        try:
            user = GetUserService(user_id="123").result
            email = user.email
        except FlextExceptions.BaseError:
            email = "fallback@example.com"

        assert email == "user123@example.com"


# ============================================================================
# Pattern 9: Infraestrutura Automática (config, logger, container)
# ============================================================================


class TestPattern9AutomaticInfrastructure:
    """Test Pattern 9: Infraestrutura Automática."""

    def test_infrastructure_config_automatic(self) -> None:
        """Infrastructure: Config available automatically."""
        service = GetUserService(user_id="123")

        # Config is automatically available
        assert service.config is not None
        assert isinstance(service.config, FlextConfig)

    def test_infrastructure_logger_automatic(self) -> None:
        """Infrastructure: Logger available automatically."""
        service = GetUserService(user_id="123")

        # Logger is automatically available
        assert service.logger is not None
        assert isinstance(service.logger, FlextLogger)

    def test_infrastructure_container_automatic(self) -> None:
        """Infrastructure: Container available automatically."""
        service = GetUserService(user_id="123")

        # Container is automatically available
        assert service.container is not None
        assert isinstance(service.container, FlextContainer)

    def test_infrastructure_lazy_initialization(self) -> None:
        """Infrastructure: Properties are lazy."""
        service = GetUserService(user_id="123")

        # Properties exist but are lazily evaluated
        config1 = service.config
        config2 = service.config

        # Same instance (singleton)
        assert config1 is config2


# ============================================================================
# Pattern 10: Múltiplas Operações (operation field)
# ============================================================================


class TestPattern10MultipleOperations:
    """Test Pattern 10: Múltiplas Operações."""

    def test_multiple_operations_double(self) -> None:
        """Multiple Operations: Double operation."""
        result = MultiOperationService(operation="double", value=5).result

        assert result["operation"] == "double"
        assert result["result"] == 10

    def test_multiple_operations_square(self) -> None:
        """Multiple Operations: Square operation."""
        result = MultiOperationService(operation="square", value=4).result

        assert result["operation"] == "square"
        assert result["result"] == 16

    def test_multiple_operations_negate(self) -> None:
        """Multiple Operations: Negate operation."""
        result = MultiOperationService(operation="negate", value=7).result

        assert result["operation"] == "negate"
        assert result["result"] == -7

    def test_multiple_operations_invalid(self) -> None:
        """Multiple Operations: Invalid operation fails."""
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            MultiOperationService(operation="invalid", value=5).result

        assert "Unknown operation" in str(exc_info.value)

    def test_multiple_operations_with_railway(self) -> None:
        """Multiple Operations: Railway pattern with operations."""
        pipeline = (
            MultiOperationService(operation="double", value=5)
            .execute()
            .map(operator.itemgetter("result"))
            .flat_map(
                lambda result: MultiOperationService(
                    operation="square", value=result
                ).execute()
            )
            .map(operator.itemgetter("result"))
        )

        assert pipeline.is_success
        # (5 * 2) ** 2 = 100
        assert pipeline.unwrap() == 100


# ============================================================================
# Integration Tests: All Patterns Together
# ============================================================================


class TestAllPatternsIntegration:
    """Integration tests combining multiple patterns."""

    def test_v1_v2_property_v2_auto_interoperability(self) -> None:
        """All patterns work together seamlessly."""
        # V1: Explicit
        v1_result = GetUserService(user_id="123").execute()
        assert v1_result.is_success

        # V2 Property: Happy path
        v2_user = GetUserService(user_id="456").result
        assert v2_user.entity_id == "456"

        # V2 Auto: Zero ceremony
        from typing import cast
        auto_user = cast("User", AutoGetUserService(user_id="789"))
        assert auto_user.entity_id == "789"

        # All return same type
        assert isinstance(v1_result.unwrap(), User)
        assert isinstance(v2_user, User)
        assert isinstance(auto_user, User)

    def test_railway_pattern_works_in_all_versions(self) -> None:
        """Railway pattern works in V1, V2 Property, and V2 Auto (manual mode)."""
        # V1: Railway
        v1_pipeline = GetUserService(user_id="123").execute().map(lambda u: u.email)
        assert v1_pipeline.is_success

        # V2 Property: Railway via .execute()
        v2_pipeline = GetUserService(user_id="456").execute().map(lambda u: u.email)
        assert v2_pipeline.is_success

        # V2 Auto (manual): Railway
        class ManualService(FlextService[User]):
            auto_execute = False
            user_id: str

            def execute(self) -> FlextResult[User]:
                return FlextResult.ok(
                    User(unique_id=self.user_id, name="Test", email="test@example.com")
                )

        v2_auto_pipeline = ManualService(user_id="789").execute().map(lambda u: u.email)
        assert v2_auto_pipeline.is_success

    def test_complete_real_world_scenario(self) -> None:
        """Complete scenario using multiple patterns."""
        # Step 1: Get user (V2 Property)
        user = GetUserService(user_id="123").result

        # Step 2: Validate and send email (Railway V1)
        email_result = (
            SendEmailService(to=user.email, subject="Welcome")
            .execute()
            .filter(lambda r: r.status == "sent", "Email not sent")
            .map(lambda r: r.message_id)
        )

        assert email_result.is_success
        message_id = email_result.unwrap()
        assert message_id.startswith("msg-")

        # Step 3: Multiple operations (V2 Property)
        calc_result = MultiOperationService(operation="double", value=10).result
        assert calc_result["result"] == 20
