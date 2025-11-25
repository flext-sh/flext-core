"""Test all patterns documented in FLEXT_SERVICE_ARCHITECTURE.md V6.1.

This module validates ALL patterns documented in the architecture guide using advanced Python 3.13 patterns,
factories, and helpers to reduce code size while maintaining and expanding functionality. Tests all edge cases
with minimal code duplication through unified class architecture and reusable test factories.

Patterns tested:
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

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
# Test Models and Factories
# ============================================================================


# Test Models
class User(FlextModels.Entity):
    """User domain model."""

    name: str
    email: str


class EmailResponse(BaseModel):
    """Email response model."""

    status: str
    message_id: str


if TYPE_CHECKING:
    from typing import TypeAlias

    UserOrStr: TypeAlias = User | str


@dataclass(frozen=True, slots=True)
class ServiceTestCase:
    """Factory for service test cases to reduce duplication."""

    user_id: str
    expected_success: bool = True
    expected_error: str | None = None
    description: str = field(default="", compare=False)

    def create_user_service(self) -> GetUserService:
        """Create GetUserService instance for this test case."""
        return GetUserService(user_id=self.user_id)

    def create_auto_user_service(self) -> AutoGetUserService:
        """Create AutoGetUserService instance for this test case."""
        return AutoGetUserService(user_id=self.user_id)


@dataclass(frozen=True, slots=True)
class RailwayTestCase:
    """Factory for railway pattern test cases."""

    user_ids: list[str]
    operations: list[str] = field(default_factory=list)
    expected_pipeline_length: int = 1
    should_fail_at: int | None = None
    description: str = field(default="", compare=False)

    def execute_v1_pipeline(self) -> FlextResult[object]:
        """Execute V1 railway pipeline for this test case."""
        if not self.user_ids:
            return FlextResult.fail("No user IDs provided")

        # Start with first user
        result: FlextResult[object] = cast(
            "FlextResult[object]", GetUserService(user_id=self.user_ids[0]).execute()
        )

        # Apply operations if specified
        for op in self.operations:
            if op == "get_email":
                result = result.map(lambda user: cast("User", user).email)  # type: ignore[arg-type]
            elif op == "send_email":
                result = result.flat_map(
                    lambda email: cast(
                        "FlextResult[object]",
                        SendEmailService(
                            to=cast("str", email), subject="Test"
                        ).execute(),
                    )
                )
            elif op == "get_status":
                result = result.map(
                    lambda response: cast("EmailResponse", response).status
                )

        return result

    def execute_v2_pipeline(self) -> User | str:
        """Execute V2 property pipeline for this test case."""
        if not self.user_ids:
            msg = "No user IDs provided"
            raise FlextExceptions.BaseError(msg)

        # Start with first user
        user: Any = GetUserService(user_id=self.user_ids[0]).result

        # Apply operations if specified
        for op in self.operations:
            if op == "get_email":
                user = cast("User", user).email
            elif op == "send_email":
                response = SendEmailService(to=cast("str", user), subject="Test").result  # type: ignore[assignment]
                user = cast("str", response.status)  # type: ignore[attr-defined]

        return user


class TestFactories:
    """Centralized test case factories for all patterns."""

    @staticmethod
    def success_cases() -> list[ServiceTestCase]:
        """Generate success test cases."""
        return [
            ServiceTestCase(user_id="123", description="Valid user ID"),
            ServiceTestCase(user_id="456", description="Another valid user ID"),
            ServiceTestCase(user_id="789", description="Third valid user ID"),
        ]

    @staticmethod
    def failure_cases() -> list[ServiceTestCase]:
        """Generate failure test cases."""
        return [
            ServiceTestCase(
                user_id="invalid",
                expected_success=False,
                expected_error="not found",
                description="Invalid user ID",
            ),
            ServiceTestCase(
                user_id="",
                expected_success=False,
                expected_error="not found",
                description="Empty user ID",
            ),
        ]

    @staticmethod
    def railway_success_cases() -> list[RailwayTestCase]:
        """Generate successful railway pattern test cases."""
        return [
            RailwayTestCase(user_ids=["123"], description="Simple user retrieval"),
            RailwayTestCase(
                user_ids=["456"],
                operations=["get_email"],
                expected_pipeline_length=2,
                description="User to email transformation",
            ),
            RailwayTestCase(
                user_ids=["789"],
                operations=["get_email", "send_email", "get_status"],
                expected_pipeline_length=4,
                description="Full pipeline: user -> email -> send -> status",
            ),
        ]

    @staticmethod
    def multi_operation_cases() -> list[tuple[str, int, Any]]:
        """Generate multiple operation test cases."""
        return [
            ("double", 5, {"operation": "double", "result": 10}),
            ("square", 4, {"operation": "square", "result": 16}),
            ("negate", 7, {"operation": "negate", "result": -7}),
            ("double", 0, {"operation": "double", "result": 0}),
            ("square", 1, {"operation": "square", "result": 1}),
        ]


# ============================================================================
# Test Services
# ============================================================================


class GetUserService(FlextService[User]):
    """Service to get user - V1/V2 compatible."""

    user_id: str

    def execute(self, **_kwargs: object) -> FlextResult[User]:
        """Get user by ID."""
        if self.user_id in {"invalid", ""}:
            return FlextResult.fail("User not found")

        return FlextResult.ok(
            User(
                unique_id=self.user_id,
                name=f"User {self.user_id}",
                email=f"user{self.user_id}@example.com",
            ),
        )


class AutoGetUserService(FlextService[User]):
    """Service with auto_execute enabled - V2 Auto pattern."""

    auto_execute = True  # Enable V2 Auto pattern
    user_id: str

    def execute(self, **_kwargs: object) -> FlextResult[User]:
        """Get user by ID."""
        if self.user_id in {"invalid", ""}:
            return FlextResult.fail("User not found")

        return FlextResult.ok(
            User(
                unique_id=self.user_id,
                name=f"User {self.user_id}",
                email=f"user{self.user_id}@example.com",
            ),
        )


class SendEmailService(FlextService[EmailResponse]):
    """Service to send email."""

    to: str
    subject: str

    def execute(self, **_kwargs: object) -> FlextResult[EmailResponse]:
        """Send email."""
        if "@" not in self.to:
            return FlextResult.fail("Invalid email address")

        return FlextResult.ok(EmailResponse(status="sent", message_id=f"msg-{self.to}"))


class ValidationService(FlextService[dict[str, Any]]):
    """Service that validates input."""

    value: int

    def execute(self, **_kwargs: object) -> FlextResult[dict[str, Any]]:
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

    def execute(self, **_kwargs: object) -> FlextResult[dict[str, Any]]:
        """Execute based on operation."""
        match self.operation:
            case "double":
                return FlextResult.ok({"operation": "double", "result": self.value * 2})
            case "square":
                return FlextResult.ok({"operation": "square", "result": self.value**2})
            case "negate":
                return FlextResult.ok({"operation": "negate", "result": -self.value})
            case _:
                return FlextResult.fail(f"Unknown operation: {self.operation}")


# ============================================================================
# Pattern 1: V1 Explícito (.execute().unwrap())
# ============================================================================


class TestPattern1V1Explicit:
    """Test Pattern 1: V1 Explícito (.execute().unwrap())."""

    @pytest.mark.parametrize("case", TestFactories.success_cases())
    def test_v1_explicit_success(self, case: ServiceTestCase) -> None:
        """V1: Execute and unwrap on success for various cases."""
        service = case.create_user_service()
        result = service.execute()

        assert result.is_success
        user = result.unwrap()

        assert isinstance(user, User)
        user_obj = cast("User", user)
        assert user_obj.user_id == case.user_id  # type: ignore[attr-defined]
        assert user_obj.name == f"User {case.user_id}"

    @pytest.mark.parametrize("case", TestFactories.failure_cases())
    def test_v1_explicit_failure(self, case: ServiceTestCase) -> None:
        """V1: Execute and check failure for various invalid cases."""
        service = case.create_user_service()
        result = service.execute()

        assert result.is_failure
        error_msg = result.error
        assert error_msg is not None
        expected = case.expected_error
        assert expected is not None
        assert expected in error_msg.lower()

    @pytest.mark.parametrize("case", TestFactories.success_cases())
    def test_v1_explicit_with_if_check(self, case: ServiceTestCase) -> None:
        """V1: Explicit error checking with if."""
        result = case.create_user_service().execute()

        if result.is_success:
            user = result.unwrap()
            user_obj = cast("User", user)
            assert user_obj.user_id == case.user_id  # type: ignore[attr-defined]
        else:
            pytest.fail("Should succeed")


# ============================================================================
# Pattern 2: V2 Property (.result)
# ============================================================================


class TestPattern2V2Property:
    """Test Pattern 2: V2 Property (.result)."""

    @pytest.mark.parametrize("case", TestFactories.success_cases())
    def test_v2_property_success(self, case: ServiceTestCase) -> None:
        """V2 Property: Use .result for happy path."""
        user = case.create_user_service().result

        assert isinstance(user, User)
        user_obj = cast("User", user)
        assert user_obj.user_id == case.user_id  # type: ignore[attr-defined]
        assert user_obj.name == f"User {case.user_id}"

    @pytest.mark.parametrize("case", TestFactories.failure_cases())
    def test_v2_property_failure_raises(self, case: ServiceTestCase) -> None:
        """V2 Property: .result raises exception on failure."""
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            case.create_user_service().result

        error_str = str(exc_info.value).lower()
        assert case.expected_error is not None
        assert case.expected_error in error_str

    @pytest.mark.parametrize("case", TestFactories.success_cases())
    def test_v2_property_execute_still_available(self, case: ServiceTestCase) -> None:
        """V2 Property: .execute() still works for railway pattern."""
        result = case.create_user_service().execute()

        assert result.is_success
        user = result.unwrap()
        user_obj = cast("User", user)
        assert user_obj.user_id == case.user_id  # type: ignore[attr-defined]


# ============================================================================
# Pattern 3: V2 Auto (auto_execute = True)
# ============================================================================


class TestPattern3V2Auto:
    """Test Pattern 3: V2 Auto (auto_execute = True)."""

    @pytest.mark.parametrize("case", TestFactories.success_cases())
    def test_v2_auto_returns_value_directly(self, case: ServiceTestCase) -> None:
        """V2 Auto: Instantiation returns unwrapped value directly."""
        user = case.create_auto_user_service()

        # Returns User directly, not service instance
        assert isinstance(user, User)
        assert not isinstance(user, AutoGetUserService)
        user_obj = cast("User", user)
        assert user_obj.user_id == case.user_id  # type: ignore[attr-defined]

    @pytest.mark.parametrize("case", TestFactories.failure_cases())
    def test_v2_auto_failure_raises(self, case: ServiceTestCase) -> None:
        """V2 Auto: Failure raises exception."""
        with pytest.raises(FlextExceptions.BaseError):
            case.create_auto_user_service()

    @pytest.mark.parametrize("case", TestFactories.success_cases())
    def test_v2_auto_manual_service_returns_instance(
        self, case: ServiceTestCase
    ) -> None:
        """V2 Auto: Default (auto_execute=False) returns service instance."""
        service = case.create_user_service()

        assert isinstance(service, GetUserService)
        assert not isinstance(service, User)


# ============================================================================
# Pattern 4: Railway Pattern em V1
# ============================================================================


class TestPattern4RailwayV1:
    """Test Pattern 4: Railway Pattern em V1."""

    @pytest.mark.parametrize("case", TestFactories.railway_success_cases())
    def test_v1_railway_complex_pipeline(self, case: RailwayTestCase) -> None:
        """V1 Railway: Full composition pipeline with various operations."""
        result = case.execute_v1_pipeline()

        assert result.is_success
        # Verify pipeline executed all expected steps
        if "get_status" in case.operations:
            assert result.unwrap() == "sent"
        elif "get_email" in case.operations:
            email = cast("str", result.unwrap())
            assert "@" in email
        else:
            assert isinstance(result.unwrap(), User)


# ============================================================================
# Pattern 5: Railway Pattern em V2 Property
# ============================================================================


class TestPattern5RailwayV2Property:
    """Test Pattern 5: Railway Pattern em V2 Property."""

    @pytest.mark.parametrize("case", TestFactories.railway_success_cases())
    def test_v2_property_can_use_execute_for_railway(
        self, case: RailwayTestCase
    ) -> None:
        """V2 Property: .execute() available for railway pattern."""
        # V2 Property: Use .result for happy path
        user = GetUserService(user_id="123").result
        assert user.user_id == "123"  # type: ignore[attr-defined]

        # V2 Property: Use .execute() for railway pattern
        result = GetUserService(user_id="123").execute().map(lambda u: u.email)

        assert result.is_success
        assert result.unwrap() == "user123@example.com"

    @pytest.mark.parametrize("case", TestFactories.railway_success_cases())
    def test_v2_property_railway_chaining(self, case: RailwayTestCase) -> None:
        """V2 Property: Full railway chaining."""
        pipeline = (
            GetUserService(user_id="456")
            .execute()
            .flat_map(
                lambda user: cast(
                    "FlextResult[object]",
                    SendEmailService(to=user.email, subject="Hello").execute(),
                ),
            )
            .map(lambda response: cast("EmailResponse", response).message_id)
        )

        assert pipeline.is_success
        message_id = cast("str", pipeline.unwrap())
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

            def execute(self, **_kwargs: object) -> FlextResult[User]:
                """Get user."""
                return FlextResult.ok(
                    User(unique_id=self.user_id, name="Test", email="test@example.com"),
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
            .flat_map(
                lambda user: cast("FlextResult[object]", FlextResult.ok(user.email))
            )
            .flat_map(
                lambda email: cast(
                    "FlextResult[object]",
                    SendEmailService(to=cast("str", email), subject="Test").execute(),
                ),
            )
        )

        assert pipeline.is_success

    def test_monadic_filter(self) -> None:
        """Monadic: filter validates predicate."""
        result = (
            ValidationService(value=50)
            .execute()
            .filter(lambda data: data["value"] < 100)
        )

        assert result.is_success

    def test_monadic_complex_pipeline(self) -> None:
        """Monadic: Complex pipeline with multiple operations."""
        pipeline = (
            GetUserService(user_id="123")
            .execute()
            .map(lambda user: user.email)
            .filter(lambda email: "@" in cast("str", email))
            .flat_map(
                lambda email: cast(
                    "FlextResult[object]",
                    SendEmailService(to=cast("str", email), subject="Test").execute(),
                ),
            )
            .map(lambda response: cast("EmailResponse", response).status)
        )

        assert pipeline.is_success
        assert pipeline.unwrap() == "sent"


# ============================================================================
# Pattern 8: Error Handling Pythonic (try/except)
# ============================================================================


class TestPattern8ErrorHandling:
    """Test Pattern 8: Error Handling Pythonic."""

    def test_error_handling_try_except_v2_property(self) -> None:
        """Error Handling: try/except with V2 Property."""
        try:
            user = GetUserService(user_id="123").result
            assert user.user_id == "123"  # type: ignore[attr-defined]
        except FlextExceptions.BaseError:
            pytest.fail("Should not raise")

    def test_error_handling_try_except_v2_property_failure(self) -> None:
        """Error Handling: try/except catches failure."""
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            GetUserService(user_id="invalid").result
        assert "not found" in str(exc_info.value).lower()

    def test_error_handling_try_except_v2_auto(self) -> None:
        """Error Handling: try/except with V2 Auto."""
        try:
            user = cast("User", AutoGetUserService(user_id="789"))
            assert user.user_id == "789"  # type: ignore[attr-defined]
        except FlextExceptions.BaseError:
            pytest.fail("Should not raise")

    def test_error_handling_graceful_degradation(self) -> None:
        """Error Handling: Graceful degradation pattern."""
        try:
            user = GetUserService(user_id="123").result
            email = user.email  # type: ignore[attr-defined]
        except FlextExceptions.BaseError:
            email = "fallback@example.com"

        assert email == "user123@example.com"  # type: ignore[attr-defined]


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

    @pytest.mark.parametrize(
        ("operation", "value", "expected"), TestFactories.multi_operation_cases()
    )
    def test_multiple_operations(
        self, operation: str, value: int, expected: dict[str, Any]
    ) -> None:
        """Multiple Operations: Various operations with different inputs."""
        result = cast(
            "dict[str, Any]",
            MultiOperationService(operation=operation, value=value).result,
        )

        assert result["operation"] == expected["operation"]
        assert result["result"] == expected["result"]

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
                lambda result: cast(
                    "FlextResult[object]",
                    MultiOperationService(
                        operation="square",
                        value=cast("int", result),
                    ).execute(),
                ),
            )
            .map(lambda data: cast("dict[str, Any]", data)["result"])
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
        assert v2_user.user_id == "456"  # type: ignore[attr-defined]

        # V2 Auto: Zero ceremony
        auto_user = cast("User", AutoGetUserService(user_id="789"))
        assert auto_user.user_id == "789"  # type: ignore[attr-defined]

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

            def execute(self, **_kwargs: object) -> FlextResult[User]:
                return FlextResult.ok(
                    User(unique_id=self.user_id, name="Test", email="test@example.com"),
                )

        v2_auto_pipeline = ManualService(user_id="789").execute().map(lambda u: u.email)
        assert v2_auto_pipeline.is_success

    def test_complete_real_world_scenario(self) -> None:
        """Complete scenario using multiple patterns."""
        # Step 1: Get user (V2 Property)
        user = GetUserService(user_id="123").result

        # Step 2: Validate and send email (Railway V1)
        email_result = (
            SendEmailService(to=user.email, subject="Welcome")  # type: ignore[attr-defined]
            .execute()
            .filter(lambda r: r.status == "sent")
            .map(lambda r: r.message_id)
        )

        assert email_result.is_success
        message_id = cast("str", email_result.unwrap())
        assert message_id.startswith("msg-")

        # Step 3: Multiple operations (V2 Property)
        calc_result = cast(
            "dict[str, Any]", MultiOperationService(operation="double", value=10).result
        )
        assert calc_result["result"] == 20
