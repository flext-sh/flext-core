"""Test all patterns documented in FLEXT_SERVICE_ARCHITECTURE.md.

This module validates ALL patterns documented in the architecture guide using advanced Python 3.13 patterns,
factories, and helpers to reduce code size while maintaining and expanding functionality. Tests all edge cases
with minimal code duplication through unified class architecture and reusable test factories.

Patterns tested:
- Pattern 1: V1 Explícito (.execute().value)
- Pattern 2: V2 Property (.result)
- Pattern 3: Railway Pattern em V1
- Pattern 4: Railway Pattern em V2 Property
- Pattern 5: Composição Monadic (map, and_then, filter, tap)
- Pattern 6: Error Handling Pythonic (try/except)
- Pattern 7: Infraestrutura Automática (config, logger, container)
- Pattern 8: Múltiplas Operações (operation field)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import operator
from typing import override

import pytest
from pydantic import BaseModel, ConfigDict, Field

from flext_core import (
    FlextContainer,
    FlextExceptions,
    FlextLogger,
    FlextService,
    FlextSettings,
    m,
    r,
)

from ._models import EmailResponse
from .test_utils import assertion_helpers


class User(BaseModel):
    """User domain model."""

    unique_id: str = Field(description="Unique user identifier")
    name: str = Field(description="User display name")
    email: str = Field(description="User email address")
    active: bool = Field(default=True, description="Whether user is active")


class ServiceTestCase(BaseModel):
    """Factory for service test cases to reduce duplication."""

    model_config = ConfigDict(frozen=True)

    user_id: str = Field(description="User identifier for test case")
    expected_success: bool = Field(
        default=True, description="Whether service call is expected to succeed"
    )
    expected_error: str | None = Field(
        default=None, description="Expected error substring for failure cases"
    )
    description: str = Field(
        default="", description="Human-readable test case description"
    )

    def create_user_service(self) -> GetUserService:
        """Create GetUserService instance for this test case."""
        return _make(GetUserService, user_id=self.user_id)


class RailwayTestCase(BaseModel):
    """Factory for railway pattern test cases."""

    model_config = ConfigDict(frozen=True)

    user_ids: list[str] = Field(description="User identifiers used in pipeline")
    operations: list[str] = Field(
        default_factory=list, description="Pipeline operations to execute"
    )
    expected_pipeline_length: int = Field(
        default=1, description="Expected number of pipeline stages"
    )
    should_fail_at: int | None = Field(
        default=None, description="Optional pipeline step expected to fail"
    )
    description: str = Field(
        default="", description="Human-readable railway test case description"
    )

    def execute_v1_pipeline(self) -> r[str | User | EmailResponse]:
        """Execute V1 railway pipeline for this test case."""
        if not self.user_ids:
            return r[str | User | EmailResponse].fail("No user IDs provided")
        user_result: r[User] = _make(
            GetUserService,
            user_id=self.user_ids[0],
        ).execute()
        result: r[User | str | EmailResponse] = user_result.map(
            lambda user: user,
        )
        for op in self.operations:
            if op == "get_email":
                result = result.map(
                    lambda user: user.email if isinstance(user, User) else str(user),
                )
            elif op == "send_email":
                email_result: r[EmailResponse] = result.flat_map(
                    lambda email: _make(
                        SendEmailService,
                        to=str(email),
                        subject="Test",
                    ).execute(),
                )
                result = email_result.map(lambda response: response)
            elif op == "get_status":
                result = result.map(
                    lambda response: (
                        response.status
                        if isinstance(response, EmailResponse)
                        else str(response)
                    ),
                )
        return result

    def execute_v2_pipeline(self) -> User | str:
        """Execute V2 property pipeline for this test case."""
        if not self.user_ids:
            msg = "No user IDs provided"
            raise FlextExceptions.BaseError(msg)
        user_result = _make(GetUserService, user_id=self.user_ids[0]).result
        user: User | str = user_result
        for op in self.operations:
            if op == "get_email":
                user = user.email if isinstance(user, User) else str(user)
            elif op == "send_email":
                email_to = str(user) if not isinstance(user, str) else user
                response_obj: EmailResponse = _make(
                    SendEmailService, to=email_to, subject="Test"
                ).result
                user = response_obj.status
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
    def multi_operation_cases() -> list[tuple[str, int, m.ConfigMap]]:
        """Generate multiple operation test cases."""
        return [
            ("double", 5, m.ConfigMap(root={"operation": "double", "result": 10})),
            ("square", 4, m.ConfigMap(root={"operation": "square", "result": 16})),
            ("negate", 7, m.ConfigMap(root={"operation": "negate", "result": -7})),
            ("double", 0, m.ConfigMap(root={"operation": "double", "result": 0})),
            ("square", 1, m.ConfigMap(root={"operation": "square", "result": 1})),
        ]


class GetUserService(FlextService[User]):
    """Service to get user - V1/V2 compatible."""

    user_id: str = ""

    @override
    def execute(self) -> r[User]:
        """Get user by ID."""
        if self.user_id in {"invalid", ""}:
            return r[User].fail("User not found")
        return r[User].ok(
            User(
                unique_id=self.user_id,
                name=f"User {self.user_id}",
                email=f"user{self.user_id}@example.com",
            ),
        )


class SendEmailService(FlextService[EmailResponse]):
    """Service to send email."""

    to: str = ""
    subject: str = ""

    @override
    def execute(self) -> r[EmailResponse]:
        """Send email."""
        if "@" not in self.to:
            return r[EmailResponse].fail("Invalid email address")
        return r[EmailResponse].ok(
            EmailResponse(status="sent", message_id=f"msg-{self.to}")
        )


class ValidationService(FlextService[m.ConfigMap]):
    """Service that validates input."""

    value: int = 0

    @override
    def execute(self) -> r[m.ConfigMap]:
        """Validate value."""
        if self.value < 0:
            return r[m.ConfigMap].fail("Value must be positive")
        if self.value > 100:
            return r[m.ConfigMap].fail("Value must be <= 100")
        return r[m.ConfigMap].ok(m.ConfigMap(root={"valid": True, "value": self.value}))


class MultiOperationService(FlextService[m.ConfigMap]):
    """Service with multiple operations."""

    operation: str = ""
    value: int = 0

    @override
    def execute(self) -> r[m.ConfigMap]:
        """Execute based on operation."""
        match self.operation:
            case "double":
                return r[m.ConfigMap].ok(
                    m.ConfigMap(root={"operation": "double", "result": self.value * 2}),
                )
            case "square":
                return r[m.ConfigMap].ok(
                    m.ConfigMap(root={"operation": "square", "result": self.value**2}),
                )
            case "negate":
                return r[m.ConfigMap].ok(
                    m.ConfigMap(root={"operation": "negate", "result": -self.value}),
                )
            case _:
                return r[m.ConfigMap].fail(f"Unknown operation: {self.operation}")


def _value_lt_100(data: m.ConfigMap) -> bool:
    value = data.get("value")
    return isinstance(value, int) and value < 100


def _make[T](cls: type[T], **kwargs: object) -> T:
    """Create a FlextService subclass — all fields must have defaults."""
    instance = cls()
    for key, value in kwargs.items():
        object.__setattr__(instance, key, value)
    return instance


class TestPattern1V1Explicit:
    """Test Pattern 1: V1 Explícito (.execute().value)."""

    @pytest.mark.parametrize("case", TestFactories.success_cases())
    def test_v1_explicit_success(self, case: ServiceTestCase) -> None:
        """V1: Execute and unwrap on success for various cases."""
        service = case.create_user_service()
        result = service.execute()
        _ = assertion_helpers.assert_flext_result_success(result)
        user = result.value
        assert isinstance(user, User)
        assert user.unique_id == case.user_id
        assert user.name == f"User {case.user_id}"

    @pytest.mark.parametrize("case", TestFactories.failure_cases())
    def test_v1_explicit_failure(self, case: ServiceTestCase) -> None:
        """V1: Execute and check failure for various invalid cases."""
        service = case.create_user_service()
        result = service.execute()
        _ = assertion_helpers.assert_flext_result_failure(result)
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
            user = result.value
            assert isinstance(user, User)
            assert user.unique_id == case.user_id
        else:
            pytest.fail("Should succeed")


class TestPattern2V2Property:
    """Test Pattern 2: V2 Property (.result)."""

    @pytest.mark.parametrize("case", TestFactories.success_cases())
    def test_v2_property_success(self, case: ServiceTestCase) -> None:
        """V2 Property: Use .result for happy path."""
        user = case.create_user_service().result
        assert isinstance(user, User)
        assert user.unique_id == case.user_id
        assert user.name == f"User {case.user_id}"

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
        _ = assertion_helpers.assert_flext_result_success(result)
        user = result.value
        assert isinstance(user, User)
        assert user.unique_id == case.user_id


class TestPattern3RailwayV1:
    """Test Pattern 3: Railway Pattern em V1."""

    @pytest.mark.parametrize("case", TestFactories.railway_success_cases())
    def test_v1_railway_complex_pipeline(self, case: RailwayTestCase) -> None:
        """V1 Railway: Full composition pipeline with various operations."""
        result = case.execute_v1_pipeline()
        _ = assertion_helpers.assert_flext_result_success(result)
        if "get_status" in case.operations:
            assert result.value == "sent"
        elif "get_email" in case.operations:
            unwrapped = result.value
            email: str = str(unwrapped) if not isinstance(unwrapped, str) else unwrapped
            assert isinstance(email, str)
            assert "@" in email
        else:
            assert isinstance(result.value, User)


class TestPattern4RailwayV2Property:
    """Test Pattern 4: Railway Pattern em V2 Property."""

    @pytest.mark.parametrize("case", TestFactories.railway_success_cases())
    def test_v2_property_can_use_execute_for_railway(
        self,
        case: RailwayTestCase,
    ) -> None:
        """V2 Property: .execute() available for railway pattern."""
        user_result = _make(GetUserService, user_id="123").result
        assert isinstance(user_result, User)
        assert user_result.unique_id == "123"
        result = _make(GetUserService, user_id="123").execute().map(lambda u: u.email)
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value == "user123@example.com"

    @pytest.mark.parametrize("case", TestFactories.railway_success_cases())
    def test_v2_property_railway_chaining(self, case: RailwayTestCase) -> None:
        """V2 Property: Full railway chaining."""
        pipeline = (
            _make(GetUserService, user_id="456")
            .execute()
            .flat_map(
                lambda user: _make(
                    SendEmailService,
                    to=user.email,
                    subject="Hello",
                ).execute(),
            )
            .map(lambda response: response.message_id)
        )
        assert pipeline.is_success
        message_id: str = pipeline.value
        assert message_id.startswith("msg-")


class TestPattern5MonadicComposition:
    """Test Pattern 5: Composição Monadic."""

    def test_monadic_map(self) -> None:
        """Monadic: map transforms value."""
        result = (
            _make(GetUserService, user_id="123")
            .execute()
            .map(lambda user: user.name.upper())
        )
        assert result.value == "USER 123"

    def test_monadic_flat_map(self) -> None:
        """Monadic: flat_map chains operations (also known as and_then)."""
        pipeline = (
            _make(GetUserService, user_id="123")
            .execute()
            .flat_map(lambda user: r[str].ok(user.email))
            .flat_map(
                lambda email: _make(
                    SendEmailService,
                    to=email,
                    subject="Test",
                ).execute(),
            )
        )
        assert pipeline.is_success

    def test_monadic_filter(self) -> None:
        """Monadic: filter validates predicate."""
        result = _make(ValidationService, value=50).execute().filter(_value_lt_100)
        _ = assertion_helpers.assert_flext_result_success(result)

    def test_monadic_complex_pipeline(self) -> None:
        """Monadic: Complex pipeline with multiple operations."""
        pipeline = (
            _make(GetUserService, user_id="123")
            .execute()
            .map(lambda user: user.email)
            .filter(lambda email: "@" in email)
            .flat_map(
                lambda email: _make(
                    SendEmailService,
                    to=email,
                    subject="Test",
                ).execute(),
            )
            .map(lambda response: response.status)
        )
        assert pipeline.is_success
        assert pipeline.value == "sent"


class TestPattern6ErrorHandling:
    """Test Pattern 6: Error Handling Pythonic."""

    def test_error_handling_try_except_v2_property(self) -> None:
        """Error Handling: try/except with V2 Property."""
        try:
            user_result = _make(GetUserService, user_id="123").result
            assert isinstance(user_result, User)
            assert user_result.unique_id == "123"
        except FlextExceptions.BaseError:
            pytest.fail("Should not raise")

    def test_error_handling_try_except_v2_property_failure(self) -> None:
        """Error Handling: try/except catches failure."""
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            _make(GetUserService, user_id="invalid").result
        assert "not found" in str(exc_info.value).lower()

    def test_error_handling_graceful_degradation(self) -> None:
        """Error Handling: Graceful degradation pattern."""
        try:
            user_result = _make(GetUserService, user_id="123").result
            assert isinstance(user_result, User)
            email = user_result.email
        except FlextExceptions.BaseError:
            email = "fallback@example.com"
        assert email == "user123@example.com"


class TestPattern7AutomaticInfrastructure:
    """Test Pattern 7: Infraestrutura Automática."""

    def test_infrastructure_config_automatic(self) -> None:
        """Infrastructure: Config available automatically."""
        service = _make(GetUserService, user_id="123")
        assert service.config is not None
        assert isinstance(service.config, FlextSettings)

    def test_infrastructure_logger_automatic(self) -> None:
        """Infrastructure: Logger available automatically."""
        service = _make(GetUserService, user_id="123")
        assert service.logger is not None
        assert isinstance(service.logger, FlextLogger)

    def test_infrastructure_container_automatic(self) -> None:
        """Infrastructure: Container available automatically."""
        service = _make(GetUserService, user_id="123")
        assert service.container is not None
        assert isinstance(service.container, FlextContainer)

    def test_infrastructure_lazy_initialization(self) -> None:
        """Infrastructure: Properties are lazy."""
        service = _make(GetUserService, user_id="123")
        config1 = service.config
        config2 = service.config
        assert config1 is config2


class TestPattern8MultipleOperations:
    """Test Pattern 8: Múltiplas Operações."""

    @pytest.mark.parametrize(
        ("operation", "value", "expected"),
        TestFactories.multi_operation_cases(),
    )
    def test_multiple_operations(
        self,
        operation: str,
        value: int,
        expected: m.ConfigMap,
    ) -> None:
        """Multiple Operations: Various operations with different inputs."""
        result: m.ConfigMap = _make(
            MultiOperationService, operation=operation, value=value
        ).result
        assert result["operation"] == expected["operation"]
        assert result["result"] == expected["result"]

    def test_multiple_operations_invalid(self) -> None:
        """Multiple Operations: Invalid operation fails."""
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            _make(MultiOperationService, operation="invalid", value=5).result
        assert "Unknown operation" in str(exc_info.value)

    def test_multiple_operations_with_railway(self) -> None:
        """Multiple Operations: Railway pattern with operations."""
        pipeline = (
            _make(MultiOperationService, operation="double", value=5)
            .execute()
            .map(operator.itemgetter("result"))
            .flat_map(
                lambda result: _make(
                    MultiOperationService,
                    operation="square",
                    value=result,
                ).execute(),
            )
            .map(operator.itemgetter("result"))
        )
        assert pipeline.is_success
        assert pipeline.value == 100


class TestAllPatternsIntegration:
    """Integration tests combining multiple patterns."""

    def test_v1_v2_property_interoperability(self) -> None:
        """All patterns work together seamlessly."""
        v1_result = _make(GetUserService, user_id="123").execute()
        assert v1_result.is_success
        v2_user_result = _make(GetUserService, user_id="456").result
        assert isinstance(v2_user_result, User)
        assert v2_user_result.unique_id == "456"
        assert isinstance(v1_result.value, User)
        assert isinstance(v2_user_result, User)

    def test_railway_pattern_works_in_all_versions(self) -> None:
        """Railway pattern works in V1 and V2 Property."""
        v1_pipeline = (
            _make(GetUserService, user_id="123").execute().map(lambda u: u.email)
        )
        assert v1_pipeline.is_success
        v2_pipeline = (
            _make(GetUserService, user_id="456").execute().map(lambda u: u.email)
        )
        assert v2_pipeline.is_success

        class CustomService(FlextService[User]):
            user_id: str = ""

            @override
            def execute(self) -> r[User]:
                return r[User].ok(
                    User(unique_id=self.user_id, name="Test", email="test@example.com"),
                )

        custom_pipeline = (
            _make(CustomService, user_id="789").execute().map(lambda u: u.email)
        )
        assert custom_pipeline.is_success

    def test_complete_real_world_scenario(self) -> None:
        """Complete scenario using multiple patterns."""
        user = _make(GetUserService, user_id="123").result
        email_result = (
            _make(SendEmailService, to=user.email, subject="Welcome")
            .execute()
            .filter(lambda r: r.status == "sent")
            .map(lambda r: r.message_id)
        )
        assert email_result.is_success
        message_id: str = email_result.value
        assert message_id.startswith("msg-")
        calc_result: m.ConfigMap = _make(
            MultiOperationService, operation="double", value=10
        ).result
        assert calc_result["result"] == 20
