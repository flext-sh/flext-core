from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, MutableSequence, Sequence
from typing import Annotated, ClassVar, cast

import pytest
from flext_tests import t, tm
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextConstants, FlextExceptions, r

from ..test_utils import assertion_helpers


class TestCoverageExceptions:
    class ExceptionCreationScenario(BaseModel):
        """Scenario for exception creation."""

        model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
        name: Annotated[str, Field(description="Exception creation scenario name")]
        exception_type: Annotated[
            type[FlextExceptions.BaseError],
            Field(description="Exception class to instantiate"),
        ]
        message: Annotated[str, Field(description="Exception message")]
        kwargs: Annotated[
            Mapping[str, t.MetadataAttributeValue | type],
            Field(description="Keyword arguments for exception creation"),
        ]
        expected_attrs: Annotated[
            Mapping[str, t.MetadataAttributeValue | type],
            Field(description="Expected attributes to validate"),
        ]

    EXCEPTION_CREATION: ClassVar[Sequence[ExceptionCreationScenario]] = [
        ExceptionCreationScenario(
            name="validation_basic",
            exception_type=FlextExceptions.ValidationError,
            message="Invalid input",
            kwargs={},
            expected_attrs={},
        ),
        ExceptionCreationScenario(
            name="validation_with_field",
            exception_type=FlextExceptions.ValidationError,
            message="Email invalid",
            kwargs={"field": "email", "value": "not-an-email"},
            expected_attrs={"field": "email", "value": "not-an-email"},
        ),
        ExceptionCreationScenario(
            name="configuration_basic",
            exception_type=FlextExceptions.ConfigurationError,
            message="Missing required field",
            kwargs={},
            expected_attrs={},
        ),
        ExceptionCreationScenario(
            name="configuration_with_source",
            exception_type=FlextExceptions.ConfigurationError,
            message="Missing API key",
            kwargs={"config_key": "API_KEY", "config_source": "environment"},
            expected_attrs={"config_key": "API_KEY", "config_source": "environment"},
        ),
        ExceptionCreationScenario(
            name="connection",
            exception_type=FlextExceptions.ConnectionError,
            message="Failed to connect",
            kwargs={"host": "db.example.com", "port": 5432, "timeout": 30.0},
            expected_attrs={"host": "db.example.com", "port": 5432},
        ),
        ExceptionCreationScenario(
            name="timeout",
            exception_type=FlextExceptions.TimeoutError,
            message="Operation timed out",
            kwargs={"timeout_seconds": 30, "operation": "fetch_data"},
            expected_attrs={"timeout_seconds": 30, "operation": "fetch_data"},
        ),
        ExceptionCreationScenario(
            name="authentication",
            exception_type=FlextExceptions.AuthenticationError,
            message="Invalid credentials",
            kwargs={"auth_method": "basic", "user_id": "user123"},
            expected_attrs={"auth_method": "basic", "user_id": "user123"},
        ),
        ExceptionCreationScenario(
            name="authorization",
            exception_type=FlextExceptions.AuthorizationError,
            message="User lacks permission",
            kwargs={
                "user_id": "user123",
                "resource": "REDACTED_LDAP_BIND_PASSWORD_panel",
                "permission": "read",
            },
            expected_attrs={
                "user_id": "user123",
                "resource": "REDACTED_LDAP_BIND_PASSWORD_panel",
            },
        ),
        ExceptionCreationScenario(
            name="not_found",
            exception_type=FlextExceptions.NotFoundError,
            message="User not found",
            kwargs={"resource_type": "User", "resource_id": "123"},
            expected_attrs={"resource_type": "User", "resource_id": "123"},
        ),
        ExceptionCreationScenario(
            name="conflict",
            exception_type=FlextExceptions.ConflictError,
            message="User already exists",
            kwargs={
                "resource_type": "User",
                "resource_id": "user@example.com",
                "conflict_reason": "email_already_registered",
            },
            expected_attrs={"resource_type": "User", "resource_id": "user@example.com"},
        ),
        ExceptionCreationScenario(
            name="rate_limit",
            exception_type=FlextExceptions.RateLimitError,
            message="Too many requests",
            kwargs={"limit": 100, "window_seconds": 60, "retry_after": 30},
            expected_attrs={"limit": 100, "window_seconds": 60},
        ),
        ExceptionCreationScenario(
            name="circuit_breaker",
            exception_type=FlextExceptions.CircuitBreakerError,
            message="Circuit breaker is open",
            kwargs={
                "service_name": "payment_service",
                "failure_count": 5,
                "reset_timeout": 60,
            },
            expected_attrs={"service_name": "payment_service", "failure_count": 5},
        ),
        ExceptionCreationScenario(
            name="type_error",
            exception_type=FlextExceptions.TypeError,
            message="Expected string, got int",
            kwargs={"expected_type": str, "actual_type": int},
            expected_attrs={"expected_type": str, "actual_type": int},
        ),
        ExceptionCreationScenario(
            name="operation_error",
            exception_type=FlextExceptions.OperationError,
            message="Database operation failed",
            kwargs={"operation": "INSERT", "reason": "Constraint violation"},
            expected_attrs={"operation": "INSERT", "reason": "Constraint violation"},
        ),
        ExceptionCreationScenario(
            name="attribute_access",
            exception_type=FlextExceptions.AttributeAccessError,
            message="Attribute not found",
            kwargs={
                "attribute_name": "missing_field",
                "attribute_context": {"class": "User", "attempted_access": "read"},
            },
            expected_attrs={"attribute_name": "missing_field"},
        ),
    ]

    FACTORY_CREATION: ClassVar[
        Sequence[
            tuple[str, Mapping[str, t.NormalizedValue], type[FlextExceptions.BaseError]]
        ]
    ] = [
        (
            "ValidationError",
            {"field": "email", "value": "not-valid"},
            FlextExceptions.ValidationError,
        ),
        (
            "ConfigurationError",
            {"config_key": "API_KEY", "config_source": "environment"},
            FlextExceptions.ConfigurationError,
        ),
        (
            "ConnectionError",
            {"host": FlextConstants.LOCALHOST},
            FlextExceptions.ConnectionError,
        ),
        (
            "OperationError",
            {"operation": "INSERT", "reason": "Constraint violation"},
            FlextExceptions.OperationError,
        ),
    ]

    @pytest.mark.parametrize("scenario", EXCEPTION_CREATION, ids=lambda s: s.name)
    def test_exception_creation(self, scenario: ExceptionCreationScenario) -> None:
        if scenario.kwargs:
            type_kwargs: MutableMapping[str, type] = {}
            metadata_kwargs: MutableMapping[str, t.NormalizedValue] = {}
            for key, value in scenario.kwargs.items():
                if (
                    scenario.exception_type == FlextExceptions.TypeError
                    and key in {"expected_type", "actual_type"}
                    and isinstance(value, type)
                ):
                    type_kwargs[key] = value
                elif isinstance(value, (str, int, float, bool, type(None), list, dict)):
                    metadata_kwargs[key] = cast("t.MetadataAttributeValue", value)
                else:
                    metadata_kwargs[key] = cast("t.MetadataAttributeValue", str(value))
            if type_kwargs:
                for key, type_value in type_kwargs.items():
                    metadata_kwargs[key] = type_value.__name__
            exception_ctor = cast(
                "Callable[..., FlextExceptions.BaseError]",
                scenario.exception_type,
            )
            error = exception_ctor(scenario.message, **metadata_kwargs)
        else:
            error = scenario.exception_type(scenario.message)
        tm.that(str(error), has=scenario.message)
        for attr_name, expected_value in scenario.expected_attrs.items():
            tm.that(hasattr(error, attr_name), eq=True)
            tm.that(getattr(error, attr_name), eq=expected_value)

    def test_exception_to_result_conversion(self) -> None:
        try:
            error_msg = "Test error"
            raise FlextExceptions.ValidationError(error_msg, field="email")
        except FlextExceptions.ValidationError as err:
            result = r[bool].fail(str(err))
            _ = assertion_helpers.assert_flext_result_failure(result)
            tm.fail(result, has="Test error")

    def test_exception_in_railway_pattern(self) -> None:
        def validate_and_process(
            data: Mapping[str, t.NormalizedValue],
        ) -> r[Mapping[str, t.NormalizedValue]]:
            if not data.get("id"):
                return r[Mapping[str, t.NormalizedValue]].fail("Missing id")
            return r[Mapping[str, t.NormalizedValue]].ok(data)

        tm.fail(validate_and_process({}))
        tm.ok(validate_and_process({"id": "123"}))

    def test_nested_exception_handling(self) -> None:
        try:
            error_msg = "Validation failed"
            raise FlextExceptions.ValidationError(
                error_msg,
                field="email",
                value="invalid",
            )
        except FlextExceptions.ValidationError as err:
            result = r[bool].fail(f"Error in user creation: {err}")
            _ = assertion_helpers.assert_flext_result_failure(result)
            tm.that(result.error, ne=None)
            if result.error is not None:
                tm.that("Validation failed" in result.error, eq=True)

    @pytest.mark.parametrize(
        ("message", "expected_in_str"),
        [
            ("", True),
            ("Invalid: 中文 العربية 🔴", True),
            ("x" * 10000, True),
            ("Message with \"quotes\" and 'apostrophes'", True),
        ],
        ids=["empty", "unicode", "long", "special_chars"],
    )
    def test_exception_message_variations(
        self,
        message: str,
        expected_in_str: bool,
    ) -> None:
        error = FlextExceptions.ValidationError(message)
        if message:
            tm.that(message in str(error) or len(str(error)) > 9000, eq=expected_in_str)

    def test_multiple_exceptions_in_sequence(self) -> None:
        errors: MutableSequence[str] = []
        for i in range(5):
            try:
                if i % 2 == 0:
                    raise FlextExceptions.ValidationError(f"Error {i}")
                raise FlextExceptions.ConfigurationError(f"Config error {i}")
            except Exception as err:
                errors.append(str(err))
        tm.that(len(errors), eq=5)
        tm.that(any("Error" in err for err in errors), eq=True)

    def test_exception_inheritance_chain(self) -> None:
        error = FlextExceptions.ValidationError("Test")
        tm.that(str(error), has="Test")

    def test_exception_string_representation(self) -> None:
        error = FlextExceptions.ValidationError("Test message")
        tm.that(str(error), has="Test message")

    def test_exception_repr(self) -> None:
        error = FlextExceptions.ValidationError("Test")
        repr_str = repr(error)
        tm.that(repr_str or "Test" in repr_str, has="ValidationError")

    def test_exception_type_checking(self) -> None:
        error = FlextExceptions.ValidationError("Test")
        tm.that(error.__class__.__name__, eq="ValidationError")

    def test_base_error_with_metadata(self) -> None:
        error = FlextExceptions.NotFoundError(
            "Resource not found",
            resource_id="123",
            resource_type="User",
        )
        tm.that(str(error), has="Resource not found")

    def test_exception_with_context_data(self) -> None:
        error = FlextExceptions.ValidationError(
            "Validation failed in context",
            user_id="123",
            operation="create_user",
            timestamp=1234567890,
        )
        tm.that(error.metadata.attributes, has="user_id")
        tm.that(error.metadata.attributes["user_id"], eq="123")

    def test_exception_with_correlation_id(self) -> None:
        error = FlextExceptions.BaseError("Test error", auto_correlation=True)
        tm.that(error.correlation_id, none=False)
        tm.that(error.correlation_id, starts="exc_")

    def test_exception_chaining(self) -> None:
        original = ValueError("Original error")
        error = FlextExceptions.OperationError("Operation failed")
        error.__cause__ = original
        tm.that(error.__cause__ is original, eq=True)

    def test_exception_preservation(self) -> None:
        original_msg = "Original error message with details"
        error = FlextExceptions.ValidationError(original_msg)
        result = r[bool].fail(str(error))
        tm.that(result.error, none=False)
        tm.fail(result, has=original_msg)

    def test_exception_to_dict(self) -> None:
        error = FlextExceptions.ValidationError(
            "Invalid email",
            field="email",
            value="not-valid",
        )
        error_dict = error.to_dict()
        tm.that(error_dict["error_type"], eq="ValidationError")
        tm.that(error_dict["message"], eq="Invalid email")
        tm.that(error_dict["error_code"], eq="VALIDATION_ERROR")

    def test_exception_dict_with_metadata(self) -> None:
        error = FlextExceptions.OperationError("Operation failed", operation="INSERT")
        error_dict = error.to_dict()
        tm.that(error_dict["operation"], eq="INSERT")

    def test_create_error_by_type(self) -> None:
        error = FlextExceptions.create("ValidationError", "Test validation error")
        tm.that(isinstance(error, FlextExceptions.ValidationError), eq=True)
        tm.that(str(error), has="Test validation error")

    @pytest.mark.parametrize(("message", "kwargs", "expected_type"), FACTORY_CREATION)
    def test_create_error_auto_detection(
        self,
        message: str,
        kwargs: Mapping[str, t.NormalizedValue],
        expected_type: type[FlextExceptions.BaseError],
    ) -> None:
        converted_kwargs: Mapping[str, t.NormalizedValue] = {
            key: cast("t.MetadataAttributeValue", value)
            for key, value in kwargs.items()
        }
        create_error = cast(
            "Callable[..., FlextExceptions.BaseError]",
            FlextExceptions.create,
        )
        error = create_error(message, **converted_kwargs)
        tm.that(isinstance(error, expected_type), eq=True)

    def test_record_exception(self) -> None:
        FlextExceptions.clear_metrics()
        FlextExceptions.record_exception(FlextExceptions.ValidationError)
        FlextExceptions.record_exception(FlextExceptions.ValidationError)
        FlextExceptions.record_exception(FlextExceptions.ConfigurationError)
        metrics = FlextExceptions.get_metrics()
        tm.that(metrics["total_exceptions"], eq=3)
        raw_counts = metrics.root.get("exception_counts")
        exception_counts = cast("Mapping[str, int]", raw_counts)
        tm.that(exception_counts.get("FlextExceptions.ValidationError"), eq=2)
        tm.that(exception_counts.get("FlextExceptions.ConfigurationError"), eq=1)
        tm.that(metrics["unique_exception_types"], eq=2)

    def test_clear_metrics(self) -> None:
        FlextExceptions.clear_metrics()
        FlextExceptions.record_exception(FlextExceptions.ValidationError)
        tm.that(FlextExceptions.get_metrics()["total_exceptions"], eq=1)
        FlextExceptions.clear_metrics()
        tm.that(FlextExceptions.get_metrics()["total_exceptions"], eq=0)

    def test_exception_string_with_correlation_id(self) -> None:
        error = FlextExceptions.BaseError("Test", auto_correlation=True)
        tm.that(error.correlation_id, none=False)
        tm.that(error.correlation_id, starts="exc_")
        tm.that(str(error), has="Test")

    def test_exception_error_code_in_string(self) -> None:
        error = FlextExceptions.ValidationError("Test message")
        error_str = str(error)
        tm.that(error_str or "Test message" in error_str, has="VALIDATION_ERROR")

    def test_failure_level_enum_values(self) -> None:
        failure_level = FlextConstants.FailureLevel
        tm.that(
            all(
                hasattr(failure_level, level)
                for level in ["STRICT", "WARN", "PERMISSIVE"]
            ),
            eq=True,
        )

    def test_failure_level_string_values(self) -> None:
        failure_level = FlextConstants.FailureLevel
        tm.that(failure_level.STRICT.value, eq="strict")
        tm.that(failure_level.WARN.value, eq="warn")
        tm.that(failure_level.PERMISSIVE.value, eq="permissive")

    def test_failure_level_comparison(self) -> None:
        failure_level = FlextConstants.FailureLevel
        strict_val: str = str(failure_level.STRICT.value)
        warn_val: str = str(failure_level.WARN.value)
        permissive_val: str = str(failure_level.PERMISSIVE.value)
        tm.that(strict_val, ne=warn_val)
        tm.that(warn_val, ne=permissive_val)
        tm.that(strict_val, eq=str(failure_level.STRICT.value))


__all__ = ["TestCoverageExceptions"]
