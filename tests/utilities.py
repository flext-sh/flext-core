"""Utilities for flext-core tests."""

from __future__ import annotations

from collections.abc import MutableSequence, Sequence
from itertools import count
from typing import ClassVar, override

from flext_tests import u
from tests import c, e, m, p, r, s, t


class TestsFlextCoreUtilities(u):
    """Utilities for flext-core tests."""

    class Core:
        """flext-core-specific test utilities namespace."""

        class Tests(u.Tests):
            """flext-core test utilities namespace."""

            @staticmethod
            def success_cases() -> Sequence[tuple[str, str]]:
                return [
                    ("123", "Valid user ID"),
                    ("456", "Another valid user ID"),
                    ("789", "Third valid user ID"),
                ]

            @staticmethod
            def failure_cases() -> Sequence[tuple[str, str, str]]:
                return [
                    ("invalid", "not found", "Invalid user ID"),
                    ("", "not found", "Empty user ID"),
                ]

            @staticmethod
            def railway_success_cases() -> Sequence[
                tuple[t.StrSequence, t.StrSequence, int, str]
            ]:
                return [
                    (["123"], [], 1, "Simple user retrieval"),
                    (["456"], ["get_email"], 2, "User to email transformation"),
                    (
                        ["789"],
                        ["get_email", "send_email", "get_status"],
                        4,
                        "Full pipeline: user -> email -> send -> status",
                    ),
                ]

            @staticmethod
            def multi_operation_cases() -> Sequence[
                tuple[str, int, t.RecursiveContainerMapping]
            ]:
                return [
                    (
                        "double",
                        5,
                        {"operation": "double", "result": 10},
                    ),
                    (
                        "square",
                        4,
                        {"operation": "square", "result": 16},
                    ),
                    (
                        "negate",
                        7,
                        {"operation": "negate", "result": -7},
                    ),
                    (
                        "double",
                        0,
                        {"operation": "double", "result": 0},
                    ),
                    (
                        "square",
                        1,
                        {"operation": "square", "result": 1},
                    ),
                ]

            class GetUserService(s[m.BaseModel]):
                """Service to get user."""

                user_id: str = ""

                @override
                def execute(self) -> p.Result[m.Core.Tests.User]:
                    if self.user_id in {"invalid", ""}:
                        return r[m.Core.Tests.User].fail(
                            c.Core.Tests.TestErrors.USER_NOT_FOUND
                        )
                    return r[m.Core.Tests.User].ok(
                        m.Core.Tests.User(
                            id=self.user_id,
                            unique_id=self.user_id,
                            name=f"User {self.user_id}",
                            email=f"user{self.user_id}@example.com",
                        ),
                    )

            class SendEmailService(s[m.BaseModel]):
                """Service to send email."""

                to: str = ""
                subject: str = ""

                @override
                def execute(self) -> p.Result[m.Core.Tests.EmailResponse]:
                    if "@" not in self.to:
                        return r[m.Core.Tests.EmailResponse].fail(
                            c.Core.Tests.TestErrors.INVALID_EMAIL
                        )
                    return r[m.Core.Tests.EmailResponse].ok(
                        m.Core.Tests.EmailResponse(
                            status="sent", message_id=f"msg-{self.to}"
                        ),
                    )

            class ValidationService(s[t.RecursiveContainerMapping]):
                """Service to validate values."""

                value: int = 0

                @override
                def execute(self) -> p.Result[t.RecursiveContainerMapping]:
                    if self.value < 0:
                        return r[t.RecursiveContainerMapping].fail(
                            c.Core.Tests.TestErrors.VALUE_TOO_LOW
                        )
                    if self.value > 100:
                        return r[t.RecursiveContainerMapping].fail(
                            c.Core.Tests.TestErrors.VALUE_TOO_HIGH
                        )
                    return r[t.RecursiveContainerMapping].ok(
                        {"valid": True, "value": self.value},
                    )

            class MultiOperationService(s[t.RecursiveContainerMapping]):
                """Service for multiple operations."""

                operation: str = ""
                value: int = 0

                @override
                def execute(self) -> p.Result[t.RecursiveContainerMapping]:
                    match self.operation:
                        case "double":
                            return r[t.RecursiveContainerMapping].ok(
                                {
                                    "operation": "double",
                                    "result": self.value * 2,
                                },
                            )
                        case "square":
                            return r[t.RecursiveContainerMapping].ok(
                                {
                                    "operation": "square",
                                    "result": self.value**2,
                                },
                            )
                        case "negate":
                            return r[t.RecursiveContainerMapping].ok(
                                {"operation": "negate", "result": -self.value},
                            )
                        case _:
                            return r[t.RecursiveContainerMapping].fail(
                                f"Unknown operation: {self.operation}"
                            )

            @staticmethod
            def value_lt_100(data: t.RecursiveContainerMapping) -> bool:
                value = data.get("value")
                return isinstance(value, int) and value < 100

            @staticmethod
            def make[T](service_type: type[T], **kwargs: t.Scalar) -> T:
                instance = service_type()
                for key, value in kwargs.items():
                    object.__setattr__(instance, key, value)
                return instance

            @staticmethod
            def create_user_service(
                case: m.Core.Tests.ServiceTestCase,
            ) -> TestsFlextCoreUtilities.Core.Tests.GetUserService:
                """Create a user service from a documented service case."""
                return TestsFlextCoreUtilities.Core.Tests.make(
                    TestsFlextCoreUtilities.Core.Tests.GetUserService,
                    user_id=case.user_id or case.input_value or "",
                )

            @staticmethod
            def execute_v1_pipeline(
                case: m.Core.Tests.RailwayTestCase,
            ) -> p.Result[str | m.Core.Tests.User | m.Core.Tests.EmailResponse]:
                """Execute the documented V1 railway pipeline."""
                if not case.user_ids:
                    return r[str | m.Core.Tests.User | m.Core.Tests.EmailResponse].fail(
                        c.Core.Tests.TestErrors.NO_USER_IDS_PROVIDED,
                    )
                user_result: p.Result[m.Core.Tests.User] = (
                    TestsFlextCoreUtilities.Core.Tests.make(
                        TestsFlextCoreUtilities.Core.Tests.GetUserService,
                        user_id=case.user_ids[0],
                    ).execute()
                )
                result: p.Result[
                    str | m.Core.Tests.User | m.Core.Tests.EmailResponse
                ] = user_result.map(lambda user: user)
                for operation in case.operations:
                    if operation == "get_email":
                        result = result.map(
                            lambda user: (
                                user.email
                                if isinstance(user, m.Core.Tests.User)
                                else str(user)
                            ),
                        )
                    elif operation == "send_email":
                        email_result: p.Result[m.Core.Tests.EmailResponse] = (
                            result.flat_map(
                                lambda email: TestsFlextCoreUtilities.Core.Tests.make(
                                    TestsFlextCoreUtilities.Core.Tests.SendEmailService,
                                    to=str(email),
                                    subject="Test",
                                ).execute(),
                            )
                        )
                        result = email_result.map(lambda response: response)
                    elif operation == "get_status":
                        result = result.map(
                            lambda response: (
                                response.status
                                if isinstance(response, m.Core.Tests.EmailResponse)
                                else str(response)
                            ),
                        )
                return result

            @staticmethod
            def execute_v2_pipeline(
                case: m.Core.Tests.RailwayTestCase,
            ) -> m.Core.Tests.User | str:
                """Execute the documented V2 railway pipeline."""
                if not case.user_ids:
                    msg = c.Core.Tests.TestErrors.NO_USER_IDS_PROVIDED
                    raise e.BaseError(msg)
                raw_user = TestsFlextCoreUtilities.Core.Tests.make(
                    TestsFlextCoreUtilities.Core.Tests.GetUserService,
                    user_id=case.user_ids[0],
                ).result
                if not isinstance(raw_user, m.Core.Tests.User):
                    msg = c.Core.Tests.TestErrors.USER_NOT_FOUND
                    raise e.BaseError(msg)
                user: m.Core.Tests.User | str = raw_user
                for operation in case.operations:
                    if operation == "get_email":
                        user = (
                            user.email
                            if isinstance(user, m.Core.Tests.User)
                            else str(user)
                        )
                    elif operation == "send_email":
                        email_to = user if isinstance(user, str) else str(user)
                        raw_response = TestsFlextCoreUtilities.Core.Tests.make(
                            TestsFlextCoreUtilities.Core.Tests.SendEmailService,
                            to=email_to,
                            subject="Test",
                        ).result
                        if not isinstance(raw_response, m.Core.Tests.EmailResponse):
                            msg = c.Core.Tests.TestErrors.INVALID_EMAIL
                            raise e.BaseError(msg)
                        response_obj: m.Core.Tests.EmailResponse = raw_response
                        user = response_obj.status
                return user

            class ValidationScenarios:
                """Centralized validation scenarios - single source of truth."""

                URI_SCENARIOS: ClassVar[Sequence[m.Core.Tests.ValidationScenario]] = [
                    m.Core.Tests.ValidationScenario(
                        name="uri_valid_http",
                        validator_type="network",
                        input_value="http://example.com",
                        should_succeed=True,
                        expected_value="http://example.com",
                        description="Valid HTTP URI",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="uri_valid_https",
                        validator_type="network",
                        input_value="https://example.com",
                        should_succeed=True,
                        expected_value="https://example.com",
                        description="Valid HTTPS URI",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="uri_with_port",
                        validator_type="network",
                        input_value="https://example.com:8080",
                        should_succeed=True,
                        expected_value="https://example.com:8080",
                        description="URI with custom port",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="uri_with_path",
                        validator_type="network",
                        input_value="https://example.com/path/to/resource",
                        should_succeed=True,
                        expected_value="https://example.com/path/to/resource",
                        description="URI with path",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="uri_with_query",
                        validator_type="network",
                        input_value="https://example.com/path?key=value",
                        should_succeed=True,
                        expected_value="https://example.com/path?key=value",
                        description="URI with query parameters",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="uri_with_fragment",
                        validator_type="network",
                        input_value="https://example.com/path#section",
                        should_succeed=True,
                        expected_value="https://example.com/path#section",
                        description="URI with fragment",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="uri_none",
                        validator_type="network",
                        input_value=None,
                        should_succeed=False,
                        expected_error_contains="cannot be None",
                        description="None URI rejection",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="uri_empty",
                        validator_type="network",
                        input_value="",
                        should_succeed=False,
                        expected_error_contains="cannot be empty",
                        description="Empty URI rejection",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="uri_invalid_scheme",
                        validator_type="network",
                        input_value="ftp://example.com",
                        should_succeed=False,
                        expected_error_contains="not in allowed",
                        description="Invalid URI scheme",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="uri_malformed",
                        validator_type="network",
                        input_value="not a valid uri",
                        should_succeed=False,
                        expected_error_contains="not a valid",
                        description="Malformed URI",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="uri_whitespace",
                        validator_type="network",
                        input_value="   ",
                        should_succeed=False,
                        expected_error_contains="cannot be empty",
                        description="Whitespace-only URI",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="uri_custom_scheme",
                        validator_type="network",
                        input_value="custom://example.com",
                        input_params={"allowed_schemes": ["http", "https", "custom"]},
                        should_succeed=True,
                        expected_value="custom://example.com",
                        description="Custom scheme with allowlist",
                    ),
                ]
                PORT_SCENARIOS: ClassVar[Sequence[m.Core.Tests.ValidationScenario]] = [
                    m.Core.Tests.ValidationScenario(
                        name="port_valid_80",
                        validator_type="network",
                        input_value=80,
                        should_succeed=True,
                        expected_value=80,
                        description="Valid port 80 (HTTP)",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="port_valid_443",
                        validator_type="network",
                        input_value=443,
                        should_succeed=True,
                        expected_value=443,
                        description="Valid port 443 (HTTPS)",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="port_valid_8080",
                        validator_type="network",
                        input_value=8080,
                        should_succeed=True,
                        expected_value=8080,
                        description="Valid port 8080",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="port_valid_1",
                        validator_type="network",
                        input_value=1,
                        should_succeed=True,
                        expected_value=1,
                        description="Valid port 1 (minimum)",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="port_valid_65535",
                        validator_type="network",
                        input_value=65535,
                        should_succeed=True,
                        expected_value=65535,
                        description="Valid port 65535 (maximum)",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="port_none",
                        validator_type="network",
                        input_value=None,
                        should_succeed=False,
                        expected_error_contains="cannot be None",
                        description="None port rejection",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="port_zero",
                        validator_type="network",
                        input_value=0,
                        should_succeed=False,
                        expected_error_contains="must be between",
                        description="Port zero rejection",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="port_negative",
                        validator_type="network",
                        input_value=-1,
                        should_succeed=False,
                        expected_error_contains="must be between",
                        description="Negative port rejection",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="port_above_max",
                        validator_type="network",
                        input_value=65536,
                        should_succeed=False,
                        expected_error_contains="at most",
                        description="Port above maximum",
                    ),
                ]
                HOSTNAME_SCENARIOS: ClassVar[
                    Sequence[m.Core.Tests.ValidationScenario]
                ] = [
                    m.Core.Tests.ValidationScenario(
                        name="hostname_simple",
                        validator_type="network",
                        input_value="example",
                        should_succeed=True,
                        expected_value="example",
                        description="Simple hostname",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="hostname_fqdn",
                        validator_type="network",
                        input_value="example.com",
                        should_succeed=True,
                        expected_value="example.com",
                        description="Fully qualified domain name",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="hostname_subdomain",
                        validator_type="network",
                        input_value="sub.example.com",
                        should_succeed=True,
                        expected_value="sub.example.com",
                        description="Hostname with subdomain",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="hostname_hyphen",
                        validator_type="network",
                        input_value="my-host.example.com",
                        should_succeed=True,
                        expected_value="my-host.example.com",
                        description="Hostname with hyphen",
                    ),
                ]
                REQUIRED_SCENARIOS: ClassVar[
                    Sequence[m.Core.Tests.ValidationScenario]
                ] = [
                    m.Core.Tests.ValidationScenario(
                        name="required_valid",
                        validator_type="string",
                        input_value="non-empty",
                        should_succeed=True,
                        expected_value="non-empty",
                        description="Valid non-empty string",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="required_unicode",
                        validator_type="string",
                        input_value="café",
                        should_succeed=True,
                        expected_value="café",
                        description="Unicode characters",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="required_special",
                        validator_type="string",
                        input_value="test@#$%",
                        should_succeed=True,
                        expected_value="test@#$%",
                        description="Special characters",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="required_none",
                        validator_type="string",
                        input_value=None,
                        should_succeed=False,
                        expected_error_contains="empty",
                        description="None value rejection",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="required_empty",
                        validator_type="string",
                        input_value="",
                        should_succeed=False,
                        expected_error_contains="empty",
                        description="Empty string rejection",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="required_whitespace",
                        validator_type="string",
                        input_value="   ",
                        should_succeed=False,
                        expected_error_contains="empty",
                        description="Whitespace-only rejection",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="required_single_char",
                        validator_type="string",
                        input_value="a",
                        should_succeed=True,
                        expected_value="a",
                        description="Single character string",
                    ),
                ]
                CHOICE_SCENARIOS: ClassVar[
                    Sequence[m.Core.Tests.ValidationScenario]
                ] = [
                    m.Core.Tests.ValidationScenario(
                        name="choice_valid_single",
                        validator_type="string",
                        input_value="option1",
                        input_params={
                            "valid_choices": ["option1", "option2", "option3"]
                        },
                        should_succeed=True,
                        expected_value="option1",
                        description="Valid single choice",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="choice_valid_second",
                        validator_type="string",
                        input_value="option2",
                        input_params={
                            "valid_choices": ["option1", "option2", "option3"]
                        },
                        should_succeed=True,
                        expected_value="option2",
                        description="Valid second choice",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="choice_invalid",
                        validator_type="string",
                        input_value="invalid",
                        input_params={"valid_choices": ["option1", "option2"]},
                        should_succeed=False,
                        expected_error_contains="Must be one of",
                        description="Invalid choice",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="choice_case_sensitive",
                        validator_type="string",
                        input_value="OPTION1",
                        input_params={
                            "valid_choices": ["option1", "option2"],
                            "case_sensitive": True,
                        },
                        should_succeed=False,
                        expected_error_contains="Must be one of",
                        description="Case-sensitive choice",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="choice_case_insensitive",
                        validator_type="string",
                        input_value="option1",
                        input_params={
                            "valid_choices": ["option1", "option2"],
                            "case_sensitive": False,
                        },
                        should_succeed=True,
                        expected_value="option1",
                        description="Case-insensitive choice",
                    ),
                ]
                LENGTH_SCENARIOS: ClassVar[
                    Sequence[m.Core.Tests.ValidationScenario]
                ] = [
                    m.Core.Tests.ValidationScenario(
                        name="length_exact",
                        validator_type="string",
                        input_value="12345",
                        input_params={"min_length": 5, "max_length": 5},
                        should_succeed=True,
                        expected_value="12345",
                        description="Exact length match",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="length_within_bounds",
                        validator_type="string",
                        input_value="hello",
                        input_params={"min_length": 3, "max_length": 10},
                        should_succeed=True,
                        expected_value="hello",
                        description="Length within bounds",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="length_below_min",
                        validator_type="string",
                        input_value="hi",
                        input_params={"min_length": 3},
                        should_succeed=False,
                        expected_error_contains="at least",
                        description="Length below minimum",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="length_above_max",
                        validator_type="string",
                        input_value="toolongstring",
                        input_params={"max_length": 5},
                        should_succeed=False,
                        expected_error_contains="no more than",
                        description="Length above maximum",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="length_zero_max",
                        validator_type="string",
                        input_value="",
                        input_params={"min_length": 0, "max_length": 0},
                        should_succeed=True,
                        expected_value="",
                        description="Zero-length string allowed",
                    ),
                ]
                PATTERN_SCENARIOS: ClassVar[
                    Sequence[m.Core.Tests.ValidationScenario]
                ] = [
                    m.Core.Tests.ValidationScenario(
                        name="pattern_email_valid",
                        validator_type="string",
                        input_value="test@example.com",
                        input_params={"pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"},
                        should_succeed=True,
                        expected_value="test@example.com",
                        description="Valid email pattern",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="pattern_digits_only",
                        validator_type="string",
                        input_value="12345",
                        input_params={"pattern": "^\\d+$"},
                        should_succeed=True,
                        expected_value="12345",
                        description="Digits-only pattern",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="pattern_alphanumeric",
                        validator_type="string",
                        input_value="abc123",
                        input_params={"pattern": "^[a-zA-Z0-9]+$"},
                        should_succeed=True,
                        expected_value="abc123",
                        description="Alphanumeric pattern",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="pattern_mismatch",
                        validator_type="string",
                        input_value="invalid@",
                        input_params={"pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"},
                        should_succeed=False,
                        expected_error_contains="format is invalid",
                        description="Pattern mismatch",
                    ),
                ]
                NON_NEGATIVE_SCENARIOS: ClassVar[
                    Sequence[m.Core.Tests.ValidationScenario]
                ] = [
                    m.Core.Tests.ValidationScenario(
                        name="non_negative_zero",
                        validator_type="numeric",
                        input_value=0,
                        should_succeed=True,
                        expected_value=0,
                        description="Zero is non-negative",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="non_negative_positive",
                        validator_type="numeric",
                        input_value=42,
                        should_succeed=True,
                        expected_value=42,
                        description="Positive number",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="non_negative_large",
                        validator_type="numeric",
                        input_value=1000000,
                        should_succeed=True,
                        expected_value=1000000,
                        description="Large positive number",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="non_negative_negative",
                        validator_type="numeric",
                        input_value=-1,
                        should_succeed=False,
                        expected_error_contains="non-negative",
                        description="Negative rejection",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="non_negative_none",
                        validator_type="numeric",
                        input_value=None,
                        should_succeed=False,
                        expected_error_contains="cannot be None",
                        description="None rejection",
                    ),
                ]
                POSITIVE_SCENARIOS: ClassVar[
                    Sequence[m.Core.Tests.ValidationScenario]
                ] = [
                    m.Core.Tests.ValidationScenario(
                        name="positive_one",
                        validator_type="numeric",
                        input_value=1,
                        should_succeed=True,
                        expected_value=1,
                        description="Positive value 1",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="positive_large",
                        validator_type="numeric",
                        input_value=999999,
                        should_succeed=True,
                        expected_value=999999,
                        description="Large positive",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="positive_float",
                        validator_type="numeric",
                        input_value=0.1,
                        should_succeed=True,
                        expected_value=0.1,
                        description="Positive float",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="positive_zero",
                        validator_type="numeric",
                        input_value=0,
                        should_succeed=False,
                        expected_error_contains="positive",
                        description="Zero rejection",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="positive_negative",
                        validator_type="numeric",
                        input_value=-5,
                        should_succeed=False,
                        expected_error_contains="positive",
                        description="Negative rejection",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="positive_none",
                        validator_type="numeric",
                        input_value=None,
                        should_succeed=False,
                        expected_error_contains="cannot be None",
                        description="None rejection",
                    ),
                ]
                RANGE_SCENARIOS: ClassVar[Sequence[m.Core.Tests.ValidationScenario]] = [
                    m.Core.Tests.ValidationScenario(
                        name="range_within_bounds",
                        validator_type="numeric",
                        input_value=5,
                        input_params={"min_value": 1, "max_value": 10},
                        should_succeed=True,
                        expected_value=5,
                        description="Value within range",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="range_at_min",
                        validator_type="numeric",
                        input_value=1,
                        input_params={"min_value": 1, "max_value": 10},
                        should_succeed=True,
                        expected_value=1,
                        description="Value at minimum",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="range_at_max",
                        validator_type="numeric",
                        input_value=10,
                        input_params={"min_value": 1, "max_value": 10},
                        should_succeed=True,
                        expected_value=10,
                        description="Value at maximum",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="range_below_min",
                        validator_type="numeric",
                        input_value=0,
                        input_params={"min_value": 1, "max_value": 10},
                        should_succeed=False,
                        expected_error_contains="at least",
                        description="Value below minimum",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="range_above_max",
                        validator_type="numeric",
                        input_value=11,
                        input_params={"min_value": 1, "max_value": 10},
                        should_succeed=False,
                        expected_error_contains="at most",
                        description="Value above maximum",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="range_negative_range",
                        validator_type="numeric",
                        input_value=-5,
                        input_params={"min_value": -10, "max_value": -1},
                        should_succeed=True,
                        expected_value=-5,
                        description="Negative range",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="range_fractional",
                        validator_type="numeric",
                        input_value=2.5,
                        input_params={"min_value": 0.5, "max_value": 5.5},
                        should_succeed=True,
                        expected_value=2.5,
                        description="Fractional range",
                    ),
                    m.Core.Tests.ValidationScenario(
                        name="range_single_value",
                        validator_type="numeric",
                        input_value=5,
                        input_params={"min_value": 5, "max_value": 5},
                        should_succeed=True,
                        expected_value=5,
                        description="Single value range",
                    ),
                ]

            class ParserScenarios:
                """Centralized parser scenarios - single source of truth."""

                PUBLIC_PARSE_CASES: ClassVar[Sequence[m.Core.Tests.PublicParseCase]] = [
                    m.Core.Tests.PublicParseCase(
                        name="string-to-int",
                        input_value="42",
                        target=int,
                        should_succeed=True,
                        expected_value=42,
                        description="Public parse coerces numeric string into int",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="int-to-str",
                        input_value=42,
                        target=str,
                        should_succeed=True,
                        expected_value="42",
                        description="Public parse coerces int into string",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="string-to-float",
                        input_value="2.2",
                        target=float,
                        should_succeed=True,
                        expected_value=2.2,
                        description="Public parse coerces numeric string into float",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="string-to-bool",
                        input_value="true",
                        target=bool,
                        should_succeed=True,
                        expected_value=True,
                        description="Public parse coerces truthy string into bool",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="none-uses-default",
                        input_value=None,
                        target=int,
                        options=u.ParseOptions[int](default=7),
                        should_succeed=True,
                        expected_value=7,
                        description="Public parse returns default when value is None",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="invalid-uses-default-factory",
                        input_value="x",
                        target=int,
                        options=u.ParseOptions[int](default_factory=lambda: 9),
                        should_succeed=True,
                        expected_value=9,
                        description="Public parse returns default_factory output on failure",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="enum-exact",
                        input_value="inactive",
                        target=c.Core.Tests.StatusEnum,
                        should_succeed=True,
                        expected_value=c.Core.Tests.StatusEnum.INACTIVE,
                        description="Public parse resolves StrEnum exact values",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="enum-case-insensitive",
                        input_value="INACTIVE",
                        target=c.Core.Tests.StatusEnum,
                        options=u.ParseOptions[c.Core.Tests.StatusEnum](
                            case_insensitive=True,
                        ),
                        should_succeed=True,
                        expected_value=c.Core.Tests.StatusEnum.INACTIVE,
                        description="Public parse resolves StrEnum values case-insensitively",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="model-from-mapping",
                        input_value={"name": "parsed", "value": 3},
                        target=m.Core.Tests.SampleModel,
                        should_succeed=True,
                        expected_data={"name": "parsed", "value": 3},
                        description="Public parse materializes canonical test model from mapping",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="invalid-int-fails",
                        input_value="x",
                        target=int,
                        should_succeed=False,
                        description="Public parse fails for non-numeric int input",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="invalid-enum-fails",
                        input_value="missing",
                        target=c.Core.Tests.StatusEnum,
                        should_succeed=False,
                        description="Public parse fails for unknown enum values",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="invalid-model-shape-fails",
                        input_value={"bad": "shape"},
                        target=m.Core.Tests.SampleModel,
                        should_succeed=False,
                        description="Public parse fails for invalid model payloads",
                    ),
                    m.Core.Tests.PublicParseCase(
                        name="invalid-bool-field-context",
                        input_value="maybe",
                        target=bool,
                        options=u.ParseOptions[bool](field_name="flag"),
                        should_succeed=False,
                        error_contains="flag",
                        description="Public parse includes field context on bool failure",
                    ),
                ]

                LDIF_PARSE_SCENARIOS: ClassVar[
                    Sequence[m.Core.Tests.ParserScenario]
                ] = [
                    m.Core.Tests.ParserScenario(
                        name="parse_simple_dn",
                        parser_method="parse",
                        input_data="dn: cn=test,dc=example,dc=com",
                        should_succeed=True,
                        description="Simple DN parsing",
                    ),
                    m.Core.Tests.ParserScenario(
                        name="parse_with_attributes",
                        parser_method="parse",
                        input_data="dn: cn=test,dc=example,dc=com\nobjectClass: person\ncn: test",
                        should_succeed=True,
                        description="DN with attributes",
                    ),
                    m.Core.Tests.ParserScenario(
                        name="parse_invalid_dn",
                        parser_method="parse",
                        input_data="invalid",
                        should_succeed=False,
                        error_contains="invalid",
                        description="Invalid DN format",
                    ),
                ]

            class ReliabilityScenarios:
                """Centralized reliability scenarios - single source of truth."""

                RETRY_SCENARIOS: ClassVar[
                    Sequence[m.Core.Tests.ReliabilityScenario]
                ] = [
                    m.Core.Tests.ReliabilityScenario(
                        name="retry_immediate_success",
                        strategy="retry",
                        settings=t.ConfigMap(
                            root={
                                "max_retries": 3,
                                "backoff_type": "constant",
                                "backoff_ms": 10,
                            },
                        ),
                        simulate_failures=0,
                        expected_state="success",
                        should_succeed=True,
                        description="Operation succeeds immediately",
                    ),
                    m.Core.Tests.ReliabilityScenario(
                        name="retry_after_one_failure",
                        strategy="retry",
                        settings=t.ConfigMap(
                            root={
                                "max_retries": 3,
                                "backoff_type": "constant",
                                "backoff_ms": 10,
                            },
                        ),
                        simulate_failures=1,
                        expected_state="success",
                        should_succeed=True,
                        description="Succeeds after one retry",
                    ),
                    m.Core.Tests.ReliabilityScenario(
                        name="retry_exhausted",
                        strategy="retry",
                        settings=t.ConfigMap(
                            root={
                                "max_retries": 2,
                                "backoff_type": "constant",
                                "backoff_ms": 10,
                            },
                        ),
                        simulate_failures=5,
                        expected_state="exhausted",
                        should_succeed=False,
                        description="All retries exhausted",
                    ),
                ]
                CIRCUIT_BREAKER_SCENARIOS: ClassVar[
                    Sequence[m.Core.Tests.ReliabilityScenario]
                ] = [
                    m.Core.Tests.ReliabilityScenario(
                        name="circuit_initial_closed",
                        strategy="circuit_breaker",
                        settings=t.ConfigMap(
                            root={"failure_threshold": 5, "timeout_ms": 1000}
                        ),
                        simulate_failures=0,
                        expected_state="closed",
                        should_succeed=True,
                        description="Circuit starts in closed state",
                    ),
                    m.Core.Tests.ReliabilityScenario(
                        name="circuit_open_on_threshold",
                        strategy="circuit_breaker",
                        settings=t.ConfigMap(
                            root={"failure_threshold": 2, "timeout_ms": 1000}
                        ),
                        simulate_failures=3,
                        expected_state="open",
                        should_succeed=False,
                        description="Circuit opens after threshold",
                    ),
                ]

            class ValidatingService(s[str]):
                """Service with validation."""

                value_input: str
                min_length: int = c.Core.Tests.TestValidation.MIN_LENGTH_DEFAULT

                @override
                def execute(self) -> p.Result[str]:
                    """Validate and return value."""
                    if len(self.value_input) < self.min_length:
                        return r[str].fail(
                            f"Value must be at least {self.min_length} characters",
                        )
                    return r[str].ok(self.value_input.upper())

            class FailingService(s[str]):
                """Service that always fails."""

                error_message: str = c.Core.Tests.Services.DEFAULT_ERROR_MESSAGE

                @override
                def execute(self) -> p.Result[str]:
                    """Always fails."""
                    return r[str].fail(self.error_message)

            class GetUserServiceAuto(GetUserService):
                """Auto-executing `GetUserService`."""

                auto_execute: ClassVar[bool] = True

            class ValidatingServiceAuto(ValidatingService):
                """Auto-executing `ValidatingService`."""

                auto_execute: ClassVar[bool] = True

            class FailingServiceAuto(FailingService):
                """Auto-executing FailingService."""

                auto_execute: ClassVar[bool] = True

            class UserFactory:
                """Factory for `m.Core.Tests.User` entities using native Python patterns."""

                _counter: ClassVar[count[int]] = count(1)
                _names: ClassVar[t.StrSequence] = [
                    "Alice Johnson",
                    "Bob Smith",
                    "Carol Williams",
                    "David Brown",
                    "Eve Davis",
                ]
                _name_index: ClassVar[int] = 0

                @classmethod
                def _next_name(cls) -> str:
                    """Get next name from rotation."""
                    name = cls._names[cls._name_index % len(cls._names)]
                    cls._name_index += 1
                    return name

                @classmethod
                def build(
                    cls,
                    *,
                    user_id: str | None = None,
                    name: str | None = None,
                    email: str | None = None,
                    is_active: bool = True,
                ) -> m.Core.Tests.User:
                    """Build a `m.Core.Tests.User` instance with optional overrides."""
                    n = next(cls._counter)
                    actual_user_id = user_id if user_id is not None else f"user_{n:03d}"
                    actual_name = name if name is not None else cls._next_name()
                    actual_email = (
                        email if email is not None else f"{actual_user_id}@example.com"
                    )
                    return m.Core.Tests.User(
                        id=actual_user_id,
                        unique_id=actual_user_id,
                        name=actual_name,
                        email=actual_email,
                        active=is_active,
                    )

                @classmethod
                def build_batch(cls, size: int) -> Sequence[m.Core.Tests.User]:
                    """Build multiple `m.Core.Tests.User` instances with auto-generated values."""
                    return [cls.build() for _ in range(size)]

                @classmethod
                def reset(cls) -> None:
                    """Reset factory state for test isolation."""
                    cls._counter = count(1)
                    cls._name_index = 0

            class GetUserServiceFactory:
                """Factory for `GetUserService`."""

                _counter: ClassVar[count[int]] = count(1)

                @classmethod
                def build(
                    cls, *, user_id: str | None = None
                ) -> TestsFlextCoreUtilities.Core.Tests.GetUserService:
                    """Build a `GetUserService` instance."""
                    n = next(cls._counter)
                    actual_user_id = user_id if user_id is not None else f"user_{n:03d}"
                    return TestsFlextCoreUtilities.Core.Tests.GetUserService(
                        user_id=actual_user_id,
                    )

                @classmethod
                def build_batch(
                    cls, size: int
                ) -> Sequence[TestsFlextCoreUtilities.Core.Tests.GetUserService]:
                    """Build multiple `GetUserService` instances with auto-generated values."""
                    return [cls.build() for _ in range(size)]

                @classmethod
                def reset(cls) -> None:
                    """Reset factory state."""
                    cls._counter = count(1)

            class FailingServiceFactory:
                """Factory for FailingService."""

                @classmethod
                def build(
                    cls,
                    *,
                    error_message: str = c.Core.Tests.Services.DEFAULT_ERROR_MESSAGE,
                ) -> TestsFlextCoreUtilities.Core.Tests.FailingService:
                    """Build a FailingService instance."""
                    return TestsFlextCoreUtilities.Core.Tests.FailingService(
                        error_message=error_message,
                    )

                @classmethod
                def build_batch(
                    cls, size: int
                ) -> Sequence[TestsFlextCoreUtilities.Core.Tests.FailingService]:
                    """Build multiple FailingService instances with default error message."""
                    return [cls.build() for _ in range(size)]

            class GetUserServiceAutoFactory:
                """Factory for GetUserServiceAuto."""

                _counter: ClassVar[count[int]] = count(1)

                @classmethod
                def build(
                    cls, *, user_id: str | None = None
                ) -> TestsFlextCoreUtilities.Core.Tests.GetUserServiceAuto:
                    """Build a GetUserServiceAuto instance."""
                    n = next(cls._counter)
                    actual_user_id = user_id if user_id is not None else f"user_{n:03d}"
                    return TestsFlextCoreUtilities.Core.Tests.GetUserServiceAuto(
                        user_id=actual_user_id,
                    )

                @classmethod
                def build_batch(
                    cls, size: int
                ) -> Sequence[TestsFlextCoreUtilities.Core.Tests.GetUserServiceAuto]:
                    """Build multiple GetUserServiceAuto instances with auto-generated values."""
                    return [cls.build() for _ in range(size)]

                @classmethod
                def reset(cls) -> None:
                    """Reset factory state."""
                    cls._counter = count(1)

            class ValidatingServiceAutoFactory:
                """Factory for ValidatingServiceAuto."""

                _words: ClassVar[t.StrSequence] = [
                    "alpha",
                    "bravo",
                    "charlie",
                    "delta",
                    "echo",
                ]
                _word_index: ClassVar[int] = 0

                @classmethod
                def _next_word(cls) -> str:
                    """Get next word from rotation."""
                    word = cls._words[cls._word_index % len(cls._words)]
                    cls._word_index += 1
                    return word

                @classmethod
                def build(
                    cls,
                    *,
                    value_input: str | None = None,
                    min_length: int = c.Core.Tests.TestValidation.MIN_LENGTH_DEFAULT,
                ) -> TestsFlextCoreUtilities.Core.Tests.ValidatingServiceAuto:
                    """Build a ValidatingServiceAuto instance."""
                    actual_value = (
                        value_input if value_input is not None else cls._next_word()
                    )
                    return TestsFlextCoreUtilities.Core.Tests.ValidatingServiceAuto(
                        value_input=actual_value,
                        min_length=min_length,
                    )

                @classmethod
                def build_batch(
                    cls, size: int
                ) -> Sequence[TestsFlextCoreUtilities.Core.Tests.ValidatingServiceAuto]:
                    """Build multiple ValidatingServiceAuto instances with auto-generated values."""
                    return [cls.build() for _ in range(size)]

                @classmethod
                def reset(cls) -> None:
                    """Reset factory state."""
                    cls._word_index = 0

            class ValidatingServiceFactory:
                """Factory for `ValidatingService`."""

                _words: ClassVar[t.StrSequence] = [
                    "alpha",
                    "bravo",
                    "charlie",
                    "delta",
                    "echo",
                ]
                _word_index: ClassVar[int] = 0

                @classmethod
                def _next_word(cls) -> str:
                    """Get next word from rotation."""
                    word = cls._words[cls._word_index % len(cls._words)]
                    cls._word_index += 1
                    return word

                @classmethod
                def build(
                    cls,
                    *,
                    value_input: str | None = None,
                    min_length: int = c.Core.Tests.TestValidation.MIN_LENGTH_DEFAULT,
                ) -> TestsFlextCoreUtilities.Core.Tests.ValidatingService:
                    """Build a `ValidatingService` instance."""
                    actual_value = (
                        value_input if value_input is not None else cls._next_word()
                    )
                    return TestsFlextCoreUtilities.Core.Tests.ValidatingService(
                        value_input=actual_value,
                        min_length=min_length,
                    )

                @classmethod
                def build_batch(
                    cls, size: int
                ) -> Sequence[TestsFlextCoreUtilities.Core.Tests.ValidatingService]:
                    """Build multiple `ValidatingService` instances with auto-generated values."""
                    return [cls.build() for _ in range(size)]

                @classmethod
                def reset(cls) -> None:
                    """Reset factory state."""
                    cls._word_index = 0

            class FailingServiceAutoFactory:
                """Factory for FailingServiceAuto."""

                @classmethod
                def build(
                    cls,
                    *,
                    error_message: str = c.Core.Tests.Services.DEFAULT_ERROR_MESSAGE,
                ) -> TestsFlextCoreUtilities.Core.Tests.FailingServiceAuto:
                    """Build a FailingServiceAuto instance."""
                    return TestsFlextCoreUtilities.Core.Tests.FailingServiceAuto(
                        error_message=error_message,
                    )

                @classmethod
                def build_batch(
                    cls, size: int
                ) -> Sequence[TestsFlextCoreUtilities.Core.Tests.FailingServiceAuto]:
                    """Build multiple FailingServiceAuto instances with default error message."""
                    return [cls.build() for _ in range(size)]

            class ServiceTestCaseFactory:
                """Factory for m.Core.Tests.ServiceTestCase."""

                _service_types: ClassVar[Sequence[c.Core.Tests.ServiceTestType]] = [
                    c.Core.Tests.ServiceTestType.GET_USER,
                    c.Core.Tests.ServiceTestType.VALIDATE,
                    c.Core.Tests.ServiceTestType.FAIL,
                ]
                _type_index: ClassVar[int] = 0
                _words: ClassVar[t.StrSequence] = [
                    "test",
                    "sample",
                    "example",
                    "demo",
                    "data",
                ]
                _word_index: ClassVar[int] = 0

                @classmethod
                def _next_type(cls) -> c.Core.Tests.ServiceTestType:
                    """Get next service type from rotation."""
                    service_type = cls._service_types[
                        cls._type_index % len(cls._service_types)
                    ]
                    cls._type_index += 1
                    return service_type

                @classmethod
                def _next_word(cls) -> str:
                    """Get next word from rotation."""
                    word = cls._words[cls._word_index % len(cls._words)]
                    cls._word_index += 1
                    return word

                @classmethod
                def build(
                    cls,
                    *,
                    service_type: c.Core.Tests.ServiceTestType | None = None,
                    input_value: str | None = None,
                    expected_success: bool = True,
                    expected_error: str | None = None,
                    extra_param: int = c.Core.Tests.TestValidation.MIN_LENGTH_DEFAULT,
                    description: str | None = None,
                ) -> m.Core.Tests.ServiceTestCase:
                    """Build a m.Core.Tests.ServiceTestCase instance."""
                    actual_type = (
                        service_type if service_type is not None else cls._next_type()
                    )
                    actual_input = (
                        input_value if input_value is not None else cls._next_word()
                    )
                    actual_description = (
                        description
                        if description is not None
                        else f"Test case for {actual_type} with {actual_input}"
                    )
                    return m.Core.Tests.ServiceTestCase(
                        service_type=actual_type,
                        input_value=actual_input,
                        expected_success=expected_success,
                        expected_error=expected_error,
                        extra_param=extra_param,
                        description=actual_description,
                    )

                @classmethod
                def build_batch(
                    cls, size: int
                ) -> Sequence[m.Core.Tests.ServiceTestCase]:
                    """Build multiple m.Core.Tests.ServiceTestCase instances with auto-generated values."""
                    return [cls.build() for _ in range(size)]

                @classmethod
                def reset(cls) -> None:
                    """Reset factory state."""
                    cls._type_index = 0
                    cls._word_index = 0

            class ServiceFactoryRegistry:
                """Registry for service factories using pattern matching."""

                @classmethod
                def create_service(
                    cls,
                    case: m.Core.Tests.ServiceTestCase,
                ) -> (
                    TestsFlextCoreUtilities.Core.Tests.GetUserService
                    | TestsFlextCoreUtilities.Core.Tests.ValidatingService
                    | TestsFlextCoreUtilities.Core.Tests.FailingService
                ):
                    """Create appropriate service based on case type using pattern matching."""
                    service: (
                        TestsFlextCoreUtilities.Core.Tests.GetUserService
                        | TestsFlextCoreUtilities.Core.Tests.ValidatingService
                        | TestsFlextCoreUtilities.Core.Tests.FailingService
                    )
                    match case.service_type:
                        case c.Core.Tests.ServiceTestType.GET_USER:
                            service = TestsFlextCoreUtilities.Core.Tests.GetUserServiceFactory.build(
                                user_id=case.input_value
                            )
                        case c.Core.Tests.ServiceTestType.VALIDATE:
                            service = TestsFlextCoreUtilities.Core.Tests.ValidatingServiceFactory.build(
                                value_input=case.input_value,
                                min_length=case.extra_param,
                            )
                        case c.Core.Tests.ServiceTestType.FAIL:
                            service = TestsFlextCoreUtilities.Core.Tests.FailingServiceFactory.build(
                                error_message=case.input_value
                                or c.Core.Tests.Services.DEFAULT_ERROR_MESSAGE
                            )
                        case _:
                            msg = f"Unsupported service type: {case.service_type}"
                            raise ValueError(msg)
                    return service

            class TestDataGenerators:
                """Advanced test data generators using comprehensions and patterns."""

                @staticmethod
                def generate_user_success_cases(
                    num_cases: int = 3,
                ) -> Sequence[m.Core.Tests.ServiceTestCase]:
                    """Generate successful user service test cases."""
                    return [
                        m.Core.Tests.ServiceTestCase(
                            service_type=c.Core.Tests.ServiceTestType.GET_USER,
                            input_value=str(i * 100 + 1),
                            description=f"Valid user ID {i}",
                        )
                        for i in range(1, num_cases + 1)
                    ]

                @staticmethod
                def generate_validation_success_cases(
                    num_cases: int = 2,
                ) -> Sequence[m.Core.Tests.ServiceTestCase]:
                    """Generate successful validation test cases."""
                    return [
                        m.Core.Tests.ServiceTestCase(
                            service_type=c.Core.Tests.ServiceTestType.VALIDATE,
                            input_value=f"value_{i}",
                            description=f"Valid input {i}",
                        )
                        for i in range(1, num_cases + 1)
                    ] + [
                        m.Core.Tests.ServiceTestCase(
                            service_type=c.Core.Tests.ServiceTestType.VALIDATE,
                            input_value="test",
                            extra_param=2,
                            description="Custom min length",
                        ),
                    ]

                @staticmethod
                def generate_validation_failure_cases() -> Sequence[
                    m.Core.Tests.ServiceTestCase
                ]:
                    """Generate validation failure test cases."""
                    return [
                        m.Core.Tests.ServiceTestCase(
                            service_type=c.Core.Tests.ServiceTestType.VALIDATE,
                            input_value="ab",
                            expected_success=False,
                            expected_error="must be at least 3 characters",
                            description="Too short input",
                        ),
                        m.Core.Tests.ServiceTestCase(
                            service_type=c.Core.Tests.ServiceTestType.VALIDATE,
                            input_value="x",
                            expected_success=False,
                            expected_error="must be at least 5 characters",
                            extra_param=5,
                            description="Custom length requirement",
                        ),
                    ]

            class ServiceTestCases:
                """Unified factory for all test cases using advanced patterns."""

                @staticmethod
                def user_success() -> Sequence[m.Core.Tests.ServiceTestCase]:
                    """Generate cached-style success cases on demand."""
                    return TestsFlextCoreUtilities.Core.Tests.TestDataGenerators.generate_user_success_cases()

                @staticmethod
                def validate_success() -> Sequence[m.Core.Tests.ServiceTestCase]:
                    """Generate cached-style validation success cases on demand."""
                    return TestsFlextCoreUtilities.Core.Tests.TestDataGenerators.generate_validation_success_cases()

                @staticmethod
                def validate_failure() -> Sequence[m.Core.Tests.ServiceTestCase]:
                    """Generate cached-style validation failure cases on demand."""
                    return TestsFlextCoreUtilities.Core.Tests.TestDataGenerators.generate_validation_failure_cases()

                @staticmethod
                def create_service(
                    case: m.Core.Tests.ServiceTestCase,
                ) -> (
                    TestsFlextCoreUtilities.Core.Tests.GetUserService
                    | TestsFlextCoreUtilities.Core.Tests.ValidatingService
                    | TestsFlextCoreUtilities.Core.Tests.FailingService
                ):
                    """Create appropriate service based on case type."""
                    return TestsFlextCoreUtilities.Core.Tests.ServiceFactoryRegistry.create_service(
                        case
                    )

            class GenericModelFactory:
                """Factories for generic reusable models (Value, Snapshot, Progress)."""

                @staticmethod
                def operation_progress(
                    success: int = 0,
                    failure: int = 0,
                    skipped: int = 0,
                ) -> m.Operation:
                    """Create OperationProgress."""
                    return m.Operation(
                        success_count=success,
                        failure_count=failure,
                        skipped_count=skipped,
                        metadata=t.Dict({}),
                    )

                @staticmethod
                def conversion_progress() -> m.Conversion:
                    """Create ConversionProgress."""
                    return m.Conversion(
                        converted=[],
                        errors=[],
                        warnings=[],
                        skipped=[],
                        metadata=t.Dict({}),
                    )

            @staticmethod
            def reset_all_factories() -> None:
                """Reset all factory states for test isolation."""
                TestsFlextCoreUtilities.Core.Tests.UserFactory.reset()
                TestsFlextCoreUtilities.Core.Tests.GetUserServiceFactory.reset()
                TestsFlextCoreUtilities.Core.Tests.ValidatingServiceFactory.reset()
                TestsFlextCoreUtilities.Core.Tests.GetUserServiceAutoFactory.reset()
                TestsFlextCoreUtilities.Core.Tests.ValidatingServiceAutoFactory.reset()
                TestsFlextCoreUtilities.Core.Tests.ServiceTestCaseFactory.reset()

            class Contract:
                """Shared contract for text utility behavior."""

                SAFE_STRING_VALID_CASES: ClassVar[Sequence[tuple[str, str]]] = (
                    c.Core.Tests.SAFE_STRING_VALID_CASES
                )
                SAFE_STRING_INVALID_CASES: ClassVar[
                    Sequence[tuple[str | None, str]]
                ] = c.Core.Tests.SAFE_STRING_INVALID_CASES
                FORMAT_APP_ID_CASES: ClassVar[Sequence[tuple[str, str]]] = (
                    c.Core.Tests.FORMAT_APP_ID_CASES
                )

                @staticmethod
                def assert_safe_string_valid(raw: str, expected: str) -> None:
                    """Assert safe string normalization for valid input."""
                    assert u.safe_string(raw) == expected

                @staticmethod
                def assert_format_app_id(raw: str, expected: str) -> None:
                    """Assert app id formatting behavior."""
                    assert u.format_app_id(raw) == expected

            @staticmethod
            def assert_safe_string_valid(raw: str, expected: str) -> None:
                """Assert safe string normalization for valid input."""
                TestsFlextCoreUtilities.Core.Tests.Contract.assert_safe_string_valid(
                    raw,
                    expected,
                )

            @staticmethod
            def assert_format_app_id(raw: str, expected: str) -> None:
                """Assert app id formatting behavior."""
                TestsFlextCoreUtilities.Core.Tests.Contract.assert_format_app_id(
                    raw,
                    expected,
                )

            class FunctionalExternalService:
                """Mock external service for integration testing.

                Provides real functionality for testing service integration patterns
                with dependency injection and result handling.
                """

                def __init__(self) -> None:
                    """Initialize external service with empty state."""
                    self.processed_items: MutableSequence[str] = []
                    self.call_count = 0

                def process(self, input_data: str) -> p.Result[str]:
                    """Process input data by prefixing with 'processed_'.

                    Args:
                        input_data: String to process

                    Returns:
                        r[str]: Processed result or failure

                    """
                    try:
                        self.call_count += 1
                        processed = f"processed_{input_data}"
                        self.processed_items.append(processed)
                        return r[str].ok(processed)
                    except (ValueError, TypeError, RuntimeError) as e:
                        return r[str].fail(f"Processing failed: {e}")

                def get_call_count(self) -> int:
                    """Get number of times process() was called."""
                    return self.call_count


u = TestsFlextCoreUtilities

__all__: list[str] = ["TestsFlextCoreUtilities", "u"]
