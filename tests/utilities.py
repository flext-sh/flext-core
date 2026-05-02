"""Utilities for flext-core tests."""

from __future__ import annotations

from collections.abc import (
    MutableSequence,
    Sequence,
)
from itertools import count
from typing import Annotated, ClassVar, override

from flext_tests import e, h, m as tm, r, u
from flext_tests.base import s

from tests import p, t
from tests.constants import c
from tests.models import m


class TestsFlextUtilities(u):
    """Utilities for flext-core tests."""

    class Tests(u.Tests):
        """flext-core test utilities namespace."""

        # --- from test_registry_full_coverage.py ---

        class Handler(h[t.JsonPayload, t.JsonPayload]):
            """Simple handler used by public registry scenarios."""

            @override
            def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
                return r[t.JsonPayload].ok(message)

        class FalseyDispatcher(p.Dispatcher):
            """Dispatcher that is present but reports itself as unavailable."""

            def __bool__(self) -> bool:
                """Return False to indicate dispatcher is unavailable."""
                return False

            @override
            def publish(
                self,
                event: p.Routable | t.SequenceOf[p.Routable],
            ) -> p.Result[bool]:
                _ = event
                return r[bool].ok(True)

            @override
            def register_handler(
                self,
                handler: t.DispatchableHandler,
                *,
                is_event: bool = False,
            ) -> p.Result[bool]:
                _ = handler
                _ = is_event
                return r[bool].ok(True)

            @override
            def dispatch(self, message: p.Routable) -> p.Result[t.JsonPayload]:
                _ = message
                return r[t.JsonPayload].fail(c.Tests.DISPATCHER_UNCONFIGURED)

        class FailDispatcher(p.Dispatcher):
            """Dispatcher that rejects public handler registration."""

            @override
            def publish(
                self,
                event: p.Routable | t.SequenceOf[p.Routable],
            ) -> p.Result[bool]:
                _ = event
                return r[bool].ok(True)

            @override
            def register_handler(
                self,
                handler: t.DispatchableHandler,
                *,
                is_event: bool = False,
            ) -> p.Result[bool]:
                _ = handler
                _ = is_event
                return r[bool].fail(c.Tests.DISPATCHER_FAIL)

            @override
            def dispatch(self, message: p.Routable) -> p.Result[t.JsonPayload]:
                _ = message
                return r[t.JsonPayload].fail(c.Tests.DISPATCHER_FAIL)

        class OkDispatcher(p.Dispatcher):
            """Dispatcher that accepts public registry operations."""

            @override
            def publish(
                self,
                event: p.Routable | t.SequenceOf[p.Routable],
            ) -> p.Result[bool]:
                _ = event
                return r[bool].ok(True)

            @override
            def register_handler(
                self,
                handler: t.DispatchableHandler,
                *,
                is_event: bool = False,
            ) -> p.Result[bool]:
                _ = handler
                _ = is_event
                return r[bool].ok(True)

            @override
            def dispatch(self, message: p.Routable) -> p.Result[t.JsonPayload]:
                _ = message
                return r[t.JsonPayload].ok(True)

        @staticmethod
        def success_cases() -> t.SequenceOf[tuple[str, str]]:
            return [
                (
                    c.Tests.USER_IDS_SUCCESS[0],
                    "Valid user ID",
                ),
                (
                    c.Tests.USER_IDS_SUCCESS[1],
                    "Another valid user ID",
                ),
                (
                    c.Tests.USER_IDS_SUCCESS[2],
                    "Third valid user ID",
                ),
            ]

        @staticmethod
        def failure_cases() -> t.SequenceOf[tuple[str, str, str]]:
            return [
                (
                    "invalid",
                    "not found",
                    "Invalid user ID",
                ),
                (
                    "",
                    "not found",
                    "Empty user ID",
                ),
            ]

        @staticmethod
        def railway_success_cases() -> t.SequenceOf[
            tuple[t.StrSequence, t.StrSequence, int, str]
        ]:
            return [
                (
                    [c.Tests.USER_IDS_SUCCESS[0]],
                    [],
                    1,
                    "Simple user retrieval",
                ),
                (
                    [c.Tests.USER_IDS_SUCCESS[1]],
                    [c.Tests.RAILWAY_OPERATION_GET_EMAIL],
                    2,
                    "User to email transformation",
                ),
                (
                    [c.Tests.USER_IDS_SUCCESS[2]],
                    [
                        c.Tests.RAILWAY_OPERATION_GET_EMAIL,
                        c.Tests.RAILWAY_OPERATION_SEND_EMAIL,
                        c.Tests.RAILWAY_OPERATION_GET_STATUS,
                    ],
                    4,
                    "Full pipeline: user -> email -> send -> status",
                ),
            ]

        @staticmethod
        def multi_operation_cases() -> t.SequenceOf[tuple[str, int, t.JsonMapping]]:
            return [
                (
                    c.Tests.RAILWAY_OPERATION_DOUBLE,
                    5,
                    {
                        c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_DOUBLE,
                        c.Tests.OPERATION_RESULT_KEY: 10,
                    },
                ),
                (
                    c.Tests.RAILWAY_OPERATION_SQUARE,
                    4,
                    {
                        c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_SQUARE,
                        c.Tests.OPERATION_RESULT_KEY: 16,
                    },
                ),
                (
                    c.Tests.RAILWAY_OPERATION_NEGATE,
                    7,
                    {
                        c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_NEGATE,
                        c.Tests.OPERATION_RESULT_KEY: -7,
                    },
                ),
                (
                    c.Tests.RAILWAY_OPERATION_DOUBLE,
                    0,
                    {
                        c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_DOUBLE,
                        c.Tests.OPERATION_RESULT_KEY: 0,
                    },
                ),
                (
                    c.Tests.RAILWAY_OPERATION_SQUARE,
                    1,
                    {
                        c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_SQUARE,
                        c.Tests.OPERATION_RESULT_KEY: 1,
                    },
                ),
            ]

        class GetUserService(s[m.BaseModel]):
            """Service to get user."""

            user_id: Annotated[
                str,
                u.Field(description="Identifier of the user to fetch."),
            ] = ""

            @override
            def execute(self) -> p.Result[tm.Tests.User]:
                if self.user_id in c.Tests.USER_IDS_INVALID:
                    return r[tm.Tests.User].fail(c.Tests.USER_NOT_FOUND)
                return r[tm.Tests.User].ok(
                    tm.Tests.User(
                        id=self.user_id,
                        unique_id=self.user_id,
                        name=f"{c.Tests.DEFAULT_USER_NAME_PREFIX}{self.user_id}",
                        email=f"user{self.user_id}{c.Tests.DEFAULT_EMAIL_DOMAIN}",
                    ),
                )

        class SendEmailService(s[m.BaseModel]):
            """Service to send email."""

            to: Annotated[
                str,
                u.Field(description="Destination email address."),
            ] = ""
            subject: Annotated[
                str,
                u.Field(description="Email subject line."),
            ] = ""

            @override
            def execute(self) -> p.Result[m.Tests.EmailResponse]:
                if "@" not in self.to:
                    return r[m.Tests.EmailResponse].fail(c.Tests.INVALID_EMAIL)
                return r[m.Tests.EmailResponse].ok(
                    m.Tests.EmailResponse(status="sent", message_id=f"msg-{self.to}"),
                )

        class ValidationService(s[t.JsonMapping]):
            """Service to validate values."""

            value: Annotated[
                int,
                u.Field(description="Integer value to validate."),
            ] = 0

            @override
            def execute(self) -> p.Result[t.JsonMapping]:
                if self.value < 0:
                    return r[t.JsonMapping].fail(c.Tests.VALUE_TOO_LOW)
                if self.value > c.Tests.MAX_VALUE:
                    return r[t.JsonMapping].fail(c.Tests.VALUE_TOO_HIGH)
                return r[t.JsonMapping].ok(
                    {"valid": True, "value": self.value},
                )

        class MultiOperationService(s[t.JsonMapping]):
            """Service for multiple operations."""

            operation: Annotated[
                str,
                u.Field(description="Operation name (double / square / ...)."),
            ] = ""
            value: Annotated[
                int,
                u.Field(description="Numeric operand for the operation."),
            ] = 0

            @override
            def execute(self) -> p.Result[t.JsonMapping]:
                match self.operation:
                    case c.Tests.RAILWAY_OPERATION_DOUBLE:
                        return r[t.JsonMapping].ok(
                            {
                                c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_DOUBLE,
                                c.Tests.OPERATION_RESULT_KEY: self.value
                                * c.Tests.OPERATION_FACTORS[
                                    c.Tests.RAILWAY_OPERATION_DOUBLE
                                ],
                            },
                        )
                    case c.Tests.RAILWAY_OPERATION_SQUARE:
                        return r[t.JsonMapping].ok(
                            {
                                c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_SQUARE,
                                c.Tests.OPERATION_RESULT_KEY: self.value**2,
                            },
                        )
                    case c.Tests.RAILWAY_OPERATION_NEGATE:
                        return r[t.JsonMapping].ok(
                            {
                                c.Tests.OPERATION_NAME_KEY: c.Tests.RAILWAY_OPERATION_NEGATE,
                                c.Tests.OPERATION_RESULT_KEY: self.value
                                * c.Tests.OPERATION_FACTORS[
                                    c.Tests.RAILWAY_OPERATION_NEGATE
                                ],
                            },
                        )
                    case _:
                        return r[t.JsonMapping].fail(
                            f"{c.Tests.UNKNOWN_OPERATION_PREFIX} {self.operation}"
                        )

        @staticmethod
        def value_lt_100(data: t.JsonMapping) -> bool:
            value = data.get("value")
            return isinstance(value, int) and value < 100

        @staticmethod
        def make[T](service_type: type[T], **kwargs: t.Scalar) -> T:
            return service_type(**kwargs)

        @staticmethod
        def create_user_service(
            case: m.Tests.ServiceTestCase,
        ) -> TestsFlextUtilities.Tests.GetUserService:
            """Create a user service from a documented service case."""
            return TestsFlextUtilities.Tests.make(
                TestsFlextUtilities.Tests.GetUserService,
                user_id=case.user_id or case.input_value or "",
            )

        @staticmethod
        def execute_v1_pipeline(
            case: m.Tests.RailwayTestCase,
        ) -> p.Result[str | tm.Tests.User | m.Tests.EmailResponse]:
            """Execute the documented V1 railway pipeline."""
            if not case.user_ids:
                return r[str | tm.Tests.User | m.Tests.EmailResponse].fail(
                    c.Tests.NO_USER_IDS_PROVIDED,
                )
            user_result: p.Result[tm.Tests.User] = TestsFlextUtilities.Tests.make(
                TestsFlextUtilities.Tests.GetUserService,
                user_id=case.user_ids[0],
            ).execute()
            result: p.Result[str | tm.Tests.User | m.Tests.EmailResponse] = (
                user_result.map(lambda user: user)
            )
            for operation in case.operations:
                if operation == "get_email":
                    result = result.map(
                        lambda user: (
                            user.email if isinstance(user, tm.Tests.User) else str(user)
                        ),
                    )
                elif operation == "send_email":
                    email_result: p.Result[m.Tests.EmailResponse] = result.flat_map(
                        lambda email: TestsFlextUtilities.Tests.make(
                            TestsFlextUtilities.Tests.SendEmailService,
                            to=str(email),
                            subject="Test",
                        ).execute(),
                    )
                    result = email_result.map(lambda response: response)
                elif operation == "get_status":
                    result = result.map(
                        lambda response: (
                            response.status
                            if isinstance(response, m.Tests.EmailResponse)
                            else str(response)
                        ),
                    )
            return result

        @staticmethod
        def execute_v2_pipeline(
            case: m.Tests.RailwayTestCase,
        ) -> tm.Tests.User | str:
            """Execute the documented V2 railway pipeline."""
            if not case.user_ids:
                msg = c.Tests.NO_USER_IDS_PROVIDED
                raise e.BaseError(msg)
            raw_user_result = TestsFlextUtilities.Tests.make(
                TestsFlextUtilities.Tests.GetUserService,
                user_id=case.user_ids[0],
            ).execute()
            if raw_user_result.failure:
                msg = raw_user_result.error or c.Tests.USER_NOT_FOUND
                raise e.BaseError(msg)
            raw_user = raw_user_result.unwrap_or(None)
            if not isinstance(raw_user, tm.Tests.User):
                msg = c.Tests.USER_NOT_FOUND
                raise e.BaseError(msg)
            user: tm.Tests.User | str = raw_user
            for operation in case.operations:
                if operation == "get_email":
                    user = user.email if isinstance(user, tm.Tests.User) else user
                elif operation == "send_email":
                    email_to = user if isinstance(user, str) else str(user)
                    raw_response_result = TestsFlextUtilities.Tests.make(
                        TestsFlextUtilities.Tests.SendEmailService,
                        to=email_to,
                        subject="Test",
                    ).execute()
                    if raw_response_result.failure:
                        msg = raw_response_result.error or c.Tests.INVALID_EMAIL
                        raise e.BaseError(msg)
                    raw_response = raw_response_result.unwrap_or(None)
                    if not isinstance(raw_response, m.Tests.EmailResponse):
                        msg = c.Tests.INVALID_EMAIL
                        raise e.BaseError(msg)
                    response_obj: m.Tests.EmailResponse = raw_response
                    user = response_obj.status
            return user

        class ValidationScenarios:
            """Centralized validation scenarios - single source of truth."""

            URI_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
                m.Tests.ValidationScenario(
                    name="uri_valid_http",
                    validator_type="network",
                    input_value="http://example.com",
                    should_succeed=True,
                    expected_value="http://example.com",
                    description="Valid HTTP URI",
                ),
                m.Tests.ValidationScenario(
                    name="uri_valid_https",
                    validator_type="network",
                    input_value="https://example.com",
                    should_succeed=True,
                    expected_value="https://example.com",
                    description="Valid HTTPS URI",
                ),
                m.Tests.ValidationScenario(
                    name="uri_with_port",
                    validator_type="network",
                    input_value="https://example.com:8080",
                    should_succeed=True,
                    expected_value="https://example.com:8080",
                    description="URI with custom port",
                ),
                m.Tests.ValidationScenario(
                    name="uri_with_path",
                    validator_type="network",
                    input_value="https://example.com/path/to/resource",
                    should_succeed=True,
                    expected_value="https://example.com/path/to/resource",
                    description="URI with path",
                ),
                m.Tests.ValidationScenario(
                    name="uri_with_query",
                    validator_type="network",
                    input_value="https://example.com/path?key=value",
                    should_succeed=True,
                    expected_value="https://example.com/path?key=value",
                    description="URI with query parameters",
                ),
                m.Tests.ValidationScenario(
                    name="uri_with_fragment",
                    validator_type="network",
                    input_value="https://example.com/path#section",
                    should_succeed=True,
                    expected_value="https://example.com/path#section",
                    description="URI with fragment",
                ),
                m.Tests.ValidationScenario(
                    name="uri_none",
                    validator_type="network",
                    input_value=None,
                    should_succeed=False,
                    expected_error_contains="cannot be None",
                    description="None URI rejection",
                ),
                m.Tests.ValidationScenario(
                    name="uri_empty",
                    validator_type="network",
                    input_value="",
                    should_succeed=False,
                    expected_error_contains="cannot be empty",
                    description="Empty URI rejection",
                ),
                m.Tests.ValidationScenario(
                    name="uri_invalid_scheme",
                    validator_type="network",
                    input_value="ftp://example.com",
                    should_succeed=False,
                    expected_error_contains="not in allowed",
                    description="Invalid URI scheme",
                ),
                m.Tests.ValidationScenario(
                    name="uri_malformed",
                    validator_type="network",
                    input_value="not a valid uri",
                    should_succeed=False,
                    expected_error_contains="not a valid",
                    description="Malformed URI",
                ),
                m.Tests.ValidationScenario(
                    name="uri_whitespace",
                    validator_type="network",
                    input_value="   ",
                    should_succeed=False,
                    expected_error_contains="cannot be empty",
                    description="Whitespace-only URI",
                ),
                m.Tests.ValidationScenario(
                    name="uri_custom_scheme",
                    validator_type="network",
                    input_value="custom://example.com",
                    input_params={"allowed_schemes": ["http", "https", "custom"]},
                    should_succeed=True,
                    expected_value="custom://example.com",
                    description="Custom scheme with allowlist",
                ),
            ]
            PORT_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
                m.Tests.ValidationScenario(
                    name="port_valid_80",
                    validator_type="network",
                    input_value=80,
                    should_succeed=True,
                    expected_value=80,
                    description="Valid port 80 (HTTP)",
                ),
                m.Tests.ValidationScenario(
                    name="port_valid_443",
                    validator_type="network",
                    input_value=443,
                    should_succeed=True,
                    expected_value=443,
                    description="Valid port 443 (HTTPS)",
                ),
                m.Tests.ValidationScenario(
                    name="port_valid_8080",
                    validator_type="network",
                    input_value=8080,
                    should_succeed=True,
                    expected_value=8080,
                    description="Valid port 8080",
                ),
                m.Tests.ValidationScenario(
                    name="port_valid_1",
                    validator_type="network",
                    input_value=1,
                    should_succeed=True,
                    expected_value=1,
                    description="Valid port 1 (minimum)",
                ),
                m.Tests.ValidationScenario(
                    name="port_valid_65535",
                    validator_type="network",
                    input_value=65535,
                    should_succeed=True,
                    expected_value=65535,
                    description="Valid port 65535 (maximum)",
                ),
                m.Tests.ValidationScenario(
                    name="port_none",
                    validator_type="network",
                    input_value=None,
                    should_succeed=False,
                    expected_error_contains="cannot be None",
                    description="None port rejection",
                ),
                m.Tests.ValidationScenario(
                    name="port_zero",
                    validator_type="network",
                    input_value=0,
                    should_succeed=False,
                    expected_error_contains="must be between",
                    description="Port zero rejection",
                ),
                m.Tests.ValidationScenario(
                    name="port_negative",
                    validator_type="network",
                    input_value=-1,
                    should_succeed=False,
                    expected_error_contains="must be between",
                    description="Negative port rejection",
                ),
                m.Tests.ValidationScenario(
                    name="port_above_max",
                    validator_type="network",
                    input_value=65536,
                    should_succeed=False,
                    expected_error_contains="at most",
                    description="Port above maximum",
                ),
            ]
            HOSTNAME_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
                m.Tests.ValidationScenario(
                    name="hostname_simple",
                    validator_type="network",
                    input_value="example",
                    should_succeed=True,
                    expected_value="example",
                    description="Simple hostname",
                ),
                m.Tests.ValidationScenario(
                    name="hostname_fqdn",
                    validator_type="network",
                    input_value="example.com",
                    should_succeed=True,
                    expected_value="example.com",
                    description="Fully qualified domain name",
                ),
                m.Tests.ValidationScenario(
                    name="hostname_subdomain",
                    validator_type="network",
                    input_value="sub.example.com",
                    should_succeed=True,
                    expected_value="sub.example.com",
                    description="Hostname with subdomain",
                ),
                m.Tests.ValidationScenario(
                    name="hostname_hyphen",
                    validator_type="network",
                    input_value="my-host.example.com",
                    should_succeed=True,
                    expected_value="my-host.example.com",
                    description="Hostname with hyphen",
                ),
            ]
            REQUIRED_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
                m.Tests.ValidationScenario(
                    name="required_valid",
                    validator_type="string",
                    input_value="non-empty",
                    should_succeed=True,
                    expected_value="non-empty",
                    description="Valid non-empty string",
                ),
                m.Tests.ValidationScenario(
                    name="required_unicode",
                    validator_type="string",
                    input_value="café",
                    should_succeed=True,
                    expected_value="café",
                    description="Unicode characters",
                ),
                m.Tests.ValidationScenario(
                    name="required_special",
                    validator_type="string",
                    input_value="test@#$%",
                    should_succeed=True,
                    expected_value="test@#$%",
                    description="Special characters",
                ),
                m.Tests.ValidationScenario(
                    name="required_none",
                    validator_type="string",
                    input_value=None,
                    should_succeed=False,
                    expected_error_contains="empty",
                    description="None value rejection",
                ),
                m.Tests.ValidationScenario(
                    name="required_empty",
                    validator_type="string",
                    input_value="",
                    should_succeed=False,
                    expected_error_contains="empty",
                    description="Empty string rejection",
                ),
                m.Tests.ValidationScenario(
                    name="required_whitespace",
                    validator_type="string",
                    input_value="   ",
                    should_succeed=False,
                    expected_error_contains="empty",
                    description="Whitespace-only rejection",
                ),
                m.Tests.ValidationScenario(
                    name="required_single_char",
                    validator_type="string",
                    input_value="a",
                    should_succeed=True,
                    expected_value="a",
                    description="Single character string",
                ),
            ]
            CHOICE_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
                m.Tests.ValidationScenario(
                    name="choice_valid_single",
                    validator_type="string",
                    input_value="option1",
                    input_params={"valid_choices": ["option1", "option2", "option3"]},
                    should_succeed=True,
                    expected_value="option1",
                    description="Valid single choice",
                ),
                m.Tests.ValidationScenario(
                    name="choice_valid_second",
                    validator_type="string",
                    input_value="option2",
                    input_params={"valid_choices": ["option1", "option2", "option3"]},
                    should_succeed=True,
                    expected_value="option2",
                    description="Valid second choice",
                ),
                m.Tests.ValidationScenario(
                    name="choice_invalid",
                    validator_type="string",
                    input_value="invalid",
                    input_params={"valid_choices": ["option1", "option2"]},
                    should_succeed=False,
                    expected_error_contains="Must be one of",
                    description="Invalid choice",
                ),
                m.Tests.ValidationScenario(
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
                m.Tests.ValidationScenario(
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
            LENGTH_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
                m.Tests.ValidationScenario(
                    name="length_exact",
                    validator_type="string",
                    input_value="12345",
                    input_params={"min_length": 5, "max_length": 5},
                    should_succeed=True,
                    expected_value="12345",
                    description="Exact length match",
                ),
                m.Tests.ValidationScenario(
                    name="length_within_bounds",
                    validator_type="string",
                    input_value="hello",
                    input_params={"min_length": 3, "max_length": 10},
                    should_succeed=True,
                    expected_value="hello",
                    description="Length within bounds",
                ),
                m.Tests.ValidationScenario(
                    name="length_below_min",
                    validator_type="string",
                    input_value="hi",
                    input_params={"min_length": 3},
                    should_succeed=False,
                    expected_error_contains="at least",
                    description="Length below minimum",
                ),
                m.Tests.ValidationScenario(
                    name="length_above_max",
                    validator_type="string",
                    input_value="toolongstring",
                    input_params={"max_length": 5},
                    should_succeed=False,
                    expected_error_contains="no more than",
                    description="Length above maximum",
                ),
                m.Tests.ValidationScenario(
                    name="length_zero_max",
                    validator_type="string",
                    input_value="",
                    input_params={"min_length": 0, "max_length": 0},
                    should_succeed=True,
                    expected_value="",
                    description="Zero-length string allowed",
                ),
            ]
            PATTERN_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
                m.Tests.ValidationScenario(
                    name="pattern_email_valid",
                    validator_type="string",
                    input_value="test@example.com",
                    input_params={"pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"},
                    should_succeed=True,
                    expected_value="test@example.com",
                    description="Valid email pattern",
                ),
                m.Tests.ValidationScenario(
                    name="pattern_digits_only",
                    validator_type="string",
                    input_value="12345",
                    input_params={"pattern": "^\\d+$"},
                    should_succeed=True,
                    expected_value="12345",
                    description="Digits-only pattern",
                ),
                m.Tests.ValidationScenario(
                    name="pattern_alphanumeric",
                    validator_type="string",
                    input_value="abc123",
                    input_params={"pattern": "^[a-zA-Z0-9]+$"},
                    should_succeed=True,
                    expected_value="abc123",
                    description="Alphanumeric pattern",
                ),
                m.Tests.ValidationScenario(
                    name="pattern_mismatch",
                    validator_type="string",
                    input_value="invalid@",
                    input_params={"pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"},
                    should_succeed=False,
                    expected_error_contains="format is invalid",
                    description="Pattern mismatch",
                ),
            ]
            NON_NEGATIVE_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
                m.Tests.ValidationScenario(
                    name="non_negative_zero",
                    validator_type="numeric",
                    input_value=0,
                    should_succeed=True,
                    expected_value=0,
                    description="Zero is non-negative",
                ),
                m.Tests.ValidationScenario(
                    name="non_negative_positive",
                    validator_type="numeric",
                    input_value=42,
                    should_succeed=True,
                    expected_value=42,
                    description="Positive number",
                ),
                m.Tests.ValidationScenario(
                    name="non_negative_large",
                    validator_type="numeric",
                    input_value=1000000,
                    should_succeed=True,
                    expected_value=1000000,
                    description="Large positive number",
                ),
                m.Tests.ValidationScenario(
                    name="non_negative_negative",
                    validator_type="numeric",
                    input_value=-1,
                    should_succeed=False,
                    expected_error_contains="non-negative",
                    description="Negative rejection",
                ),
                m.Tests.ValidationScenario(
                    name="non_negative_none",
                    validator_type="numeric",
                    input_value=None,
                    should_succeed=False,
                    expected_error_contains="cannot be None",
                    description="None rejection",
                ),
            ]
            POSITIVE_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
                m.Tests.ValidationScenario(
                    name="positive_one",
                    validator_type="numeric",
                    input_value=1,
                    should_succeed=True,
                    expected_value=1,
                    description="Positive value 1",
                ),
                m.Tests.ValidationScenario(
                    name="positive_large",
                    validator_type="numeric",
                    input_value=999999,
                    should_succeed=True,
                    expected_value=999999,
                    description="Large positive",
                ),
                m.Tests.ValidationScenario(
                    name="positive_float",
                    validator_type="numeric",
                    input_value=0.1,
                    should_succeed=True,
                    expected_value=0.1,
                    description="Positive float",
                ),
                m.Tests.ValidationScenario(
                    name="positive_zero",
                    validator_type="numeric",
                    input_value=0,
                    should_succeed=False,
                    expected_error_contains="positive",
                    description="Zero rejection",
                ),
                m.Tests.ValidationScenario(
                    name="positive_negative",
                    validator_type="numeric",
                    input_value=-5,
                    should_succeed=False,
                    expected_error_contains="positive",
                    description="Negative rejection",
                ),
                m.Tests.ValidationScenario(
                    name="positive_none",
                    validator_type="numeric",
                    input_value=None,
                    should_succeed=False,
                    expected_error_contains="cannot be None",
                    description="None rejection",
                ),
            ]
            RANGE_SCENARIOS: ClassVar[Sequence[m.Tests.ValidationScenario]] = [
                m.Tests.ValidationScenario(
                    name="range_within_bounds",
                    validator_type="numeric",
                    input_value=5,
                    input_params={"min_value": 1, "max_value": 10},
                    should_succeed=True,
                    expected_value=5,
                    description="Value within range",
                ),
                m.Tests.ValidationScenario(
                    name="range_at_min",
                    validator_type="numeric",
                    input_value=1,
                    input_params={"min_value": 1, "max_value": 10},
                    should_succeed=True,
                    expected_value=1,
                    description="Value at minimum",
                ),
                m.Tests.ValidationScenario(
                    name="range_at_max",
                    validator_type="numeric",
                    input_value=10,
                    input_params={"min_value": 1, "max_value": 10},
                    should_succeed=True,
                    expected_value=10,
                    description="Value at maximum",
                ),
                m.Tests.ValidationScenario(
                    name="range_below_min",
                    validator_type="numeric",
                    input_value=0,
                    input_params={"min_value": 1, "max_value": 10},
                    should_succeed=False,
                    expected_error_contains="at least",
                    description="Value below minimum",
                ),
                m.Tests.ValidationScenario(
                    name="range_above_max",
                    validator_type="numeric",
                    input_value=11,
                    input_params={"min_value": 1, "max_value": 10},
                    should_succeed=False,
                    expected_error_contains="at most",
                    description="Value above maximum",
                ),
                m.Tests.ValidationScenario(
                    name="range_negative_range",
                    validator_type="numeric",
                    input_value=-5,
                    input_params={"min_value": -10, "max_value": -1},
                    should_succeed=True,
                    expected_value=-5,
                    description="Negative range",
                ),
                m.Tests.ValidationScenario(
                    name="range_fractional",
                    validator_type="numeric",
                    input_value=2.5,
                    input_params={"min_value": 0.5, "max_value": 5.5},
                    should_succeed=True,
                    expected_value=2.5,
                    description="Fractional range",
                ),
                m.Tests.ValidationScenario(
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

            PUBLIC_PARSE_CASES: ClassVar[Sequence[m.Tests.PublicParseCase]] = [
                m.Tests.PublicParseCase(
                    name="string-to-int",
                    input_value="42",
                    target=int,
                    should_succeed=True,
                    expected_value=42,
                    description="Public parse coerces numeric string into int",
                ),
                m.Tests.PublicParseCase(
                    name="int-to-str",
                    input_value=42,
                    target=str,
                    should_succeed=True,
                    expected_value="42",
                    description="Public parse coerces int into string",
                ),
                m.Tests.PublicParseCase(
                    name="string-to-float",
                    input_value="2.2",
                    target=float,
                    should_succeed=True,
                    expected_value=2.2,
                    description="Public parse coerces numeric string into float",
                ),
                m.Tests.PublicParseCase(
                    name="string-to-bool",
                    input_value="true",
                    target=bool,
                    should_succeed=True,
                    expected_value=True,
                    description="Public parse coerces truthy string into bool",
                ),
                m.Tests.PublicParseCase(
                    name="none-uses-default",
                    input_value=None,
                    target=int,
                    options=u.ParseOptions(default=7),
                    should_succeed=True,
                    expected_value=7,
                    description="Public parse returns default when value is None",
                ),
                m.Tests.PublicParseCase(
                    name="invalid-uses-default-factory",
                    input_value="x",
                    target=int,
                    options=u.ParseOptions(default_factory=lambda: 9),
                    should_succeed=True,
                    expected_value=9,
                    description="Public parse returns default_factory output on failure",
                ),
                m.Tests.PublicParseCase(
                    name="enum-exact",
                    input_value="inactive",
                    target=c.Tests.STATUS_ENUM,
                    should_succeed=True,
                    expected_value=c.Tests.STATUS_INACTIVE,
                    description="Public parse resolves StrEnum exact values",
                ),
                m.Tests.PublicParseCase(
                    name="enum-case-insensitive",
                    input_value="INACTIVE",
                    target=c.Tests.STATUS_ENUM,
                    options=u.ParseOptions(
                        case_insensitive=True,
                    ),
                    should_succeed=True,
                    expected_value=c.Tests.STATUS_INACTIVE,
                    description="Public parse resolves StrEnum values case-insensitively",
                ),
                m.Tests.PublicParseCase(
                    name="model-from-mapping",
                    input_value={"name": "parsed", "value": 3},
                    target=m.Tests.SampleModel,
                    should_succeed=True,
                    expected_data={"name": "parsed", "value": 3},
                    description="Public parse materializes canonical test model from mapping",
                ),
                m.Tests.PublicParseCase(
                    name="invalid-int-fails",
                    input_value="x",
                    target=int,
                    should_succeed=False,
                    description="Public parse fails for non-numeric int input",
                ),
                m.Tests.PublicParseCase(
                    name="invalid-enum-fails",
                    input_value="missing",
                    target=c.Tests.STATUS_ENUM,
                    should_succeed=False,
                    description="Public parse fails for unknown enum values",
                ),
                m.Tests.PublicParseCase(
                    name="invalid-model-shape-fails",
                    input_value={"bad": "shape"},
                    target=m.Tests.SampleModel,
                    should_succeed=False,
                    description="Public parse fails for invalid model payloads",
                ),
                m.Tests.PublicParseCase(
                    name="invalid-bool-field-context",
                    input_value="maybe",
                    target=bool,
                    options=u.ParseOptions(field_name="flag"),
                    should_succeed=False,
                    error_contains="flag",
                    description="Public parse includes field context on bool failure",
                ),
            ]

            LDIF_PARSE_SCENARIOS: ClassVar[Sequence[m.Tests.ParserScenario]] = [
                m.Tests.ParserScenario(
                    name="parse_simple_dn",
                    parser_method="parse",
                    input_data="dn: cn=test,dc=example,dc=com",
                    should_succeed=True,
                    description="Simple DN parsing",
                ),
                m.Tests.ParserScenario(
                    name="parse_with_attributes",
                    parser_method="parse",
                    input_data="dn: cn=test,dc=example,dc=com\nobjectClass: person\ncn: test",
                    should_succeed=True,
                    description="DN with attributes",
                ),
                m.Tests.ParserScenario(
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

            _RETRY_BASE_SETTINGS: ClassVar[m.ConfigMap] = m.ConfigMap(
                root={"max_retries": 3, "backoff_type": "constant", "backoff_ms": 10}
            )
            _RETRY_EXHAUSTED_SETTINGS: ClassVar[m.ConfigMap] = m.ConfigMap(
                root={"max_retries": 2, "backoff_type": "constant", "backoff_ms": 10}
            )

            RETRY_SCENARIOS: ClassVar[Sequence[m.Tests.ReliabilityScenario]] = [
                m.Tests.ReliabilityScenario(
                    name="retry_immediate_success",
                    strategy="retry",
                    settings=_RETRY_BASE_SETTINGS,
                    simulate_failures=0,
                    expected_state="success",
                    should_succeed=True,
                    description="Operation succeeds immediately",
                ),
                m.Tests.ReliabilityScenario(
                    name="retry_after_one_failure",
                    strategy="retry",
                    settings=_RETRY_BASE_SETTINGS,
                    simulate_failures=1,
                    expected_state="success",
                    should_succeed=True,
                    description="Succeeds after one retry",
                ),
                m.Tests.ReliabilityScenario(
                    name="retry_exhausted",
                    strategy="retry",
                    settings=_RETRY_EXHAUSTED_SETTINGS,
                    simulate_failures=5,
                    expected_state="exhausted",
                    should_succeed=False,
                    description="All retries exhausted",
                ),
            ]
            CIRCUIT_BREAKER_SCENARIOS: ClassVar[
                t.SequenceOf[m.Tests.ReliabilityScenario]
            ] = [
                m.Tests.ReliabilityScenario(
                    name="circuit_initial_closed",
                    strategy="circuit_breaker",
                    settings=m.ConfigMap(
                        root={"failure_threshold": 5, "timeout_ms": 1000}
                    ),
                    simulate_failures=0,
                    expected_state="closed",
                    should_succeed=True,
                    description="Circuit starts in closed state",
                ),
                m.Tests.ReliabilityScenario(
                    name="circuit_open_on_threshold",
                    strategy="circuit_breaker",
                    settings=m.ConfigMap(
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

            value_input: Annotated[
                str,
                u.Field(description="String input validated by business rules."),
            ]
            min_length: Annotated[
                int,
                u.Field(description="Minimum accepted input length."),
            ] = c.Tests.MIN_LENGTH_DEFAULT

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

            error_message: Annotated[
                str,
                u.Field(description="Failure message emitted by execute()."),
            ] = c.Tests.DEFAULT_ERROR_MESSAGE

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
            """Factory for `m.Tests.User` entities using native Python patterns."""

            _counter: ClassVar[count[int]] = count(1)
            _names: ClassVar[Sequence[str]] = [
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
            ) -> tm.Tests.User:
                """Build a `tm.Tests.User` instance with optional overrides."""
                n = next(cls._counter)
                actual_user_id = user_id if user_id is not None else f"user_{n:03d}"
                actual_name = name if name is not None else cls._next_name()
                actual_email = (
                    email if email is not None else f"{actual_user_id}@example.com"
                )
                return tm.Tests.User(
                    id=actual_user_id,
                    unique_id=actual_user_id,
                    name=actual_name,
                    email=actual_email,
                    active=is_active,
                )

            @classmethod
            def build_batch(cls, size: int) -> t.SequenceOf[tm.Tests.User]:
                """Build multiple `tm.Tests.User` instances with auto-generated values."""
                return [cls.build() for _ in range(size)]

            @classmethod
            def reset(cls) -> None:
                """Reset factory state for test isolation."""
                cls._counter = count(1)
                cls._name_index = 0

        class _GetUserFactoryBase[T]:
            """Shared counter-rotating state for GetUser-style factories.

            Subclasses provide their own typed ``build``;
            the base owns the per-subclass ``_counter``, ``build_batch``, and ``reset``.
            """

            _counter: ClassVar[count[int]] = count(1)

            @classmethod
            def _resolve_user_id(cls, user_id: str | None) -> str:
                """Return ``user_id`` or the next auto-generated identifier."""
                if user_id is not None:
                    return user_id
                return f"user_{next(cls._counter):03d}"

            @classmethod
            def build(cls, *, user_id: str | None = None) -> T:
                """Build a GetUser-style instance; subclasses override with typed return."""
                raise NotImplementedError

            @classmethod
            def build_batch(cls, size: int) -> list[T]:
                """Build multiple GetUser-style instances with auto-generated values."""
                return [cls.build() for _ in range(size)]

            @classmethod
            def reset(cls) -> None:
                """Reset per-subclass factory counter."""
                cls._counter = count(1)

        class GetUserServiceFactory(_GetUserFactoryBase[GetUserService]):
            """Factory for `GetUserService`."""

            @classmethod
            @override
            def build(
                cls, *, user_id: str | None = None
            ) -> TestsFlextUtilities.Tests.GetUserService:
                """Build a `GetUserService` instance."""
                return TestsFlextUtilities.Tests.GetUserService(
                    user_id=cls._resolve_user_id(user_id),
                )

        class _FailingFactoryBase[T]:
            """Shared constructor contract for failing-service factories."""

            @classmethod
            def build(
                cls,
                *,
                error_message: str = c.Tests.DEFAULT_ERROR_MESSAGE,
            ) -> T:
                """Build a failing-service instance; subclasses provide the type."""
                raise NotImplementedError

            @classmethod
            def build_batch(cls, size: int) -> list[T]:
                """Build multiple failing-service instances."""
                return [cls.build() for _ in range(size)]

        class FailingServiceFactory(_FailingFactoryBase[FailingService]):
            """Factory for FailingService."""

            @classmethod
            @override
            def build(
                cls,
                *,
                error_message: str = c.Tests.DEFAULT_ERROR_MESSAGE,
            ) -> TestsFlextUtilities.Tests.FailingService:
                """Build a FailingService instance."""
                return TestsFlextUtilities.Tests.FailingService(
                    error_message=error_message,
                )

        class GetUserServiceAutoFactory(_GetUserFactoryBase[GetUserServiceAuto]):
            """Factory for GetUserServiceAuto."""

            @classmethod
            @override
            def build(
                cls, *, user_id: str | None = None
            ) -> TestsFlextUtilities.Tests.GetUserServiceAuto:
                """Build a GetUserServiceAuto instance."""
                return TestsFlextUtilities.Tests.GetUserServiceAuto(
                    user_id=cls._resolve_user_id(user_id),
                )

        class _ValidatingFactoryBase[T]:
            """Shared word-rotating state for validating-service factories.

            Subclasses override ``_make_instance`` with the typed return;
            the base owns ``_words``, ``_word_index``, ``_next_word``,
            ``build``, ``build_batch``, and ``reset``.
            """

            _words: ClassVar[Sequence[str]] = (
                "alpha",
                "bravo",
                "charlie",
                "delta",
                "echo",
            )
            _word_index: ClassVar[int] = 0

            @classmethod
            def _next_word(cls) -> str:
                """Get next word from rotation."""
                word = cls._words[cls._word_index % len(cls._words)]
                cls._word_index += 1
                return word

            @classmethod
            def _make_instance(cls, value_input: str, min_length: int) -> T:
                """Construct one instance; subclasses override with typed return."""
                raise NotImplementedError

            @classmethod
            def build(
                cls,
                *,
                value_input: str | None = None,
                min_length: int = c.Tests.MIN_LENGTH_DEFAULT,
            ) -> T:
                """Build one validating-service instance."""
                actual_value = (
                    value_input if value_input is not None else cls._next_word()
                )
                return cls._make_instance(actual_value, min_length)

            @classmethod
            def build_batch(cls, size: int) -> list[T]:
                """Build multiple validating-service instances."""
                return [cls.build() for _ in range(size)]

            @classmethod
            def reset(cls) -> None:
                """Reset per-subclass factory counter."""
                cls._word_index = 0

        class ValidatingServiceAutoFactory(
            _ValidatingFactoryBase[ValidatingServiceAuto]
        ):
            """Factory for ValidatingServiceAuto."""

            @classmethod
            @override
            def _make_instance(
                cls, value_input: str, min_length: int
            ) -> TestsFlextUtilities.Tests.ValidatingServiceAuto:
                """Construct a ValidatingServiceAuto instance."""
                return TestsFlextUtilities.Tests.ValidatingServiceAuto(
                    value_input=value_input,
                    min_length=min_length,
                )

        class ValidatingServiceFactory(_ValidatingFactoryBase[ValidatingService]):
            """Factory for ``ValidatingService``."""

            @classmethod
            @override
            def _make_instance(
                cls, value_input: str, min_length: int
            ) -> TestsFlextUtilities.Tests.ValidatingService:
                """Construct a ValidatingService instance."""
                return TestsFlextUtilities.Tests.ValidatingService(
                    value_input=value_input,
                    min_length=min_length,
                )

        class FailingServiceAutoFactory(_FailingFactoryBase[FailingServiceAuto]):
            """Factory for FailingServiceAuto."""

            @classmethod
            @override
            def build(
                cls,
                *,
                error_message: str = c.Tests.DEFAULT_ERROR_MESSAGE,
            ) -> TestsFlextUtilities.Tests.FailingServiceAuto:
                """Build a FailingServiceAuto instance."""
                return TestsFlextUtilities.Tests.FailingServiceAuto(
                    error_message=error_message,
                )

        class ServiceTestCaseFactory:
            """Factory for m.Tests.ServiceTestCase."""

            _service_types: ClassVar[Sequence[c.Tests.ServiceType]] = [
                c.Tests.SERVICE_TEST_TYPE_GET_USER,
                c.Tests.SERVICE_TEST_TYPE_VALIDATE,
                c.Tests.SERVICE_TEST_TYPE_FAIL,
            ]
            _type_index: ClassVar[int] = 0
            _words: ClassVar[Sequence[str]] = [
                "test",
                "sample",
                "example",
                "demo",
                "data",
            ]
            _word_index: ClassVar[int] = 0

            @classmethod
            def _next_type(cls) -> c.Tests.ServiceType:
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
                service_type: c.Tests.ServiceType | None = None,
                input_value: str | None = None,
                expected_success: bool = True,
                expected_error: str | None = None,
                extra_param: int = c.Tests.MIN_LENGTH_DEFAULT,
                description: str | None = None,
            ) -> m.Tests.ServiceTestCase:
                """Build a m.Tests.ServiceTestCase instance."""
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
                return m.Tests.ServiceTestCase(
                    service_type=actual_type,
                    input_value=actual_input,
                    expected_success=expected_success,
                    expected_error=expected_error,
                    extra_param=extra_param,
                    description=actual_description,
                )

            @classmethod
            def build_batch(cls, size: int) -> t.SequenceOf[m.Tests.ServiceTestCase]:
                """Build multiple m.Tests.ServiceTestCase instances with auto-generated values."""
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
                case: m.Tests.ServiceTestCase,
            ) -> (
                TestsFlextUtilities.Tests.GetUserService
                | TestsFlextUtilities.Tests.ValidatingService
                | TestsFlextUtilities.Tests.FailingService
            ):
                """Create appropriate service based on case type using pattern matching."""
                service: (
                    TestsFlextUtilities.Tests.GetUserService
                    | TestsFlextUtilities.Tests.ValidatingService
                    | TestsFlextUtilities.Tests.FailingService
                )
                match case.service_type:
                    case c.Tests.SERVICE_TEST_TYPE_GET_USER:
                        service = TestsFlextUtilities.Tests.GetUserServiceFactory.build(
                            user_id=case.input_value
                        )
                    case c.Tests.SERVICE_TEST_TYPE_VALIDATE:
                        service = (
                            TestsFlextUtilities.Tests.ValidatingServiceFactory.build(
                                value_input=case.input_value,
                                min_length=case.extra_param,
                            )
                        )
                    case c.Tests.SERVICE_TEST_TYPE_FAIL:
                        service = TestsFlextUtilities.Tests.FailingServiceFactory.build(
                            error_message=case.input_value
                            or c.Tests.DEFAULT_ERROR_MESSAGE
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
            ) -> t.SequenceOf[m.Tests.ServiceTestCase]:
                """Generate successful user service test cases."""
                return [
                    m.Tests.ServiceTestCase(
                        service_type=c.Tests.SERVICE_TEST_TYPE_GET_USER,
                        input_value=str(i * 100 + 1),
                        description=f"Valid user ID {i}",
                    )
                    for i in range(1, num_cases + 1)
                ]

            @staticmethod
            def generate_validation_success_cases(
                num_cases: int = 2,
            ) -> t.SequenceOf[m.Tests.ServiceTestCase]:
                """Generate successful validation test cases."""
                return [
                    m.Tests.ServiceTestCase(
                        service_type=c.Tests.SERVICE_TEST_TYPE_VALIDATE,
                        input_value=f"value_{i}",
                        description=f"Valid input {i}",
                    )
                    for i in range(1, num_cases + 1)
                ] + [
                    m.Tests.ServiceTestCase(
                        service_type=c.Tests.SERVICE_TEST_TYPE_VALIDATE,
                        input_value="test",
                        extra_param=2,
                        description="Custom min length",
                    ),
                ]

            @staticmethod
            def generate_validation_failure_cases() -> t.SequenceOf[
                m.Tests.ServiceTestCase
            ]:
                """Generate validation failure test cases."""
                return [
                    m.Tests.ServiceTestCase(
                        service_type=c.Tests.SERVICE_TEST_TYPE_VALIDATE,
                        input_value="ab",
                        expected_success=False,
                        expected_error="must be at least 3 characters",
                        description="Too short input",
                    ),
                    m.Tests.ServiceTestCase(
                        service_type=c.Tests.SERVICE_TEST_TYPE_VALIDATE,
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
            def user_success() -> t.SequenceOf[m.Tests.ServiceTestCase]:
                """Generate cached-style success cases on demand."""
                return TestsFlextUtilities.Tests.TestDataGenerators.generate_user_success_cases()

            @staticmethod
            def validate_success() -> t.SequenceOf[m.Tests.ServiceTestCase]:
                """Generate cached-style validation success cases on demand."""
                return TestsFlextUtilities.Tests.TestDataGenerators.generate_validation_success_cases()

            @staticmethod
            def validate_failure() -> t.SequenceOf[m.Tests.ServiceTestCase]:
                """Generate cached-style validation failure cases on demand."""
                return TestsFlextUtilities.Tests.TestDataGenerators.generate_validation_failure_cases()

            @staticmethod
            def create_service(
                case: m.Tests.ServiceTestCase,
            ) -> (
                TestsFlextUtilities.Tests.GetUserService
                | TestsFlextUtilities.Tests.ValidatingService
                | TestsFlextUtilities.Tests.FailingService
            ):
                """Create appropriate service based on case type."""
                return TestsFlextUtilities.Tests.ServiceFactoryRegistry.create_service(
                    case
                )

        class GenericModelFactory:
            """Factories for generic reusable models (Value, Snapshot, Progress)."""

            @staticmethod
            def operation_progress(
                success: int = 0,
                failure: int = 0,
                skipped: int = 0,
            ) -> m.Tests.Operation:
                """Create OperationProgress."""
                return m.Tests.Operation(
                    success_count=success,
                    failure_count=failure,
                    skipped_count=skipped,
                    metadata={},
                )

            @staticmethod
            def conversion_progress() -> m.Tests.Conversion:
                """Create ConversionProgress."""
                return m.Tests.Conversion(
                    converted=[],
                    errors=[],
                    warnings=[],
                    skipped=[],
                    metadata={},
                )

        @staticmethod
        def reset_all_factories() -> None:
            """Reset all factory states for test isolation."""
            TestsFlextUtilities.Tests.UserFactory.reset()
            TestsFlextUtilities.Tests.GetUserServiceFactory.reset()
            TestsFlextUtilities.Tests.ValidatingServiceFactory.reset()
            TestsFlextUtilities.Tests.GetUserServiceAutoFactory.reset()
            TestsFlextUtilities.Tests.ValidatingServiceAutoFactory.reset()
            TestsFlextUtilities.Tests.ServiceTestCaseFactory.reset()

        class Contract:
            """Shared contract for text utility behavior."""

            SAFE_STRING_VALID_CASES: ClassVar[Sequence[tuple[str, str]]] = (
                c.Tests.CORE_SAFE_STRING_VALID_CASES
            )
            SAFE_STRING_INVALID_CASES: ClassVar[Sequence[tuple[str | None, str]]] = (
                c.Tests.CORE_SAFE_STRING_INVALID_CASES
            )
            FORMAT_APP_ID_CASES: ClassVar[Sequence[tuple[str, str]]] = (
                c.Tests.CORE_FORMAT_APP_ID_CASES
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
            TestsFlextUtilities.Tests.Contract.assert_safe_string_valid(
                raw,
                expected,
            )

        @staticmethod
        def assert_format_app_id(raw: str, expected: str) -> None:
            """Assert app id formatting behavior."""
            TestsFlextUtilities.Tests.Contract.assert_format_app_id(
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


u = TestsFlextUtilities

__all__: list[str] = ["TestsFlextUtilities", "u"]
