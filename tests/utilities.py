"""Utilities for flext-core tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from enum import StrEnum, unique
from pathlib import Path
from typing import Annotated, ClassVar, override

from flext_tests import FlextTestsUtilities, h, r

from tests._utilities.case_factories import TestsFlextUtilitiesCaseFactoriesMixin
from tests._utilities.contracts import TestsFlextUtilitiesContractsMixin
from tests._utilities.dispatch import TestsFlextUtilitiesDispatchMixin
from tests._utilities.parser_reliability import (
    TestsFlextUtilitiesParserReliabilityMixin,
)
from tests._utilities.railway import TestsFlextUtilitiesRailwayMixin
from tests._utilities.service_factories import TestsFlextUtilitiesServiceFactoriesMixin
from tests._utilities.services import TestsFlextUtilitiesServicesMixin
from tests.constants import c
from tests.models import m
from tests.protocols import p
from tests.typings import t


class TestsFlextUtilities(FlextTestsUtilities):
    """Utilities for flext-core tests."""

    class TestsFlextResultExceptionCarrying:
        class BrokenSized:
            """Sized t.JsonValue that raises on __len__."""

            def __len__(self) -> int:
                """Raise TypeError on length call.

                Raises:
                    TypeError: If no length.

                """
                msg = "no length"
                raise TypeError(msg)

        class UserModel(m.Value):
            """User model for testing."""

            name: Annotated[str, m.Field(description="User name")]
            age: Annotated[int, m.Field(description="User age")]

    class TestsFlextFlextHandlers:
        class ConcreteTestHandler(h[t.JsonPayload, t.JsonPayload]):
            """Test handler for string messages."""

            def __init__(self, *, settings: m.Handler | None = None) -> None:
                super().__init__(settings=settings)

            @override
            def dispatch_message(
                self,
                message: t.JsonPayload,
                operation: str = c.DEFAULT_HANDLER_MODE,
            ) -> p.Result[t.JsonPayload]:
                handler_mode = getattr(
                    self._config_model.handler_mode,
                    "value",
                    self._config_model.handler_mode,
                )
                valid_operations = {
                    c.DEFAULT_HANDLER_MODE,
                    c.HandlerMode.QUERY,
                    c.HandlerType.EVENT.value,
                }
                if operation != handler_mode and operation in valid_operations:
                    error_msg = c.ERR_HANDLER_INCOMPATIBLE_PIPELINE_MODE.format(
                        handler_mode=handler_mode,
                        operation=operation,
                    )
                    return r[t.JsonPayload].fail_op(
                        "validate handler pipeline mode",
                        error_msg,
                    )
                message_type = message.__class__
                if not self.can_handle(message_type):
                    error_msg = c.ERR_HANDLER_CANNOT_HANDLE_MESSAGE_TYPE.format(
                        type_name=message_type.__name__,
                    )
                    return r[t.JsonPayload].fail_op(
                        "validate handler message type",
                        error_msg,
                    )
                validation = self.validate_message(message)
                if validation.failure:
                    error_detail = validation.error or c.ERR_VALIDATION_FAILED
                    error_msg = c.ERR_HANDLER_MESSAGE_VALIDATION_FAILED.format(
                        error=error_detail,
                    )
                    return r[t.JsonPayload].fail_op(
                        "validate handler message",
                        error_msg,
                    )
                try:
                    return self.handle(message)
                except c.EXC_BROAD_RUNTIME as exc:
                    return r[t.JsonPayload].fail_op(
                        "run handler pipeline",
                        c.ERR_HANDLER_CRITICAL_FAILURE.format(error=str(exc)),
                    )

            @override
            def execute(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
                validation = self.validate_message(message)
                if validation.failure:
                    return r[t.JsonPayload].fail_op(
                        "execute handler validation",
                        validation.error or c.ERR_VALIDATION_FAILED,
                    )
                return self.handle(message)

            @override
            def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
                if not isinstance(message, str):
                    return r[t.JsonPayload].fail(c.Tests.UNEXPECTED_MESSAGE_TYPE)
                return r[t.JsonPayload].ok(f"processed_{message}")

            @override
            def validate_message(self, data: t.JsonPayload) -> p.Result[bool]:
                if data is None:
                    return r[bool].fail_op(
                        "validate handler message",
                        c.ERR_MESSAGE_CANNOT_BE_NONE,
                    )
                return r[bool].ok(value=True)

        class ValidationTestHandler(h[t.JsonPayload, t.JsonPayload]):
            """Test handler for validation."""

            def __init__(self, *, settings: m.Handler | None = None) -> None:
                super().__init__(settings=settings)

            @override
            def validate_message(self, data: t.JsonPayload) -> p.Result[bool]:
                return (
                    r[bool].ok(value=True)
                    if data
                    else r[bool].fail(c.Tests.VALIDATION_FAILED_FOR_TEST)
                )

            @override
            def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
                return r[t.JsonPayload].ok(f"processed_{message}")

        class FailingTestHandler(h[t.JsonPayload, t.JsonPayload]):
            """Test handler that fails."""

            def __init__(self, *, settings: m.Handler | None = None) -> None:
                super().__init__(settings=settings)

            @override
            def handle(self, message: t.JsonPayload) -> p.Result[t.JsonPayload]:
                if not isinstance(message, str):
                    return r[t.JsonPayload].fail(c.Tests.UNEXPECTED_MESSAGE_TYPE)
                return r[t.JsonPayload].fail(f"Handler failed for: {message}")

        class HandlerTypeScenario(m.Value):
            """Scenario for handler types."""

            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
            name: Annotated[str, m.Field(description="Handler type scenario name")]
            handler_type: Annotated[c.HandlerType, m.Field(description="Type")]
            handler_mode: Annotated[c.HandlerType, m.Field(description="Mode")]

        HANDLER_TYPES: ClassVar[Sequence[HandlerTypeScenario]] = [
            HandlerTypeScenario(
                name="command",
                handler_type=c.HandlerType.COMMAND,
                handler_mode=c.HandlerType.COMMAND,
            ),
            HandlerTypeScenario(
                name="query",
                handler_type=c.HandlerType.QUERY,
                handler_mode=c.HandlerType.QUERY,
            ),
            HandlerTypeScenario(
                name="event",
                handler_type=c.HandlerType.EVENT,
                handler_mode=c.HandlerType.EVENT,
            ),
            HandlerTypeScenario(
                name="saga",
                handler_type=c.HandlerType.SAGA,
                handler_mode=c.HandlerType.SAGA,
            ),
        ]

        VALIDATION_TYPES: ClassVar[Sequence[t.Pair[str, t.JsonPayload]]] = [
            ("str", "test_message"),
            ("int", 42),
            ("float", math.pi),
            ("bool", True),
            ("dict", {"key": "value", "number": 42}),
        ]

    class TestsFlextDecoratorsLegacy:
        @unique
        class DecoratorOperationType(StrEnum):
            """Decorator operation types."""

            INJECT_BASIC = "inject_basic"
            INJECT_MISSING = "inject_missing"
            INJECT_PROVIDED = "inject_provided"
            LOG_OPERATION_BASIC = "log_operation_basic"
            LOG_OPERATION_EXCEPTION = "log_operation_exception"
            TRACK_PERFORMANCE_BASIC = "track_performance_basic"
            TRACK_PERFORMANCE_EXCEPTION = "track_performance_exception"
            RAILWAY_SUCCESS = "railway_success"
            RAILWAY_EXCEPTION = "railway_exception"
            RETRY_SUCCESS_FIRST = "retry_success_first"
            RETRY_SUCCESS_AFTER_FAILURES = "retry_success_after_failures"
            RETRY_EXHAUSTED = "retry_exhausted"
            TIMEOUT_SUCCESS = "timeout_success"
            TIMEOUT_EXCEEDED = "timeout_exceeded"
            COMBINED_BASIC = "combined_basic"
            COMBINED_WITH_RAILWAY = "combined_with_railway"

        class DecoratorTestCase(m.BaseModel):
            """Test case for decorator."""

            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
            name: Annotated[str, m.Field(description="Decorator test case name")]
            operation: Annotated[
                str,
                m.Field(description="Decorator operation under test"),
            ]

        class TestService:
            """Service for testing."""

            @staticmethod
            def get_value() -> str:
                """Provide ``get_value``.

                Returns:
                    The resulting ``str``.

                """
                return "test_value"

        class ServiceWithLogger:
            """Service with logger for testing."""

            def __init__(self) -> None:
                self.logger = u.fetch_logger(__name__)
                self.attempts = 0

            def flaky_method(self) -> str:
                """Provide ``flaky_method``.

                Returns:
                    The resulting ``str``.

                Raises:
                    RuntimeError: If First attempt fails.

                """
                self.attempts += 1
                if self.attempts == 1:
                    error_msg = "First attempt fails"
                    raise RuntimeError(error_msg)
                return "success"

        INJECT_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="inject_basic_dependency",
                operation=DecoratorOperationType.INJECT_BASIC,
            ),
            DecoratorTestCase(
                name="inject_missing_dependency",
                operation=DecoratorOperationType.INJECT_MISSING,
            ),
            DecoratorTestCase(
                name="inject_with_provided_kwarg",
                operation=DecoratorOperationType.INJECT_PROVIDED,
            ),
        ]
        LOG_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="log_operation_basic",
                operation=DecoratorOperationType.LOG_OPERATION_BASIC,
            ),
            DecoratorTestCase(
                name="log_operation_exception",
                operation=DecoratorOperationType.LOG_OPERATION_EXCEPTION,
            ),
        ]
        TRACK_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="track_performance_basic",
                operation=DecoratorOperationType.TRACK_PERFORMANCE_BASIC,
            ),
            DecoratorTestCase(
                name="track_performance_exception",
                operation=DecoratorOperationType.TRACK_PERFORMANCE_EXCEPTION,
            ),
        ]
        RAILWAY_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="railway_success",
                operation=DecoratorOperationType.RAILWAY_SUCCESS,
            ),
            DecoratorTestCase(
                name="railway_exception",
                operation=DecoratorOperationType.RAILWAY_EXCEPTION,
            ),
        ]
        RETRY_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="retry_success_first_attempt",
                operation=DecoratorOperationType.RETRY_SUCCESS_FIRST,
            ),
            DecoratorTestCase(
                name="retry_success_after_failures",
                operation=DecoratorOperationType.RETRY_SUCCESS_AFTER_FAILURES,
            ),
            DecoratorTestCase(
                name="retry_exhausted",
                operation=DecoratorOperationType.RETRY_EXHAUSTED,
            ),
        ]
        TIMEOUT_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="timeout_success",
                operation=DecoratorOperationType.TIMEOUT_SUCCESS,
            ),
            DecoratorTestCase(
                name="timeout_exceeded",
                operation=DecoratorOperationType.TIMEOUT_EXCEEDED,
            ),
        ]
        COMBINED_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
            DecoratorTestCase(
                name="combined_basic",
                operation=DecoratorOperationType.COMBINED_BASIC,
            ),
            DecoratorTestCase(
                name="combined_with_railway",
                operation=DecoratorOperationType.COMBINED_WITH_RAILWAY,
            ),
        ]

    class TestsFlextBeartypeEngine:
        """Shared beartype engine test support."""

        FORBIDDEN: frozenset[str] = frozenset({"dict", "list", "set"})

        @staticmethod
        def _run_python(script: str, cwd: Path) -> p.Cli.CommandOutput:
            """Run a Python snippet in a subprocess and capture text output.

            Returns:
                The resulting ``p.Cli.CommandOutput``.

            """
            result = u.Cli.run_raw([sys.executable, "-c", script], cwd=cwd)
            if result.success:
                output: p.Cli.CommandOutput = result.value
                return output
            return m.Cli.CommandOutput(
                stdout="",
                stderr=result.error or "python snippet execution failed",
                outcome=m.Cli.ProcessOutcome(
                    raw_return_code=1,
                    timed_out=False,
                    forwarded_signal=None,
                ),
            )

    class Tests(
        TestsFlextUtilitiesCaseFactoriesMixin,
        TestsFlextUtilitiesContractsMixin,
        TestsFlextUtilitiesParserReliabilityMixin,
        TestsFlextUtilitiesServiceFactoriesMixin,
        TestsFlextUtilitiesServicesMixin,
        TestsFlextUtilitiesRailwayMixin,
        TestsFlextUtilitiesDispatchMixin,
        FlextTestsUtilities.Tests,
    ):
        """flext-core test utilities namespace."""


u = TestsFlextUtilities

__all__: list[str] = ["TestsFlextUtilities"]
