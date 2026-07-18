"""Shared fixtures for split decorator unit tests."""

from __future__ import annotations

import io
import time
from contextlib import redirect_stdout
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Annotated, ClassVar

from tests import m, t
from tests import u

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def capture_stdout[T](emit: Callable[[], T], *, contains: str) -> T:
    stream = io.StringIO()
    with redirect_stdout(stream):
        result = emit()
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline and contains not in stream.getvalue():
            time.sleep(0.01)
    assert contains in stream.getvalue()
    return result


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

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)
        name: Annotated[str, m.Field(description="Decorator test case name")]
        operation: Annotated[str, m.Field(description="Decorator operation under test")]

    class TestService:
        """Service for testing."""

        def get_value(self) -> str:
            return "test_value"

    class ServiceWithLogger:
        """Service with logger for testing."""

        def __init__(self) -> None:
            self.logger = u.fetch_logger(__name__)
            self.attempts = 0

        def flaky_method(self) -> str:
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
            name="railway_success", operation=DecoratorOperationType.RAILWAY_SUCCESS
        ),
        DecoratorTestCase(
            name="railway_exception", operation=DecoratorOperationType.RAILWAY_EXCEPTION
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
            name="retry_exhausted", operation=DecoratorOperationType.RETRY_EXHAUSTED
        ),
    ]
    TIMEOUT_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="timeout_success", operation=DecoratorOperationType.TIMEOUT_SUCCESS
        ),
        DecoratorTestCase(
            name="timeout_exceeded", operation=DecoratorOperationType.TIMEOUT_EXCEEDED
        ),
    ]
    COMBINED_SCENARIOS: ClassVar[Sequence[DecoratorTestCase]] = [
        DecoratorTestCase(
            name="combined_basic", operation=DecoratorOperationType.COMBINED_BASIC
        ),
        DecoratorTestCase(
            name="combined_with_railway",
            operation=DecoratorOperationType.COMBINED_WITH_RAILWAY,
        ),
    ]
