from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum, unique
from typing import Annotated, ClassVar, cast, override

import pytest
from flext_tests import t, u
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextRegistry, h, r
from tests import c, m


class TestFlextRegistry:
    @unique
    class RegistryOperationType(StrEnum):
        """Registry operation types."""

        REGISTER_HANDLER = "register_handler"
        REGISTER_HANDLERS = "register_handlers"
        REGISTER_BINDINGS = "register_bindings"
        REGISTER_FUNCTION_MAP = "register_function_map"
        RESOLVE_BINDING_KEY = "resolve_binding_key"
        RESOLVE_HANDLER_KEY = "resolve_handler_key"
        SUMMARY_MANAGEMENT = "summary_management"
        ERROR_HANDLING = "error_handling"

    class RegistryTestCase(BaseModel):
        """Test case for registry operations."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(description="Registry test case name")]
        operation: Annotated[StrEnum, Field(description="Registry operation type")]
        handler_count: Annotated[
            int, Field(default=1, description="Number of handlers to generate")
        ] = 1
        should_succeed: Annotated[
            bool, Field(default=True, description="Expected operation success")
        ] = True
        error_pattern: Annotated[
            str | None,
            Field(default=None, description="Expected error message pattern"),
        ] = None
        with_bindings: Annotated[
            bool, Field(default=False, description="Whether bindings are included")
        ] = False
        with_function_map: Annotated[
            bool, Field(default=False, description="Whether function map is included")
        ] = False
        with_summary: Annotated[
            bool, Field(default=False, description="Whether summary is included")
        ] = False
        duplicate_registration: Annotated[
            bool,
            Field(
                default=False,
                description="Whether registration is intentionally duplicated",
            ),
        ] = False

    class ConcreteTestHandler(h[t.NormalizedValue, t.NormalizedValue]):
        """Test handler for registry."""

        @override
        def handle(self, message: t.NormalizedValue) -> r[t.NormalizedValue]:
            return r[t.NormalizedValue].ok(f"processed_{message}")

        def __call__(self, message: t.NormalizedValue) -> r[t.NormalizedValue]:
            return self.handle(message)

    _HANDLER_REGISTRATION: ClassVar[list[TestFlextRegistry.RegistryTestCase]] = [
        RegistryTestCase(
            name="single_handler_success",
            operation=RegistryOperationType.REGISTER_HANDLER,
            handler_count=1,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="idempotent_registration",
            operation=RegistryOperationType.REGISTER_HANDLER,
            handler_count=1,
            should_succeed=True,
            error_pattern=None,
            with_bindings=False,
            with_function_map=False,
            with_summary=False,
            duplicate_registration=True,
        ),
        RegistryTestCase(
            name="none_handler_failure",
            operation=RegistryOperationType.REGISTER_HANDLER,
            handler_count=0,
            should_succeed=False,
            error_pattern="Handler must expose message_type",
        ),
    ]
    _BATCH_REGISTRATION: ClassVar[list[TestFlextRegistry.RegistryTestCase]] = [
        RegistryTestCase(
            name="multiple_handlers_success",
            operation=RegistryOperationType.REGISTER_HANDLERS,
            handler_count=2,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="empty_handlers_list",
            operation=RegistryOperationType.REGISTER_HANDLERS,
            handler_count=0,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="duplicate_handlers",
            operation=RegistryOperationType.REGISTER_HANDLERS,
            handler_count=2,
            should_succeed=True,
            error_pattern=None,
            with_bindings=False,
            with_function_map=False,
            with_summary=False,
            duplicate_registration=True,
        ),
    ]
    _BINDING_REGISTRATION: ClassVar[list[TestFlextRegistry.RegistryTestCase]] = [
        RegistryTestCase(
            name="single_binding_success",
            operation=RegistryOperationType.REGISTER_BINDINGS,
            handler_count=1,
            should_succeed=True,
            error_pattern=None,
            with_bindings=True,
        ),
        RegistryTestCase(
            name="empty_bindings_list",
            operation=RegistryOperationType.REGISTER_BINDINGS,
            handler_count=0,
            should_succeed=True,
            error_pattern=None,
            with_bindings=True,
        ),
        RegistryTestCase(
            name="duplicate_bindings",
            operation=RegistryOperationType.REGISTER_BINDINGS,
            handler_count=1,
            should_succeed=True,
            error_pattern=None,
            with_bindings=True,
            with_function_map=False,
            with_summary=False,
            duplicate_registration=True,
        ),
    ]
    _FUNCTION_MAP_SCENARIOS: ClassVar[list[TestFlextRegistry.RegistryTestCase]] = [
        RegistryTestCase(
            name="function_map_with_handler",
            operation=RegistryOperationType.REGISTER_FUNCTION_MAP,
            handler_count=1,
            should_succeed=True,
            error_pattern=None,
            with_bindings=False,
            with_function_map=True,
        ),
        RegistryTestCase(
            name="empty_function_map",
            operation=RegistryOperationType.REGISTER_FUNCTION_MAP,
            handler_count=0,
            should_succeed=True,
            error_pattern=None,
            with_bindings=False,
            with_function_map=True,
        ),
        RegistryTestCase(
            name="duplicate_function_map",
            operation=RegistryOperationType.REGISTER_FUNCTION_MAP,
            handler_count=1,
            should_succeed=True,
            error_pattern=None,
            with_bindings=False,
            with_function_map=True,
            with_summary=False,
            duplicate_registration=True,
        ),
    ]
    _SUMMARY_SCENARIOS: ClassVar[list[TestFlextRegistry.RegistryTestCase]] = [
        RegistryTestCase(
            name="empty_summary",
            operation=RegistryOperationType.SUMMARY_MANAGEMENT,
            handler_count=0,
            should_succeed=True,
            error_pattern=None,
            with_bindings=False,
            with_function_map=False,
            with_summary=True,
        ),
        RegistryTestCase(
            name="summary_with_registrations",
            operation=RegistryOperationType.SUMMARY_MANAGEMENT,
            handler_count=2,
            should_succeed=True,
            error_pattern=None,
            with_bindings=False,
            with_function_map=False,
            with_summary=True,
        ),
        RegistryTestCase(
            name="summary_with_errors",
            operation=RegistryOperationType.SUMMARY_MANAGEMENT,
            handler_count=1,
            should_succeed=False,
            error_pattern=None,
            with_bindings=False,
            with_function_map=False,
            with_summary=True,
        ),
    ]
    _ERROR_SCENARIOS: ClassVar[list[TestFlextRegistry.RegistryTestCase]] = [
        RegistryTestCase(
            name="register_none_handler",
            operation=RegistryOperationType.ERROR_HANDLING,
            handler_count=0,
            should_succeed=False,
            error_pattern="Handler must expose message_type",
        ),
        RegistryTestCase(
            name="dispatcher_integration",
            operation=RegistryOperationType.ERROR_HANDLING,
            handler_count=1,
            should_succeed=True,
        ),
    ]

    @staticmethod
    def _create_handlers(count: int) -> list[t.HandlerLike]:
        return [TestFlextRegistry.ConcreteTestHandler() for _ in range(count)]

    @staticmethod
    def _create_bindings(
        handlers: Sequence[t.HandlerLike],
    ) -> list[tuple[type, t.HandlerLike]]:
        return [(str, handler) for handler in handlers]

    @staticmethod
    def _create_function_map(
        handlers: Sequence[t.HandlerLike],
    ) -> dict[type, t.HandlerLike]:
        result: dict[type, t.HandlerLike] = {}
        for idx, handler in enumerate(handlers):
            result[str if idx == 0 else int] = handler
        return result

    @pytest.mark.parametrize("test_case", _HANDLER_REGISTRATION, ids=lambda c: c.name)
    def test_handler_registration(
        self,
        test_case: TestFlextRegistry.RegistryTestCase,
    ) -> None:
        registry = u.Tests.RegistryHelpers.create_test_registry()
        if test_case.handler_count == 0:
            result = registry.register_handler(
                cast("t.HandlerLike", cast("object", None)),
            )
        else:
            handler = self.ConcreteTestHandler()
            result = registry.register_handler(handler)
            if test_case.duplicate_registration:
                result = registry.register_handler(handler)
        if test_case.should_succeed:
            _ = u.Tests.Result.assert_success(result)
        else:
            _ = u.Tests.Result.assert_failure(result)
            if test_case.error_pattern:
                u.Tests.Result.assert_failure_with_error(
                    result, test_case.error_pattern
                )

    @pytest.mark.parametrize("test_case", _BATCH_REGISTRATION, ids=lambda c: c.name)
    def test_batch_registration(
        self,
        test_case: TestFlextRegistry.RegistryTestCase,
    ) -> None:
        registry = u.Tests.RegistryHelpers.create_test_registry()
        handlers = self._create_handlers(test_case.handler_count)
        if test_case.duplicate_registration and handlers:
            registry.register_handlers(handlers)
            result = registry.register_handlers(handlers)
        else:
            result = registry.register_handlers(handlers)
        _ = (
            u.Tests.Result.assert_success(result)
            if test_case.should_succeed
            else u.Tests.Result.assert_failure(result)
        )
        assert isinstance(result.value, FlextRegistry.Summary)

    @pytest.mark.parametrize("test_case", _BINDING_REGISTRATION, ids=lambda c: c.name)
    def test_binding_registration(
        self,
        test_case: TestFlextRegistry.RegistryTestCase,
    ) -> None:
        registry = u.Tests.RegistryHelpers.create_test_registry()
        handlers = self._create_handlers(test_case.handler_count)
        if test_case.duplicate_registration and handlers:
            registry.register_handlers(handlers)
            result = registry.register_handlers(handlers)
        else:
            result = registry.register_handlers(handlers)
        _ = (
            u.Tests.Result.assert_success(result)
            if test_case.should_succeed
            else u.Tests.Result.assert_failure(result)
        )
        assert result.value is not None

    @pytest.mark.parametrize("test_case", _FUNCTION_MAP_SCENARIOS, ids=lambda c: c.name)
    def test_function_map_registration(
        self,
        test_case: TestFlextRegistry.RegistryTestCase,
    ) -> None:
        registry = u.Tests.RegistryHelpers.create_test_registry()
        handlers = self._create_handlers(test_case.handler_count)
        if test_case.duplicate_registration and handlers:
            registry.register_handlers(handlers)
            result = registry.register_handlers(handlers)
        else:
            result = registry.register_handlers(handlers)
        _ = (
            u.Tests.Result.assert_success(result)
            if test_case.should_succeed
            else u.Tests.Result.assert_failure(result)
        )
        assert result.value is not None

    @pytest.mark.parametrize("test_case", _SUMMARY_SCENARIOS, ids=lambda c: c.name)
    def test_summary_management(
        self,
        test_case: TestFlextRegistry.RegistryTestCase,
    ) -> None:
        summary = FlextRegistry.Summary(registered=[], skipped=[], errors=[])
        if test_case.handler_count > 0:
            for i in range(test_case.handler_count):
                summary.registered.append(
                    m.RegistrationDetails(
                        registration_id=f"test_{i}",
                        handler_mode=c.Cqrs.HandlerType.COMMAND,
                        timestamp="2025-01-01T00:00:00Z",
                        status=c.Cqrs.CommonStatus.RUNNING,
                    ),
                )
        if not test_case.should_succeed:
            summary.errors.append("test_error")
        assert len(summary.registered) == test_case.handler_count
        assert (len(summary.errors) > 0) == (not test_case.should_succeed)
        assert summary.is_failure == (not test_case.should_succeed)

    @pytest.mark.parametrize("test_case", _ERROR_SCENARIOS, ids=lambda c: c.name)
    def test_error_handling(
        self,
        test_case: TestFlextRegistry.RegistryTestCase,
    ) -> None:
        registry = u.Tests.RegistryHelpers.create_test_registry()
        if test_case.handler_count == 0:
            result = registry.register_handler(
                cast("t.HandlerLike", cast("object", None)),
            )
            _ = u.Tests.Result.assert_failure(result)
            u.Tests.Result.assert_failure_with_error(
                result,
                "Handler must expose message_type",
            )
        else:
            handler = self.ConcreteTestHandler()
            result = registry.register_handler(handler)
            _ = u.Tests.Result.assert_success(result)
            assert isinstance(result.value, m.RegistrationDetails)

    def test_registry_initialization(self) -> None:
        registry = u.Tests.RegistryHelpers.create_test_registry()
        assert registry is not None
        assert isinstance(registry, FlextRegistry)

    def test_registry_with_dispatcher(self) -> None:
        registry = u.Tests.RegistryHelpers.create_test_registry()
        handler = self.ConcreteTestHandler()
        result = registry.register_handler(handler)
        _ = u.Tests.Result.assert_success(result)
        assert isinstance(result.value, m.RegistrationDetails)

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("command", c.Cqrs.HandlerType.COMMAND),
            ("query", c.Cqrs.HandlerType.QUERY),
            ("invalid", c.Cqrs.HandlerType.COMMAND),
            (None, c.Cqrs.HandlerType.COMMAND),
        ],
        ids=["command", "query", "invalid", "none"],
    )
    def test_safe_handler_mode_extraction(
        self,
        mode: str | None,
        expected: str,
    ) -> None:
        registry = u.Tests.RegistryHelpers.create_test_registry()
        assert registry._get_handler_mode(mode or "") == expected

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("active", c.Cqrs.CommonStatus.ACTIVE),
            ("inactive", c.Cqrs.CommonStatus.INACTIVE),
            ("invalid", c.Cqrs.CommonStatus.ACTIVE),
            ("", c.Cqrs.CommonStatus.ACTIVE),
        ],
        ids=["active", "inactive", "invalid", "empty"],
    )
    def test_safe_status_extraction(
        self,
        status: str,
        expected: c.Cqrs.CommonStatus,
    ) -> None:
        registry = u.Tests.RegistryHelpers.create_test_registry()
        assert registry._get_status(status) == expected
