"""Refactored comprehensive tests for FlextRegistry - Service Registry.

Module: flext_core.registry
Scope: FlextRegistry - handler registration, bindings, function maps, key resolution

Tests FlextRegistry functionality including:
- Handler registration (single and batch)
- Binding registration
- Function map registration
- Key resolution
- Summary management
- Error handling
- Dispatcher integration

Uses Python 3.13 patterns, FlextTestsUtilities, FlextConstants,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

import pytest

from flext_core import (
    FlextConstants,
    FlextDispatcher,
    FlextHandlers,
    FlextModels,
    FlextRegistry,
    FlextResult,
)


class RegistryOperationType(StrEnum):
    """Registry operation types for test parametrization."""

    REGISTER_HANDLER = "register_handler"
    REGISTER_HANDLERS = "register_handlers"
    REGISTER_BINDINGS = "register_bindings"
    REGISTER_FUNCTION_MAP = "register_function_map"
    RESOLVE_BINDING_KEY = "resolve_binding_key"
    RESOLVE_HANDLER_KEY = "resolve_handler_key"
    SUMMARY_MANAGEMENT = "summary_management"
    ERROR_HANDLING = "error_handling"


@dataclass(frozen=True, slots=True)
class RegistryTestCase:
    """Registry test case definition with parametrization data."""

    name: str
    operation: RegistryOperationType
    handler_count: int = 1
    should_succeed: bool = True
    error_pattern: str | None = None
    with_bindings: bool = False
    with_function_map: bool = False
    with_summary: bool = False
    duplicate_registration: bool = False


class ConcreteTestHandler(FlextHandlers[object, object]):
    """Concrete implementation of FlextHandlers for testing."""

    def handle(self, message: object) -> FlextResult[object]:
        """Handle the message."""
        return FlextResult[object].ok(f"processed_{message}")


class RegistryScenarios:
    """Centralized registry test scenarios using FlextConstants."""

    HANDLER_REGISTRATION: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            "single_handler_success", RegistryOperationType.REGISTER_HANDLER, 1, True,
        ),
        RegistryTestCase(
            "idempotent_registration",
            RegistryOperationType.REGISTER_HANDLER,
            1,
            True,
            None,
            False,
            False,
            False,
            True,
        ),
        RegistryTestCase(
            "none_handler_failure",
            RegistryOperationType.REGISTER_HANDLER,
            0,
            False,
            "Handler cannot be None",
        ),
    ]

    BATCH_REGISTRATION: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            "multiple_handlers_success",
            RegistryOperationType.REGISTER_HANDLERS,
            2,
            True,
        ),
        RegistryTestCase(
            "empty_handlers_list", RegistryOperationType.REGISTER_HANDLERS, 0, True,
        ),
        RegistryTestCase(
            "duplicate_handlers",
            RegistryOperationType.REGISTER_HANDLERS,
            2,
            True,
            None,
            False,
            False,
            False,
            True,
        ),
    ]

    BINDING_REGISTRATION: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            "single_binding_success",
            RegistryOperationType.REGISTER_BINDINGS,
            1,
            True,
            None,
            True,
        ),
        RegistryTestCase(
            "empty_bindings_list",
            RegistryOperationType.REGISTER_BINDINGS,
            0,
            True,
            None,
            True,
        ),
        RegistryTestCase(
            "duplicate_bindings",
            RegistryOperationType.REGISTER_BINDINGS,
            1,
            True,
            None,
            True,
            False,
            False,
            True,
        ),
    ]

    FUNCTION_MAP_SCENARIOS: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            "function_map_with_handler",
            RegistryOperationType.REGISTER_FUNCTION_MAP,
            1,
            True,
            None,
            False,
            True,
        ),
        RegistryTestCase(
            "empty_function_map",
            RegistryOperationType.REGISTER_FUNCTION_MAP,
            0,
            True,
            None,
            False,
            True,
        ),
        RegistryTestCase(
            "duplicate_function_map",
            RegistryOperationType.REGISTER_FUNCTION_MAP,
            1,
            True,
            None,
            False,
            True,
            False,
            True,
        ),
    ]

    SUMMARY_SCENARIOS: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            "empty_summary",
            RegistryOperationType.SUMMARY_MANAGEMENT,
            0,
            True,
            None,
            False,
            False,
            True,
        ),
        RegistryTestCase(
            "summary_with_registrations",
            RegistryOperationType.SUMMARY_MANAGEMENT,
            2,
            True,
            None,
            False,
            False,
            True,
        ),
        RegistryTestCase(
            "summary_with_errors",
            RegistryOperationType.SUMMARY_MANAGEMENT,
            1,
            False,
            None,
            False,
            False,
            True,
        ),
    ]

    KEY_RESOLUTION: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            "resolve_handler_key_string_type",
            RegistryOperationType.RESOLVE_HANDLER_KEY,
            1,
            True,
        ),
        RegistryTestCase(
            "resolve_handler_key_class_type",
            RegistryOperationType.RESOLVE_HANDLER_KEY,
            1,
            True,
        ),
        RegistryTestCase(
            "resolve_binding_key", RegistryOperationType.RESOLVE_BINDING_KEY, 1, True,
        ),
    ]

    ERROR_SCENARIOS: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            "register_none_handler",
            RegistryOperationType.ERROR_HANDLING,
            0,
            False,
            "Handler cannot be None",
        ),
        RegistryTestCase(
            "dispatcher_integration", RegistryOperationType.ERROR_HANDLING, 1, True,
        ),
    ]

    @staticmethod
    def create_handlers(count: int) -> list[FlextHandlers[object, object]]:
        """Create test handlers."""
        return [ConcreteTestHandler() for _ in range(count)]

    @staticmethod
    def create_bindings(
        handlers: list[FlextHandlers[object, object]],
    ) -> list[tuple[type[object], FlextHandlers[object, object]]]:
        """Create test bindings using str message type."""
        return [(str, handler) for handler in handlers]

    @staticmethod
    def create_function_map(
        handlers: list[FlextHandlers[object, object]],
    ) -> dict[type[object], FlextHandlers[object, object]]:
        """Create test function map using str message type."""
        result: dict[type[object], FlextHandlers[object, object]] = {}
        for idx, handler in enumerate(handlers):
            result[str if idx == 0 else int] = handler
        return result


class RegistryTestHelpers:
    """Generalized helpers for registry testing."""

    @staticmethod
    def create_test_registry() -> FlextRegistry:
        """Create test registry with dispatcher."""
        dispatcher = FlextDispatcher()
        return FlextRegistry(dispatcher=dispatcher)

    @staticmethod
    def assert_registration_result(
        result: FlextResult[object],
        should_succeed: bool,
        error_pattern: str | None = None,
    ) -> None:
        """Assert registration result matches expectations."""
        assert result.is_success == should_succeed
        if error_pattern:
            assert result.error is not None
            assert error_pattern in result.error


class TestFlextRegistry:
    """Refactored registry test suite using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "test_case", RegistryScenarios.HANDLER_REGISTRATION, ids=lambda tc: tc.name,
    )
    def test_handler_registration(self, test_case: RegistryTestCase) -> None:
        """Test handler registration with various scenarios."""
        registry = RegistryTestHelpers.create_test_registry()
        if test_case.handler_count == 0:
            result = registry.register_handler(None)
        else:
            handler = ConcreteTestHandler()
            result = registry.register_handler(handler)
            if test_case.duplicate_registration:
                result = registry.register_handler(handler)
        assert result.is_success == test_case.should_succeed
        if test_case.error_pattern:
            assert result.error is not None
            assert test_case.error_pattern in result.error

    @pytest.mark.parametrize(
        "test_case", RegistryScenarios.BATCH_REGISTRATION, ids=lambda tc: tc.name,
    )
    def test_batch_registration(self, test_case: RegistryTestCase) -> None:
        """Test batch handler registration."""
        registry = RegistryTestHelpers.create_test_registry()
        handlers = RegistryScenarios.create_handlers(test_case.handler_count)
        if test_case.duplicate_registration and handlers:
            registry.register_handlers(handlers)
            result = registry.register_handlers(handlers)
        else:
            result = registry.register_handlers(handlers)
        assert result.is_success == test_case.should_succeed
        assert isinstance(result.value, FlextRegistry.Summary)

    @pytest.mark.parametrize(
        "test_case", RegistryScenarios.BINDING_REGISTRATION, ids=lambda tc: tc.name,
    )
    def test_binding_registration(self, test_case: RegistryTestCase) -> None:
        """Test binding registration."""
        registry = RegistryTestHelpers.create_test_registry()
        handlers = RegistryScenarios.create_handlers(test_case.handler_count)
        bindings = RegistryScenarios.create_bindings(handlers)
        if test_case.duplicate_registration and bindings:
            registry.register_bindings(bindings)
            result = registry.register_bindings(bindings)
        else:
            result = registry.register_bindings(bindings)
        assert result.is_success == test_case.should_succeed
        assert isinstance(result.value, FlextRegistry.Summary)

    @pytest.mark.parametrize(
        "test_case", RegistryScenarios.FUNCTION_MAP_SCENARIOS, ids=lambda tc: tc.name,
    )
    def test_function_map_registration(self, test_case: RegistryTestCase) -> None:
        """Test function map registration."""
        registry = RegistryTestHelpers.create_test_registry()
        handlers = RegistryScenarios.create_handlers(test_case.handler_count)
        function_map = RegistryScenarios.create_function_map(handlers)
        if test_case.duplicate_registration and function_map:
            registry.register_function_map(function_map)
            result = registry.register_function_map(function_map)
        else:
            result = registry.register_function_map(function_map)
        assert result.is_success == test_case.should_succeed
        assert isinstance(result.value, FlextRegistry.Summary)

    @pytest.mark.parametrize(
        "test_case", RegistryScenarios.SUMMARY_SCENARIOS, ids=lambda tc: tc.name,
    )
    def test_summary_management(self, test_case: RegistryTestCase) -> None:
        """Test registry summary creation and properties."""
        summary = FlextRegistry.Summary()
        if test_case.handler_count > 0:
            for i in range(test_case.handler_count):
                summary.registered.append(
                    FlextModels.RegistrationDetails(
                        registration_id=f"test_{i}",
                        handler_mode=FlextConstants.Cqrs.HandlerType.COMMAND,
                        timestamp="2025-01-01T00:00:00Z",
                        status=FlextConstants.Cqrs.Status.RUNNING,
                    ),
                )
        if not test_case.should_succeed:
            summary.errors.append("test_error")
        assert len(summary.registered) == test_case.handler_count
        assert (len(summary.errors) > 0) == (not test_case.should_succeed)
        assert (not summary) == (not test_case.should_succeed)

    @pytest.mark.parametrize(
        "test_case", RegistryScenarios.KEY_RESOLUTION, ids=lambda tc: tc.name,
    )
    def test_key_resolution(self, test_case: RegistryTestCase) -> None:
        """Test binding and handler key resolution."""
        registry = RegistryTestHelpers.create_test_registry()
        handler = ConcreteTestHandler()
        if test_case.operation == RegistryOperationType.RESOLVE_HANDLER_KEY:
            key = registry._resolve_handler_key(handler)
            assert isinstance(key, str) and len(key) > 0
        elif test_case.operation == RegistryOperationType.RESOLVE_BINDING_KEY:
            key = registry._resolve_binding_key(handler, str)
            assert isinstance(key, str) and len(key) > 0

    @pytest.mark.parametrize(
        "test_case", RegistryScenarios.ERROR_SCENARIOS, ids=lambda tc: tc.name,
    )
    def test_error_handling(self, test_case: RegistryTestCase) -> None:
        """Test error handling scenarios."""
        registry = RegistryTestHelpers.create_test_registry()
        if test_case.handler_count == 0:
            result = registry.register_handler(None)
            assert result.is_failure
            assert "Handler cannot be None" in (result.error or "")
        else:
            handler = ConcreteTestHandler()
            result = registry.register_handler(handler)
            assert result.is_success
            assert isinstance(result.value, FlextModels.RegistrationDetails)

    def test_registry_initialization(self) -> None:
        """Test registry initialization."""
        registry = RegistryTestHelpers.create_test_registry()
        assert registry is not None
        assert isinstance(registry, FlextRegistry)

    def test_registry_with_dispatcher(self) -> None:
        """Test registry integration with dispatcher."""
        registry = RegistryTestHelpers.create_test_registry()
        handler = ConcreteTestHandler()
        result = registry.register_handler(handler)
        assert result.is_success
        assert isinstance(result.value, FlextModels.RegistrationDetails)

    @pytest.mark.parametrize(
        ("mode", "expected"),
        [
            ("command", FlextConstants.Cqrs.HandlerType.COMMAND),
            ("query", FlextConstants.Cqrs.HandlerType.QUERY),
            ("invalid", FlextConstants.Cqrs.HandlerType.COMMAND),
            (None, FlextConstants.Cqrs.HandlerType.COMMAND),
        ],
        ids=["command", "query", "invalid", "none"],
    )
    def test_safe_handler_mode_extraction(
        self, mode: str | None, expected: str,
    ) -> None:
        """Test safe handler mode extraction."""
        registry = RegistryTestHelpers.create_test_registry()
        assert registry._safe_get_handler_mode(mode) == expected

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("active", "running"),
            ("inactive", "completed"),
            ("invalid", "running"),
            (None, "running"),
        ],
        ids=["active", "inactive", "invalid", "none"],
    )
    def test_safe_status_extraction(self, status: str | None, expected: str) -> None:
        """Test safe status extraction."""
        registry = RegistryTestHelpers.create_test_registry()
        assert registry._safe_get_status(status) == expected


__all__ = ["ConcreteTestHandler", "TestFlextRegistry"]
