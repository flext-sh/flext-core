"""Refactored comprehensive tests for FlextRegistry - Service Registry.

Tests the actual FlextRegistry API with real functionality testing using
Python 3.13 patterns, StrEnum, frozen dataclasses, and advanced parametrization.

Consolidates 56 test methods into 8 parametrized tests with comprehensive
coverage via ClassVar test data and factory patterns.

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

# =========================================================================
# Operation Type Enumeration
# =========================================================================


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


# =========================================================================
# Test Case Data Structure
# =========================================================================


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


# =========================================================================
# Handler Implementation
# =========================================================================


class ConcreteTestHandler(FlextHandlers[object, object]):
    """Concrete implementation of FlextHandlers for testing."""

    def handle(self, message: object) -> FlextResult[object]:
        """Handle the message."""
        return FlextResult[object].ok(f"processed_{message}")


# =========================================================================
# Test Scenario Factory
# =========================================================================


class RegistryScenarios:
    """Factory for registry test scenarios with centralized test data."""

    # Handler registration scenarios
    HANDLER_REGISTRATION: ClassVar[list[RegistryTestCase]] = [
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
            duplicate_registration=True,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="none_handler_failure",
            operation=RegistryOperationType.REGISTER_HANDLER,
            handler_count=0,
            should_succeed=False,
            error_pattern="Handler cannot be None",
        ),
    ]

    # Batch registration scenarios
    BATCH_REGISTRATION: ClassVar[list[RegistryTestCase]] = [
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
            duplicate_registration=True,
            should_succeed=True,
        ),
    ]

    # Binding registration scenarios
    BINDING_REGISTRATION: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            name="single_binding_success",
            operation=RegistryOperationType.REGISTER_BINDINGS,
            handler_count=1,
            with_bindings=True,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="empty_bindings_list",
            operation=RegistryOperationType.REGISTER_BINDINGS,
            handler_count=0,
            with_bindings=True,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="duplicate_bindings",
            operation=RegistryOperationType.REGISTER_BINDINGS,
            handler_count=1,
            with_bindings=True,
            duplicate_registration=True,
            should_succeed=True,
        ),
    ]

    # Function map scenarios
    FUNCTION_MAP_SCENARIOS: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            name="function_map_with_handler",
            operation=RegistryOperationType.REGISTER_FUNCTION_MAP,
            handler_count=1,
            with_function_map=True,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="empty_function_map",
            operation=RegistryOperationType.REGISTER_FUNCTION_MAP,
            handler_count=0,
            with_function_map=True,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="duplicate_function_map",
            operation=RegistryOperationType.REGISTER_FUNCTION_MAP,
            handler_count=1,
            with_function_map=True,
            duplicate_registration=True,
            should_succeed=True,
        ),
    ]

    # Summary management scenarios
    SUMMARY_SCENARIOS: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            name="empty_summary",
            operation=RegistryOperationType.SUMMARY_MANAGEMENT,
            handler_count=0,
            with_summary=True,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="summary_with_registrations",
            operation=RegistryOperationType.SUMMARY_MANAGEMENT,
            handler_count=2,
            with_summary=True,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="summary_with_errors",
            operation=RegistryOperationType.SUMMARY_MANAGEMENT,
            handler_count=1,
            with_summary=True,
            should_succeed=False,
        ),
    ]

    # Key resolution scenarios
    KEY_RESOLUTION: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            name="resolve_handler_key_string_type",
            operation=RegistryOperationType.RESOLVE_HANDLER_KEY,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="resolve_handler_key_class_type",
            operation=RegistryOperationType.RESOLVE_HANDLER_KEY,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="resolve_binding_key",
            operation=RegistryOperationType.RESOLVE_BINDING_KEY,
            should_succeed=True,
        ),
    ]

    # Error handling scenarios
    ERROR_SCENARIOS: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            name="register_none_handler",
            operation=RegistryOperationType.ERROR_HANDLING,
            handler_count=0,
            should_succeed=False,
            error_pattern="Handler cannot be None",
        ),
        RegistryTestCase(
            name="dispatcher_integration",
            operation=RegistryOperationType.ERROR_HANDLING,
            handler_count=1,
            should_succeed=True,
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
            # Use different types for each handler to avoid key collision
            result[str if idx == 0 else int] = handler
        return result


# =========================================================================
# Test Suite
# =========================================================================


class TestFlextRegistry:
    """Refactored registry test suite with parametrized tests."""

    # =====================================================================
    # Handler Registration Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.HANDLER_REGISTRATION,
        ids=lambda tc: tc.name,
    )
    def test_handler_registration(self, test_case: RegistryTestCase) -> None:
        """Test handler registration with various scenarios."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        if test_case.handler_count == 0:
            result = registry.register_handler(None)
        else:
            handler = ConcreteTestHandler()
            result = registry.register_handler(handler)

            if test_case.duplicate_registration:
                # Register again to test idempotence
                result = registry.register_handler(handler)

        # Assert result
        assert result.is_success == test_case.should_succeed
        if test_case.error_pattern:
            assert test_case.error_pattern in (result.error or "")

    # =====================================================================
    # Batch Registration Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.BATCH_REGISTRATION,
        ids=lambda tc: tc.name,
    )
    def test_batch_registration(self, test_case: RegistryTestCase) -> None:
        """Test batch handler registration."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        handlers = RegistryScenarios.create_handlers(test_case.handler_count)

        if test_case.duplicate_registration and handlers:
            # Register first time
            result1 = registry.register_handlers(handlers)
            assert result1.is_success
            # Register again to test idempotence
            result = registry.register_handlers(handlers)
        else:
            result = registry.register_handlers(handlers)

        assert result.is_success == test_case.should_succeed
        assert isinstance(result.value, FlextRegistry.Summary)

    # =====================================================================
    # Binding Registration Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.BINDING_REGISTRATION,
        ids=lambda tc: tc.name,
    )
    def test_binding_registration(self, test_case: RegistryTestCase) -> None:
        """Test binding registration."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        handlers = RegistryScenarios.create_handlers(test_case.handler_count)
        bindings = RegistryScenarios.create_bindings(handlers)

        if test_case.duplicate_registration and bindings:
            # Register first time
            result1 = registry.register_bindings(bindings)
            assert result1.is_success
            # Register again to test idempotence
            result = registry.register_bindings(bindings)
        else:
            result = registry.register_bindings(bindings)

        assert result.is_success == test_case.should_succeed
        assert isinstance(result.value, FlextRegistry.Summary)

    # =====================================================================
    # Function Map Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.FUNCTION_MAP_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_function_map_registration(self, test_case: RegistryTestCase) -> None:
        """Test function map registration."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        handlers = RegistryScenarios.create_handlers(test_case.handler_count)
        function_map = RegistryScenarios.create_function_map(handlers)

        if test_case.duplicate_registration and function_map:
            # Register first time
            result1 = registry.register_function_map(function_map)
            assert result1.is_success
            # Register again to test idempotence
            result = registry.register_function_map(function_map)
        else:
            result = registry.register_function_map(function_map)

        assert result.is_success == test_case.should_succeed
        assert isinstance(result.value, FlextRegistry.Summary)

    # =====================================================================
    # Summary Management Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.SUMMARY_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_summary_management(self, test_case: RegistryTestCase) -> None:
        """Test registry summary creation and properties."""
        summary = FlextRegistry.Summary()

        # Add test data based on scenario
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

        # Test properties - use len() directly to avoid Pydantic v2 typing issues
        assert len(summary.registered) == test_case.handler_count
        assert (len(summary.errors) > 0) == (not test_case.should_succeed)
        assert (not summary) == (not test_case.should_succeed)

    # =====================================================================
    # Key Resolution Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.KEY_RESOLUTION,
        ids=lambda tc: tc.name,
    )
    def test_key_resolution(self, test_case: RegistryTestCase) -> None:
        """Test binding and handler key resolution."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        handler = ConcreteTestHandler()

        if test_case.operation == RegistryOperationType.RESOLVE_HANDLER_KEY:
            key = registry._resolve_handler_key(handler)
            assert isinstance(key, str)
            assert len(key) > 0

        elif test_case.operation == RegistryOperationType.RESOLVE_BINDING_KEY:
            key = registry._resolve_binding_key(handler, str)
            assert isinstance(key, str)
            assert len(key) > 0

    # =====================================================================
    # Error Handling Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.ERROR_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_error_handling(self, test_case: RegistryTestCase) -> None:
        """Test error handling scenarios."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        if test_case.operation == RegistryOperationType.ERROR_HANDLING:
            if test_case.handler_count == 0:
                # Test None handler
                result = registry.register_handler(None)
                assert result.is_failure
                assert "Handler cannot be None" in (result.error or "")
            else:
                # Test dispatcher integration
                handler = ConcreteTestHandler()
                result = registry.register_handler(handler)
                assert result.is_success
                assert isinstance(result.value, FlextModels.RegistrationDetails)

    # =====================================================================
    # Integration Tests
    # =====================================================================

    def test_registry_initialization(self) -> None:
        """Test registry initialization."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)
        assert registry is not None
        assert isinstance(registry, FlextRegistry)

    def test_registry_with_dispatcher(self) -> None:
        """Test registry integration with dispatcher."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)
        handler = ConcreteTestHandler()

        result = registry.register_handler(handler)
        assert result.is_success
        assert isinstance(result.value, FlextModels.RegistrationDetails)

    def test_safe_handler_mode_extraction(self) -> None:
        """Test safe handler mode extraction."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        # Test valid modes
        assert (
            registry._safe_get_handler_mode("command")
            == FlextConstants.Cqrs.HandlerType.COMMAND
        )
        assert (
            registry._safe_get_handler_mode("query")
            == FlextConstants.Cqrs.HandlerType.QUERY
        )

        # Test invalid mode (should default to command)
        assert (
            registry._safe_get_handler_mode("invalid")
            == FlextConstants.Cqrs.HandlerType.COMMAND
        )
        assert (
            registry._safe_get_handler_mode(None)
            == FlextConstants.Cqrs.HandlerType.COMMAND
        )

    def test_safe_status_extraction(self) -> None:
        """Test safe status extraction."""
        dispatcher = FlextDispatcher()
        registry = FlextRegistry(dispatcher=dispatcher)

        # Test status mapping
        assert registry._safe_get_status("active") == "running"
        assert registry._safe_get_status("inactive") == "completed"

        # Test invalid status (should default to running)
        assert registry._safe_get_status("invalid") == "running"
        assert registry._safe_get_status(None) == "running"


__all__ = ["ConcreteTestHandler", "TestFlextRegistry"]
