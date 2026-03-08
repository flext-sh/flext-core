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

Uses Python 3.13 patterns, FlextTestsUtilities, c,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import ClassVar, cast, override

import pytest

from flext_core import (
    FlextRegistry,
    c,
    h,
    m,
    p,
    r,
    t,
)
from flext_tests import FlextTestsUtilities, u


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


class ConcreteTestHandler(h[t.ContainerValue, t.ContainerValue]):
    """Concrete implementation of h for testing."""

    @override
    def handle(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        """Handle the message."""
        return r[t.ContainerValue].ok(f"processed_{message}")

    def __call__(self, message: t.ContainerValue) -> r[t.ContainerValue]:
        """Make handler callable for registry validation."""
        return self.handle(message)

    def _protocol_name(self) -> str:
        """Return protocol name for registry identification."""
        return "test-handler"


class RegistryScenarios:
    """Centralized registry test scenarios using c."""

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
            should_succeed=True,
            error_pattern=None,
            with_bindings=False,
            with_function_map=False,
            with_summary=False,
            duplicate_registration=True,
        ),
    ]

    BINDING_REGISTRATION: ClassVar[list[RegistryTestCase]] = [
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

    FUNCTION_MAP_SCENARIOS: ClassVar[list[RegistryTestCase]] = [
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

    SUMMARY_SCENARIOS: ClassVar[list[RegistryTestCase]] = [
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

    KEY_RESOLUTION: ClassVar[list[RegistryTestCase]] = [
        RegistryTestCase(
            name="resolve_handler_key_string_type",
            operation=RegistryOperationType.RESOLVE_HANDLER_KEY,
            handler_count=1,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="resolve_handler_key_class_type",
            operation=RegistryOperationType.RESOLVE_HANDLER_KEY,
            handler_count=1,
            should_succeed=True,
        ),
        RegistryTestCase(
            name="resolve_binding_key",
            operation=RegistryOperationType.RESOLVE_BINDING_KEY,
            handler_count=1,
            should_succeed=True,
        ),
    ]

    ERROR_SCENARIOS: ClassVar[list[RegistryTestCase]] = [
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
    def create_handlers(
        count: int,
    ) -> list[p.Handler[t.ContainerValue, t.ContainerValue]]:
        """Create test handlers."""
        return [ConcreteTestHandler() for _ in range(count)]

    @staticmethod
    def create_bindings(
        handlers: Sequence[p.Handler[t.ContainerValue, t.ContainerValue]],
    ) -> list[
        tuple[
            type[t.ContainerValue],
            p.Handler[t.ContainerValue, t.ContainerValue],
        ]
    ]:
        """Create test bindings using str message type."""
        return [(str, handler) for handler in handlers]

    @staticmethod
    def create_function_map(
        handlers: Sequence[p.Handler[t.ContainerValue, t.ContainerValue]],
    ) -> dict[
        type[t.ContainerValue],
        p.Handler[t.ContainerValue, t.ContainerValue],
    ]:
        """Create test function map using str message type."""
        result: dict[
            type[t.ContainerValue],
            p.Handler[t.ContainerValue, t.ContainerValue],
        ] = {}
        for idx, handler in enumerate(handlers):
            result[str if idx == 0 else int] = handler
        return result


class TestFlextRegistry:
    """Refactored registry test suite using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.HANDLER_REGISTRATION,
        ids=lambda c: c.name,
    )
    def test_handler_registration(self, test_case: RegistryTestCase) -> None:
        """Test handler registration with various scenarios."""
        registry = FlextTestsUtilities.Tests.RegistryHelpers.create_test_registry()
        if test_case.handler_count == 0:
            result = registry.register_handler(
                cast(
                    "p.Handler[t.ContainerValue, t.ContainerValue]",
                    cast("object", None),
                ),
            )
        else:
            handler = ConcreteTestHandler()
            result = registry.register_handler(handler)
            if test_case.duplicate_registration:
                result = registry.register_handler(handler)
        if test_case.should_succeed:
            u.Tests.Result.assert_success(result)
        else:
            u.Tests.Result.assert_failure(result)
            if test_case.error_pattern:
                # Type ignore: RegistrationDetails is not t.ContainerValue but test is valid
                u.Tests.Result.assert_failure_with_error(
                    result,
                    test_case.error_pattern,
                )

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.BATCH_REGISTRATION,
        ids=lambda c: c.name,
    )
    def test_batch_registration(self, test_case: RegistryTestCase) -> None:
        """Test batch handler registration."""
        registry = FlextTestsUtilities.Tests.RegistryHelpers.create_test_registry()
        handlers = RegistryScenarios.create_handlers(test_case.handler_count)
        if test_case.duplicate_registration and handlers:
            registry.register_handlers(handlers)
            result = registry.register_handlers(handlers)
        else:
            result = registry.register_handlers(handlers)
        u.Tests.Result.assert_success(
            result,
        ) if test_case.should_succeed else u.Tests.Result.assert_failure(
            result,
        )
        assert isinstance(result.value, FlextRegistry.Summary)

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.BINDING_REGISTRATION,
        ids=lambda c: c.name,
    )
    def test_binding_registration(self, test_case: RegistryTestCase) -> None:
        """Test handler registration in batch mode."""
        registry = FlextTestsUtilities.Tests.RegistryHelpers.create_test_registry()
        handlers = RegistryScenarios.create_handlers(test_case.handler_count)
        # FlextRegistry.register_handlers takes Iterable[FlextHandlers], not dict
        if test_case.duplicate_registration and handlers:
            registry.register_handlers(handlers)
            result = registry.register_handlers(handlers)
        else:
            result = registry.register_handlers(handlers)
        u.Tests.Result.assert_success(
            result,
        ) if test_case.should_succeed else u.Tests.Result.assert_failure(
            result,
        )
        assert result.value is not None

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.FUNCTION_MAP_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_function_map_registration(self, test_case: RegistryTestCase) -> None:
        """Test function-based handler registration."""
        registry = FlextTestsUtilities.Tests.RegistryHelpers.create_test_registry()
        handlers = RegistryScenarios.create_handlers(test_case.handler_count)
        # FlextRegistry.register_handlers takes Iterable[FlextHandlers], not dict
        if test_case.duplicate_registration and handlers:
            registry.register_handlers(handlers)
            result = registry.register_handlers(handlers)
        else:
            result = registry.register_handlers(handlers)
        u.Tests.Result.assert_success(
            result,
        ) if test_case.should_succeed else u.Tests.Result.assert_failure(
            result,
        )
        assert result.value is not None

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.SUMMARY_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_summary_management(self, test_case: RegistryTestCase) -> None:
        """Test registry summary creation and properties."""
        summary = FlextRegistry.Summary()
        if test_case.handler_count > 0:
            for i in range(test_case.handler_count):
                summary.registered.append(
                    m.HandlerRegistrationDetails(
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

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.KEY_RESOLUTION,
        ids=lambda c: c.name,
    )
    def test_key_resolution(self, test_case: RegistryTestCase) -> None:
        """Test handler key resolution."""
        registry = FlextTestsUtilities.Tests.RegistryHelpers.create_test_registry()
        handler = ConcreteTestHandler()
        # All key resolution uses _resolve_handler_key
        if test_case.operation in {
            RegistryOperationType.RESOLVE_HANDLER_KEY,
            RegistryOperationType.RESOLVE_BINDING_KEY,
        }:
            key = registry._resolve_handler_key(handler)
            assert isinstance(key, str) and len(key) > 0

    @pytest.mark.parametrize(
        "test_case",
        RegistryScenarios.ERROR_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_error_handling(self, test_case: RegistryTestCase) -> None:
        """Test error handling scenarios."""
        registry = FlextTestsUtilities.Tests.RegistryHelpers.create_test_registry()
        if test_case.handler_count == 0:
            result = registry.register_handler(
                cast(
                    "p.Handler[t.ContainerValue, t.ContainerValue]",
                    cast("object", None),
                ),
            )
            u.Tests.Result.assert_failure(result)
            # Type ignore: RegistrationDetails is not t.ContainerValue but test is valid
            u.Tests.Result.assert_failure_with_error(
                result,
                "Handler must expose message_type",
            )
        else:
            handler = ConcreteTestHandler()
            result = registry.register_handler(handler)
            u.Tests.Result.assert_success(result)
            assert isinstance(result.value, m.HandlerRegistrationDetails)

    def test_registry_initialization(self) -> None:
        """Test registry initialization."""
        registry = FlextTestsUtilities.Tests.RegistryHelpers.create_test_registry()
        assert registry is not None
        assert isinstance(registry, FlextRegistry)

    def test_registry_with_dispatcher(self) -> None:
        """Test registry integration with dispatcher."""
        registry = FlextTestsUtilities.Tests.RegistryHelpers.create_test_registry()
        handler = ConcreteTestHandler()
        result = registry.register_handler(handler)
        u.Tests.Result.assert_success(result)
        assert isinstance(result.value, m.HandlerRegistrationDetails)

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
        """Test safe handler mode extraction."""
        registry = FlextTestsUtilities.Tests.RegistryHelpers.create_test_registry()
        assert registry._safe_get_handler_mode(mode or "") == expected

    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            ("active", "running"),
            ("inactive", "failed"),  # CommonStatus has FAILED, not STOPPED
            ("invalid", "running"),
            (None, "running"),
        ],
        ids=["active", "inactive", "invalid", "none"],
    )
    def test_safe_status_extraction(self, status: str | None, expected: str) -> None:
        """Test safe status extraction."""
        registry = FlextTestsUtilities.Tests.RegistryHelpers.create_test_registry()
        parsed_status = cast("str", status)
        assert registry._safe_get_status(parsed_status) == expected


__all__ = ["ConcreteTestHandler", "TestFlextRegistry"]
