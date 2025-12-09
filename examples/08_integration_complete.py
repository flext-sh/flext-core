"""Flext complete integration demonstration.

Demonstrates integration of all flext-core components: Result, Container,
Context, Logger, Config, Models, Decorators, Registry, Dispatcher, and
Utilities using Python 3.13+ strict patterns with PEP 695 type aliases.

**Expected Output:**
- Complete workflow integration across all FLEXT components
- Service orchestration with dependency injection
- End-to-end request processing with context propagation
- Error handling with railway patterns
- Logging and monitoring integration
- Configuration-driven behavior
- CQRS command/query patterns in practice

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from pydantic import BaseModel, Field

from flext_core import (
    FlextConfig,
    FlextConstants,
    FlextContainer,
    FlextContext,
    FlextDecorators,
    FlextDispatcher,
    FlextLogger,
    FlextModels,
    FlextRegistry,
    FlextResult,
    s,
    t,
    u,
)
from flext_core.protocols import p

# ═══════════════════════════════════════════════════════════════════
# DOMAIN MODELS
# ═══════════════════════════════════════════════════════════════════


class User(FlextModels.Entity):
    """User entity."""

    name: str
    email: str


class Order(FlextModels.AggregateRoot):
    """Order aggregate root."""

    customer_id: str
    items: list[str] = Field(default_factory=list)
    status: FlextConstants.Domain.Status = Field(
        default=FlextConstants.Domain.Status.PENDING,
    )


# ═══════════════════════════════════════════════════════════════════
# SERVICE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════


class IntegrationService(s[t.Types.ServiceMetadataMapping]):
    """Service demonstrating complete flext-core integration."""

    def execute(
        self,
    ) -> FlextResult[t.Types.ServiceMetadataMapping]:
        """Execute complete integration demonstration."""
        print("Starting complete integration demonstration")

        try:
            self._demonstrate_result_patterns()
            self._demonstrate_container_integration()
            self._demonstrate_context_integration()
            self._demonstrate_logger_integration()
            self._demonstrate_config_integration()
            self._demonstrate_models_integration()
            self._demonstrate_decorators_integration()
            self._demonstrate_registry_dispatcher_integration()
            self._demonstrate_utilities_integration()

            return FlextResult[t.Types.ServiceMetadataMapping].ok({
                "components_integrated": [
                    "FlextResult",
                    "FlextContainer",
                    "FlextContext",
                    "FlextLogger",
                    "FlextConfig",
                    "FlextModels",
                    "FlextDecorators",
                    "FlextRegistry",
                    "FlextDispatcher",
                    "u",
                ],
                "integration_patterns": [
                    "railway_oriented",
                    "dependency_injection",
                    "context_propagation",
                    "structured_logging",
                    "configuration_management",
                    "domain_modeling",
                    "decorator_composition",
                    "cqrs_patterns",
                    "utility_functions",
                ],
                "total_components": 10,
            })

        except Exception as e:
            error_msg = f"Integration demonstration failed: {e}"
            return FlextResult[t.Types.ServiceMetadataMapping].fail(error_msg)

    @staticmethod
    def _demonstrate_result_patterns() -> None:
        """Show FlextResult patterns."""
        print("\n=== FlextResult Patterns ===")

        def to_upper(x: str) -> str:
            return x.upper()

        def add_processed(x: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"{x}_processed")

        # Railway pattern
        result = FlextResult[str].ok("initial").map(to_upper).flat_map(add_processed)
        if result.is_success:
            print(f"✅ Railway pattern: {result.unwrap()}")

    @staticmethod
    def _demonstrate_container_integration() -> None:
        """Show FlextContainer integration."""
        print("\n=== FlextContainer Integration ===")

        container = FlextContainer()
        logger = FlextLogger.create_module_logger(__name__)
        # Business Rule: Container accepts any object type including FlextLogger
        # Cast to container.register() compatible type for type checker
        logger_typed: (
            t.GeneralValueType | BaseModel | Callable[..., t.GeneralValueType] | object
        ) = logger
        _ = container.register("logger", logger_typed)

        logger_result: FlextResult[FlextLogger] = container.get("logger")
        if logger_result.is_success:
            print("✅ Container service resolution")

    @staticmethod
    def _demonstrate_context_integration() -> None:
        """Show FlextContext integration."""
        print("\n=== FlextContext Integration ===")

        with FlextContext.Request.request_context(operation_name="integration_demo"):
            correlation_id = (
                FlextContext.Variables.Correlation.CORRELATION_ID.get() or "unknown"
            )
            print(f"✅ Context correlation: {correlation_id}")

    @staticmethod
    def _demonstrate_logger_integration() -> None:
        """Show FlextLogger integration."""
        print("\n=== FlextLogger Integration ===")

        logger = FlextLogger.create_module_logger(__name__)
        logger.info("Integration demonstration", extra={"component": "logger"})
        print("✅ Structured logging")

    @staticmethod
    def _demonstrate_config_integration() -> None:
        """Show FlextConfig integration."""
        print("\n=== FlextConfig Integration ===")

        config = FlextConfig.get_global_instance()
        log_level = config.log_level
        print(f"✅ Config access: log_level={log_level}")

    @staticmethod
    def _demonstrate_models_integration() -> None:
        """Show FlextModels integration."""
        print("\n=== FlextModels Integration ===")

        user = User(
            unique_id=u.generate("entity"),
            name="Integration User",
            email="integration@example.com",
        )
        print(f"✅ Entity created: {user.name}")

        order = Order(
            unique_id=u.generate("entity"),
            customer_id=user.entity_id,
        )
        print(f"✅ Aggregate created: {order.status.value}")

    @staticmethod
    def _demonstrate_decorators_integration() -> None:
        """Show FlextDecorators integration."""
        print("\n=== FlextDecorators Integration ===")

        @FlextDecorators.log_operation(operation_name="integration_demo")
        def decorated_function(value: int) -> int:
            """Function with decorator."""
            return value * 2

        result = decorated_function(5)
        print(f"✅ Decorated function: {result}")

    @staticmethod
    def _demonstrate_registry_dispatcher_integration() -> None:
        """Show FlextRegistry and FlextDispatcher integration."""
        print("\n=== Registry/Dispatcher Integration ===")

        dispatcher = FlextDispatcher()
        _registry = FlextRegistry(
            dispatcher=cast("p.CommandBus | None", dispatcher),
        )
        print("✅ Registry/Dispatcher initialized")

    @staticmethod
    def _demonstrate_utilities_integration() -> None:
        """Show u integration."""
        print("\n=== u Integration ===")

        # Validation
        email_result = u.Validation.validate_pattern(
            "test@example.com",
            FlextConstants.Platform.PATTERN_EMAIL,
            "email",
        )
        if email_result.is_success:
            print("✅ Validation utility")

        # ID generation
        correlation_id = u.generate("correlation")
        print(f"✅ ID generation: {correlation_id[:12]}...")

        # Text processing
        cleaned = u.Text.clean_text("  test  ")
        print(f"✅ Text processing: '{cleaned}'")


def main() -> None:
    """Main entry point."""
    print("=" * 60)
    print("FLEXT COMPLETE INTEGRATION DEMONSTRATION")
    print("All components working together")
    print("=" * 60)

    service = IntegrationService()
    result = service.execute()

    if result.is_success:
        data = result.unwrap()
        components = data["components_integrated"]
        total = data["total_components"]
        if isinstance(components, Sequence) and isinstance(total, int):
            components_list = list(components)
            print(f"\n✅ Integrated {total} components")
            print(f"✅ Demonstrated {len(components_list)} integration patterns")
    else:
        print(f"\n❌ Failed: {result.error}")

    print("\n" + "=" * 60)
    print("🎯 Complete Integration: All flext-core components")
    print("🎯 Patterns: Railway, DI, Context, Logging, CQRS")
    print("🎯 Type Safety: PEP 695, collections.abc, Python 3.13+")
    print("=" * 60)


if __name__ == "__main__":
    main()
