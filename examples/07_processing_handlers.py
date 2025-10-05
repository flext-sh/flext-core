# !/usr/bin/env python3
"""07 - FlextProcessors: Handler Pipeline and Strategy Patterns.

This example demonstrates the COMPLETE FlextProcessors API for building
handler pipelines, strategy patterns, and message processing systems.

Key Concepts Demonstrated:
- Handler Creation: BasicHandler and custom handlers
- Pipeline Building: Chain of responsibility pattern
- Strategy Pattern: Dynamic algorithm selection
- Registry Pattern: Handler registration and discovery
- Message Processing: Request/response handling
- Error Recovery: Pipeline error handling and fallback

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import warnings
from typing import cast

from flext_core import (
    FlextLogger,
    FlextProcessors,
    FlextResult,
    FlextService,
    FlextTypes,
)

from .example_scenarios import ExampleScenarios


class ProcessingPatternsService(FlextService[FlextTypes.Dict]):
    """Service demonstrating ALL FlextProcessors patterns."""

    def __init__(self) -> None:
        """Initialize with automatic FlextCore infrastructure."""
        super().__init__()
        self._scenarios = ExampleScenarios()
        self._user = self._scenarios.user()
        self._order = self._scenarios.realistic_data()["order"]
        self._admin_user: dict[str, object] = {
            **self._user,
            "role": "admin",
            "token": "valid-token",
        }
        self._invalid_user: dict[str, object] = {
            **self._user,
            "email": "invalid",
            "role": "user",
        }
        self._metadata = self._scenarios.metadata(tags=["processors", "demo"])

    def execute(self) -> FlextResult[FlextTypes.Dict]:
        """Execute method required by FlextService."""
        self._logger.info("Executing processing demo")
        return FlextResult[FlextTypes.Dict].ok({
            "status": "processed",
            "handlers_executed": True,
        })

    # ========== BASIC HANDLER PATTERNS ==========

    def demonstrate_basic_handlers(self) -> None:
        """Show basic handler creation and execution."""
        print("\n=== Basic Handler Patterns ===")

        class ValidationHandler(FlextProcessors.Implementation.BasicHandler):
            """Handler for data validation."""

            def __init__(self, name: str) -> None:
                """Initialize handler with logger."""
                super().__init__(name)
                self._logger = FlextLogger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Validate and process the request."""
                self._logger.info(f"Validating request in {self.name}")

                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                email_value: str | None = request.get("email", None)
                if not email_value:
                    return FlextResult[str].fail("Email required")

                if not isinstance(email_value, str):
                    return FlextResult[str].fail("Email must be a string")

                if "@" not in email_value:
                    return FlextResult[str].fail("Invalid email format")

                request["validated_at"] = str(time.time())
                return FlextResult[str].ok(f"Validation passed for {request['email']}")

        validator = ValidationHandler("EmailValidator")

        valid_request = {
            "email": self._user["email"],
            "name": self._user["name"],
        }
        result = validator.handle(valid_request)
        if result.is_success:
            print(f"✅ Validation passed: {result.unwrap()}")

        invalid_request: dict[str, object] = {
            "email": self._invalid_user["email"],
            "name": self._invalid_user["name"],
        }
        result = validator.handle(invalid_request)
        if result.is_failure:
            print(f"❌ Validation failed: {result.error}")

    # ========== HANDLER PIPELINE ==========

    def demonstrate_handler_pipeline(self) -> None:
        """Show chain of responsibility pattern with handler pipeline."""
        print("\n=== Handler Pipeline ===")

        class AuthenticationHandler(FlextProcessors.Implementation.BasicHandler):
            """Authenticate the request."""

            def __init__(self, name: str) -> None:
                """Initialize handler with logger."""
                super().__init__(name)
                self._logger = FlextLogger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Check authentication."""
                self._logger.info("Authenticating request")

                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                request_dict = cast("dict[str, object]", request)
                token_value: str | None = request_dict.get("token", None)
                if not token_value:
                    return FlextResult[str].fail("Authentication required")

                if not isinstance(token_value, str):
                    return FlextResult[str].fail("Token must be a string")

                if token_value != "valid-token":
                    return FlextResult[str].fail("Invalid token")

                request["authenticated"] = True
                return FlextResult[str].ok("Authentication successful")

        class AuthorizationHandler(FlextProcessors.Implementation.BasicHandler):
            """Authorize the request."""

            def __init__(self, name: str) -> None:
                """Initialize handler with logger."""
                super().__init__(name)
                self._logger = FlextLogger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Check authorization."""
                self._logger.info("Authorizing request")

                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                request_dict = cast("dict[str, object]", request)
                authenticated_value: bool | None = request_dict.get(
                    "authenticated", None
                )
                if not authenticated_value:
                    return FlextResult[str].fail("Not authenticated")

                role_value: str | None = request_dict.get("role", None)
                if role_value != "admin":
                    return FlextResult[str].fail("Insufficient permissions")

                request["authorized"] = True
                return FlextResult[str].ok("Authorization successful")

        class ProcessingHandler(FlextProcessors.Implementation.BasicHandler):
            """Process the authorized request."""

            def __init__(self, name: str) -> None:
                """Initialize handler with logger."""
                super().__init__(name)
                self._logger = FlextLogger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Process the business logic."""
                self._logger.info("Processing request")

                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                request_dict = cast("dict[str, object]", request)
                authorized_value: bool | None = request_dict.get("authorized", None)
                if not authorized_value:
                    return FlextResult[str].fail("Not authorized")

                request["processed"] = True
                request["result"] = "Operation completed successfully"
                request["timestamp"] = str(time.time())

                return FlextResult[str].ok("Processing completed successfully")

        auth_handler = AuthenticationHandler("Authenticator")
        authz_handler = AuthorizationHandler("Authorizer")
        process_handler = ProcessingHandler("Processor")

        def execute_pipeline(request: FlextTypes.Dict) -> FlextResult[str]:
            """Execute the handler pipeline."""
            result = auth_handler.handle(request)
            if result.is_failure:
                return result

            result = authz_handler.handle(request)
            if result.is_failure:
                return result

            return process_handler.handle(request)

        print("\n1. Valid request through pipeline:")
        valid_request: dict[str, object] = {
            "token": self._admin_user["token"],
            "role": self._admin_user["role"],
            "action": "delete_user",
            "user": self._admin_user,
        }
        result = execute_pipeline(valid_request)
        if result.is_success:
            print(f"✅ Pipeline success: {result.unwrap()}")

        print("\n2. Request with invalid token:")
        invalid_token: FlextTypes.Dict = {
            "token": "invalid",
            "role": self._admin_user["role"],
            "action": "delete_user",
            "user": self._admin_user,
        }
        result = execute_pipeline(invalid_token)
        if result.is_failure:
            print(f"❌ Pipeline failed at auth: {result.error}")

        print("\n3. Request with insufficient permissions:")
        insufficient: dict[str, object] = {
            "token": self._admin_user["token"],
            "role": self._invalid_user.get("role", "user"),
            "action": "delete_user",
            "user": self._invalid_user,
        }
        result = execute_pipeline(insufficient)
        if result.is_failure:
            print(f"❌ Pipeline failed at authz: {result.error}")

    # ========== STRATEGY PATTERN ==========

    def demonstrate_strategy_pattern(self) -> None:
        """Show strategy pattern for algorithm selection."""
        print("\n=== Strategy Pattern ===")

        class PaymentStrategy:
            """Base payment processing strategy."""

            def process(self, amount: float) -> FlextResult[FlextTypes.Dict]:
                """Process payment with specific strategy."""
                raise NotImplementedError

        class CreditCardStrategy(PaymentStrategy):
            """Credit card payment strategy."""

            def process(self, amount: float) -> FlextResult[FlextTypes.Dict]:
                """Process credit card payment."""
                fee = amount * 0.029
                return FlextResult[FlextTypes.Dict].ok({
                    "method": "credit_card",
                    "amount": amount,
                    "fee": round(fee, 2),
                    "total": round(amount + fee, 2),
                    "status": "processed",
                })

        class PayPalStrategy(PaymentStrategy):
            """PayPal payment strategy."""

            def process(self, amount: float) -> FlextResult[FlextTypes.Dict]:
                """Process PayPal payment."""
                fee = amount * 0.034 + 0.30
                return FlextResult[FlextTypes.Dict].ok({
                    "method": "paypal",
                    "amount": amount,
                    "fee": round(fee, 2),
                    "total": round(amount + fee, 2),
                    "status": "processed",
                })

        class BankTransferStrategy(PaymentStrategy):
            """Bank transfer payment strategy."""

            def process(self, amount: float) -> FlextResult[FlextTypes.Dict]:
                """Process bank transfer."""
                fee = 5.00
                return FlextResult[FlextTypes.Dict].ok({
                    "method": "bank_transfer",
                    "amount": amount,
                    "fee": fee,
                    "total": amount + fee,
                    "status": "pending",
                })

        class PaymentProcessor:
            """Payment processor using strategy pattern."""

            def __init__(self) -> None:
                """Initialize with strategies."""
                self._strategies: dict[str, PaymentStrategy] = {
                    "credit_card": CreditCardStrategy(),
                    "paypal": PayPalStrategy(),
                    "bank_transfer": BankTransferStrategy(),
                }

            def process(
                self,
                method: str,
                amount: float,
            ) -> FlextResult[FlextTypes.Dict]:
                """Process payment with selected strategy."""
                strategy = self._strategies.get(method)
                if strategy is None:
                    return FlextResult[FlextTypes.Dict].fail("Unknown payment method")
                return strategy.process(amount)

        processor = PaymentProcessor()
        amount = float(self._order["total"])

        card_result = processor.process("credit_card", amount)
        if card_result.is_success:
            print(f"✅ Credit card processed: {card_result.unwrap()['total']}")

        paypal_result = processor.process("paypal", amount)
        if paypal_result.is_success:
            print(f"✅ PayPal processed: {paypal_result.unwrap()['total']}")

        bank_result = processor.process("bank_transfer", amount)
        if bank_result.is_success:
            print(f"✅ Bank transfer pending: {bank_result.unwrap()['status']}")

    # ========== REGISTRY PATTERN ==========

    def demonstrate_registry_pattern(self) -> None:
        """Show handler registry for dynamic discovery."""
        print("\n=== Registry Pattern ===")

        # Handler registry
        class HandlerRegistry:
            """Registry for dynamic handler management."""

            def __init__(self) -> None:
                """Initialize registry."""
                self._handlers: dict[
                    str,
                    FlextProcessors.Implementation.BasicHandler,
                ] = {}
                self._logger = FlextLogger(__name__)

            def register(
                self,
                name: str,
                handler: FlextProcessors.Implementation.BasicHandler,
            ) -> FlextResult[None]:
                """Register a handler."""
                if name in self._handlers:
                    return FlextResult[None].fail(f"Handler {name} already registered")

                self._handlers[name] = handler
                self._logger.info("Registered handler: %s", name)
                return FlextResult[None].ok(None)

            def unregister(self, name: str) -> FlextResult[None]:
                """Unregister a handler."""
                if name not in self._handlers:
                    return FlextResult[None].fail(f"Handler {name} not found")

                del self._handlers[name]
                self._logger.info("Unregistered handler: %s", name)
                return FlextResult[None].ok(None)

            def get(
                self,
                name: str,
            ) -> FlextResult[FlextProcessors.Implementation.BasicHandler]:
                """Get a handler by name."""
                handler = self._handlers.get(name)
                if not handler:
                    return FlextResult[
                        FlextProcessors.Implementation.BasicHandler
                    ].fail(f"Handler {name} not found")

                return FlextResult[FlextProcessors.Implementation.BasicHandler].ok(
                    handler,
                )

            def execute(
                self,
                name: str,
                request: FlextTypes.Dict,
            ) -> FlextResult[str]:
                """Execute a handler by name."""
                handler_result = self.get(name)
                if handler_result.is_failure:
                    return FlextResult[str].fail(
                        handler_result.error or "Handler not found",
                    )

                handler = handler_result.unwrap()
                return handler.handle(request)

            def list_handlers(self) -> FlextTypes.StringList:
                """List all registered handlers."""
                return list(self._handlers.keys())

        # Create registry and handlers
        registry = HandlerRegistry()

        # Register various handlers
        class UpperCaseHandler(FlextProcessors.Implementation.BasicHandler):
            """Convert text to uppercase."""

            def handle(self, request: object) -> FlextResult[str]:
                """Process text."""
                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                request_dict = cast("dict[str, object]", request)
                text_value: str = request_dict.get("text", "")
                if not isinstance(text_value, str):
                    text_value = str(text_value)
                return FlextResult[str].ok(f"Uppercase: {text_value.upper()}")

        class LowerCaseHandler(FlextProcessors.Implementation.BasicHandler):
            """Convert text to lowercase."""

            def handle(self, request: object) -> FlextResult[str]:
                """Process text."""
                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                request_dict = cast("dict[str, object]", request)
                text_value: str = request_dict.get("text", "")
                if not isinstance(text_value, str):
                    text_value = str(text_value)
                return FlextResult[str].ok(f"Lowercase: {text_value.lower()}")

        class ReverseHandler(FlextProcessors.Implementation.BasicHandler):
            """Reverse text."""

            def handle(self, request: object) -> FlextResult[str]:
                """Process text."""
                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                text_value: str = request.get("text", "")
                if not isinstance(text_value, str):
                    text_value = str(text_value)
                return FlextResult[str].ok(f"Reversed: {text_value[::-1]}")

        # Register handlers
        registry.register("uppercase", UpperCaseHandler("UpperCase"))
        registry.register("lowercase", LowerCaseHandler("LowerCase"))
        registry.register("reverse", ReverseHandler("Reverse"))

        print(f"Registered handlers: {registry.list_handlers()}")

        # Dynamic handler execution
        test_request: FlextTypes.Dict = {"text": "Hello FLEXT"}

        print("\n1. Uppercase handler:")
        result = registry.execute("uppercase", test_request)
        if result.is_success:
            print(f"✅ {result.unwrap()}")

        print("\n2. Lowercase handler:")
        result = registry.execute("lowercase", test_request)
        if result.is_success:
            print(f"✅ {result.unwrap()}")

        print("\n3. Reverse handler:")
        result = registry.execute("reverse", test_request)
        if result.is_success:
            print(f"✅ {result.unwrap()}")

        print("\n4. Unknown handler:")
        result = registry.execute("unknown", test_request)
        if result.is_failure:
            print(f"❌ {result.error}")

        # Unregister and retry
        registry.unregister("reverse")
        print(f"\nHandlers after unregister: {registry.list_handlers()}")

    # ========== ERROR RECOVERY ==========

    def demonstrate_error_recovery(self) -> None:
        """Show error recovery in processing pipelines."""
        print("\n=== Error Recovery Patterns ===")

        # Handler with retry logic
        class RetryableHandler(FlextProcessors.Implementation.BasicHandler):
            """Handler that can retry on failure."""

            def __init__(self, name: str, max_retries: int = 3) -> None:
                """Initialize with retry configuration."""
                super().__init__(name)
                self._max_retries = max_retries
                self._attempt = 0
                self._logger = FlextLogger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Handle with retry logic."""
                self._attempt += 1
                _ = request  # Mark as used to avoid linting warning

                # Simulate intermittent failure
                if self._attempt < self._max_retries:
                    self._logger.warning(
                        f"Attempt {self._attempt} failed",
                        extra={"handler": self.name},
                    )
                    return FlextResult[str].fail(f"Attempt {self._attempt} failed")

                # Success on final attempt
                self._logger.info(
                    f"Success on attempt {self._attempt}",
                    extra={"handler": self.name},
                )
                return FlextResult[str].ok(f"Success on attempt {self._attempt}")

        # Circuit breaker pattern
        class CircuitBreakerHandler(FlextProcessors.Implementation.BasicHandler):
            """Handler with circuit breaker pattern."""

            def __init__(self, name: str, threshold: int = 3) -> None:
                """Initialize with circuit breaker settings."""
                super().__init__(name)
                self._failure_count = 0
                self._threshold = threshold
                self._is_open = False
                self._logger = FlextLogger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Handle with circuit breaker."""
                # Check if circuit is open
                if self._is_open:
                    return FlextResult[str].fail("Circuit breaker is open")

                # Convert object to dict for processing
                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                # Simulate processing
                request_dict = cast("dict[str, object]", request)
                force_fail_value: bool = request_dict.get("force_fail", False)
                should_fail = bool(force_fail_value)

                if should_fail:
                    self._failure_count += 1
                    if self._failure_count >= self._threshold:
                        self._is_open = True
                        self._logger.error(
                            "Circuit breaker opened",
                            extra={"failures": self._failure_count},
                        )
                    return FlextResult[str].fail("Processing failed")

                # Reset on success
                self._failure_count = 0
                return FlextResult[str].ok("Processing successful")

            def reset(self) -> None:
                """Reset circuit breaker."""
                self._is_open = False
                self._failure_count = 0
                self._logger.info("Circuit breaker reset")

        # Test retry handler
        print("\n1. Retry Handler:")
        retry_handler = RetryableHandler("RetryHandler", max_retries=3)

        # Keep trying until success
        for i in range(3):
            result = retry_handler.handle({"data": "test"})
            if result.is_success:
                print(f"✅ Success: {result.unwrap()}")
                break
            print(f"⚠️ Retry {i + 1}: {result.error}")

        # Test circuit breaker
        print("\n2. Circuit Breaker Handler:")
        breaker = CircuitBreakerHandler("BreakerHandler", threshold=2)

        # Successful request
        result = breaker.handle({"data": "test", "force_fail": False})
        print(
            f"Request 1: {'✅' if result.is_success else '❌'} {result.unwrap() if result.is_success else result.error}",
        )

        # Failing requests trigger circuit breaker
        for i in range(3):
            result = breaker.handle({"data": "test", "force_fail": True})
            print(
                f"Request {i + 2}: {'✅' if result.is_success else '❌'} {result.error if result.is_failure else ''}",
            )

        # Circuit is now open
        result = breaker.handle({"data": "test", "force_fail": False})
        print(f"Request 5 (circuit open): ❌ {result.error}")

        # Reset and retry
        breaker.reset()
        result = breaker.handle({"data": "test", "force_fail": False})
        print(
            f"Request 6 (after reset): {'✅' if result.is_success else '❌'} Success after reset",
        )

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated processing patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Direct exception handling (DEPRECATED)
        warnings.warn(
            "Exception-based handlers are DEPRECATED! Use FlextResult.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (exceptions):")
        print("try:")
        print("    result = handler.process(data)")
        print("except ProcessingError as e:")
        print("    handle_error(e)")

        print("\n✅ CORRECT WAY (FlextResult):")
        print("result = handler.handle(data)")
        print("if result.is_failure:")
        print("    logger.error(f'Processing failed: {result.error}')")

        # OLD: Global handler state (DEPRECATED)
        warnings.warn(
            "Global handler state is DEPRECATED! Use handler instances.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (global state):")
        print("HANDLER_REGISTRY = {}")
        print("def register_handler(name, func): ...")

        print("\n✅ CORRECT WAY (instance registry):")
        print("registry = HandlerRegistry()")
        print("registry.register('name', handler_instance)")

        # OLD: Synchronous-only processing (DEPRECATED)
        warnings.warn(
            "Sync-only processing is DEPRECATED! Design for compatibility.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (sync only):")
        print("def process(data):")
        print("    return processed_data")

        print("\n✅ CORRECT WAY (-ready):")
        print("def handle(self, request: dict) -> FlextResult[FlextTypes.Dict]:")
        print("    # Returns FlextResult for composition")


def main() -> None:
    """Main entry point demonstrating all FlextProcessors capabilities."""
    service = ProcessingPatternsService()

    print("=" * 60)
    print("FlextProcessors COMPLETE API DEMONSTRATION")
    print("Handler Pipelines and Strategy Patterns")
    print("=" * 60)

    # Core patterns
    service.demonstrate_basic_handlers()
    service.demonstrate_handler_pipeline()

    # Advanced patterns
    service.demonstrate_strategy_pattern()
    service.demonstrate_registry_pattern()

    # Professional patterns
    service.demonstrate_error_recovery()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextProcessors patterns demonstrated!")
    print("🎯 Next: See 08_*.py for additional advanced patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
