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
from copy import deepcopy
from typing import ClassVar, cast

from flext_core import (
    FlextConstants,
    FlextExceptions,
    FlextLogger,
    FlextProcessors,
    FlextResult,
    FlextRuntime,
    FlextService,
    FlextTypes,
)


class DemoScenarios:
    """Inline scenario helpers for processing handler demonstrations."""

    _DATASET: ClassVar[FlextTypes.Dict] = {
        "users": [
            {
                "id": 1,
                "name": "Alice Example",
                "email": "alice@example.com",
                "age": 30,
            }
        ],
    }

    _REALISTIC: ClassVar[FlextTypes.Dict] = {
        "order": {
            "order_id": "order-456",
            "customer_id": "cust-123",
            "items": [
                {"product_id": "prod-001", "name": "Widget", "quantity": 1},
            ],
            "total": "29.99",
        }
    }

    @staticmethod
    def user(**overrides: object) -> FlextTypes.Dict:
        """Create user data dictionary for processing examples."""
        users_list = cast("list[FlextTypes.Dict]", DemoScenarios._DATASET["users"])
        user = deepcopy(users_list[0])
        user.update(overrides)
        return user

    @staticmethod
    def realistic_data() -> FlextTypes.Dict:
        """Create realistic order data dictionary for processing examples."""
        return deepcopy(DemoScenarios._REALISTIC)

    @staticmethod
    def metadata(
        *,
        source: str = "examples",
        tags: FlextTypes.StringList | None = None,
        **extra: object,
    ) -> FlextTypes.Dict:
        """Create metadata dictionary for processing examples."""
        data: FlextTypes.Dict = {
            "source": source,
            "component": "flext_core",
            "tags": tags or ["processors", "demo"],
        }
        data.update(extra)
        return data


class ProcessingPatternsService(FlextService[FlextTypes.Dict]):
    """Service demonstrating ALL FlextProcessors patterns with FlextMixins infrastructure.

    This service inherits from FlextService to demonstrate:
    - Inherited container property (FlextContainer singleton)
    - Inherited logger property (FlextLogger with service context - PROCESSORS FOCUS!)
    - Inherited context property (FlextContext for handler execution tracking)
    - Inherited config property (FlextConfig with processing settings)
    - Inherited metrics property (FlextMetrics for handler observability)

    The focus is on demonstrating FlextProcessors patterns (handlers, pipelines,
    strategies, registry) with structured logging and handler execution tracking,
    while leveraging complete FlextMixins infrastructure for orchestration.
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextMixins infrastructure.

        Note: No manual logger initialization needed!
        All infrastructure is inherited from FlextService base class:
        - self.logger: FlextLogger with service context (ALREADY CONFIGURED!)
        - self.container: FlextContainer global singleton
        - self.context: FlextContext for handler execution tracking
        - self.config: FlextConfig with processing configuration
        - self.metrics: FlextMetrics for handler observability
        """
        super().__init__()
        # Use self.logger from FlextMixins, not logger
        self._scenarios = DemoScenarios()
        self._user = self._scenarios.user()
        self._order: FlextTypes.Dict = cast(
            "FlextTypes.Dict", self._scenarios.realistic_data()["order"]
        )
        self._REDACTED_LDAP_BIND_PASSWORD_user: FlextTypes.Dict = {
            **self._user,
            "role": "REDACTED_LDAP_BIND_PASSWORD",
            "token": "valid-token",
        }
        self._invalid_user: FlextTypes.Dict = {
            **self._user,
            "email": "invalid",
            "role": "user",
        }
        self._metadata = self._scenarios.metadata(tags=["processors", "demo"])

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "ProcessingPatternsService initialized with inherited infrastructure",
            extra={
                "service_type": "FlextProcessors & Handler Patterns demonstration",
                "handler_types": ["BasicHandler", "Pipeline", "Strategy", "Registry"],
                "processing_patterns": True,
            },
        )

    def execute(self) -> FlextResult[FlextTypes.Dict]:
        """Execute all FlextProcessors pattern demonstrations.

        Demonstrates inherited infrastructure alongside handler patterns:
        - Inherited logger for structured handler execution logs
        - Inherited context for handler execution tracking
        - Complete handler pipeline and strategy patterns
        - Registry pattern with handler discovery

        Returns:
            FlextResult[Dict] with demonstration summary including infrastructure details

        """
        self.logger.info("Starting comprehensive FlextProcessors demonstration")

        try:
            # Core patterns
            self.demonstrate_basic_handlers()
            self.demonstrate_handler_pipeline()

            # Advanced patterns
            self.demonstrate_strategy_pattern()
            self.demonstrate_registry_pattern()

            # Professional patterns
            self.demonstrate_error_recovery()

            # NEW: FlextResult v0.9.9+ methods for handlers
            self.demonstrate_from_callable_handlers()
            self.demonstrate_flow_through_handlers()
            self.demonstrate_lash_handlers()
            self.demonstrate_alt_handlers()
            self.demonstrate_value_or_call_handlers()

            # Deprecation warnings
            self.demonstrate_deprecated_patterns()

            summary: FlextTypes.Dict = {
                "demonstrations_completed": 11,
                "status": "completed",
                "infrastructure": {
                    "logger": type(self.logger).__name__,
                    "container": type(self.container).__name__,
                    "context": type(self.context).__name__,
                    "config": type(self.config).__name__,
                },
                "processing_features": {
                    "basic_handlers": True,
                    "handler_pipeline": True,
                    "strategy_pattern": True,
                    "registry_pattern": True,
                    "error_recovery": True,
                },
            }

            self.logger.info(
                "FlextProcessors demonstration completed successfully",
                extra=summary,
            )

            return FlextResult[FlextTypes.Dict].ok(summary)

        except Exception as e:
            error_msg = f"FlextProcessors demonstration failed: {e}"
            self.logger.exception(error_msg)
            return FlextResult[FlextTypes.Dict].fail(
                error_msg, error_code=FlextConstants.Errors.VALIDATION_ERROR
            )

    # ========== BASIC HANDLER PATTERNS ==========

    def demonstrate_basic_handlers(self) -> None:
        """Show basic handler creation and execution."""
        print("\n=== Basic Handler Patterns ===")

        class ValidationHandler(FlextProcessors.Implementation.BasicHandler):
            """Handler for data validation."""

            def __init__(self, name: str) -> None:
                """Initialize handler with logger."""
                super().__init__(name)
                self.logger = FlextLogger.create_module_logger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Validate and process the request."""
                self.logger.info(f"Validating request in {self.name}")
                data = cast("FlextTypes.Dict", request)

                # request is typed as FlextTypes.Dict
                email_value = cast("str | None", data.get("email", None))
                if not email_value:
                    return FlextResult[str].fail("Email required")

                # email_value is guaranteed to be str after cast and None check

                if "@" not in email_value:
                    return FlextResult[str].fail("Invalid email format")

                data["validated_at"] = str(time.time())
                return FlextResult[str].ok(f"Validation passed for {data['email']}")

        validator = ValidationHandler("EmailValidator")

        valid_request: FlextTypes.Dict = {
            "email": self._user["email"],
            "name": self._user["name"],
        }
        result: FlextResult[str] = validator.handle(valid_request)
        if result.is_success:
            print(f"✅ Validation passed: {result.unwrap()}")

        invalid_request: FlextTypes.Dict = {
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
                self.logger = FlextLogger.create_module_logger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Check authentication."""
                self.logger.info("Authenticating request")

                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                request_dict = cast("FlextTypes.Dict", request)
                token_value = cast("str | None", request_dict.get("token", None))
                if not token_value:
                    return FlextResult[str].fail("Authentication required")

                # token_value is guaranteed to be str after cast and None check

                if token_value != "valid-token":
                    return FlextResult[str].fail("Invalid token")

                request["authenticated"] = True
                return FlextResult[str].ok("Authentication successful")

        class AuthorizationHandler(FlextProcessors.Implementation.BasicHandler):
            """Authorize the request."""

            def __init__(self, name: str) -> None:
                """Initialize handler with logger."""
                super().__init__(name)
                self.logger = FlextLogger.create_module_logger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Check authorization."""
                self.logger.info("Authorizing request")

                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                request_dict = cast("FlextTypes.Dict", request)
                authenticated_value = cast(
                    "bool | None", request_dict.get("authenticated", None)
                )
                if not authenticated_value:
                    return FlextResult[str].fail("Not authenticated")

                role_value = cast("str | None", request_dict.get("role", None))
                if role_value != "REDACTED_LDAP_BIND_PASSWORD":
                    return FlextResult[str].fail("Insufficient permissions")

                request["authorized"] = True
                return FlextResult[str].ok("Authorization successful")

        class ProcessingHandler(FlextProcessors.Implementation.BasicHandler):
            """Process the authorized request."""

            def __init__(self, name: str) -> None:
                """Initialize handler with logger."""
                super().__init__(name)
                self.logger = FlextLogger.create_module_logger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Process the business logic."""
                self.logger.info("Processing request")

                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                request_dict = cast("FlextTypes.Dict", request)
                authorized_value = cast(
                    "bool | None", request_dict.get("authorized", None)
                )
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
        valid_request: FlextTypes.Dict = {
            "token": self._REDACTED_LDAP_BIND_PASSWORD_user["token"],
            "role": self._REDACTED_LDAP_BIND_PASSWORD_user["role"],
            "action": "delete_user",
            "user": self._REDACTED_LDAP_BIND_PASSWORD_user,
        }
        result = execute_pipeline(valid_request)
        if result.is_success:
            print(f"✅ Pipeline success: {result.unwrap()}")

        print("\n2. Request with invalid token:")
        invalid_token: FlextTypes.Dict = {
            "token": "invalid",
            "role": self._REDACTED_LDAP_BIND_PASSWORD_user["role"],
            "action": "delete_user",
            "user": self._REDACTED_LDAP_BIND_PASSWORD_user,
        }
        result = execute_pipeline(invalid_token)
        if result.is_failure:
            print(f"❌ Pipeline failed at auth: {result.error}")

        print("\n3. Request with insufficient permissions:")
        insufficient: FlextTypes.Dict = {
            "token": self._REDACTED_LDAP_BIND_PASSWORD_user["token"],
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
                super().__init__()
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
        amount = float(cast("str", self._order["total"]))

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
                super().__init__()
                self._handlers: dict[
                    str, FlextProcessors.Implementation.BasicHandler
                ] = {}
                self.logger = FlextLogger.create_module_logger(__name__)

            def register(
                self,
                name: str,
                handler: FlextProcessors.Implementation.BasicHandler,
            ) -> FlextResult[bool]:
                """Register a handler."""
                if name in self._handlers:
                    return FlextResult[bool].fail(f"Handler {name} already registered")

                self._handlers[name] = handler
                self.logger.info("Registered handler: %s", name)
                return FlextResult[bool].ok(True)

            def unregister(self, name: str) -> FlextResult[bool]:
                """Unregister a handler."""
                if name not in self._handlers:
                    return FlextResult[bool].fail(f"Handler {name} not found")

                del self._handlers[name]
                self.logger.info("Unregistered handler: %s", name)
                return FlextResult[bool].ok(True)

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

                request_dict = cast("FlextTypes.Dict", request)
                text_value = cast("str", request_dict.get("text", ""))
                return FlextResult[str].ok(f"Uppercase: {text_value.upper()}")

        class LowerCaseHandler(FlextProcessors.Implementation.BasicHandler):
            """Convert text to lowercase."""

            def handle(self, request: object) -> FlextResult[str]:
                """Process text."""
                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                request_dict = cast("FlextTypes.Dict", request)
                text_value = cast("str", request_dict.get("text", ""))
                return FlextResult[str].ok(f"Lowercase: {text_value.lower()}")

        class ReverseHandler(FlextProcessors.Implementation.BasicHandler):
            """Reverse text."""

            def handle(self, request: object) -> FlextResult[str]:
                """Process text."""
                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                request_dict = cast("FlextTypes.Dict", request)
                text_raw = request_dict.get("text", "")
                text_value: str = (
                    text_raw if isinstance(text_raw, str) else str(text_raw)
                )
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
                self.logger = FlextLogger.create_module_logger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Handle with retry logic."""
                self._attempt += 1
                _ = request  # Mark as used to avoid linting warning

                # Simulate intermittent failure
                if self._attempt < self._max_retries:
                    self.logger.warning(
                        f"Attempt {self._attempt} failed",
                        extra={"handler": self.name},
                    )
                    return FlextResult[str].fail(f"Attempt {self._attempt} failed")

                # Success on final attempt
                self.logger.info(
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
                self.logger = FlextLogger.create_module_logger(__name__)

            def handle(self, request: object) -> FlextResult[str]:
                """Handle with circuit breaker."""
                # Check if circuit is open
                if self._is_open:
                    return FlextResult[str].fail("Circuit breaker is open")

                # Convert object to FlextTypes.Dict for processing
                if not isinstance(request, dict):
                    return FlextResult[str].fail("Request must be a dictionary")

                # Simulate processing
                request_dict = cast("FlextTypes.Dict", request)
                force_fail_value = cast("bool", request_dict.get("force_fail", False))
                should_fail = bool(force_fail_value)

                if should_fail:
                    self._failure_count += 1
                    if self._failure_count >= self._threshold:
                        self._is_open = True
                        self.logger.error(
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
                self.logger.info("Circuit breaker reset")

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

    # ========== NEW FlextResult METHODS (v0.9.9+) ==========

    def demonstrate_from_callable_handlers(self) -> None:
        """Show from_callable for safe handler execution."""
        print("\n=== from_callable: Safe Handler Execution ===")

        def risky_handler(data: FlextTypes.Dict) -> str:
            """Handler that might raise exceptions."""
            if not data.get("user_id"):
                msg = "User ID required"
                raise ValueError(msg)
            return f"Processed user {data['user_id']}"

        # Safe handler execution without try/except
        result: FlextResult[str] = FlextResult[str].from_callable(
            lambda: risky_handler({"user_id": self._user["id"]}),
        )
        if result.is_success:
            print(f"✅ Handler success: {result.unwrap()}")

        # Failed execution captured as Result
        failed: FlextResult[str] = FlextResult[str].from_callable(
            lambda: risky_handler({}),
        )
        if failed.is_failure:
            print(f"❌ Handler failed: {failed.error}")

    def demonstrate_flow_through_handlers(self) -> None:
        """Show flow_through for handler pipeline composition."""
        print("\n=== flow_through: Handler Pipeline ===")

        def validate_handler(
            data: FlextTypes.Dict,
        ) -> FlextResult[FlextTypes.Dict]:
            """Validate request data."""
            if not data.get("email"):
                return FlextResult[FlextTypes.Dict].fail("Email required")
            return FlextResult[FlextTypes.Dict].ok(data)

        def authenticate_handler(
            data: FlextTypes.Dict,
        ) -> FlextResult[FlextTypes.Dict]:
            """Authenticate the request."""
            authenticated: FlextTypes.Dict = {**data, "authenticated": True}
            return FlextResult[FlextTypes.Dict].ok(authenticated)

        def authorize_handler(
            data: FlextTypes.Dict,
        ) -> FlextResult[FlextTypes.Dict]:
            """Authorize the request."""
            if data.get("role") != "REDACTED_LDAP_BIND_PASSWORD":
                return FlextResult[FlextTypes.Dict].fail("Insufficient permissions")
            authorized: FlextTypes.Dict = {**data, "authorized": True}
            return FlextResult[FlextTypes.Dict].ok(authorized)

        # Flow through handler pipeline
        request_data: FlextTypes.Dict = {
            "email": self._REDACTED_LDAP_BIND_PASSWORD_user["email"],
            "role": self._REDACTED_LDAP_BIND_PASSWORD_user["role"],
            "action": "delete",
        }
        result: FlextResult[FlextTypes.Dict] = (
            FlextResult[FlextTypes.Dict]
            .ok(request_data)
            .flow_through(
                validate_handler,
                authenticate_handler,
                authorize_handler,
            )
        )

        if result.is_success:
            final_data: FlextTypes.Dict = result.unwrap()
            print(f"✅ Pipeline complete: action={final_data.get('action')}")
            authenticated = bool(final_data.get("authenticated", False))
            authorized = bool(final_data.get("authorized", False))
            print(f"   Authenticated: {authenticated}")
            print(f"   Authorized: {authorized}")

    def demonstrate_lash_handlers(self) -> None:
        """Show lash for handler retry and fallback."""
        print("\n=== lash: Handler Retry with Fallback ===")

        attempt_count = {"count": 0}

        def primary_handler(data: FlextTypes.Dict) -> FlextResult[str]:
            """Try primary handler (might fail)."""
            attempt_count["count"] += 1
            if attempt_count["count"] < FlextConstants.Validation.RETRY_COUNT_MAX:
                return FlextResult[str].fail("Primary handler unavailable")
            return FlextResult[str].ok(f"Primary: Processed {data.get('action')}")

        def fallback_handler(error: str) -> FlextResult[str]:
            """Fallback handler on failure."""
            print(f"   Falling back after: {error}")
            return FlextResult[str].ok("Fallback: Request queued for retry")

        request: FlextTypes.Dict = {
            "action": "process_order",
            "order_id": self._order["order_id"],
        }

        # Try primary, fall back on failure
        result: FlextResult[str] = primary_handler(request).lash(fallback_handler)
        if result.is_success:
            print(f"✅ Handler result: {result.unwrap()}")

    def demonstrate_alt_handlers(self) -> None:
        """Show alt for handler selection fallback."""
        print("\n=== alt: Handler Selection Fallback ===")

        def get_specialized_handler(action: str) -> FlextResult[str]:
            """Try to get specialized handler."""
            handlers = {
                "payment": "PaymentHandler",
                "shipping": "ShippingHandler",
                "inventory": "InventoryHandler",
            }
            if action in handlers:
                return FlextResult[str].ok(handlers[action])
            return FlextResult[str].fail(f"No specialized handler for {action}")

        def get_generic_handler() -> FlextResult[str]:
            """Fallback to generic handler."""
            return FlextResult[str].ok("GenericActionHandler")

        # Try specialized, use generic as fallback
        action = "notification"
        handler: FlextResult[str] = get_specialized_handler(action).alt(
            get_generic_handler()
        )

        if handler.is_success:
            print(f"✅ Selected handler: {handler.unwrap()}")
            print(f"   For action: {action}")

    def demonstrate_value_or_call_handlers(self) -> None:
        """Show value_or_call for lazy handler defaults."""
        print("\n=== value_or_call: Lazy Handler Defaults ===")

        def create_default_handler_result() -> str:
            """Create default handler result (only called if needed)."""
            print("   Generating default handler result...")
            return "Default: Request acknowledged"

        # Success case - default not called
        handler_result = FlextResult[str].ok("Primary: Request processed successfully")
        response = handler_result.value_or_call(create_default_handler_result)
        print(f"✅ Got primary response: {response}")

        # Failure case - default called lazily
        failed_handler = FlextResult[str].fail("Handler error")
        default_response = failed_handler.value_or_call(create_default_handler_result)
        print(f"✅ Got default response: {default_response}")

    def demonstrate_flext_constants_processing(self) -> None:
        """Show FlextConstants integration with processing patterns."""
        print("\n=== FlextConstants.Processing Integration (Layer 1) ===")

        logger = FlextLogger.create_module_logger(__name__)

        # Processing timeout and batch constants
        print(f"  DEFAULT_TIMEOUT: {FlextConstants.Config.DEFAULT_TIMEOUT}s")
        print(f"  DEFAULT_MAX_WORKERS: {FlextConstants.Processing.DEFAULT_MAX_WORKERS}")
        print(f"  DEFAULT_BATCH_SIZE: {FlextConstants.Processing.DEFAULT_BATCH_SIZE}")
        print(
            f"  DEFAULT_RETRY_ATTEMPTS: {FlextConstants.Reliability.MAX_RETRY_ATTEMPTS}"
        )

        # Error codes for handler failures
        print(f"  VALIDATION_ERROR: {FlextConstants.Errors.VALIDATION_ERROR}")
        print(f"  TIMEOUT_ERROR: {FlextConstants.Errors.TIMEOUT_ERROR}")
        print(f"  NOT_FOUND_ERROR: {FlextConstants.Errors.NOT_FOUND_ERROR}")

        # Use constants in handler configuration
        handler_config: FlextTypes.Dict = {
            "timeout": FlextConstants.Config.DEFAULT_TIMEOUT,
            "max_workers": FlextConstants.Processing.DEFAULT_MAX_WORKERS,
            "batch_size": FlextConstants.Processing.DEFAULT_BATCH_SIZE,
            "retry_attempts": FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
        }

        logger.info(
            "Handler configuration established",
            extra={"config": handler_config, "pattern": "processing_handler"},
        )
        print("✅ Processing configuration constants demonstrated")

    def demonstrate_flext_exceptions_processing(self) -> None:
        """Show FlextExceptions integration with handler error handling."""
        print("\n=== FlextExceptions Integration (Layer 2) ===")

        logger = FlextLogger.create_module_logger(__name__)

        # Handler validation error
        try:
            handler_name = ""
            if not handler_name:
                error_message = "Handler name is required"
                raise FlextExceptions.ValidationError(
                    error_message,
                    field="handler_name",
                    value=handler_name,
                )
        except FlextExceptions.ValidationError as e:
            logger.exception(
                "Handler validation failed",
                extra={
                    "error_code": e.error_code,
                    "field": e.field,
                    "correlation_id": e.correlation_id,
                },
            )
            print(f"✅ ValidationError logged: {e.error_code}")
            print(f"   Field: {e.field}")

        # Handler not found
        try:
            error_message = "Handler not found in registry"
            raise FlextExceptions.NotFoundError(
                error_message,
                resource_type="handler",
                resource_id="DataTransformHandler",
            )
        except FlextExceptions.NotFoundError as e:
            logger.exception(
                "Handler not found",
                extra={
                    "error_code": e.error_code,
                    "resource_type": e.resource_type,
                    "resource_id": e.resource_id,
                    "correlation_id": e.correlation_id,
                },
            )
            print(f"✅ NotFoundError logged: {e.error_code}")
            print(f"   Resource: {e.resource_type}/{e.resource_id}")

        # Handler timeout error
        try:
            error_message = "Handler processing timeout"
            raise FlextExceptions.TimeoutError(
                error_message,
                timeout_seconds=FlextConstants.Config.DEFAULT_TIMEOUT,
                operation="process_request",
            )
        except FlextExceptions.TimeoutError as e:
            logger.exception(
                "Handler timeout",
                extra={
                    "error_code": e.error_code,
                    "timeout_seconds": e.timeout_seconds,
                    "operation": e.operation,
                    "correlation_id": e.correlation_id,
                },
            )
            print(f"✅ TimeoutError logged: {e.error_code}")
            print(f"   Timeout: {e.timeout_seconds}s for {e.operation}")

        # Handler configuration error
        try:
            error_message = "Handler pipeline configuration invalid: max_workers=-1"
            raise FlextExceptions.ConfigurationError(
                error_message,
                config_key="max_workers",
                config_source="handler_config.yaml",
            )
        except FlextExceptions.ConfigurationError as e:
            logger.exception(
                "Handler configuration error",
                extra={
                    "error_code": e.error_code,
                    "config_key": e.config_key,
                    "config_source": e.config_source,
                    "correlation_id": e.correlation_id,
                },
            )
            print(f"✅ ConfigurationError logged: {e.error_code}")
            print(f"   Config: {e.config_key} from {e.config_source}")

    def demonstrate_flext_runtime_processing(self) -> None:
        """Show FlextRuntime integration with processing defaults."""
        print("\n=== FlextRuntime Integration (Layer 0.5) ===")

        # FlextRuntime configuration defaults for processing
        print(f"  DEFAULT_TIMEOUT: {FlextConstants.Config.DEFAULT_TIMEOUT}")
        print(f"  DEFAULT_MAX_WORKERS: {FlextConstants.Processing.DEFAULT_MAX_WORKERS}")
        print(f"  DEFAULT_BATCH_SIZE: {FlextConstants.Processing.DEFAULT_BATCH_SIZE}")
        print(
            f"  DEFAULT_RETRY_ATTEMPTS: {FlextConstants.Reliability.MAX_RETRY_ATTEMPTS}"
        )
        print(f"  DEFAULT_PAGE_SIZE: {FlextConstants.Pagination.DEFAULT_PAGE_SIZE}")

        # Handler processing configuration
        processing_config: FlextTypes.Dict = {
            "timeout": FlextConstants.Config.DEFAULT_TIMEOUT,
            "max_workers": FlextConstants.Processing.DEFAULT_MAX_WORKERS,
            "batch_size": FlextConstants.Processing.DEFAULT_BATCH_SIZE,
            "retry_attempts": FlextConstants.Reliability.MAX_RETRY_ATTEMPTS,
            "page_size": FlextConstants.Pagination.DEFAULT_PAGE_SIZE,
        }

        print("✅ Processing configuration:")
        for key, value in processing_config.items():
            print(f"   {key}: {value}")

        # Type guards for handler input validation
        handler_input = {
            "handler_id": "data_processor",
            "handler_type": "TransformHandler",
        }

        handler_id = handler_input.get("handler_id", "")
        if FlextRuntime.is_valid_identifier(handler_id):
            print(f"✅ Valid handler ID: {handler_id}")

        handler_type = handler_input.get("handler_type", "")
        if FlextRuntime.is_valid_identifier(handler_type):
            print(f"✅ Valid handler type: {handler_type}")

        # Path validation for handler configuration files
        config_path = "/etc/flext/handlers.yaml"
        if FlextRuntime.is_valid_path(config_path):  # config_path is a string literal
            print(f"✅ Valid config path: {config_path}")

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

    # NEW: FlextResult v0.9.9+ methods for handlers
    service.demonstrate_from_callable_handlers()
    service.demonstrate_flow_through_handlers()
    service.demonstrate_lash_handlers()
    service.demonstrate_alt_handlers()
    service.demonstrate_value_or_call_handlers()

    # Foundation layer integration (NEW in Phase 2)
    service.demonstrate_flext_constants_processing()
    service.demonstrate_flext_exceptions_processing()
    service.demonstrate_flext_runtime_processing()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextProcessors patterns demonstrated!")
    print(
        "✨ Including new v0.9.9+ methods: from_callable, flow_through, lash, alt, value_or_call"
    )
    print(
        "🔧 Including foundation integration: FlextConstants processing, FlextRuntime (Layer 0.5), FlextExceptions (Layer 2)"
    )
    print("🎯 Next: See 08_*.py for additional advanced patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
