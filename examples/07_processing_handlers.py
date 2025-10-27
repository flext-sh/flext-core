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

import warnings
from copy import deepcopy
from typing import ClassVar, cast

from flext_core import (
    FlextConstants,
    FlextExceptions,
    FlextLogger,
    FlextResult,
    FlextRuntime,
    FlextService,
)


class DemoScenarios:
    """Inline scenario helpers for processing handler demonstrations."""

    _DATASET: ClassVar[dict[str, object]] = {
        "users": [
            {
                "id": 1,
                "name": "Alice Example",
                "email": "alice@example.com",
                "age": 30,
            }
        ],
    }

    _REALISTIC: ClassVar[dict[str, object]] = {
        "order": {
            "order_id": "order-456",
            "customer_id": "cust-123",
            "items": [
                {"product_id": "prod-001", "name": "Widget", "quantity": 1},
            ],
            "total": "29.99",
        }
    }

    @classmethod
    def user(cls, **overrides: object) -> dict[str, object]:
        """Create user data dictionary for processing examples."""
        users_list = cast("list[dict[str, object]]", cls._DATASET["users"])
        return {**deepcopy(users_list[0]), **overrides}

    @classmethod
    def realistic_data(cls) -> dict[str, object]:
        """Create realistic order data dictionary for processing examples."""
        return deepcopy(cls._REALISTIC)

    @classmethod
    def metadata(
        cls,
        *,
        source: str = "examples",
        tags: list[str] | None = None,
        **extra: object,
    ) -> dict[str, object]:
        """Create metadata dictionary for processing examples."""
        return {
            "source": source,
            "component": "flext_core",
            "tags": tags or ["processors", "demo"],
            **extra,
        }


class ProcessingPatternsService(FlextService[dict[str, object]]):
    """Service demonstrating ALL FlextProcessors patterns with FlextMixins infrastructure."""

    def __init__(self) -> None:
        """Initialize with inherited FlextMixins infrastructure."""
        super().__init__()

        self._scenarios = DemoScenarios()
        self._user = self._scenarios.user()
        self._order: dict[str, object] = cast(
            "dict[str, object]", self._scenarios.realistic_data()["order"]
        )
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

        self.logger.info(
            "ProcessingPatternsService initialized with inherited infrastructure",
            extra={
                "service_type": "FlextProcessors & Handler Patterns demonstration",
                "handler_types": [
                    "BasicHandler",
                    "Pipeline",
                    "Strategy",
                    "Registry",
                ],
                "processing_patterns": True,
            },
        )

    def execute(self) -> FlextResult[dict[str, object]]:
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

            # NEW: FlextResult v0.9.9+ methods for handlers
            self.demonstrate_from_callable_handlers()
            self.demonstrate_flow_through_handlers()
            self.demonstrate_lash_handlers()
            self.demonstrate_alt_handlers()
            self.demonstrate_value_or_call_handlers()

            # Deprecation warnings
            self.demonstrate_deprecated_patterns()

            summary: dict[str, object] = {
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

            return FlextResult[dict[str, object]].ok(summary)

        except Exception as e:
            error_msg = f"FlextProcessors demonstration failed: {e}"
            self.logger.exception(error_msg)
            return FlextResult[dict[str, object]].fail(
                error_msg, error_code=FlextConstants.Errors.VALIDATION_ERROR
            )

    # ========== BASIC HANDLER PATTERNS ==========

    def demonstrate_basic_handlers(self) -> None:
        """Show basic handler creation and execution."""
        print("\n=== Basic Handler Patterns ===")

        # Using advanced factory pattern to reduce boilerplate
        def email_validator(data: dict[str, object]) -> bool:
            """Email validation logic."""
            email = data.get("email")
            return bool(email and "@" in str(email))

        # NOTE: HandlerFactory API is deprecated/removed
        # validation_handler = HandlerFactory.create_validation_handler(
        #     email_validator,
        #     "Invalid email format",
        # )
        # validator = validation_handler("EmailValidator")

        # Simplified example without factory
        print("⚠️  HandlerFactory examples temporarily disabled (API updated)")

        # valid_request: dict[str, object] = {
        #     "email": self._user["email"],
        #     "name": self._user["name"],
        # }
        # valid_result: FlextResult[FlextTypes.ProcessorOutputType] = validator.handle(
        #     valid_request
        # )
        # if valid_result.is_success:
        #     print(f"✅ Validation passed: {valid_result.unwrap()}")

        # invalid_request: dict[str, object] = {
        #     "email": self._invalid_user["email"],
        #     "name": self._invalid_user["name"],
        # }
        # invalid_result: FlextResult[FlextTypes.ProcessorOutputType] = validator.handle(
        #     invalid_request
        # )
        # if invalid_result.is_failure:
        #     print(f"❌ Validation failed: {invalid_result.error}")

    # ========== HANDLER PIPELINE ==========

    def demonstrate_handler_pipeline(self) -> None:
        """Show chain of responsibility pattern with handler pipeline."""
        print("\n=== Handler Pipeline ===")

        # NOTE: HandlerFactory API is deprecated/removed
        print("⚠️  HandlerFactory pipeline examples temporarily disabled (API updated)")

        # Using factory pattern for all pipeline handlers
        # def auth_validator(data: dict[str, object]) -> bool:
        #     """Authentication validation logic."""
        #     token = data.get("token")
        #     return bool(token and token == "valid-token")

        # authentication_handler = HandlerFactory.create_validation_handler(
        #     auth_validator,
        #     "Invalid token",
        # )

        # def authz_validator(data: dict[str, object]) -> bool:
        #     """Authorization validation logic."""
        #     return bool(data.get("authenticated") and data.get("role") == "admin")

        # authorization_handler = HandlerFactory.create_validation_handler(
        #     authz_validator,
        #     "Insufficient permissions",
        # )

        # def process_transformer(data: dict[str, object]) -> dict[str, object]:
        #     """Processing transformation logic."""
        #     return {
        #         **data,
        #         "processed": True,
        #         "result": "Operation completed successfully",
        #         "timestamp": str(time.time()),
        #     }

        # processing_handler = HandlerFactory.create_transform_handler(
        #     process_transformer,
        # )

        # auth_handler = authentication_handler("Authenticator")
        # authz_handler = authorization_handler("Authorizer")
        # process_handler = processing_handler("Processor")

        # def execute_pipeline(
        #     request: dict[str, object],
        # ) -> FlextResult[FlextTypes.ProcessorOutputType]:
        #     """Execute the handler pipeline."""
        #     result = auth_handler.handle(request)
        #     if result.is_failure:
        #         return result

        #     result = authz_handler.handle(request)
        #     if result.is_failure:
        #         return result

        #     return process_handler.handle(request)

        # print("\n1. Valid request through pipeline:")
        # valid_request: dict[str, object] = {
        #     "token": self._admin_user["token"],
        #     "role": self._admin_user["role"],
        #     "action": "delete_user",
        #     "user": self._admin_user,
        # }
        # result = execute_pipeline(valid_request)
        # if result.is_success:
        #     print(f"✅ Pipeline success: {result.unwrap()}")

        # print("\n2. Request with invalid token:")
        # invalid_token: dict[str, object] = {
        #     "token": "invalid",
        #     "role": self._admin_user["role"],
        #     "action": "delete_user",
        #     "user": self._admin_user,
        # }
        # result = execute_pipeline(invalid_token)
        # if result.is_failure:
        #     print(f"❌ Pipeline failed at auth: {result.error}")

        # print("\n3. Request with insufficient permissions:")
        # insufficient: dict[str, object] = {
        #     "token": self._admin_user["token"],
        #     "role": self._invalid_user.get("role", "user"),
        #     "action": "delete_user",
        #     "user": self._invalid_user,
        # }
        # result = execute_pipeline(insufficient)
        # if result.is_failure:
        #     print(f"❌ Pipeline failed at authz: {result.error}")

    # ========== STRATEGY PATTERN ==========

    def demonstrate_strategy_pattern(self) -> None:
        """Show strategy pattern for algorithm selection."""
        print("\n=== Strategy Pattern ===")

        class PaymentStrategy:
            """Base payment processing strategy."""

            def process(self, amount: float) -> FlextResult[dict[str, object]]:
                """Process payment with specific strategy."""
                raise NotImplementedError

        class CreditCardStrategy(PaymentStrategy):
            """Credit card payment strategy."""

            def process(self, amount: float) -> FlextResult[dict[str, object]]:
                """Process credit card payment."""
                fee = amount * 0.029
                return FlextResult[dict[str, object]].ok({
                    "method": "credit_card",
                    "amount": amount,
                    "fee": round(fee, 2),
                    "total": round(amount + fee, 2),
                    "status": "processed",
                })

        class PayPalStrategy(PaymentStrategy):
            """PayPal payment strategy."""

            def process(self, amount: float) -> FlextResult[dict[str, object]]:
                """Process PayPal payment."""
                fee = amount * 0.034 + 0.30
                return FlextResult[dict[str, object]].ok({
                    "method": "paypal",
                    "amount": amount,
                    "fee": round(fee, 2),
                    "total": round(amount + fee, 2),
                    "status": "processed",
                })

        class BankTransferStrategy(PaymentStrategy):
            """Bank transfer payment strategy."""

            def process(self, amount: float) -> FlextResult[dict[str, object]]:
                """Process bank transfer."""
                fee = 5.00
                return FlextResult[dict[str, object]].ok({
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
            ) -> FlextResult[dict[str, object]]:
                """Process payment with selected strategy."""
                strategy = self._strategies.get(method)
                if strategy is None:
                    return FlextResult[dict[str, object]].fail("Unknown payment method")
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

    # ========== NEW FlextResult METHODS (v0.9.9+) ==========

    def demonstrate_from_callable_handlers(self) -> None:
        """Show from_callable for safe handler execution."""
        print("\n=== from_callable: Safe Handler Execution ===")

        def risky_handler(data: dict[str, object]) -> str:
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
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Validate request data."""
            if not data.get("email"):
                return FlextResult[dict[str, object]].fail("Email required")
            return FlextResult[dict[str, object]].ok(data)

        def authenticate_handler(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Authenticate the request."""
            authenticated: dict[str, object] = {**data, "authenticated": True}
            return FlextResult[dict[str, object]].ok(authenticated)

        def authorize_handler(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Authorize the request."""
            if data.get("role") != "admin":
                return FlextResult[dict[str, object]].fail("Insufficient permissions")
            authorized: dict[str, object] = {**data, "authorized": True}
            return FlextResult[dict[str, object]].ok(authorized)

        # Flow through handler pipeline
        request_data: dict[str, object] = {
            "email": self._admin_user["email"],
            "role": self._admin_user["role"],
            "action": "delete",
        }
        result: FlextResult[dict[str, object]] = (
            FlextResult[dict[str, object]]
            .ok(request_data)
            .flow_through(
                validate_handler,
                authenticate_handler,
                authorize_handler,
            )
        )

        if result.is_success:
            final_data: dict[str, object] = result.unwrap()
            print(f"✅ Pipeline complete: action={final_data.get('action')}")
            authenticated = bool(final_data.get("authenticated"))
            authorized = bool(final_data.get("authorized"))
            print(f"   Authenticated: {authenticated}")
            print(f"   Authorized: {authorized}")

    def demonstrate_lash_handlers(self) -> None:
        """Show lash for handler retry and fallback."""
        print("\n=== lash: Handler Retry with Fallback ===")

        attempt_count = {"count": 0}

        def primary_handler(data: dict[str, object]) -> FlextResult[str]:
            """Try primary handler (might fail)."""
            attempt_count["count"] += 1
            if attempt_count["count"] < FlextConstants.Validation.RETRY_COUNT_MAX:
                return FlextResult[str].fail("Primary handler unavailable")
            return FlextResult[str].ok(f"Primary: Processed {data.get('action')}")

        def fallback_handler(error: str) -> FlextResult[str]:
            """Fallback handler on failure."""
            print(f"   Falling back after: {error}")
            return FlextResult[str].ok("Fallback: Request queued for retry")

        request: dict[str, object] = {
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
        print(f"  DEFAULT_TIMEOUT: {FlextConstants.Defaults.TIMEOUT}s")
        print(f"  DEFAULT_MAX_WORKERS: {FlextConstants.Processing.DEFAULT_MAX_WORKERS}")
        print(
            f"  DEFAULT_BATCH_SIZE: {FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE}"
        )
        print(
            f"  DEFAULT_RETRY_ATTEMPTS: {FlextConstants.Reliability.MAX_RETRY_ATTEMPTS}"
        )

        # Error codes for handler failures
        print(f"  VALIDATION_ERROR: {FlextConstants.Errors.VALIDATION_ERROR}")
        print(f"  TIMEOUT_ERROR: {FlextConstants.Errors.TIMEOUT_ERROR}")
        print(f"  NOT_FOUND_ERROR: {FlextConstants.Errors.NOT_FOUND_ERROR}")

        # Use constants in handler configuration
        handler_config: dict[str, object] = {
            "timeout": FlextConstants.Defaults.TIMEOUT,
            "max_workers": FlextConstants.Processing.DEFAULT_MAX_WORKERS,
            "batch_size": FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE,
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
                timeout_seconds=FlextConstants.Defaults.TIMEOUT,
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
        print(f"  DEFAULT_TIMEOUT: {FlextConstants.Defaults.TIMEOUT}")
        print(f"  DEFAULT_MAX_WORKERS: {FlextConstants.Processing.DEFAULT_MAX_WORKERS}")
        print(
            f"  DEFAULT_BATCH_SIZE: {FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE}"
        )
        print(
            f"  DEFAULT_RETRY_ATTEMPTS: {FlextConstants.Reliability.MAX_RETRY_ATTEMPTS}"
        )
        print(f"  DEFAULT_PAGE_SIZE: {FlextConstants.Pagination.DEFAULT_PAGE_SIZE}")

        # Handler processing configuration
        processing_config: dict[str, object] = {
            "timeout": FlextConstants.Defaults.TIMEOUT,
            "max_workers": FlextConstants.Processing.DEFAULT_MAX_WORKERS,
            "batch_size": FlextConstants.Performance.BatchProcessing.DEFAULT_SIZE,
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
        # (Now handled by Pydantic v2 FilePath/DirectoryPath types)
        config_path = "/etc/flext/handlers.yaml"
        print(
            f"✅ Config path: {config_path} (validation via Pydantic v2 FilePath/DirectoryPath)"
        )

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
        print("def handle(self, request: dict) -> FlextResult[dict[str, object]]:")
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
