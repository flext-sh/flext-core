#!/usr/bin/env python3
"""01 - FlextResult Fundamentals: Complete Railway-Oriented Programming.

This example demonstrates the COMPLETE FlextResult[T] API - the foundation
for error handling across the entire FLEXT ecosystem. FlextResult provides
railway-oriented programming that eliminates exceptions in business logic.

Key Concepts Demonstrated:
- Factory methods: .ok() and .fail() with error codes
- Value extraction: .unwrap(), .unwrap_or(), .expect()
- Railway operations: .map(), .flat_map(), .filter()
- Error recovery: .recover(), .or_else()
- Advanced combinators: .tap(), .traverse()
- Collection operations: .sequence()
- Validation chaining: .chain_validations(), .validate_all()
- Context and logging: .with_context()
- Operators: >>, <<, @, /, %, &, |, ^

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from datetime import UTC, datetime
from typing import ClassVar, cast

from flext_core import (
    FlextConstants,
    FlextCore,
    FlextResult,
    FlextService,
    FlextTypes,
)


class DemoScenarios:
    """Lightweight scenario data declared inline for demonstration."""

    _DATASET: ClassVar[dict[str, object]] = {
        "users": [
            {
                "id": 1,
                "name": "Alice Example",
                "email": "alice@example.com",
                "age": 30,
            },
            {
                "id": 2,
                "name": "Bob Example",
                "email": "bob@example.com",
                "age": 28,
            },
            {
                "id": 3,
                "name": "Charlie Example",
                "email": "charlie@example.com",
                "age": 35,
            },
        ],
        "configs": {"app_name": "Flext Demo", "version": "1.0.0"},
    }
    _VALIDATION: ClassVar[FlextTypes.Dict] = {
        "valid_emails": ["user@example.com", "contact@flext.dev"],
        "invalid_emails": ["invalid", "missing-at-symbol"],
    }
    _REALISTIC: ClassVar[FlextTypes.Dict] = {
        "order": {
            "customer_id": "cust-123",
            "order_id": "order-456",
            "total": "59.98",
            "items": [
                {
                    "product_id": "prod-001",
                    "name": "Widget",
                    "price": "29.99",
                    "quantity": 1,
                },
                {
                    "product_id": "prod-002",
                    "name": "Gadget",
                    "price": "29.99",
                    "quantity": 1,
                },
            ],
        },
        "api_response": {
            "status": "ok",
            "processed_at": "2025-01-01T00:00:00Z",
        },
        "user_registration": {
            "user_id": "usr-789",
            "plan": "standard",
        },
    }
    _CONFIG: ClassVar[FlextTypes.Dict] = {
        "database_url": "sqlite:///:memory:",
        "api_timeout": 30,
        "retry": 3,
    }
    _PAYLOAD: ClassVar[FlextTypes.Dict] = {
        "event": "user_registered",
        "user_id": "usr-123",
        "metadata": {"source": "examples", "version": "1.0"},
    }

    @staticmethod
    def dataset() -> FlextTypes.Dict:
        """Get a copy of the demo dataset for testing and examples."""
        return deepcopy(DemoScenarios._DATASET)

    @staticmethod
    def validation_data() -> FlextTypes.Dict:
        """Get a copy of the demo validation data for testing and examples."""
        return deepcopy(DemoScenarios._VALIDATION)

    @staticmethod
    def realistic_data() -> FlextTypes.Dict:
        """Get a simple realistic dataset for aggregate demonstrations."""
        return deepcopy(DemoScenarios._REALISTIC)

    @staticmethod
    def config(**overrides: object) -> FlextTypes.Dict:
        """Get a copy of the demo config for testing and examples."""
        value = deepcopy(DemoScenarios._CONFIG)
        value.update(overrides)
        return value

    @staticmethod
    def user(**overrides: object) -> FlextTypes.Dict:
        """Get a demo user object with optional overrides."""
        users_list = cast("list[dict[str, object]]", DemoScenarios._DATASET["users"])
        user = deepcopy(users_list[0])
        user.update(overrides)
        return user

    @staticmethod
    def users(count: int = 5) -> list[FlextTypes.Dict]:
        """Get a list of demo users (default: 5 users)."""
        users_list = cast("list[dict[str, object]]", DemoScenarios._DATASET["users"])
        return [deepcopy(user) for user in users_list[:count]]

    @staticmethod
    def service_batch(logger_name: str = "example_batch") -> FlextTypes.Dict:
        """Create a demo service batch with logger, config, and metrics."""
        return {
            "logger": FlextCore.create_logger(logger_name),
            "config": DemoScenarios.config(),
            "metrics": {"requests": 0, "errors": 0},
        }

    @staticmethod
    def payload(**overrides: object) -> FlextTypes.Dict:
        """Get a demo payload with optional overrides."""
        payload = deepcopy(DemoScenarios._PAYLOAD)
        payload.update(overrides)
        return payload

    @staticmethod
    def metadata(
        *, source: str = "examples", tags: list[str] | None = None, **extra: object
    ) -> FlextTypes.Dict:
        """Generate metadata for scenarios with optional tags and extra data."""
        data: FlextTypes.Dict = {
            "source": source,
            "component": "flext_core",
            "tags": tags or ["scenario", "demo"],
        }
        data.update(extra)
        return data

    @staticmethod
    def result_success(data: object | None = None) -> FlextResult[object]:
        """Create a successful FlextResult for testing and examples."""
        return FlextResult[object].ok(data)

    @staticmethod
    def result_failure(message: str = "Scenario error") -> FlextResult[object]:
        """Create a failed FlextResult for testing and examples."""
        return FlextResult[object].fail(message)

    @staticmethod
    def user_result(success: bool = True) -> FlextResult[dict[str, object]]:
        """Get a demo user result (success or failure)."""
        user = cast("list[dict[str, object]]", DemoScenarios._DATASET["users"])[0]
        if success:
            return FlextResult[dict[str, object]].ok(user)
        return FlextResult[dict[str, object]].fail("User lookup failed")

    @staticmethod
    def error_scenario(error_type: str = "ValidationError") -> FlextTypes.Dict:
        """Get a demo error scenario dictionary."""
        return {
            "error_type": error_type,
            "error_code": f"{error_type.upper()}_001",
            "message": f"Example {error_type} scenario",
            "timestamp": "2025-01-01T00:00:00Z",
            "severity": "error",
        }


class ComprehensiveResultService(FlextService[FlextTypes.Dict]):
    """Service demonstrating ALL FlextResult patterns with FlextMixins.Service infrastructure.

    This service now inherits from FlextService to demonstrate:
    - Inherited container property (FlextCore.Container singleton)
    - Inherited logger property (FlextLogger with service context)
    - Inherited context property (FlextCore.Context for request tracking)
    - Inherited config property (FlextCore.Config with settings)
    - Inherited metrics property (FlextMetrics for observability)

    These inherited properties showcase the foundation infrastructure available
    to all services in the FLEXT ecosystem.
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextMixins.Service infrastructure.

        Note: No manual logger or container initialization needed!
        All infrastructure is inherited from FlextService base class:
        - self.logger: FlextLogger with service context
        - self.container: FlextCore.Container global singleton
        - self.context: FlextCore.Context for request tracking
        - self.config: FlextCore.Config with application settings
        - self.metrics: FlextMetrics for observability
        """
        super().__init__()
        self._scenarios = DemoScenarios()
        self._dataset: FlextTypes.Dict = self._scenarios.dataset()
        self._validation: FlextTypes.Dict = self._scenarios.validation_data()
        self._metadata: FlextTypes.Dict = self._scenarios.metadata(
            tags=["result", "demo"]
        )

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "ComprehensiveResultService initialized with inherited infrastructure",
            extra={
                "dataset_keys": list[object](self._dataset.keys()),
                "service_type": "FlextResult demonstration",
            },
        )

    def execute(self) -> FlextResult[FlextTypes.Dict]:
        """Execute all FlextResult demonstrations and return summary.

        This method satisfies the FlextService abstract interface while
        demonstrating all FlextResult capabilities as a comprehensive example.

        Returns:
            FlextResult containing demonstration summary with method counts

        """
        self.logger.info("Starting comprehensive FlextResult demonstration")

        try:
            # Run all demonstrations
            self.demonstrate_flext_runtime_integration()
            self.demonstrate_factory_methods()
            self.demonstrate_value_extraction()
            self.demonstrate_railway_operations()
            self.demonstrate_error_recovery()
            self.demonstrate_advanced_combinators()
            self.demonstrate_collection_operations()
            self.demonstrate_validation_chaining()
            self.demonstrate_operators()
            self.demonstrate_context_operations()
            self.demonstrate_flext_exceptions_integration()
            self.demonstrate_from_callable()
            self.demonstrate_flow_through()
            self.demonstrate_lash()
            self.demonstrate_alt()
            self.demonstrate_value_or_call()
            self.demonstrate_deprecated_patterns()

            summary: dict[str, object] = {
                "demonstrations_completed": 17,
                "methods_covered": [
                    "FlextCore.Runtime integration",
                    "factory methods",
                    "value extraction",
                    "railway operations",
                    "error recovery",
                    "advanced combinators",
                    "collection operations",
                    "validation chaining",
                    "operators",
                    "context operations",
                    "FlextCore.Exceptions integration",
                    "from_callable",
                    "flow_through",
                    "lash",
                    "alt",
                    "value_or_call",
                    "deprecated patterns",
                ],
                "infrastructure": {
                    "logger": type(self.logger).__name__,
                    "container": type(self.container).__name__,
                    "context": type(self.context).__name__,
                },
                "completed_at": datetime.now(UTC).isoformat(),
            }

            self.logger.info(
                "FlextResult demonstration completed successfully", extra=summary
            )

            return FlextResult[FlextTypes.Dict].ok(summary)

        except Exception as e:
            error_msg = f"Demonstration failed: {e}"
            self.logger.exception(error_msg)
            return FlextResult[FlextTypes.Dict].fail(
                error_msg, error_code="VALIDATION_ERROR"
            )

    # ========== BASIC OPERATIONS ==========

    def demonstrate_flext_runtime_integration(self) -> None:
        """Show FlextCore.Runtime (Layer 0.5) integration with FlextResult."""
        print("\n=== FlextCore.Runtime Integration (Layer 0.5) ===")

        # FlextCore.Runtime type guards with FlextResult
        email = "test@example.com"
        if FlextCore.Runtime.is_valid_email(email):
            result = FlextResult[str].ok(email)
            print(f"✅ Valid email via FlextCore.Runtime: {result.unwrap()}")
        else:
            result = FlextResult[str].fail(
                "Invalid email", error_code=FlextConstants.Errors.VALIDATION_ERROR
            )

        # JSON validation with FlextCore.Runtime
        json_str = '{"key": "value"}'
        if FlextCore.Runtime.is_valid_json(json_str):
            result = FlextResult[str].ok(json_str)
            print("✅ Valid JSON via FlextCore.Runtime: validated")

        # UUID validation
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        if FlextCore.Runtime.is_valid_uuid(uuid_str):
            result = FlextResult[str].ok(uuid_str)
            print(f"✅ Valid UUID via FlextCore.Runtime: {uuid_str[:8]}...")

        # Configuration defaults from FlextCore.Runtime
        timeout = FlextConstants.Network.DEFAULT_TIMEOUT
        print(f"✅ FlextConstants.Network.DEFAULT_TIMEOUT: {timeout}s")

        # Demonstrate inherited logger usage
        self.logger.debug(
            "FlextCore.Runtime integration demonstrated",
            extra={"validated": ["email", "json", "uuid"]},
        )

    def demonstrate_factory_methods(self) -> None:
        """Show all ways to create FlextResult instances."""
        print("\n=== Factory Methods ===")

        success = self._scenarios.result_success({"scenario": "factory"})
        print(f"✅ .ok(): {success}")

        # Using FlextConstants for error codes
        failure = FlextResult[object].fail(
            "Validation failed", error_code=FlextConstants.Errors.VALIDATION_ERROR
        )
        print(f"❌ .fail() with FlextConstants error code: {failure}")

        def risky_operation() -> int:
            return 1 // 0  # Will raise ZeroDivisionError

        from_exc = FlextResult[int].safe_call(risky_operation)
        print(f"🔥 .safe_call() for exceptions: {from_exc}")

        # Log factory method demonstration
        self.logger.info(
            "FlextResult factory methods demonstrated",
            extra={"methods": ["ok", "fail", "safe_call"]},
        )

    def demonstrate_value_extraction(self) -> None:
        """Show all ways to extract values from FlextResult."""
        print("\n=== Value Extraction ===")

        dataset = self._dataset
        users_list = cast("FlextTypes.List", dataset["users"])
        user_payload = cast("FlextTypes.Dict", users_list[0])
        success = FlextResult[FlextTypes.Dict].ok(user_payload)
        failure = self._scenarios.result_failure("error")

        print(f".unwrap() on success: {success.unwrap()['email']}")
        print(f".unwrap_or('default'): {failure.unwrap_or({'email': 'default'})}")
        print(f".value_or_none: {failure.value_or_none}")

        print(f".value property: {success.value['name']}")
        print(f".expect('Must have value'): {success.expect('Must have value')}")

    # ========== RAILWAY OPERATIONS ==========

    def demonstrate_railway_operations(self) -> None:
        """Core railway-oriented programming patterns."""
        print("\n=== Railway Operations ===")

        def validate_length(s: str) -> FlextResult[str]:
            if len(s) < FlextConstants.Validation.MIN_USERNAME_LENGTH:
                return FlextResult[str].fail("Too short")
            return FlextResult[str].ok(s)

        def to_upper(s: str) -> str:
            return s.upper()

        def add_prefix(s: str) -> FlextResult[str]:
            return FlextResult[str].ok(f"PREFIX_{s}")

        # Map: transform success value
        result = FlextResult[str].ok("test").map(to_upper)
        print(f".map(to_upper): {result.unwrap()}")

        # FlatMap: chain operations that return FlextResult
        result = (
            FlextResult[str].ok("hello").flat_map(validate_length).flat_map(add_prefix)
        )
        print(f".flat_map chain: {result.unwrap()}")

        # Filter: conditional success
        filtered_result: FlextResult[int] = (
            FlextResult[int]
            .ok(10)
            .filter(
                lambda x: x > FlextConstants.Validation.FILTER_THRESHOLD,
                "Too small",
            )
        )
        print(
            f".filter(>{FlextConstants.Validation.FILTER_THRESHOLD}): {filtered_result}"
        )

        # Using operators (syntactic sugar)
        result = FlextResult[str].ok("test") >> validate_length >> add_prefix
        print(f">> operator chain: {result}")

    # ========== ERROR RECOVERY ==========

    def demonstrate_error_recovery(self) -> None:
        """Show error recovery patterns."""
        print("\n=== Error Recovery ===")

        failure = FlextResult[str].fail("Initial error")

        # Recover: transform error to success
        recovered = failure.recover(lambda e: f"Recovered from: {e}")
        print(f".recover(): {recovered.unwrap()}")

        # OrElse: provide fallback value
        fallback = failure.or_else(FlextResult[str].ok("fallback"))
        print(f".or_else(): {fallback.unwrap()}")

    # ========== ADVANCED COMBINATORS ==========

    def demonstrate_advanced_combinators(self) -> None:
        """Advanced functional programming patterns."""
        print("\n=== Advanced Combinators ===")

        # Tap: side effects without changing value
        result = (
            FlextResult[int]
            .ok(42)
            .tap(lambda x: print(f"  Tapping success: {x}"))
            .map(lambda x: x * 2)
        )
        print(f".tap() result: {result.unwrap()}")

        # Traverse: map and sequence
        items = [1, 2, 3]

        def process(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        traversed = FlextResult.traverse(items, process)
        print(f".traverse(): {traversed.unwrap()}")

    # ========== COLLECTION OPERATIONS ==========

    def demonstrate_collection_operations(self) -> None:
        """Operations on collections of FlextResult instances."""
        print("\n=== Collection Operations ===")

        results: list[FlextResult[FlextTypes.Dict]] = [
            self._scenarios.user_result(success=True),
            self._scenarios.user_result(success=True),
            self._scenarios.user_result(success=True),
        ]

        sequenced = FlextResult.sequence(results)
        print(f".sequence(): {len(sequenced.unwrap())} successful users")

        results.append(self._scenarios.user_result(success=False))

    # ========== VALIDATION PATTERNS ==========

    def demonstrate_validation_chaining(self) -> None:
        """Complex validation scenarios."""
        print("\n=== Validation Chaining ===")

        validation_data = self._validation
        sample_email = cast("FlextTypes.List", validation_data["valid_emails"])[0]
        invalid_email = cast("FlextTypes.List", validation_data["invalid_emails"])[0]

        def validate_not_empty(value: object) -> FlextResult[str]:
            str_value = cast("str", value)
            if not str_value:
                return FlextResult[str].fail("Empty string")
            return FlextResult[str].ok(str_value)

        def validate_email(value: object) -> FlextResult[str]:
            str_value = cast("str", value)
            if "@" not in str_value:
                return FlextResult[str].fail("Invalid email")
            return FlextResult[str].ok(str_value)

        def validate_domain(value: object) -> FlextResult[str]:
            str_value = cast("str", value)
            if not str_value.endswith(".com"):
                return FlextResult[str].fail("Must be .com domain")
            return FlextResult[str].ok(str_value)

        result = (
            FlextResult[str]
            .ok(cast("str", sample_email))
            .flat_map(validate_not_empty)
            .flat_map(validate_email)
            .flat_map(validate_domain)
        )
        print(f".chain_validations(): {result}")

        def validate_not_empty_none(value: object) -> FlextResult[None]:
            str_value = cast("str", value)
            if not str_value:
                return FlextResult[None].fail("Empty string")
            return FlextResult[None].ok(None)

        def validate_email_none(value: object) -> FlextResult[None]:
            str_value = cast("str", value)
            if "@" not in str_value:
                return FlextResult[None].fail("Invalid email")
            return FlextResult[None].ok(None)

        def validate_domain_none(value: object) -> FlextResult[None]:
            str_value = cast("str", value)
            if not str_value.endswith(".com"):
                return FlextResult[None].fail("Must be .com domain")
            return FlextResult[None].ok(None)

        all_results = FlextResult.validate_all(
            invalid_email,
            validate_not_empty_none,
            validate_email_none,
            validate_domain_none,
        )
        print(f".validate_all(): {all_results}")

    # ========== MONADIC OPERATORS ==========

    def demonstrate_operators(self) -> None:
        """All operator overloads for ergonomic usage."""
        print("\n=== Operator Overloads ===")

        def double(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        def add_ten(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 10)

        # >> operator (flat_map)
        result = FlextResult[int].ok(5) >> double >> add_ten
        print(f">> (flat_map): 5 >> double >> add_ten = {result.unwrap()}")

        # << operator (map)
        def multiply_by_three(x: int) -> int:
            return x * 3

        mapped_result = FlextResult[int].ok(5) << multiply_by_three
        print(f"<< (map): 5 << (*3) = {mapped_result.unwrap()}")

        # NOTE: @ (matmul) and & (and) operators removed in optimization
        # They were over-engineering - use .map()/.flat_map() or manual tuples instead
        print(
            "⚠️  @ and & operators removed - use .map()/.flat_map() or manual patterns"
        )

        # | operator (or_else)
        failure = FlextResult[int].fail("error")
        fallback = FlextResult[int].ok(999)
        result = failure.or_else(fallback)
        print(f"| (or_else): failure | 999 = {result.unwrap()}")

    # ========== CONTEXT AND LOGGING ==========

    def demonstrate_context_operations(self) -> None:
        """Context enrichment and logging with inherited infrastructure."""
        print("\n=== Context and Logging ===")

        def risky_operation() -> FlextResult[int]:
            message = str(self._scenarios.error_scenario("ValidationError")["message"])
            return FlextResult[int].fail(message)

        result = risky_operation().with_context(
            lambda err: f"{self._metadata.get('component', 'component')} error: {err}",
        )
        print(f".with_context(): {result}")

        # Demonstrate context tracking with inherited context property
        print("\n✨ Inherited Infrastructure Demonstration:")
        print(f"   - Logger: {type(self.logger).__name__}")
        print(f"   - Container: {type(self.container).__name__}")
        print(f"   - Context: {type(self.context).__name__}")

    def demonstrate_flext_exceptions_integration(self) -> None:
        """Show FlextCore.Exceptions (Layer 2) structured error handling."""
        print("\n=== FlextCore.Exceptions Integration (Layer 2) ===")

        # ValidationError with field context
        try:
            error_message = "Invalid email format"
            raise FlextCore.Exceptions.ValidationError(
                error_message,
                field="email",
                value="invalid-email",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        except FlextCore.Exceptions.ValidationError as e:
            result = FlextResult[str].fail(
                e.message,
                error_code=e.error_code,
            )
            print(f"✅ ValidationError: {result.error_code} - {result.error}")
            print(f"   Field: {e.field}, Value: {e.value}")

        # ConfigurationError with config key tracking
        try:
            config_error_message = "Missing required configuration"
            raise FlextCore.Exceptions.ConfigurationError(
                config_error_message,
                config_key="DATABASE_URL",
                config_source="environment",
            )
        except FlextCore.Exceptions.ConfigurationError as e:
            result = FlextResult[str].fail(
                e.message,
                error_code=e.error_code,
            )
            print(f"✅ ConfigurationError: {e.error_code}")
            print(f"   Config Key: {e.config_key}, Source: {e.config_source}")

        # NotFoundError with resource details
        try:
            not_found_message = "User not found"
            raise FlextCore.Exceptions.NotFoundError(
                not_found_message,
                resource_type="user",
                resource_id="123",
            )
        except FlextCore.Exceptions.NotFoundError as e:
            print(f"✅ NotFoundError: {e.error_code}")
            print(f"   Resource: {e.resource_type}, ID: {e.resource_id}")

    # ========== NEW ADVANCED METHODS (v0.9.9+) ==========

    def demonstrate_from_callable(self) -> None:
        """Show from_callable pattern for safe exception handling."""
        print("\n=== from_callable(): Safe Exception Handling ===")

        # Safe division using from_callable
        def risky_division() -> float:
            return 10 / 0  # Will raise ZeroDivisionError

        result = FlextResult[float].from_callable(risky_division)
        print(
            f"✅ Caught exception: {result.error if result.is_failure else 'Unexpected success'}"
        )

        # Successful callable
        def safe_addition() -> int:
            return 5 + 10

        addition_result: FlextResult[int] = FlextResult[int].from_callable(
            safe_addition
        )
        print(f".from_callable(safe): {addition_result.unwrap()}")

        # With custom error code
        def parse_int_string() -> int:
            return int("not a number")

        parse_result: FlextResult[int] = FlextResult[int].from_callable(
            parse_int_string,
            error_code=FlextConstants.Errors.VALIDATION_ERROR,
        )
        print(f".from_callable(with error_code): {parse_result.error}")

    def demonstrate_flow_through(self) -> None:
        """Show pipeline composition with flow_through."""
        print("\n=== flow_through(): Pipeline Composition ===")

        def validate_positive(x: int) -> FlextResult[int]:
            if x <= 0:
                return FlextResult[int].fail("Must be positive")
            return FlextResult[int].ok(x)

        def add_ten(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x + 10)

        def multiply_by_two(x: int) -> FlextResult[int]:
            return FlextResult[int].ok(x * 2)

        # Success pipeline: 5 → validate → +10 → *2 = 30
        result = (
            FlextResult[int]
            .ok(5)
            .flow_through(
                validate_positive,
                add_ten,
                multiply_by_two,
            )
        )
        print(f"✅ Pipeline success: 5 → +10 → *2 = {result.unwrap()}")

        # Failure pipeline (stops at validation)
        result = (
            FlextResult[int]
            .ok(-5)
            .flow_through(
                validate_positive,
                add_ten,
                multiply_by_two,
            )
        )
        print(f"Pipeline failure: {result.error}")

    def demonstrate_lash(self) -> None:
        """Show error recovery with lash (opposite of flat_map)."""
        print("\n=== lash(): Error Recovery ===")

        # Primary operation that fails
        def try_primary_database() -> FlextResult[str]:
            return FlextResult[str].fail("Primary database unavailable")

        # Recovery function for errors
        def recover_to_cache(error: str) -> FlextResult[str]:
            print(f"  Recovering from error: {error}")
            return FlextResult[str].ok("Data from cache")

        result = try_primary_database().lash(recover_to_cache)
        print(f"✅ Recovered: {result.unwrap()}")

        # Success case - lash not called
        def try_successful_operation() -> FlextResult[str]:
            return FlextResult[str].ok("Primary success")

        result = try_successful_operation().lash(recover_to_cache)
        print(f"Success (no recovery): {result.unwrap()}")

    def demonstrate_alt(self) -> None:
        """Show fallback pattern with alt."""
        print("\n=== alt(): Fallback Pattern ===")

        # Primary fails, use fallback
        primary = FlextResult[str].fail("Primary service unavailable")
        fallback = FlextResult[str].ok("Fallback value")

        result = primary.alt(fallback)
        print(f"✅ Using fallback: {result.unwrap()}")

        # Primary succeeds, ignore fallback
        primary = FlextResult[str].ok("Primary value")
        fallback = FlextResult[str].ok("Fallback value")

        result = primary.alt(fallback)
        print(f"Primary success: {result.unwrap()}")

        # Chain multiple fallbacks
        first = FlextResult[str].fail("First failed")
        second = FlextResult[str].fail("Second failed")
        third = FlextResult[str].ok("Third succeeded")

        result = first.alt(second).alt(third)
        print(f"Fallback chain: {result.unwrap()}")

    def demonstrate_value_or_call(self) -> None:
        """Show lazy default evaluation with value_or_call."""
        print("\n=== value_or_call(): Lazy Defaults ===")

        expensive_called = False

        def expensive_default() -> str:
            nonlocal expensive_called
            expensive_called = True
            print("  Computing expensive default...")
            return "expensive computed default"

        # Success case - default NOT called (lazy evaluation)
        success = FlextResult[str].ok("success value")
        result = success.value_or_call(expensive_default)
        print(f"✅ Success: {result}, expensive_called={expensive_called}")

        # Failure case - default IS called
        expensive_called = False
        failure = FlextResult[str].fail("error occurred")
        result = failure.value_or_call(expensive_default)
        print(f"Failure: {result}, expensive_called={expensive_called}")

    # ========== DEPRECATED PATTERNS (WITH WARNINGS) ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated patterns with proper warnings."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Manual try/except (DEPRECATED)
        warnings.warn(
            "Manual try/except is DEPRECATED! Use FlextResult.safe_call() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (try/except):")
        print("try:")
        print("    result = risky_function()")
        print("except Exception as e:")
        print("    handle_error(e)")

        # NEW: FlextResult pattern
        print("\n✅ CORRECT WAY (FlextResult):")

        def risky_function() -> int:
            error_message = "Division by zero"
            raise ZeroDivisionError(error_message)  # Will raise

        result = FlextResult[int].safe_call(risky_function)
        print(f"FlextResult.safe_call(): {result}")

        # OLD: Multiple return types (DEPRECATED)
        warnings.warn(
            "Returning Optional[T] or Union[T, None] is DEPRECATED! Always return FlextResult[T].",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (Optional return):")
        print("def find_user(id: int) -> Optional[User]:")
        print("    return None  # or User")

        print("\n✅ CORRECT WAY (FlextResult):")
        print("def find_user(id: int) -> FlextResult[User]:")
        print("    return FlextResult[User].fail('Not found')")

        # OLD: Boolean success flags (DEPRECATED)
        warnings.warn(
            "Returning (bool, T) tuples is DEPRECATED! Use FlextResult[T].",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (success flag):")
        print("def process() -> tuple[bool, str]:")
        print("    return (False, 'error message')")

        print("\n✅ CORRECT WAY (FlextResult):")
        print("def process() -> FlextResult[str]:")
        print("    return FlextResult[str].fail('error message')")


def main() -> None:
    """Main entry point demonstrating all FlextResult capabilities."""
    service = ComprehensiveResultService()

    print("=" * 60)
    print("FlextResult COMPLETE API DEMONSTRATION")
    print("Foundation for 32+ FLEXT Ecosystem Projects")
    print(
        "With Layer 0.5-3 Integration (FlextCore.Runtime → FlextConstants → FlextProtocols → FlextCore.Exceptions)"
    )
    print("=" * 60)

    # Foundation layer integration (NEW)
    service.demonstrate_flext_runtime_integration()

    # Core patterns
    service.demonstrate_factory_methods()
    service.demonstrate_value_extraction()
    service.demonstrate_railway_operations()

    # Advanced patterns
    service.demonstrate_error_recovery()
    service.demonstrate_advanced_combinators()
    service.demonstrate_collection_operations()

    # Professional patterns
    service.demonstrate_validation_chaining()
    service.demonstrate_operators()
    service.demonstrate_context_operations()

    # Structured error handling (NEW)
    service.demonstrate_flext_exceptions_integration()

    # New advanced methods (v0.9.9+)
    service.demonstrate_from_callable()
    service.demonstrate_flow_through()
    service.demonstrate_lash()
    service.demonstrate_alt()
    service.demonstrate_value_or_call()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextResult methods demonstrated!")
    print(
        "✨ Including new v0.9.9+ methods: from_callable, flow_through, lash, alt, value_or_call"
    )
    print(
        "🎯 Layer Integration: FlextCore.Runtime (0.5) → FlextConstants (1) → FlextCore.Exceptions (2)"
    )
    print("🎯 Next: See 02_dependency_injection.py for FlextCore.Container patterns")
    print("=" * 60)


if __name__ == "__main__":
    main()
