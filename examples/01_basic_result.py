"""FlextResult fundamentals example with single class structure.

This module demonstrates the complete FlextResult API with railway-oriented programming.
Scope: Factory methods, value extraction, railway operations, error recovery, combinators, validation, operators, context.

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
    FlextExceptions,
    FlextLogger,
    FlextResult,
    FlextRuntime,
    FlextService,
    FlextTypes,
)


class Example01BasicResult:
    """FlextResult fundamentals example.

    This module demonstrates the complete FlextResult API with railway-oriented programming.
    Scope: Factory methods, value extraction, railway operations, error recovery, combinators, validation, operators, context.
    """

    class DemoScenarios:
        """Lightweight scenario data declared inline for demonstration."""

        DATASET: ClassVar[dict[str, FlextTypes.GeneralValueType]] = {
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
            ],
            "configs": {"app_name": "Flext Demo", "version": "1.0.0"},
        }
        VALIDATION: ClassVar[dict[str, FlextTypes.GeneralValueType]] = {
            "valid_emails": ["user@example.com", "contact@flext.dev"],
            "invalid_emails": ["invalid", "missing-at-symbol"],
        }
        REALISTIC: ClassVar[dict[str, FlextTypes.GeneralValueType]] = {
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
        CONFIG: ClassVar[dict[str, FlextTypes.GeneralValueType]] = {
            "database_url": "sqlite:///:memory:",
            "api_timeout": 30,
            "retry": 3,
        }
        PAYLOAD: ClassVar[dict[str, FlextTypes.GeneralValueType]] = {
            "event": "user_registered",
            "user_id": "usr-123",
            "metadata": {"source": "examples", "version": "1.0"},
        }

        @staticmethod
        def dataset() -> dict[str, FlextTypes.GeneralValueType]:
            """Get a copy of the demo dataset for testing and examples."""
            return deepcopy(Example01BasicResult.DemoScenarios.DATASET)

        @staticmethod
        def validation_data() -> dict[str, FlextTypes.GeneralValueType]:
            """Get a copy of the demo validation data for testing and examples."""
            return deepcopy(Example01BasicResult.DemoScenarios.VALIDATION)

        @staticmethod
        def realistic_data() -> dict[str, FlextTypes.GeneralValueType]:
            """Get a simple realistic dataset for aggregate demonstrations."""
            return deepcopy(Example01BasicResult.DemoScenarios.REALISTIC)

        @staticmethod
        def config(
            **overrides: FlextTypes.GeneralValueType,
        ) -> dict[str, FlextTypes.GeneralValueType]:
            """Get a copy of the demo config for testing and examples."""
            value = deepcopy(Example01BasicResult.DemoScenarios.CONFIG)
            value.update(overrides)
            return value

        @staticmethod
        def user(
            **overrides: FlextTypes.GeneralValueType,
        ) -> dict[str, FlextTypes.GeneralValueType]:
            """Get a demo user object with optional overrides."""
            users_list = cast(
                "list[dict[str, FlextTypes.GeneralValueType]]",
                Example01BasicResult.DemoScenarios.DATASET["users"],
            )
            user = deepcopy(users_list[0])
            user.update(overrides)
            return user

        @staticmethod
        def users(count: int = 5) -> list[dict[str, FlextTypes.GeneralValueType]]:
            """Get a list of demo users (default: 5 users)."""
            users_list = cast(
                "list[dict[str, FlextTypes.GeneralValueType]]",
                Example01BasicResult.DemoScenarios.DATASET["users"],
            )
            return [deepcopy(user) for user in users_list[:count]]

        @staticmethod
        def service_batch(
            logger_name: str = "example_batch",
        ) -> dict[str, FlextTypes.GeneralValueType | FlextLogger]:
            """Create a demo service batch with logger, config, and metrics."""
            return {
                "logger": FlextLogger.create_module_logger(logger_name),
                "config": Example01BasicResult.DemoScenarios.config(),
                "metrics": {"requests": 0, "errors": 0},
            }

        @staticmethod
        def payload(
            **overrides: FlextTypes.GeneralValueType,
        ) -> dict[str, FlextTypes.GeneralValueType]:
            """Get a demo payload with optional overrides."""
            payload = deepcopy(Example01BasicResult.DemoScenarios.PAYLOAD)
            payload.update(overrides)
            return payload

        @staticmethod
        def metadata(
            *,
            source: str = "examples",
            tags: list[str] | None = None,
            **extra: FlextTypes.GeneralValueType,
        ) -> dict[str, FlextTypes.GeneralValueType]:
            """Generate metadata for scenarios with optional tags and extra data."""
            data: dict[str, FlextTypes.GeneralValueType] = {
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
        def user_result(
            success: bool = True,
        ) -> FlextResult[dict[str, FlextTypes.GeneralValueType]]:
            """Get a demo user result (success or failure)."""
            user = cast(
                "list[dict[str, FlextTypes.GeneralValueType]]",
                Example01BasicResult.DemoScenarios.DATASET["users"],
            )[0]
            if success:
                return FlextResult[dict[str, FlextTypes.GeneralValueType]].ok(user)
            return FlextResult[dict[str, FlextTypes.GeneralValueType]].fail(
                "User lookup failed"
            )

        @staticmethod
        def error_scenario(
            error_type: str = "ValidationError",
        ) -> dict[str, FlextTypes.GeneralValueType]:
            """Get a demo error scenario dictionary."""
            return {
                "error_type": error_type,
                "error_code": f"{error_type.upper()}_001",
                "message": f"Example {error_type} scenario",
                "timestamp": "2025-01-01T00:00:00Z",
                "severity": "error",
            }

    class ComprehensiveResultService(FlextService[FlextTypes.GeneralValueType]):
        """Service demonstrating ALL FlextResult patterns with FlextMixins infrastructure.

        This service now inherits from FlextService to demonstrate:
        - Inherited container property (FlextContainer singleton)
        - Inherited logger property (FlextLogger with service context)
        - Inherited context property (FlextContext for request tracking)
        - Inherited config property (FlextConfig with settings)
        - Inherited metrics property (FlextMetrics for observability)

        These inherited properties showcase the foundation infrastructure available
        to all services in the FLEXT ecosystem.
        """

        def __init__(self) -> None:
            """Initialize with inherited FlextMixins infrastructure.

            Note: No manual logger or container initialization needed!
            All infrastructure is inherited from FlextService base class:
            - self.logger: FlextLogger with service context
            - self.container: FlextContainer global singleton
            - self.context: FlextContext for request tracking
            - self.config: FlextConfig with application settings
            - self.metrics: FlextMetrics for observability
            """
            super().__init__()
            self._scenarios = Example01BasicResult.DemoScenarios()
            self._dataset: dict[str, FlextTypes.GeneralValueType] = (
                self._scenarios.dataset()
            )
            self._validation: dict[str, FlextTypes.GeneralValueType] = (
                self._scenarios.validation_data()
            )
            self._metadata: dict[str, FlextTypes.GeneralValueType] = (
                self._scenarios.metadata(
                    tags=["result", "demo"],
                )
            )

            # Demonstrate inherited logger (no manual instantiation needed!)
            self.logger.info(
                "ComprehensiveResultService initialized with inherited infrastructure",
                extra={
                    "dataset_keys": list(self._dataset.keys()),
                    "service_type": "FlextResult demonstration",
                },
            )

        def execute(
            self,
            **_kwargs: object,
        ) -> FlextResult[dict[str, FlextTypes.GeneralValueType]]:
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

                summary: dict[str, FlextTypes.GeneralValueType] = {
                    "demonstrations_completed": 17,
                    "methods_covered": [
                        "FlextRuntime integration",
                        "factory methods",
                        "value extraction",
                        "railway operations",
                        "error recovery",
                        "advanced combinators",
                        "collection operations",
                        "validation chaining",
                        "operators",
                        "context operations",
                        "FlextExceptions integration",
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

                # Convert summary to GeneralValueType for type safety
                normalized_summary = FlextRuntime.normalize_to_general_value(summary)
                # Type narrowing: ensure normalized_summary is a dict for unpacking
                if isinstance(normalized_summary, dict):
                    normalized_summary_dict: dict[str, FlextTypes.GeneralValueType] = {
                        str(k): v
                        for k, v in normalized_summary.items()
                        if isinstance(k, str)
                    }
                    # Pass summary as context - avoid unpacking to prevent parameter conflicts
                    self.logger.info(
                        "FlextResult demonstration completed successfully",
                        summary=normalized_summary_dict,
                    )
                    return FlextResult[dict[str, FlextTypes.GeneralValueType]].ok(
                        normalized_summary_dict,
                    )
                normalized_summary_dict: dict[str, FlextTypes.GeneralValueType] = {
                    "summary": normalized_summary,
                }
                self.logger.info(
                    "FlextResult demonstration completed successfully",
                    summary=normalized_summary,
                )
                return FlextResult[dict[str, FlextTypes.GeneralValueType]].ok(
                    normalized_summary_dict,
                )

            except Exception as e:
                error_msg = f"Demonstration failed: {e}"
                self.logger.exception(error_msg)
                return FlextResult[dict[str, FlextTypes.GeneralValueType]].fail(
                    error_msg,
                    error_code="VALIDATION_ERROR",
                )

        # ========== BASIC OPERATIONS ==========

        def demonstrate_flext_runtime_integration(self) -> None:
            """Show FlextRuntime (Layer 0.5) integration with FlextResult."""
            print("\n=== FlextRuntime Integration (Layer 0.5) ===")

            # Email validation via Pydantic v2 EmailStr with FlextResult
            email = "test@example.com"
            result = FlextResult[str].ok(email)
            print(f"✅ Valid email (Pydantic v2): {result.unwrap()}")

            # JSON validation with FlextRuntime
            json_str = '{"key": "value"}'
            if FlextRuntime.is_valid_json(json_str):
                result = FlextResult[str].ok(json_str)
                print("✅ Valid JSON via FlextRuntime: validated")

            # UUID validation via Pydantic v2 UUID4 type
            uuid_str = "550e8400-e29b-41d4-a716-446655440000"
            result = FlextResult[str].ok(uuid_str)
            print(f"✅ Valid UUID (Pydantic v2): {uuid_str[:8]}...")

            # Configuration defaults from FlextRuntime
            timeout = FlextConstants.Network.DEFAULT_TIMEOUT
            print(f"✅ FlextConstants.Network.DEFAULT_TIMEOUT: {timeout}s")

            # Demonstrate inherited logger usage
            self.logger.debug(
                "FlextRuntime integration demonstrated",
                extra={"validated": ["email", "json", "uuid"]},
            )

        def demonstrate_factory_methods(self) -> None:
            """Show all ways to create FlextResult instances."""
            print("\n=== Factory Methods ===")

            success = self._scenarios.result_success({"scenario": "factory"})
            print(f"✅ .ok(): {success}")

            # Using FlextConstants for error codes
            failure = FlextResult[object].fail(
                "Validation failed",
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
            print(f"❌ .fail() with FlextConstants error code: {failure}")

            def risky_operation() -> int:
                return 1 // 0  # Will raise ZeroDivisionError

            # NEW: Using create_from_callable instead of manual try/except
            from_callable_result = FlextResult[int].create_from_callable(
                risky_operation,
            )
            print(f"🔥 .create_from_callable() for exceptions: {from_callable_result}")

            # Log factory method demonstration
            self.logger.info(
                "FlextResult factory methods demonstrated",
                extra={"methods": ["ok", "fail", "from_callable"]},
            )

        def demonstrate_value_extraction(self) -> None:
            """Show all ways to extract values from FlextResult."""
            print("\n=== Value Extraction ===")

            dataset = self._dataset
            users_list = cast("list[FlextTypes.GeneralValueType]", dataset["users"])
            user_payload = cast("dict[str, FlextTypes.GeneralValueType]", users_list[0])
            success = FlextResult[dict[str, FlextTypes.GeneralValueType]].ok(
                user_payload
            )
            failure = self._scenarios.result_failure("error")

            print(f".unwrap() on success: {success.unwrap()['email']}")
            print(f".unwrap_or('default'): {failure.unwrap_or({'email': 'default'})}")
            print(f".unwrap_or(None): {failure.unwrap_or(None)}")

            print(f".value property: {success.value['name']}")
            print(f".unwrap() with message: {success.unwrap()}")

            # Using unwrap_or for defaults
            result = failure.unwrap_or({
                "email": "default@example.com",
                "name": "Default User",
            })
            print(
                f".unwrap_or() default: {cast('dict[str, FlextTypes.GeneralValueType]', result)['name']}"
            )

        # ========== RAILWAY OPERATIONS ==========

        def demonstrate_railway_operations(self) -> None:
            """Core railway-oriented programming patterns."""
            print("\n=== Railway Operations ===")

            def validate_length(s: object) -> FlextResult[object]:
                str_val = str(s)
                if len(str_val) < FlextConstants.Validation.MIN_USERNAME_LENGTH:
                    return FlextResult[object].fail("Too short")
                return FlextResult[object].ok(str_val)

            def to_upper(s: str) -> str:
                return s.upper()

            def add_prefix(s: object) -> FlextResult[object]:
                return FlextResult[object].ok(f"PREFIX_{s}")

            # Map: transform success value
            result = FlextResult[str].ok("test").map(to_upper)
            print(f".map(to_upper): {result.unwrap()}")

            # FlatMap: chain operations that return FlextResult
            result = (
                FlextResult[str]
                .ok("hello")
                .flat_map(validate_length)
                .flat_map(add_prefix)
            )
            print(f".flat_map chain: {result.unwrap()}")

            # NEW: flow_through for clean pipeline composition
            def to_upper_result(s: object) -> FlextResult[object]:
                str_val = str(s)
                return FlextResult[object].ok(str_val.upper())

            def double_length(s: object) -> FlextResult[object]:
                str_val = str(s)
                return FlextResult[object].ok(str_val * 2)

            result = (
                FlextResult[str]
                .ok("hello")
                .flow_through(
                    validate_length,
                    to_upper_result,
                    double_length,
                    add_prefix,
                )
            )
            print(f".flow_through pipeline: {result.unwrap()}")

            # Filter: conditional success
            filtered_result: FlextResult[int] = (
                FlextResult[int]
                .ok(10)
                .filter(
                    lambda x: x > FlextConstants.Validation.FILTER_THRESHOLD,
                )
            )
            print(
                f".filter(>{FlextConstants.Validation.FILTER_THRESHOLD}): {filtered_result}",
            )

            # Using flat_map for chaining
            result = (
                FlextResult[str]
                .ok("test")
                .flat_map(validate_length)
                .flat_map(add_prefix)
            )
            print(f"flat_map chain: {result}")

        # ========== ERROR RECOVERY ==========

        def demonstrate_error_recovery(self) -> None:
            """Show error recovery patterns."""
            print("\n=== Error Recovery ===")

            failure = FlextResult[str].fail("Initial error")

            # alt - transform error string to success value
            def recover_from_error(error: str) -> str:
                return f"Recovered from: {error}"

            recovered = failure.alt(recover_from_error)
            print(f".alt() (error transform): {recovered.unwrap()}")

            # lash - provide fallback result
            def provide_fallback(_error: str) -> FlextResult[str]:
                return FlextResult[str].ok("fallback")

            fallback = failure.lash(provide_fallback)
            print(f".lash() (fallback): {fallback.unwrap()}")

            # NEW: lash - apply function to error (opposite of flat_map)
            def log_and_recover(error: str) -> FlextResult[str]:
                print(f"  Logging error: {error}")
                return FlextResult[str].ok("Recovered via lash")

            lashed = failure.lash(log_and_recover)
            print(f".lash(): {lashed.unwrap()}")

            # Success case - lash not applied
            success = FlextResult[str].ok("Success value")
            success_lashed = success.lash(log_and_recover)
            print(f"Success with .lash() (unchanged): {success_lashed.unwrap()}")

        # ========== ADVANCED COMBINATORS ==========

        def demonstrate_advanced_combinators(self) -> None:
            """Advanced functional programming patterns."""
            print("\n=== Advanced Combinators ===")

            # Side effects using map (tap equivalent)
            def tap_effect(x: object) -> object:
                int_val = int(x) if isinstance(x, (int, float, str)) else 0
                print(f"  Side effect: {int_val}")
                return int_val

            result = (
                FlextResult[int]
                .ok(42)
                .map(tap_effect)
                .map(lambda x: int(x) * 2 if isinstance(x, (int, float, str)) else 0)
            )
            print(f"Side effect result: {result.unwrap()}")

            # Traverse: map and sequence
            items = [1, 2, 3]

            def process(x: object) -> FlextResult[int]:
                int_val = int(x) if isinstance(x, (int, float, str)) else 0
                return FlextResult[int].ok(int_val * 2)

            traversed = FlextResult.traverse(items, process)
            print(f".traverse(): {traversed.unwrap()}")

        # ========== COLLECTION OPERATIONS ==========

        def demonstrate_collection_operations(self) -> None:
            """Operations on collections of FlextResult instances."""
            print("\n=== Collection Operations ===")

            results: list[FlextResult[dict[str, FlextTypes.GeneralValueType]]] = [
                self._scenarios.user_result(success=True),
                self._scenarios.user_result(success=True),
                self._scenarios.user_result(success=True),
            ]

            # Use traverse to sequence results
            items = [r.value for r in results if r.is_success]
            sequenced = FlextResult.traverse(
                items,
                lambda item: FlextResult[dict[str, FlextTypes.GeneralValueType]].ok(
                    cast("dict[str, FlextTypes.GeneralValueType]", item),
                ),
            )
            print(
                f".traverse() (sequenced): {len(sequenced.unwrap())} successful users",
            )

        # ========== VALIDATION PATTERNS ==========

        def demonstrate_validation_chaining(self) -> None:
            """Complex validation scenarios."""
            print("\n=== Validation Chaining ===")

            validation_data = self._validation
            sample_email = cast("list[object]", validation_data["valid_emails"])[0]
            invalid_email = cast("list[object]", validation_data["invalid_emails"])[0]

            def validate_not_empty(value: object) -> FlextResult[object]:
                str_value = str(value)
                if not str_value:
                    return FlextResult[object].fail("Empty string")
                return FlextResult[object].ok(str_value)

            def validate_email(value: object) -> FlextResult[object]:
                str_value = str(value)
                if "@" not in str_value:
                    return FlextResult[object].fail("Invalid email")
                return FlextResult[object].ok(str_value)

            def validate_domain(value: object) -> FlextResult[object]:
                str_value = str(value)
                if not str_value.endswith(".com"):
                    return FlextResult[object].fail("Must be .com domain")
                return FlextResult[object].ok(str_value)

            result = (
                FlextResult[str]
                .ok(cast("str", sample_email))
                .flat_map(validate_not_empty)
                .flat_map(validate_email)
                .flat_map(validate_domain)
            )
            print(f".chain_validations(): {result}")

            def validate_not_empty_bool(value: object) -> FlextResult[bool]:
                str_value = str(value)
                if not str_value:
                    return FlextResult[bool].fail("Empty string")
                return FlextResult[bool].ok(True)

            def validate_email_bool(value: object) -> FlextResult[bool]:
                str_value = str(value)
                if "@" not in str_value:
                    return FlextResult[bool].fail("Invalid email")
                return FlextResult[bool].ok(True)

            def validate_domain_bool(value: object) -> FlextResult[bool]:
                str_value = str(value)
                if not str_value.endswith(".com"):
                    return FlextResult[bool].fail("Must be .com domain")
                return FlextResult[bool].ok(True)

            # Use traverse for multiple validations
            validators = [
                validate_not_empty_bool,
                validate_email_bool,
                validate_domain_bool,
            ]
            validation_results = [validator(invalid_email) for validator in validators]
            all_results = FlextResult.traverse(
                validation_results,
                lambda r: r
                if isinstance(r, FlextResult)
                else FlextResult[bool].ok(True),
            )
            print(f"Multiple validations: {all_results}")

        # ========== MONADIC OPERATORS ==========

        def demonstrate_operators(self) -> None:
            """All operator overloads for ergonomic usage."""
            print("\n=== Operator Overloads ===")

            def double(x: object) -> FlextResult[object]:
                int_val = int(x) if isinstance(x, (int, float, str)) else 0
                return FlextResult[object].ok(int_val * 2)

            def add_ten(x: object) -> FlextResult[object]:
                int_val = int(x) if isinstance(x, (int, float, str)) else 0
                return FlextResult[object].ok(int_val + 10)

            # Use flat_map for chaining (>> operator not available)
            result = FlextResult[int].ok(5).flat_map(double).flat_map(add_ten)
            print(f"flat_map chain: 5 → double → add_ten = {result.unwrap()}")

            # << operator (map)
            def multiply_by_three(x: object) -> object:
                int_val = int(x) if isinstance(x, (int, float, str)) else 0
                return int_val * 3

            mapped_result = FlextResult[int].ok(5).map(multiply_by_three)
            print(f"map (*3): 5 → (*3) = {mapped_result.unwrap()}")

            # NOTE: @ (matmul) and & (and) operators removed in optimization
            # They were over-engineering - use .map()/.flat_map() or manual tuples instead
            print(
                "⚠️  @ and & operators removed - use .map()/.flat_map() or manual patterns",
            )

            # | operator (unwrap_or via __or__)
            failure = FlextResult[int].fail("error")
            result_value = failure | 999
            print(f"| (unwrap_or): failure | 999 = {result_value}")

        # ========== CONTEXT AND LOGGING ==========

        def demonstrate_context_operations(self) -> None:
            """Context enrichment and logging with inherited infrastructure."""
            print("\n=== Context and Logging ===")

            def risky_operation() -> FlextResult[int]:
                message = str(
                    self._scenarios.error_scenario("ValidationError")["message"],
                )
                return FlextResult[int].fail(message)

            result = risky_operation()
            error_with_context = (
                f"{self._metadata.get('component', 'component')} error: {result.error}"
            )
            print(f"Error with context: {error_with_context}")

            # Demonstrate context tracking with inherited context property
            print("\n✨ Inherited Infrastructure Demonstration:")
            print(f"   - Logger: {type(self.logger).__name__}")
            print(f"   - Container: {type(self.container).__name__}")
            print(f"   - Context: {type(self.context).__name__}")

        def demonstrate_flext_exceptions_integration(self) -> None:
            """Show FlextExceptions (Layer 2) structured error handling."""
            print("\n=== FlextExceptions Integration (Layer 2) ===")

            # ValidationError with field context
            try:
                error_message = "Invalid email format"
                from flext_core._models.config import FlextModelsConfig

                raise FlextExceptions.ValidationError(
                    error_message,
                    config=FlextModelsConfig.ValidationErrorConfig(
                        field="email",
                        value="invalid-email",
                        error_code=FlextConstants.Errors.VALIDATION_ERROR,
                    ),
                )
            except FlextExceptions.ValidationError as e:
                result = FlextResult[str].fail(
                    e.message,
                    error_code=e.error_code,
                )
                print(f"✅ ValidationError: {result.error_code} - {result.error}")
                print(f"   Field: {e.field}, Value: {e.value}")

            # ConfigurationError with config key tracking
            try:
                config_error_message = "Missing required configuration"
                raise FlextExceptions.ConfigurationError(
                    config_error_message,
                    config_key="DATABASE_URL",
                    config_source="environment",
                )
            except FlextExceptions.ConfigurationError as e:
                result = FlextResult[str].fail(
                    e.message,
                    error_code=e.error_code,
                )
                print(f"✅ ConfigurationError: {e.error_code}")
                print(f"   Config Key: {e.config_key}, Source: {e.config_source}")

            # NotFoundError with resource details
            try:
                not_found_message = "User not found"
                raise FlextExceptions.NotFoundError(
                    not_found_message,
                    resource_type="user",
                    resource_id="123",
                )
            except FlextExceptions.NotFoundError as e:
                print(f"✅ NotFoundError: {e.error_code}")
                print(f"   Resource: {e.resource_type}, ID: {e.resource_id}")

            # ========== NEW ADVANCED METHODS (v0.9.9+) ==========

        def demonstrate_from_callable(self) -> None:
            """Show create_from_callable pattern for safe exception handling."""
            print("\n=== create_from_callable(): Safe Exception Handling ===")

            # Safe division using create_from_callable
            def risky_division() -> float:
                return 10 / 0  # Will raise ZeroDivisionError

            result = FlextResult[float].create_from_callable(risky_division)
            print(
                f"✅ Caught exception: {result.error if result.is_failure else 'Unexpected success'}",
            )

            # Successful callable
            def safe_addition() -> int:
                return 5 + 10

            addition_result: FlextResult[int] = FlextResult[int].create_from_callable(
                safe_addition,
            )
            print(f".create_from_callable(safe): {addition_result.unwrap()}")

            # With custom error code
            def parse_int_string() -> int:
                return int("not a number")

            parse_result: FlextResult[int] = FlextResult[int].create_from_callable(
                parse_int_string,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
            print(f".create_from_callable(with error_code): {parse_result.error}")

        def demonstrate_flow_through(self) -> None:
            """Show pipeline composition with flow_through."""
            print("\n=== flow_through(): Pipeline Composition ===")

            def validate_positive(x: object) -> FlextResult[object]:
                int_val = int(x) if isinstance(x, (int, float, str)) else 0
                if int_val <= 0:
                    return FlextResult[object].fail("Must be positive")
                return FlextResult[object].ok(int_val)

            def add_ten(x: object) -> FlextResult[object]:
                int_val = int(x) if isinstance(x, (int, float, str)) else 0
                return FlextResult[object].ok(int_val + 10)

            def multiply_by_two(x: object) -> FlextResult[object]:
                int_val = int(x) if isinstance(x, (int, float, str)) else 0
                return FlextResult[object].ok(int_val * 2)

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

            # Primary fails, transform error
            primary = FlextResult[str].fail("Primary service unavailable")

            def transform_error(error: str) -> str:
                return f"Fallback: {error}"

            result = primary.alt(transform_error)
            print(f"✅ Using alt (error transform): {result.unwrap()}")

            # Primary succeeds, alt not applied
            primary = FlextResult[str].ok("Primary value")
            result = primary.alt(transform_error)
            print(f"Primary success (alt not applied): {result.unwrap()}")

            # Chain multiple fallbacks using lash
            first = FlextResult[str].fail("First failed")

            def second_fallback(_error: str) -> FlextResult[str]:
                return FlextResult[str].fail("Second failed")

            def third_fallback(_error: str) -> FlextResult[str]:
                return FlextResult[str].ok("Third succeeded")

            result = first.lash(second_fallback).lash(third_fallback)
            print(f"Fallback chain (lash): {result.unwrap()}")

        def demonstrate_value_or_call(self) -> None:
            """Show default evaluation with unwrap_or."""
            print("\n=== unwrap_or(): Default Values ===")

            # Success case - default NOT used
            success = FlextResult[str].ok("success value")
            result = success.unwrap_or("expensive computed default")
            print(f"✅ Success: {result}")

            # Failure case - default IS used
            failure = FlextResult[str].fail("error occurred")
            result = failure.unwrap_or("expensive computed default")
            print(f"Failure (using default): {result}")

            # ========== DEPRECATED PATTERNS (WITH WARNINGS) ==========

        def demonstrate_deprecated_patterns(self) -> None:
            """Show deprecated patterns with proper warnings."""
            print("\n=== ⚠️ DEPRECATED PATTERNS ===")

            # OLD: Manual try/except (DEPRECATED)
            warnings.warn(
                "Manual try/except is DEPRECATED! Use FlextResult.create_from_callable() instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            print("❌ OLD WAY (try/except):")
            print("try:")
            print("    result = risky_function()")
            print("except Exception as e:")
            print("    handle_error(e)")

            # NEW: FlextResult pattern with create_from_callable
            print("\n✅ CORRECT WAY (FlextResult):")

            def risky_function() -> int:
                error_message = "Division by zero"
                raise ZeroDivisionError(error_message)  # Will raise

            result = FlextResult[int].create_from_callable(risky_function)
            print(f"FlextResult.create_from_callable(): {result}")

            # Show the clean new pattern
            print("\n✨ NEW PATTERN (create_from_callable with error code):")
            result_with_code = FlextResult[int].create_from_callable(
                risky_function,
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
            print(f"With custom error code: {result_with_code.error_code}")

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

    @staticmethod
    def main() -> None:
        """Main entry point demonstrating all FlextResult capabilities."""
        service: Example01BasicResult.ComprehensiveResultService = (
            Example01BasicResult.ComprehensiveResultService()
        )

        print("=" * 60)
        print("FlextResult COMPLETE API DEMONSTRATION")
        print("Foundation for 32+ FLEXT Ecosystem Projects")
        print(
            "With Layer 0.5-3 Integration (FlextRuntime → FlextConstants → FlextProtocols → FlextExceptions)",
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
            "✨ NEW v0.9.9+ methods prominently featured: from_callable, flow_through, lash, alt, value_or_call",
        )
        print(
            "🎯 Railway patterns: Replaced manual try/except with functional composition",
        )
        print(
            "🎯 Layer Integration: FlextRuntime (0.5) → FlextConstants (1) → FlextExceptions (2)",
        )
        print("🎯 Next: See 02_dependency_injection.py for FlextContainer patterns")
        print("=" * 60)


if __name__ == "__main__":
    Example01BasicResult.main()
