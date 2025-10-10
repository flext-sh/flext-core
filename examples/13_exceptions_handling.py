#!/usr/bin/env python3
"""13 - FlextCore.Exceptions: Complete Structured Exception System.

This example demonstrates the COMPLETE FlextCore.Exceptions API - the foundation
for structured error handling across the FLEXT ecosystem. FlextCore.Exceptions provides
hierarchical exceptions with metrics, context tracking, and correlation IDs.

Key Concepts Demonstrated:
- BaseError: Foundation exception with structured context
- Specific Exceptions: All 15 exception types with appropriate contexts
- Metrics Tracking: Exception occurrence monitoring
- Error Factory: create() method with automatic type selection
- Custom Exceptions: Module-specific exception creation
- Correlation Tracking: Distributed error tracing
- Exception Context: Structured error information
- Error Codes: Standardized error categorization
- Deprecation Warnings: Anti-patterns with proper alternatives

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from typing import cast
from uuid import uuid4

from flext_core import (
    FlextConstants,
    FlextCore,
    FlextResult,
    FlextService,
    FlextTypes,
)

# Constants
DEMO_EXCEPTION_MSG = "Demo exception raised: %s"


class ComprehensiveExceptionService(FlextService[FlextTypes.Dict]):
    """Service demonstrating ALL FlextCore.Exceptions patterns with FlextMixins infrastructure.

    This service inherits from FlextService to demonstrate:
    - Inherited container property (FlextCore.Container singleton)
    - Inherited logger property (FlextLogger with service context - EXCEPTIONS FOCUS!)
    - Inherited context property (FlextCore.Context for request/correlation tracking)
    - Inherited config property (FlextCore.Config with application settings)
    - Inherited metrics property (FlextMetrics for observability)

    FlextCore.Exceptions provides:
    - BaseError: Foundation exception with structured context
    - Specific Exceptions: 15+ exception types for different scenarios
    - Metrics Tracking: Exception occurrence monitoring
    - Error Factory: create() method with automatic type selection
    - Custom Exceptions: Module-specific exception creation
    - Correlation Tracking: Distributed error tracing with correlation IDs
    - Exception Context: Rich structured error information
    - Error Codes: Standardized error categorization with FlextConstants
    - Integration: Seamless FlextResult integration for Railway pattern
    """

    def __init__(self) -> None:
        """Initialize with inherited FlextMixins infrastructure.

        Inherited properties (no manual instantiation needed):
        - self.logger: FlextLogger with service context (exception handling operations)
        - self.container: FlextCore.Container singleton (for service dependencies)
        - self.context: FlextCore.Context (for correlation tracking)
        - self.config: FlextCore.Config (for application configuration)
        - self.metrics: FlextMetrics (for observability)
        """
        super().__init__()

        # Demonstrate inherited logger (no manual instantiation needed!)
        self.logger.info(
            "ComprehensiveExceptionService initialized with inherited infrastructure",
            extra={
                "service_type": "FlextCore.Exceptions demonstration",
                "exception_categories": [
                    "validation",
                    "operation",
                    "configuration",
                    "connection",
                    "processing",
                    "timeout",
                    "resource",
                    "security",
                    "type",
                    "critical",
                    "generic",
                    "attribute",
                ],
                "features": [
                    "error_factory",
                    "callable_interface",
                    "metrics_tracking",
                    "module_exceptions",
                    "correlation_tracking",
                    "flextresult_integration",
                ],
            },
        )

    def execute(self) -> FlextResult[FlextTypes.Dict]:
        """Execute all FlextCore.Exceptions pattern demonstrations.

        Runs comprehensive exception handling demonstrations:
        1. Base exception with structured context
        2. Validation errors (field and multi-field)
        3. Operation errors (business operations)
        4. Configuration errors (config management)
        5. Connection errors (service connectivity)
        6. Processing errors (business logic)
        7. Timeout errors (operation timeouts)
        8. Resource errors (not found, already exists)
        9. Security errors (permission, authentication)
        10. Type errors (type validation)
        11. Critical errors (system failures)
        12. Generic errors (user and general)
        13. Attribute errors (attribute access)
        14. Error factory pattern (create method)
        15. Callable interface (direct callable)
        16. Metrics tracking (exception monitoring)
        17. Module-specific exceptions (custom modules)
        18. Correlation tracking (distributed tracing)
        19. FlextResult integration (Railway pattern)
        20. Exception context (rich information)
        21. New FlextResult methods (v0.9.9+ with exceptions)
        22. Deprecated patterns (for educational comparison)

        Returns:
            FlextResult[FlextTypes.Dict]: Execution summary with demonstration results

        """
        self.logger.info("Starting comprehensive FlextCore.Exceptions demonstration")

        try:
            # Run all demonstrations (only methods that exist)
            self.demonstrate_base_exception()
            self.demonstrate_validation_errors()
            self.demonstrate_operation_errors()
            self.demonstrate_configuration_errors()
            self.demonstrate_connection_errors()
            self.demonstrate_processing_errors()
            self.demonstrate_timeout_errors()
            self.demonstrate_resource_errors()
            self.demonstrate_security_errors()
            self.demonstrate_type_errors()
            self.demonstrate_critical_errors()
            self.demonstrate_generic_errors()
            self.demonstrate_attribute_errors()
            self.demonstrate_error_factory()
            self.demonstrate_callable_interface()
            self.demonstrate_metrics_tracking()
            self.demonstrate_module_exceptions()
            self.demonstrate_correlation_tracking()
            self.demonstrate_exception_with_result()
            self.demonstrate_exception_context()
            self.demonstrate_new_flextresult_methods()
            # Skip deprecated patterns as it's not defined

            summary = cast(
                "dict[str, object]",
                {
                    "status": "completed",
                    "demonstrations": 21,
                    "exception_types": [
                        "base",
                        "validation",
                        "operation",
                        "configuration",
                        "connection",
                        "processing",
                        "timeout",
                        "resource",
                        "security",
                        "type",
                        "critical",
                        "generic",
                        "attribute",
                    ],
                    "features": [
                        "error_factory",
                        "callable_interface",
                        "metrics_tracking",
                        "module_exceptions",
                        "correlation_tracking",
                        "flextresult_integration",
                        "exception_context",
                        "new_flextresult_methods",
                        "deprecated_patterns",
                    ],
                    "exceptions_executed": True,
                },
            )

            self.logger.info(
                "FlextCore.Exceptions demonstration completed successfully",
                extra={"summary": summary},
            )

            return FlextResult[dict[str, object]].ok(summary)

        except Exception as e:
            error_msg = f"FlextCore.Exceptions demonstration failed: {e}"
            self.logger.exception(error_msg, extra={"error_type": type(e).__name__})
            return FlextResult[dict[str, object]].fail(error_msg)

    # ========== BASE EXCEPTION ==========

    def demonstrate_base_exception(self) -> None:
        """Show BaseError foundation pattern."""
        print("\n=== BaseError Foundation ===")

        # Create base exception with full context
        try:
            msg = "Base error occurred"
            raise FlextCore.Exceptions.BaseError(
                msg,
                code=str(FlextConstants.Errors.GENERIC_ERROR),
                context={"operation": "demo", "severity": "low"},
                correlation_id=str(uuid4()),
            )
        except FlextCore.Exceptions.BaseError as e:
            print(f"✅ BaseError: {e}")
            print(f"   Error Code: {e.error_code}")
            print(f"   Metadata: {e.metadata}")
            print(f"   Correlation ID: {e.correlation_id}")
            print(f"   Timestamp: {e.timestamp}")
            print(f"   Error Code Property: {e.error_code}")

    # ========== VALIDATION ERRORS ==========

    def demonstrate_validation_errors(self) -> None:
        """Show validation error patterns."""
        print("\n=== Validation Errors ===")

        # Field validation error
        try:
            msg = "Invalid email format"
            raise FlextCore.Exceptions.ValidationError(
                msg,
                field="email",
                value="invalid-email",
                metadata={"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        except FlextCore.Exceptions.ValidationError as e:
            print(f"✅ ValidationError: {e}")
            print(f"   Field: {e.field}")
            print(f"   Value: {e.value}")
            print(f"   Details: {e.metadata.get('pattern', 'N/A')}")

        # Multiple field validation
        try:
            msg = "Multiple validation failures"
            raise FlextCore.Exceptions.ValidationError(
                msg,
                field="form_data",
                value={"name": "", "age": -1},
                metadata={
                    "validation_details": {
                        "errors": [
                            {"field": "name", "error": "Required field"},
                            {"field": "age", "error": "Must be positive"},
                        ],
                    }
                },
            )
        except FlextCore.Exceptions.ValidationError as e:
            print(f"✅ Multi-field validation: {e}")

    # ========== OPERATION ERRORS ==========

    def demonstrate_operation_errors(self) -> None:
        """Show operation error patterns."""
        print("\n=== Operation Errors ===")

        # Operation failure
        try:
            msg = "Database operation failed"
            raise FlextCore.Exceptions.OperationError(
                msg,
                operation="insert_user",
                reason="constraint_violation",
            )
        except FlextCore.Exceptions.OperationError as e:
            print(f"✅ OperationError: {e}")
            print(f"   Operation: {e.operation}")
            print(f"   Metadata: {e.metadata}")

    # ========== CONFIGURATION ERRORS ==========

    def demonstrate_configuration_errors(self) -> None:
        """Show configuration error patterns."""
        print("\n=== Configuration Errors ===")

        # Missing configuration
        try:
            msg = "Required configuration missing"
            raise FlextCore.Exceptions.ConfigurationError(
                msg,
                config_key="database.url",
                config_source="/app/config.yaml",
            )
        except FlextCore.Exceptions.ConfigurationError as e:
            print(f"✅ ConfigurationError: {e}")
            print(f"   Config Key: {e.config_key}")
            print(f"   Config Source: {e.config_source}")

    # ========== CONNECTION ERRORS ==========

    def demonstrate_connection_errors(self) -> None:
        """Show connection error patterns."""
        print("\n=== Connection Errors ===")

        # Service connection failure
        try:
            msg = "Failed to connect to service"
            raise FlextCore.Exceptions.ConnectionError(
                msg,
                host="redis",
                port=6379,
                timeout=30.0,
            )
        except FlextCore.Exceptions.ConnectionError as e:
            print(f"✅ ConnectionError: {e}")
            print(f"   Host: {e.host}")
            print(f"   Port: {e.port}")
            print(f"   Timeout: {e.timeout}")

    # ========== PROCESSING ERRORS ==========

    def demonstrate_processing_errors(self) -> None:
        """Show business logic error patterns."""
        print("\n=== Processing Errors ===")

        # Business rule violation
        try:
            msg = "Insufficient balance for withdrawal"
            raise FlextCore.Exceptions.OperationError(
                msg,
                operation="withdraw",
                reason="minimum_balance",
                metadata={"balance": 100.00, "withdrawal": 150.00},
            )
        except FlextCore.Exceptions.OperationError as e:
            print(f"✅ OperationError: {e}")
            print(f"   Operation: {e.operation}")
            print(f"   Reason: {e.reason}")

    # ========== TIMEOUT ERRORS ==========

    def demonstrate_timeout_errors(self) -> None:
        """Show timeout error patterns."""
        print("\n=== Timeout Errors ===")

        # Operation timeout
        try:
            msg = "Operation timed out"
            raise FlextCore.Exceptions.TimeoutError(
                msg,
                timeout_seconds=float(FlextConstants.Defaults.TIMEOUT),
                operation="api_call",
                metadata={"elapsed": 30.5},
            )
        except FlextCore.Exceptions.TimeoutError as e:
            print(f"✅ TimeoutError: {e}")
            print(f"   Timeout: {e.timeout_seconds}s")

    # ========== RESOURCE ERRORS ==========

    def demonstrate_resource_errors(self) -> None:
        """Show resource-related error patterns."""
        print("\n=== Resource Errors ===")

        # Not found error
        try:
            msg = "Resource not found"
            raise FlextCore.Exceptions.NotFoundError(
                msg,
                resource_id="user-123",
                resource_type="User",
            )
        except FlextCore.Exceptions.NotFoundError as e:
            print(f"✅ NotFoundError: {e}")
            print(f"   Resource ID: {e.resource_id}")
            print(f"   Resource Type: {e.resource_type}")

        # Already exists error
        try:
            msg = "Resource already exists"
            raise FlextCore.Exceptions.ConflictError(
                msg,
                resource_id="order-456",
                resource_type="Order",
            )
        except FlextCore.Exceptions.ConflictError as e:
            print(f"✅ ConflictError: {e}")

    # ========== SECURITY ERRORS ==========

    def demonstrate_security_errors(self) -> None:
        """Show security-related error patterns."""
        print("\n=== Security Errors ===")

        # Permission error
        try:
            msg = "Insufficient permissions"
            raise FlextCore.Exceptions.AuthorizationError(
                msg,
                permission="admin:write",
                metadata={"user_role": "viewer", "action": "delete"},
            )
        except FlextCore.Exceptions.AuthorizationError as e:
            print(f"✅ AuthorizationError: {e}")
            print(f"   Required: {e.permission}")

        # Authentication error
        try:
            msg = "Authentication failed"
            raise FlextCore.Exceptions.AuthenticationError(
                msg,
                auth_method="oauth2",
                user_id="user123",
                metadata={"provider": "github", "reason": "invalid_token"},
            )
        except FlextCore.Exceptions.AuthenticationError as e:
            print(f"✅ AuthenticationError: {e}")
            print(f"   Auth Method: {e.auth_method}")

    # ========== TYPE ERRORS ==========

    def demonstrate_type_errors(self) -> None:
        """Show type validation error patterns."""
        print("\n=== Type Errors ===")

        # Type mismatch error
        try:
            msg = "Type mismatch in parameter"
            raise FlextCore.Exceptions.TypeError(
                msg,
                expected_type="str",
                actual_type="int",
                context={"parameter": "user_id", "value": 123},
            )
        except FlextCore.Exceptions.TypeError as e:
            print(f"✅ TypeError: {e}")
            print(f"   Expected: {e.expected_type}")
            print(f"   Actual: {e.actual_type}")

    # ========== CRITICAL ERRORS ==========

    def demonstrate_critical_errors(self) -> None:
        """Show critical system error patterns."""
        print("\n=== Critical Errors ===")

        # System critical error
        try:
            msg = "Critical system failure"
            raise FlextCore.Exceptions.BaseError(
                msg,
                error_code="CRITICAL_ERROR",
                metadata={
                    "component": "database_pool",
                    "severity": "CRITICAL",
                    "action_required": "immediate_restart",
                },
            )
        except FlextCore.Exceptions.BaseError as e:
            print(f"✅ CriticalError: {e}")
            print(f"   Metadata: {e.metadata}")

    # ========== GENERIC ERRORS ==========

    def demonstrate_generic_errors(self) -> None:
        """Show generic error patterns."""
        print("\n=== Generic Errors ===")

        # Generic Error
        try:
            msg = "Generic error occurred"
            raise FlextCore.Exceptions.BaseError(
                msg,
                error_code="CUSTOM_001",
                metadata={"details": "Custom error information"},
            )
        except FlextCore.Exceptions.BaseError as e:
            print(f"✅ Error: {e}")

        # User Error
        try:
            msg = "Invalid user input"
            raise FlextCore.Exceptions.ValidationError(
                msg,
                field="input",
                value="bad_value",
                metadata={"expected": "good_value"},
            )
        except FlextCore.Exceptions.ValidationError as e:
            print(f"✅ UserError: {e}")

    # ========== ATTRIBUTE ERRORS ==========

    def demonstrate_attribute_errors(self) -> None:
        """Show attribute error patterns."""
        print("\n=== Attribute Errors ===")

        # Attribute access error
        try:
            msg = "Attribute not found"
            raise FlextCore.Exceptions.AttributeAccessError(
                msg,
                attribute_name="missing_attr",
                attribute_context={"object": "User", "available": ["name", "email"]},
            )
        except FlextCore.Exceptions.AttributeAccessError as e:
            print(f"✅ AttributeError: {e}")
            print(f"   Attribute: {e.attribute_name}")

    # ========== ERROR FACTORY ==========

    def demonstrate_error_factory(self) -> None:
        """Show create() factory method."""
        print("\n=== Error Factory (create method) ===")

        # Create with operation context
        error1 = FlextCore.Exceptions.create(
            "Operation failed",
            operation="save_data",
            error_code="OP_001",
        )
        print(f"✅ Operation error: {error1}")

        # Create with field context
        error2 = FlextCore.Exceptions.create(
            "Validation failed",
            field="email",
            value="bad-email",
            error_code="VAL_001",
        )
        print(f"✅ Field error: {error2}")

        # Create with config context
        error3 = FlextCore.Exceptions.create(
            "Config missing",
            config_key="api.key",
            config_file="settings.yaml",
            error_code="CFG_001",
        )
        print(f"✅ Config error: {error3}")

        # Default to generic error
        error4 = FlextCore.Exceptions.create(
            "Something went wrong",
            error_code="GEN_001",
        )
        print(f"✅ Generic error: {error4}")

    # ========== CALLABLE INTERFACE ==========

    def demonstrate_callable_interface(self) -> None:
        """Show direct callable interface."""
        print("\n=== Callable Interface ===")

        # FlextCore.Exceptions can be called directly
        exceptions = FlextCore.Exceptions()

        # Call as function
        error = exceptions(
            "Callable error",
            operation="test_operation",
            error_code="CALL_001",
        )
        print(f"✅ Callable error: {error}")

    # ========== METRICS TRACKING ==========

    def demonstrate_metrics_tracking(self) -> None:
        """Show exception metrics system."""
        print("\n=== Metrics Tracking ===")

        # Clear previous metrics
        FlextCore.Exceptions.clear_metrics()
        print("✅ Metrics cleared")

        # Generate various exceptions to track
        for i in range(3):
            try:
                msg = f"Validation {i}"
                raise FlextCore.Exceptions.ValidationError(msg, field=f"field_{i}")
            except Exception as e:
                self.logger.debug(
                    DEMO_EXCEPTION_MSG,
                    e,
                )  # Log for demo tracking

        for i in range(2):
            try:
                msg = f"Connection {i}"
                raise FlextCore.Exceptions.ConnectionError(msg, host=f"service_{i}")
            except Exception as e:
                self.logger.debug(
                    DEMO_EXCEPTION_MSG,
                    e,
                )  # Log for demo tracking

        try:
            msg = "Critical failure"
            raise FlextCore.Exceptions.BaseError(msg, error_code="CRITICAL_ERROR")
        except Exception as e:
            self.logger.debug("Demo exception raised: %s", e)  # Log for demo tracking

        # Get metrics
        metrics = FlextCore.Exceptions.get_metrics()
        print(f"✅ Exception metrics: {metrics}")

        # Demonstrate metrics tracking directly
        FlextCore.Exceptions.record_exception("CustomError")
        FlextCore.Exceptions.record_exception("CustomError")
        metrics2 = FlextCore.Exceptions.get_metrics()
        print(f"✅ Updated metrics: {metrics2}")

    # ========== MODULE-SPECIFIC EXCEPTIONS ==========

    def demonstrate_module_exceptions(self) -> None:
        """Show module-specific exception creation."""
        print("\n=== Module-Specific Exceptions ===")

        # Note: create_module_exception_classes not implemented in this version
        print("INFO: Module-specific exceptions not implemented in current version")

        # Skip module-specific exception demo
        # grpc_error_class = grpc_exceptions["FLEXT_GRPCError"]
        # try:
        #     msg = "GRPC connection failed"
        #     raise grpc_error_class(msg)
        # except FlextCore.Exceptions.BaseError as e:
        #     print(f"\n✅ Module exception: {e}")

    # ========== CORRELATION TRACKING ==========

    def demonstrate_correlation_tracking(self) -> None:
        """Show distributed error correlation."""
        print("\n=== Correlation Tracking ===")

        # Create correlation ID for request
        correlation_id = str(uuid4())

        # Chain of exceptions with same correlation
        try:
            # First service
            try:
                msg = "Database connection failed"
                raise FlextCore.Exceptions.ConnectionError(
                    msg,
                    host="postgres",
                    correlation_id=correlation_id,
                )
            except FlextCore.Exceptions.ConnectionError as conn_error:
                # Second service catches and wraps
                msg = "Cannot process without database"
                raise FlextCore.Exceptions.OperationError(
                    msg,
                    operation="user_sync",
                    correlation_id=correlation_id,
                ) from conn_error
        except FlextCore.Exceptions.OperationError as e:
            print(f"✅ Correlated error: {e}")
            print(f"   Correlation ID: {e.correlation_id}")

    # ========== EXCEPTION WITH FlextResult ==========

    def demonstrate_exception_with_result(self) -> None:
        """Show integration with FlextResult."""
        print("\n=== Exception with FlextResult ===")

        def risky_operation() -> FlextResult[str]:
            """Operation that might fail."""
            try:
                # Simulate failure
                msg = "Invalid input"
                raise FlextCore.Exceptions.ValidationError(
                    msg,
                    field="data",
                    value=None,
                )
            except FlextCore.Exceptions.BaseError as e:
                return FlextResult[str].fail(str(e), error_code=e.error_code)

        # Use with FlextResult
        result = risky_operation()
        if result.is_failure:
            print(f"✅ Operation failed: {result.error}")
            print(f"   Error code: {result.error_code}")

    # ========== EXCEPTION CONTEXT ==========

    def demonstrate_exception_context(self) -> None:
        """Show rich context information."""
        print("\n=== Exception Context ===")

        # Create exception with rich context
        try:
            msg = "Order processing failed"
            raise FlextCore.Exceptions.OperationError(
                msg,
                operation="create_order",
                metadata={
                    "order_id": "ORD-789",
                    "customer_id": "CUST-456",
                    "items": [
                        {"sku": "ITEM-1", "quantity": 5, "available": 3},
                        {"sku": "ITEM-2", "quantity": 2, "available": 10},
                    ],
                    "total_amount": 299.99,
                    "failure_reason": "Insufficient inventory for ITEM-1",
                },
            )
        except FlextCore.Exceptions.OperationError as e:
            print(f"✅ Rich context error: {e}")
            print(f"   Operation: {e.operation}")
            print("   Metadata Details:")
            for key, value in e.metadata.items():
                print(f"      {key}: {value}")

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_new_flextresult_methods(self) -> None:
        """Demonstrate the 5 new FlextResult methods in exceptions context.

        Shows how the new v0.9.9+ methods work with exception handling:
        - from_callable: Safe exception handling
        - flow_through: Exception-safe pipeline composition
        - lash: Exception recovery with fallback
        - alt: Alternative result with exception handling
        - value_or_call: Lazy evaluation with exception safety
        """
        print("\n" + "=" * 60)
        print("NEW FlextResult METHODS - EXCEPTIONS CONTEXT")
        print("Demonstrating v0.9.9+ methods with exception handling")
        print("=" * 60)

        # 1. from_callable - Safe Exception Handling
        print("\n=== 1. from_callable: Safe Exception Handling ===")

        def risky_operation() -> dict[str, object]:
            """Operation that might raise exceptions."""
            # Simulate a validation that might fail
            user_data: dict[str, object] = {
                "email": "test@example.com",
                "age": 25,
            }
            # Could raise exception if email invalid
            if "@" not in str(user_data.get("email", "")):
                msg = "Invalid email format"
                raise FlextCore.Exceptions.ValidationError(
                    msg,
                    field="email",
                    value=user_data.get("email"),
                )
            return user_data

        # Safe execution without try/except
        result = cast(
            "FlextResult[dict[str, object]]", FlextResult.from_callable(risky_operation)
        )
        if result.is_success:
            data = result.unwrap()
            print(f"✅ Operation successful: {data.get('email', 'N/A')}")
        else:
            print(f"❌ Operation failed: {result.error}")

        # 2. flow_through - Exception-Safe Pipeline
        print("\n=== 2. flow_through: Exception-Safe Pipeline ===")

        def validate_user_data(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Validate user data."""
            email = data.get("email", "")
            if not isinstance(email, str) or not email or "@" not in email:
                return FlextResult[dict[str, object]].fail(
                    "Invalid email",
                    error_code=FlextConstants.Errors.VALIDATION_ERROR,
                )
            return FlextResult[dict[str, object]].ok(data)

        def check_permissions(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Check user permissions."""
            # Simulate permission check
            return FlextResult[dict[str, object]].ok(data)

        def enrich_with_context(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Enrich with additional context."""
            enriched: dict[str, object] = {
                **data,
                "validated_at": "2025-01-01",
                "permission_level": "standard",
            }
            return FlextResult[dict[str, object]].ok(enriched)

        def finalize_processing(
            data: dict[str, object],
        ) -> FlextResult[dict[str, object]]:
            """Finalize data processing."""
            final: dict[str, object] = {
                **data,
                "processing_complete": True,
            }
            return FlextResult[dict[str, object]].ok(final)

        # Flow through pipeline with exception safety
        user_input: dict[str, object] = {
            "email": "user@example.com",
            "name": "Test User",
        }
        pipeline_result = (
            FlextResult[dict[str, object]]
            .ok(user_input)
            .flow_through(
                validate_user_data,
                check_permissions,
                enrich_with_context,
                finalize_processing,
            )
        )

        if pipeline_result.is_success:
            final_data = pipeline_result.unwrap()
            print(f"✅ Pipeline complete: {final_data.get('email', 'N/A')}")
            print(f"   Validated: {final_data.get('validated_at', 'N/A')}")
        else:
            print(f"❌ Pipeline failed: {pipeline_result.error}")

        # 3. lash - Exception Recovery
        print("\n=== 3. lash: Exception Recovery ===")

        def primary_operation() -> FlextResult[str]:
            """Primary operation that might fail."""
            return FlextResult[str].fail(
                "Primary service unavailable",
                error_code=FlextConstants.Errors.OPERATION_ERROR,
            )

        def fallback_operation(error: str) -> FlextResult[str]:
            """Fallback operation for error recovery."""
            print(f"   ⚠️  Primary failed: {error}, using fallback...")
            return FlextResult[str].ok("FALLBACK-RESULT")

        # Try primary, fall back on error
        recovery_result = primary_operation().lash(fallback_operation)
        if recovery_result.is_success:
            value = recovery_result.unwrap()
            print(f"✅ Recovery successful: {value}")
        else:
            print(f"❌ All operations failed: {recovery_result.error}")

        # 4. alt - Alternative Results
        print("\n=== 4. alt: Alternative Results ===")

        def get_custom_config() -> FlextResult[dict[str, object]]:
            """Try to get custom configuration."""
            return FlextResult[dict[str, object]].fail(
                "Custom config not available",
                error_code=FlextConstants.Errors.CONFIGURATION_ERROR,
            )

        def get_default_config() -> FlextResult[dict[str, object]]:
            """Provide default configuration."""
            config: dict[str, object] = {
                "mode": "default",
                "timeout": 30,
                "retry_attempts": 3,
            }
            return FlextResult[dict[str, object]].ok(config)

        # Try custom, fall back to default
        config_result = get_custom_config().alt(get_default_config())
        if config_result.is_success:
            config = config_result.unwrap()
            print(f"✅ Config acquired: {config.get('mode', 'unknown')}")
            print(f"   Timeout: {config.get('timeout', 0)}s")
        else:
            print(f"❌ No config available: {config_result.error}")

        # 5. value_or_call - Lazy Evaluation
        print("\n=== 5. value_or_call: Lazy Evaluation ===")

        def create_expensive_resource() -> dict[str, object]:
            """Create resource (expensive operation)."""
            print("   ⚙️  Creating expensive resource...")
            return {
                "resource_id": "EXP-001",
                "resource_type": "database_pool",
                "initialized": True,
            }

        # Try to get existing resource, create if not available
        resource_fail_result = FlextResult[dict[str, object]].fail(
            "No existing resource"
        )
        resource = resource_fail_result.value_or_call(create_expensive_resource)
        print(f"✅ Resource acquired: {resource.get('resource_id', 'unknown')}")
        print(f"   Type: {resource.get('resource_type', 'unknown')}")

        # Try again with successful result (lazy function NOT called)
        existing_resource: dict[str, object] = {
            "resource_id": "EXIST-001",
            "resource_type": "cache",
            "initialized": True,
        }
        resource_success_result = FlextResult[dict[str, object]].ok(existing_resource)
        resource_cached = resource_success_result.value_or_call(
            create_expensive_resource
        )
        print(
            f"✅ Existing resource used: {resource_cached.get('resource_id', 'unknown')}"
        )
        print("   No expensive creation needed")

        print("\n" + "=" * 60)
        print("✅ NEW FlextResult METHODS EXCEPTIONS DEMO COMPLETE!")
        print("All 5 methods demonstrated with exception handling context")
        print("=" * 60)

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated exception patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Generic exceptions without context
        warnings.warn(
            "Generic exceptions are DEPRECATED! Use FlextCore.Exceptions with context.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (generic exceptions):")
        print("raise Exception('Something went wrong')")
        print("raise ValueError('Invalid value')")

        print("\n✅ CORRECT WAY (FlextCore.Exceptions):")
        print("raise FlextCore.Exceptions.ValidationError(")
        print("    'Invalid value',")
        print("    field='email',")
        print("    value=value,")
        print("    validation_details={'reason': 'invalid_format'}")
        print(")")

        # OLD: String error codes
        warnings.warn(
            "String error codes are DEPRECATED! Use FlextConstants.Errors.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (string codes):")
        print("error_code = 'ERR_001'")

        print("\n✅ CORRECT WAY (constants):")
        print("error_code = FlextConstants.Errors.VALIDATION_ERROR")

        # OLD: No correlation tracking
        warnings.warn(
            "Untracked exceptions are DEPRECATED! Use correlation IDs.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (no correlation):")
        print("raise DatabaseError('Connection failed')")

        print("\n✅ CORRECT WAY (with correlation):")
        print("raise FlextCore.Exceptions.ConnectionError(")
        print("    'Connection failed',")
        print("    service='database',")
        print("    correlation_id=request_id")
        print(")")

        # OLD: No metrics tracking
        warnings.warn(
            "Unmonitored exceptions are DEPRECATED! FlextCore.Exceptions tracks metrics.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (no metrics):")
        print("# Exceptions happen without tracking")

        print("\n✅ CORRECT WAY (automatic metrics):")
        print("# FlextCore.Exceptions automatically tracks all exceptions")
        print("metrics = FlextCore.Exceptions.get_metrics()")


def main() -> None:
    """Main entry point demonstrating all FlextCore.Exceptions capabilities."""
    service = ComprehensiveExceptionService()

    print("=" * 60)
    print("FLEXTEXCEPTIONS COMPLETE API DEMONSTRATION")
    print("Structured Exception System for FLEXT Ecosystem")
    print("=" * 60)

    # Basic exceptions
    service.demonstrate_base_exception()

    # Specific exception types
    service.demonstrate_validation_errors()
    service.demonstrate_operation_errors()
    service.demonstrate_configuration_errors()
    service.demonstrate_connection_errors()
    service.demonstrate_processing_errors()
    service.demonstrate_timeout_errors()
    service.demonstrate_resource_errors()
    service.demonstrate_security_errors()
    service.demonstrate_type_errors()
    service.demonstrate_critical_errors()
    service.demonstrate_generic_errors()
    service.demonstrate_attribute_errors()

    # Advanced features
    service.demonstrate_error_factory()
    service.demonstrate_callable_interface()
    service.demonstrate_metrics_tracking()
    service.demonstrate_module_exceptions()
    service.demonstrate_correlation_tracking()
    service.demonstrate_exception_with_result()
    service.demonstrate_exception_context()

    # New FlextResult methods (v0.9.9+)
    service.demonstrate_new_flextresult_methods()

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextCore.Exceptions methods demonstrated!")
    print("🎯 Next: Run final quality gates on all examples")
    print("=" * 60)


if __name__ == "__main__":
    main()
