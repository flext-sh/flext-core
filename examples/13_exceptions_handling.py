#!/usr/bin/env python3
"""13 - FlextExceptions: Complete Structured Exception System.

This example demonstrates the COMPLETE FlextExceptions API - the foundation
for structured error handling across the FLEXT ecosystem. FlextExceptions provides
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
from uuid import uuid4

from flext_core import (
    FlextConstants,
    FlextContainer,
    FlextExceptions,
    FlextLogger,
    FlextResult,
    FlextService,
    FlextTypes,
)

# Constants
DEMO_EXCEPTION_MSG = "Demo exception raised: %s"


class ComprehensiveExceptionService(FlextService[FlextTypes.Dict]):
    """Service demonstrating ALL FlextExceptions patterns and methods."""

    def __init__(self) -> None:
        """Initialize with dependencies."""
        super().__init__()
        manager = FlextContainer.ensure_global_manager()
        self._container = manager.get_or_create()
        self._logger = FlextLogger(__name__)

    def execute(self) -> FlextResult[FlextTypes.Dict]:
        """Execute method required by FlextService."""
        self._logger.info("Executing exception demonstration")
        return FlextResult[FlextTypes.Dict].ok({
            "status": "completed",
            "exceptions_demonstrated": True,
        })

    # ========== BASE EXCEPTION ==========

    def demonstrate_base_exception(self) -> None:
        """Show BaseError foundation pattern."""
        print("\n=== BaseError Foundation ===")

        # Create base exception with full context
        try:
            msg = "Base error occurred"
            raise FlextExceptions.BaseError(
                msg,
                code=str(FlextConstants.Errors.GENERIC_ERROR),
                context={"operation": "demo", "severity": "low"},
                correlation_id=str(uuid4()),
            )
        except FlextExceptions.BaseError as e:
            print(f"✅ BaseError: {e}")
            print(f"   Code: {e.code}")
            print(f"   Context: {e.context}")
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
            raise FlextExceptions.ValidationError(
                msg,
                field="email",
                value="invalid-email",
                validation_details={"pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$"},
                error_code=FlextConstants.Errors.VALIDATION_ERROR,
            )
        except FlextExceptions.ValidationError as e:
            print(f"✅ ValidationError: {e}")
            print(f"   Field: {e.field}")
            print(f"   Value: {e.value}")
            print(f"   Details: {e.validation_details}")

        # Multiple field validation
        try:
            msg = "Multiple validation failures"
            raise FlextExceptions.ValidationError(
                msg,
                field="form_data",
                value={"name": "", "age": -1},
                validation_details={
                    "errors": [
                        {"field": "name", "error": "Required field"},
                        {"field": "age", "error": "Must be positive"},
                    ],
                },
            )
        except FlextExceptions.ValidationError as e:
            print(f"✅ Multi-field validation: {e}")

    # ========== OPERATION ERRORS ==========

    def demonstrate_operation_errors(self) -> None:
        """Show operation error patterns."""
        print("\n=== Operation Errors ===")

        # Operation failure
        try:
            msg = "Database operation failed"
            raise FlextExceptions.OperationError(
                msg,
                operation="insert_user",
                context={"table": "users", "rows_affected": 0},
            )
        except FlextExceptions.OperationError as e:
            print(f"✅ OperationError: {e}")
            print(f"   Operation: {e.operation}")
            print(f"   Context: {e.context}")

    # ========== CONFIGURATION ERRORS ==========

    def demonstrate_configuration_errors(self) -> None:
        """Show configuration error patterns."""
        print("\n=== Configuration Errors ===")

        # Missing configuration
        try:
            msg = "Required configuration missing"
            raise FlextExceptions.ConfigurationError(
                msg,
                config_key="database.url",
                config_file="/app/config.yaml",
            )
        except FlextExceptions.ConfigurationError as e:
            print(f"✅ ConfigurationError: {e}")
            print(f"   Config Key: {e.config_key}")
            print(f"   Config File: {e.config_file}")

    # ========== CONNECTION ERRORS ==========

    def demonstrate_connection_errors(self) -> None:
        """Show connection error patterns."""
        print("\n=== Connection Errors ===")

        # Service connection failure
        try:
            msg = "Failed to connect to service"
            raise FlextExceptions.ConnectionError(
                msg,
                service="redis",
                endpoint="redis://localhost:6379",
                context={"retry_count": 3, "timeout": 30},
            )
        except FlextExceptions.ConnectionError as e:
            print(f"✅ ConnectionError: {e}")
            print(f"   Service: {e.service}")
            print(f"   Endpoint: {e.endpoint}")

    # ========== PROCESSING ERRORS ==========

    def demonstrate_processing_errors(self) -> None:
        """Show business logic error patterns."""
        print("\n=== Processing Errors ===")

        # Business rule violation
        try:
            msg = "Insufficient balance for withdrawal"
            raise FlextExceptions.ProcessingError(
                msg,
                business_rule="minimum_balance",
                operation="withdraw",
                context={"balance": 100.00, "withdrawal": 150.00},
            )
        except FlextExceptions.ProcessingError as e:
            print(f"✅ ProcessingError: {e}")
            print(f"   Business Rule: {e.business_rule}")
            print(f"   Operation: {e.operation}")

    # ========== TIMEOUT ERRORS ==========

    def demonstrate_timeout_errors(self) -> None:
        """Show timeout error patterns."""
        print("\n=== Timeout Errors ===")

        # Operation timeout
        try:
            msg = "Operation timed out"
            raise FlextExceptions.TimeoutError(
                msg,
                timeout_seconds=float(FlextConstants.Defaults.TIMEOUT),
                context={"operation": "api_call", "elapsed": 30.5},
            )
        except FlextExceptions.TimeoutError as e:
            print(f"✅ TimeoutError: {e}")
            print(f"   Timeout: {e.timeout_seconds}s")

    # ========== RESOURCE ERRORS ==========

    def demonstrate_resource_errors(self) -> None:
        """Show resource-related error patterns."""
        print("\n=== Resource Errors ===")

        # Not found error
        try:
            msg = "Resource not found"
            raise FlextExceptions.NotFoundError(
                msg,
                resource_id="user-123",
                resource_type="User",
            )
        except FlextExceptions.NotFoundError as e:
            print(f"✅ NotFoundError: {e}")
            print(f"   Resource ID: {e.resource_id}")
            print(f"   Resource Type: {e.resource_type}")

        # Already exists error
        try:
            msg = "Resource already exists"
            raise FlextExceptions.AlreadyExistsError(
                msg,
                resource_id="order-456",
                resource_type="Order",
            )
        except FlextExceptions.AlreadyExistsError as e:
            print(f"✅ AlreadyExistsError: {e}")

    # ========== SECURITY ERRORS ==========

    def demonstrate_security_errors(self) -> None:
        """Show security-related error patterns."""
        print("\n=== Security Errors ===")

        # Permission error
        try:
            msg = "Insufficient permissions"
            raise FlextExceptions.PermissionError(
                msg,
                required_permission="admin:write",
                context={"user_role": "viewer", "action": "delete"},
            )
        except FlextExceptions.PermissionError as e:
            print(f"✅ PermissionError: {e}")
            print(f"   Required: {e.required_permission}")

        # Authentication error
        try:
            msg = "Authentication failed"
            raise FlextExceptions.AuthenticationError(
                msg,
                auth_method="oauth2",
                context={"provider": "github", "reason": "invalid_token"},
            )
        except FlextExceptions.AuthenticationError as e:
            print(f"✅ AuthenticationError: {e}")
            print(f"   Auth Method: {e.auth_method}")

    # ========== TYPE ERRORS ==========

    def demonstrate_type_errors(self) -> None:
        """Show type validation error patterns."""
        print("\n=== Type Errors ===")

        # Type mismatch error
        try:
            msg = "Type mismatch in parameter"
            raise FlextExceptions.TypeError(
                msg,
                expected_type="str",
                actual_type="int",
                context={"parameter": "user_id", "value": 123},
            )
        except FlextExceptions.TypeError as e:
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
            raise FlextExceptions.CriticalError(
                msg,
                component="database_pool",
                severity="CRITICAL",
                action_required="immediate_restart",
            )
        except FlextExceptions.CriticalError as e:
            print(f"✅ CriticalError: {e}")
            print(f"   Context: {e.context}")

    # ========== GENERIC ERRORS ==========

    def demonstrate_generic_errors(self) -> None:
        """Show generic error patterns."""
        print("\n=== Generic Errors ===")

        # Generic Error
        try:
            msg = "Generic error occurred"
            raise FlextExceptions.Error(
                msg,
                error_code="CUSTOM_001",
                details="Custom error information",
            )
        except FlextExceptions.Error as e:
            print(f"✅ Error: {e}")

        # User Error
        try:
            msg = "Invalid user input"
            raise FlextExceptions.UserError(
                msg,
                context={"input": "bad_value", "expected": "good_value"},
            )
        except FlextExceptions.UserError as e:
            print(f"✅ UserError: {e}")

    # ========== ATTRIBUTE ERRORS ==========

    def demonstrate_attribute_errors(self) -> None:
        """Show attribute error patterns."""
        print("\n=== Attribute Errors ===")

        # Attribute access error
        try:
            msg = "Attribute not found"
            raise FlextExceptions.AttributeError(
                msg,
                attribute_name="missing_attr",
                attribute_context={"object": "User", "available": ["name", "email"]},
            )
        except FlextExceptions.AttributeError as e:
            print(f"✅ AttributeError: {e}")
            print(f"   Attribute: {e.attribute_name}")

    # ========== ERROR FACTORY ==========

    def demonstrate_error_factory(self) -> None:
        """Show create() factory method."""
        print("\n=== Error Factory (create method) ===")

        # Create with operation context
        error1 = FlextExceptions.create(
            "Operation failed",
            operation="save_data",
            error_code="OP_001",
        )
        print(f"✅ Operation error: {error1}")

        # Create with field context
        error2 = FlextExceptions.create(
            "Validation failed",
            field="email",
            value="bad-email",
            error_code="VAL_001",
        )
        print(f"✅ Field error: {error2}")

        # Create with config context
        error3 = FlextExceptions.create(
            "Config missing",
            config_key="api.key",
            config_file="settings.yaml",
            error_code="CFG_001",
        )
        print(f"✅ Config error: {error3}")

        # Default to generic error
        error4 = FlextExceptions.create(
            "Something went wrong",
            error_code="GEN_001",
        )
        print(f"✅ Generic error: {error4}")

    # ========== CALLABLE INTERFACE ==========

    def demonstrate_callable_interface(self) -> None:
        """Show direct callable interface."""
        print("\n=== Callable Interface ===")

        # FlextExceptions can be called directly
        exceptions = FlextExceptions()

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
        FlextExceptions.clear_metrics()
        print("✅ Metrics cleared")

        # Generate various exceptions to track
        for i in range(3):
            try:
                msg = f"Validation {i}"
                raise FlextExceptions.ValidationError(msg, field=f"field_{i}")
            except Exception as e:
                self._logger.debug(
                    DEMO_EXCEPTION_MSG,
                    e,
                )  # Log for demo tracking

        for i in range(2):
            try:
                msg = f"Connection {i}"
                raise FlextExceptions.ConnectionError(msg, service=f"service_{i}")
            except Exception as e:
                self._logger.debug(
                    DEMO_EXCEPTION_MSG,
                    e,
                )  # Log for demo tracking

        try:
            msg = "Critical failure"
            raise FlextExceptions.CriticalError(msg)
        except Exception as e:
            self._logger.debug("Demo exception raised: %s", e)  # Log for demo tracking

        # Get metrics
        metrics = FlextExceptions.get_metrics()
        print(f"✅ Exception metrics: {metrics}")

        # Demonstrate metrics tracking directly
        FlextExceptions.record_exception("CustomError")
        FlextExceptions.record_exception("CustomError")
        metrics2 = FlextExceptions.get_metrics()
        print(f"✅ Updated metrics: {metrics2}")

    # ========== MODULE-SPECIFIC EXCEPTIONS ==========

    def demonstrate_module_exceptions(self) -> None:
        """Show module-specific exception creation."""
        print("\n=== Module-Specific Exceptions ===")

        # Create exceptions for a specific module
        grpc_exceptions = FlextExceptions.create_module_exception_classes("flext-grpc")

        print("✅ Created FLEXT_GRPC exceptions:")
        for name, exc_class in grpc_exceptions.items():
            print(f"   - {name}: {exc_class.__name__}")

        # Use module-specific exception
        grpc_error_class = grpc_exceptions["FLEXT_GRPCError"]
        try:
            msg = "GRPC connection failed"
            raise grpc_error_class(msg)
        except FlextExceptions.BaseError as e:
            print(f"\n✅ Module exception: {e}")

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
                raise FlextExceptions.ConnectionError(
                    msg,
                    service="postgres",
                    correlation_id=correlation_id,
                )
            except FlextExceptions.ConnectionError as conn_error:
                # Second service catches and wraps
                msg = "Cannot process without database"
                raise FlextExceptions.ProcessingError(
                    msg,
                    operation="user_sync",
                    correlation_id=correlation_id,
                ) from conn_error
        except FlextExceptions.ProcessingError as e:
            print(f"✅ Correlated error: {e}")
            print(f"   Correlation ID: {e.correlation_id}")

    # ========== EXCEPTION WITH FLEXTRESULT ==========

    def demonstrate_exception_with_result(self) -> None:
        """Show integration with FlextResult."""
        print("\n=== Exception with FlextResult ===")

        def risky_operation() -> FlextResult[str]:
            """Operation that might fail."""
            try:
                # Simulate failure
                msg = "Invalid input"
                raise FlextExceptions.ValidationError(
                    msg,
                    field="data",
                    value=None,
                )
            except FlextExceptions.BaseError as e:
                return FlextResult[str].fail(str(e), error_code=e.code)

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
            raise FlextExceptions.ProcessingError(
                msg,
                business_rule="inventory_check",
                operation="create_order",
                context={
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
        except FlextExceptions.ProcessingError as e:
            print(f"✅ Rich context error: {e}")
            print(f"   Business Rule: {e.business_rule}")
            print("   Context Details:")
            for key, value in e.context.items():
                print(f"      {key}: {value}")

    # ========== DEPRECATED PATTERNS ==========

    def demonstrate_deprecated_patterns(self) -> None:
        """Show deprecated exception patterns."""
        print("\n=== ⚠️ DEPRECATED PATTERNS ===")

        # OLD: Generic exceptions without context
        warnings.warn(
            "Generic exceptions are DEPRECATED! Use FlextExceptions with context.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("❌ OLD WAY (generic exceptions):")
        print("raise Exception('Something went wrong')")
        print("raise ValueError('Invalid value')")

        print("\n✅ CORRECT WAY (FlextExceptions):")
        print("raise FlextExceptions.ValidationError(")
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
        print("raise FlextExceptions.ConnectionError(")
        print("    'Connection failed',")
        print("    service='database',")
        print("    correlation_id=request_id")
        print(")")

        # OLD: No metrics tracking
        warnings.warn(
            "Unmonitored exceptions are DEPRECATED! FlextExceptions tracks metrics.",
            DeprecationWarning,
            stacklevel=2,
        )
        print("\n❌ OLD WAY (no metrics):")
        print("# Exceptions happen without tracking")

        print("\n✅ CORRECT WAY (automatic metrics):")
        print("# FlextExceptions automatically tracks all exceptions")
        print("metrics = FlextExceptions.get_metrics()")


def main() -> None:
    """Main entry point demonstrating all FlextExceptions capabilities."""
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

    # Deprecation warnings
    service.demonstrate_deprecated_patterns()

    print("\n" + "=" * 60)
    print("✅ ALL FlextExceptions methods demonstrated!")
    print("🎯 Next: Run final quality gates on all examples")
    print("=" * 60)


if __name__ == "__main__":
    main()
