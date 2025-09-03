#!/usr/bin/env python3
"""01 - Railway-Oriented Programming with FLEXT patterns.

Demonstrates FlextResult[T] railway patterns using maximum FLEXT integration.
Follows FLEXT_REFACTORING_PROMPT.md strictly for proper ABI compliance.

Architecture Overview:
    Uses maximum FlextModels, FlextConstants, FlextUtilities for centralized patterns.
    All business logic returns FlextResult[T] for type-safe error handling.
    Implements proper SOLID principles with protocol-based design.
    Consolidated into single service class following FLEXT patterns.
"""

from __future__ import annotations

import json
from decimal import Decimal

from flext_core import (
    FlextConstants,
    FlextModels,
    FlextResult,
    FlextTypes,
    FlextUtilities,
)

# =============================================================================
# DOMAIN ENTITIES - Using FlextModels with proper inheritance
# =============================================================================


class User(FlextModels.Entity):
    """Domain user entity with built-in lifecycle management.

    Uses FlextModels.Entity for proper domain modeling with validation.
    """

    name: str
    email: str
    age: int
    status: str = FlextConstants.Status.ACTIVE

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules for user entity."""
        min_name_length = 2
        max_age = 120

        if len(self.name) < min_name_length:
            return FlextResult[None].fail("Name must be at least 2 characters")

        if "@" not in self.email:
            return FlextResult[None].fail("Invalid email format")

        if self.age < 0 or self.age > max_age:
            return FlextResult[None].fail(f"Age must be between 0 and {max_age}")

        return FlextResult[None].ok(None)


# =============================================================================
# STRUCTURED DATA MODELS - Using FlextModels with proper inheritance
# =============================================================================


class UserRegistrationRequest(FlextModels.Value):
    """Registration request using FlextModels.Value for immutability and validation.

    Follows SOLID principles with proper validation using FlextResult patterns.
    """

    name: str
    email: str
    age: int

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules using centralized validation protocols.

        Returns:
            FlextResult indicating validation success or failure.

        """
        if not self.name.strip():
            return FlextResult[None].fail(FlextConstants.Errors.VALIDATION_ERROR)
        if "@" not in self.email:
            return FlextResult[None].fail(FlextConstants.Errors.VALIDATION_ERROR)
        return FlextResult[None].ok(None)


class RegistrationResult(FlextModels.BaseConfig):
    """Registration result using FlextModels.BaseConfig for enterprise features.

    Follows SOLID principles with proper typing and validation.
    """

    user_id: str
    status: str
    processing_time_ms: float
    correlation_id: str


class BatchResult(FlextModels.BaseConfig):
    """Batch processing result with enterprise metrics.

    Uses proper typing for consistent data handling.
    """

    total: int
    successful: int
    failed: int
    success_rate: float
    results: list[RegistrationResult]
    errors: list[str]


# =============================================================================
# RAILWAY PROCESSING - Consolidated service using flext-core patterns
# =============================================================================


class UserRegistrationService:
    """Consolidated user registration service using FlextUtilities and FlextResult patterns.

    Implements all registration functionality in a single service following FLEXT patterns.
    Uses FlextUtilities for ID generation and FlextResult for error handling.
    """

    def __init__(self) -> None:
        """Initialize service with FlextUtilities."""
        self._utilities = FlextUtilities()

    def process_registration(
        self, request: UserRegistrationRequest
    ) -> FlextResult[RegistrationResult]:
        """Process user registration with proper validation and error handling.

        Args:
            request: Registration request to process.

        Returns:
            FlextResult containing RegistrationResult or error.

        """
        # Validate request
        validation_result = request.validate_business_rules()
        if validation_result.is_failure:
            return FlextResult[RegistrationResult].fail(
                validation_result.error or FlextConstants.Errors.VALIDATION_ERROR
            )

        # Generate correlation ID
        correlation_id = self._utilities.generate_correlation_id()

        # Create user entity with proper ID generation
        entity_id = self._utilities.generate_entity_id()
        try:
            user = User(
                id=entity_id,
                name=request.name,
                email=request.email,
                age=request.age,
                status=FlextConstants.Status.ACTIVE,
            )
        except Exception as e:
            return FlextResult[RegistrationResult].fail(f"Entity creation failed: {e}")

        # Build registration result
        processing_time = 1.0  # Simplified timing
        result = RegistrationResult(
            user_id=str(user.id),
            status=str(user.status),
            processing_time_ms=processing_time,
            correlation_id=correlation_id,
        )

        return FlextResult[RegistrationResult].ok(result)

    def process_batch(
        self, requests: list[UserRegistrationRequest]
    ) -> FlextResult[BatchResult]:
        """Process batch of registration requests.

        Args:
            requests: List of registration requests to process.

        Returns:
            FlextResult containing BatchResult with success metrics.

        """
        if not requests:
            return FlextResult[BatchResult].fail(FlextConstants.Errors.VALIDATION_ERROR)

        successful_results: list[RegistrationResult] = []
        errors: list[str] = []

        # Process each request individually
        for request in requests:
            result = self.process_registration(request)
            if result.is_success:
                successful_results.append(result.unwrap())
            else:
                errors.append(result.error or "Unknown error")

        # Build batch result with metrics
        total = len(requests)
        successful_count = len(successful_results)
        failed_count = len(errors)
        success_rate = successful_count / total if total > 0 else 0.0

        batch_result = BatchResult(
            total=total,
            successful=successful_count,
            failed=failed_count,
            success_rate=success_rate,
            results=successful_results,
            errors=errors,
        )

        return FlextResult[BatchResult].ok(batch_result)

    def process_json_registration(
        self, json_data: str
    ) -> FlextResult[RegistrationResult]:
        """Process JSON registration data.

        Args:
            json_data: JSON string containing registration data.

        Returns:
            FlextResult containing RegistrationResult or error.

        """
        # Parse JSON data using FlextUtilities
        try:
            data = FlextUtilities.ProcessingUtils.safe_json_parse(json_data)
            if not data:
                return FlextResult[RegistrationResult].fail("Invalid JSON data")

            # Type-safe conversion for UserRegistrationRequest using FlextUtilities
            age_value = data.get("age", 0)
            request = UserRegistrationRequest(
                name=str(data.get("name", "")),
                email=str(data.get("email", "")),
                age=FlextUtilities.Conversions.safe_int(age_value, 0),
            )
        except Exception as e:
            return FlextResult[RegistrationResult].fail(f"JSON parsing failed: {e}")

        # Process the request
        return self.process_registration(request)


# =============================================================================
# DEMONSTRATIONS - Consolidated demo using single service
# =============================================================================


def demonstrate_railway_patterns() -> None:
    """Demonstrate all railway patterns using consolidated UserRegistrationService."""
    service = UserRegistrationService()

    print("🚀 FLEXT Railway Pattern Demonstrations")
    print("=" * 50)

    # 1. Single registration processing
    print("\n1. Single Registration Processing:")
    request = UserRegistrationRequest(
        name="Alice Johnson", email="alice@company.com", age=28
    )

    result = service.process_registration(request)
    if result.is_success:
        print(f"✅ Registration successful: {result.value.user_id}")
        print(f"   Status: {result.value.status}")
        print(f"   Processing time: {result.value.processing_time_ms}ms")
    else:
        print(f"❌ Registration failed: {result.error}")

    # 2. Batch processing
    print("\n2. Batch Processing:")
    batch_requests = [
        UserRegistrationRequest(name="User 1", email="user1@company.com", age=25),
        UserRegistrationRequest(name="User 2", email="user2@company.com", age=30),
        UserRegistrationRequest(name="User 3", email="user3@company.com", age=35),
    ]

    batch_result = service.process_batch(batch_requests)
    if batch_result.is_success:
        batch_data = batch_result.value
        print(f"✅ Batch completed: {batch_data.successful}/{batch_data.total}")
        print(f"   Success rate: {batch_data.success_rate}")
        print(f"   Errors: {len(batch_data.errors)}")
    else:
        print(f"❌ Batch processing failed: {batch_result.error}")

    # 3. JSON processing
    print("\n3. JSON Processing:")
    valid_json = '{"name": "JSON User", "email": "json@company.com", "age": 32}'
    json_result = service.process_json_registration(valid_json)
    if json_result.is_success:
        print(f"✅ JSON processed: {json_result.value.user_id}")
        print(f"   Correlation ID: {json_result.value.correlation_id}")
    else:
        print(f"❌ JSON processing failed: {json_result.error}")

    # 4. Error handling demonstration
    print("\n4. Error Handling Demonstration:")
    invalid_request = UserRegistrationRequest(name="", email="invalid-email", age=-5)

    error_result = service.process_registration(invalid_request)
    if error_result.is_failure:
        print(f"✅ Error handling working: {error_result.error}")
    else:
        print("❌ Error handling failed - should have caught validation errors")

    print("\n🎯 Railway Pattern Demo Complete!")


def main() -> None:
    """Main entry point for railway pattern demonstrations."""
    demonstrate_railway_patterns()


# Rebuild Pydantic models to resolve forward references
RegistrationResult.model_rebuild()
BatchResult.model_rebuild()

if __name__ == "__main__":
    main()
