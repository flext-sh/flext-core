#!/usr/bin/env python3
"""01 - Railway-Oriented Programming with FLEXT patterns.

Demonstrates FlextResult[T] railway patterns using maximum FLEXT integration.
Follows FLEXT_REFACTORING_PROMPT.md strictly for proper ABI compliance.

Architecture Overview:
    Uses maximum FlextTypes, FlextConstants, FlextProtocols for centralized patterns.
    All business logic returns FlextResult[T] for type-safe error handling.
    Implements proper SOLID principles with protocol-based design.
"""

from __future__ import annotations

import json
from decimal import Decimal

from flext_core import (
    FlextConstants,
    FlextCore,
    FlextModels,
    FlextResult,
    FlextTypes,
    FlextUtilities,
)

# Singleton FlextCore instance for all utilities
core = FlextCore.get_instance()

# =============================================================================
# DOMAIN ENTITIES - Using FlextModels.Entity with FlextCore factory
# =============================================================================


class User(FlextModels.Entity):
    """Domain user entity with built-in lifecycle management.

    Uses FlextFields for validation and FlextTypes for annotations.
    """

    name: FlextTypes.Core.String
    email: FlextTypes.Core.String
    age: FlextTypes.Core.Integer
    status: FlextTypes.Core.String = FlextConstants.Status.ACTIVE


# =============================================================================
# STRUCTURED DATA MODELS - Using FlextModels + FlextModels.Value base
# =============================================================================


class UserRegistrationRequest(FlextModels.Value):
    """Registration request using FlextModels.Value for immutability and validation.

    Follows SOLID principles with proper validation using FlextProtocols.
    """

    name: FlextTypes.Core.String
    email: FlextTypes.Core.String
    age: FlextTypes.Core.Integer

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
    """Registration result using FlextModels for enterprise features.

    Follows SOLID principles with proper typing using FlextTypes.
    """

    user_id: FlextTypes.Core.String
    status: FlextTypes.Core.String
    processing_time_ms: FlextTypes.Core.Float
    correlation_id: FlextTypes.Core.String


class BatchResult(FlextModels.BaseConfig):
    """Batch processing result with enterprise metrics.

    Uses FlextTypes for consistent typing across the ecosystem.
    """

    total: FlextTypes.Core.Integer
    successful: FlextTypes.Core.Integer
    failed: FlextTypes.Core.Integer
    success_rate: Decimal
    results: list[RegistrationResult]
    errors: list[FlextTypes.Core.String]


# =============================================================================
# RAILWAY PROCESSING - Maximum flext-core usage
# =============================================================================


class RegistrationProcessor:
    """Registration processor using FlextProtocols for proper typing.

    Implements ProcessorProtocol following SOLID principles and proper ABI.
    """

    def __init__(self) -> None:
        """Initialize processor with FlextUtilities."""
        self._utilities = FlextUtilities()

    def process(self, request: UserRegistrationRequest) -> FlextResult[User]:
        """Process user registration with proper validation.

        Args:
            request: Registration request to process.

        Returns:
            FlextResult containing created User or error message.

        """
        validation_result = request.validate_business_rules()
        if validation_result.is_failure:
            return FlextResult[User].fail(
                validation_result.error or FlextConstants.Errors.VALIDATION_ERROR
            )

        # Create user entity with proper ID generation
        entity_id = self._utilities.generate_uuid()
        try:
            user = User(
                id=f"user_{entity_id[:10]}",
                name=request.name,
                email=request.email,
                age=request.age,
                status=FlextConstants.Status.ACTIVE,
            )
            return FlextResult[User].ok(user)
        except Exception as e:
            return FlextResult[User].fail(f"Entity creation failed: {e}")

    def build_result(
        self, user: User, correlation_id: FlextTypes.Core.String
    ) -> RegistrationResult:
        """Build registration result with timing metrics.

        Args:
            user: Created user entity.
            correlation_id: Operation correlation ID.

        Returns:
            Registration result with timing information.

        """
        processing_time = 1.0  # Simplified timing
        return RegistrationResult(
            user_id=str(user.id),
            status=str(user.status),
            processing_time_ms=processing_time,
            correlation_id=correlation_id,
        )

    def process_registration(
        self, request: UserRegistrationRequest
    ) -> FlextResult[RegistrationResult]:
        """Process user registration with proper error handling.

        Args:
            request: User registration request.

        Returns:
            FlextResult containing RegistrationResult or error.

        """
        correlation_id = self._utilities.generate_uuid()

        # Process the request
        user_result = self.process(request)
        if user_result.is_failure:
            return FlextResult[RegistrationResult].fail(
                user_result.error or "Registration processing failed"
            )

        # Build the result
        result = self.build_result(user_result.unwrap(), correlation_id)
        return FlextResult[RegistrationResult].ok(result)


class BatchProcessor:
    """Ultra-lean batch processor using FlextServiceProcessor template - ZERO logic."""

    def __init__(self) -> None:
        """Initialize batch processor with utilities."""
        self._registration_processor = RegistrationProcessor()
        self._utilities = FlextUtilities()

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
        errors: list[FlextTypes.Core.String] = []

        # Process each request individually
        for request in requests:
            result = self._registration_processor.process_registration(request)
            if result.is_success:
                successful_results.append(result.unwrap())
            else:
                errors.append(result.error or "Unknown error")

        # Build batch result with metrics
        total = len(requests)
        successful_count = len(successful_results)
        failed_count = len(errors)
        success_rate = (
            Decimal(str(successful_count / total)) if total > 0 else Decimal("0.0")
        )

        batch_result = BatchResult(
            total=total,
            successful=successful_count,
            failed=failed_count,
            success_rate=success_rate,
            results=successful_results,
            errors=errors,
        )

        return FlextResult[BatchResult].ok(batch_result)


# =============================================================================
# JSON PROCESSING - Simple JSON handling
# =============================================================================


class JSONProcessor:
    """JSON processor using standard FLEXT patterns.

    Handles JSON parsing with proper error handling.
    """

    def __init__(self) -> None:
        """Initialize processor with utilities."""
        self._registration_processor = RegistrationProcessor()

    def process_json_registration(
        self, json_data: str
    ) -> FlextResult[RegistrationResult]:
        """Process JSON registration data.

        Args:
            json_data: JSON string containing registration data.

        Returns:
            FlextResult containing RegistrationResult or error.

        """
        # Parse JSON data
        try:
            data = json.loads(json_data)
            request = UserRegistrationRequest(**data)
        except Exception as e:
            return FlextResult[RegistrationResult].fail(f"JSON parsing failed: {e}")

        # Process the request
        return self._registration_processor.process_registration(request)


# =============================================================================
# DEMONSTRATIONS - Simple demos using corrected classes
# =============================================================================


def demo_railway_processing() -> None:
    """Demonstrate railway processing with flext-core components."""
    processor = RegistrationProcessor()

    # Valid request using FlextModels.Value
    request = UserRegistrationRequest(
        name="Alice Johnson", email="alice@company.com", age=28
    )

    result = processor.process_registration(request)
    if result.is_success:
        print(f"Registration successful: {result.value.user_id}")


def demo_batch_processing() -> None:
    """Demonstrate batch processing."""
    batch_processor = BatchProcessor()

    # Create batch requests using FlextModels.Value
    requests = [
        UserRegistrationRequest(name="User 1", email="user1@company.com", age=25),
        UserRegistrationRequest(name="User 2", email="user2@company.com", age=30),
        UserRegistrationRequest(name="User 3", email="user3@company.com", age=35),
    ]

    result = batch_processor.process_batch(requests)
    if result.is_success:
        batch_result = result.value
        print(f"Batch completed: {batch_result.successful}/{batch_result.total}")


def demo_json_processing() -> None:
    """Demonstrate JSON processing with structured validation."""
    json_processor = JSONProcessor()

    # Process valid JSON
    valid_json = '{"name": "JSON User", "email": "json@company.com", "age": 32}'
    result = json_processor.process_json_registration(valid_json)
    if result.is_success:
        print(f"JSON processed: {result.value.user_id}")


def main() -> None:
    """Railway Pattern with Maximum FLEXT-Core Usage."""
    print("Railway Pattern Demonstrations:")
    demo_railway_processing()
    demo_batch_processing()
    demo_json_processing()


if __name__ == "__main__":
    main()
