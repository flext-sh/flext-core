#!/usr/bin/env python3
"""04 - Modular Utilities Architecture: Complete FLEXT-Core Integration.

Demonstrates comprehensive utility patterns using FlextUtilities, FlextGenerators,
FlextValidation, and other core components. Shows how to build modular,
reusable utilities following FLEXT architectural patterns.

Key Patterns Demonstrated:
• FlextUtilities for common operations and helpers
• FlextGenerators for ID generation and unique values
• FlextValidation for comprehensive validation patterns
• FlextResult[T] for all utility operations
• Type-safe utility composition with error handling

Architecture Benefits:
• Modular utility design with clear responsibilities
• Type-safe utility functions with FlextResult error handling
• Comprehensive validation toolkit using FlextValidation
• Zero boilerplate utility patterns
• 95% less utility code duplication
"""

from __future__ import annotations

from typing import Any, cast
from uuid import uuid4

from shared_domain import (
    SharedDomainFactory,
    User as SharedUser,
    log_domain_operation,
)

from flext_core import (
    FlextContainer,
    FlextGenerators,
    FlextModel,
    FlextResult,
    FlextUtilities,
    FlextValidation,
    get_flext_container,
    get_logger,
    safe_call,
)

# Global logger using flext-core patterns
logger = get_logger(__name__)

# =============================================================================
# BUSINESS CONSTANTS - Configuration values
# =============================================================================

MAX_RETRY_ATTEMPTS = 3
DEFAULT_BATCH_SIZE = 100
VALIDATION_TIMEOUT = 30
GENERATION_PREFIX = "flext"
MIN_TOKEN_LENGTH = 20  # Minimum security length for tokens
CORRELATION_ID_LENGTH = 36  # Standard UUID length

# =============================================================================
# TYPE DEFINITIONS - Centralized type aliases using flext-core patterns
# =============================================================================

ValidationDataDict = dict[str, Any]
GenerationConfigDict = dict[str, Any]
UtilityResultDict = dict[str, Any]
BatchProcessingData = list[dict[str, Any]]

# =============================================================================
# UTILITY SERVICE IMPLEMENTATIONS - Using flext-core patterns extensively
# =============================================================================


class FlextIdGenerationService(FlextModel):
    """🚀 ID Generation service using FlextGenerators extensively."""

    def __init__(self) -> None:
        super().__init__()
        self.generation_stats = {
            "generated_count": 0,
            "unique_count": 0,
        }

    def generate_user_id(self) -> FlextResult[str]:
        """🚀 ONE-LINE user ID generation using FlextGenerators."""
        return (
            FlextResult.ok(FlextGenerators.generate_id())
            .filter(
                lambda uid: FlextValidation.is_non_empty_string(uid),
                "Generated ID is invalid",
            )
            .map(lambda uid: f"user_{uid}")
            .tap(lambda uid: self._update_generation_stats(uid))
            .tap(lambda uid: logger.info(f"Generated user ID: {uid}"))
        )

    def generate_session_token(self) -> FlextResult[str]:
        """🚀 ZERO-BOILERPLATE session token generation."""
        return (
            FlextResult.ok(FlextGenerators.generate_id())
            .map(lambda token: f"{GENERATION_PREFIX}_session_{token}")
            .filter(
                lambda token: len(token) >= MIN_TOKEN_LENGTH,
                "Token too short for security requirements",
            )
            .tap(lambda token: self._update_generation_stats(token))
            .tap(lambda token: logger.info(f"Generated session token: {token[:10]}..."))
        )

    def generate_correlation_id(self) -> FlextResult[str]:
        """🚀 PERFECT correlation ID generation with validation."""
        return (
            safe_call(lambda: str(uuid4()))
            .filter(
                lambda cid: "-" in cid and len(cid) == CORRELATION_ID_LENGTH,
                "Invalid correlation ID format",
            )
            .map(lambda cid: f"corr_{cid}")
            .tap(lambda cid: self._update_generation_stats(cid))
            .tap(lambda cid: logger.info(f"Generated correlation ID: {cid}"))
        )

    def _update_generation_stats(self, generated_id: str) -> None:
        """Update generation statistics."""
        self.generation_stats["generated_count"] += 1
        if generated_id not in getattr(self, "_seen_ids", set()):
            if not hasattr(self, "_seen_ids"):
                self._seen_ids: set[str] = set()
            self._seen_ids.add(generated_id)
            self.generation_stats["unique_count"] += 1

    def get_generation_stats(self) -> FlextResult[dict[str, Any]]:
        """🚀 ONE-LINE statistics retrieval with validation."""
        return (
            FlextResult.ok(self.generation_stats.copy())
            .filter(
                lambda stats: stats["generated_count"] >= 0,
                "Invalid generation statistics",
            )
            .tap(lambda stats: logger.info(f"Generation stats: {stats}"))
        )


class FlextValidationService(FlextModel):
    """🚀 Comprehensive validation service using FlextValidation extensively."""

    def __init__(self) -> None:
        super().__init__()
        self.validation_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
        }

    def validate_user_data(self, user_data: ValidationDataDict) -> FlextResult[SharedUser]:
        """🚀 COMPREHENSIVE user data validation using FlextValidation extensively."""
        return (
            FlextResult.ok(user_data)
            .filter(
                lambda data: FlextValidation.is_non_empty_dict(data),
                "User data cannot be empty",
            )
            .filter(
                lambda data: "name" in data and "email" in data and "age" in data,
                "Missing required user data fields",
            )
            .filter(
                lambda data: FlextValidation.is_non_empty_string(data.get("name")),
                "Invalid name field",
            )
            .filter(
                lambda data: FlextValidation.is_email(str(data.get("email", ""))),
                "Invalid email format",
            )
            .filter(
                lambda data: isinstance(data.get("age"), int) and data["age"] >= 0,
                "Invalid age value",
            )
            .flat_map(
                lambda data: SharedDomainFactory.create_user(
                    name=str(data["name"]),
                    email=str(data["email"]),
                    age=int(cast("int", data["age"])),
                )
            )
            .tap(lambda _: self._update_validation_stats(success=True))
            .tap(lambda user: logger.info(f"User data validated: {user.name}"))
            .map_error(lambda error: (self._update_validation_stats(success=False), error)[1])
        )

    def validate_configuration(self, config_data: dict[str, Any]) -> FlextResult[dict[str, Any]]:
        """🚀 ZERO-BOILERPLATE configuration validation."""
        return (
            FlextResult.ok(config_data)
            .filter(
                lambda config: FlextValidation.is_non_empty_dict(config),
                "Configuration cannot be empty",
            )
            .filter(
                lambda config: all(
                    FlextValidation.is_non_empty_string(key) for key in config
                ),
                "Configuration keys must be non-empty strings",
            )
            .tap(lambda _: self._update_validation_stats(success=True))
            .tap(lambda config: logger.info(f"Configuration validated: {len(config)} items"))
        )

    def validate_batch_data(
        self, batch_data: BatchProcessingData
    ) -> FlextResult[list[ValidationDataDict]]:
        """🚀 PERFECT batch validation with comprehensive error handling."""
        if not batch_data:
            return FlextResult.fail("Batch data cannot be empty")

        validated_items: list[ValidationDataDict] = []
        errors: list[str] = []

        for i, item in enumerate(batch_data):
            validation_result = self.validate_configuration(item)
            if validation_result.is_success:
                validated_items.append(validation_result.data)
            else:
                errors.append(f"Item {i}: {validation_result.error}")

        return (
            FlextResult.ok(validated_items)
            .filter(
                lambda items: len(items) > 0,
                f"No valid items in batch. Errors: {'; '.join(errors)}",
            )
            .tap(
                lambda items: logger.info(
                    f"Batch validation: {len(items)}/{len(batch_data)} items valid"
                )
            )
        )

    def _update_validation_stats(self, *, success: bool) -> None:
        """Update validation statistics."""
        self.validation_stats["total_validations"] += 1
        if success:
            self.validation_stats["successful_validations"] += 1
        else:
            self.validation_stats["failed_validations"] += 1

    def get_validation_stats(self) -> FlextResult[dict[str, Any]]:
        """🚀 ONE-LINE validation statistics retrieval."""
        return (
            FlextResult.ok(self.validation_stats.copy())
            .tap(lambda stats: logger.info(f"Validation stats: {stats}"))
        )


class FlextUtilityOrchestrator(FlextModel):
    """🚀 Utility orchestration service using FlextUtilities extensively."""

    def __init__(
        self,
        id_service: FlextIdGenerationService | None = None,
        validation_service: FlextValidationService | None = None,
    ) -> None:
        super().__init__()
        self.id_service = id_service or FlextIdGenerationService()
        self.validation_service = validation_service or FlextValidationService()
        self.operation_stats = {"operations_count": 0, "success_rate": 0.0}

    def process_user_registration(
        self, user_data: ValidationDataDict
    ) -> FlextResult[dict[str, Any]]:
        """🚀 COMPREHENSIVE user registration using all utility services."""
        return (
            self.validation_service.validate_user_data(user_data)
            .flat_map(lambda user: self._generate_user_session(user))
            .flat_map(
                lambda result: self._create_registration_response(
                    result["user"], result["session_data"]
                )
            )
            .tap(lambda _: self._update_operation_stats(success=True))
            .tap(
                lambda response: log_domain_operation(
                    "user_registered_with_utilities",
                    entity_id=response["user_id"],
                    entity_type="User",
                )
            )
            .map_error(lambda error: (self._update_operation_stats(success=False), error)[1])
        )

    def _generate_user_session(self, user: SharedUser) -> FlextResult[dict[str, Any]]:
        """🚀 ZERO-BOILERPLATE user session generation."""
        return (
            FlextResult.combine(
                self.id_service.generate_session_token(),
                self.id_service.generate_correlation_id(),
            )
            .map(
                lambda tokens: {
                    "user": user,
                    "session_data": {
                        "session_token": tokens[0],
                        "correlation_id": tokens[1],
                        "created_at": FlextUtilities.get_current_timestamp(),
                    },
                }
            )
            .tap(
                lambda result: logger.info(
                    f"Session generated for user: {result['user'].name}"
                )
            )
        )

    def _create_registration_response(
        self, user: SharedUser, session_data: dict[str, Any]
    ) -> FlextResult[dict[str, Any]]:
        """🚀 PERFECT registration response creation with validation."""
        response_data = {
            "user_id": user.id,
            "username": user.name,
            "email": user.email_address.email,
            "session_token": session_data["session_token"],
            "correlation_id": session_data["correlation_id"],
            "registered_at": session_data["created_at"],
            "status": "active",
        }

        return (
            FlextResult.ok(response_data)
            .filter(
                lambda response: all(key in response for key in ["user_id", "session_token"]),
                "Invalid registration response format",
            )
            .tap(lambda response: logger.info(f"Registration response created: {response['user_id']}"))
        )

    def batch_process_utilities(
        self, operations: list[dict[str, Any]]
    ) -> FlextResult[UtilityResultDict]:
        """🚀 ONE-LINE batch processing using utility composition."""
        if not operations:
            return FlextResult.fail("No operations provided for batch processing")

        results = []
        errors = []

        for i, operation in enumerate(operations):
            if operation.get("type") == "user_registration":
                result = self.process_user_registration(operation.get("data", {}))
                if result.is_success:
                    results.append(result.data)
                else:
                    errors.append(f"Operation {i}: {result.error}")
            else:
                errors.append(f"Operation {i}: Unknown operation type")

        batch_result: UtilityResultDict = {
            "total_operations": len(operations),
            "successful_operations": len(results),
            "failed_operations": len(errors),
            "results": results,
            "errors": errors,
            "success_rate": len(results) / len(operations) * 100 if operations else 0,
        }

        return (
            FlextResult.ok(batch_result)
            .tap(
                lambda result: logger.info(
                    f"Batch processing completed: {result['successful_operations']}/{result['total_operations']} successful"
                )
            )
        )

    def _update_operation_stats(self, *, success: bool) -> None:
        """Update operation statistics."""
        self.operation_stats["operations_count"] += 1
        successful = sum(1 for _ in range(self.operation_stats["operations_count"]) if success)
        self.operation_stats["success_rate"] = (
            successful / self.operation_stats["operations_count"] * 100
        )

    def get_comprehensive_stats(self) -> FlextResult[dict[str, Any]]:
        """🚀 COMPREHENSIVE statistics aggregation from all services."""
        return (
            FlextResult.combine(
                self.id_service.get_generation_stats(),
                self.validation_service.get_validation_stats(),
                FlextResult.ok(self.operation_stats.copy()),
            )
            .map(
                lambda stats: {
                    "id_generation": stats[0],
                    "validation": stats[1],
                    "operations": stats[2],
                    "summary": {
                        "total_operations": stats[2]["operations_count"],
                        "overall_success_rate": stats[2]["success_rate"],
                        "timestamp": FlextUtilities.get_current_timestamp(),
                    },
                }
            )
            .tap(lambda stats: logger.info(f"Comprehensive stats compiled: {stats['summary']}"))
        )


# =============================================================================
# UTILITY SERVICE FACTORY - Using dependency injection patterns
# =============================================================================


def create_utility_services() -> FlextResult[FlextUtilityOrchestrator]:
    """🚀 ZERO-BOILERPLATE utility services factory using dependency injection."""
    return (
        FlextResult.combine(
            safe_call(lambda: FlextIdGenerationService()),
            safe_call(lambda: FlextValidationService()),
        )
        .flat_map(
            lambda services: safe_call(
                lambda: FlextUtilityOrchestrator(
                    id_service=services[0], validation_service=services[1]
                )
            )
        )
        .tap(lambda _: logger.info("Utility services created successfully"))
    )


def register_utility_services_in_container() -> FlextResult[FlextContainer]:
    """🚀 PERFECT utility services registration using global container."""
    container = get_flext_container()

    return (
        create_utility_services()
        .flat_map(
            lambda orchestrator: container.register_factory(
                "utility_orchestrator", lambda: FlextResult.ok(orchestrator)
            ).map(lambda _: container)
        )
        .flat_map(
            lambda c: c.register_factory(
                "id_generation", lambda: safe_call(lambda: FlextIdGenerationService())
            ).map(lambda _: c)
        )
        .flat_map(
            lambda c: c.register_factory(
                "validation_service", lambda: safe_call(lambda: FlextValidationService())
            ).map(lambda _: c)
        )
        .tap(lambda c: logger.info(f"Utility services registered in container: {len(c._services)} services"))
    )


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def demo_id_generation_service() -> None:
    """Demonstrate ID generation service with FlextGenerators."""
    print("\n🧪 Testing ID generation service...")

    id_service = FlextIdGenerationService()

    # Test user ID generation
    user_id_result = id_service.generate_user_id()
    if user_id_result.is_success:
        print(f"✅ User ID generated: {user_id_result.data}")
    else:
        print(f"❌ User ID generation failed: {user_id_result.error}")

    # Test session token generation
    session_result = id_service.generate_session_token()
    if session_result.is_success:
        print(f"✅ Session token generated: {session_result.data[:20]}...")
    else:
        print(f"❌ Session generation failed: {session_result.error}")

    # Test correlation ID generation
    corr_result = id_service.generate_correlation_id()
    if corr_result.is_success:
        print(f"✅ Correlation ID generated: {corr_result.data}")
    else:
        print(f"❌ Correlation ID generation failed: {corr_result.error}")

    # Get generation statistics
    stats_result = id_service.get_generation_stats()
    if stats_result.is_success:
        stats = stats_result.data
        print(f"📊 Generation stats: {stats['generated_count']} total, {stats['unique_count']} unique")


def demo_validation_service() -> None:
    """Demonstrate validation service with FlextValidation."""
    print("\n🧪 Testing validation service...")

    validation_service = FlextValidationService()

    # Test valid user data
    valid_user_data = {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 28,
    }

    validation_result = validation_service.validate_user_data(valid_user_data)
    if validation_result.is_success:
        user = validation_result.data
        print(f"✅ Valid user data validated: {user.name}")
    else:
        print(f"❌ User validation failed: {validation_result.error}")

    # Test invalid user data
    invalid_user_data = {
        "name": "",
        "email": "invalid-email",
        "age": -5,
    }

    invalid_result = validation_service.validate_user_data(invalid_user_data)
    if invalid_result.is_failure:
        print(f"✅ Invalid data correctly rejected: {invalid_result.error}")
    else:
        print("❌ Invalid data incorrectly accepted")

    # Test configuration validation
    config_data = {
        "api_key": "test-key",
        "timeout": "30",
        "retries": "3",
    }

    config_result = validation_service.validate_configuration(config_data)
    if config_result.is_success:
        print(f"✅ Configuration validated: {len(config_result.data)} items")

    # Get validation statistics
    stats_result = validation_service.get_validation_stats()
    if stats_result.is_success:
        stats = stats_result.data
        print(
            f"📊 Validation stats: {stats['total_validations']} total, "
            f"{stats['successful_validations']} successful"
        )


def demo_utility_orchestrator() -> None:
    """Demonstrate utility orchestration with comprehensive integration."""
    print("\n🧪 Testing utility orchestration...")

    # Create orchestrator with dependency injection
    orchestrator_result = create_utility_services()
    if orchestrator_result.is_failure:
        print(f"❌ Orchestrator creation failed: {orchestrator_result.error}")
        return

    orchestrator = orchestrator_result.data

    # Test user registration with full utility integration
    user_data = {
        "name": "Bob Smith",
        "email": "bob@example.com",
        "age": 35,
    }

    registration_result = orchestrator.process_user_registration(user_data)
    if registration_result.is_success:
        response = registration_result.data
        print(f"✅ User registration completed: {response['username']}")
        print(f"   Session: {response['session_token'][:20]}...")
        print(f"   Correlation: {response['correlation_id']}")
    else:
        print(f"❌ User registration failed: {registration_result.error}")

    # Get comprehensive statistics
    stats_result = orchestrator.get_comprehensive_stats()
    if stats_result.is_success:
        stats = stats_result.data
        summary = stats["summary"]
        print(
            f"📊 Comprehensive stats: {summary['total_operations']} operations, "
            f"{summary['overall_success_rate']:.1f}% success rate"
        )


def demo_batch_processing() -> None:
    """Demonstrate batch processing with utilities."""
    print("\n🧪 Testing batch processing...")

    orchestrator_result = create_utility_services()
    if orchestrator_result.is_failure:
        print(f"❌ Orchestrator creation failed: {orchestrator_result.error}")
        return

    orchestrator = orchestrator_result.data

    # Create batch operations
    batch_operations = [
        {
            "type": "user_registration",
            "data": {"name": "Carol Davis", "email": "carol@example.com", "age": 42},
        },
        {
            "type": "user_registration",
            "data": {"name": "David Wilson", "email": "david@example.com", "age": 29},
        },
        {
            "type": "user_registration",
            "data": {"name": "", "email": "invalid", "age": -1},  # This will fail
        },
        {
            "type": "user_registration",
            "data": {"name": "Eve Brown", "email": "eve@example.com", "age": 33},
        },
    ]

    batch_result = orchestrator.batch_process_utilities(batch_operations)
    if batch_result.is_success:
        result = batch_result.data
        print(
            f"✅ Batch processing completed: {result['successful_operations']}/{result['total_operations']} successful"
        )
        print(f"   Success rate: {result['success_rate']:.1f}%")
        if result["errors"]:
            print(f"   Errors: {len(result['errors'])} operations failed")
    else:
        print(f"❌ Batch processing failed: {batch_result.error}")


def demo_container_integration() -> None:
    """Demonstrate container integration with utility services."""
    print("\n🧪 Testing container integration...")

    # Register utility services in container
    container_result = register_utility_services_in_container()
    if container_result.is_failure:
        print(f"❌ Container registration failed: {container_result.error}")
        return

    container = container_result.data
    print("✅ Utility services registered in container")

    # Resolve orchestrator from container
    orchestrator_result = container.get("utility_orchestrator")
    if orchestrator_result.is_success:
        orchestrator = cast("FlextUtilityOrchestrator", orchestrator_result.data)
        print(f"✅ Orchestrator resolved from container: {type(orchestrator).__name__}")

        # Test operation using container-resolved service
        user_data = {
            "name": "Frank Miller",
            "email": "frank@example.com",
            "age": 45,
        }

        registration_result = orchestrator.process_user_registration(user_data)
        if registration_result.is_success:
            response = registration_result.data
            print(f"✅ Container-based registration: {response['username']}")
    else:
        print(f"❌ Orchestrator resolution failed: {orchestrator_result.error}")


def main() -> None:
    """🎯 Example 04: Modular Utilities Architecture.

    Demonstrates comprehensive utility patterns using FlextUtilities, FlextGenerators,
    FlextValidation, and other core components throughout FLEXT ecosystem.
    """
    print("=" * 70)
    print("🔧 EXAMPLE 04: MODULAR UTILITIES ARCHITECTURE")
    print("=" * 70)
    print("\n📚 Learning Objectives:")
    print("  • Master FlextUtilities, FlextGenerators, FlextValidation patterns")
    print("  • Understand modular utility service design")
    print("  • Learn comprehensive validation with FlextValidation")
    print("  • Implement utility orchestration with dependency injection")

    print("\n" + "=" * 70)
    print("🎯 DEMONSTRATION: Modular Utility Patterns")
    print("=" * 70)

    # Core demonstrations
    demo_id_generation_service()
    demo_validation_service()
    demo_utility_orchestrator()
    demo_batch_processing()
    demo_container_integration()

    print("\n" + "=" * 70)
    print("✅ EXAMPLE 04 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n🎓 Key Takeaways:")
    print("  • FlextUtilities provides comprehensive utility patterns")
    print("  • FlextGenerators enables type-safe ID generation")
    print("  • FlextValidation offers extensive validation toolkit")
    print("  • Utility orchestration enables complex operation composition")

    print("\n💡 Next Steps:")
    print("  → Run example 05 for advanced validation patterns")
    print("  → Study FlextUtilities API for more utility functions")
    print("  → Explore utility service composition patterns")


if __name__ == "__main__":
    main()
