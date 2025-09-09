#!/usr/bin/env python3
"""02 - Dependency Injection using FlextCore DIRECTLY.

Demonstrates DIRECT usage of FlextCore components eliminating ALL duplication:
- FlextContainer.get_global() for dependency injection
- FlextModels.Entity for domain models
- FlextValidations for validation
- FlextUtilities for utilities

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations
from typing import cast

from flext_core import (
    FlextContainer,
    FlextDomainService,
    FlextModels,
    FlextResult,
    FlextTypes,
    FlextUtilities,
    FlextValidations,
)


# =============================================================================
# UNIFIED DEPENDENCY INJECTION SERVICE - Single class using FlextCore DIRECTLY
# =============================================================================


class ProfessionalDependencyInjectionService(FlextDomainService[object]):
    """UNIFIED service demonstrating dependency injection using FlextCore DIRECTLY.

    Eliminates ALL duplication by using FlextCore components directly:
    - FlextContainer for dependency injection
    - FlextModels.Entity for domain models
    - FlextValidations for validation
    - FlextUtilities for utilities
    """

    def __init__(self) -> None:
        """Initialize with FlextCore components."""
        super().__init__()
        self._container = FlextContainer.get_global()
        self._validator = FlextValidations.create_user_validator()

    class _ContainerHelper:
        """Nested helper for container operations."""

        @staticmethod
        def register_service(
            container: FlextContainer, key: str, service: object
        ) -> FlextResult[None]:
            """Register service in container."""
            result = container.register(key, service)
            if result.is_failure:
                return FlextResult[None].fail(f"Failed to register {key}")
            return FlextResult[None].ok(None)

        @staticmethod
        def get_service(container: FlextContainer, key: str) -> FlextResult[object]:
            """Get service from container."""
            return container.get(key)

    class UserEntity(FlextModels.Entity):
        """User entity using FlextModels.Entity DIRECTLY."""

        name: str
        email: str
        age: int
        status: str = "active"

        def validate_business_rules(self) -> FlextResult[None]:
            """Validate using FlextValidations DIRECTLY."""
            user_validator = FlextValidations.create_user_validator()
            user_data: FlextTypes.Core.Dict = {"name": self.name, "email": self.email}
            result = user_validator.validate_business_rules(user_data)
            if result.is_failure:
                return FlextResult[None].fail(result.error or "Validation failed")
            return FlextResult[None].ok(None)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate using FlextDomainService pattern."""
        return FlextResult[None].ok(None)

    def execute(self) -> FlextResult[object]:
        """Execute default operation - required by FlextDomainService."""
        result = self.create_user_with_container("Demo User", "demo@example.com", 25)
        if result.is_success:
            return FlextResult[object].ok(result.value)
        return FlextResult[object].fail(result.error or "Execution failed")

    def setup_container(self) -> FlextResult[None]:
        """Setup container with services using FlextCore DIRECTLY."""
        # Register storage using FlextCore
        user_storage: dict[str, ProfessionalDependencyInjectionService.UserEntity] = {}
        storage_result = self._ContainerHelper.register_service(
            self._container, "user_storage", user_storage
        )
        if storage_result.is_failure:
            return storage_result

        # Register validator using FlextValidations DIRECTLY
        validator_result = self._ContainerHelper.register_service(
            self._container, "user_validator", self._validator
        )
        if validator_result.is_failure:
            return validator_result

        return FlextResult[None].ok(None)

    def create_user_with_container(
        self, name: str, email: str, age: int
    ) -> FlextResult[UserEntity]:
        """Create user using FlextContainer services DIRECTLY."""
        # Get services from container using FlextCore
        storage_result = self._ContainerHelper.get_service(
            self._container, "user_storage"
        )
        validator_result = self._ContainerHelper.get_service(
            self._container, "user_validator"
        )

        if storage_result.is_failure or validator_result.is_failure:
            return FlextResult[ProfessionalDependencyInjectionService.UserEntity].fail(
                "Required services not found in container"
            )

        storage = cast(
            "dict[str, ProfessionalDependencyInjectionService.UserEntity]",
            storage_result.unwrap(),
        )
        validator = validator_result.unwrap()

        # Check for duplicates
        if email in storage:
            return FlextResult[ProfessionalDependencyInjectionService.UserEntity].fail(
                f"User {email} already exists"
            )

        # Validate using existing validator
        user_data: FlextTypes.Core.Dict = {"name": name, "email": email}
        if hasattr(validator, "validate_business_rules"):
            validation_result = validator.validate_business_rules(user_data)  # type: ignore[attr-defined]
            if validation_result.is_failure:
                return FlextResult[
                    ProfessionalDependencyInjectionService.UserEntity
                ].fail("Validation failed")
        else:
            return FlextResult[ProfessionalDependencyInjectionService.UserEntity].fail(
                "Invalid validator"
            )

        # Create user using FlextUtilities DIRECTLY
        user = self.UserEntity(
            id=FlextUtilities.Generators.generate_entity_id(),
            name=name,
            email=email,
            age=age,
        )

        # Store in container-managed storage
        storage[email] = user
        return FlextResult[ProfessionalDependencyInjectionService.UserEntity].ok(user)

    def find_user_with_container(self, email: str) -> FlextResult[UserEntity | None]:
        """Find user using FlextContainer storage DIRECTLY."""
        storage_result = self._ContainerHelper.get_service(
            self._container, "user_storage"
        )
        if storage_result.is_failure:
            return FlextResult[
                ProfessionalDependencyInjectionService.UserEntity | None
            ].fail("Storage service not found")

        # Use existing email validation from FlextValidations
        email_result = FlextValidations.Rules.StringRules.validate_email(email)
        if email_result.is_failure:
            return FlextResult[
                ProfessionalDependencyInjectionService.UserEntity | None
            ].fail("Invalid email format")

        storage = cast(
            "dict[str, ProfessionalDependencyInjectionService.UserEntity]",
            storage_result.unwrap(),
        )
        user = storage.get(email)
        return FlextResult[ProfessionalDependencyInjectionService.UserEntity | None].ok(
            user
        )

    def send_notification(self, user: UserEntity) -> FlextResult[None]:
        """Send notification using FlextValidations DIRECTLY."""
        # Use existing email validation from FlextCore
        email_result = FlextValidations.Rules.StringRules.validate_email(user.email)
        if email_result.is_failure:
            return FlextResult[None].fail("Invalid email format")

        # Simple notification simulation
        print(f"Welcome email sent to {user.name} at {user.email}")
        return FlextResult[None].ok(None)

    def send_welcome(self, user: UserEntity) -> FlextResult[None]:
        """Simple alias for send_notification method."""
        return self.send_notification(user)


def main() -> None:
    """Main demonstration using FlextCore DIRECTLY - ZERO duplication."""
    service = ProfessionalDependencyInjectionService()

    print("🚀 FlextCore Dependency Injection Showcase - ZERO Duplication")
    print("=" * 50)
    print("Features: FlextContainer.get_global() • FlextValidations • FlextUtilities")
    print()

    # Setup container using FlextCore DIRECTLY
    setup_result = service.setup_container()
    if setup_result.is_failure:
        print(f"❌ Container setup failed: {setup_result.error}")
        return

    print("✅ FlextContainer.get_global() configured successfully")

    # Test user creation using container services
    print("\n1. User Creation using FlextContainer:")
    test_users = [
        ("Alice Johnson", "alice@example.com", 25),
        ("Bob Smith", "bob@company.com", 30),
        ("Charlie Brown", "charlie@test.org", 28),
    ]

    for name, email, age in test_users:
        result = service.create_user_with_container(name, email, age)
        if result.is_success:
            user = result.value
            print(f"✅ Created: {user.name} ({user.email}) [ID: {user.id}]")

            # Send notification using existing validation
            notification_result = service.send_notification(user)
            if notification_result.is_failure:
                print(f"⚠️  Notification failed: {notification_result.error}")
        else:
            print(f"❌ Failed: {name} - {result.error}")

    # Test duplicate prevention
    print("\n2. Duplicate Prevention using FlextContainer:")
    duplicate_result = service.create_user_with_container(
        "Alice Duplicate", "alice@example.com", 26
    )
    if duplicate_result.is_failure:
        print(f"✅ Duplicate prevented: {duplicate_result.error}")
    else:
        print("❌ Duplicate prevention failed")

    # Test user lookup using container
    print("\n3. User Lookup using FlextContainer:")
    lookup_result = service.find_user_with_container("bob@company.com")
    if lookup_result.is_success:
        found_user = lookup_result.value
        if found_user:
            print(f"✅ Found user: {found_user.name} ({found_user.email})")
        else:
            print("❌ User not found in container storage")
    else:
        print(f"❌ Lookup failed: {lookup_result.error}")

    # Test validation using existing FlextValidations
    print("\n4. Validation using FlextValidations:")
    invalid_tests = [
        ("", "empty@test.com", 25, "Empty name"),
        ("Valid Name", "invalid-email", 30, "Invalid email"),
    ]

    for name, email, age, error_type in invalid_tests:
        result = service.create_user_with_container(name, email, age)
        if result.is_failure:
            print(f"✅ {error_type} validation: {result.error}")
        else:
            print(f"❌ {error_type} validation should have failed")

    print("\n✅ FlextCore Dependency Injection Demo Completed Successfully!")
    print(
        "💪 ZERO code duplication - using only existing FlextContainer & FlextValidations!"
    )


# Simple aliases for test compatibility (CLAUDE.md compliant)
DependencyInjectionService = ProfessionalDependencyInjectionService
DependencyInjectionShowcase = ProfessionalDependencyInjectionService
UserService = ProfessionalDependencyInjectionService
NotificationService = ProfessionalDependencyInjectionService
UserRegistrationService = ProfessionalDependencyInjectionService

# Aliases for nested classes
UserData = ProfessionalDependencyInjectionService.UserEntity
User = ProfessionalDependencyInjectionService.UserEntity


def setup_container() -> FlextResult[ProfessionalDependencyInjectionService]:
    """Simple setup function for test compatibility."""
    service = ProfessionalDependencyInjectionService()
    init_result = service.setup_container()
    if init_result.is_failure:
        return FlextResult[ProfessionalDependencyInjectionService].fail("Setup failed")
    return FlextResult[ProfessionalDependencyInjectionService].ok(service)


if __name__ == "__main__":
    main()
