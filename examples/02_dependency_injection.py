#!/usr/bin/env python3
"""02 - Dependency Injection using FlextCore facade.

Demonstrates simplified dependency injection patterns using FlextCore facade:
- FlextCore facade for unified access
- FlextContainer for dependency injection
- FlextDomainService inheritance
- Railway-oriented programming with FlextResult

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextContainer,
    FlextCore,
    FlextDomainService,
    FlextResult,
    FlextUtilities,
    FlextValidations,
)


class ProfessionalDependencyInjectionService(FlextDomainService[dict[str, object]]):
    """Simplified dependency injection service using FlextCore facade.

    This service demonstrates:
    - FlextCore facade usage for unified access
    - Container-based dependency injection
    - Type-safe service operations
    - Railway-oriented error handling
    """

    def __init__(self) -> None:
        """Initialize service with FlextCore facade."""
        super().__init__()
        self._core = FlextCore()
        self._container = FlextContainer.get_global()
        self._users: dict[str, dict[str, object]] = {}

    def setup_container(self) -> FlextResult[None]:
        """Setup container with required services."""
        # Register user storage
        storage_result = self._container.register("user_storage", self._users)
        if storage_result.is_failure:
            return FlextResult[None].fail("Failed to register user storage")

        # Register validator
        validator = FlextValidations.create_user_validator()
        validator_result = self._container.register("validator", validator)
        if validator_result.is_failure:
            return FlextResult[None].fail("Failed to register validator")

        return FlextResult[None].ok(None)

    def create_user(
        self, name: str, email: str, age: int
    ) -> FlextResult[dict[str, object]]:
        """Create a new user with validation using FlextCore utilities."""
        # Use FlextValidations for email validation
        email_validation = FlextValidations.validate_email(email)
        if email_validation.is_failure:
            return FlextResult[dict[str, object]].fail("Invalid email format")

        # Use basic validations for name and age (staying in domain)
        if not name or len(name.strip()) < 2:
            return FlextResult[dict[str, object]].fail(
                "Name must be at least 2 characters"
            )

        if age < 0 or age > 150:
            return FlextResult[dict[str, object]].fail("Age must be between 0 and 150")

        # Check for existing user
        storage_result = self._container.get("user_storage")
        if storage_result.is_failure:
            return FlextResult[dict[str, object]].fail("Storage not available")

        storage = storage_result.unwrap()
        if isinstance(storage, dict) and email in storage:
            return FlextResult[dict[str, object]].fail(f"User {email} already exists")

        # Use FlextUtilities for UUID generation
        user_id = f"user_{FlextUtilities.Generators.generate_id()[:8]}"
        user = {
            "id": user_id,
            "name": name,
            "email": email,
            "age": age,
            "status": "active",
        }

        # Store user
        if isinstance(storage, dict):
            storage[email] = user

        return FlextResult[dict[str, object]].ok(user)

    def find_user(self, email: str) -> FlextResult[dict[str, object] | None]:
        """Find user by email."""
        storage_result = self._container.get("user_storage")
        if storage_result.is_failure:
            return FlextResult[dict[str, object] | None].fail("Storage not available")

        storage = storage_result.unwrap()
        if isinstance(storage, dict):
            user = storage.get(email)
            return FlextResult[dict[str, object] | None].ok(user)

        return FlextResult[dict[str, object] | None].ok(None)

    def execute(self) -> FlextResult[dict[str, object]]:
        """Execute demo functionality - required by FlextDomainService."""
        # Setup container
        setup_result = self.setup_container()
        if setup_result.is_failure:
            return FlextResult[dict[str, object]].fail(
                f"Setup failed: {setup_result.error or 'Unknown error'}"
            )

        # Create demo user
        user_result = self.create_user("Demo User", "demo@example.com", 25)
        if user_result.is_failure:
            return FlextResult[dict[str, object]].fail(
                f"Demo user creation failed: {user_result.error or 'Unknown error'}"
            )

        return FlextResult[dict[str, object]].ok(
            {"message": "Demo completed", "user": user_result.value}
        )


def main() -> None:
    """Main demonstration function."""
    print("🚀 FlextCore Dependency Injection Demo")
    print("=" * 40)

    # Create service
    service = ProfessionalDependencyInjectionService()

    # Setup container
    setup_result = service.setup_container()
    if setup_result.is_failure:
        print(f"❌ Setup failed: {setup_result.error}")
        return

    print("✅ Container setup successful")

    # Test user creation
    print("\n1. Creating users:")
    users = [
        ("Alice Johnson", "alice@example.com", 28),
        ("Bob Smith", "bob@example.com", 32),
        ("Charlie Brown", "charlie@example.com", 25),
    ]

    for name, email, age in users:
        result = service.create_user(name, email, age)
        if result.is_success:
            user = result.value
            print(f"✅ Created: {user['name']} ({user['email']})")
        else:
            print(f"❌ Failed: {name} - {result.error}")

    # Test duplicate prevention
    print("\n2. Testing duplicate prevention:")
    duplicate_result = service.create_user("Alice Duplicate", "alice@example.com", 30)
    if duplicate_result.is_failure:
        print(f"✅ Duplicate prevented: {duplicate_result.error}")
    else:
        print("❌ Duplicate prevention failed")

    # Test user lookup
    print("\n3. Looking up users:")
    lookup_result = service.find_user("bob@example.com")
    if lookup_result.is_success and lookup_result.value:
        user = lookup_result.value
        print(f"✅ Found: {user['name']} ({user['email']})")
    else:
        print("❌ User not found")

    # Test validation
    print("\n4. Testing validation:")
    invalid_tests = [
        ("", "test@example.com", 25, "Empty name"),
        ("Valid Name", "invalid-email", 30, "Invalid email"),
        ("Valid Name", "valid@example.com", 200, "Invalid age"),
    ]

    for name, email, age, test_type in invalid_tests:
        result = service.create_user(name, email, age)
        if result.is_failure:
            print(f"✅ {test_type} validation: {result.error}")
        else:
            print(f"❌ {test_type} validation should have failed")

    print("\n✅ FlextCore Dependency Injection Demo Completed!")


if __name__ == "__main__":
    main()
