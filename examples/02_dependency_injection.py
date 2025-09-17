#!/usr/bin/env python3
"""02 - Dependency Injection using direct FlextContainer access.

Demonstrates dependency injection patterns using direct component access:
- Direct FlextContainer access for dependency injection
- FlextDomainService inheritance
- Railway-oriented programming with FlextResult

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextContainer,
    FlextDomainService,
    FlextModels,
    FlextResult,
    FlextUtilities,
)


class User(FlextModels.Value):
    """User domain model with validation."""

    id: str
    name: str
    email: str
    age: int
    status: str


class ProfessionalDependencyInjectionService(FlextDomainService[User]):
    """Dependency injection service using direct FlextContainer access.

    This service demonstrates:
    - Direct FlextContainer access for dependency injection
    - Type-safe service operations
    - Railway-oriented error handling
    """

    def __init__(self) -> None:
        """Initialize service with direct container access."""
        super().__init__()
        self._container = FlextContainer.get_global()
        self._users: dict[str, User] = {}

    def setup_container(self) -> FlextResult[None]:
        """Setup container with required services."""
        # Register user storage
        storage_result = self._container.register("user_storage", self._users)
        if storage_result.is_failure:
            return FlextResult[None].fail("Failed to register user storage")

        # Register validation function
        def validate_user_data(data: dict[str, object]) -> FlextResult[None]:
            # Validate required fields exist
            if not data:
                return FlextResult[None].fail("User data cannot be empty")
            name = data.get("name", "")
            email = data.get("email", "")
            age = data.get("age")
            if not isinstance(name, str) or len(name) < 2:
                return FlextResult[None].fail("Name must be at least 2 characters")
            if not isinstance(email, str) or "@" not in email:
                return FlextResult[None].fail("Invalid email format")
            if not isinstance(age, int) or age < 0 or age > 150:
                return FlextResult[None].fail("Age must be between 0 and 150")
            return FlextResult[None].ok(None)

        validator_result = self._container.register("validator", validate_user_data)
        if validator_result.is_failure:
            return FlextResult[None].fail("Failed to register validator")

        return FlextResult[None].ok(None)

    def get_user_storage(self) -> FlextResult[dict[str, User]]:
        """Get user storage from container - eliminates duplicate access pattern."""
        storage_result = self._container.get("user_storage")
        if storage_result.is_failure:
            return FlextResult[dict[str, User]].fail("Storage not available")

        storage = storage_result.unwrap()
        if isinstance(storage, dict):
            return FlextResult[dict[str, User]].ok(storage)

        return FlextResult[dict[str, User]].fail("Invalid storage type")

    def create_user(self, name: str, email: str, age: int) -> FlextResult[User]:
        """Create a new user with validation using container-based validator."""
        # Consolidated validation using container validator
        user_data = {"name": name, "email": email, "age": age}
        validator_result = self._container.get("validator")
        if validator_result.is_failure:
            return FlextResult[User].fail("Validator not available")

        validator = validator_result.unwrap()
        if callable(validator):
            validation_result = validator(user_data)
            # Check if the result has the expected FlextResult interface
            if hasattr(validation_result, "is_failure") and getattr(
                validation_result, "is_failure", False
            ):
                error_msg = (
                    getattr(validation_result, "error", None) or "Validation failed"
                )
                return FlextResult[User].fail(error_msg)
        else:
            return FlextResult[User].fail("Invalid validator type")

        # Check for existing user using helper method
        storage_result = self.get_user_storage()
        if storage_result.is_failure:
            return FlextResult[User].fail(storage_result.error or "Storage error")

        storage = storage_result.unwrap()
        if email in storage:
            return FlextResult[User].fail(f"User {email} already exists")

        # Create User domain object
        user_id = f"user_{FlextUtilities.Generators.generate_id()[:8]}"
        user = User(
            id=user_id,
            name=name,
            email=email,
            age=age,
            status="active",
        )

        # Store user
        storage[email] = user
        return FlextResult[User].ok(user)

    def find_user(self, email: str) -> FlextResult[User | None]:
        """Find user by email using helper method."""
        storage_result = self.get_user_storage()
        if storage_result.is_failure:
            return FlextResult[User | None].fail(
                storage_result.error or "Storage error"
            )

        storage = storage_result.unwrap()
        user = storage.get(email)
        return FlextResult[User | None].ok(user)

    def execute(self) -> FlextResult[User]:
        """Execute demo functionality - required by FlextDomainService."""
        # Setup container
        setup_result = self.setup_container()
        if setup_result.is_failure:
            return FlextResult[User].fail(
                f"Setup failed: {setup_result.error or 'Unknown error'}"
            )

        # Create demo user
        user_result = self.create_user("Demo User", "demo@example.com", 25)
        if user_result.is_failure:
            return FlextResult[User].fail(
                f"Demo user creation failed: {user_result.error or 'Unknown error'}"
            )

        return FlextResult[User].ok(user_result.value)


def main() -> None:
    """Compact demonstration with data-driven testing."""
    print("🚀 Advanced Dependency Injection Demo")
    print("=" * 45)

    service = ProfessionalDependencyInjectionService()

    # Setup container
    setup_result = service.setup_container()
    if setup_result.is_failure:
        print(f"❌ Setup failed: {setup_result.error}")
        return

    print("✅ Container setup successful")

    # Data-driven test scenarios
    test_scenarios = [
        # Valid users
        {
            "data": ("Alice Johnson", "alice@example.com", 28),
            "expect": "success",
            "desc": "Valid user",
        },
        {
            "data": ("Bob Smith", "bob@example.com", 32),
            "expect": "success",
            "desc": "Valid user",
        },
        {
            "data": ("Charlie Brown", "charlie@example.com", 25),
            "expect": "success",
            "desc": "Valid user",
        },
        # Duplicate test
        {
            "data": ("Alice Duplicate", "alice@example.com", 30),
            "expect": "failure",
            "desc": "Duplicate email",
        },
        # Validation tests
        {
            "data": ("", "test@example.com", 25),
            "expect": "failure",
            "desc": "Empty name",
        },
        {
            "data": ("Valid Name", "invalid-email", 30),
            "expect": "failure",
            "desc": "Invalid email",
        },
        {
            "data": ("Valid Name", "valid@example.com", 200),
            "expect": "failure",
            "desc": "Invalid age",
        },
    ]

    print("\n1. Testing user operations:")
    print("-" * 30)

    for i, scenario in enumerate(test_scenarios, 1):
        data_tuple = scenario["data"]
        expected = scenario["expect"]
        description = scenario["desc"]

        # Type-safe unpacking
        if isinstance(data_tuple, tuple) and len(data_tuple) == 3:
            name, email, age = data_tuple
            # Ensure types are correct
            if (
                isinstance(name, str)
                and isinstance(email, str)
                and isinstance(age, int)
            ):
                result = service.create_user(name, email, age)

                if result.is_success and expected == "success":
                    user = result.value
                    print(
                        f"✅ {i}. Created: {user.name} ({user.email}) - {description}"
                    )
                elif result.is_failure and expected == "failure":
                    print(f"✅ {i}. Expected failure: {description} - {result.error}")
                else:
                    status = "success" if result.is_success else "failure"
                    print(f"❌ {i}. Unexpected {status}: {description}")

    # Test lookup functionality
    print("\n2. Testing user lookup:")
    lookup_result = service.find_user("bob@example.com")
    if lookup_result.is_success and lookup_result.value:
        user = lookup_result.value
        print(f"✅ Found: {user.name} ({user.email})")
    else:
        print("❌ User lookup failed")

    print("\n✅ Advanced DI Demo Completed!")


if __name__ == "__main__":
    main()
