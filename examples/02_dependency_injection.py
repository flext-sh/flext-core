#!/usr/bin/env python3
"""02 - Dependency Injection using FlextCore.

Simplified demonstration of dependency injection patterns using FlextCore's built-in
container and service management capabilities with minimum boilerplate

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations
from typing import Protocol, cast

from flext_core import (
    FlextConstants,
    FlextContainer,
    FlextModels,
    FlextResult,
    FlextUtilities,
)

# =============================================================================
# DOMAIN MODELS - Using FlextCore's built-in patterns
# =============================================================================


class UserData(FlextModels.Value):
    """User data as immutable value object using FlextModels."""

    name: str
    email: str
    age: int

    def validate_business_rules(self) -> FlextResult[None]:
        """Basic validation for user data."""
        # Basic validation is delegated to User entity
        return FlextResult[None].ok(None)


class User(FlextModels.Entity):
    """User entity with business logic using FlextCore patterns."""

    name: str
    email: str
    age: int
    status: str = FlextConstants.Status.ACTIVE

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate user business rules."""
        if len(self.name) < 2:
            return FlextResult[None].fail("Name must be at least 2 characters")
        if "@" not in self.email or "." not in self.email.split("@")[-1]:
            return FlextResult[None].fail("Invalid email format")
        if not 0 <= self.age <= 150:
            return FlextResult[None].fail("Age must be between 0 and 150")
        return FlextResult[None].ok(None)


# =============================================================================
# SERVICE PROTOCOLS - Dependency inversion
# =============================================================================


class UserServiceProtocol(Protocol):
    """Protocol for user service operations."""

    def create_user(self, data: UserData) -> FlextResult[User]: ...
    def find_user_by_email(self, email: str) -> FlextResult[User | None]: ...


class NotificationServiceProtocol(Protocol):
    """Protocol for notification operations."""

    def send_welcome(self, user: User) -> FlextResult[None]: ...


# =============================================================================
# SERVICE IMPLEMENTATIONS - Minimal and focused
# =============================================================================


class UserService:
    """User service implementation using FlextCore patterns."""

    def __init__(self) -> None:
        """Initialize with in-memory storage."""
        self._users: dict[str, User] = {}

    def create_user(self, data: UserData) -> FlextResult[User]:
        """Create a new user with validation."""
        # Check for duplicates
        if data.email in self._users:
            return FlextResult[User].fail(f"User {data.email} already exists")

        # Create user with generated ID
        user = User(
            id=FlextUtilities.Generators.generate_uuid(),
            name=data.name,
            email=data.email,
            age=data.age,
        )

        # Validate business rules
        validation_result = user.validate_business_rules()
        if validation_result.is_failure:
            return FlextResult[User].fail(
                validation_result.error or "Validation failed",
            )

        # Store user
        self._users[data.email] = user
        return FlextResult[User].ok(user)

    def find_user_by_email(self, email: str) -> FlextResult[User | None]:
        """Find user by email."""
        if "@" not in email:
            return FlextResult[User | None].fail("Invalid email format")

        user = self._users.get(email)
        return FlextResult[User | None].ok(user)


class NotificationService:
    """Notification service implementation."""

    def send_welcome(self, user: User) -> FlextResult[None]:
        """Send welcome notification."""
        if "@" not in user.email:
            return FlextResult[None].fail("Invalid email format")

        # In production, this would send actual notification
        print(f"Welcome email sent to {user.name} at {user.email}")
        return FlextResult[None].ok(None)


# =============================================================================
# APPLICATION SERVICE - Orchestration with dependency injection
# =============================================================================


class UserRegistrationService:
    """User registration orchestration using dependency injection."""

    def __init__(
        self,
        user_service: UserServiceProtocol,
        notification_service: NotificationServiceProtocol,
    ) -> None:
        """Initialize with injected dependencies."""
        self._user_service = user_service
        self._notification_service = notification_service

    def register_user(self, name: str, email: str, age: int) -> FlextResult[User]:
        """Register a new user with orchestrated workflow."""
        # Create user data
        user_data = UserData(name=name, email=email, age=age)

        # Chain operations using FlextResult
        return self._user_service.create_user(user_data).flat_map(
            self._send_welcome_notification,
        )

    def _send_welcome_notification(self, user: User) -> FlextResult[User]:
        """Send welcome notification and return user."""
        notification_result = self._notification_service.send_welcome(user)

        # Don't fail registration if notification fails
        if notification_result.is_failure:
            print(f"Warning: Welcome notification failed for {user.email}")

        return FlextResult[User].ok(user)


# =============================================================================
# DEPENDENCY INJECTION SETUP
# =============================================================================


def setup_container() -> FlextResult[FlextContainer]:
    """Setup dependency injection container with services."""
    # Get global container
    container = FlextContainer.get_global()

    # Register services
    container.register("user_service", UserService())
    container.register("notification_service", NotificationService())

    # Register factory for registration service
    def create_registration_service() -> UserRegistrationService:
        user_service_result = container.get("user_service")
        if not user_service_result.success:
            msg = "User service not found"
            raise RuntimeError(msg)

        notification_service_result = container.get("notification_service")
        if not notification_service_result.success:
            msg = "Notification service not found"
            raise RuntimeError(msg)

        user_service = cast("UserServiceProtocol", user_service_result.unwrap())
        notification_service = cast(
            "NotificationServiceProtocol", notification_service_result.unwrap(),
        )
        return UserRegistrationService(user_service, notification_service)

    container.register_factory("registration_service", create_registration_service)

    return FlextResult[FlextContainer].ok(container)


# =============================================================================
# DEMONSTRATION
# =============================================================================


def main() -> None:
    """Demonstrate dependency injection with FlextCore."""
    print("=== FlextCore Dependency Injection Demo ===\n")

    # Setup container
    container_result = setup_container()
    if container_result.is_failure:
        print(f"❌ Container setup failed: {container_result.error}")
        return

    container = container_result.unwrap()
    print("✅ Container configured successfully")

    # Get registration service
    registration_result = container.get("registration_service")
    if registration_result.is_failure:
        print(f"❌ Failed to get registration service: {registration_result.error}")
        return

    registration_service = cast("UserRegistrationService", registration_result.unwrap())
    print("✅ Registration service retrieved\n")

    # Test user registrations
    test_users = [
        ("Alice Johnson", "alice@example.com", 25),
        ("Bob Smith", "bob@company.com", 30),
        ("Charlie Brown", "charlie@test.org", 28),
    ]

    print("=== Registering Users ===")
    for name, email, age in test_users:
        result = registration_service.register_user(name, email, age)

        if result.success:
            user = result.unwrap()
            print(f"✅ Registered: {user.name} ({user.email})")
        else:
            print(f"❌ Failed: {name} - {result.error}")

    # Test duplicate prevention
    print("\n=== Testing Duplicate Prevention ===")
    duplicate_result = registration_service.register_user(
        "Alice Duplicate", "alice@example.com", 26,
    )
    if duplicate_result.is_failure:
        print(f"✅ Duplicate prevented: {duplicate_result.error}")

    # Test user lookup
    print("\n=== Testing User Lookup ===")
    user_service = cast("UserService", container.get("user_service").unwrap())
    lookup_result = user_service.find_user_by_email("bob@company.com")

    if lookup_result.success:
        found_user = lookup_result.unwrap()
        if found_user:
            print(f"✅ Found user: {found_user.name} ({found_user.email})")
    else:
        print("❌ User not found")

    # Test validation errors
    print("\n=== Testing Validation ===")
    invalid_tests = [
        ("", "empty@test.com", 25, "Empty name"),
        ("Valid Name", "invalid-email", 30, "Invalid email"),
        ("Valid Name", "valid@test.com", -5, "Invalid age"),
    ]

    for name, email, age, error_type in invalid_tests:
        result = registration_service.register_user(name, email, age)
        if result.is_failure:
            print(f"✅ {error_type} validation: {result.error}")
        else:
            print(f"❌ {error_type} validation should have failed")

    print("\n✅ Dependency injection demo completed successfully!")


if __name__ == "__main__":
    main()
