#!/usr/bin/env python3
"""FLEXT Core - Working Examples.

Comprehensive examples demonstrating all major FLEXT Core functionality.
This file shows practical usage patterns that work when flext_core is installed.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys

from pydantic import Field

from flext_core import FlextConfig, FlextModels, FlextResult

MINIMUM_AGE_REQUIREMENT = 21


def _print_header() -> None:
    """Print example header."""
    print("\n" + "=" * 60)
    print("🚀 FLEXT Core - Working Examples")
    print("=" * 60)


class SimpleUser(FlextModels.Config):
    """Simple user model for demonstrations."""

    user_name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., min_length=5, max_length=254)
    age: int = Field(..., ge=18, le=120)

    @classmethod
    def create(cls, name: str, email: str, age: int) -> FlextResult[SimpleUser]:
        """Create user with validation."""
        try:
            if "@" not in email:
                return FlextResult[SimpleUser].fail("Invalid email format")

            user = cls(user_name=name.strip(), email=email.lower(), age=age)
            return FlextResult[SimpleUser].ok(user)
        except Exception as e:
            return FlextResult[SimpleUser].fail(f"Failed to create user: {e}")


class SimpleConfig(FlextConfig):
    """Simple configuration example."""

    app_name: str = "flext-working-examples"
    version: str = "1.0.0"
    debug: bool = False
    max_users: int = Field(default=1000, ge=1)

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate configuration."""
        if self.max_users <= 0:
            return FlextResult[None].fail("max_users must be positive")
        return FlextResult[None].ok(None)


def demonstrate_basic_result_pattern() -> FlextResult[None]:
    """Demonstrate basic FlextResult usage."""
    print("\n📋 Demonstrating FlextResult Pattern")
    print("-" * 40)

    # Success case
    user_result = SimpleUser.create("John Doe", "john@example.com", 30)

    if user_result.success:
        user = user_result.value
        print(f"✅ User created: {user.user_name} ({user.email})")
    else:
        print(f"❌ User creation failed: {user_result.error}")
        return FlextResult[None].fail("User creation failed")

    # Failure case
    invalid_user_result = SimpleUser.create("", "invalid-email", 15)

    if not invalid_user_result.success:
        print(f"✅ Invalid user correctly rejected: {invalid_user_result.error}")
    else:
        print("❌ Invalid user was unexpectedly accepted")

    return FlextResult[None].ok(None)


def demonstrate_config_validation() -> FlextResult[None]:
    """Demonstrate configuration validation."""
    print("\n⚙️  Demonstrating Configuration Validation")
    print("-" * 40)

    # Valid configuration
    config = SimpleConfig()
    validation_result = config.validate_business_rules()

    if validation_result.success:
        print(f"✅ Valid config: {config.app_name} v{config.version}")
    else:
        print(f"❌ Config validation failed: {validation_result.error}")
        return FlextResult[None].fail("Config validation failed")

    # Invalid configuration
    try:
        invalid_config = SimpleConfig(max_users=-1)
        invalid_validation = invalid_config.validate_business_rules()

        if not invalid_validation.success:
            print(f"✅ Invalid config correctly rejected: {invalid_validation.error}")
        else:
            print("❌ Invalid config was unexpectedly accepted")
    except Exception as e:
        print(f"✅ Invalid config rejected during creation: {e}")

    return FlextResult[None].ok(None)


class UserService:
    """Service for managing users."""

    def __init__(self, config: SimpleConfig) -> None:
        """Initialize service with configuration."""
        self.config = config
        self.users: list[SimpleUser] = []

    def add_user(self, name: str, email: str, age: int) -> FlextResult[SimpleUser]:
        """Add a new user."""
        if len(self.users) >= self.config.max_users:
            return FlextResult[SimpleUser].fail("Maximum users reached")

        user_result = SimpleUser.create(name, email, age)
        if not user_result.success:
            return user_result

        user = user_result.value
        self.users.append(user)

        return FlextResult[SimpleUser].ok(user)

    def get_user_count(self) -> int:
        """Get current user count."""
        return len(self.users)

    def get_users_by_age_range(
        self,
        min_age: int,
        max_age: int,
    ) -> FlextResult[list[SimpleUser]]:
        """Get users within age range."""
        if min_age > max_age:
            return FlextResult[list[SimpleUser]].fail("Invalid age range")

        filtered_users = [user for user in self.users if min_age <= user.age <= max_age]

        return FlextResult[list[SimpleUser]].ok(filtered_users)


def demonstrate_service_patterns() -> FlextResult[None]:
    """Demonstrate service layer patterns."""
    print("\n🔧 Demonstrating Service Patterns")
    print("-" * 40)

    # Create service
    config = SimpleConfig(max_users=3)
    service = UserService(config)

    # Add some users
    test_users = [
        ("Alice Johnson", "alice@example.com", 25),
        ("Bob Smith", "bob@example.com", 35),
        ("Carol Davis", "carol@example.com", 45),
    ]

    for name, email, age in test_users:
        user_result = service.add_user(name, email, age)
        if user_result.success:
            user = user_result.value
            print(f"✅ Added user: {user.user_name} (age {user.age})")
        else:
            print(f"❌ Failed to add user {name}: {user_result.error}")

    print(f"📊 Total users: {service.get_user_count()}")

    # Test age filtering
    young_users_result = service.get_users_by_age_range(20, 30)
    if young_users_result.success:
        young_users = young_users_result.value
        print(f"👥 Young users (20-30): {len(young_users)}")
        for user in young_users:
            print(f"   - {user.user_name} (age {user.age})")

    # Test max users limit
    overflow_result = service.add_user("Dave Wilson", "dave@example.com", 28)
    if not overflow_result.success:
        print(f"✅ Max users limit enforced: {overflow_result.error}")

    return FlextResult[None].ok(None)


def demonstrate_error_handling() -> FlextResult[None]:
    """Demonstrate comprehensive error handling."""
    print("\n🚫 Demonstrating Error Handling")
    print("-" * 40)

    # Chain operations with error handling
    def process_user_pipeline(name: str, email: str, age: int) -> FlextResult[str]:
        """Process user through validation pipeline."""
        # Step 1: Create user
        user_result = SimpleUser.create(name, email, age)
        if not user_result.success:
            return FlextResult[str].fail(f"User creation failed: {user_result.error}")

        user = user_result.value

        # Step 2: Validate business rules (example)
        if user.age < MINIMUM_AGE_REQUIREMENT:
            return FlextResult[str].fail(
                f"User must be at least {MINIMUM_AGE_REQUIREMENT} years old",
            )

        # Step 3: Format result
        result = f"User {user.user_name} ({user.email}) processed successfully"
        return FlextResult[str].ok(result)

    # Test successful pipeline
    success_result = process_user_pipeline("John Doe", "john@example.com", 30)
    if success_result.success:
        print(f"✅ Pipeline success: {success_result.value}")
    else:
        print(f"❌ Pipeline failed: {success_result.error}")

    # Test pipeline failures
    failure_cases = [
        ("", "john@example.com", 30),  # Empty name
        ("John Doe", "invalid-email", 30),  # Invalid email
        ("John Doe", "john@example.com", 20),  # Too young
    ]

    for name, email, age in failure_cases:
        result = process_user_pipeline(name, email, age)
        if not result.success:
            print(f"✅ Expected failure caught: {result.error}")
        else:
            print("❌ Unexpected success for invalid input")

    return FlextResult[None].ok(None)


def main() -> int:
    """Main demonstration function."""
    _print_header()

    demonstrations = [
        ("Basic FlextResult Pattern", demonstrate_basic_result_pattern),
        ("Configuration Validation", demonstrate_config_validation),
        ("Service Layer Patterns", demonstrate_service_patterns),
        ("Error Handling", demonstrate_error_handling),
    ]

    for demo_name, demo_func in demonstrations:
        try:
            print(f"\n🎯 Running: {demo_name}")
            result = demo_func()

            if result.success:
                print(f"✅ {demo_name} completed successfully")
            else:
                print(f"❌ {demo_name} failed: {result.error}")
                return 1

        except Exception as e:
            print(f"❌ {demo_name} crashed: {e}")
            return 1

    print("\n🎉 All demonstrations completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
