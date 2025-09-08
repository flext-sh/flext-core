#!/usr/bin/env python3
"""FlextMixins Composition Patterns - Working Example.

This example demonstrates FlextMixins composition patterns for behavioral functionality.
Shows how to use FlextMixins instead of multiple inheritance for clean separation
of concerns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time

from flext_core import FlextMixins, FlextResult
from flext_core import FlextTypes

# Constants
MIN_USERNAME_LENGTH = 3
MINIMUM_USER_AGE = 13


class User:
    """Simple user class for demonstration."""

    def __init__(self, username: str, email: str, age: int) -> None:
        self.username = username
        self.email = email
        self.age = age

    def __str__(self) -> str:
        """String representation of user."""
        return f"User({self.username}, {self.email}, {self.age})"


class EnhancedUserService:
    """User service with FlextMixins functionality."""

    def __init__(self) -> None:
        """Initialize service with FlextMixins support."""
        # Initialize mixins functionality
        self._mixins = FlextMixins()

        # Get logger for this instance
        self._logger = FlextMixins.flext_logger(self)

    def create_user(self, username: str, email: str, age: int) -> FlextResult[User]:
        """Create a user with validation and logging."""
        # Validate input using mixins utility
        if not username or len(username) < MIN_USERNAME_LENGTH:
            error = f"Username must be at least {MIN_USERNAME_LENGTH} characters"
            self._logger.error("User creation failed", error=error)
            return FlextResult[User].fail(error)

        if not email or "@" not in email:
            error = "Invalid email address"
            self._logger.error("User creation failed", error=error)
            return FlextResult[User].fail(error)

        if age < MINIMUM_USER_AGE:
            error = f"User must be at least {MINIMUM_USER_AGE} years old"
            self._logger.error("User creation failed", error=error)
            return FlextResult[User].fail(error)

        # Create user
        try:
            user = User(username, email, age)
            self._logger.info("User created successfully", username=username)
            return FlextResult[User].ok(user)
        except Exception as e:
            error = f"Failed to create user: {e}"
            self._logger.exception("User creation failed", error=error)
            return FlextResult[User].fail(error)

    def get_user_summary(self, user: User) -> FlextTypes.Core.Dict:
        """Get user summary with mixins functionality."""
        # Use mixins to add metadata
        summary = {
            "username": user.username,
            "email": user.email,
            "age": user.age,
            "created_at": time.time(),
        }

        self._logger.info("User summary generated", username=user.username)
        return summary


class CacheableService:
    """Service with caching capabilities using FlextMixins."""

    def __init__(self) -> None:
        """Initialize cacheable service."""
        self._cache: FlextTypes.Core.Dict = {}
        self._logger = FlextMixins.flext_logger(self)

    def get_cached_data(self, key: str) -> FlextResult[object]:
        """Get data from cache."""
        if key in self._cache:
            self._logger.info("Cache hit", key=key)
            return FlextResult[object].ok(self._cache[key])

        self._logger.info("Cache miss", key=key)
        return FlextResult[object].fail(f"Key '{key}' not found in cache")

    def set_cached_data(self, key: str, value: object) -> FlextResult[None]:
        """Set data in cache."""
        try:
            self._cache[key] = value
            self._logger.info("Data cached", key=key)
            return FlextResult[None].ok(None)
        except Exception as e:
            error = f"Failed to cache data: {e}"
            self._logger.exception("Cache set failed", error=error)
            return FlextResult[None].fail(error)

    def clear_cache(self) -> FlextResult[int]:
        """Clear cache and return count of items cleared."""
        count = len(self._cache)
        self._cache.clear()
        self._logger.info("Cache cleared", items_cleared=count)
        return FlextResult[int].ok(count)


class ValidationService:
    """Service for validation operations using FlextMixins."""

    def __init__(self) -> None:
        """Initialize validation service."""
        self._logger = FlextMixins.flext_logger(self)

    def validate_email(self, email: str) -> FlextResult[bool]:
        """Validate email format."""
        if not email:
            return FlextResult[bool].fail("Email cannot be empty")

        if "@" not in email or "." not in email:
            return FlextResult[bool].fail("Invalid email format")

        # Basic validation passed
        self._logger.info("Email validation passed", email=email)
        return FlextResult[bool].ok(data=True)

    def validate_age(self, age: int, min_age: int = 13) -> FlextResult[bool]:
        """Validate age requirements."""
        if age < min_age:
            error = f"Age {age} is below minimum {min_age}"
            self._logger.warning("Age validation failed", age=age, min_age=min_age)
            return FlextResult[bool].fail(error)

        self._logger.info("Age validation passed", age=age)
        return FlextResult[bool].ok(data=True)


class ComposedUserManager:
    """User manager that composes multiple services using FlextMixins patterns."""

    def __init__(self) -> None:
        """Initialize composed user manager."""
        self._user_service = EnhancedUserService()
        self._cache_service = CacheableService()
        self._validation_service = ValidationService()
        self._logger = FlextMixins.flext_logger(self)

    def create_and_cache_user(
        self,
        username: str,
        email: str,
        age: int,
    ) -> FlextResult[User]:
        """Create user with validation and caching."""
        # Step 1: Validate email
        email_validation = self._validation_service.validate_email(email)
        if email_validation.is_failure:
            return FlextResult[User].fail(
                f"Email validation failed: {email_validation.error}",
            )

        # Step 2: Validate age
        age_validation = self._validation_service.validate_age(age)
        if age_validation.is_failure:
            return FlextResult[User].fail(
                f"Age validation failed: {age_validation.error}",
            )

        # Step 3: Check cache
        cache_key = f"user_{username}"
        cached_result = self._cache_service.get_cached_data(cache_key)
        if cached_result.success:
            self._logger.info("User found in cache", username=username)
            # Cast from cache (assume it's a User object)
            cached_user = cached_result.value
            if isinstance(cached_user, User):
                return FlextResult[User].ok(cached_user)
            # If not a User, continue to create new one
            self._logger.warning("Invalid cached data type", type=type(cached_user))

        # Step 4: Create user
        create_result = self._user_service.create_user(username, email, age)
        if create_result.is_failure:
            return create_result

        # Step 5: Cache user
        user = create_result.value
        cache_result = self._cache_service.set_cached_data(cache_key, user)
        if cache_result.is_failure:
            self._logger.warning("Failed to cache user", error=cache_result.error)

        self._logger.info("User created and cached", username=username)
        return FlextResult[User].ok(user)

    def get_user_info(self, username: str) -> FlextResult[FlextTypes.Core.Dict]:
        """Get comprehensive user information."""
        cache_key = f"user_{username}"
        cached_result = self._cache_service.get_cached_data(cache_key)

        if cached_result.is_failure:
            return FlextResult[FlextTypes.Core.Dict].fail("User not found")

        user = cached_result.value
        if not isinstance(user, User):
            return FlextResult[FlextTypes.Core.Dict].fail("Invalid user data in cache")

        # Get user summary
        summary = self._user_service.get_user_summary(user)

        self._logger.info("User info retrieved", username=username)
        return FlextResult[FlextTypes.Core.Dict].ok(summary)


def demonstrate_mixins_composition() -> None:
    """Demonstrate FlextMixins composition patterns."""
    print("=== FlextMixins Composition Demo ===")

    # Create composed user manager
    user_manager = ComposedUserManager()

    # Create users with validation and caching
    users_to_create = [
        ("alice", "alice@example.com", 25),
        ("bob", "bob@example.com", 30),
        ("charlie", "charlie@example.com", 22),
    ]

    created_users = []
    for username, email, age in users_to_create:
        result = user_manager.create_and_cache_user(username, email, age)
        if result.success:
            created_users.append(result.value)
            print(f"✅ Created user: {result.value}")
        else:
            print(f"❌ Failed to create {username}: {result.error}")

    print(f"\n📊 Successfully created {len(created_users)} users")


def demonstrate_validation_patterns() -> None:
    """Demonstrate validation with FlextMixins."""
    print("\n=== Validation Patterns Demo ===")

    validation_service = ValidationService()

    # Test email validation
    emails = ["valid@example.com", "invalid-email", "", "another@test.org"]

    for email in emails:
        result = validation_service.validate_email(email)
        status = "✅" if result.success else "❌"
        message = "Valid" if result.success else result.error
        print(f"{status} Email '{email}': {message}")

    # Test age validation
    ages = [25, 16, 12, 18, 65]

    for age in ages:
        result = validation_service.validate_age(age)
        status = "✅" if result.success else "❌"
        message = "Valid" if result.success else result.error
        print(f"{status} Age {age}: {message}")


def demonstrate_caching_patterns() -> None:
    """Demonstrate caching with FlextMixins."""
    print("\n=== Caching Patterns Demo ===")

    cache_service = CacheableService()

    # Set some data
    test_data = [
        ("key1", {"name": "Test Data 1", "value": 100}),
        ("key2", {"name": "Test Data 2", "value": 200}),
        ("key3", "Simple string value"),
    ]

    for key, value in test_data:
        set_result = cache_service.set_cached_data(key, value)
        if set_result.success:
            print(f"✅ Cached: {key}")

    # Retrieve data
    for key, _ in test_data:
        get_result = cache_service.get_cached_data(key)
        if get_result.success:
            print(f"✅ Retrieved {key}: {get_result.value}")

    # Clear cache
    clear_result = cache_service.clear_cache()
    if clear_result.success:
        print(f"🧹 Cleared {clear_result.value} items from cache")


def main() -> None:
    """Run FlextMixins composition patterns demonstration."""
    print("FlextCore Mixins Composition Patterns")
    print("=" * 40)

    demonstrate_mixins_composition()
    demonstrate_validation_patterns()
    demonstrate_caching_patterns()

    print("\n" + "=" * 40)
    print("✅ All FlextMixins patterns demonstrated successfully!")


if __name__ == "__main__":
    main()
