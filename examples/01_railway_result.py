#!/usr/bin/env python3
"""01 - Advanced Railway Pattern using FLEXT Core Domain Services.

Demonstrates advanced flext-core patterns with significant code reduction:
- FlextDomainService for structured service architecture
- FlextModels.Value for type-safe domain modeling
- Direct validation with FlextResult for clean validation without try/catch
- Advanced access patterns reducing nested utility calls

This showcase demonstrates production-ready patterns that reduce bloat
while maintaining full type safety and error handling capabilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import Field

from flext_core import (
    FlextDomainService,
    FlextModels,
    FlextResult,
    FlextUtilities,
)


class UserData(FlextModels.Value):
    """Immutable user data model with validation."""

    name: str = Field(min_length=2)
    email: str = Field(...)
    age: int = Field(ge=0, le=150)


class ProcessedUser(FlextModels.Value):
    """Complete processed user with metadata."""

    id: str
    name: str
    email: str
    age: int
    status: str
    created_at: str
    correlation_id: str


class AdvancedRailwayService(FlextDomainService[ProcessedUser]):
    """Domain service demonstrating advanced flext-core patterns."""

    def execute(self) -> FlextResult[ProcessedUser]:
        """Main execution method required by FlextDomainService."""
        return FlextResult[ProcessedUser].fail("Use process_pipeline instead")

    def validate_user_input(
        self, name: str, email: str, age: int,
    ) -> FlextResult[UserData]:
        """Validate user input using direct validation patterns."""
        # Single validation call replaces manual try/catch
        user_dict = {"name": name, "email": email, "age": age}

        # Validate required fields
        name_val = user_dict.get("name", "")
        email_val = user_dict.get("email", "")
        age_val = user_dict.get("age")

        if not isinstance(name_val, str) or len(name_val) < 2:
            return FlextResult[UserData].fail("Name must be at least 2 characters")

        if (
            not isinstance(email_val, str)
            or "@" not in email_val
            or email_val.count("@") != 1
        ):
            return FlextResult[UserData].fail("Invalid email format")

        if not isinstance(age_val, int) or age_val < 0 or age_val > 150:
            return FlextResult[UserData].fail(
                "Age must be an integer between 0 and 150",
            )

        # Create strongly-typed value object
        try:
            validated_dict = {"name": name_val, "email": email_val, "age": age_val}
            user_data = UserData.model_validate(validated_dict)
            return FlextResult[UserData].ok(user_data)
        except Exception as e:
            return FlextResult[UserData].fail(f"User model validation failed: {e}")

    def create_processed_user(self, user_data: UserData) -> FlextResult[ProcessedUser]:
        """Create processed user with generated metadata."""
        processed = ProcessedUser(
            id=FlextUtilities.Generators.generate_id()[:8],
            name=user_data.name,
            email=user_data.email,
            age=user_data.age,
            status="active",
            created_at=FlextUtilities.generate_iso_timestamp(),
            correlation_id=FlextUtilities.Generators.generate_correlation_id(),
        )
        return FlextResult[ProcessedUser].ok(processed)

    def save_user(self, user: ProcessedUser) -> FlextResult[ProcessedUser]:
        """Simulate saving user with structured logging."""
        print(f"💾 Saving user: {user.name} ({user.email})")
        return FlextResult[ProcessedUser].ok(user)

    def process_pipeline(
        self, name: str, email: str, age: int,
    ) -> FlextResult[ProcessedUser]:
        """Railway-oriented pipeline with type-safe domain models."""
        return (
            self.validate_user_input(name, email, age)
            .flat_map(self.create_processed_user)
            .flat_map(self.save_user)
        )


def main() -> None:
    """Compact demonstration using data-driven approach."""
    print("🚀 Advanced FLEXT Core Railway Pattern Showcase")
    print("=" * 50)

    service = AdvancedRailwayService()

    # Data-driven test cases with expected outcomes
    test_scenarios: list[dict[str, tuple[str, str, int] | str]] = [
        {"data": ("Alice Johnson", "alice@example.com", 25), "expect": "success"},
        {"data": ("Bob", "bob@company.com", 30), "expect": "success"},
        {"data": ("", "empty@test.com", 25), "expect": "failure"},
        {"data": ("Charlie", "invalid-email", 28), "expect": "failure"},
        {"data": ("David", "david@test.org", 200), "expect": "failure"},
    ]

    print("\nAdvanced Railway Pipeline Results:")
    print("-" * 40)

    for i, scenario in enumerate(test_scenarios, 1):
        data_tuple = scenario["data"]
        expected = scenario["expect"]

        # Type-safe unpacking
        if isinstance(data_tuple, tuple) and len(data_tuple) == 3:
            name, email, age = data_tuple
            print(f"\n{i}. {name or '(empty)'} | {email} | {age}")
            result = service.process_pipeline(name, email, age)

            if result.is_success and expected == "success":
                user = result.value
                print(f"✅ Success: {user.name} (ID: {user.id})")
                print(f"   Created: {user.created_at[:19]}")
            elif result.is_failure and expected == "failure":
                error_msg = result.error or "Unknown error"
                print(f"✅ Expected failure: {error_msg[:50]}...")
            elif result.is_success:
                print("❌ Unexpected success (expected failure)")
            else:
                print(f"❌ Unexpected failure: {result.error or 'Unknown error'}")

    # Demonstrate railway composition benefits
    print("\n" + "=" * 50)
    print("Railway Composition Benefits:")
    print("-" * 40)

    # Single pipeline call handles all validation, processing, and saving
    success_result = service.process_pipeline("Emma Wilson", "emma@company.com", 32)
    if success_result.is_success:
        user = success_result.value
        print(f"✅ Complete pipeline: {user.name} processed and saved")
        print(f"   Type safety: {type(user).__name__} (strongly typed)")
        print("   Immutability: Value object pattern enforced")

    # Error propagation demonstration
    error_result = service.process_pipeline("X", "bad-email", -1)
    print(f"\n❌ Error propagation: {error_result.error}")
    print("   → Pipeline stops at first failure (railway pattern)")

    print("\n✅ Advanced Pattern Showcase Completed!")


if __name__ == "__main__":
    main()
