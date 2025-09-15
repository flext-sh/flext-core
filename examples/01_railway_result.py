#!/usr/bin/env python3
"""01 - Basic Result Pattern using FlextCore.

Demonstrates basic result pattern with FlextCore:
- FlextResult for success/failure handling
- Clean error propagation
- Safe data extraction with unwrap()
- Pydantic v2 validation integration

Note: This demonstrates the current basic implementation.
Advanced railway-oriented composition is planned for future versions.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pydantic import BaseModel, EmailStr, Field, ValidationError

from flext_core import FlextCore, FlextResult, FlextUtilities


class UserInput(BaseModel):
    """Validated input for user processing (Pydantic v2)."""

    name: str = Field(min_length=2)
    email: EmailStr
    age: int = Field(ge=0, le=150)


class RailwayExample:
    """Simple railway-oriented programming example using FlextCore facade."""

    def __init__(self) -> None:
        """Initialize with FlextCore facade."""
        self._core = FlextCore()

    def process_user_data(
        self, name: str, email: str, age: int
    ) -> FlextResult[dict[str, object]]:
        """Process user data with Pydantic validation and railway composition."""
        try:
            validated = UserInput.model_validate(
                {"name": name, "email": email, "age": age}
            )
        except ValidationError as exc:  # Pydantic v2
            return FlextResult[dict[str, object]].fail(
                f"Validation failed: {exc.errors()}"
            )

        # All validations passed, create user data
        user_data: dict[str, object] = {
            "id": FlextUtilities.Generators.generate_id()[:8],
            "name": validated.name,
            "email": str(validated.email),
            "age": validated.age,
            "status": "active",
        }

        return FlextResult[dict[str, object]].ok(user_data)

    def enrich_user_data(
        self, user_data: dict[str, object]
    ) -> FlextResult[dict[str, object]]:
        """Enrich user data with additional fields."""
        enriched = user_data.copy()
        enriched["created_at"] = FlextUtilities.generate_iso_timestamp()
        enriched["correlation_id"] = FlextUtilities.Generators.generate_correlation_id()
        return FlextResult[dict[str, object]].ok(enriched)

    def save_user(self, user_data: dict[str, object]) -> FlextResult[dict[str, object]]:
        """Simulate saving user (in real app, would save to database)."""
        print(f"💾 Saving user: {user_data['name']} ({user_data['email']})")
        return FlextResult[dict[str, object]].ok(user_data)

    def process_pipeline(
        self, name: str, email: str, age: int
    ) -> FlextResult[dict[str, object]]:
        """Complete processing pipeline using railway composition."""
        # Chain operations using flat_map for railway-oriented programming
        return (
            self.process_user_data(name, email, age)
            .flat_map(self.enrich_user_data)
            .flat_map(self.save_user)
        )


def main() -> None:
    """Main demonstration."""
    print("🚀 FlextCore Railway-Oriented Programming Example")
    print("=" * 50)

    service = RailwayExample()

    # Test cases
    test_cases = [
        ("Alice Johnson", "alice@example.com", 25),
        ("Bob", "bob@company.com", 30),
        ("", "empty@test.com", 25),  # Invalid: empty name
        ("Charlie Brown", "invalid-email", 28),  # Invalid: bad email
        ("David Lee", "david@test.org", 200),  # Invalid: age out of range
    ]

    print("\nProcessing users through railway pipeline:")
    print("-" * 40)

    for name, email, age in test_cases:
        print(f"\nProcessing: {name or '(empty)'} | {email} | {age}")
        result = service.process_pipeline(name, email, age)

        if result.is_success:
            user = result.value
            print(f"✅ Success: User {user['id']} created")
            print(f"   Created at: {user.get('created_at', 'N/A')}")
            correlation_id = user.get("correlation_id", "N/A")
            if isinstance(correlation_id, str):
                print(f"   Correlation: {correlation_id[:20]}...")
            else:
                print(f"   Correlation: {correlation_id}")
        else:
            print(f"❌ Failed: {result.error}")

    # Demonstrate railway composition
    print("\n" + "=" * 50)
    print("Railway Composition Example:")
    print("-" * 40)

    # Create a valid user
    valid_result = service.process_user_data("Jane Doe", "jane@example.com", 28)

    if valid_result.is_success:
        print("✅ Initial validation passed")

        # Chain multiple operations
        enriched = valid_result.flat_map(service.enrich_user_data)
        if enriched.is_success:
            print("✅ Data enrichment successful")

        # Use map to transform without changing success/failure
        transformed = enriched.map(lambda d: {**d, "processed": True})
        if transformed.is_success:
            print("✅ Data transformation successful")
            print(f"   Final data: processed={transformed.value.get('processed')}")

    # Demonstrate error propagation
    print("\nError Propagation Example:")
    print("-" * 40)

    invalid_result = service.process_user_data("X", "bad-email", -5)
    print(f"❌ First error encountered: {invalid_result.error}")

    # Chain will stop at first error
    chained = (
        invalid_result.flat_map(service.enrich_user_data).flat_map(
            service.save_user
        )  # Won't execute  # Won't execute
    )
    print(f"❌ Error propagated through chain: {chained.error}")

    print("\n✅ Railway-Oriented Programming Example Completed!")


if __name__ == "__main__":
    main()
