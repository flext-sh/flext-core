#!/usr/bin/env python3
"""03 - CQRS Commands: FLEXT Core Ecosystem Showcase.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

from flext_core import (
    FlextBus,
    FlextContainer,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextTypes,
)

# FLEXT Core ecosystem setup
logger = FlextLogger(__name__)  # FlextLogger structured output


class User(FlextModels.Entity):
    """User entity using FlextModels for better type safety."""

    name: str
    email: str  # Will use FlextModels.EmailAddress for validation
    age: int = 25

    def to_dict(self) -> FlextTypes.Core.Dict:
        """Serialization using FlextModels capabilities.

        Returns:
            FlextTypes.Core.Dict: Dictionary representation of the user

        """
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "age": self.age,
        }


class UserRepository:
    """Repository pattern encapsulating container access."""

    def __init__(self) -> None:
        """Initialize repository with container services."""
        self._container = FlextContainer.get_global()
        # Setup container services
        self._container.register("user_db", dict[str, User]())
        self._container.register("events", list[FlextModels.Event]())

    def save_user(self, user: User) -> None:
        """Save user with type-safe container access."""
        db_result = self._container.get("user_db")
        events_result = self._container.get("events")

        if db_result.is_success and events_result.is_success:
            db = cast("dict[str, User]", db_result.value)
            events = cast("list[FlextModels.Event]", events_result.value)

            if user.id:
                db[user.id] = user
                event = FlextModels.create_event(
                    "UserCreated",
                    {"user_id": user.id},
                    user.id,
                )
                events.append(event)

    def get_stats(self) -> tuple[int, int]:
        """Get repository statistics.

        Returns:
            tuple[int, int]: Tuple of (user_count, event_count)

        """
        db_result = self._container.get("user_db")
        events_result = self._container.get("events")

        if db_result.is_success and events_result.is_success:
            db = cast("dict[str, User]", db_result.value)
            events = cast("list[FlextModels.Event]", events_result.value)
            return len(db), len(events)
        return 0, 0


class CreateUserCommand(FlextModels.Command):
    """Flext CQRS pattern."""

    name: str
    email: str
    age: int = 25


class UserCommandHandler(FlextModels.Entity):
    """Command handler using proper FlextModels.Entity pattern."""

    def __init__(self, repository: UserRepository) -> None:
        """Initialize with repository dependency."""
        super().__init__()
        self._repository = repository

    def handle(self, message: CreateUserCommand) -> FlextResult[User]:
        """Create user with FlextModels validation and repository.

        Args:
            message: Create user command containing user data

        Returns:
            FlextResult[User]: Created user or error

        """
        # Use FlextModels.EmailAddress for validation
        email_result = FlextModels.EmailAddress.create(message.email)
        if email_result.is_failure:
            return FlextResult[User].fail("Invalid email format")

        # Create user with FlextModels.Entity
        user = User(
            name=message.name,
            email=message.email,
            age=message.age,
        )

        # Save using repository
        self._repository.save_user(user)

        logger.info(
            "🎯 User created with FLEXT Core!",
            user_id=user.id,
            features=[
                "FlextResult",
                "FlextHandlers",
                "FlextBus",
                "FlextModels",
            ],
        )

        return FlextResult[User].ok(user)


def demo_flext_ecosystem() -> None:
    """Advanced FLEXT Core features with repository pattern."""
    # Setup repository and command bus
    repository = UserRepository()
    bus = FlextBus()
    bus.register_handler(CreateUserCommand, UserCommandHandler(repository))

    # Execute command with FlextResult railway
    command = CreateUserCommand(
        name="Alice FLEXT Core",
        email="alice@flext.dev",
        age=28,
        command_type="CreateUser",
    )
    result = bus.execute(command)

    if result.success:
        user = cast("User", result.value)
        print(f"✅ User created: {user.name} ({user.email})")

        # Show statistics using repository
        users_count, events_count = repository.get_stats()
        print(f"💾 Database: {users_count} users, {events_count} events")
        print("📊 Features: FlextResult • FlextHandlers • FlextBus • FlextModels")
    else:
        print(f"❌ Failed: {result.error}")


def main() -> None:
    """Advanced FLEXT Core CQRS with Repository Pattern."""
    print("🚀 Advanced FLEXT Core CQRS Demo")
    print("=" * 35)
    print("Architecture: CQRS • Repository • DI • Events • Railway")
    print()
    demo_flext_ecosystem()
    print("\n✅ Advanced patterns with clean architecture!")


if __name__ == "__main__":
    main()
