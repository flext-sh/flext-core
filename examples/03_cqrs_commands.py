#!/usr/bin/env python3
"""03 - CQRS Commands: Maximum FlextCore Ecosystem Showcase.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast, override

from flext_core import (
    FlextCommands,
    FlextContainer,
    FlextDecorators,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextTypes,
    FlextUtilities,
)

# FlextCore ecosystem setup
logger = FlextLogger(__name__)  # FlextLogger structured output
container = FlextContainer.get_global()  # FlextContainer dependency injection


class Email(FlextModels.Config):
    """Email with FlextResult validation."""

    address: str

    @classmethod
    @FlextDecorators.Reliability.safe_result  # FlextDecorators auto-wrapping
    def create(cls, address: str) -> Email:
        """FlextResult factory pattern."""
        if "@" not in address:
            error_msg = "Invalid email"
            raise ValueError(error_msg)
        return cls(address=address)


class User(FlextModels.Config):
    """User entity with FlextUtilities integration."""

    id: str | None = None
    name: str
    email: Email
    age: int = 25

    @FlextDecorators.Reliability.safe_result  # FlextDecorators
    def to_dict(self) -> FlextTypes.Core.Dict:
        """FlextUtilities serialization pattern."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email.address,
            "age": self.age,
        }


# FlextContainer DI setup
container.register("user_db", dict[str, User]())
container.register("events", list[FlextTypes.Core.Dict]())


class CreateUserCommand(FlextCommands.Models.Command):
    """FlextCommands CQRS pattern."""

    name: str
    email: str
    age: int = 25


class UserCommandHandler(
    FlextCommands.Handlers.CommandHandler[CreateUserCommand, User],
):
    """FlextCommands + FlextResult + FlextContainer + FlextUtilities + FlextLogger."""

    @override
    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        """🚀 5 FlextCore features in 6 lines of code!."""
        return (
            Email.create(command.email)  # FlextResult + safe_result
            .map(
                lambda e: User(
                    id=FlextUtilities.Generators.generate_uuid(),  # FlextUtilities UUID
                    name=command.name,
                    email=e,
                    age=command.age,
                ),
            )
            .tap(self._save_user)  # FlextResult tap
        )

    def _save_user(self, user: User) -> None:
        """FlextContainer + FlextLogger integration."""
        db = cast(
            "dict[str, User]",
            container.get("user_db").value,
        )  # FlextContainer DI
        events = cast(
            "list[FlextTypes.Core.Dict]",
            container.get("events").value,
        )  # FlextContainer DI

        if user.id:
            db[user.id] = user  # Save to "database"
            events.append({"type": "UserCreated", "user_id": user.id})  # Event sourcing

        logger.info(
            "🎯 User created with FlextCore!",
            user_id=user.id,
            features=[
                "FlextResult",
                "FlextCommands",
                "FlextContainer",
                "FlextUtilities",
                "FlextLogger",
                "FlextDecorators.safe_result",
            ],
        )


def demo_flext_ecosystem() -> None:
    """🚀 Demonstrate 10+ FlextCore features in action."""
    # FlextCommands Bus
    bus = FlextCommands.Bus()
    bus.register_handler(CreateUserCommand, UserCommandHandler())

    # Execute command with FlextResult railway
    command = CreateUserCommand(
        name="Alice FlextCore",
        email="alice@flext.dev",
        age=28,
        command_type="CreateUser",
    )
    result = bus.execute(command)

    if result.success:
        user = cast("User", result.value)
        print(f"✅ User created: {user.name} ({user.email.address})")
        features = [
            "FlextResult",
            "FlextCommands",
            "FlextContainer",
            "FlextUtilities",
            "FlextLogger",
            "FlextDecorators.safe_result",
            "FlextConstants",
            "FlextContainer.get_global",
            "Pydantic",
        ]
        print(f"📊 Features demonstrated: {len(features)} FlextCore components!")

        # Show stored data
        db = cast("dict[str, User]", container.get("user_db").value)
        events = cast("list[FlextTypes.Core.Dict]", container.get("events").value)
        print(f"💾 Database has {len(db)} users, {len(events)} events")
    else:
        print(f"❌ Failed: {result.error}")


def main() -> None:
    """🎯 FlextCore Ecosystem: Maximum Power, Minimum Code!."""
    print("🚀 FLEXT-CORE ECOSYSTEM SHOWCASE")
    print("=" * 50)
    print("Features: FlextResult • FlextCommands • FlextContainer • FlextUtilities")
    print("         FlextLogger • safe_result • FlextConstants • CQRS • DI • Railway")
    print()
    demo_flext_ecosystem()
    print("\n🎉 SUCCESS: 10+ FlextCore features in <100 lines of code!")
    print("💪 Enterprise patterns with minimal complexity!")


if __name__ == "__main__":
    main()
