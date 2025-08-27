#!/usr/bin/env python3
"""FLEXT Core - Working Examples.

Comprehensive examples demonstrating all major FLEXT Core functionality.
"""

import contextlib
from typing import cast

from flext_core import (
    FlextCommands,
    FlextFields,
    FlextResult,
    get_flext_container,
)

from .shared_domain import SharedDomainFactory, User as SharedUser


def _print_header() -> None:
    pass


def _demo_flext_result() -> None:
    FlextResult[str].ok("Operation successful")
    FlextResult[str].fail("Something went wrong")


def _demo_entity_shared_domain() -> SharedUser | None:
    user_result = SharedDomainFactory.create_user(
        name="John Doe",
        email="john@example.com",
        age=30,
    )
    if user_result.is_failure or user_result.value is None:
        return None
    user = user_result.value
    (user.status.value if hasattr(user.status, "value") else str(user.status))
    user.validate_domain_rules()
    return user


def _demo_commands() -> tuple[object, object]:
    class CreateUserCommand(FlextCommands.Models.Command):
        email: str
        name: str

        def validate_command(self) -> FlextResult[None]:
            if not self.email or "@" not in self.email:
                return FlextResult[None].fail("Invalid email")
            if not self.name.strip():
                return FlextResult[None].fail("Name required")
            return FlextResult[None].ok(None)

    class CreateUserHandler(
        FlextCommands.Handlers.CommandHandler[CreateUserCommand, SharedUser]
    ):
        def handle(self, command: CreateUserCommand) -> FlextResult[SharedUser]:
            return SharedDomainFactory.create_user(
                name=command.name,
                email=command.email,
                age=30,
            )

    # Ensure Pydantic forward references are resolved for local command
    with contextlib.suppress(Exception):
        CreateUserCommand.model_rebuild()
    command = CreateUserCommand(email="alice@example.com", name="Alice Smith")
    handler = CreateUserHandler()
    result = handler.execute(command)
    # Modern pattern: Check success and use value directly
    if result.success:
        pass
        # Process the user data
    return command, handler


def _demo_container() -> None:
    class UserService:
        def __init__(self, repository: object) -> None:
            self.repository = repository

        def create_user(self, email: str, name: str) -> SharedUser:
            result = SharedDomainFactory.create_user(name=name, email=email, age=25)
            # Modern pattern: Use expect() for unwrapping with custom error
            return result.expect("User creation failed")

    class UserRepository:
        def __init__(self) -> None:
            self.users: dict[str, SharedUser] = {}

        def save(self, user: SharedUser) -> SharedUser:
            self.users[str(user.id)] = user
            return user

    container = get_flext_container()
    repository = UserRepository()
    container.register("user_repository", repository)
    service = UserService(repository)
    container.register("user_service", service)
    service_result = container.get("user_service")
    if service_result.success:
        user_service = service_result.value
        if hasattr(user_service, "create_user"):
            user_service.create_user("bob@example.com", "Bob Wilson")  # type: ignore[attr-defined]


def _demo_fields() -> None:
    # Using FlextFields Factory to create string field
    builder = FlextFields.Factory.FieldBuilder("string", "email")
    email_field = builder.build().unwrap()
    # Field validation - using getattr for safe method access
    validate_method = getattr(email_field, "validate", None)
    if validate_method is not None:
        validate_method("user@example.com")
        validate_method("invalid-email")
    else:
        # Field created successfully but validation method may differ
        pass


def _demo_command_bus(command: object, handler: object) -> None:
    bus = FlextCommands.Bus()
    bus.register_handler(cast("type", type(command)), handler)
    bus_result = bus.execute(cast("FlextCommands.Models.Command", command))
    if bus_result.success:
        bus_user = bus_result.value
        if hasattr(bus_user, "name"):
            pass


def main() -> None:
    """Execute main function for working examples."""
    _print_header()
    _demo_flext_result()
    user = _demo_entity_shared_domain()
    if user is None:
        return
    command, handler = _demo_commands()
    _demo_container()
    _demo_fields()
    _demo_command_bus(command, handler)


if __name__ == "__main__":
    main()
