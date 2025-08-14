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
    print("=== FLEXT Core Working Examples ===\n")


def _demo_flext_result() -> None:
    print("1. FlextResult Examples:")
    success_result = FlextResult.ok("Operation successful")
    print(f"  Success: {success_result.success}, Data: {success_result.data}")
    error_result: FlextResult[str] = FlextResult.fail("Something went wrong")
    print(f"  Error: {error_result.is_failure}, Error: {error_result.error}")
    print()


def _demo_entity_shared_domain() -> SharedUser | None:
    print("2. FlextEntity Examples (using shared domain):")
    user_result = SharedDomainFactory.create_user(
        name="John Doe",
        email="john@example.com",
        age=30,
    )
    if user_result.is_failure or user_result.data is None:
        print(f"Failed to create user: {user_result.error}")
        return None
    user = user_result.data
    status_value = (
        user.status.value if hasattr(user.status, "value") else str(user.status)
    )
    print(
        f"  User: {user.name} ({user.email_address.email}), Status: {status_value}",
    )
    validation = user.validate_domain_rules()
    print(f"  Validation: {validation.success}")
    print()
    return user


def _demo_commands() -> tuple[object, object]:
    print("3. FlextCommands Examples:")

    class CreateUserCommand(FlextCommands.Command):
        email: str
        name: str

        def validate_command(self) -> FlextResult[None]:
            if not self.email or "@" not in self.email:
                return FlextResult.fail("Invalid email")
            if not self.name.strip():
                return FlextResult.fail("Name required")
            return FlextResult.ok(None)

    class CreateUserHandler(FlextCommands.Handler[CreateUserCommand, SharedUser]):
        def handle(self, command: CreateUserCommand) -> FlextResult[SharedUser]:
            return SharedDomainFactory.create_user(
                name=command.name,
                email=command.email,
                age=30,
            )

    # Ensure Pydantic forward references are resolved for local command
    with contextlib.suppress(Exception):
        CreateUserCommand.model_rebuild(
            types_namespace={
                "TEntityId": str,
                "TServiceName": str,
                "TUserId": str,
                "TCorrelationId": str,
            },
        )
    command = CreateUserCommand(email="alice@example.com", name="Alice Smith")
    handler = CreateUserHandler()
    print(f"  Command: {command.command_type}")
    print(f"  Command ID: {command.command_id}")
    result = handler.execute(command)
    if result.success and result.data is not None:
        created_user = result.data
        print(f"  Created: {created_user.name} ({created_user.email_address.email})")
    print()
    return command, handler


def _demo_container() -> None:
    print("4. FlextContainer Examples:")

    class UserService:
        def __init__(self, repository: object) -> None:
            self.repository = repository

        def create_user(self, email: str, name: str) -> SharedUser:
            result = SharedDomainFactory.create_user(name=name, email=email, age=25)
            if result.success and result.data is not None:
                return result.data
            msg: str = f"Failed to create user: {result.error}"
            raise ValueError(msg)

    class UserRepository:
        def __init__(self) -> None:
            self.users: dict[str, SharedUser] = {}

        def save(self, user: SharedUser) -> SharedUser:
            self.users[user.id] = user
            return user

    container = get_flext_container()
    repository = UserRepository()
    container.register("user_repository", repository)
    service = UserService(repository)
    container.register("user_service", service)
    service_result = container.get("user_service")
    if service_result.success:
        user_service = service_result.data
        if hasattr(user_service, "create_user"):
            new_user = user_service.create_user("bob@example.com", "Bob Wilson")
            print(
                f"  Service created: {new_user.name} ({new_user.email_address.email})",
            )
        else:
            print("  Service creation failed: no create_user method")
    print()


def _demo_fields() -> None:
    print("5. FlextFields Examples:")
    email_field = FlextFields.create_string_field(
        field_id="user_email",
        field_name="email",
        pattern=r"^[^@]+@[^@]+\.[^@]+$",
        required=True,
        description="User email address",
    )
    print(f"  Field: {email_field.field_name} ({email_field.field_type})")
    valid_email = email_field.validate_value("user@example.com")
    invalid_email = email_field.validate_value("invalid-email")
    print(f"  Valid email: {valid_email.success}")
    print(f"  Invalid email: {invalid_email.success} - {invalid_email.error}")
    print()


def _demo_command_bus(command: object, handler: object) -> None:
    print("6. Command Bus Examples:")
    bus = FlextCommands.create_command_bus()
    bus.register_handler(cast("type", type(command)), handler)  # type: ignore[arg-type]
    print("  Handler registered successfully")
    bus_result = bus.execute(command)
    if bus_result.success:
        bus_user = bus_result.data
        if hasattr(bus_user, "name"):
            print(f"  Bus executed: {bus_user.name} created")
        else:
            print(f"  Bus executed: {bus_user}")
    print()


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
    print("=== All Examples Completed Successfully! ===")


if __name__ == "__main__":
    main()
