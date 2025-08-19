#!/usr/bin/env python3
"""FLEXT Core - Working Examples.

Comprehensive examples demonstrating all major FLEXT Core functionality.
"""

import contextlib
from typing import cast

from examples.shared_domain import SharedDomainFactory, User as SharedUser
from flext_core import (
    FlextCommands,
    FlextFields,
    FlextResult,
    get_flext_container,
)


def _print_header() -> None:
    pass


def _demo_flext_result() -> None:
    FlextResult[None].ok("Operation successful")
    FlextResult[None].fail("Something went wrong")


def _demo_entity_shared_domain() -> SharedUser | None:
    user_result = SharedDomainFactory.create_user(
        name="John Doe",
        email="john@example.com",
        age=30,
    )
    if user_result.is_failure or user_result.data is None:
        return None
    user = user_result.data
    (user.status.value if hasattr(user.status, "value") else str(user.status))
    user.validate_domain_rules()
    return user


def _demo_commands() -> tuple[object, object]:
    class CreateUserCommand(FlextCommands.Command):
        email: str
        name: str

        def validate_command(self) -> FlextResult[None]:
            if not self.email or "@" not in self.email:
                return FlextResult[None].fail("Invalid email")
            if not self.name.strip():
                return FlextResult[None].fail("Name required")
            return FlextResult[None].ok(None)

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
    result = handler.execute(command)
    if result.success and result.data is not None:
        pass
    return command, handler


def _demo_container() -> None:
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
            user_service.create_user("bob@example.com", "Bob Wilson")


def _demo_fields() -> None:
    email_field = FlextFields.create_string_field(
        field_id="user_email",
        field_name="email",
        pattern=r"^[^@]+@[^@]+\.[^@]+$",
        required=True,
        description="User email address",
    )
    email_field.validate_value("user@example.com")
    email_field.validate_value("invalid-email")


def _demo_command_bus(command: object, handler: object) -> None:
    bus = FlextCommands.create_command_bus()
    bus.register_handler(cast("type", type(command)), handler)
    bus_result = bus.execute(command)
    if bus_result.success:
        bus_user = bus_result.data
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
