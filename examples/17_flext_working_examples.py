#!/usr/bin/env python3
"""FLEXT Core - Working Examples.

Comprehensive examples demonstrating all major FLEXT Core functionality.
"""

from shared_domain import SharedDomainFactory, User as SharedUser

from flext_core import (
    FlextCommands,
    FlextFields,
    FlextResult,
    get_flext_container,
)


def main() -> None:  # noqa: PLR0915
    """Execute main function for working examples."""
    print("=== FLEXT Core Working Examples ===\n")

    # 1. FlextResult - Railway Pattern
    print("1. FlextResult Examples:")
    success_result = FlextResult.ok("Operation successful")
    print(f"  Success: {success_result.success}, Data: {success_result.data}")

    error_result: FlextResult[str] = FlextResult.fail("Something went wrong")
    print(f"  Error: {error_result.is_failure}, Error: {error_result.error}")
    print()

    # 2. FlextEntity - Domain Modeling (using shared domain)
    print("2. FlextEntity Examples (using shared domain):")

    # Use SharedUser from shared_domain instead of local User class
    user_result = SharedDomainFactory.create_user(
        name="John Doe",
        email="john@example.com",
        age=30,
    )

    if user_result.is_failure:
        print(f"Failed to create user: {user_result.error}")
        return

    user = user_result.data

    if user is None:
        print("❌ Operation returned None data")

        return

    print(
        f"  User: {user.name} ({user.email_address.email}), "
        f"Status: {user.status.value}",
    )

    validation = user.validate_domain_rules()
    print(f"  Validation: {validation.success}")
    print()

    # 3. FlextCommands - CQRS Pattern
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
            # Create user using SharedDomainFactory
            return SharedDomainFactory.create_user(
                name=command.name,
                email=command.email,
                age=30,  # Default age
            )

    # Create and execute command
    command = CreateUserCommand(email="alice@example.com", name="Alice Smith")
    handler = CreateUserHandler()

    print(f"  Command: {command.command_type}")
    print(f"  Command ID: {command.command_id}")

    result = handler.execute(command)
    if result.success:
        created_user = result.data

        if created_user is None:
            print("❌ Operation returned None data")

            return
        print(f"  Created: {created_user.name} ({created_user.email_address.email})")
    print()

    # 4. FlextContainer - Dependency Injection
    print("4. FlextContainer Examples:")

    class UserService:
        def __init__(self, repository: object) -> None:
            self.repository = repository

        def create_user(self, email: str, name: str) -> SharedUser:
            # Use SharedDomainFactory for consistent user creation
            result = SharedDomainFactory.create_user(name=name, email=email, age=25)
            if result.success and result.data is not None:
                return result.data
            # Fallback - this shouldn't happen in practice
            msg: str = f"Failed to create user: {result.error}"
            raise ValueError(msg)

    class UserRepository:
        def __init__(self) -> None:
            self.users: dict[str, SharedUser] = {}

        def save(self, user: SharedUser) -> SharedUser:
            self.users[user.id] = user
            return user

    # Get global container
    container = get_flext_container()

    # Register dependencies
    repository = UserRepository()
    container.register("user_repository", repository)

    service = UserService(repository)
    container.register("user_service", service)

    # Retrieve and use service
    service_result = container.get("user_service")
    if service_result.success:
        user_service = service_result.data
        if hasattr(user_service, "create_user"):
            new_user = user_service.create_user("bob@example.com", "Bob Wilson")
            print(f"  Service created: {new_user.name} ({new_user.email_address.email})")
        else:
            print("  Service creation failed: no create_user method")
    print()

    # 5. FlextFields - Field Validation
    print("5. FlextFields Examples:")

    # Create email field
    email_field = FlextFields.create_string_field(
        field_id="user_email",
        field_name="email",
        pattern=r"^[^@]+@[^@]+\.[^@]+$",
        required=True,
        description="User email address",
    )

    print(f"  Field: {email_field.field_name} ({email_field.field_type})")

    # Test validation
    valid_email = email_field.validate_value("user@example.com")
    invalid_email = email_field.validate_value("invalid-email")

    print(f"  Valid email: {valid_email.success}")
    print(f"  Invalid email: {invalid_email.success} - {invalid_email.error}")
    print()

    # 6. Command Bus Pattern
    print("6. Command Bus Examples:")

    bus = FlextCommands.create_command_bus()

    # Register handler
    registration = bus.register_handler(CreateUserCommand, handler)
    print(f"  Handler registered: {registration.success}")

    # Execute via bus
    bus_result = bus.execute(command)
    if bus_result.success:
        bus_user = bus_result.data
        if hasattr(bus_user, "name"):
            print(f"  Bus executed: {bus_user.name} created")
        else:
            print(f"  Bus executed: {bus_user}")
    print()

    print("=== All Examples Completed Successfully! ===")


if __name__ == "__main__":
    main()
