#!/usr/bin/env python3
"""03 - CQRS Commands: Clean Command Query Separation.

Shows how FlextCommands simplify CQRS implementation.
Demonstrates commands, queries, handlers, and event sourcing.

Key Patterns:
• FlextCommands for CQRS separation
• Command validation and execution
• Query handling with projections
• Simple event sourcing
"""

from datetime import UTC, datetime
from typing import cast, override

from shared_domain import EmailAddress, SharedDomainFactory, User

from flext_core import FlextCommands, FlextResult

# =============================================================================
# SIMPLE EVENT STORE - For demonstration
# =============================================================================


class SimpleEventStore:
    """Simple event store for CQRS demonstration."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def add_event(self, event_type: str, data: dict[str, object]) -> None:
        """Add event to store."""
        event: dict[str, object] = {
            "type": event_type,
            "data": data,
            "timestamp": "now",
        }
        self.events.append(event)

    def get_events(self) -> list[dict[str, object]]:
        """Get all events."""
        return self.events.copy()


# Global event store
event_store = SimpleEventStore()

# =============================================================================
# USER DATABASE - Simple in-memory storage
# =============================================================================


class UserDatabase:
    """Simple user database for demonstration."""

    def __init__(self) -> None:
        self.users: dict[str, User] = {}

    def save(self, user: User) -> None:
        """Save user to database."""
        self.users[str(user.id)] = user
        event_store.add_event("UserSaved", {"user_id": str(user.id), "name": user.name})

    def get(self, user_id: str) -> User | None:
        """Get user by ID."""
        return self.users.get(user_id)

    def list_all(self) -> list[User]:
        """List all users."""
        return list(self.users.values())

    def delete(self, user_id: str) -> bool:
        """Delete user by ID."""
        if user_id in self.users:
            del self.users[user_id]
            event_store.add_event("UserDeleted", {"user_id": user_id})
            return True
        return False


# Global database
user_db = UserDatabase()

# =============================================================================
# COMMANDS - Write operations with business intent
# =============================================================================


class CreateUserCommand(FlextCommands.Command):
    """Command to create a new user."""

    name: str
    email: str
    age: int


class UpdateUserCommand(FlextCommands.Command):
    """Command to update a user."""

    target_user_id: str  # Renamed to avoid conflict with base class user_id
    name: str | None = None
    email: str | None = None


class DeleteUserCommand(FlextCommands.Command):
    """Command to delete a user."""

    target_user_id: str  # Renamed to avoid conflict with base class user_id


# =============================================================================
# QUERIES - Read operations
# =============================================================================


class GetUserQuery(FlextCommands.Query):
    """Query to get a specific user."""

    user_id: str


class ListUsersQuery(FlextCommands.Query):
    """Query to list all users."""


class GetEventsQuery(FlextCommands.Query):
    """Query to get all events."""


# =============================================================================
# COMMAND HANDLERS - Business logic for write operations
# =============================================================================


class CreateUserHandler(FlextCommands.Handler[CreateUserCommand, User]):
    """Handler for user creation."""

    @override
    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        """Create user with validation."""
        return SharedDomainFactory.create_user(
            command.name, command.email, command.age
        ).map(self._save_user)

    def _save_user(self, user: User) -> User:
        """Save user to database."""
        user_db.save(user)
        return user


class UpdateUserHandler(FlextCommands.Handler[UpdateUserCommand, User]):
    """Handler for user updates."""

    @override
    def handle(self, command: UpdateUserCommand) -> FlextResult[User]:
        """Update existing user."""
        user = user_db.get(command.target_user_id)
        if not user:
            return FlextResult[User].fail(f"User not found: {command.target_user_id}")
        if command.name:
            user.name = command.name
        if command.email:
            try:
                email_address = EmailAddress(email=command.email)
                validation_result = email_address.validate_business_rules()
                if validation_result.is_failure:
                    return FlextResult[User].fail(
                        f"Invalid email: {validation_result.error}"
                    )
                user.email_address = email_address
            except Exception as e:
                return FlextResult[User].fail(f"Invalid email: {e}")

        user_db.save(user)
        return FlextResult[User].ok(user)


class DeleteUserHandler(FlextCommands.Handler[DeleteUserCommand, bool]):
    """Handler for user deletion."""

    @override
    def handle(self, command: DeleteUserCommand) -> FlextResult[bool]:
        """Delete user by ID."""
        success = user_db.delete(command.target_user_id)
        if success:
            success = True
            return FlextResult[bool].ok(success)
        return FlextResult[bool].fail(f"User not found: {command.target_user_id}")


# =============================================================================
# QUERY HANDLERS - Read operations
# =============================================================================


class GetUserQueryHandler(FlextCommands.QueryHandler[GetUserQuery, User]):
    """Handler for user lookup."""

    @override
    def handle(self, query: GetUserQuery) -> FlextResult[User]:
        """Get user by ID."""
        user = user_db.get(query.user_id)
        if user:
            return FlextResult[User].ok(user)
        return FlextResult[User].fail(f"User not found: {query.user_id}")


class ListUsersQueryHandler(FlextCommands.QueryHandler[ListUsersQuery, list[User]]):
    """Handler for user listing."""

    @override
    def handle(self, query: ListUsersQuery) -> FlextResult[list[User]]:
        """List all users."""
        users = user_db.list_all()
        return FlextResult[list[User]].ok(users)


class GetEventsQueryHandler(
    FlextCommands.QueryHandler[GetEventsQuery, list[dict[str, object]]]
):
    """Handler for event listing."""

    @override
    def handle(self, query: GetEventsQuery) -> FlextResult[list[dict[str, object]]]:
        """Get all events."""
        events = event_store.get_events()
        return FlextResult[list[dict[str, object]]].ok(events)


# =============================================================================
# CQRS SETUP - Configure command bus and query handlers
# =============================================================================


def setup_cqrs() -> tuple[FlextCommands.Bus, dict[str, object]]:
    """Setup CQRS infrastructure."""
    # Create command bus
    bus = FlextCommands.Bus()

    # Register command handlers
    bus.register_handler(CreateUserCommand, CreateUserHandler())
    bus.register_handler(UpdateUserCommand, UpdateUserHandler())
    bus.register_handler(DeleteUserCommand, DeleteUserHandler())

    # Create query handlers
    query_handlers: dict[str, object] = {
        "get_user": GetUserQueryHandler(),
        "list_users": ListUsersQueryHandler(),
        "get_events": GetEventsQueryHandler(),
    }

    return bus, query_handlers


# =============================================================================
# APPLICATION SERVICE - High-level CQRS orchestration
# =============================================================================


class UserService:
    """Simple user management service using CQRS."""

    def __init__(
        self, command_bus: FlextCommands.Bus, query_handlers: dict[str, object]
    ) -> None:
        self.bus = command_bus
        self.queries = query_handlers

    def create_user(self, name: str, email: str, age: int) -> FlextResult[User]:
        """Create a new user."""
        command = CreateUserCommand(
            name=name,
            email=email,
            age=age,
            command_type="",
            timestamp=datetime.now(tz=UTC),
            user_id=None,
            correlation_id="",
        )
        result = self.bus.execute(command)
        return result.map(lambda data: cast("User", data))

    def update_user(
        self, user_id: str, name: str | None = None, email: str | None = None
    ) -> FlextResult[User]:
        """Update an existing user."""
        command = UpdateUserCommand(
            target_user_id=user_id,
            name=name,
            email=email,
            command_type="",
            timestamp=datetime.now(tz=UTC),
            user_id=None,
            correlation_id="",
        )
        result = self.bus.execute(command)
        return result.map(lambda data: cast("User", data))

    def delete_user(self, user_id: str) -> FlextResult[bool]:
        """Delete a user."""
        command = DeleteUserCommand(
            target_user_id=user_id,
            command_id="",
            command_type="",
            timestamp=datetime.now(tz=UTC),
            user_id=None,
            correlation_id="",
        )
        result = self.bus.execute(command)
        return result.map(lambda data: cast("bool", data))

    def get_user(self, user_id: str) -> FlextResult[User]:
        """Get user by ID."""
        query = GetUserQuery(
            user_id=user_id,
            query_id=None,
            query_type=None,
            page_size=100,
            page_number=1,
            sort_by=None,
            sort_order="asc",
        )
        handler = self.queries["get_user"]
        result = cast("FlextCommands.QueryHandler[object, object]", handler).handle(
            query
        )
        return cast("FlextResult[User]", result)

    def list_users(self) -> FlextResult[list[User]]:
        """List all users."""
        query = ListUsersQuery(
            query_id=None,
            query_type=None,
            page_size=100,
            page_number=1,
            sort_by=None,
            sort_order="asc",
        )
        handler = self.queries["list_users"]
        result = cast("FlextCommands.QueryHandler[object, object]", handler).handle(
            query
        )
        return cast("FlextResult[list[User]]", result)

    def get_events(self) -> FlextResult[list[dict[str, object]]]:
        """Get all events."""
        query = GetEventsQuery(
            query_id=None,
            query_type=None,
            page_size=100,
            page_number=1,
            sort_by=None,
            sort_order="asc",
        )
        handler = self.queries["get_events"]
        result = cast("FlextCommands.QueryHandler[object, object]", handler).handle(
            query
        )
        return cast("FlextResult[list[dict[str, object]]]", result)


# =============================================================================
# DEMONSTRATIONS - Real-world CQRS usage
# =============================================================================


def demo_command_operations() -> None:
    """Demonstrate command operations (writes)."""
    bus, queries = setup_cqrs()
    service = UserService(bus, queries)

    # Create user
    create_result = service.create_user("Alice Johnson", "alice@example.com", 25)
    if create_result.success:
        user = create_result.value

        # Update user
        update_result = service.update_user(str(user.id), name="Alice Smith")
        if update_result.success:
            pass


def demo_query_operations() -> None:
    """Demonstrate query operations (reads)."""
    bus, queries = setup_cqrs()
    service = UserService(bus, queries)

    # List users
    list_result = service.list_users()
    if list_result.success:
        users = list_result.value
        for _user in users:
            pass


def demo_event_sourcing() -> None:
    """Demonstrate event sourcing."""
    bus, queries = setup_cqrs()
    service = UserService(bus, queries)

    # Get events
    events_result = service.get_events()
    if events_result.success:
        events = events_result.value
        for _event in events:
            pass


def demo_validation_handling() -> None:
    """Demonstrate validation and error handling."""
    bus, queries = setup_cqrs()
    service = UserService(bus, queries)

    # Try invalid data
    invalid_result = service.create_user("", "invalid-email", 15)
    if invalid_result.is_failure:
        pass


def main() -> None:
    """🎯 Example 03: CQRS Commands Pattern."""
    # Show the refactored CQRS patterns
    demo_command_operations()
    demo_query_operations()
    demo_event_sourcing()
    demo_validation_handling()


if __name__ == "__main__":
    main()
