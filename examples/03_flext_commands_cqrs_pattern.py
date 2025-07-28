#!/usr/bin/env python3
"""FLEXT Commands - CQRS Pattern Example.

Demonstrates Command Query Responsibility Segregation (CQRS) using FlextCommands
with command handling, event sourcing, and real-world patterns.

Features demonstrated:
- CQRS command and query separation
- Command handlers with business logic
- Command bus for routing and execution
- Event sourcing integration
- Command validation and metadata
- Performance monitoring and metrics
- Type-safe command patterns
"""

from __future__ import annotations

from typing import Any

from flext_core import FlextResult
from flext_core.commands import FlextCommands
from flext_core.utilities import FlextUtilities

# =============================================================================
# DOMAIN EVENTS - Event sourcing support
# =============================================================================


class DomainEvent:
    """Base domain event for event sourcing."""

    def __init__(self, event_type: str, data: dict[str, Any]) -> None:
        self.event_id = FlextUtilities.generate_entity_id()
        self.event_type = event_type
        self.data = data
        self.timestamp = FlextUtilities.generate_iso_timestamp()
        self.correlation_id = FlextUtilities.generate_correlation_id()

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
        }


class EventStore:
    """Simple event store for demonstration."""

    def __init__(self) -> None:
        self.events: list[DomainEvent] = []

    def append_event(self, event: DomainEvent) -> FlextResult[str]:
        """Append event to store."""
        self.events.append(event)
        print(f"📝 Event stored: {event.event_type} ({event.event_id})")
        return FlextResult.ok(event.event_id)

    def get_events_by_correlation(self, correlation_id: str) -> list[DomainEvent]:
        """Get events by correlation ID."""
        return [e for e in self.events if e.correlation_id == correlation_id]

    def get_all_events(self) -> list[DomainEvent]:
        """Get all events."""
        return self.events.copy()


# Global event store instance
event_store = EventStore()


# =============================================================================
# COMMANDS - Write operations with business intent
# =============================================================================


class CreateUserCommand(FlextCommands.Command):
    """Command to create a new user."""

    name: str
    email: str
    age: int

    def validate_command(self) -> FlextResult[None]:
        """Validate create user command."""
        print(f"🔍 Validating CreateUserCommand: {self.name}")

        # Business rule validation
        if not self.name or len(self.name.strip()) == 0:
            return FlextResult.fail("User name cannot be empty")

        if "@" not in self.email:
            return FlextResult.fail("Invalid email format")

        if self.age < 18 or self.age > 120:
            return FlextResult.fail("Age must be between 18 and 120")

        print("✅ CreateUserCommand validation passed")
        return FlextResult.ok(None)


class UpdateUserCommand(FlextCommands.Command):
    """Command to update an existing user."""

    target_user_id: str
    name: str | None = None
    email: str | None = None

    def validate_command(self) -> FlextResult[None]:
        """Validate update user command."""
        print(f"🔍 Validating UpdateUserCommand: {self.target_user_id}")

        if not self.target_user_id:
            return FlextResult.fail("Target user ID is required")

        if self.email and "@" not in self.email:
            return FlextResult.fail("Invalid email format")

        if not self.name and not self.email:
            return FlextResult.fail("At least one field must be updated")

        print("✅ UpdateUserCommand validation passed")
        return FlextResult.ok(None)


class DeleteUserCommand(FlextCommands.Command):
    """Command to delete a user."""

    target_user_id: str
    reason: str

    def validate_command(self) -> FlextResult[None]:
        """Validate delete user command."""
        print(f"🔍 Validating DeleteUserCommand: {self.target_user_id}")

        if not self.target_user_id:
            return FlextResult.fail("Target user ID is required")

        if not self.reason or len(self.reason.strip()) < 10:
            return FlextResult.fail("Deletion reason must be at least 10 characters")

        print("✅ DeleteUserCommand validation passed")
        return FlextResult.ok(None)


# =============================================================================
# QUERIES - Read operations with data retrieval
# =============================================================================


class GetUserQuery(FlextCommands.Query):
    """Query to get a specific user."""

    target_user_id: str


class ListUsersQuery(FlextCommands.Query):
    """Query to list users with filtering."""

    active_only: bool = True
    min_age: int | None = None
    max_age: int | None = None


class GetUserEventsQuery(FlextCommands.Query):
    """Query to get events for a user."""

    correlation_id: str


# =============================================================================
# COMMAND HANDLERS - Business logic execution
# =============================================================================


class CreateUserCommandHandler(
    FlextCommands.Handler[CreateUserCommand, dict[str, Any]],
):
    """Handler for user creation commands."""

    def __init__(self) -> None:
        super().__init__()
        self.users_db: dict[str, dict[str, Any]] = {}

    def handle(self, command: CreateUserCommand) -> FlextResult[dict[str, Any]]:
        """Handle user creation command."""
        print(f"🏗️ Handling CreateUserCommand: {command.name}")

        # Generate user ID
        user_id = FlextUtilities.generate_entity_id()

        # Create user data
        user_data = {
            "user_id": user_id,
            "name": command.name,
            "email": command.email,
            "age": command.age,
            "active": True,
            "created_at": FlextUtilities.generate_iso_timestamp(),
            "updated_at": FlextUtilities.generate_iso_timestamp(),
        }

        # Simulate business logic
        if command.email.endswith("@spam.com"):
            return FlextResult.fail("Email domain is blacklisted")

        # Store in "database"
        self.users_db[user_id] = user_data

        # Create domain event
        event = DomainEvent(
            "UserCreated",
            {
                "user_id": user_id,
                "name": command.name,
                "email": command.email,
                "age": command.age,
                "command_id": command.command_id,
            },
        )

        # Store event
        event_result = event_store.append_event(event)
        if event_result.is_failure:
            return FlextResult.fail(f"Failed to store event: {event_result.error}")

        print(f"✅ User created successfully: {user_id}")

        # Return created user data
        result_data = {
            "user": user_data,
            "event_id": event.event_id,
            "command_id": command.command_id,
        }

        return FlextResult.ok(result_data)


class UpdateUserCommandHandler(
    FlextCommands.Handler[UpdateUserCommand, dict[str, Any]],
):
    """Handler for user update commands."""

    def __init__(self, users_db: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.users_db = users_db

    def handle(self, command: UpdateUserCommand) -> FlextResult[dict[str, Any]]:
        """Handle user update command."""
        print(f"✏️ Handling UpdateUserCommand: {command.target_user_id}")

        # Check if user exists
        if command.target_user_id not in self.users_db:
            return FlextResult.fail(f"User not found: {command.target_user_id}")

        user_data = self.users_db[command.target_user_id].copy()
        original_data = user_data.copy()

        # Update fields
        changes = {}
        if command.name:
            user_data["name"] = command.name
            changes["name"] = {"old": original_data["name"], "new": command.name}

        if command.email:
            user_data["email"] = command.email
            changes["email"] = {"old": original_data["email"], "new": command.email}

        user_data["updated_at"] = FlextUtilities.generate_iso_timestamp()

        # Store updated data
        self.users_db[command.target_user_id] = user_data

        # Create domain event
        event = DomainEvent(
            "UserUpdated",
            {
                "user_id": command.target_user_id,
                "changes": changes,
                "command_id": command.command_id,
            },
        )

        event_store.append_event(event)

        print(f"✅ User updated successfully: {command.target_user_id}")

        result_data = {
            "user": user_data,
            "changes": changes,
            "event_id": event.event_id,
            "command_id": command.command_id,
        }

        return FlextResult.ok(result_data)


class DeleteUserCommandHandler(
    FlextCommands.Handler[DeleteUserCommand, dict[str, Any]],
):
    """Handler for user deletion commands."""

    def __init__(self, users_db: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.users_db = users_db

    def handle(self, command: DeleteUserCommand) -> FlextResult[dict[str, Any]]:
        """Handle user deletion command."""
        print(f"🗑️ Handling DeleteUserCommand: {command.target_user_id}")

        # Check if user exists
        if command.target_user_id not in self.users_db:
            return FlextResult.fail(f"User not found: {command.target_user_id}")

        user_data = self.users_db[command.target_user_id].copy()

        # Soft delete (mark as inactive)
        user_data["active"] = False
        user_data["deleted_at"] = FlextUtilities.generate_iso_timestamp()
        user_data["deletion_reason"] = command.reason

        self.users_db[command.target_user_id] = user_data

        # Create domain event
        event = DomainEvent(
            "UserDeleted",
            {
                "user_id": command.target_user_id,
                "reason": command.reason,
                "command_id": command.command_id,
            },
        )

        event_store.append_event(event)

        print(f"✅ User deleted successfully: {command.target_user_id}")

        result_data = {
            "user": user_data,
            "deletion_reason": command.reason,
            "event_id": event.event_id,
            "command_id": command.command_id,
        }

        return FlextResult.ok(result_data)


# =============================================================================
# QUERY HANDLERS - Data retrieval logic
# =============================================================================


class GetUserQueryHandler(FlextCommands.QueryHandler[GetUserQuery, dict[str, Any]]):
    """Handler for get user queries."""

    def __init__(self, users_db: dict[str, dict[str, Any]]) -> None:
        self.users_db = users_db

    def handle(self, query: GetUserQuery) -> FlextResult[dict[str, Any]]:
        """Handle get user query."""
        print(f"🔍 Handling GetUserQuery: {query.target_user_id}")

        if query.target_user_id not in self.users_db:
            return FlextResult.fail(f"User not found: {query.target_user_id}")

        user_data = self.users_db[query.target_user_id]

        print(f"✅ User retrieved: {user_data['name']}")
        return FlextResult.ok(user_data)


class ListUsersQueryHandler(
    FlextCommands.QueryHandler[ListUsersQuery, list[dict[str, Any]]],
):
    """Handler for list users queries."""

    def __init__(self, users_db: dict[str, dict[str, Any]]) -> None:
        self.users_db = users_db

    def handle(self, query: ListUsersQuery) -> FlextResult[list[dict[str, Any]]]:
        """Handle list users query."""
        print(f"📋 Handling ListUsersQuery (active_only: {query.active_only})")

        users = []
        for user_data in self.users_db.values():
            # Filter by active status
            if query.active_only and not user_data.get("active", True):
                continue

            # Filter by age range
            if query.min_age and user_data["age"] < query.min_age:
                continue

            if query.max_age and user_data["age"] > query.max_age:
                continue

            users.append(user_data)

        print(f"✅ Found {len(users)} users matching criteria")
        return FlextResult.ok(users)


class GetUserEventsQueryHandler(
    FlextCommands.QueryHandler[GetUserEventsQuery, list[dict[str, Any]]],
):
    """Handler for user events queries."""

    def handle(self, query: GetUserEventsQuery) -> FlextResult[list[dict[str, Any]]]:
        """Handle get user events query."""
        print(f"📜 Handling GetUserEventsQuery: {query.correlation_id}")

        events = event_store.get_events_by_correlation(query.correlation_id)
        event_data = [event.to_dict() for event in events]

        print(f"✅ Found {len(event_data)} events for correlation ID")
        return FlextResult.ok(event_data)


# =============================================================================
# COMMAND BUS SETUP - Enterprise command routing
# =============================================================================


def setup_command_bus() -> FlextResult[FlextCommands.Bus]:
    """Setups command bus with all handlers."""
    print("\n🚌 Setting up command bus...")

    # Create command bus
    bus = FlextCommands.create_command_bus()

    # Shared database for handlers
    users_db: dict[str, dict[str, Any]] = {}

    # Create command handlers
    create_handler = CreateUserCommandHandler()
    update_handler = UpdateUserCommandHandler(users_db)
    delete_handler = DeleteUserCommandHandler(users_db)

    # Share database reference with create handler
    create_handler.users_db = users_db

    # Register command handlers
    handlers = [
        (CreateUserCommand, create_handler),
        (UpdateUserCommand, update_handler),
        (DeleteUserCommand, delete_handler),
    ]

    for command_type, handler in handlers:
        result = bus.register_handler(command_type, handler)
        if result.is_failure:
            return FlextResult.fail(f"Failed to register handler: {result.error}")

    print(f"✅ Command bus setup completed with {len(handlers)} handlers")
    return FlextResult.ok(bus)


def setup_query_handlers(users_db: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Setups query handlers."""
    print("\n🔍 Setting up query handlers...")

    handlers = {
        "get_user": GetUserQueryHandler(users_db),
        "list_users": ListUsersQueryHandler(users_db),
        "get_user_events": GetUserEventsQueryHandler(),
    }

    print(f"✅ Query handlers setup completed with {len(handlers)} handlers")
    return handlers


# =============================================================================
# APPLICATION SERVICE - High-level orchestration
# =============================================================================


class UserManagementApplicationService:
    """Application service orchestrating commands and queries."""

    def __init__(
        self,
        command_bus: FlextCommands.Bus,
        query_handlers: dict[str, Any],
    ) -> None:
        self.command_bus = command_bus
        self.query_handlers = query_handlers
        self.service_id = FlextUtilities.generate_entity_id()

        print(f"🏢 User management application service created: {self.service_id}")

    def create_user(
        self,
        name: str,
        email: str,
        age: int,
    ) -> FlextResult[dict[str, Any]]:
        """Create user through command bus."""
        print(f"\n🏢 Creating user: {name}")

        # Create command
        command = CreateUserCommand(
            name=name,
            email=email,
            age=age,
            user_id="system",
        )

        # Execute through command bus
        result = self.command_bus.execute(command)
        if result.is_failure:
            return FlextResult.fail(f"User creation failed: {result.error}")

        return FlextResult.ok(result.data)

    def update_user(
        self,
        user_id: str,
        name: str | None = None,
        email: str | None = None,
    ) -> FlextResult[dict[str, Any]]:
        """Update user through command bus."""
        print(f"\n🏢 Updating user: {user_id}")

        command = UpdateUserCommand(
            target_user_id=user_id,
            name=name,
            email=email,
            user_id="system",
        )

        result = self.command_bus.execute(command)
        if result.is_failure:
            return FlextResult.fail(f"User update failed: {result.error}")

        return FlextResult.ok(result.data)

    def delete_user(self, user_id: str, reason: str) -> FlextResult[dict[str, Any]]:
        """Delete user through command bus."""
        print(f"\n🏢 Deleting user: {user_id}")

        command = DeleteUserCommand(
            target_user_id=user_id,
            reason=reason,
            user_id="system",
        )

        result = self.command_bus.execute(command)
        if result.is_failure:
            return FlextResult.fail(f"User deletion failed: {result.error}")

        return FlextResult.ok(result.data)

    def get_user(self, user_id: str) -> FlextResult[dict[str, Any]]:
        """Get user through query handler."""
        print(f"\n🏢 Getting user: {user_id}")

        query = GetUserQuery(target_user_id=user_id)
        handler = self.query_handlers["get_user"]

        result = handler.handle(query)
        if result.is_failure:
            return FlextResult.fail(f"User retrieval failed: {result.error}")

        return FlextResult.ok(result.data)

    def list_users(
        self,
        min_age: int | None = None,
        max_age: int | None = None,
        *,
        active_only: bool = True,
    ) -> FlextResult[list[dict[str, Any]]]:
        """List users through query handler."""
        print(f"\n🏢 Listing users (active_only: {active_only})")

        query = ListUsersQuery(
            active_only=active_only,
            min_age=min_age,
            max_age=max_age,
        )
        handler = self.query_handlers["list_users"]

        result = handler.handle(query)
        if result.is_failure:
            return FlextResult.fail(f"User listing failed: {result.error}")

        return FlextResult.ok(result.data)

    def get_user_events(self, correlation_id: str) -> FlextResult[list[dict[str, Any]]]:
        """Get user events through query handler."""
        print(f"\n🏢 Getting user events: {correlation_id}")

        query = GetUserEventsQuery(correlation_id=correlation_id)
        handler = self.query_handlers["get_user_events"]

        result = handler.handle(query)
        if result.is_failure:
            return FlextResult.fail(f"Events retrieval failed: {result.error}")

        return FlextResult.ok(result.data)


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def main() -> None:
    """Run comprehensive FlextCommands CQRS demonstration."""
    print("=" * 80)
    print("🚌 FLEXT COMMANDS - CQRS PATTERN DEMONSTRATION")
    print("=" * 80)

    # Setup command bus and query handlers
    bus_result = setup_command_bus()
    if bus_result.is_failure:
        print(f"❌ Failed to setup command bus: {bus_result.error}")
        return

    command_bus = bus_result.unwrap()

    # Get users database from create handler for query handlers
    create_handler = None
    for handler in command_bus._handlers.values():
        if isinstance(handler, CreateUserCommandHandler):
            create_handler = handler
            break

    if not create_handler:
        print("❌ Failed to get create handler reference")
        return

    query_handlers = setup_query_handlers(create_handler.users_db)

    # Create application service
    app_service = UserManagementApplicationService(command_bus, query_handlers)

    # Example 1: Create users using commands
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 1: User Creation via Commands")
    print("=" * 60)

    users_to_create = [
        ("Alice Johnson", "alice@example.com", 28),
        ("Bob Smith", "bob@example.com", 35),
        ("Carol Davis", "carol@example.com", 42),
    ]

    created_users = []
    for name, email, age in users_to_create:
        result = app_service.create_user(name, email, age)
        if result.is_success:
            user_data = result.data
            created_users.append(user_data["user"])
            print(f"✅ Created: {name} (ID: {user_data['user']['user_id']})")
        else:
            print(f"❌ Failed to create {name}: {result.error}")

    # Example 2: Query users
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 2: User Queries")
    print("=" * 60)

    # List all users
    list_result = app_service.list_users()
    if list_result.is_success:
        users = list_result.data
        print(f"📋 Total users found: {len(users)}")
        for user in users:
            print(f"  👤 {user['name']} ({user['email']}) - Age: {user['age']}")

    # Get specific user
    if created_users:
        first_user = created_users[0]
        get_result = app_service.get_user(first_user["user_id"])
        if get_result.is_success:
            user = get_result.data
            print(f"\n🔍 Retrieved user: {user['name']} ({user['email']})")

    # Example 3: Update user via command
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 3: User Update via Commands")
    print("=" * 60)

    if created_users:
        user_to_update = created_users[0]
        update_result = app_service.update_user(
            user_id=user_to_update["user_id"],
            name="Alice Johnson-Smith",
            email="alice.smith@example.com",
        )

        if update_result.is_success:
            updated_data = update_result.data
            print("✅ User updated successfully")
            print(f"📝 Changes: {updated_data['changes']}")

    # Example 4: Event sourcing - view events
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 4: Event Sourcing - View Events")
    print("=" * 60)

    all_events = event_store.get_all_events()
    print(f"📜 Total events in store: {len(all_events)}")

    for event in all_events:
        print(f"  📝 {event.event_type} - {event.timestamp}")
        print(f"     ID: {event.event_id}")
        print(f"     Data: {event.data}")

    # Example 5: Age-based filtering query
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 5: Age-based Filtering Queries")
    print("=" * 60)

    # Users over 30
    older_users_result = app_service.list_users(active_only=True, min_age=30)
    if older_users_result.is_success:
        older_users = older_users_result.data
        print(f"👥 Users over 30: {len(older_users)}")
        for user in older_users:
            print(f"  👤 {user['name']} - Age: {user['age']}")

    # Example 6: Delete user via command
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 6: User Deletion via Commands")
    print("=" * 60)

    if created_users:
        user_to_delete = created_users[-1]  # Delete last user
        delete_result = app_service.delete_user(
            user_id=user_to_delete["user_id"],
            reason="User requested account deletion for privacy reasons",
        )

        if delete_result.is_success:
            deleted_data = delete_result.data
            print("✅ User deleted successfully")
            print(f"🗑️ Reason: {deleted_data['deletion_reason']}")

    # Example 7: View all events after operations
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 7: Complete Event History")
    print("=" * 60)

    final_events = event_store.get_all_events()
    print(f"📜 Final event count: {len(final_events)}")

    # Group events by type
    event_types = {}
    for event in final_events:
        event_type = event.event_type
        if event_type not in event_types:
            event_types[event_type] = 0
        event_types[event_type] += 1

    print("📊 Event statistics:")
    for event_type, count in event_types.items():
        print(f"  📝 {event_type}: {count} events")

    # Example 8: Command validation failures
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 8: Command Validation Failures")
    print("=" * 60)

    # Try to create invalid user
    invalid_result = app_service.create_user("", "invalid-email", 15)
    if invalid_result.is_failure:
        print(f"❌ Expected validation failure: {invalid_result.error}")

    print("\n" + "=" * 80)
    print("🎉 FLEXT COMMANDS CQRS DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
