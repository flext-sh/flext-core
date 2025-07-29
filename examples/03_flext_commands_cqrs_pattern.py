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
- Maximum type safety using flext_core.types
"""

from __future__ import annotations

from typing import cast

# Import shared domain models to eliminate duplication
from shared_domain import (
    SharedDomainFactory,
    log_domain_operation,
)

# Import additional flext-core patterns for enhanced functionality
from flext_core import (
    FlextCommands,
    FlextResult,
    FlextTypes,
    FlextUtilities,
    TAnyObject,
    TEntityId,
    TErrorMessage,
    TLogMessage,
    TUserData,
)

# =============================================================================
# VALIDATION CONSTANTS - Business rule constraints
# =============================================================================

# Age validation constants
MIN_USER_AGE = 18  # Minimum legal age for user registration
MAX_USER_AGE = 120  # Maximum reasonable age for validation

# Deletion reason validation constants
MIN_DELETION_REASON_LENGTH = 10  # Minimum characters for deletion justification

# =============================================================================
# NO LOCAL DOMAIN MODELS - Use ONLY shared_domain.py models
# =============================================================================

# All domain functionality comes from shared_domain.py
# This eliminates ALL code duplication and uses standard SharedUser and SharedProduct


# =============================================================================
# DOMAIN EVENTS - Event sourcing support
# =============================================================================


class DomainEvent:
    """Base domain event for event sourcing using flext_core.types."""

    def __init__(self, event_type: str, data: TAnyObject) -> None:
        """Initialize domain event with type and data using TAnyObject."""
        self.event_id: TEntityId = FlextUtilities.generate_entity_id()
        self.event_type = event_type
        self.data = data
        self.timestamp = FlextUtilities.generate_iso_timestamp()
        self.correlation_id = FlextUtilities.generate_correlation_id()

    def to_dict(self) -> TAnyObject:
        """Convert event to dictionary using TAnyObject."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
        }


class EventStore:
    """Simple event store for demonstration using flext_core.types."""

    def __init__(self) -> None:
        """Initialize empty event store."""
        self.events: list[DomainEvent] = []

    def append_event(self, event: DomainEvent) -> FlextResult[TEntityId]:
        """Append event to store using TEntityId return type."""
        self.events.append(event)
        log_message: TLogMessage = (
            f"📝 Event stored: {event.event_type} ({event.event_id})"
        )
        print(log_message)
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
    """Command to create a new user using flext_core.types."""

    name: str
    email: str
    age: int

    def validate_command(self) -> FlextResult[None]:
        """Validate create user command."""
        log_message: TLogMessage = f"🔍 Validating CreateUserCommand: {self.name}"
        print(log_message)

        # Validate name
        if not self.name or len(self.name.strip()) == 0:
            error_message: TErrorMessage = "Name cannot be empty"
            return FlextResult.fail(error_message)

        # Validate email format
        if "@" not in self.email:
            error_message: TErrorMessage = f"Invalid email format: {self.email}"
            return FlextResult.fail(error_message)

        # Validate age
        if not FlextTypes.TypeGuards.is_instance_of(self.age, int):
            error_message: TErrorMessage = "Age must be an integer"
            return FlextResult.fail(error_message)

        if self.age < MIN_USER_AGE or self.age > MAX_USER_AGE:
            error_message: TErrorMessage = (
                f"Age must be between {MIN_USER_AGE} and {MAX_USER_AGE}"
            )
            return FlextResult.fail(error_message)

        print(f"✅ CreateUserCommand validation passed: {self.name}")
        return FlextResult.ok(None)


class UpdateUserCommand(FlextCommands.Command):
    """Command to update an existing user using flext_core.types."""

    target_user_id: TEntityId
    name: str | None = None
    email: str | None = None

    def validate_command(self) -> FlextResult[None]:
        """Validate update user command."""
        log_message: TLogMessage = (
            f"🔍 Validating UpdateUserCommand: {self.target_user_id}"
        )
        print(log_message)

        # Validate user ID
        if not self.target_user_id:
            error_message: TErrorMessage = "Target user ID cannot be empty"
            return FlextResult.fail(error_message)

        # Validate at least one field to update
        if not self.name and not self.email:
            error_message: TErrorMessage = (
                "At least one field (name or email) must be provided"
            )
            return FlextResult.fail(error_message)

        # Validate name if provided
        if self.name is not None and len(self.name.strip()) == 0:
            error_message: TErrorMessage = "Name cannot be empty if provided"
            return FlextResult.fail(error_message)

        # Validate email if provided
        if self.email is not None and "@" not in self.email:
            error_message: TErrorMessage = f"Invalid email format: {self.email}"
            return FlextResult.fail(error_message)

        print(f"✅ UpdateUserCommand validation passed: {self.target_user_id}")
        return FlextResult.ok(None)


class DeleteUserCommand(FlextCommands.Command):
    """Command to delete a user using flext_core.types."""

    target_user_id: TEntityId
    reason: str

    def validate_command(self) -> FlextResult[None]:
        """Validate delete user command."""
        log_message: TLogMessage = (
            f"🔍 Validating DeleteUserCommand: {self.target_user_id}"
        )
        print(log_message)

        # Validate user ID
        if not self.target_user_id:
            error_message: TErrorMessage = "Target user ID cannot be empty"
            return FlextResult.fail(error_message)

        # Validate deletion reason
        if not self.reason or len(self.reason.strip()) < MIN_DELETION_REASON_LENGTH:
            error_message: TErrorMessage = (
                f"Deletion reason must be at least"
                f" {MIN_DELETION_REASON_LENGTH} characters"
            )
            return FlextResult.fail(error_message)

        print(f"✅ DeleteUserCommand validation passed: {self.target_user_id}")
        return FlextResult.ok(None)


# =============================================================================
# QUERIES - Read operations
# =============================================================================


class GetUserQuery(FlextCommands.Query):
    """Query to get a specific user using flext_core.types."""

    target_user_id: TEntityId


class ListUsersQuery(FlextCommands.Query):
    """Query to list users with filtering using flext_core.types."""

    active_only: bool = True
    min_age: int | None = None
    max_age: int | None = None


class GetUserEventsQuery(FlextCommands.Query):
    """Query to get events for a user using flext_core.types."""

    correlation_id: str


# =============================================================================
# COMMAND HANDLERS - Business logic implementation
# =============================================================================


class CreateUserCommandHandler(
    FlextCommands.Handler[CreateUserCommand, TAnyObject],
):
    """Handler for CreateUserCommand using flext_core.types."""

    def __init__(self) -> None:
        """Initialize command handler."""
        self.handler_id: TEntityId = FlextUtilities.generate_entity_id()
        log_message: TLogMessage = (
            f"🔧 CreateUserCommandHandler initialized: {self.handler_id}"
        )
        print(log_message)

    def handle(self, command: CreateUserCommand) -> FlextResult[TAnyObject]:
        """Handle create user command using shared domain models."""
        log_message: TLogMessage = (
            f"👤 Creating enhanced user: {command.name} ({command.email})"
        )
        print(log_message)

        # Use SharedDomainFactory for robust user creation
        user_result = SharedDomainFactory.create_user(
            name=command.name,
            email=command.email,
            age=command.age,
        )

        if user_result.is_failure:
            return FlextResult.fail(f"User creation failed: {user_result.error}")

        shared_user = user_result.data

        # Create enhanced CQRS demo user
        try:
            user = shared_user

            # Log domain operation using shared user
            log_domain_operation(
                "user_created_via_command",
                "SharedUser",
                user.id,
                handler_id=self.handler_id,
                command_type="CreateUserCommand",
                name=user.name,
                email=user.email_address.email,
            )

            # Create query projection data using shared user
            query_projection: TAnyObject = {
                "id": user.id,
                "name": user.name,
                "email": user.email_address.email,
                "age": user.age.value,
                "status": user.status.value,
                "created_at": str(user.created_at) if user.created_at else None,
                "version": user.version,
            }

            # Store event with shared user data
            event = DomainEvent("UserCreated", query_projection)
            event_result = event_store.append_event(event)
            if event_result.is_failure:
                return FlextResult.fail(event_result.error)

            print(f"✅ Shared user created successfully: {user.id}")
            return FlextResult.ok(query_projection)

        except (TypeError, ValueError) as e:
            return FlextResult.fail(f"Failed to create shared user: {e}")


class UpdateUserCommandHandler(
    FlextCommands.Handler[UpdateUserCommand, TAnyObject],
):
    """Handler for UpdateUserCommand using flext_core.types."""

    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None:
        """Initialize command handler with user database."""
        self.users_db = users_db
        self.handler_id: TEntityId = FlextUtilities.generate_entity_id()
        log_message: TLogMessage = (
            f"🔧 UpdateUserCommandHandler initialized: {self.handler_id}"
        )
        print(log_message)

    def handle(self, command: UpdateUserCommand) -> FlextResult[TAnyObject]:
        """Handle update user command using TAnyObject return type."""
        log_message: TLogMessage = f"🔄 Updating user: {command.target_user_id}"
        print(log_message)

        # Check if user exists
        if command.target_user_id not in self.users_db:
            error_message: TErrorMessage = f"User not found: {command.target_user_id}"
            return FlextResult.fail(error_message)

        user_data = self.users_db[command.target_user_id]

        # Update fields
        update_data: TAnyObject = {
            "updated_at": FlextUtilities.generate_iso_timestamp(),
        }
        if command.name is not None:
            user_data["name"] = command.name
            update_data["name"] = command.name

        if command.email is not None:
            user_data["email"] = command.email
            update_data["email"] = command.email

        # Store event
        event = DomainEvent("UserUpdated", update_data)
        event_result = event_store.append_event(event)
        if event_result.is_failure:
            return FlextResult.fail(event_result.error)

        print(f"✅ User updated successfully: {command.target_user_id}")
        return FlextResult.ok(user_data)


class DeleteUserCommandHandler(
    FlextCommands.Handler[DeleteUserCommand, TAnyObject],
):
    """Handler for DeleteUserCommand using flext_core.types."""

    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None:
        """Initialize command handler with user database."""
        self.users_db = users_db
        self.handler_id: TEntityId = FlextUtilities.generate_entity_id()
        log_message: TLogMessage = (
            f"🔧 DeleteUserCommandHandler initialized: {self.handler_id}"
        )
        print(log_message)

    def handle(self, command: DeleteUserCommand) -> FlextResult[TAnyObject]:
        """Handle delete user command using TAnyObject return type."""
        log_message: TLogMessage = f"🗑️ Deleting user: {command.target_user_id}"
        print(log_message)

        # Check if user exists
        if command.target_user_id not in self.users_db:
            error_message: TErrorMessage = f"User not found: {command.target_user_id}"
            return FlextResult.fail(error_message)

        user_data = self.users_db[command.target_user_id]

        # Mark as deleted
        user_data["status"] = "deleted"
        user_data["deleted_at"] = FlextUtilities.generate_iso_timestamp()
        user_data["deletion_reason"] = command.reason

        # Store event
        deletion_data: TAnyObject = {
            "user_id": command.target_user_id,
            "reason": command.reason,
            "deleted_at": user_data["deleted_at"],
        }
        event = DomainEvent("UserDeleted", deletion_data)
        event_result = event_store.append_event(event)
        if event_result.is_failure:
            return FlextResult.fail(event_result.error)

        print(f"✅ User deleted successfully: {command.target_user_id}")
        return FlextResult.ok(user_data)


# =============================================================================
# QUERY HANDLERS - Read operations
# =============================================================================


class GetUserQueryHandler(FlextCommands.QueryHandler[GetUserQuery, TAnyObject]):
    """Handler for GetUserQuery using flext_core.types."""

    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None:
        """Initialize query handler with user database."""
        self.users_db = users_db

    def handle(self, query: GetUserQuery) -> FlextResult[TAnyObject]:
        """Handle get user query using TAnyObject return type."""
        log_message: TLogMessage = f"🔍 Getting user: {query.target_user_id}"
        print(log_message)

        if query.target_user_id not in self.users_db:
            error_message: TErrorMessage = f"User not found: {query.target_user_id}"
            return FlextResult.fail(error_message)

        user_data = self.users_db[query.target_user_id]
        print(f"✅ User retrieved: {query.target_user_id}")
        return FlextResult.ok(user_data)


class ListUsersQueryHandler(
    FlextCommands.QueryHandler[ListUsersQuery, list[TAnyObject]],
):
    """Handler for ListUsersQuery using flext_core.types."""

    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None:
        """Initialize query handler with user database."""
        self.users_db = users_db

    def handle(self, query: ListUsersQuery) -> FlextResult[list[TAnyObject]]:
        """Handle list users query using list[TAnyObject] return type."""
        log_message: TLogMessage = f"📋 Listing users (active_only={query.active_only})"
        print(log_message)

        users: list[TAnyObject] = []
        for user_data in self.users_db.values():
            # Apply active filter
            if query.active_only and user_data.get("status") != "active":
                continue

            # Apply age filters
            if query.min_age is not None and user_data.get("age", 0) < query.min_age:
                continue
            if query.max_age is not None and user_data.get("age", 0) > query.max_age:
                continue

            users.append(user_data)

        print(f"✅ Found {len(users)} users matching criteria")
        return FlextResult.ok(users)


class GetUserEventsQueryHandler(
    FlextCommands.QueryHandler[GetUserEventsQuery, list[TAnyObject]],
):
    """Handler for GetUserEventsQuery using flext_core.types."""

    def handle(self, query: GetUserEventsQuery) -> FlextResult[list[TAnyObject]]:
        """Handle get user events query using list[TAnyObject] return type."""
        log_message: TLogMessage = (
            f"📝 Getting events for correlation: {query.correlation_id}"
        )
        print(log_message)

        events = event_store.get_events_by_correlation(query.correlation_id)
        event_data: list[TAnyObject] = [event.to_dict() for event in events]

        print(
            f"✅ Found {len(event_data)} events for correlation {query.correlation_id}",
        )
        return FlextResult.ok(event_data)


# =============================================================================
# COMMAND BUS SETUP - Routing and execution
# =============================================================================


def setup_command_bus() -> FlextResult[FlextCommands.Bus]:
    """Setups command bus with handlers using flext_core.types."""
    log_message: TLogMessage = "🚌 Setting up command bus..."
    print(log_message)

    # Initialize user database
    users_db: dict[TEntityId, TUserData] = {}

    # Create command bus
    command_bus = FlextCommands.Bus()

    # Register command handlers
    create_handler = CreateUserCommandHandler()
    update_handler = UpdateUserCommandHandler(users_db)
    delete_handler = DeleteUserCommandHandler(users_db)

    command_bus.register_handler(CreateUserCommand, create_handler)
    command_bus.register_handler(UpdateUserCommand, update_handler)
    command_bus.register_handler(DeleteUserCommand, delete_handler)

    print("✅ Command bus setup completed")
    return FlextResult.ok(command_bus)


def setup_query_handlers(users_db: dict[TEntityId, TUserData]) -> dict[str, object]:
    """Setups query handlers using flext_core.types."""
    log_message: TLogMessage = "🔍 Setting up query handlers..."
    print(log_message)

    query_handlers: dict[str, object] = {
        "get_user": GetUserQueryHandler(users_db),
        "list_users": ListUsersQueryHandler(users_db),
        "get_user_events": GetUserEventsQueryHandler(),
    }

    print("✅ Query handlers setup completed")
    return query_handlers


# =============================================================================
# APPLICATION SERVICE - High-level orchestration
# =============================================================================


class UserManagementApplicationService:
    """Application service for user management using flext_core.types."""

    def __init__(
        self,
        command_bus: FlextCommands.Bus,
        query_handlers: dict[str, object],
    ) -> None:
        """Initialize application service."""
        self.command_bus = command_bus
        self.query_handlers = query_handlers
        self.service_id: TEntityId = FlextUtilities.generate_entity_id()
        log_message: TLogMessage = (
            f"👥 UserManagementApplicationService initialized: {self.service_id}"
        )
        print(log_message)

    def create_user(
        self,
        name: str,
        email: str,
        age: int,
    ) -> FlextResult[TAnyObject]:
        """Create user using TAnyObject return type."""
        log_message: TLogMessage = f"👤 Creating user via application service: {name}"
        print(log_message)

        command = CreateUserCommand(name=name, email=email, age=age)
        return self.command_bus.execute(command)

    def update_user(
        self,
        user_id: TEntityId,
        name: str | None = None,
        email: str | None = None,
    ) -> FlextResult[TAnyObject]:
        """Update user using TEntityId and TAnyObject types."""
        log_message: TLogMessage = (
            f"🔄 Updating user via application service: {user_id}"
        )
        print(log_message)

        command = UpdateUserCommand(target_user_id=user_id, name=name, email=email)
        return self.command_bus.execute(command)

    def delete_user(self, user_id: TEntityId, reason: str) -> FlextResult[TAnyObject]:
        """Delete user using TEntityId and TAnyObject types."""
        log_message: TLogMessage = f"🗑️ Deleting user via application service: {user_id}"
        print(log_message)

        command = DeleteUserCommand(target_user_id=user_id, reason=reason)
        return self.command_bus.execute(command)

    def get_user(self, user_id: TEntityId) -> FlextResult[TAnyObject]:
        """Get user using TEntityId and TAnyObject types."""
        log_message: TLogMessage = f"🔍 Getting user via application service: {user_id}"
        print(log_message)

        query = GetUserQuery(target_user_id=user_id)
        handler = cast("GetUserQueryHandler", self.query_handlers["get_user"])
        return handler.handle(query)

    def list_users(
        self,
        min_age: int | None = None,
        max_age: int | None = None,
        *,
        active_only: bool = True,
    ) -> FlextResult[list[TAnyObject]]:
        """List users using list[TAnyObject] return type."""
        log_message: TLogMessage = (
            f"📋 Listing users via application service (active_only={active_only})"
        )
        print(log_message)

        query = ListUsersQuery(
            active_only=active_only,
            min_age=min_age,
            max_age=max_age,
        )
        handler = cast("ListUsersQueryHandler", self.query_handlers["list_users"])
        return handler.handle(query)

    def get_user_events(
        self,
        correlation_id: str,
    ) -> FlextResult[list[TAnyObject]]:
        """Get user events using list[TAnyObject] return type."""
        log_message: TLogMessage = (
            f"📝 Getting user events via application service: {correlation_id}"
        )
        print(log_message)

        query = GetUserEventsQuery(correlation_id=correlation_id)
        handler = cast(
            "GetUserEventsQueryHandler",
            self.query_handlers["get_user_events"],
        )
        return handler.handle(query)


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def main() -> None:  # noqa: PLR0912, PLR0915
    """Run comprehensive FlextCommands demonstration with maximum type safety."""
    print("=" * 80)
    print("🚀 FLEXT COMMANDS - CQRS PATTERN DEMONSTRATION")
    print("=" * 80)

    # Setup command bus and query handlers
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 1: Command Bus Setup")
    print("=" * 60)

    command_bus_result = setup_command_bus()
    if command_bus_result.is_failure:
        print(f"❌ Command bus setup failed: {command_bus_result.error}")
        return

    command_bus = command_bus_result.data

    # Initialize user database for query handlers
    users_db: dict[TEntityId, TUserData] = {}
    query_handlers = setup_query_handlers(users_db)

    # Create application service
    app_service = UserManagementApplicationService(command_bus, query_handlers)

    # Test user creation
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 2: User Creation Commands")
    print("=" * 60)

    create_result = app_service.create_user("Alice Johnson", "alice@example.com", 28)
    if create_result.is_success:
        user_data = create_result.data
        if isinstance(user_data, dict) and "id" in user_data:
            user_id = user_data["id"]
            print(f"✅ User created successfully: {user_id}")
            # Store in database for queries
            users_db[user_id] = user_data
        else:
            print("✅ User created successfully")
    else:
        print(f"❌ User creation failed: {create_result.error}")

    # Test user update
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 3: User Update Commands")
    print("=" * 60)

    if (
        create_result.is_success
        and isinstance(create_result.data, dict)
        and "id" in create_result.data
    ):
        user_id = create_result.data["id"]
        update_result = app_service.update_user(user_id, name="Alice Smith")
        if update_result.is_success:
            print(f"✅ User updated successfully: {user_id}")
            # Update database
            if isinstance(update_result.data, dict):
                users_db[user_id] = update_result.data
        else:
            print(f"❌ User update failed: {update_result.error}")

    # Test user queries
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 4: User Queries")
    print("=" * 60)

    list_result = app_service.list_users(active_only=True)
    if list_result.is_success:
        users = list_result.data
        if isinstance(users, list):
            print(f"✅ Found {len(users)} active users")
            for user in users:
                if isinstance(user, dict) and "name" in user:
                    print(f"   - {user['name']} ({user.get('email', 'N/A')})")
        else:
            print("✅ Users listed successfully")
    else:
        print(f"❌ User listing failed: {list_result.error}")

    # Test event sourcing
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 5: Event Sourcing")
    print("=" * 60)

    all_events = event_store.get_all_events()
    print(f"📝 Total events in store: {len(all_events)}")
    for event in all_events:
        print(f"   - {event.event_type}: {event.event_id}")

    # Test user deletion
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 6: User Deletion Commands")
    print("=" * 60)

    if (
        create_result.is_success
        and isinstance(create_result.data, dict)
        and "id" in create_result.data
    ):
        user_id = create_result.data["id"]
        delete_result = app_service.delete_user(
            user_id,
            "User requested account deletion",
        )
        if delete_result.is_success:
            print(f"✅ User deleted successfully: {user_id}")
            # Update database
            if isinstance(delete_result.data, dict):
                users_db[user_id] = delete_result.data
        else:
            print(f"❌ User deletion failed: {delete_result.error}")

    # Test validation failure
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 7: Command Validation")
    print("=" * 60)

    # Try to create user with invalid data
    invalid_result = app_service.create_user("", "invalid-email", 15)
    if invalid_result.is_failure:
        print(f"❌ Expected validation failure: {invalid_result.error}")
    else:
        print("⚠️  Unexpected success for invalid data")

    print("\n" + "=" * 80)
    print("🎉 FLEXT COMMANDS DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
