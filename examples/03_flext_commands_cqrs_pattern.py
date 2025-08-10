#!/usr/bin/env python3
"""CQRS pattern implementation using FlextCommands.

Demonstrates command and query separation with handlers, routing,
validation, and event sourcing integration.
"""

from __future__ import annotations

from typing import cast

# Import shared domain models to eliminate duplication
from .shared_domain import (
    SharedDomainFactory,
    log_domain_operation,
)

# Import additional flext-core patterns for enhanced functionality
from flext_core import (
    FlextCommands,
    FlextResult,
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
    """Base domain event for event sourcing using flext_core.typings."""

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
    """Simple event store for demonstration using flext_core.typings."""

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
# COMPLEXITY REDUCTION HELPERS - SOLID SRP: Eliminate repetitive patterns
# =============================================================================


class BaseCommandHandler:
    """Base handler with common patterns - reduces complexity."""

    def __init__(self, handler_type: str) -> None:
        """Initialize base handler with type identifier."""
        self.handler_id: TEntityId = FlextUtilities.generate_entity_id()
        self.handler_type = handler_type
        log_message: TLogMessage = f"🔧 {handler_type} initialized: {self.handler_id}"
        print(log_message)

    def create_query_projection(self, shared_user: object) -> TAnyObject:
        """DRY Helper: Create standardized query projection."""
        return {
            "id": getattr(shared_user, "id", None),
            "name": getattr(shared_user, "name", None),
            "email": getattr(getattr(shared_user, "email_address", None), "email", None)
            if hasattr(shared_user, "email_address")
            else None,
            "age": getattr(getattr(shared_user, "age", None), "value", None)
            if hasattr(shared_user, "age")
            else None,
            "status": getattr(getattr(shared_user, "status", None), "value", None)
            if hasattr(shared_user, "status")
            else None,
            "created_at": str(getattr(shared_user, "created_at", None))
            if getattr(shared_user, "created_at", None)
            else None,
            "version": getattr(shared_user, "version", None),
        }

    def store_domain_event(
        self, event_type: str, data: TAnyObject
    ) -> FlextResult[TEntityId]:
        """DRY Helper: Store domain event with error handling."""
        event = DomainEvent(event_type, data)
        return event_store.append_event(event)


class DemonstrationFlowHelper:
    """Helper to reduce repetitive demonstration patterns - SOLID SRP."""

    @staticmethod
    def print_section_header(example_num: int, title: str) -> None:
        """DRY Helper: Print standardized section headers."""
        print("\n" + "=" * 60)
        print(f"📋 EXAMPLE {example_num}: {title}")
        print("=" * 60)

    @staticmethod
    def handle_result_with_state_update(
        result: FlextResult[TAnyObject],
        success_message: str,
        state_dict: dict[TEntityId, TUserData],
        user_id: TEntityId | None = None,
    ) -> FlextResult[TAnyObject]:
        """DRY Helper: Handle result and update state consistently."""
        if result.success:
            print(f"✅ {success_message}")
            if user_id and isinstance(result.data, dict):
                state_dict[user_id] = result.data
            return result
        return FlextResult.fail(
            f"{success_message.split(' ', maxsplit=1)[0]} failed: {result.error}"
        )


# =============================================================================
# COMMANDS - Write operations with business intent
# =============================================================================


class CreateUserCommand(FlextCommands.Command):
    """Command to create a new user using flext_core.typings."""

    name: str
    email: str
    age: int

    def validate_command(self) -> FlextResult[None]:
        """Validate create user command."""
        log_message: TLogMessage = f"🔍 Validating CreateUserCommand: {self.name}"
        print(log_message)

        # Validate name
        if not self.name or len(self.name.strip()) == 0:
            return FlextResult.fail("Name cannot be empty")

        # Validate email format
        if "@" not in self.email:
            return FlextResult.fail(f"Invalid email format: {self.email}")

        # Validate age range
        if self.age < MIN_USER_AGE or self.age > MAX_USER_AGE:
            age_error: TErrorMessage = (
                f"Age must be between {MIN_USER_AGE} and {MAX_USER_AGE}"
            )
            return FlextResult.fail(age_error)

        print(f"✅ CreateUserCommand validation passed: {self.name}")
        return FlextResult.ok(None)


class UpdateUserCommand(FlextCommands.Command):
    """Command to update an existing user using flext_core.typings."""

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
            return FlextResult.fail("Target user ID cannot be empty")

        # Validate at least one field to update
        if not self.name and not self.email:
            return FlextResult.fail(
                "At least one field (name or email) must be provided"
            )

        # Validate name if provided
        if self.name is not None and len(self.name.strip()) == 0:
            return FlextResult.fail("Name cannot be empty if provided")

        # Validate email if provided
        if self.email is not None and "@" not in self.email:
            return FlextResult.fail(f"Invalid email format: {self.email}")

        print(f"✅ UpdateUserCommand validation passed: {self.target_user_id}")
        return FlextResult.ok(None)


class DeleteUserCommand(FlextCommands.Command):
    """Command to delete a user using flext_core.typings."""

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
            return FlextResult.fail("Target user ID cannot be empty")

        # Validate deletion reason
        if not self.reason or len(self.reason.strip()) < MIN_DELETION_REASON_LENGTH:
            return FlextResult.fail(
                f"Deletion reason must be at least"
                f" {MIN_DELETION_REASON_LENGTH} characters"
            )

        print(f"✅ DeleteUserCommand validation passed: {self.target_user_id}")
        return FlextResult.ok(None)


# =============================================================================
# QUERIES - Read operations
# =============================================================================


class GetUserQuery(FlextCommands.Query):
    """Query to get a specific user using flext_core.typings."""

    target_user_id: TEntityId


class ListUsersQuery(FlextCommands.Query):
    """Query to list users with filtering using flext_core.typings."""

    active_only: bool = True
    min_age: int | None = None
    max_age: int | None = None


class GetUserEventsQuery(FlextCommands.Query):
    """Query to get events for a user using flext_core.typings."""

    correlation_id: str


# =============================================================================
# COMMAND HANDLERS - Business logic implementation
# =============================================================================


class CreateUserCommandHandler(
    BaseCommandHandler,
    FlextCommands.Handler[CreateUserCommand, TAnyObject],
):
    """Handler for CreateUserCommand using flext_core.typings."""

    def __init__(self) -> None:
        """Initialize command handler."""
        super().__init__("CreateUserCommandHandler")

    def handle(self, command: CreateUserCommand) -> FlextResult[TAnyObject]:
        """Handle create user command using shared domain models.

        Reduced complexity through domain model reuse.
        """
        log_message: TLogMessage = f"👤 Creating user: {command.name} ({command.email})"
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
        if shared_user is None:
            return FlextResult.fail("User creation returned None data")

        try:
            # Log domain operation using shared user
            log_domain_operation(
                "user_created_via_command",
                "SharedUser",
                shared_user.id,
                handler_id=self.handler_id,
                command_type="CreateUserCommand",
                name=shared_user.name,
                email=shared_user.email_address.email,
            )

            # Use helper to create query projection - DRY principle
            query_projection = self.create_query_projection(shared_user)

            # Use helper to store domain event - DRY principle
            event_result = self.store_domain_event("UserCreated", query_projection)
            if event_result.is_failure:
                return FlextResult.fail(event_result.error or "Event storage failed")

            print(f"✅ User created successfully: {shared_user.id}")
            return FlextResult.ok(query_projection)

        except (TypeError, ValueError) as e:
            return FlextResult.fail(f"Failed to create shared user: {e}")


class UpdateUserCommandHandler(
    BaseCommandHandler,
    FlextCommands.Handler[UpdateUserCommand, TAnyObject],
):
    """Handler for UpdateUserCommand using flext_core.typings."""

    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None:
        """Initialize command handler with user database."""
        super().__init__("UpdateUserCommandHandler")
        self.users_db = users_db

    def handle(self, command: UpdateUserCommand) -> FlextResult[TAnyObject]:
        """Handle update user command - reduced complexity."""
        log_message: TLogMessage = f"🔄 Updating user: {command.target_user_id}"
        print(log_message)

        # Check if user exists
        if command.target_user_id not in self.users_db:
            return FlextResult.fail(f"User not found: {command.target_user_id}")

        user_data = self.users_db[command.target_user_id]

        # Update fields using helper data structure
        update_data: TAnyObject = {
            "updated_at": FlextUtilities.generate_iso_timestamp(),
        }

        # Apply updates if provided
        if command.name is not None:
            user_data["name"] = command.name
            update_data["name"] = command.name
        if command.email is not None:
            user_data["email"] = command.email
            update_data["email"] = command.email

        # Use helper to store domain event - DRY principle
        event_result = self.store_domain_event("UserUpdated", update_data)
        if event_result.is_failure:
            return FlextResult.fail(event_result.error or "Event storage failed")

        print(f"✅ User updated successfully: {command.target_user_id}")
        return FlextResult.ok(user_data)


class DeleteUserCommandHandler(
    BaseCommandHandler,
    FlextCommands.Handler[DeleteUserCommand, TAnyObject],
):
    """Handler for DeleteUserCommand using flext_core.typings."""

    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None:
        """Initialize command handler with user database."""
        super().__init__("DeleteUserCommandHandler")
        self.users_db = users_db

    def handle(self, command: DeleteUserCommand) -> FlextResult[TAnyObject]:
        """Handle delete user command - reduced complexity."""
        log_message: TLogMessage = f"🗑️ Deleting user: {command.target_user_id}"
        print(log_message)

        # Check if user exists
        if command.target_user_id not in self.users_db:
            return FlextResult.fail(f"User not found: {command.target_user_id}")

        user_data = self.users_db[command.target_user_id]

        # Mark as deleted
        deleted_at = FlextUtilities.generate_iso_timestamp()
        user_data.update(
            {
                "status": "deleted",
                "deleted_at": deleted_at,
                "deletion_reason": command.reason,
            }
        )

        # Prepare deletion event data
        deletion_data: TAnyObject = {
            "user_id": command.target_user_id,
            "reason": command.reason,
            "deleted_at": deleted_at,
        }

        # Use helper to store domain event - DRY principle
        event_result = self.store_domain_event("UserDeleted", deletion_data)
        if event_result.is_failure:
            return FlextResult.fail(event_result.error or "Event storage failed")

        print(f"✅ User deleted successfully: {command.target_user_id}")
        return FlextResult.ok(user_data)


# =============================================================================
# QUERY HANDLERS - Read operations
# =============================================================================


class GetUserQueryHandler(FlextCommands.QueryHandler[GetUserQuery, TAnyObject]):
    """Handler for GetUserQuery using flext_core.typings."""

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
    """Handler for ListUsersQuery using flext_core.typings."""

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
            user_age = int(cast("int", user_data.get("age", 0)))
            if query.min_age is not None and user_age < query.min_age:
                continue
            if query.max_age is not None and user_age > query.max_age:
                continue

            users.append(user_data)

        print(f"✅ Found {len(users)} users matching criteria")
        return FlextResult.ok(users)


class GetUserEventsQueryHandler(
    FlextCommands.QueryHandler[GetUserEventsQuery, list[TAnyObject]],
):
    """Handler for GetUserEventsQuery using flext_core.typings."""

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
    """Setups command bus with handlers using flext_core.typings."""
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
    """Setups query handlers using flext_core.typings."""
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
    """Application service for user management using flext_core.typings."""

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
        result = self.command_bus.execute(command)
        return result.map(lambda x: cast("TAnyObject", x))

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
        result = self.command_bus.execute(command)
        return result.map(lambda x: cast("TAnyObject", x))

    def delete_user(self, user_id: TEntityId, reason: str) -> FlextResult[TAnyObject]:
        """Delete user using TEntityId and TAnyObject types."""
        log_message: TLogMessage = f"🗑️ Deleting user via application service: {user_id}"
        print(log_message)

        command = DeleteUserCommand(target_user_id=user_id, reason=reason)
        result = self.command_bus.execute(command)
        return result.map(lambda x: cast("TAnyObject", x))

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


# Extract CQRS demonstration methods to reduce main complexity
class CQRSDemonstrator:
    """Handles CQRS pattern demonstration following SOLID principles."""

    def __init__(self) -> None:
        self.users_db: dict[TEntityId, TUserData] = {}
        self.app_service: UserManagementApplicationService | None = None
        self.created_user_id: TEntityId | None = None

    def setup_cqrs_infrastructure(self) -> FlextResult[None]:
        """Setup command bus, query handlers, and application service."""
        # Use helper for standardized section header
        DemonstrationFlowHelper.print_section_header(1, "Command Bus Setup")

        # Setup command bus
        command_bus_result = setup_command_bus()
        if command_bus_result.is_failure:
            return FlextResult.fail(
                f"Command bus setup failed: {command_bus_result.error}"
            )

        # Setup query handlers
        query_handlers = setup_query_handlers(self.users_db)

        # Create application service
        command_bus = command_bus_result.data
        if command_bus is None:
            return FlextResult.fail("Command bus setup returned None")

        self.app_service = UserManagementApplicationService(command_bus, query_handlers)
        return FlextResult.ok(None)

    def demonstrate_user_creation(self) -> FlextResult[TEntityId]:
        """Demonstrate user creation command - reduced complexity."""
        if not self.app_service:
            return FlextResult.fail("Application service not initialized")

        # Use helper for standardized section header
        DemonstrationFlowHelper.print_section_header(2, "User Creation Commands")

        create_result = self.app_service.create_user(
            "Alice Johnson", "alice@example.com", 28
        )

        if create_result.success:
            user_data = create_result.data
            if isinstance(user_data, dict) and "id" in user_data:
                user_id = str(user_data["id"])
                # Use helper for consistent result handling with state update
                DemonstrationFlowHelper.handle_result_with_state_update(
                    create_result,
                    f"User created successfully: {user_id}",
                    self.users_db,
                    user_id,
                )
                self.created_user_id = user_id
                return FlextResult.ok(user_id)
            print("✅ User created successfully")
            return FlextResult.ok("unknown_id")
        return FlextResult.fail(f"User creation failed: {create_result.error}")

    def demonstrate_user_update(self) -> FlextResult[None]:
        """Demonstrate user update command."""
        if not self.app_service or not self.created_user_id:
            return FlextResult.fail("User not created or service not initialized")

        print("\n" + "=" * 60)
        print("📋 EXAMPLE 3: User Update Commands")
        print("=" * 60)

        update_result = self.app_service.update_user(
            self.created_user_id, name="Alice Smith"
        )
        if update_result.success:
            print(f"✅ User updated successfully: {self.created_user_id}")
            # Update database
            if isinstance(update_result.data, dict):
                self.users_db[self.created_user_id] = update_result.data
            return FlextResult.ok(None)
        return FlextResult.fail(f"User update failed: {update_result.error}")

    def demonstrate_user_queries(self) -> FlextResult[None]:
        """Demonstrate user query operations."""
        if not self.app_service:
            return FlextResult.fail("Application service not initialized")

        print("\n" + "=" * 60)
        print("📋 EXAMPLE 4: User Queries")
        print("=" * 60)

        list_result = self.app_service.list_users(active_only=True)
        if list_result.success:
            users = list_result.data
            if isinstance(users, list):
                print(f"✅ Found {len(users)} active users")
                for user in users:
                    if isinstance(user, dict) and "name" in user:
                        print(f"   - {user['name']} ({user.get('email', 'N/A')})")
            else:
                print("✅ Users listed successfully")
            return FlextResult.ok(None)
        return FlextResult.fail(f"User listing failed: {list_result.error}")

    def demonstrate_event_sourcing(self) -> FlextResult[None]:
        """Demonstrate event sourcing functionality."""
        print("\n" + "=" * 60)
        print("📋 EXAMPLE 5: Event Sourcing")
        print("=" * 60)

        all_events = event_store.get_all_events()
        print(f"📝 Total events in store: {len(all_events)}")
        for event in all_events:
            print(f"   - {event.event_type}: {event.event_id}")
        return FlextResult.ok(None)

    def demonstrate_user_deletion(self) -> FlextResult[None]:
        """Demonstrate user deletion command."""
        if not self.app_service or not self.created_user_id:
            return FlextResult.fail("User not created or service not initialized")

        print("\n" + "=" * 60)
        print("📋 EXAMPLE 6: User Deletion Commands")
        print("=" * 60)

        delete_result = self.app_service.delete_user(
            self.created_user_id,
            "User requested account deletion",
        )
        if delete_result.success:
            print(f"✅ User deleted successfully: {self.created_user_id}")
            # Update database
            if isinstance(delete_result.data, dict):
                self.users_db[self.created_user_id] = delete_result.data
            return FlextResult.ok(None)
        return FlextResult.fail(f"User deletion failed: {delete_result.error}")

    def demonstrate_validation_failure(self) -> FlextResult[None]:
        """Demonstrate command validation with invalid data."""
        if not self.app_service:
            return FlextResult.fail("Application service not initialized")

        print("\n" + "=" * 60)
        print("📋 EXAMPLE 7: Command Validation")
        print("=" * 60)

        # Try to create user with invalid data
        invalid_result = self.app_service.create_user("", "invalid-email", 15)
        if invalid_result.is_failure:
            print(f"❌ Expected validation failure: {invalid_result.error}")
            return FlextResult.ok(None)
        print("⚠️  Unexpected success for invalid data")
        return FlextResult.fail("Validation should have failed")


def main() -> None:
    """Run comprehensive FlextCommands demonstration using SOLID principles."""
    print("=" * 80)
    print("🚀 FLEXT COMMANDS - CQRS PATTERN DEMONSTRATION")
    print("=" * 80)

    demonstrator = CQRSDemonstrator()

    # Run all demonstration steps
    steps = [
        demonstrator.setup_cqrs_infrastructure,
        demonstrator.demonstrate_user_creation,
        demonstrator.demonstrate_user_update,
        demonstrator.demonstrate_user_queries,
        demonstrator.demonstrate_event_sourcing,
        demonstrator.demonstrate_user_deletion,
        demonstrator.demonstrate_validation_failure,
    ]

    for step in steps:
        result = step()
        if hasattr(result, "is_failure") and getattr(result, "is_failure", False):
            print(
                f"❌ Demonstration step failed: "
                f"{getattr(result, 'error', 'Unknown error')}"
            )
            return

    print("\n" + "=" * 80)
    print("🎉 FLEXT COMMANDS DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
