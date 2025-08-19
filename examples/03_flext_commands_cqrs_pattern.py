#!/usr/bin/env python3
"""CQRS pattern implementation using FlextCommands.

Demonstrates command and query separation with handlers, routing,
validation, and event sourcing integration.
"""

from __future__ import annotations

import contextlib
from typing import Protocol, cast

from examples.shared_domain import (
    SharedDomainFactory,
    log_domain_operation,
)
from flext_core import (
    FlextCommands,
    FlextResult,
    FlextUtilities,
    TAnyObject,
    TEntityId,
    TErrorMessage,
    TUserData,
)

_types_ns = {
    "TServiceName": str,
    "TUserId": str,
    "TCorrelationId": str,
    "TEntityId": str,
}
with contextlib.suppress(Exception):  # Defensive: ignore if already built
    FlextCommands.Command.model_rebuild(types_namespace=_types_ns)
    FlextCommands.Query.model_rebuild(types_namespace=_types_ns)

# =============================================================================
# VALIDATION CONSTANTS - Business rule constraints
# =============================================================================

# Age validation constants
MIN_USER_AGE = 18  # Minimum legal age for user registration
MAX_USER_AGE = 120  # Maximum reasonable age for validation

# Deletion reason validation constants
MIN_DELETION_REASON_LENGTH = 10  # Minimum characters for deletion justification

# =============================================================================
# HANDLER PROTOCOLS - Type-safe handler interfaces
# =============================================================================


class QueryHandlerProtocol(Protocol):
    """Protocol for query handlers with handle method."""

    def handle(self, query: object) -> FlextResult[object]:
        """Handle a query and return result."""
        ...


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

    def to_dict(self) -> dict[str, object]:
        """Convert event to dictionary."""
        result: dict[str, object] = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
        }
        return result


class EventStore:
    """Simple event store for demonstration using flext_core.typings."""

    def __init__(self) -> None:
        """Initialize empty event store."""
        self.events: list[DomainEvent] = []

    def append_event(self, event: DomainEvent) -> FlextResult[TEntityId]:
        """Append event to store using TEntityId return type."""
        self.events.append(event)
        return FlextResult[None].ok(event.event_id)

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

    def create_query_projection(self, shared_user: object) -> dict[str, object]:
        """DRY Helper: Create standardized query projection."""
        result: dict[str, object] = {
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
        return result

    def store_domain_event(
        self,
        event_type: str,
        data: TAnyObject,
    ) -> FlextResult[TEntityId]:
        """DRY Helper: Store domain event with error handling."""
        event = DomainEvent(event_type, data)
        return event_store.append_event(event)


class DemonstrationFlowHelper:
    """Helper to reduce repetitive demonstration patterns - SOLID SRP."""

    @staticmethod
    def print_section_header(example_num: int, title: str) -> None:
        """DRY Helper: Print standardized section headers."""

    @staticmethod
    def handle_result_with_state_update(
        result: FlextResult[TAnyObject],
        success_message: str,
        state_dict: dict[TEntityId, TUserData],
        user_id: TEntityId | None = None,
    ) -> FlextResult[TAnyObject]:
        """DRY Helper: Handle result and update state consistently."""
        if result.success:
            if user_id and isinstance(result.data, dict):
                state_dict[user_id] = result.data
            return result
        return FlextResult[None].fail(
            f"{success_message.split(' ', maxsplit=1)[0]} failed: {result.error}",
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
        # Validate name
        if not self.name or len(self.name.strip()) == 0:
            return FlextResult[None].fail("Name cannot be empty")

        # Validate email format
        if "@" not in self.email:
            return FlextResult[None].fail(f"Invalid email format: {self.email}")

        # Validate age range
        if self.age < MIN_USER_AGE or self.age > MAX_USER_AGE:
            age_error: TErrorMessage = (
                f"Age must be between {MIN_USER_AGE} and {MAX_USER_AGE}"
            )
            return FlextResult[None].fail(age_error)

        return FlextResult[None].ok(None)


class UpdateUserCommand(FlextCommands.Command):
    """Command to update an existing user using flext_core.typings."""

    target_user_id: TEntityId
    name: str | None = None
    email: str | None = None

    def validate_command(self) -> FlextResult[None]:
        """Validate update user command."""
        # Validate user ID
        if not self.target_user_id:
            return FlextResult[None].fail("Target user ID cannot be empty")

        # Validate at least one field to update
        if not self.name and not self.email:
            return FlextResult[None].fail(
                "At least one field (name or email) must be provided",
            )

        # Validate name if provided
        if self.name is not None and len(self.name.strip()) == 0:
            return FlextResult[None].fail("Name cannot be empty if provided")

        # Validate email if provided
        if self.email is not None and "@" not in self.email:
            return FlextResult[None].fail(f"Invalid email format: {self.email}")

        return FlextResult[None].ok(None)


class DeleteUserCommand(FlextCommands.Command):
    """Command to delete a user using flext_core.typings."""

    target_user_id: TEntityId
    reason: str

    def validate_command(self) -> FlextResult[None]:
        """Validate delete user command."""
        # Validate user ID
        if not self.target_user_id:
            return FlextResult[None].fail("Target user ID cannot be empty")

        # Validate deletion reason
        if not self.reason or len(self.reason.strip()) < MIN_DELETION_REASON_LENGTH:
            return FlextResult[None].fail(
                f"Deletion reason must be at least"
                f" {MIN_DELETION_REASON_LENGTH} characters",
            )

        return FlextResult[None].ok(None)


# Rebuild Pydantic models for local command classes
with contextlib.suppress(Exception):
    CreateUserCommand.model_rebuild(types_namespace=_types_ns)
    UpdateUserCommand.model_rebuild(types_namespace=_types_ns)
    DeleteUserCommand.model_rebuild(types_namespace=_types_ns)


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

    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None:
        """Initialize command handler with shared users database."""
        super().__init__("CreateUserCommandHandler")
        self.users_db = users_db

    def handle(self, command: CreateUserCommand) -> FlextResult[TAnyObject]:
        """Handle create user command using shared domain models.

        Reduced complexity through domain model reuse.
        """
        # Use SharedDomainFactory for robust user creation
        user_result = SharedDomainFactory.create_user(
            name=command.name,
            email=command.email,
            age=command.age,
        )

        if user_result.is_failure:
            return FlextResult[None].fail(f"User creation failed: {user_result.error}")

        shared_user = user_result.data
        if shared_user is None:
            return FlextResult[None].fail("User creation returned None data")

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
            query_projection: dict[str, object] = self.create_query_projection(
                shared_user,
            )

            # Use helper to store domain event - DRY principle
            event_result = self.store_domain_event("UserCreated", query_projection)
            if event_result.is_failure:
                return FlextResult[None].fail(event_result.error or "Event storage failed")

            # Persist projection in shared database for subsequent queries/updates
            user_id_str = str(shared_user.id)
            self.users_db[user_id_str] = cast("TUserData", query_projection)

            return FlextResult[None].ok(cast("TAnyObject", query_projection))

        except (TypeError, ValueError) as e:
            return FlextResult[None].fail(f"Failed to create shared user: {e}")


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
        # Check if user exists
        if command.target_user_id not in self.users_db:
            return FlextResult[None].fail(f"User not found: {command.target_user_id}")

        user_data = self.users_db[command.target_user_id]

        # Update fields using helper data structure
        update_data: dict[str, object] = {
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
            return FlextResult[None].fail(event_result.error or "Event storage failed")

        return FlextResult[None].ok(cast("TAnyObject", user_data))


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
        # Check if user exists
        if command.target_user_id not in self.users_db:
            return FlextResult[None].fail(f"User not found: {command.target_user_id}")

        user_data = self.users_db[command.target_user_id]

        # Mark as deleted
        deleted_at = FlextUtilities.generate_iso_timestamp()
        user_data.update(
            {
                "status": "deleted",
                "deleted_at": deleted_at,
                "deletion_reason": command.reason,
            },
        )

        # Prepare deletion event data
        deletion_data: dict[str, object] = {
            "user_id": command.target_user_id,
            "reason": command.reason,
            "deleted_at": deleted_at,
        }

        # Use helper to store domain event - DRY principle
        event_result = self.store_domain_event("UserDeleted", deletion_data)
        if event_result.is_failure:
            return FlextResult[None].fail(event_result.error or "Event storage failed")

        return FlextResult[None].ok(cast("TAnyObject", user_data))


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
        if query.target_user_id not in self.users_db:
            error_message: TErrorMessage = f"User not found: {query.target_user_id}"
            return FlextResult[None].fail(error_message)

        user_data = self.users_db[query.target_user_id]
        return FlextResult[None].ok(cast("TAnyObject", user_data))


class ListUsersQueryHandler(
    FlextCommands.QueryHandler[ListUsersQuery, list[TAnyObject]],
):
    """Handler for ListUsersQuery using flext_core.typings."""

    def __init__(self, users_db: dict[TEntityId, TUserData]) -> None:
        """Initialize query handler with user database."""
        self.users_db = users_db

    def handle(self, query: ListUsersQuery) -> FlextResult[list[TAnyObject]]:
        """Handle list users query using list[TAnyObject] return type."""
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

            users.append(cast("TAnyObject", user_data))

        return FlextResult[None].ok(users)


class GetUserEventsQueryHandler(
    FlextCommands.QueryHandler[GetUserEventsQuery, list[TAnyObject]],
):
    """Handler for GetUserEventsQuery using flext_core.typings."""

    def handle(self, query: GetUserEventsQuery) -> FlextResult[list[TAnyObject]]:
        """Handle get user events query using list[TAnyObject] return type."""
        events = event_store.get_events_by_correlation(query.correlation_id)
        event_data: list[TAnyObject] = [
            cast("TAnyObject", event.to_dict()) for event in events
        ]

        return FlextResult[None].ok(event_data)


# =============================================================================
# COMMAND BUS SETUP - Routing and execution
# =============================================================================


def setup_command_bus() -> FlextResult[FlextCommands.Bus]:
    """Setups command bus with handlers using flext_core.typings."""
    # Initialize user database
    users_db: dict[TEntityId, TUserData] = {}

    # Create command bus
    command_bus = FlextCommands.Bus()

    # Register command handlers
    create_handler: object = CreateUserCommandHandler(users_db)
    update_handler: object = UpdateUserCommandHandler(users_db)
    delete_handler: object = DeleteUserCommandHandler(users_db)

    command_bus.register_handler(CreateUserCommand, create_handler)
    command_bus.register_handler(UpdateUserCommand, update_handler)
    command_bus.register_handler(DeleteUserCommand, delete_handler)

    return FlextResult[None].ok(command_bus)


def setup_query_handlers(users_db: dict[TEntityId, TUserData]) -> dict[str, object]:
    """Setups query handlers using flext_core.typings."""
    query_handlers: dict[str, object] = {
        "get_user": GetUserQueryHandler(users_db),
        "list_users": ListUsersQueryHandler(users_db),
        "get_user_events": GetUserEventsQueryHandler(),
    }

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

    def create_user(
        self,
        name: str,
        email: str,
        age: int,
    ) -> FlextResult[TAnyObject]:
        """Create user using TAnyObject return type."""
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
        command = UpdateUserCommand(target_user_id=user_id, name=name, email=email)
        result = self.command_bus.execute(command)
        return result.map(lambda x: cast("TAnyObject", x))

    def delete_user(self, user_id: TEntityId, reason: str) -> FlextResult[TAnyObject]:
        """Delete user using TEntityId and TAnyObject types."""
        command = DeleteUserCommand(target_user_id=user_id, reason=reason)
        result = self.command_bus.execute(command)
        return result.map(lambda x: cast("TAnyObject", x))

    def get_user(self, user_id: TEntityId) -> FlextResult[TAnyObject]:
        """Get user using TEntityId and TAnyObject types."""
        query = GetUserQuery(target_user_id=user_id)
        handler = cast("QueryHandlerProtocol", self.query_handlers["get_user"])
        return cast("FlextResult[TAnyObject]", handler.handle(query))

    def list_users(
        self,
        min_age: int | None = None,
        max_age: int | None = None,
        *,
        active_only: bool = True,
    ) -> FlextResult[list[TAnyObject]]:
        """List users using list[TAnyObject] return type."""
        query = ListUsersQuery(
            active_only=active_only,
            min_age=min_age,
            max_age=max_age,
        )
        handler = cast("QueryHandlerProtocol", self.query_handlers["list_users"])
        return cast("FlextResult[list[TAnyObject]]", handler.handle(query))

    def get_user_events(
        self,
        correlation_id: str,
    ) -> FlextResult[list[TAnyObject]]:
        """Get user events using list[TAnyObject] return type."""
        query = GetUserEventsQuery(correlation_id=correlation_id)
        handler = cast(
            "QueryHandlerProtocol",
            self.query_handlers["get_user_events"],
        )
        return cast("FlextResult[list[TAnyObject]]", handler.handle(query))


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
            return FlextResult[None].fail(
                f"Command bus setup failed: {command_bus_result.error}",
            )

        # Setup query handlers
        query_handlers = setup_query_handlers(self.users_db)

        # Create application service
        command_bus = command_bus_result.data
        if command_bus is None:
            return FlextResult[None].fail("Command bus setup returned None")

        self.app_service = UserManagementApplicationService(command_bus, query_handlers)
        return FlextResult[None].ok(None)

    def demonstrate_user_creation(self) -> FlextResult[TEntityId]:
        """Demonstrate user creation command - reduced complexity."""
        if not self.app_service:
            return FlextResult[None].fail("Application service not initialized")

        # Use helper for standardized section header
        DemonstrationFlowHelper.print_section_header(2, "User Creation Commands")

        create_result: FlextResult[object] = self.app_service.create_user(
            "Alice Johnson",
            "alice@example.com",
            28,
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
                return FlextResult[None].ok(user_id)
            return FlextResult[None].ok("unknown_id")
        return FlextResult[None].fail(f"User creation failed: {create_result.error}")

    def demonstrate_user_update(self) -> FlextResult[None]:
        """Demonstrate user update command."""
        if not self.app_service or not self.created_user_id:
            return FlextResult[None].fail("User not created or service not initialized")

        update_result: FlextResult[object] = self.app_service.update_user(
            self.created_user_id,
            name="Alice Smith",
        )
        if update_result.success:
            # Update database
            if isinstance(update_result.data, dict):
                self.users_db[self.created_user_id] = update_result.data
            return FlextResult[None].ok(None)
        return FlextResult[None].fail(f"User update failed: {update_result.error}")

    def demonstrate_user_queries(self) -> FlextResult[None]:
        """Demonstrate user query operations."""
        if not self.app_service:
            return FlextResult[None].fail("Application service not initialized")

        list_result: FlextResult[list[object]] = self.app_service.list_users(
            active_only=True,
        )
        if list_result.success:
            users = list_result.data
            if isinstance(users, list):
                for user in users:
                    if isinstance(user, dict) and "name" in user:
                        user_dict = cast("dict[str, object]", user)
                        user_dict.get("name", "Unknown")
                        user_dict.get("email", "N/A")
            return FlextResult[None].ok(None)
        return FlextResult[None].fail(f"User listing failed: {list_result.error}")

    def demonstrate_event_sourcing(self) -> FlextResult[None]:
        """Demonstrate event sourcing functionality."""
        all_events = event_store.get_all_events()
        for _event in all_events:
            pass
        return FlextResult[None].ok(None)

    def demonstrate_user_deletion(self) -> FlextResult[None]:
        """Demonstrate user deletion command."""
        if not self.app_service or not self.created_user_id:
            return FlextResult[None].fail("User not created or service not initialized")

        delete_result: FlextResult[object] = self.app_service.delete_user(
            self.created_user_id,
            "User requested account deletion",
        )
        if delete_result.success:
            # Update database
            if isinstance(delete_result.data, dict):
                self.users_db[self.created_user_id] = delete_result.data
            return FlextResult[None].ok(None)
        return FlextResult[None].fail(f"User deletion failed: {delete_result.error}")

    def demonstrate_validation_failure(self) -> FlextResult[None]:
        """Demonstrate command validation with invalid data."""
        if not self.app_service:
            return FlextResult[None].fail("Application service not initialized")

        # Try to create user with invalid data
        invalid_result: FlextResult[object] = self.app_service.create_user(
            "",
            "invalid-email",
            15,
        )
        if invalid_result.is_failure:
            return FlextResult[None].ok(None)
        return FlextResult[None].fail("Validation should have failed")


def main() -> None:
    """Run comprehensive FlextCommands demonstration using SOLID principles."""
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
            return


if __name__ == "__main__":
    main()
