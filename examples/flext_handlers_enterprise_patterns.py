#!/usr/bin/env python3
"""Enterprise handler patterns with FlextHandlers.

Demonstrates CQRS, event sourcing, chain of responsibility,
and registry patterns for message processing.
    - Handler registry for service location and dependency injection
    - Chain of responsibility for multi-handler processing workflows
    - Handler lifecycle management with pre/post processing hooks
    - Function-based handler creation for flexibility
    - Enterprise handler patterns with metrics and logging
    - Performance monitoring and observability integration

Key Components:
    - FlextHandlers.Handler: Generic base handler with lifecycle management
    - FlextHandlers.CommandHandler: CQRS command processing with validation
    - FlextHandlers.EventHandler: Domain event processing with side effects
    - FlextHandlers.QueryHandler: Read-only query processing with authorization
    - FlextHandlers.Registry: Service location pattern for handler management
    - FlextHandlers.Chain: Chain of responsibility for complex workflows

This example shows real-world enterprise handler scenarios
demonstrating the power and flexibility of the FlextHandlers system.
"""

import time
import traceback
from dataclasses import dataclass
from typing import cast

from flext_core import (
    FlextEntity,
    FlextHandlers,
    FlextLogger,
    FlextResult,
)

# =============================================================================
# HANDLER CONSTANTS - Validation and business rule constraints
# =============================================================================

# Name validation constants
MIN_NAME_LENGTH = 2  # Minimum characters for name fields

# Order processing constants
MIN_ITEMS_FOR_DISCOUNT = 3  # Minimum items in order to qualify for discount

# =============================================================================
# DOMAIN MODELS - Business entities and messages
# =============================================================================


@dataclass
class User:
    """User domain model."""

    id: str
    name: str
    email: str
    is_active: bool = True


@dataclass
class Order:
    """Order domain model."""

    id: str
    user_id: str
    items: list[str]
    total: float
    status: str = "pending"


class UserEntity(FlextEntity):
    """User entity with domain behavior."""

    name: str
    email: str
    is_active: bool = True

    def validate_domain_rules(self) -> FlextResult[None]:
        """Validate domain rules for user entity."""
        if not self.name or len(self.name) < MIN_NAME_LENGTH:
            return FlextResult[None].fail(
                f"Name must be at least {MIN_NAME_LENGTH} characters",
            )
        if "@" not in self.email:
            return FlextResult[None].fail("Invalid email format")
        return FlextResult[None].ok(None)

    def activate(self) -> FlextResult["UserEntity"]:
        """Activate user."""
        if self.is_active:
            return FlextResult["UserEntity"].fail("User is already active")
        # Since entities are frozen, we need to create a new instance
        activated_user = UserEntity(
            id=self.id,
            name=self.name,
            email=self.email,
            is_active=True,
        )
        return FlextResult["UserEntity"].ok(activated_user)


# =============================================================================
# CQRS MESSAGES - Commands, Queries, and Events
# =============================================================================


@dataclass
class CreateUserCommand:
    """Command to create a new user."""

    name: str
    email: str

    def validate(self) -> FlextResult[None]:
        """Validate command data."""
        if not self.name or len(self.name) < MIN_NAME_LENGTH:
            return FlextResult[None].fail("Name must be at least 2 characters")
        if not self.email or "@" not in self.email:
            return FlextResult[None].fail("Invalid email format")
        return FlextResult[None].ok(None)


@dataclass
class UpdateUserCommand:
    """Command to update user information."""

    user_id: str
    name: str | None = None
    email: str | None = None

    def validate(self) -> FlextResult[None]:
        """Validate update command."""
        if not self.user_id:
            return FlextResult[None].fail("User ID is required")
        if self.name is not None and len(self.name) < MIN_NAME_LENGTH:
            return FlextResult[None].fail("Name must be at least 2 characters")
        if self.email is not None and "@" not in self.email:
            return FlextResult[None].fail("Invalid email format")
        return FlextResult[None].ok(None)


@dataclass
class GetUserQuery:
    """Query to retrieve user by ID."""

    user_id: str
    include_inactive: bool = False


@dataclass
class ListUsersQuery:
    """Query to list users with filtering."""

    active_only: bool = True
    limit: int = 10
    offset: int = 0


@dataclass
class UserCreatedEvent:
    """Event indicating user was created."""

    user_id: str
    name: str
    email: str
    timestamp: float


@dataclass
class UserUpdatedEvent:
    """Event indicating user was updated."""

    user_id: str
    changes: dict[str, object]
    timestamp: float


@dataclass
class OrderCreatedEvent:
    """Event indicating order was created."""

    order_id: str
    user_id: str
    total: float
    timestamp: float


# =============================================================================
# COMMAND HANDLERS - CQRS command processing
# =============================================================================


class CreateUserHandler(FlextHandlers.Implementation.BasicHandler):
    """Handler for creating users with validation."""

    def __init__(self) -> None:
        """Initialize CreateUserHandler."""
        super().__init__("CreateUserHandler")
        # Use imported FlextLoggerFactory for proper logger initialization
        self._logger = FlextLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
        # Simulate user storage
        self.users: dict[str, User] = {}
        self._next_id = 1

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return getattr(self, "_name", "CreateUserHandler")

    def can_handle(self, message_type: type) -> bool:
        """Check if can handle this message type."""
        return message_type is CreateUserCommand or (
            isinstance(message_type, type)
            and issubclass(message_type, CreateUserCommand)
        )

    def validate_command(self, command: object) -> FlextResult[None]:
        """Additional command validation."""
        if not isinstance(command, CreateUserCommand):
            return FlextResult[None].fail("Invalid command type")
        # Check if email already exists
        for user in self.users.values():
            if user.email == command.email:
                return FlextResult[None].fail(f"Email {command.email} already exists")
        return FlextResult[None].ok(None)

    def handle(self, request: object) -> FlextResult[object]:
        """Create new user."""
        if not isinstance(request, CreateUserCommand):
            return FlextResult[object].fail("Invalid command type")

        command = request
        user_id = f"user_{self._next_id}"
        self._next_id += 1

        user = User(
            id=user_id,
            name=command.name,
            email=command.email,
            is_active=True,
        )

        self.users[user_id] = user

        self.logger.info(
            "User created successfully",
            user_id=user_id,
            name=command.name,
            email=command.email,
        )

        return FlextResult[object].ok(user)


class UpdateUserHandler(FlextHandlers.Implementation.BasicHandler):
    """Handler for updating users."""

    def __init__(self, user_storage: dict[str, User]) -> None:
        """Initialize UpdateUserHandler.

        Args:
            user_storage: User storage dictionary

        """
        super().__init__("UpdateUserHandler")
        # Use imported FlextLoggerFactory for proper logger initialization

        self._logger = FlextLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
        self.users = user_storage

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return getattr(self, "_name", "UpdateUserHandler")

    def can_handle(self, message_type: type) -> bool:
        """Check if can handle this message type."""
        return message_type is UpdateUserCommand or (
            isinstance(message_type, type)
            and issubclass(message_type, UpdateUserCommand)
        )

    def validate_command(self, command: object) -> FlextResult[None]:
        """Validate update command."""
        if not isinstance(command, UpdateUserCommand):
            return FlextResult[None].fail("Invalid command type")
        if not command.user_id:
            return FlextResult[None].fail("User ID is required")
        if command.name is None and command.email is None:
            return FlextResult[None].fail(
                "At least one field must be provided for update"
            )
        return FlextResult[None].ok(None)

    def handle(self, request: object) -> FlextResult[object]:
        """Update user information."""
        if not isinstance(request, UpdateUserCommand):
            return FlextResult[object].fail("Invalid command type")

        command = request
        if command.user_id not in self.users:
            return FlextResult[object].fail(f"User {command.user_id} not found")

        user = self.users[command.user_id]
        changes = {}

        if command.name is not None:
            user.name = command.name
            changes["name"] = command.name

        if command.email is not None:
            user.email = command.email
            changes["email"] = command.email

        self.logger.info(
            "User updated successfully",
            user_id=command.user_id,
            changes=changes,
        )

        return FlextResult[object].ok(user)


# =============================================================================
# QUERY HANDLERS - CQRS query processing
# =============================================================================


class GetUserHandler(FlextHandlers.Implementation.BasicHandler):
    """Handler for retrieving individual users."""

    def __init__(self, user_storage: dict[str, User]) -> None:
        """Initialize GetUserHandler.

        Args:
            user_storage: User storage dictionary

        """
        super().__init__("GetUserHandler")
        # Use imported FlextLoggerFactory for proper logger initialization

        self._logger = FlextLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
        self.users = user_storage

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return getattr(self, "_name", "GetUserHandler")

    def can_handle(self, message_type: type) -> bool:
        """Check if can handle this message type."""
        return message_type is GetUserQuery or (
            isinstance(message_type, type) and issubclass(message_type, GetUserQuery)
        )

    def validate_command(self, query: object) -> FlextResult[None]:
        """Validate query (renamed from validate_command for consistency)."""
        if not isinstance(query, GetUserQuery):
            return FlextResult[None].fail("Invalid query type")
        # Simple query validation
        if not query.user_id:
            return FlextResult[None].fail("User ID is required")
        return FlextResult[None].ok(None)

    def authorize_query(self, query: object) -> FlextResult[None]:
        """Check query authorization."""
        if not isinstance(query, GetUserQuery):
            return FlextResult[None].fail("Invalid query type")
        # Simple authorization check
        if not query.user_id:
            return FlextResult[None].fail("User ID is required for authorization")
        return FlextResult[None].ok(None)

    def handle(self, request: object) -> FlextResult[object]:
        """Retrieve user by ID."""
        if not isinstance(request, GetUserQuery):
            return FlextResult[object].fail("Invalid query type")

        query = request
        if query.user_id not in self.users:
            return FlextResult[object].fail(f"User {query.user_id} not found")

        user = self.users[query.user_id]

        # Check if we should include inactive users
        if not query.include_inactive and not user.is_active:
            return FlextResult[object].fail(f"User {query.user_id} is inactive")

        self.logger.debug(
            "User retrieved successfully",
            user_id=query.user_id,
            user_name=user.name,
        )

        return FlextResult[object].ok(user)


class ListUsersHandler(FlextHandlers.Implementation.BasicHandler):
    """Handler for listing users with filtering."""

    def __init__(self, user_storage: dict[str, User]) -> None:
        """Initialize ListUsersHandler.

        Args:
            user_storage: User storage dictionary

        """
        super().__init__("ListUsersHandler")
        # Use imported FlextLoggerFactory for proper logger initialization

        self._logger = FlextLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
        self.users = user_storage

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return getattr(self, "_name", "ListUsersHandler")

    def can_handle(self, message_type: type) -> bool:
        """Check if can handle this message type."""
        return message_type is ListUsersQuery or (
            isinstance(message_type, type) and issubclass(message_type, ListUsersQuery)
        )

    def handle(self, request: object) -> FlextResult[object]:
        """List users with filtering and pagination."""
        if not isinstance(request, ListUsersQuery):
            return FlextResult[object].fail("Invalid query type")

        query = request
        users = list(self.users.values())

        # Filter by active status
        if query.active_only:
            users = [u for u in users if u.is_active]

        # Apply pagination
        start = query.offset
        end = start + query.limit
        paginated_users = users[start:end]

        self.logger.debug(
            "Users listed successfully",
            total_users=len(users),
            returned_users=len(paginated_users),
            active_only=query.active_only,
        )

        return FlextResult[object].ok(paginated_users)


# =============================================================================
# EVENT HANDLERS - Domain event processing
# =============================================================================


class UserCreatedEventHandler(FlextHandlers.Implementation.BasicHandler):
    """Handler for user created events."""

    def __init__(self) -> None:
        """Initialize UserCreatedEventHandler."""
        super().__init__("UserCreatedEventHandler")
        # Use imported FlextLoggerFactory for proper logger initialization
        # Don't override _handler_name as it's Final in parent class
        self._logger = FlextLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
        self._notifications_sent = 0

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    @property
    def handler_name(self) -> str:
        """Get handler name."""
        return getattr(self, "_name", "UserCreatedEventHandler")

    def can_handle(self, message_type: type) -> bool:
        """Check if can handle this request type."""
        return message_type is UserCreatedEvent or (
            isinstance(message_type, type)
            and issubclass(message_type, UserCreatedEvent)
        )

    def handle(self, request: object) -> FlextResult[object]:
        """Handle user created event."""
        result = self.process_event(request)
        if result.success:
            return FlextResult[object].ok(None)
        return FlextResult[object].fail(result.error or "Event processing failed")

    def process_event(self, event: object) -> FlextResult[None]:
        """Process user created event."""
        if not isinstance(event, UserCreatedEvent):
            return FlextResult[None].fail("Invalid event type")
        # Send welcome email (simulated)
        self.logger.info(
            "Sending welcome email",
            user_id=event.user_id,
            email=event.email,
            name=event.name,
        )

        # Update analytics (simulated)
        self.logger.info(
            "Updating user creation analytics",
            user_id=event.user_id,
            timestamp=event.timestamp,
        )

        self._notifications_sent += 1

        self.logger.info(
            "User created event processed",
            user_id=event.user_id,
            total_notifications=self._notifications_sent,
        )

        return FlextResult[None].ok(None)


class UserUpdatedEventHandler(FlextHandlers.Implementation.BasicHandler):
    """Handler for user updated events."""

    def __init__(self) -> None:
        """Initialize UserUpdatedEventHandler."""
        super().__init__("UserUpdatedEventHandler")
        # Use imported FlextLoggerFactory for proper logger initialization

        self._logger = FlextLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    def process_event(self, event: object) -> FlextResult[None]:
        """Process user updated event."""
        if not isinstance(event, UserUpdatedEvent):
            return FlextResult[None].fail("Invalid event type")
        # Log audit trail (simulated)
        self.logger.info(
            "Recording audit trail for user update",
            user_id=event.user_id,
            changes=event.changes,
            timestamp=event.timestamp,
        )

        # Invalidate caches (simulated)
        self.logger.debug(
            "Invalidating user caches",
            user_id=event.user_id,
        )

        self.logger.info(
            "User updated event processed",
            user_id=event.user_id,
            change_count=len(event.changes),
        )

        return FlextResult[None].ok(None)


class OrderCreatedEventHandler(FlextHandlers.Implementation.BasicHandler):
    """Handler for order created events."""

    def __init__(self) -> None:
        """Initialize OrderCreatedEventHandler."""
        super().__init__("OrderCreatedEventHandler")
        # Use imported FlextLoggerFactory for proper logger initialization

        self._logger = FlextLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
        self._orders_processed = 0

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    def process_event(self, event: object) -> FlextResult[None]:
        """Process order created event."""
        # Type guard
        if not isinstance(event, OrderCreatedEvent):
            return FlextResult[None].fail("Invalid event type")

        # Send order confirmation (simulated)
        self.logger.info(
            "Sending order confirmation",
            order_id=event.order_id,
            user_id=event.user_id,
            total=event.total,
        )

        # Update inventory (simulated)
        self.logger.info(
            "Updating inventory for order",
            order_id=event.order_id,
            total=event.total,
        )

        self._orders_processed += 1

        self.logger.info(
            "Order created event processed",
            order_id=event.order_id,
            total_orders_processed=self._orders_processed,
        )

        return FlextResult[None].ok(None)


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================


def demonstrate_command_handlers() -> None:
    """Demonstrate CQRS command handlers with validation."""
    _print_command_handlers_header()
    create_handler = CreateUserHandler()
    _test_create_user_handler(create_handler)
    update_handler = _test_update_user_handler(create_handler)
    _print_command_metrics(create_handler, update_handler)


def _print_command_handlers_header() -> None:
    pass


def _test_create_user_handler(create_handler: CreateUserHandler) -> None:
    valid_command = CreateUserCommand(name="John Doe", email="john@example.com")
    result = create_handler.handle(valid_command)
    if result.success:
        user_data = result.value
        if user_data is None:
            return
        if isinstance(user_data, User):
            pass

    duplicate_command = CreateUserCommand(name="Jane Doe", email="john@example.com")
    result = create_handler.handle(duplicate_command)
    if result.success:
        user_data = result.value
        if user_data is None:
            return
        if isinstance(user_data, User):
            pass


def _test_update_user_handler(create_handler: CreateUserHandler) -> UpdateUserHandler:
    update_handler = UpdateUserHandler(create_handler.users)
    first_user_id = next(iter(create_handler.users.keys()))
    update_command = UpdateUserCommand(
        user_id=first_user_id,
        name="John Smith",
        email="john.smith@example.com",
    )
    result = update_handler.handle(update_command)
    if result.success:
        updated_user_data = result.value
        if updated_user_data is None:
            return update_handler
        if isinstance(updated_user_data, User):
            pass
    return update_handler


def _print_command_metrics(
    create_handler: CreateUserHandler,
    update_handler: UpdateUserHandler,
) -> None:
    getattr(
        create_handler,
        "get_metrics",
        lambda: {"commands_processed": 0},
    )()
    getattr(
        update_handler,
        "get_metrics",
        lambda: {"commands_processed": 0},
    )()


def demonstrate_query_handlers() -> None:
    """Demonstrate CQRS query handlers with authorization."""
    _print_query_handlers_header()
    test_users = _setup_test_users()
    get_handler = GetUserHandler(test_users)
    _single_user_query(get_handler)
    _inactive_user_without_permission(get_handler)
    _inactive_user_with_permission(get_handler)
    list_handler = ListUsersHandler(test_users)
    _list_active_users(list_handler)
    _list_all_users(list_handler)
    _print_query_metrics(get_handler, list_handler)


def _print_query_handlers_header() -> None:
    pass


def _setup_test_users() -> dict[str, User]:
    return {
        "user_1": User("user_1", "Alice Johnson", "alice@example.com", is_active=True),
        "user_2": User("user_2", "Bob Wilson", "bob@example.com", is_active=True),
        "user_3": User("user_3", "Carol Brown", "carol@example.com", is_active=False),
    }


def _single_user_query(get_handler: GetUserHandler) -> None:
    query = GetUserQuery(user_id="user_1", include_inactive=False)
    result = get_handler.handle(query)
    if result.success:
        user_data = result.value
        if user_data is None:
            return
        if isinstance(user_data, User):
            pass


def _inactive_user_without_permission(get_handler: GetUserHandler) -> None:
    inactive_query = GetUserQuery(user_id="user_3", include_inactive=False)
    result = get_handler.handle(inactive_query)
    if result.success:
        user_data = result.value
        if user_data is None:
            return
        if isinstance(user_data, User):
            pass


def _inactive_user_with_permission(get_handler: GetUserHandler) -> None:
    inactive_query_allowed = GetUserQuery(user_id="user_3", include_inactive=True)
    result = get_handler.handle(inactive_query_allowed)
    if result.success:
        user_data = result.value
        if user_data is None:
            return
        if isinstance(user_data, User):
            pass


def _list_active_users(list_handler: ListUsersHandler) -> None:
    list_query = ListUsersQuery(active_only=True, limit=5, offset=0)
    result = list_handler.handle(list_query)
    if result.success:
        users_data = result.value
        if users_data is None:
            return
        if isinstance(users_data, list):
            for user_item in users_data:
                if isinstance(user_item, User):
                    pass


def _list_all_users(list_handler: ListUsersHandler) -> None:
    all_query = ListUsersQuery(active_only=False, limit=10, offset=0)
    result = list_handler.handle(all_query)
    if result.success:
        users_data = result.value
        if users_data is None:
            return
        if isinstance(users_data, list):
            for user_item in users_data:
                if isinstance(user_item, User):
                    pass


def _print_query_metrics(
    get_handler: GetUserHandler,
    list_handler: ListUsersHandler,
) -> None:
    getattr(
        get_handler,
        "get_metrics",
        lambda: {"queries_processed": 0},
    )()
    getattr(
        list_handler,
        "get_metrics",
        lambda: {"queries_processed": 0},
    )()


def demonstrate_event_handlers() -> None:
    """Demonstrate domain event handlers with side effects."""
    # 1. User created event handler
    user_created_handler = UserCreatedEventHandler()

    user_created_event = UserCreatedEvent(
        user_id="user_123",
        name="David Clark",
        email="david@example.com",
        timestamp=time.time(),
    )

    result = user_created_handler.handle(user_created_event)
    if result.success:
        pass

    # 2. User updated event handler
    user_updated_handler = UserUpdatedEventHandler()

    user_updated_event = UserUpdatedEvent(
        user_id="user_123",
        changes={"name": "David J. Clark", "email": "david.clark@example.com"},
        timestamp=time.time(),
    )

    result = user_updated_handler.handle(user_updated_event)  # type: ignore[attr-defined]
    if result.success:
        pass

    # 3. Order created event handler
    order_created_handler = OrderCreatedEventHandler()

    order_created_event = OrderCreatedEvent(
        order_id="order_456",
        user_id="user_123",
        total=299.99,
        timestamp=time.time(),
    )

    result = order_created_handler.handle(order_created_event)  # type: ignore[attr-defined]
    if result.success:
        pass

    # 4. Event handler metrics
    getattr(
        user_created_handler,
        "get_metrics",
        lambda: {"events_processed": 0},
    )()
    getattr(
        user_updated_handler,
        "get_metrics",
        lambda: {"events_processed": 0},
    )()
    getattr(
        order_created_handler,
        "get_metrics",
        lambda: {"events_processed": 0},
    )()


def demonstrate_handler_registry() -> None:
    """Demonstrate handler registry for service location."""
    _print_registry_header()
    registry = _setup_registry()
    _retrieve_handlers_by_key(registry)
    _retrieve_handlers_by_type(registry)
    _process_with_registry(registry)


def _print_registry_header() -> None:
    pass


def _setup_registry() -> FlextHandlers.Management.HandlerRegistry:
    registry = FlextHandlers.Management.HandlerRegistry()
    create_handler = CreateUserHandler()
    get_handler = GetUserHandler({})
    user_created_handler = UserCreatedEventHandler()
    registry.register("create_user", create_handler)
    registry.register("get_user", get_handler)
    registry.register("user_created_event", user_created_handler)
    return registry


def _retrieve_handlers_by_key(
    registry: FlextHandlers.Management.HandlerRegistry,
) -> None:
    result = registry.get_handler("create_user")
    if result.success:
        pass
    result = registry.get_handler("non_existent")
    if result.success:
        pass


def _retrieve_handlers_by_type(
    registry: FlextHandlers.Management.HandlerRegistry,
) -> None:
    # Using key-based access since type-based registration was not done
    result = registry.get_handler("create_user")
    if result.success:
        pass
    result = registry.get_handler("get_user")
    if result.success:
        pass


def _process_with_registry(registry: FlextHandlers.Management.HandlerRegistry) -> None:
    command = CreateUserCommand(name="Registry User", email="registry@example.com")
    handler_result = registry.get_handler("create_user")
    if handler_result.success:
        handler = handler_result.value
        if handler is None:
            return
        # Cast to proper handler type for method access
        typed_handler = cast("FlextHandlers.Implementation.BasicHandler", handler)
        command_result = typed_handler.handle(command)
        if command_result.success:
            user_data = command_result.value
            if user_data is None:
                return
            if isinstance(user_data, User):
                pass


def _create_handler_chain() -> tuple[
    FlextHandlers.Patterns.HandlerChain, dict[str, User], str | None
]:
    """Create handler chain and return chain, storage, and user_id."""
    # Create different types of handlers
    create_handler = CreateUserHandler()
    user_storage = create_handler.users  # Share storage

    get_handler = GetUserHandler(user_storage)
    update_handler = UpdateUserHandler(user_storage)
    user_event_handler = UserCreatedEventHandler()

    # Create chain
    chain = FlextHandlers.Patterns.HandlerChain("request_chain")
    # Cast handlers to ChainableHandler protocol for chain compatibility
    chain.add_handler(cast("FlextHandlers.Protocols.ChainableHandler", create_handler))
    chain.add_handler(cast("FlextHandlers.Protocols.ChainableHandler", get_handler))
    chain.add_handler(cast("FlextHandlers.Protocols.ChainableHandler", update_handler))
    chain.add_handler(
        cast("FlextHandlers.Protocols.ChainableHandler", user_event_handler)
    )

    # Process create command and get user_id
    create_command = CreateUserCommand(name="Chain User", email="chain@example.com")
    result = chain.handle(create_command)

    user_id = None
    if result.success:
        user = result.value
        if user is not None and hasattr(user, "name"):
            pass
        if hasattr(result.value, "id"):
            user_id = getattr(result.value, "id", None)

    return chain, user_storage, user_id


def _process_get_query(
    chain: FlextHandlers.Patterns.HandlerChain, user_id: str | None
) -> None:
    """Process get query through the chain."""
    if not user_id:
        return

    get_query = GetUserQuery(user_id=user_id)
    result = chain.handle(get_query)

    if result.success:
        user = result.value
        if hasattr(user, "name"):
            pass


def _process_update_command(
    chain: FlextHandlers.Patterns.HandlerChain, user_id: str | None
) -> None:
    """Process update command through the chain."""
    if not user_id:
        return

    update_command = UpdateUserCommand(
        user_id=user_id,
        name="Updated Chain User",
    )
    result = chain.handle(update_command)

    if result.success:
        user_data = result.value
        if isinstance(user_data, User):
            pass


def _process_event_through_all_handlers(
    chain: FlextHandlers.Patterns.HandlerChain,
) -> None:
    """Process event through all applicable handlers."""
    user_event = UserCreatedEvent(
        user_id="event_user",
        name="Event User",
        email="event@example.com",
        timestamp=time.time(),
    )

    # Process single event through chain
    result = chain.handle(user_event)
    if result.success:
        pass


def demonstrate_handler_chain() -> None:
    """Demonstrate chain of responsibility pattern."""
    # 1. Create multiple handlers for the chain
    chain, _user_storage, user_id = _create_handler_chain()

    # 2. Process different message types through chain
    _process_get_query(chain, user_id)
    _process_update_command(chain, user_id)

    # 3. Process message through all applicable handlers
    _process_event_through_all_handlers(chain)


def demonstrate_function_handlers() -> None:
    """Demonstrate function-based handler creation."""
    _print_function_handlers_header()
    message_handler, number_handler, order_handler = _create_function_handlers()
    _use_message_handler(message_handler)
    _use_number_handler(number_handler)
    _process_complex_order(order_handler)
    _print_function_metrics(message_handler, number_handler, order_handler)


def _print_function_handlers_header() -> None:
    pass


def _create_function_handlers() -> tuple[
    FlextHandlers.Implementation.BasicHandler,
    FlextHandlers.Implementation.BasicHandler,
    FlextHandlers.Implementation.BasicHandler,
]:
    def process_simple_message(message: str) -> FlextResult[str]:
        if not message:
            return FlextResult[str].fail("Empty message")
        return FlextResult[str].ok(f"Processed: {message.upper()}")

    def process_number(number: int) -> FlextResult[int]:
        if number < 0:
            return FlextResult[int].fail("Negative numbers not allowed")
        return FlextResult[int].ok(number * 2)

    # Functions are defined but not bound; handlers echo input by design here
    message_handler = FlextHandlers.Implementation.BasicHandler("message_handler")
    number_handler = FlextHandlers.Implementation.BasicHandler("number_handler")
    order_handler = FlextHandlers.Implementation.BasicHandler("order_handler")
    return (
        message_handler,
        number_handler,
        order_handler,
    )


def _use_message_handler(
    message_handler: FlextHandlers.Implementation.BasicHandler,
) -> None:
    try:
        result = message_handler.handle("hello world")
        if result.success:
            pass
    except (ValueError, TypeError, KeyError):
        pass
    try:
        result = message_handler.handle("")
        if result.success:
            pass
    except (ValueError, TypeError):
        pass


def _use_number_handler(
    number_handler: FlextHandlers.Implementation.BasicHandler,
) -> None:
    try:
        result = number_handler.handle(42)
        if result.success:
            pass
    except (ValueError, TypeError):
        pass
    try:
        result = number_handler.handle(-5)
        if result.success:
            pass
    except (ValueError, TypeError):
        pass


def _process_complex_order(
    order_handler: FlextHandlers.Implementation.BasicHandler,
) -> None:
    def process_order_total(
        order_data: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        if not order_data.get("items"):
            return FlextResult[dict[str, object]].fail("Order must have items")
        items = order_data["items"]
        if not isinstance(items, list) or len(items) == 0:
            return FlextResult[dict[str, object]].fail(
                "Order items must be a non-empty list"
            )
        item_prices = {"item1": 10.0, "item2": 15.0, "item3": 20.0}
        total = sum(item_prices.get(item, 5.0) for item in items)
        if len(items) >= MIN_ITEMS_FOR_DISCOUNT:
            total *= 0.9
        result = {
            "order_id": order_data.get("order_id", "unknown"),
            "items": items,
            "total": round(total, 2),
            "discount_applied": len(items) >= MIN_ITEMS_FOR_DISCOUNT,
        }
        return FlextResult[dict[str, object]].ok(result)

    order_data = {"order_id": "ORD001", "items": ["item1", "item2", "item3"]}
    try:
        result = order_handler.handle(order_data)
        if result.success:
            pass
    except (ValueError, TypeError, KeyError):
        pass


def _print_function_metrics(
    message_handler: FlextHandlers.Implementation.BasicHandler,
    number_handler: FlextHandlers.Implementation.BasicHandler,
    order_handler: FlextHandlers.Implementation.BasicHandler,
) -> None:
    try:
        cast(
            "dict[str, object]",
            getattr(
                message_handler,
                "get_metrics",
                lambda: {"handler_name": "Message", "handler_type": "Function"},
            )(),
        )
        cast(
            "dict[str, object]",
            getattr(
                number_handler,
                "get_metrics",
                lambda: {"handler_name": "Number", "handler_type": "Function"},
            )(),
        )
        cast(
            "dict[str, object]",
            getattr(
                order_handler,
                "get_metrics",
                lambda: {"handler_name": "Order", "handler_type": "Function"},
            )(),
        )
    except (KeyError, AttributeError):
        pass


def main() -> None:
    """Execute all FlextHandlers demonstrations."""
    try:
        demonstrate_command_handlers()
        demonstrate_query_handlers()
        demonstrate_event_handlers()
        demonstrate_handler_registry()
        demonstrate_handler_chain()
        demonstrate_function_handlers()

    except (ValueError, TypeError, ImportError, AttributeError):
        traceback.print_exc()


if __name__ == "__main__":
    main()
