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
from typing import TYPE_CHECKING, cast

from flext_core import (
    FlextBaseHandler,
    FlextEntity,
    FlextEventHandler,
    FlextHandlerChain,
    FlextHandlerRegistry,
    FlextLogger,
    FlextLoggerFactory,
    FlextResult,
)

if TYPE_CHECKING:
    from flext_core.handlers import FlextBaseHandler as FlextMessageHandler

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
            return FlextResult.fail(
                f"Name must be at least {MIN_NAME_LENGTH} characters",
            )
        if "@" not in self.email:
            return FlextResult.fail("Invalid email format")
        return FlextResult.ok(None)

    def activate(self) -> FlextResult["UserEntity"]:
        """Activate user."""
        if self.is_active:
            return FlextResult.fail("User is already active")
        # Since entities are frozen, we need to create a new instance
        activated_user = UserEntity(
            id=self.id, name=self.name, email=self.email, is_active=True,
        )
        return FlextResult.ok(activated_user)


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
            return FlextResult.fail("Name must be at least 2 characters")
        if not self.email or "@" not in self.email:
            return FlextResult.fail("Invalid email format")
        return FlextResult.ok(None)


@dataclass
class UpdateUserCommand:
    """Command to update user information."""

    user_id: str
    name: str | None = None
    email: str | None = None

    def validate(self) -> FlextResult[None]:
        """Validate update command."""
        if not self.user_id:
            return FlextResult.fail("User ID is required")
        if self.name is not None and len(self.name) < MIN_NAME_LENGTH:
            return FlextResult.fail("Name must be at least 2 characters")
        if self.email is not None and "@" not in self.email:
            return FlextResult.fail("Invalid email format")
        return FlextResult.ok(None)


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


class CreateUserHandler(FlextBaseHandler):
    """Handler for creating users with validation."""

    def __init__(self) -> None:
        """Initialize CreateUserHandler."""
        super().__init__("CreateUserHandler")
        # Use imported FlextLoggerFactory for proper logger initialization
        self._logger = FlextLoggerFactory.get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
        # Simulate user storage
        self.users: dict[str, User] = {}
        self._next_id = 1

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    def can_handle(self, message: object) -> bool:
        """Check if can handle this message type."""
        return isinstance(message, CreateUserCommand)

    def validate_command(self, command: object) -> FlextResult[None]:
        """Additional command validation."""
        if not isinstance(command, CreateUserCommand):
            return FlextResult.fail("Invalid command type")
        # Check if email already exists
        for user in self.users.values():
            if user.email == command.email:
                return FlextResult.fail(f"Email {command.email} already exists")
        return FlextResult.ok(None)

    def handle(self, command: object) -> FlextResult[object]:
        """Create new user."""
        if not isinstance(command, CreateUserCommand):
            return FlextResult.fail("Invalid command type")

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

        return FlextResult.ok(user)


class UpdateUserHandler(FlextBaseHandler):
    """Handler for updating users."""

    def __init__(self, user_storage: dict[str, User]) -> None:
        """Initialize UpdateUserHandler.

        Args:
            user_storage: User storage dictionary

        """
        super().__init__("UpdateUserHandler")
        # Use imported FlextLoggerFactory for proper logger initialization

        self._logger = FlextLoggerFactory.get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
        self.users = user_storage

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    def can_handle(self, message: object) -> bool:
        """Check if can handle this message type."""
        return isinstance(message, UpdateUserCommand)

    def validate_command(self, command: object) -> FlextResult[None]:
        """Validate update command."""
        if not isinstance(command, UpdateUserCommand):
            return FlextResult.fail("Invalid command type")
        if not command.user_id:
            return FlextResult.fail("User ID is required")
        if command.name is None and command.email is None:
            return FlextResult.fail("At least one field must be provided for update")
        return FlextResult.ok(None)

    def handle(self, command: object) -> FlextResult[object]:
        """Update user information."""
        if not isinstance(command, UpdateUserCommand):
            return FlextResult.fail("Invalid command type")
        if command.user_id not in self.users:
            return FlextResult.fail(f"User {command.user_id} not found")

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

        return FlextResult.ok(user)


# =============================================================================
# QUERY HANDLERS - CQRS query processing
# =============================================================================


class GetUserHandler(FlextBaseHandler):
    """Handler for retrieving individual users."""

    def __init__(self, user_storage: dict[str, User]) -> None:
        """Initialize GetUserHandler.

        Args:
            user_storage: User storage dictionary

        """
        super().__init__("GetUserHandler")
        # Use imported FlextLoggerFactory for proper logger initialization

        self._logger = FlextLoggerFactory.get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
        self.users = user_storage

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    def can_handle(self, message: object) -> bool:
        """Check if can handle this message type."""
        return isinstance(message, GetUserQuery)

    def validate_command(self, query: object) -> FlextResult[None]:
        """Validate query (renamed from validate_command for consistency)."""
        if not isinstance(query, GetUserQuery):
            return FlextResult.fail("Invalid query type")
        # Simple query validation
        if not query.user_id:
            return FlextResult.fail("User ID is required")
        return FlextResult.ok(None)

    def authorize_query(self, query: object) -> FlextResult[None]:
        """Check query authorization."""
        if not isinstance(query, GetUserQuery):
            return FlextResult.fail("Invalid query type")
        # Simple authorization check
        if not query.user_id:
            return FlextResult.fail("User ID is required for authorization")
        return FlextResult.ok(None)

    def handle(self, query: object) -> FlextResult[object]:
        """Retrieve user by ID."""
        if not isinstance(query, GetUserQuery):
            return FlextResult.fail("Invalid query type")
        if query.user_id not in self.users:
            return FlextResult.fail(f"User {query.user_id} not found")

        user = self.users[query.user_id]

        # Check if we should include inactive users
        if not query.include_inactive and not user.is_active:
            return FlextResult.fail(f"User {query.user_id} is inactive")

        self.logger.debug(
            "User retrieved successfully",
            user_id=query.user_id,
            user_name=user.name,
        )

        return FlextResult.ok(user)


class ListUsersHandler(FlextBaseHandler):
    """Handler for listing users with filtering."""

    def __init__(self, user_storage: dict[str, User]) -> None:
        """Initialize ListUsersHandler.

        Args:
            user_storage: User storage dictionary

        """
        super().__init__("ListUsersHandler")
        # Use imported FlextLoggerFactory for proper logger initialization

        self._logger = FlextLoggerFactory.get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
        self.users = user_storage

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    def handle(self, query: object) -> FlextResult[object]:
        """List users with filtering and pagination."""
        if not isinstance(query, ListUsersQuery):
            return FlextResult.fail("Invalid query type")
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

        return FlextResult.ok(paginated_users)


# =============================================================================
# EVENT HANDLERS - Domain event processing
# =============================================================================


class UserCreatedEventHandler(FlextEventHandler):
    """Handler for user created events."""

    def __init__(self) -> None:
        """Initialize UserCreatedEventHandler."""
        super().__init__("UserCreatedEventHandler")
        # Use imported FlextLoggerFactory for proper logger initialization

        self._logger = FlextLoggerFactory.get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )
        self._notifications_sent = 0

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    def process_event(self, event: object) -> FlextResult[None]:
        """Process user created event."""
        if not isinstance(event, UserCreatedEvent):
            return FlextResult.fail("Invalid event type")
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

        return FlextResult.ok(None)


class UserUpdatedEventHandler(FlextEventHandler):
    """Handler for user updated events."""

    def __init__(self) -> None:
        """Initialize UserUpdatedEventHandler."""
        super().__init__("UserUpdatedEventHandler")
        # Use imported FlextLoggerFactory for proper logger initialization

        self._logger = FlextLoggerFactory.get_logger(
            f"{self.__class__.__module__}.{self.__class__.__name__}",
        )

    @property
    def logger(self) -> FlextLogger:
        """Access logger."""
        return self._logger

    def process_event(self, event: object) -> FlextResult[None]:
        """Process user updated event."""
        if not isinstance(event, UserUpdatedEvent):
            return FlextResult.fail("Invalid event type")
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

        return FlextResult.ok(None)


class OrderCreatedEventHandler(FlextEventHandler):
    """Handler for order created events."""

    def __init__(self) -> None:
        """Initialize OrderCreatedEventHandler."""
        super().__init__("OrderCreatedEventHandler")
        # Use imported FlextLoggerFactory for proper logger initialization

        self._logger = FlextLoggerFactory.get_logger(
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
            return FlextResult.fail("Invalid event type")

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

        return FlextResult.ok(None)


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
    print("\n" + "=" * 80)
    print("⚡ COMMAND HANDLERS - CQRS PATTERN")
    print("=" * 80)


def _test_create_user_handler(create_handler: CreateUserHandler) -> None:
    print("\n1. Creating and using command handlers:")
    valid_command = CreateUserCommand(name="John Doe", email="john@example.com")
    result = create_handler.handle(valid_command)
    if result.success:
        user_data = result.data
        if user_data is None:
            print("❌ User creation returned None data")
            return
        if isinstance(user_data, User):
            print(
                f"✅ User created: {user_data.name} ({user_data.email}) - ID: {user_data.id}",
            )
        else:
            print(f"✅ User created: {user_data}")
    else:
        print(f"❌ User creation failed: {result.error}")

    duplicate_command = CreateUserCommand(name="Jane Doe", email="john@example.com")
    result = create_handler.handle(duplicate_command)
    if result.success:
        user_data = result.data
        if user_data is None:
            print("❌ User creation returned None data")
            return
        if isinstance(user_data, User):
            print(f"✅ User created: {user_data.name} ({user_data.email})")
        else:
            print(f"✅ User created: {user_data}")
    else:
        print(f"❌ Duplicate email prevented: {result.error}")


def _test_update_user_handler(create_handler: CreateUserHandler) -> UpdateUserHandler:
    print("\n2. Update command handler:")
    update_handler = UpdateUserHandler(create_handler.users)
    first_user_id = next(iter(create_handler.users.keys()))
    update_command = UpdateUserCommand(
        user_id=first_user_id, name="John Smith", email="john.smith@example.com",
    )
    result = update_handler.handle(update_command)
    if result.success:
        updated_user_data = result.data
        if updated_user_data is None:
            print("❌ User update returned None data")
            return update_handler
        if isinstance(updated_user_data, User):
            print(
                f"✅ User updated: {updated_user_data.name} ({updated_user_data.email})",
            )
        else:
            print(f"✅ User updated: {updated_user_data}")
    else:
        print(f"❌ User update failed: {result.error}")
    return update_handler


def _print_command_metrics(
    create_handler: CreateUserHandler, update_handler: UpdateUserHandler,
) -> None:
    print("\n3. Command handler metrics:")
    create_metrics = getattr(
        create_handler, "get_metrics", lambda: {"commands_processed": 0},
    )()
    update_metrics = getattr(
        update_handler, "get_metrics", lambda: {"commands_processed": 0},
    )()
    print("📊 Create Handler Metrics:")
    print(f"   Handler name: {create_metrics.get('handler_name', 'Unknown')}")
    print(f"   Handler type: {create_metrics.get('handler_type', 'Unknown')}")
    print("📊 Update Handler Metrics:")
    print(f"   Handler name: {update_metrics.get('handler_name', 'Unknown')}")
    print(f"   Handler type: {update_metrics.get('handler_type', 'Unknown')}")


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
    print("\n" + "=" * 80)
    print("🔍 QUERY HANDLERS - READ OPERATIONS")
    print("=" * 80)


def _setup_test_users() -> dict[str, User]:
    return {
        "user_1": User("user_1", "Alice Johnson", "alice@example.com", is_active=True),
        "user_2": User("user_2", "Bob Wilson", "bob@example.com", is_active=True),
        "user_3": User("user_3", "Carol Brown", "carol@example.com", is_active=False),
    }


def _single_user_query(get_handler: GetUserHandler) -> None:
    print("\n1. Single user query handler:")
    query = GetUserQuery(user_id="user_1", include_inactive=False)
    result = get_handler.handle(query)
    if result.success:
        user_data = result.data
        if user_data is None:
            print("❌ User query returned None data")
            return
        if isinstance(user_data, User):
            print(
                f"✅ User found: {user_data.name} ({user_data.email}) - Active: {user_data.is_active}",
            )
        else:
            print(f"✅ User found: {user_data}")
    else:
        print(f"❌ User query failed: {result.error}")


def _inactive_user_without_permission(get_handler: GetUserHandler) -> None:
    inactive_query = GetUserQuery(user_id="user_3", include_inactive=False)
    result = get_handler.handle(inactive_query)
    if result.success:
        user_data = result.data
        if user_data is None:
            print("❌ User query returned None data")
            return
        if isinstance(user_data, User):
            print(f"✅ Inactive user found: {user_data.name}")
        else:
            print(f"✅ Inactive user found: {user_data}")
    else:
        print(f"❌ Inactive user query failed (expected): {result.error}")


def _inactive_user_with_permission(get_handler: GetUserHandler) -> None:
    inactive_query_allowed = GetUserQuery(user_id="user_3", include_inactive=True)
    result = get_handler.handle(inactive_query_allowed)
    if result.success:
        user_data = result.data
        if user_data is None:
            print("❌ User query returned None data")
            return
        if isinstance(user_data, User):
            print(f"✅ Inactive user found with permission: {user_data.name}")
        else:
            print(f"✅ Inactive user found with permission: {user_data}")
    else:
        print(f"❌ Inactive user query failed: {result.error}")


def _list_active_users(list_handler: ListUsersHandler) -> None:
    print("\n2. List users query handler:")
    list_query = ListUsersQuery(active_only=True, limit=5, offset=0)
    result = list_handler.handle(list_query)
    if result.success:
        users_data = result.data
        if users_data is None:
            print("❌ Users query returned None data")
            return
        if isinstance(users_data, list):
            print(f"✅ Active users found: {len(users_data)}")
            for user_item in users_data:
                if isinstance(user_item, User):
                    print(f"   - {user_item.name} ({user_item.email})")
                else:
                    print(f"   - {user_item}")
        else:
            print(f"✅ Active users found: {users_data}")
    else:
        print(f"❌ List query failed: {result.error}")


def _list_all_users(list_handler: ListUsersHandler) -> None:
    all_query = ListUsersQuery(active_only=False, limit=10, offset=0)
    result = list_handler.handle(all_query)
    if result.success:
        users_data = result.data
        if users_data is None:
            print("❌ Users query returned None data")
            return
        if isinstance(users_data, list):
            print(f"✅ All users found: {len(users_data)}")
            for user_item in users_data:
                if isinstance(user_item, User):
                    status = "Active" if user_item.is_active else "Inactive"
                    print(f"   - {user_item.name} ({user_item.email}) - {status}")
                else:
                    print(f"   - {user_item}")
        else:
            print(f"✅ All users found: {users_data}")
    else:
        print(f"❌ All users query failed: {result.error}")


def _print_query_metrics(
    get_handler: GetUserHandler, list_handler: ListUsersHandler,
) -> None:
    print("\n3. Query handler metrics:")
    get_metrics = getattr(
        get_handler, "get_metrics", lambda: {"queries_processed": 0},
    )()
    list_metrics = getattr(
        list_handler, "get_metrics", lambda: {"queries_processed": 0},
    )()
    print("📊 Get Handler Metrics:")
    print(f"   Handler name: {get_metrics.get('handler_name', 'Unknown')}")
    print(f"   Handler type: {get_metrics.get('handler_type', 'Unknown')}")
    print("📊 List Handler Metrics:")
    print(f"   Handler name: {list_metrics.get('handler_name', 'Unknown')}")
    print(f"   Handler type: {list_metrics.get('handler_type', 'Unknown')}")


def demonstrate_event_handlers() -> None:
    """Demonstrate domain event handlers with side effects."""
    print("\n" + "=" * 80)
    print("📡 EVENT HANDLERS - DOMAIN EVENTS")
    print("=" * 80)

    # 1. User created event handler
    print("\n1. User created event handling:")
    user_created_handler = UserCreatedEventHandler()

    user_created_event = UserCreatedEvent(
        user_id="user_123",
        name="David Clark",
        email="david@example.com",
        timestamp=time.time(),
    )

    result = user_created_handler.handle(user_created_event)
    if result.success:
        print("✅ User created event processed successfully")
    else:
        print(f"❌ User created event failed: {result.error}")

    # 2. User updated event handler
    print("\n2. User updated event handling:")
    user_updated_handler = UserUpdatedEventHandler()

    user_updated_event = UserUpdatedEvent(
        user_id="user_123",
        changes={"name": "David J. Clark", "email": "david.clark@example.com"},
        timestamp=time.time(),
    )

    result = user_updated_handler.handle(user_updated_event)
    if result.success:
        print("✅ User updated event processed successfully")
    else:
        print(f"❌ User updated event failed: {result.error}")

    # 3. Order created event handler
    print("\n3. Order created event handling:")
    order_created_handler = OrderCreatedEventHandler()

    order_created_event = OrderCreatedEvent(
        order_id="order_456",
        user_id="user_123",
        total=299.99,
        timestamp=time.time(),
    )

    result = order_created_handler.handle(order_created_event)
    if result.success:
        print("✅ Order created event processed successfully")
    else:
        print(f"❌ Order created event failed: {result.error}")

    # 4. Event handler metrics
    print("\n4. Event handler metrics:")
    user_created_metrics = getattr(
        user_created_handler, "get_metrics", lambda: {"events_processed": 0},
    )()
    user_updated_metrics = getattr(
        user_updated_handler, "get_metrics", lambda: {"events_processed": 0},
    )()
    order_created_metrics = getattr(
        order_created_handler, "get_metrics", lambda: {"events_processed": 0},
    )()

    print("📊 User Created Handler:")
    print(f"   Handler name: {user_created_metrics.get('handler_name', 'Unknown')}")
    print(f"   Handler type: {user_created_metrics.get('handler_type', 'Unknown')}")

    print("📊 User Updated Handler:")
    print(f"   Handler name: {user_updated_metrics.get('handler_name', 'Unknown')}")
    print(f"   Handler type: {user_updated_metrics.get('handler_type', 'Unknown')}")

    print("📊 Order Created Handler:")
    print(f"   Handler name: {order_created_metrics.get('handler_name', 'Unknown')}")
    print(f"   Handler type: {order_created_metrics.get('handler_type', 'Unknown')}")


def demonstrate_handler_registry() -> None:
    """Demonstrate handler registry for service location."""
    _print_registry_header()
    registry = _setup_registry()
    _retrieve_handlers_by_key(registry)
    _retrieve_handlers_by_type(registry)
    _process_with_registry(registry)


def _print_registry_header() -> None:
    print("\n" + "=" * 80)
    print("📋 HANDLER REGISTRY - SERVICE LOCATION")
    print("=" * 80)


def _setup_registry() -> FlextHandlerRegistry:
    print("\n1. Creating and populating handler registry:")
    registry = FlextHandlerRegistry()
    create_handler = CreateUserHandler()
    get_handler = GetUserHandler({})
    user_created_handler = UserCreatedEventHandler()
    registry.register("create_user", create_handler)
    registry.register("get_user", get_handler)
    registry.register("user_created_event", user_created_handler)
    registry.register_for_type(CreateUserCommand, "create_user", create_handler)
    registry.register_for_type(GetUserQuery, "get_user", get_handler)
    registry.register_for_type(UserCreatedEvent, "user_created", user_created_handler)
    print("✅ Handlers registered by string keys and message types")
    return registry


def _retrieve_handlers_by_key(registry: FlextHandlerRegistry) -> None:
    print("\n2. Retrieving handlers by string key:")
    result = registry.get_handler("create_user")
    if result.success:
        handler = result.data
        print(f"✅ Found handler: {handler.__class__.__name__}")
    else:
        print(f"❌ Handler not found: {result.error}")
    result = registry.get_handler("non_existent")
    if result.success:
        handler = result.data
        print(f"✅ Found handler: {handler}")
    else:
        print(f"❌ Handler not found (expected): {result.error}")


def _retrieve_handlers_by_type(registry: FlextHandlerRegistry) -> None:
    print("\n3. Retrieving handlers by message type:")
    result = registry.get_handler_for_type(CreateUserCommand)
    if result.success:
        handler = result.data
        print(f"✅ Found handler for CreateUserCommand: {handler.__class__.__name__}")
    else:
        print(f"❌ Handler not found: {result.error}")
    result = registry.get_handler_for_type(GetUserQuery)
    if result.success:
        handler = result.data
        print(f"✅ Found handler for GetUserQuery: {handler.__class__.__name__}")
    else:
        print(f"❌ Handler not found: {result.error}")


def _process_with_registry(registry: FlextHandlerRegistry) -> None:
    print("\n4. Using registry for message processing:")
    command = CreateUserCommand(name="Registry User", email="registry@example.com")
    handler_result = registry.get_handler_for_type(CreateUserCommand)
    if handler_result.success:
        handler = handler_result.data
        if handler is None:
            print("❌ Handler registry returned None handler")
            return
        command_result = handler.handle(command)
        if command_result.success:
            user_data = command_result.data
            if user_data is None:
                print("❌ Handler returned None user data")
                return
            if isinstance(user_data, User):
                print(f"✅ Command processed via registry: {user_data.name}")
            else:
                print(f"✅ Command processed via registry: {user_data}")
        else:
            print(f"❌ Command processing failed: {command_result.error}")
    else:
        print(f"❌ No handler found for command: {handler_result.error}")


def _create_handler_chain() -> tuple[FlextHandlerChain, dict[str, User], str | None]:
    """Create handler chain and return chain, storage, and user_id."""
    # Create different types of handlers
    create_handler = CreateUserHandler()
    user_storage = create_handler.users  # Share storage

    get_handler = GetUserHandler(user_storage)
    update_handler = UpdateUserHandler(user_storage)
    user_event_handler = UserCreatedEventHandler()

    # Create chain - cast handlers to expected type
    chain = FlextHandlerChain()
    chain.add_handler(create_handler)
    chain.add_handler(get_handler)
    chain.add_handler(update_handler)
    chain.add_handler(user_event_handler)

    print("✅ Handler chain created with 4 handlers")

    # Process create command and get user_id
    create_command = CreateUserCommand(name="Chain User", email="chain@example.com")
    result = chain.process(create_command)

    user_id = None
    if result.success:
        user = result.data
        if user is not None and hasattr(user, "name"):
            print(f"✅ Create command handled by chain: {user.name}")
        else:
            print("✅ Create command handled by chain")
        if hasattr(result.data, "id"):
            user_id = result.data.id
    else:
        print(f"❌ Create command failed: {result.error}")

    return chain, user_storage, user_id


def _process_get_query(chain: FlextHandlerChain, user_id: str | None) -> None:
    """Process get query through the chain."""
    if not user_id:
        return

    get_query = GetUserQuery(user_id=user_id)
    result = chain.process(get_query)

    if result.success:
        user = result.data
        if hasattr(user, "name"):
            print(f"✅ Get query handled by chain: {user.name}")
        else:
            print(f"❌ Get query result type error: got {type(user)}, expected User")
    else:
        print(f"❌ Get query failed: {result.error}")


def _process_update_command(chain: FlextHandlerChain, user_id: str | None) -> None:
    """Process update command through the chain."""
    if not user_id:
        return

    update_command = UpdateUserCommand(
        user_id=user_id,
        name="Updated Chain User",
    )
    result = chain.process(update_command)

    if result.success:
        user_data = result.data
        if isinstance(user_data, User):
            print(f"✅ Update command handled by chain: {user_data.name}")
        else:
            print(f"✅ Update command handled by chain: {user_data}")
    else:
        print(f"❌ Update command failed: {result.error}")


def _process_event_through_all_handlers(chain: FlextHandlerChain) -> None:
    """Process event through all applicable handlers."""
    user_event = UserCreatedEvent(
        user_id="event_user",
        name="Event User",
        email="event@example.com",
        timestamp=time.time(),
    )

    results: FlextResult[list[object]] = chain.process_all([user_event])
    result_list: list[object] = results.unwrap_or([])
    print(f"📊 Event processed by {len(result_list)} handlers")

    for i, result in enumerate(result_list, 1):
        # Each result in the list should be a FlextResult - need to check that
        if hasattr(result, "success") and result.success:
            print(f"   ✅ Handler {i}: Success")
        elif hasattr(result, "error"):
            print(f"   ❌ Handler {i}: {result.error}")
        else:
            print(f"   Info Handler {i}: {result}")


def demonstrate_handler_chain() -> None:
    """Demonstrate chain of responsibility pattern."""
    print("\n" + "=" * 80)
    print("🔗 HANDLER CHAIN - CHAIN OF RESPONSIBILITY")
    print("=" * 80)

    # 1. Create multiple handlers for the chain
    print("\n1. Creating handler chain:")
    chain, _user_storage, user_id = _create_handler_chain()

    # 2. Process different message types through chain
    print("\n2. Processing messages through chain:")
    _process_get_query(chain, user_id)
    _process_update_command(chain, user_id)

    # 3. Process message through all applicable handlers
    print("\n3. Processing event through all applicable handlers:")
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
    print("\n" + "=" * 80)
    print("🔧 FUNCTION HANDLERS - FUNCTIONAL STYLE")
    print("=" * 80)


def _create_function_handlers() -> tuple[
    FlextBaseHandler, FlextBaseHandler, FlextBaseHandler,
]:
    print("\n1. Creating function-based handlers:")

    def process_simple_message(message: str) -> FlextResult[str]:
        if not message:
            return FlextResult.fail("Empty message")
        return FlextResult.ok(f"Processed: {message.upper()}")

    def process_number(number: int) -> FlextResult[int]:
        if number < 0:
            return FlextResult.fail("Negative numbers not allowed")
        return FlextResult.ok(number * 2)

    # Functions are defined but not bound; handlers echo input by design here
    message_handler: FlextMessageHandler = FlextBaseHandler("message_handler")
    number_handler: FlextMessageHandler = FlextBaseHandler("number_handler")
    order_handler: FlextMessageHandler = FlextBaseHandler("order_handler")
    print("✅ Function handlers created")
    return (
        cast("FlextBaseHandler", message_handler),
        cast("FlextBaseHandler", number_handler),
        cast("FlextBaseHandler", order_handler),
    )


def _use_message_handler(message_handler: FlextBaseHandler) -> None:
    print("\n2. Using function-based handlers:")
    try:
        result = message_handler.handle("hello world")
        if result.success:
            print(f"✅ Message handler result: {result.data}")
        else:
            print(f"❌ Message handler failed: {result.error}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"❌ Message handler error: {e}")
    try:
        result = message_handler.handle("")
        if result.success:
            print(f"✅ Empty message result: {result.data}")
        else:
            print(f"❌ Empty message failed (expected): {result.error}")
    except (ValueError, TypeError) as e:
        print(f"❌ Empty message error: {e}")


def _use_number_handler(number_handler: FlextBaseHandler) -> None:
    try:
        result = number_handler.handle(42)
        if result.success:
            print(f"✅ Number handler result: {result.data}")
        else:
            print(f"❌ Number handler failed: {result.error}")
    except (ValueError, TypeError) as e:
        print(f"❌ Number handler error: {e}")
    try:
        result = number_handler.handle(-5)
        if result.success:
            print(f"✅ Negative number result: {result.data}")
        else:
            print(f"❌ Negative number failed (expected): {result.error}")
    except (ValueError, TypeError) as e:
        print(f"❌ Negative number error: {e}")


def _process_complex_order(order_handler: FlextBaseHandler) -> None:
    print("\n3. Complex function handler:")

    def process_order_total(
        order_data: dict[str, object],
    ) -> FlextResult[dict[str, object]]:
        if not order_data.get("items"):
            return FlextResult.fail("Order must have items")
        items = order_data["items"]
        if not isinstance(items, list) or len(items) == 0:
            return FlextResult.fail("Order items must be a non-empty list")
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
        return FlextResult.ok(result)

    order_data = {"order_id": "ORD001", "items": ["item1", "item2", "item3"]}
    try:
        result = order_handler.handle(order_data)
        if result.success:
            order_result = result.data
            print(f"✅ Order processed: {order_result}")
        else:
            print(f"❌ Order processing failed: {result.error}")
    except (ValueError, TypeError, KeyError) as e:
        print(f"❌ Order processing error: {e}")


def _print_function_metrics(
    message_handler: FlextBaseHandler,
    number_handler: FlextBaseHandler,
    order_handler: FlextBaseHandler,
) -> None:
    print("\n4. Function handler metrics:")
    try:
        message_metrics = cast(
            "dict[str, object]",
            getattr(
                message_handler,
                "get_metrics",
                lambda: {"handler_name": "Message", "handler_type": "Function"},
            )(),
        )
        number_metrics = cast(
            "dict[str, object]",
            getattr(
                number_handler,
                "get_metrics",
                lambda: {"handler_name": "Number", "handler_type": "Function"},
            )(),
        )
        order_metrics = cast(
            "dict[str, object]",
            getattr(
                order_handler,
                "get_metrics",
                lambda: {"handler_name": "Order", "handler_type": "Function"},
            )(),
        )
        print("📊 Message Handler:")
        print(f"   Handler name: {message_metrics.get('handler_name', 'Unknown')}")
        print(f"   Handler type: {message_metrics.get('handler_type', 'Unknown')}")
        print("📊 Number Handler:")
        print(f"   Handler name: {number_metrics.get('handler_name', 'Unknown')}")
        print(f"   Handler type: {number_metrics.get('handler_type', 'Unknown')}")
        print("📊 Order Handler:")
        print(f"   Handler name: {order_metrics.get('handler_name', 'Unknown')}")
        print(f"   Handler type: {order_metrics.get('handler_type', 'Unknown')}")
    except (KeyError, AttributeError) as e:
        print(f"❌ Error getting metrics: {e}")


def main() -> None:
    """Execute all FlextHandlers demonstrations."""
    print("🚀 FLEXT HANDLERS - ENTERPRISE PATTERNS EXAMPLE")
    print("Demonstrating comprehensive handler patterns for enterprise applications")

    try:
        demonstrate_command_handlers()
        demonstrate_query_handlers()
        demonstrate_event_handlers()
        demonstrate_handler_registry()
        demonstrate_handler_chain()
        demonstrate_function_handlers()

        print("\n" + "=" * 80)
        print("✅ ALL FLEXT HANDLERS DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📊 Summary of patterns demonstrated:")
        print("   ⚡ Command handlers with CQRS validation and business logic")
        print("   🔍 Query handlers with authorization and read-only operations")
        print("   📡 Event handlers with domain events and side effect processing")
        print("   📋 Handler registry with service location and dependency injection")
        print("   🔗 Handler chain with chain of responsibility pattern")
        print("   🔧 Function handlers with functional programming style")
        print("\n💡 FlextHandlers provides enterprise-grade message processing")
        print(
            "   with CQRS patterns, service location, and flexible handler "
            "composition!",
        )

    except (ValueError, TypeError, ImportError, AttributeError) as e:
        print(f"\n❌ Error during FlextHandlers demonstration: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
