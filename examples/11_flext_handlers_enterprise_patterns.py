#!/usr/bin/env python3
"""FLEXT Handlers Enterprise Patterns Example.

Comprehensive demonstration of FlextHandlers system showing enterprise-grade
handler patterns with CQRS, event sourcing, chain of responsibility, and
registry patterns for message processing.

Features demonstrated:
    - Base handler patterns with generic type safety
    - CQRS pattern implementation (Commands, Queries, Events)
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
from dataclasses import dataclass
from typing import Any

from flext_core.entities import FlextEntity
from flext_core.handlers import FlextHandlers
from flext_core.result import FlextResult

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

    def __init__(self, user_id: str, name: str, email: str) -> None:
        super().__init__(user_id)
        self.name = name
        self.email = email
        self.is_active = True

    def activate(self) -> FlextResult[None]:
        """Activate user."""
        if self.is_active:
            return FlextResult.fail("User is already active")
        self.is_active = True
        return FlextResult.ok(None)


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
        if not self.name or len(self.name) < 2:
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
        if self.name is not None and len(self.name) < 2:
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
    changes: dict[str, Any]
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


class CreateUserHandler(FlextHandlers.CommandHandler[CreateUserCommand, User]):
    """Handler for creating users with validation."""

    def __init__(self) -> None:
        super().__init__("CreateUserHandler")
        # Simulate user storage
        self._users: dict[str, User] = {}
        self._next_id = 1

    def validate_command(self, command: CreateUserCommand) -> FlextResult[None]:
        """Additional command validation."""
        # Check if email already exists
        for user in self._users.values():
            if user.email == command.email:
                return FlextResult.fail(f"Email {command.email} already exists")
        return FlextResult.ok(None)

    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        """Create new user."""
        user_id = f"user_{self._next_id}"
        self._next_id += 1

        user = User(
            id=user_id,
            name=command.name,
            email=command.email,
            is_active=True,
        )

        self._users[user_id] = user

        self.logger.info(
            "User created successfully",
            user_id=user_id,
            name=command.name,
            email=command.email,
        )

        return FlextResult.ok(user)


class UpdateUserHandler(FlextHandlers.CommandHandler[UpdateUserCommand, User]):
    """Handler for updating users."""

    def __init__(self, user_storage: dict[str, User]) -> None:
        super().__init__("UpdateUserHandler")
        self._users = user_storage

    def handle(self, command: UpdateUserCommand) -> FlextResult[User]:
        """Update user information."""
        if command.user_id not in self._users:
            return FlextResult.fail(f"User {command.user_id} not found")

        user = self._users[command.user_id]
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


class GetUserHandler(FlextHandlers.QueryHandler[GetUserQuery, User]):
    """Handler for retrieving individual users."""

    def __init__(self, user_storage: dict[str, User]) -> None:
        super().__init__("GetUserHandler")
        self._users = user_storage

    def authorize_query(self, query: GetUserQuery) -> FlextResult[None]:
        """Check query authorization."""
        # Simple authorization check
        if not query.user_id:
            return FlextResult.fail("User ID is required for authorization")
        return FlextResult.ok(None)

    def handle(self, query: GetUserQuery) -> FlextResult[User]:
        """Retrieve user by ID."""
        if query.user_id not in self._users:
            return FlextResult.fail(f"User {query.user_id} not found")

        user = self._users[query.user_id]

        # Check if we should include inactive users
        if not query.include_inactive and not user.is_active:
            return FlextResult.fail(f"User {query.user_id} is inactive")

        self.logger.debug(
            "User retrieved successfully",
            user_id=query.user_id,
            user_name=user.name,
        )

        return FlextResult.ok(user)


class ListUsersHandler(FlextHandlers.QueryHandler[ListUsersQuery, list[User]]):
    """Handler for listing users with filtering."""

    def __init__(self, user_storage: dict[str, User]) -> None:
        super().__init__("ListUsersHandler")
        self._users = user_storage

    def handle(self, query: ListUsersQuery) -> FlextResult[list[User]]:
        """List users with filtering and pagination."""
        users = list(self._users.values())

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


class UserCreatedEventHandler(FlextHandlers.EventHandler[UserCreatedEvent]):
    """Handler for user created events."""

    def __init__(self) -> None:
        super().__init__("UserCreatedEventHandler")
        self._notifications_sent = 0

    def process_event(self, event: UserCreatedEvent) -> None:
        """Process user created event."""
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


class UserUpdatedEventHandler(FlextHandlers.EventHandler[UserUpdatedEvent]):
    """Handler for user updated events."""

    def __init__(self) -> None:
        super().__init__("UserUpdatedEventHandler")

    def process_event(self, event: UserUpdatedEvent) -> None:
        """Process user updated event."""
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


class OrderCreatedEventHandler(FlextHandlers.EventHandler[OrderCreatedEvent]):
    """Handler for order created events."""

    def __init__(self) -> None:
        super().__init__("OrderCreatedEventHandler")
        self._orders_processed = 0

    def process_event(self, event: OrderCreatedEvent) -> None:
        """Process order created event."""
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


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================


def demonstrate_command_handlers() -> None:
    """Demonstrate CQRS command handlers with validation."""
    print("\n" + "=" * 80)
    print("⚡ COMMAND HANDLERS - CQRS PATTERN")
    print("=" * 80)

    # 1. Create user command handler
    print("\n1. Creating and using command handlers:")
    create_handler = CreateUserHandler()

    # Test valid command
    valid_command = CreateUserCommand(
        name="John Doe",
        email="john@example.com",
    )

    result = create_handler.handle_with_hooks(valid_command)
    if result.is_success:
        user = result.data
        print(f"✅ User created: {user.name} ({user.email}) - ID: {user.id}")
    else:
        print(f"❌ User creation failed: {result.error}")

    # Test invalid command (duplicate email)
    duplicate_command = CreateUserCommand(
        name="Jane Doe",
        email="john@example.com",  # Same email
    )

    result = create_handler.handle_with_hooks(duplicate_command)
    if result.is_success:
        user = result.data
        print(f"✅ User created: {user.name} ({user.email})")
    else:
        print(f"❌ Duplicate email prevented: {result.error}")

    # 2. Update user command handler
    print("\n2. Update command handler:")
    update_handler = UpdateUserHandler(create_handler._users)

    # Get first user ID
    first_user_id = next(iter(create_handler._users.keys()))

    update_command = UpdateUserCommand(
        user_id=first_user_id,
        name="John Smith",
        email="john.smith@example.com",
    )

    result = update_handler.handle_with_hooks(update_command)
    if result.is_success:
        updated_user = result.data
        print(f"✅ User updated: {updated_user.name} ({updated_user.email})")
    else:
        print(f"❌ User update failed: {result.error}")

    # 3. Command handler metrics
    print("\n3. Command handler metrics:")
    create_metrics = create_handler.get_metrics()
    update_metrics = update_handler.get_metrics()

    print("📊 Create Handler Metrics:")
    print(f"   Messages handled: {create_metrics['messages_handled']}")
    print(f"   Success rate: {create_metrics['success_rate']:.2%}")
    print(f"   Average time: {create_metrics['average_processing_time_ms']:.2f}ms")

    print("📊 Update Handler Metrics:")
    print(f"   Messages handled: {update_metrics['messages_handled']}")
    print(f"   Success rate: {update_metrics['success_rate']:.2%}")
    print(f"   Average time: {update_metrics['average_processing_time_ms']:.2f}ms")


def demonstrate_query_handlers() -> None:
    """Demonstrate CQRS query handlers with authorization."""
    print("\n" + "=" * 80)
    print("🔍 QUERY HANDLERS - READ OPERATIONS")
    print("=" * 80)

    # Set up test data
    test_users = {
        "user_1": User("user_1", "Alice Johnson", "alice@example.com", True),
        "user_2": User("user_2", "Bob Wilson", "bob@example.com", True),
        "user_3": User("user_3", "Carol Brown", "carol@example.com", False),  # Inactive
    }

    # 1. Get single user query
    print("\n1. Single user query handler:")
    get_handler = GetUserHandler(test_users)

    # Query active user
    query = GetUserQuery(user_id="user_1", include_inactive=False)
    result = get_handler.handle_with_hooks(query)

    if result.is_success:
        user = result.data
        print(f"✅ User found: {user.name} ({user.email}) - Active: {user.is_active}")
    else:
        print(f"❌ User query failed: {result.error}")

    # Query inactive user without permission
    inactive_query = GetUserQuery(user_id="user_3", include_inactive=False)
    result = get_handler.handle_with_hooks(inactive_query)

    if result.is_success:
        user = result.data
        print(f"✅ Inactive user found: {user.name}")
    else:
        print(f"❌ Inactive user query failed (expected): {result.error}")

    # Query inactive user with permission
    inactive_query_allowed = GetUserQuery(user_id="user_3", include_inactive=True)
    result = get_handler.handle_with_hooks(inactive_query_allowed)

    if result.is_success:
        user = result.data
        print(f"✅ Inactive user found with permission: {user.name}")
    else:
        print(f"❌ Inactive user query failed: {result.error}")

    # 2. List users query
    print("\n2. List users query handler:")
    list_handler = ListUsersHandler(test_users)

    # List active users only
    list_query = ListUsersQuery(active_only=True, limit=5, offset=0)
    result = list_handler.handle_with_hooks(list_query)

    if result.is_success:
        users = result.data
        print(f"✅ Active users found: {len(users)}")
        for user in users:
            print(f"   - {user.name} ({user.email})")
    else:
        print(f"❌ List query failed: {result.error}")

    # List all users including inactive
    all_query = ListUsersQuery(active_only=False, limit=10, offset=0)
    result = list_handler.handle_with_hooks(all_query)

    if result.is_success:
        users = result.data
        print(f"✅ All users found: {len(users)}")
        for user in users:
            status = "Active" if user.is_active else "Inactive"
            print(f"   - {user.name} ({user.email}) - {status}")
    else:
        print(f"❌ All users query failed: {result.error}")

    # 3. Query handler metrics
    print("\n3. Query handler metrics:")
    get_metrics = get_handler.get_metrics()
    list_metrics = list_handler.get_metrics()

    print("📊 Get Handler Metrics:")
    print(f"   Messages handled: {get_metrics['messages_handled']}")
    print(f"   Success rate: {get_metrics['success_rate']:.2%}")

    print("📊 List Handler Metrics:")
    print(f"   Messages handled: {list_metrics['messages_handled']}")
    print(f"   Success rate: {list_metrics['success_rate']:.2%}")


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

    result = user_created_handler.handle_with_hooks(user_created_event)
    if result.is_success:
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

    result = user_updated_handler.handle_with_hooks(user_updated_event)
    if result.is_success:
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

    result = order_created_handler.handle_with_hooks(order_created_event)
    if result.is_success:
        print("✅ Order created event processed successfully")
    else:
        print(f"❌ Order created event failed: {result.error}")

    # 4. Event handler metrics
    print("\n4. Event handler metrics:")
    user_created_metrics = user_created_handler.get_metrics()
    user_updated_metrics = user_updated_handler.get_metrics()
    order_created_metrics = order_created_handler.get_metrics()

    print("📊 User Created Handler:")
    print(f"   Events processed: {user_created_metrics['messages_handled']}")
    print(f"   Success rate: {user_created_metrics['success_rate']:.2%}")

    print("📊 User Updated Handler:")
    print(f"   Events processed: {user_updated_metrics['messages_handled']}")
    print(f"   Success rate: {user_updated_metrics['success_rate']:.2%}")

    print("📊 Order Created Handler:")
    print(f"   Events processed: {order_created_metrics['messages_handled']}")
    print(f"   Success rate: {order_created_metrics['success_rate']:.2%}")


def demonstrate_handler_registry() -> None:
    """Demonstrate handler registry for service location."""
    print("\n" + "=" * 80)
    print("📋 HANDLER REGISTRY - SERVICE LOCATION")
    print("=" * 80)

    # 1. Create registry and register handlers
    print("\n1. Creating and populating handler registry:")
    registry = FlextHandlers.create_registry()

    # Create handlers
    create_handler = CreateUserHandler()
    get_handler = GetUserHandler({})
    user_created_handler = UserCreatedEventHandler()

    # Register by string keys
    registry.register("create_user", create_handler)
    registry.register("get_user", get_handler)
    registry.register("user_created_event", user_created_handler)

    print("✅ Handlers registered by string keys")

    # Register by type
    registry.register_for_type(CreateUserCommand, create_handler)
    registry.register_for_type(GetUserQuery, get_handler)
    registry.register_for_type(UserCreatedEvent, user_created_handler)

    print("✅ Handlers registered by message types")

    # 2. Retrieve handlers by key
    print("\n2. Retrieving handlers by string key:")

    result = registry.get_handler("create_user")
    if result.is_success:
        handler = result.data
        print(f"✅ Found handler: {handler.__class__.__name__}")
    else:
        print(f"❌ Handler not found: {result.error}")

    # Try to get non-existent handler
    result = registry.get_handler("non_existent")
    if result.is_success:
        handler = result.data
        print(f"✅ Found handler: {handler}")
    else:
        print(f"❌ Handler not found (expected): {result.error}")

    # 3. Retrieve handlers by type
    print("\n3. Retrieving handlers by message type:")

    result = registry.get_handler_for_type(CreateUserCommand)
    if result.is_success:
        handler = result.data
        print(f"✅ Found handler for CreateUserCommand: {handler.__class__.__name__}")
    else:
        print(f"❌ Handler not found: {result.error}")

    result = registry.get_handler_for_type(GetUserQuery)
    if result.is_success:
        handler = result.data
        print(f"✅ Found handler for GetUserQuery: {handler.__class__.__name__}")
    else:
        print(f"❌ Handler not found: {result.error}")

    # 4. Use registry for message processing
    print("\n4. Using registry for message processing:")

    # Process command through registry
    command = CreateUserCommand(name="Registry User", email="registry@example.com")
    handler_result = registry.get_handler_for_type(CreateUserCommand)

    if handler_result.is_success:
        handler = handler_result.data
        result = handler.handle_with_hooks(command)
        if result.is_success:
            user = result.data
            print(f"✅ Command processed via registry: {user.name}")
        else:
            print(f"❌ Command processing failed: {result.error}")
    else:
        print(f"❌ No handler found for command: {handler_result.error}")


def demonstrate_handler_chain() -> None:
    """Demonstrate chain of responsibility pattern."""
    print("\n" + "=" * 80)
    print("🔗 HANDLER CHAIN - CHAIN OF RESPONSIBILITY")
    print("=" * 80)

    # 1. Create multiple handlers for the chain
    print("\n1. Creating handler chain:")

    # Create different types of handlers
    create_handler = CreateUserHandler()
    user_storage = create_handler._users  # Share storage

    get_handler = GetUserHandler(user_storage)
    update_handler = UpdateUserHandler(user_storage)
    user_event_handler = UserCreatedEventHandler()

    # Create chain
    chain = FlextHandlers.create_chain()
    chain.add_handler(create_handler)
    chain.add_handler(get_handler)
    chain.add_handler(update_handler)
    chain.add_handler(user_event_handler)

    print("✅ Handler chain created with 4 handlers")

    # 2. Process different message types through chain
    print("\n2. Processing messages through chain:")

    # Process create command
    create_command = CreateUserCommand(name="Chain User", email="chain@example.com")
    result = chain.process(create_command)

    if result.is_success:
        user = result.data
        print(f"✅ Create command handled by chain: {user.name}")
    else:
        print(f"❌ Create command failed: {result.error}")

    # Get the created user's ID for subsequent operations
    user_id = None
    if result.is_success and hasattr(result.data, "id"):
        user_id = result.data.id

    # Process get query
    if user_id:
        get_query = GetUserQuery(user_id=user_id)
        result = chain.process(get_query)

        if result.is_success:
            user = result.data
            print(f"✅ Get query handled by chain: {user.name}")
        else:
            print(f"❌ Get query failed: {result.error}")

    # Process update command
    if user_id:
        update_command = UpdateUserCommand(
            user_id=user_id,
            name="Updated Chain User",
        )
        result = chain.process(update_command)

        if result.is_success:
            user = result.data
            print(f"✅ Update command handled by chain: {user.name}")
        else:
            print(f"❌ Update command failed: {result.error}")

    # 3. Process message through all applicable handlers
    print("\n3. Processing event through all applicable handlers:")

    user_event = UserCreatedEvent(
        user_id="event_user",
        name="Event User",
        email="event@example.com",
        timestamp=time.time(),
    )

    results = chain.process_all(user_event)
    print(f"📊 Event processed by {len(results)} handlers")

    for i, result in enumerate(results, 1):
        if result.is_success:
            print(f"   ✅ Handler {i}: Success")
        else:
            print(f"   ❌ Handler {i}: {result.error}")


def demonstrate_function_handlers() -> None:
    """Demonstrate function-based handler creation."""
    print("\n" + "=" * 80)
    print("🔧 FUNCTION HANDLERS - FUNCTIONAL STYLE")
    print("=" * 80)

    # 1. Create simple function handlers
    print("\n1. Creating function-based handlers:")

    def process_simple_message(message: str) -> FlextResult[str]:
        """Simple message processing function."""
        if not message:
            return FlextResult.fail("Empty message")
        return FlextResult.ok(f"Processed: {message.upper()}")

    def process_number(number: int) -> FlextResult[int]:
        """Number processing function."""
        if number < 0:
            return FlextResult.fail("Negative numbers not allowed")
        return FlextResult.ok(number * 2)

    # Create handlers from functions
    message_handler = FlextHandlers.create_function_handler(process_simple_message)
    number_handler = FlextHandlers.create_function_handler(process_number)

    print("✅ Function handlers created")

    # 2. Use function handlers directly (avoiding type checking issues)
    print("\n2. Using function-based handlers:")

    # Test message handler - use handle() directly to avoid type checking
    try:
        result = message_handler.handle("hello world")
        if result.is_success:
            print(f"✅ Message handler result: {result.data}")
        else:
            print(f"❌ Message handler failed: {result.error}")
    except Exception as e:
        print(f"❌ Message handler error: {e}")

    # Test with empty message
    try:
        result = message_handler.handle("")
        if result.is_success:
            print(f"✅ Empty message result: {result.data}")
        else:
            print(f"❌ Empty message failed (expected): {result.error}")
    except Exception as e:
        print(f"❌ Empty message error: {e}")

    # Test number handler
    try:
        result = number_handler.handle(42)
        if result.is_success:
            print(f"✅ Number handler result: {result.data}")
        else:
            print(f"❌ Number handler failed: {result.error}")
    except Exception as e:
        print(f"❌ Number handler error: {e}")

    # Test with negative number
    try:
        result = number_handler.handle(-5)
        if result.is_success:
            print(f"✅ Negative number result: {result.data}")
        else:
            print(f"❌ Negative number failed (expected): {result.error}")
    except Exception as e:
        print(f"❌ Negative number error: {e}")

    # 3. Complex function handler with business logic
    print("\n3. Complex function handler:")

    def process_order_total(order_data: dict[str, Any]) -> FlextResult[dict[str, Any]]:
        """Complex order processing function."""
        if not order_data.get("items"):
            return FlextResult.fail("Order must have items")

        items = order_data["items"]
        if not isinstance(items, list) or len(items) == 0:
            return FlextResult.fail("Order items must be a non-empty list")

        # Calculate total (simulate item prices)
        item_prices = {"item1": 10.0, "item2": 15.0, "item3": 20.0}
        total = sum(item_prices.get(item, 5.0) for item in items)

        # Apply discount for large orders
        if len(items) >= 3:
            total *= 0.9  # 10% discount

        result = {
            "order_id": order_data.get("order_id", "unknown"),
            "items": items,
            "total": round(total, 2),
            "discount_applied": len(items) >= 3,
        }

        return FlextResult.ok(result)

    order_handler = FlextHandlers.create_function_handler(process_order_total)

    # Test complex handler - use handle() directly
    order_data = {
        "order_id": "ORD001",
        "items": ["item1", "item2", "item3"],
    }

    try:
        result = order_handler.handle(order_data)
        if result.is_success:
            order_result = result.data
            print(f"✅ Order processed: {order_result}")
        else:
            print(f"❌ Order processing failed: {result.error}")
    except Exception as e:
        print(f"❌ Order processing error: {e}")

    # 4. Function handler metrics
    print("\n4. Function handler metrics:")
    try:
        message_metrics = message_handler.get_metrics()
        number_metrics = number_handler.get_metrics()
        order_metrics = order_handler.get_metrics()

        print("📊 Message Handler:")
        print(f"   Messages handled: {message_metrics['messages_handled']}")
        print(f"   Success rate: {message_metrics['success_rate']:.2%}")

        print("📊 Number Handler:")
        print(f"   Messages handled: {number_metrics['messages_handled']}")
        print(f"   Success rate: {number_metrics['success_rate']:.2%}")

        print("📊 Order Handler:")
        print(f"   Messages handled: {order_metrics['messages_handled']}")
        print(f"   Success rate: {order_metrics['success_rate']:.2%}")
    except Exception as e:
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
            "   with CQRS patterns, service location, and flexible handler composition!",
        )

    except Exception as e:
        print(f"\n❌ Error during FlextHandlers demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
