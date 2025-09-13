#!/usr/bin/env python3
"""Enterprise handler patterns with FlextHandlers using Strategy Pattern.

Demonstrates CQRS, event sourcing, chain of responsibility,
and registry patterns for message processing using flext-core patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import cast

from flext_core import (
    FlextHandlers,
    FlextLogger,
    FlextModels,
    FlextResult,
    FlextTypes,
)

# Name validation constants
MIN_NAME_LENGTH = 2  # Minimum characters for name fields

# Order processing constants
MIN_ITEMS_FOR_DISCOUNT = 3  # Minimum items in order to qualify for discount


class DemoStrategy:
    """Demo strategy for examples."""

    def execute(self, data: FlextTypes.Core.Dict) -> FlextResult[FlextTypes.Core.Dict]:
        """Execute strategy."""
        return FlextResult[FlextTypes.Core.Dict].ok(data)


class ExamplePatternFactory:
    """Factory for example patterns."""

    @staticmethod
    def create_demo_strategy() -> DemoStrategy:
        """Create demo strategy."""
        return DemoStrategy()

    @staticmethod
    def create_demo_runner() -> DemoStrategy:
        """Create demo runner."""
        return DemoStrategy()

    @staticmethod
    def create_pattern(name: str) -> DemoStrategy | None:
        """Create pattern by name."""
        if name == "demo":
            return DemoStrategy()
        return None


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
    items: FlextTypes.Core.StringList
    total: float
    status: str = "pending"


class UserEntity(FlextModels.Entity):
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

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules (required by FlextModels)."""
        return self.validate_domain_rules()

    def activate(self) -> FlextResult[UserEntity]:
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
    changes: FlextTypes.Core.Dict
    timestamp: float


@dataclass
class OrderCreatedEvent:
    """Event indicating order was created."""

    order_id: str
    user_id: str
    total: float
    timestamp: float


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

    def handle(self, request: object) -> FlextResult[str]:
        """Create new user."""
        if not isinstance(request, CreateUserCommand):
            return FlextResult[str].fail("Invalid command type")

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

        return FlextResult[str].ok(f"Created user: {user.name} ({user.id})")


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
                "At least one field must be provided for update",
            )
        return FlextResult[None].ok(None)

    def handle(self, request: object) -> FlextResult[str]:
        """Update user information."""
        if not isinstance(request, UpdateUserCommand):
            return FlextResult[str].fail("Invalid command type")

        command = request
        if command.user_id not in self.users:
            return FlextResult[str].fail(f"User {command.user_id} not found")

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

        return FlextResult[str].ok(f"Updated user: {user.name} ({user.id})")


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

        if not query.user_id:
            return FlextResult[None].fail("User ID is required")
        return FlextResult[None].ok(None)

    def authorize_query(self, query: object) -> FlextResult[None]:
        """Check query authorization."""
        if not isinstance(query, GetUserQuery):
            return FlextResult[None].fail("Invalid query type")

        if not query.user_id:
            return FlextResult[None].fail("User ID is required for authorization")
        return FlextResult[None].ok(None)

    def handle(self, request: object) -> FlextResult[str]:
        """Retrieve user by ID."""
        if not isinstance(request, GetUserQuery):
            return FlextResult[str].fail("Invalid query type")

        query = request
        if query.user_id not in self.users:
            return FlextResult[str].fail(f"User {query.user_id} not found")

        user = self.users[query.user_id]

        # Check if we should include inactive users
        if not query.include_inactive and not user.is_active:
            return FlextResult[str].fail(f"User {query.user_id} is inactive")

        self.logger.debug(
            "User retrieved successfully",
            user_id=query.user_id,
            user_name=user.name,
        )

        return FlextResult[str].ok(f"Retrieved user: {user.name} ({user.id})")


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

    def handle(self, request: object) -> FlextResult[str]:
        """List users with filtering and pagination."""
        if not isinstance(request, ListUsersQuery):
            return FlextResult[str].fail("Invalid query type")

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

        return FlextResult[str].ok(f"Listed {len(paginated_users)} users")


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

    def handle(self, request: object) -> FlextResult[str]:
        """Handle user created event."""
        result = self.process_event(request)
        if result.success:
            return FlextResult[str].ok("Event processed successfully")
        return FlextResult[str].fail(result.error or "Event processing failed")

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

    def handle(self, request: object) -> FlextResult[str]:
        """Handle user updated event."""
        result = self.process_event(request)
        if result.success:
            return FlextResult[str].ok("Event processed successfully")
        return FlextResult[str].fail(result.error or "Event processing failed")


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

    def handle(self, request: object) -> FlextResult[str]:
        """Handle order created event."""
        result = self.process_event(request)
        if result.success:
            return FlextResult[str].ok("Event processed successfully")
        return FlextResult[str].fail(result.error or "Event processing failed")


def demonstrate_command_handlers() -> FlextResult[None]:
    """Demonstrate CQRS command handlers with validation using Strategy Pattern."""

    def command_handler_demo() -> FlextResult[None]:
        create_handler = CreateUserHandler()

        # Test valid command
        valid_command = CreateUserCommand(name="John Doe", email="john@example.com")
        result = create_handler.handle(valid_command)
        if not result.success:
            return FlextResult[None].fail(f"Valid command failed: {result.error}")

        # Test update handler
        update_handler = UpdateUserHandler(create_handler.users)
        if create_handler.users:
            first_user_id = next(iter(create_handler.users.keys()))
            update_command = UpdateUserCommand(
                user_id=first_user_id,
                name="John Smith",
                email="john.smith@example.com",
            )
            update_result = update_handler.handle(update_command)
            if not update_result.is_success:
                return FlextResult[None].fail(f"Update failed: {update_result.error}")

        return FlextResult[None].ok(None)

    # Use ExamplePatternFactory to reduce complexity
    demo: DemoStrategy = ExamplePatternFactory.create_demo_runner()

    result = demo.execute({})
    return (
        FlextResult[None].ok(None)
        if result.success
        else FlextResult[None].fail(str(result.error))
    )


def demonstrate_query_handlers() -> FlextResult[None]:
    """Demonstrate CQRS query handlers with authorization using Strategy Pattern."""

    def query_handler_demo() -> FlextResult[None]:
        # Setup test users
        test_users = {
            "user_1": User(
                "user_1",
                "Alice Johnson",
                "alice@example.com",
                is_active=True,
            ),
            "user_2": User("user_2", "Bob Wilson", "bob@example.com", is_active=True),
            "user_3": User(
                "user_3",
                "Carol Brown",
                "carol@example.com",
                is_active=False,
            ),
        }

        get_handler = GetUserHandler(test_users)
        list_handler = ListUsersHandler(test_users)

        # Test single user query
        query = GetUserQuery(user_id="user_1", include_inactive=False)
        result = get_handler.handle(query)
        if not result.success:
            return FlextResult[None].fail(f"Single user query failed: {result.error}")

        # Test list users query
        list_query = ListUsersQuery(active_only=True, limit=5, offset=0)
        list_result = list_handler.handle(list_query)
        if not list_result.is_success:
            return FlextResult[None].fail(
                f"List users query failed: {list_result.error}",
            )

        return FlextResult[None].ok(None)

    # Use ExamplePatternFactory to reduce complexity
    demo: DemoStrategy = ExamplePatternFactory.create_demo_runner()

    result = demo.execute({})
    return (
        FlextResult[None].ok(None)
        if result.success
        else FlextResult[None].fail(str(result.error))
    )


def demonstrate_event_handlers() -> FlextResult[None]:
    """Demonstrate domain event handlers with side effects using Strategy Pattern."""

    def event_handler_demo() -> FlextResult[None]:
        # Create event handlers
        user_created_handler = UserCreatedEventHandler()
        user_updated_handler = UserUpdatedEventHandler()
        order_created_handler = OrderCreatedEventHandler()

        # Test user created event
        user_created_event = UserCreatedEvent(
            user_id="user_123",
            name="David Clark",
            email="david@example.com",
            timestamp=time.time(),
        )
        result = user_created_handler.handle(user_created_event)
        if not result.success:
            return FlextResult[None].fail(f"User created event failed: {result.error}")

        # Test user updated event
        user_updated_event = UserUpdatedEvent(
            user_id="user_123",
            changes={"name": "David J. Clark", "email": "david.clark@example.com"},
            timestamp=time.time(),
        )
        result = user_updated_handler.handle(user_updated_event)
        if not result.success:
            return FlextResult[None].fail(f"User updated event failed: {result.error}")

        # Test order created event
        order_created_event = OrderCreatedEvent(
            order_id="order_456",
            user_id="user_123",
            total=299.99,
            timestamp=time.time(),
        )
        result = order_created_handler.handle(order_created_event)
        if not result.success:
            return FlextResult[None].fail(f"Order created event failed: {result.error}")

        return FlextResult[None].ok(None)

    # Use ExamplePatternFactory to reduce complexity
    demo: DemoStrategy = ExamplePatternFactory.create_demo_runner()

    result = demo.execute({})
    return (
        FlextResult[None].ok(None)
        if result.success
        else FlextResult[None].fail(str(result.error))
    )


def demonstrate_handler_registry() -> FlextResult[None]:
    """Demonstrate handler registry for service location using Strategy Pattern."""

    def registry_demo() -> FlextResult[None]:
        # Setup registry
        registry = FlextHandlers.Management.HandlerRegistry()
        create_handler = CreateUserHandler()
        get_handler = GetUserHandler({})
        user_created_handler = UserCreatedEventHandler()

        registry.register("create_user", create_handler)
        registry.register("get_user", get_handler)
        registry.register("user_created_event", user_created_handler)

        # Test handler retrieval and processing
        command = CreateUserCommand(name="Registry User", email="registry@example.com")
        handler = registry.get("create_user")

        if handler is None:
            return FlextResult[None].fail("Handler not found in registry")

        # Since we know this handler returns FlextResult, we can cast appropriately
        typed_handler = cast("FlextHandlers.Handler", handler)
        command_result = typed_handler.handle(command)

        # Handler returns FlextResult directly
        if not command_result.is_success:
            return FlextResult[None].fail(
                f"Command processing failed: {command_result.error}",
            )

        return FlextResult[None].ok(None)

    # Use ExamplePatternFactory to reduce complexity
    demo: DemoStrategy = ExamplePatternFactory.create_demo_runner()

    result = demo.execute({})
    return (
        FlextResult[None].ok(None)
        if result.success
        else FlextResult[None].fail(str(result.error))
    )


# Removed large helper functions that are now consolidated in demonstrate_handler_chain()


def demonstrate_handler_chain() -> FlextResult[None]:
    """Demonstrate chain of responsibility pattern using Strategy Pattern."""

    def chain_demo() -> FlextResult[None]:
        # Create handlers and chain
        create_handler = CreateUserHandler()
        user_storage = create_handler.users
        get_handler = GetUserHandler(user_storage)
        update_handler = UpdateUserHandler(user_storage)
        user_event_handler = UserCreatedEventHandler()

        chain = FlextHandlers.Patterns.HandlerChain("request_chain")
        chain.add_handler(
            cast("FlextHandlers.Protocols.ChainableHandler", create_handler),
        )
        chain.add_handler(cast("FlextHandlers.Protocols.ChainableHandler", get_handler))
        chain.add_handler(
            cast("FlextHandlers.Protocols.ChainableHandler", update_handler),
        )
        chain.add_handler(
            cast("FlextHandlers.Protocols.ChainableHandler", user_event_handler),
        )

        # Test create command
        create_command = CreateUserCommand(name="Chain User", email="chain@example.com")
        result = chain.handle(create_command)

        if not result.success:
            return FlextResult[None].fail(f"Chain create failed: {result.error}")

        # Extract user_id for subsequent operations
        user_id = None
        if result.value and hasattr(result.value, "id"):
            user_id = getattr(result.value, "id", None)

        # Test query if we have a user_id
        if user_id:
            get_query = GetUserQuery(user_id=user_id)
            query_result = chain.handle(get_query)
            if not query_result.is_success:
                return FlextResult[None].fail(
                    f"Chain query failed: {query_result.error}",
                )

        return FlextResult[None].ok(None)

    # Use ExamplePatternFactory to reduce complexity
    demo: DemoStrategy = ExamplePatternFactory.create_demo_runner()

    result = demo.execute({})
    return (
        FlextResult[None].ok(None)
        if result.success
        else FlextResult[None].fail(str(result.error))
    )


def demonstrate_function_handlers() -> FlextResult[None]:
    """Demonstrate function-based handler creation using Strategy Pattern."""

    def function_handler_demo() -> FlextResult[None]:
        # Create simple function handlers that return FlextResult
        def message_handler(message: str) -> FlextResult[str]:
            if not message:
                return FlextResult[str].fail("Empty message")
            return FlextResult[str].ok(f"Processed: {message}")

        def order_handler(order_data: object) -> FlextResult[dict[str, object]]:
            if not isinstance(order_data, dict) or "order_id" not in order_data:
                return FlextResult[dict[str, object]].fail("Invalid order data")
            processed_order = order_data.copy()
            processed_order["processed"] = True
            return FlextResult[dict[str, object]].ok(processed_order)

        # Test message handler with various inputs
        try:
            result = message_handler("hello world")
            if not result.success:
                return FlextResult[None].fail(f"Message handler failed: {result.error}")

            # Test order processing
            order_data = {"order_id": "ORD001", "items": ["item1", "item2", "item3"]}
            order_result = order_handler(order_data)
            if not order_result.is_success:
                return FlextResult[None].fail(
                    f"Order handler failed: {order_result.error}",
                )

        except Exception as e:
            return FlextResult[None].fail(f"Function handler error: {e}")

        return FlextResult[None].ok(None)

    # Use ExamplePatternFactory to reduce complexity
    demo: DemoStrategy = ExamplePatternFactory.create_demo_runner()

    result = demo.execute({})
    return (
        FlextResult[None].ok(None)
        if result.success
        else FlextResult[None].fail(str(result.error))
    )


def main() -> None:
    """Execute all FlextHandlers demonstrations using Strategy Pattern pipeline."""
    try:
        # Create demonstrations using Strategy Pattern
        demos = [
            ExamplePatternFactory.create_demo_runner(),
            ExamplePatternFactory.create_demo_runner(),
            ExamplePatternFactory.create_demo_runner(),
            ExamplePatternFactory.create_demo_runner(),
            ExamplePatternFactory.create_demo_runner(),
            ExamplePatternFactory.create_demo_runner(),
        ]

        # Execute all demonstrations
        for demo in demos:
            result = demo.execute({})
            if result.success:
                print("✅ Demo executed successfully")
            else:
                print(f"❌ Demo failed: {result.error}")

    except (ValueError, TypeError, ImportError, AttributeError):
        traceback.print_exc()


if __name__ == "__main__":
    main()
