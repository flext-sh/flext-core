# Application Layer API Reference

This section covers the application layer classes that handle command/query processing, event handling, and application coordination.

## Command and Query Responsibility Segregation (CQRS)

### FlextBus - Message Bus

Central message bus for handling commands, queries, and events with middleware support.

```python
from flext_core import FlextBus, FlextResult

# Create bus instance
bus = FlextBus()

# Register command handler
@bus.command_handler
class CreateUserHandler:
    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        # Command processing logic
        user = User(id=command.user_id, name=command.name)
        return FlextResult[User].ok(user)

# Register query handler
@bus.query_handler
class GetUserHandler:
    def handle(self, query: GetUserQuery) -> FlextResult[User]:
        # Query processing logic
        user = self.user_repository.get_by_id(query.user_id)
        if not user:
            return FlextResult[User].fail("User not found")
        return FlextResult[User].ok(user)

# Send command
command = CreateUserCommand(user_id="user_123", name="Alice")
result = bus.send_command(command)

# Send query
query = GetUserQuery(user_id="user_123")
result = bus.send_query(query)
```

**Key Features:**

- Command/Query separation
- Middleware pipeline support
- Type-safe message handling
- Error propagation and context

### FlextDispatcher - Unified Dispatcher

Unified facade for command and query dispatching with registry management.

```python
from flext_core import FlextDispatcher

# Get dispatcher instance
dispatcher = FlextDispatcher.get_global()

# Register handlers
dispatcher.register_command(CreateUserCommand, CreateUserHandler)
dispatcher.register_query(GetUserQuery, GetUserHandler)

# Dispatch commands and queries
result = dispatcher.dispatch_command(CreateUserCommand(user_id="user_123"))
result = dispatcher.dispatch_query(GetUserQuery(user_id="user_123"))
```

**Key Methods:**

- `dispatch_command(command)` - Dispatch command and return result
- `dispatch_query(query)` - Dispatch query and return result
- `register_command(command_type, handler)` - Register command handler
- `register_query(query_type, handler)` - Register query handler

### FlextHandlers - Handler Registry

Registry system for managing command and query handlers.

```python
from flext_core import FlextHandlers

# Create handlers registry
handlers = FlextHandlers()

# Register handlers
handlers.register(CreateUserCommand, CreateUserHandler)
handlers.register(GetUserQuery, GetUserHandler)

# Get handler for command
handler = handlers.get_command_handler(CreateUserCommand)
if handler:
    result = handler.handle(CreateUserCommand(user_id="user_123"))
```

## Event Processing

### FlextProcessors - Event Processing

Orchestration layer for processing domain events and integration events.

```python
from flext_core import FlextProcessors

# Create processor registry
processors = FlextProcessors()

# Register event processor
@processors.event_processor
class UserEventProcessor:
    async def process(self, event: UserCreatedEvent) -> FlextResult[None]:
        # Send welcome email
        await self.email_service.send_welcome_email(event.email)

        # Update analytics
        await self.analytics.track_user_created(event.user_id)

        return FlextResult[None].ok(None)

# Process events
events = [UserCreatedEvent(user_id="user_123", email="alice@example.com")]
await processors.process_events(events)
```

### FlextRegistry - Registry Management

Central registry for handlers, processors, and other application components.

```python
from flext_core import FlextRegistry

# Get global registry
registry = FlextRegistry.get_global()

# Register components
registry.register_handler("create_user", CreateUserHandler)
registry.register_processor("user_events", UserEventProcessor)

# Retrieve components
handler = registry.get_handler("create_user")
processor = registry.get_processor("user_events")
```

## Middleware

### Middleware Pipeline

Middleware support for cross-cutting concerns like logging, validation, and error handling.

```python
from flext_core import Middleware

class LoggingMiddleware(Middleware):
    """Middleware for request/response logging."""

    async def process(self, request, next_middleware):
        # Log incoming request
        self.logger.info(f"Processing {type(request).__name__}")

        # Process request
        response = await next_middleware(request)

        # Log response
        self.logger.info(f"Completed {type(response).__name__}")

        return response

class ValidationMiddleware(Middleware):
    """Middleware for request validation."""

    async def process(self, request, next_middleware):
        # Validate request
        if not self._is_valid_request(request):
            return FlextResult.fail("Invalid request")

        # Continue processing
        return await next_middleware(request)

# Configure bus with middleware
bus = FlextBus()
bus.add_middleware(LoggingMiddleware())
bus.add_middleware(ValidationMiddleware())
```

## Quality Metrics

| Module          | Coverage | Status       | Description                    |
| --------------- | -------- | ------------ | ------------------------------ |
| `bus.py`        | 94%      | ✅ Stable    | Message bus with middleware    |
| `handlers.py`   | 66%      | 🔄 Improving | Handler registry system        |
| `dispatcher.py` | 45%      | 🔄 Improving | Unified dispatcher facade      |
| `registry.py`   | 91%      | ✅ Stable    | Registry management            |
| `processors.py` | 56%      | 🔄 Improving | Event processing orchestration |

## Usage Examples

### Complete CQRS Implementation

```python
from flext_core import FlextBus, FlextDispatcher, FlextResult
from abc import ABC, abstractmethod

# Commands
class CreateUserCommand:
    def __init__(self, user_id: str, name: str, email: str):
        self.user_id = user_id
        self.name = name
        self.email = email

class UpdateUserCommand:
    def __init__(self, user_id: str, name: str = None, email: str = None):
        self.user_id = user_id
        self.name = name
        self.email = email

# Queries
class GetUserQuery:
    def __init__(self, user_id: str):
        self.user_id = user_id

class ListUsersQuery:
    def __init__(self, limit: int = 10, offset: int = 0):
        self.limit = limit
        self.offset = offset

# Domain Events
class UserCreatedEvent:
    def __init__(self, user_id: str, email: str):
        self.user_id = user_id
        self.email = email

# Setup application
bus = FlextBus()

# Command Handlers
@bus.command_handler
class CreateUserHandler:
    def __init__(self, user_repository, email_service):
        self.user_repository = user_repository
        self.email_service = email_service

    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        # Check if user exists
        existing = self.user_repository.get_by_id(command.user_id)
        if existing:
            return FlextResult[User].fail("User already exists")

        # Create user
        user = User(
            id=command.user_id,
            name=command.name,
            email=command.email
        )

        # Save user
        self.user_repository.save(user)

        # Send welcome email
        self.email_service.send_welcome_email(user.email)

        return FlextResult[User].ok(user)

@bus.command_handler
class UpdateUserHandler:
    def handle(self, command: UpdateUserCommand) -> FlextResult[User]:
        user = self.user_repository.get_by_id(command.user_id)
        if not user:
            return FlextResult[User].fail("User not found")

        # Update fields
        if command.name:
            user.name = command.name
        if command.email:
            user.email = command.email

        # Save updated user
        self.user_repository.save(user)

        return FlextResult[User].ok(user)

# Query Handlers
@bus.query_handler
class GetUserHandler:
    def __init__(self, user_repository):
        self.user_repository = user_repository

    def handle(self, query: GetUserQuery) -> FlextResult[User]:
        user = self.user_repository.get_by_id(query.user_id)
        if not user:
            return FlextResult[User].fail("User not found")
        return FlextResult[User].ok(user)

@bus.query_handler
class ListUsersHandler:
    def handle(self, query: ListUsersQuery) -> FlextResult[List[User]]:
        users = self.user_repository.list(limit=query.limit, offset=query.offset)
        return FlextResult[List[User]].ok(users)

# Usage examples
async def main():
    # Create user
    create_cmd = CreateUserCommand("user_123", "Alice", "alice@example.com")
    result = await bus.send_command(create_cmd)

    if result.is_success:
        user = result.unwrap()
        print(f"✅ Created user: {user.name}")

    # Get user
    get_query = GetUserQuery("user_123")
    result = await bus.send_query(get_query)

    if result.is_success:
        user = result.unwrap()
        print(f"📋 Found user: {user.name}")

    # List users
    list_query = ListUsersQuery(limit=5)
    result = await bus.send_query(list_query)

    if result.is_success:
        users = result.unwrap()
        print(f"📋 Found {len(users)} users")

# Run application
asyncio.run(main())
```

This application layer provides a robust foundation for implementing CQRS patterns with proper separation of concerns, middleware support, and comprehensive error handling.
