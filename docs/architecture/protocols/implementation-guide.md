# FlextProtocols Implementation Guide

**Version**: 0.9.0  
**Module**: `flext_core.protocols`  
**Target Audience**: Senior Developers, Software Architects, Platform Engineers

## Quick Start

This guide provides step-by-step implementation patterns for integrating FlextProtocols across FLEXT ecosystem services, from basic protocol adoption to enterprise-grade contract architecture.

**Prerequisite**: Ensure `flext-core` is installed and available in your environment.

---

## 🚀 Basic Implementation

### Step 1: Import and Basic Setup

```python
from flext_core import FlextProtocols, FlextResult
from flext_core.protocols import FlextProtocolsConfig
import time
from typing import Protocol, runtime_checkable

# Basic protocol configuration
config_result = FlextProtocolsConfig.configure_protocols_system({
    "environment": "development",
    "protocol_level": "loose",
    "enable_runtime_checking": True,
    "log_level": "DEBUG"
})

if not config_result.success:
    raise Exception(f"Failed to configure protocols: {config_result.error}")

print("✅ FlextProtocols configured successfully")
```

### Step 2: Foundation Layer Implementation

```python
# Implement basic Foundation protocols

# Type-safe validator
class EmailValidator(FlextProtocols.Foundation.Validator[str]):
    """Email validation using Foundation.Validator protocol."""

    def validate(self, data: str) -> object:
        if not isinstance(data, str):
            return FlextResult[None].fail("Email must be a string")

        if "@" not in data or "." not in data:
            return FlextResult[None].fail("Invalid email format")

        return FlextResult[None].ok(None)

# Factory pattern implementation
class UserFactory(FlextProtocols.Foundation.Factory[dict]):
    """User creation factory using Foundation.Factory protocol."""

    def create(self, **kwargs: object) -> object:
        name = kwargs.get("name")
        email = kwargs.get("email")

        if not name or not email:
            return FlextResult[dict].fail("Name and email are required")

        # Validate email
        validator = EmailValidator()
        email_validation = validator.validate(str(email))

        if not email_validation.success:
            return FlextResult[dict].fail(f"Invalid email: {email_validation.error}")

        user = {
            "id": f"user_{int(time.time())}",
            "name": str(name),
            "email": str(email),
            "created_at": time.time()
        }

        return FlextResult[dict].ok(user)

# Serialization protocol
class User(FlextProtocols.Foundation.HasToDict):
    """User model with serialization protocol."""

    def __init__(self, user_id: str, name: str, email: str):
        self.id = user_id
        self.name = name
        self.email = email
        self.created_at = time.time()

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "created_at": self.created_at
        }

# Test basic implementations
def test_foundation_protocols():
    """Test Foundation layer protocols."""

    # Test factory
    factory = UserFactory()
    user_result = factory.create(name="John Doe", email="john@example.com")

    if user_result.success:
        user_data = user_result.value
        print(f"✅ User created: {user_data['id']}")

        # Create User object and test serialization
        user_obj = User(user_data["id"], user_data["name"], user_data["email"])

        # Test runtime checking
        if isinstance(user_obj, FlextProtocols.Foundation.HasToDict):
            serialized = user_obj.to_dict()
            print(f"✅ User serialized: {serialized}")
        else:
            print("❌ User does not implement HasToDict protocol")
    else:
        print(f"❌ User creation failed: {user_result.error}")

test_foundation_protocols()
```

### Step 3: Domain Layer Implementation

```python
# Domain service implementation
class UserDomainService(FlextProtocols.Domain.Service):
    """User domain service following Domain.Service protocol."""

    def __init__(self):
        self._running = False
        self._users: dict[str, dict] = {}
        self.user_factory = UserFactory()

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Callable interface for service operations."""
        operation = kwargs.get("operation")

        if operation == "create_user":
            return self.create_user(kwargs.get("name"), kwargs.get("email"))
        elif operation == "get_user":
            return self.get_user(kwargs.get("user_id"))
        else:
            return FlextResult[dict].fail(f"Unknown operation: {operation}")

    def start(self) -> object:
        """Start the domain service."""
        if self._running:
            return FlextResult[None].fail("Service already running")

        self._running = True
        print("🚀 User domain service started")
        return FlextResult[None].ok(None)

    def stop(self) -> object:
        """Stop the domain service."""
        if not self._running:
            return FlextResult[None].fail("Service not running")

        self._running = False
        print("🛑 User domain service stopped")
        return FlextResult[None].ok(None)

    def health_check(self) -> object:
        """Perform health check."""
        if not self._running:
            return FlextResult[dict].fail("Service not running")

        health_data = {
            "status": "healthy",
            "running": self._running,
            "user_count": len(self._users),
            "timestamp": time.time()
        }

        return FlextResult[dict].ok(health_data)

    def create_user(self, name: object, email: object) -> object:
        """Create a new user."""
        if not self._running:
            return FlextResult[dict].fail("Service not running")

        # Use factory to create user
        create_result = self.user_factory.create(name=name, email=email)

        if create_result.success:
            user_data = create_result.value
            self._users[user_data["id"]] = user_data
            return create_result

        return create_result

    def get_user(self, user_id: object) -> object:
        """Get user by ID."""
        if not self._running:
            return FlextResult[dict].fail("Service not running")

        if not isinstance(user_id, str):
            return FlextResult[dict].fail("User ID must be a string")

        if user_id in self._users:
            return FlextResult[dict].ok(self._users[user_id])
        else:
            return FlextResult[dict].fail(f"User not found: {user_id}")

# Repository pattern implementation
class UserRepository(FlextProtocols.Domain.Repository[dict]):
    """User repository following Domain.Repository protocol."""

    def __init__(self):
        self._storage: dict[str, dict] = {}

    def get_by_id(self, entity_id: str) -> object:
        """Get user by ID."""
        if entity_id in self._storage:
            return FlextResult[dict].ok(self._storage[entity_id])
        else:
            return FlextResult[dict].fail(f"User not found: {entity_id}")

    def save(self, entity: dict) -> object:
        """Save user entity."""
        if not isinstance(entity, dict) or "id" not in entity:
            return FlextResult[dict].fail("Invalid entity: missing 'id' field")

        entity_id = entity["id"]
        self._storage[entity_id] = entity.copy()

        return FlextResult[dict].ok(entity)

    def delete(self, entity_id: str) -> object:
        """Delete user by ID."""
        if entity_id in self._storage:
            deleted_entity = self._storage.pop(entity_id)
            return FlextResult[dict].ok(deleted_entity)
        else:
            return FlextResult[dict].fail(f"User not found: {entity_id}")

    def find_all(self) -> object:
        """Find all users."""
        return FlextResult[list].ok(list(self._storage.values()))

# Test domain layer
def test_domain_protocols():
    """Test Domain layer protocols."""

    # Test domain service
    service = UserDomainService()

    # Start service
    start_result = service.start()
    if start_result.success:
        print("✅ Domain service started")

        # Test service operations
        create_result = service.create_user("Alice Smith", "alice@example.com")
        if create_result.success:
            user = create_result.value
            print(f"✅ User created via service: {user['id']}")

            # Test callable interface
            get_result = service(operation="get_user", user_id=user["id"])
            if get_result.success:
                print(f"✅ User retrieved via callable: {get_result.value['name']}")

        # Test health check
        health = service.health_check()
        if health.success:
            print(f"✅ Service health: {health.value}")

        # Stop service
        service.stop()

    # Test repository
    repo = UserRepository()

    # Create and save user
    factory = UserFactory()
    user_result = factory.create(name="Bob Johnson", email="bob@example.com")

    if user_result.success:
        user = user_result.value

        # Save to repository
        save_result = repo.save(user)
        if save_result.success:
            print(f"✅ User saved to repository: {user['id']}")

            # Retrieve from repository
            get_result = repo.get_by_id(user["id"])
            if get_result.success:
                print(f"✅ User retrieved from repository: {get_result.value['name']}")

test_domain_protocols()
```

---

## 🏗️ Enterprise Implementation

### Step 1: Application Layer with CQRS

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Command and Query objects
@dataclass
class CreateUserCommand:
    name: str
    email: str
    department: str = "general"

@dataclass
class GetUserQuery:
    user_id: str

@dataclass
class UserCreatedEvent:
    user_id: str
    name: str
    email: str
    timestamp: float

# Application layer handlers
class CreateUserHandler(FlextProtocols.Application.ValidatingHandler):
    """CQRS Command handler with validation."""

    def __init__(self,
                 user_service: FlextProtocols.Domain.Service,
                 user_repo: FlextProtocols.Domain.Repository):
        self.user_service = user_service
        self.user_repo = user_repo

    def handle(self, message: object) -> object:
        """Handle CreateUserCommand."""
        if not isinstance(message, CreateUserCommand):
            return FlextResult[dict].fail("Invalid message type")

        # Validate command
        validation_result = self.validate(message)
        if not validation_result.success:
            return validation_result

        # Create user through domain service
        create_result = self.user_service.create_user(message.name, message.email)

        if create_result.success:
            user = create_result.value

            # Save to repository
            save_result = self.user_repo.save(user)

            if save_result.success:
                # Create domain event
                event = UserCreatedEvent(
                    user_id=user["id"],
                    name=user["name"],
                    email=user["email"],
                    timestamp=time.time()
                )

                return FlextResult[dict].ok({
                    "user": user,
                    "event": event
                })
            else:
                return save_result
        else:
            return create_result

    def validate(self, message: object) -> object:
        """Validate CreateUserCommand."""
        if not isinstance(message, CreateUserCommand):
            return FlextResult[None].fail("Invalid message type")

        if not message.name or len(message.name.strip()) < 2:
            return FlextResult[None].fail("Name must be at least 2 characters")

        if not message.email or "@" not in message.email:
            return FlextResult[None].fail("Invalid email address")

        return FlextResult[None].ok(None)

    def can_handle(self, message_type: type) -> bool:
        """Check if handler can process message type."""
        return message_type == CreateUserCommand

class GetUserQueryHandler(FlextProtocols.Application.Handler[GetUserQuery, dict]):
    """CQRS Query handler."""

    def __init__(self, user_repo: FlextProtocols.Domain.Repository):
        self.user_repo = user_repo

    def __call__(self, input_data: GetUserQuery) -> object:
        """Process GetUserQuery."""
        return self.user_repo.get_by_id(input_data.user_id)

    def validate(self, data: GetUserQuery) -> object:
        """Validate GetUserQuery."""
        if not isinstance(data, GetUserQuery):
            return FlextResult[None].fail("Invalid query type")

        if not data.user_id:
            return FlextResult[None].fail("User ID is required")

        return FlextResult[None].ok(None)

# Transaction management
class DatabaseUnitOfWork(FlextProtocols.Application.UnitOfWork):
    """Unit of Work for transaction management."""

    def __init__(self):
        self._in_transaction = False
        self._transaction_data = {}
        self._rollback_data = {}

    def begin(self) -> object:
        """Begin transaction."""
        if self._in_transaction:
            return FlextResult[None].fail("Transaction already active")

        self._in_transaction = True
        self._transaction_data.clear()
        self._rollback_data.clear()

        print("📊 Transaction begun")
        return FlextResult[None].ok(None)

    def commit(self) -> object:
        """Commit transaction."""
        if not self._in_transaction:
            return FlextResult[None].fail("No active transaction")

        # Simulate committing transaction data
        print(f"✅ Transaction committed with {len(self._transaction_data)} operations")

        self._in_transaction = False
        self._transaction_data.clear()
        self._rollback_data.clear()

        return FlextResult[None].ok(None)

    def rollback(self) -> object:
        """Rollback transaction."""
        if not self._in_transaction:
            return FlextResult[None].fail("No active transaction")

        # Simulate rollback
        print(f"↩️  Transaction rolled back, {len(self._rollback_data)} operations undone")

        self._in_transaction = False
        self._transaction_data.clear()
        self._rollback_data.clear()

        return FlextResult[None].ok(None)

    def add_operation(self, operation: str, data: object):
        """Add operation to transaction."""
        if self._in_transaction:
            self._transaction_data[operation] = data

# Application service orchestrator
class UserApplicationService:
    """Application service orchestrating commands and queries."""

    def __init__(self,
                 command_handler: FlextProtocols.Application.ValidatingHandler,
                 query_handler: FlextProtocols.Application.Handler,
                 unit_of_work: FlextProtocols.Application.UnitOfWork):
        self.command_handler = command_handler
        self.query_handler = query_handler
        self.unit_of_work = unit_of_work

    def create_user_with_transaction(self, command: CreateUserCommand) -> FlextResult[dict]:
        """Create user within a transaction."""

        # Begin transaction
        begin_result = self.unit_of_work.begin()
        if not begin_result.success:
            return begin_result

        try:
            # Handle command
            result = self.command_handler.handle(command)

            if result.success:
                # Commit transaction
                commit_result = self.unit_of_work.commit()
                if not commit_result.success:
                    return commit_result

                print(f"✅ User creation transaction completed successfully")
                return result
            else:
                # Rollback on failure
                self.unit_of_work.rollback()
                return result

        except Exception as e:
            # Rollback on exception
            self.unit_of_work.rollback()
            return FlextResult[dict].fail(f"Transaction failed: {e}")

    def get_user(self, query: GetUserQuery) -> FlextResult[dict]:
        """Get user (no transaction needed for read)."""
        return self.query_handler(query)

# Test application layer
def test_application_protocols():
    """Test Application layer protocols."""

    # Setup dependencies
    user_service = UserDomainService()
    user_repo = UserRepository()

    # Start domain service
    user_service.start()

    # Setup application handlers
    command_handler = CreateUserHandler(user_service, user_repo)
    query_handler = GetUserQueryHandler(user_repo)
    unit_of_work = DatabaseUnitOfWork()

    # Setup application service
    app_service = UserApplicationService(command_handler, query_handler, unit_of_work)

    # Test command handling with transaction
    create_command = CreateUserCommand(
        name="Charlie Brown",
        email="charlie@example.com",
        department="engineering"
    )

    create_result = app_service.create_user_with_transaction(create_command)

    if create_result.success:
        user_data = create_result.value["user"]
        print(f"✅ User created with transaction: {user_data['id']}")

        # Test query handling
        get_query = GetUserQuery(user_id=user_data["id"])
        get_result = app_service.get_user(get_query)

        if get_result.success:
            retrieved_user = get_result.value
            print(f"✅ User retrieved: {retrieved_user['name']}")
        else:
            print(f"❌ Failed to retrieve user: {get_result.error}")
    else:
        print(f"❌ Failed to create user: {create_result.error}")

    # Cleanup
    user_service.stop()

test_application_protocols()
```

### Step 2: Infrastructure Layer Implementation

```python
# Connection management
class DatabaseConnection(FlextProtocols.Infrastructure.Connection):
    """Database connection following Infrastructure.Connection protocol."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connected = False
        self.connection = None

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Callable interface for executing operations."""
        operation = kwargs.get("operation")

        if operation == "query":
            return self.execute_query(kwargs.get("sql", ""))
        elif operation == "execute":
            return self.execute_command(kwargs.get("sql", ""))
        else:
            return FlextResult[object].fail(f"Unknown operation: {operation}")

    def test_connection(self) -> object:
        """Test database connection."""
        try:
            # Simulate connection test
            if "invalid" in self.connection_string:
                return FlextResult[None].fail("Invalid connection string")

            print(f"🔗 Database connection test successful")
            return FlextResult[None].ok(None)
        except Exception as e:
            return FlextResult[None].fail(f"Connection test failed: {e}")

    def get_connection_string(self) -> str:
        """Get connection string."""
        # Return sanitized connection string (hide password)
        return self.connection_string.replace(":password", ":***")

    def close_connection(self) -> object:
        """Close database connection."""
        if not self.connected:
            return FlextResult[None].fail("No active connection")

        self.connected = False
        self.connection = None
        print("🔌 Database connection closed")
        return FlextResult[None].ok(None)

    def connect(self) -> FlextResult[None]:
        """Connect to database."""
        if self.connected:
            return FlextResult[None].fail("Already connected")

        test_result = self.test_connection()
        if not test_result.success:
            return test_result

        self.connected = True
        self.connection = f"db_conn_{int(time.time())}"
        print(f"✅ Connected to database: {self.get_connection_string()}")
        return FlextResult[None].ok(None)

    def execute_query(self, sql: str) -> FlextResult[list]:
        """Execute query and return results."""
        if not self.connected:
            return FlextResult[list].fail("Not connected to database")

        # Simulate query execution
        print(f"🔍 Executing query: {sql[:50]}...")

        # Mock results
        results = [
            {"id": 1, "name": "Mock User 1"},
            {"id": 2, "name": "Mock User 2"}
        ]

        return FlextResult[list].ok(results)

# Configurable logger implementation
class StructuredLogger(FlextProtocols.Infrastructure.LoggerProtocol):
    """Structured logger following Infrastructure.LoggerProtocol."""

    def __init__(self, name: str, level: str = "INFO"):
        self.name = name
        self.level = level
        self.log_entries: list[dict] = []

    def _log(self, level: str, message: str, **kwargs: object) -> None:
        """Internal logging method."""
        entry = {
            "timestamp": time.time(),
            "level": level,
            "logger": self.name,
            "message": message,
            "context": kwargs
        }

        self.log_entries.append(entry)
        print(f"[{level}] {self.name}: {message} {kwargs}")

    def debug(self, message: str, **kwargs: object) -> None:
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs: object) -> None:
        """Log info message."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: object) -> None:
        """Log warning message."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: object) -> None:
        """Log error message."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs: object) -> None:
        """Log critical message."""
        self._log("CRITICAL", message, **kwargs)

    def exception(self, message: str, *, exc_info: bool = True, **kwargs: object) -> None:
        """Log exception message."""
        self._log("EXCEPTION", message, exc_info=exc_info, **kwargs)

# Configurable service
class ConfigurableUserService(FlextProtocols.Infrastructure.Configurable):
    """User service with configuration support."""

    def __init__(self, logger: FlextProtocols.Infrastructure.LoggerProtocol):
        self.logger = logger
        self.config: dict[str, object] = {}
        self.initialized = False

    def configure(self, config: dict[str, object]) -> object:
        """Configure service with settings."""
        try:
            self.config.update(config)

            # Validate required config
            required_keys = ["database_url", "max_connections"]
            missing_keys = [key for key in required_keys if key not in config]

            if missing_keys:
                error_msg = f"Missing required configuration keys: {missing_keys}"
                self.logger.error("Configuration failed", missing_keys=missing_keys)
                return FlextResult[None].fail(error_msg)

            self.initialized = True
            self.logger.info("Service configured successfully",
                           config_keys=list(config.keys()))

            return FlextResult[None].ok(None)

        except Exception as e:
            self.logger.exception("Configuration error")
            return FlextResult[None].fail(f"Configuration failed: {e}")

    def get_config(self) -> dict[str, object]:
        """Get current configuration."""
        return self.config.copy()

# Test infrastructure layer
def test_infrastructure_protocols():
    """Test Infrastructure layer protocols."""

    # Test database connection
    db_conn = DatabaseConnection("postgresql://user:password@localhost:5432/testdb")

    connection_result = db_conn.connect()
    if connection_result.success:
        print("✅ Database connected successfully")

        # Test callable interface
        query_result = db_conn(operation="query", sql="SELECT * FROM users")
        if query_result.success:
            results = query_result.value
            print(f"✅ Query executed, {len(results)} results")

        # Test connection info
        print(f"📋 Connection string: {db_conn.get_connection_string()}")

        # Close connection
        db_conn.close_connection()

    # Test logger
    logger = StructuredLogger("test-service", "DEBUG")
    logger.info("Service starting", version="1.0.0", env="development")
    logger.warning("This is a test warning", component="user-service")
    logger.error("Simulated error", error_code=500, user_id="12345")

    print(f"📝 Logger created {len(logger.log_entries)} entries")

    # Test configurable service
    configurable_service = ConfigurableUserService(logger)

    config_result = configurable_service.configure({
        "database_url": "postgresql://localhost:5432/users",
        "max_connections": 10,
        "timeout_seconds": 30
    })

    if config_result.success:
        current_config = configurable_service.get_config()
        print(f"✅ Service configured with {len(current_config)} settings")
    else:
        print(f"❌ Configuration failed: {config_result.error}")

test_infrastructure_protocols()
```

### Step 3: Extensions Layer Implementation

```python
import uuid
from typing import Callable

# Plugin system implementation
class AnalyticsPlugin(FlextProtocols.Extensions.Plugin):
    """Analytics plugin following Extensions.Plugin protocol."""

    def __init__(self):
        self.config: dict[str, object] = {}
        self.context: FlextProtocols.Extensions.PluginContext | None = None
        self.initialized = False
        self.analytics_data: list[dict] = []

    def configure(self, config: dict[str, object]) -> object:
        """Configure plugin with settings."""
        self.config.update(config)
        return FlextResult[None].ok(None)

    def get_config(self) -> dict[str, object]:
        """Get current plugin configuration."""
        return self.config.copy()

    def initialize(self, context: FlextProtocols.Extensions.PluginContext) -> object:
        """Initialize plugin with context."""
        try:
            self.context = context
            logger = context.FlextLogger()

            # Get plugin-specific config
            plugin_config = context.get_config().get("analytics", {})
            config_result = self.configure(plugin_config)

            if not config_result.success:
                return config_result

            self.initialized = True
            logger.info("Analytics plugin initialized",
                       plugin_version="1.0.0",
                       config=plugin_config)

            return FlextResult[None].ok(None)

        except Exception as e:
            return FlextResult[None].fail(f"Plugin initialization failed: {e}")

    def shutdown(self) -> object:
        """Shutdown plugin and cleanup resources."""
        if self.context:
            logger = self.context.FlextLogger()
            logger.info("Analytics plugin shutting down",
                       events_collected=len(self.analytics_data))

        self.initialized = False
        self.analytics_data.clear()
        return FlextResult[None].ok(None)

    def get_info(self) -> dict[str, object]:
        """Get plugin information."""
        return {
            "name": "AnalyticsPlugin",
            "version": "1.0.0",
            "description": "User behavior analytics collection",
            "initialized": self.initialized,
            "events_collected": len(self.analytics_data),
            "config": self.config
        }

    def track_event(self, event_type: str, data: dict[str, object]) -> FlextResult[None]:
        """Track analytics event."""
        if not self.initialized:
            return FlextResult[None].fail("Plugin not initialized")

        event = {
            "id": str(uuid.uuid4()),
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        }

        self.analytics_data.append(event)

        if self.context:
            logger = self.context.FlextLogger()
            logger.debug("Event tracked", event_type=event_type, event_id=event["id"])

        return FlextResult[None].ok(None)

# Plugin context implementation
class SimplePluginContext(FlextProtocols.Extensions.PluginContext):
    """Simple plugin context implementation."""

    def __init__(self):
        self.services: dict[str, object] = {}
        self.config: dict[str, object] = {}
        self.logger = StructuredLogger("plugin-context")

    def get_service(self, service_name: str) -> object:
        """Get service instance by name."""
        if service_name in self.services:
            return self.services[service_name]
        else:
            raise ValueError(f"Service not found: {service_name}")

    def get_config(self) -> dict[str, object]:
        """Get configuration for plugin."""
        return self.config.copy()

    def FlextLogger(self) -> FlextProtocols.Infrastructure.LoggerProtocol:
        """Get logger instance for plugin."""
        return self.logger

    def register_service(self, name: str, service: object):
        """Register service in context."""
        self.services[name] = service
        self.logger.info("Service registered", service_name=name)

    def set_config(self, config: dict[str, object]):
        """Set plugin configuration."""
        self.config.update(config)

# Middleware implementation
class LoggingMiddleware(FlextProtocols.Extensions.Middleware):
    """Request logging middleware."""

    def __init__(self, logger: FlextProtocols.Infrastructure.LoggerProtocol):
        self.logger = logger

    def process(self, request: object, next_handler: Callable[[object], object]) -> object:
        """Process request with logging."""
        request_id = getattr(request, 'id', f"req_{int(time.time())}")

        self.logger.info("Processing request", request_id=request_id)
        start_time = time.time()

        try:
            result = next_handler(request)

            duration_ms = (time.time() - start_time) * 1000
            self.logger.info("Request processed successfully",
                           request_id=request_id,
                           duration_ms=duration_ms)

            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.exception("Request processing failed",
                                request_id=request_id,
                                duration_ms=duration_ms)

            return FlextResult[object].fail(f"Request failed: {e}")

# Observability implementation
class SimpleObservabilityCollector(FlextProtocols.Extensions.Observability):
    """Simple observability metrics collector."""

    def __init__(self):
        self.metrics: dict[str, dict] = {}
        self.traces: dict[str, dict] = {}

    def record_metric(self, name: str, value: float, tags: dict[str, str] | None = None) -> object:
        """Record metric value."""
        metric_key = f"{name}:{tags}" if tags else name

        self.metrics[metric_key] = {
            "name": name,
            "value": value,
            "tags": tags or {},
            "timestamp": time.time()
        }

        print(f"📊 Metric recorded: {name} = {value} {tags or ''}")
        return FlextResult[None].ok(None)

    def start_trace(self, operation_name: str) -> object:
        """Start distributed trace."""
        trace_id = f"{operation_name}_{uuid.uuid4()}"

        self.traces[trace_id] = {
            "id": trace_id,
            "operation": operation_name,
            "start_time": time.time(),
            "status": "active"
        }

        print(f"🔍 Trace started: {operation_name} ({trace_id})")
        return FlextResult[str].ok(trace_id)

    def health_check(self) -> object:
        """Perform health check."""
        active_traces = len([t for t in self.traces.values() if t["status"] == "active"])

        health_data = {
            "status": "healthy",
            "metrics_count": len(self.metrics),
            "active_traces": active_traces,
            "total_traces": len(self.traces),
            "timestamp": time.time()
        }

        return FlextResult[dict].ok(health_data)

# Test extensions layer
def test_extensions_protocols():
    """Test Extensions layer protocols."""

    # Setup plugin context
    context = SimplePluginContext()
    context.set_config({
        "analytics": {
            "enabled": True,
            "batch_size": 100,
            "flush_interval": 60
        }
    })

    # Test plugin
    plugin = AnalyticsPlugin()

    init_result = plugin.initialize(context)
    if init_result.success:
        print("✅ Plugin initialized successfully")

        # Track some events
        plugin.track_event("user_login", {"user_id": "123", "source": "web"})
        plugin.track_event("page_view", {"page": "/dashboard", "user_id": "123"})

        plugin_info = plugin.get_info()
        print(f"📋 Plugin info: {plugin_info['name']} - {plugin_info['events_collected']} events")

        # Shutdown plugin
        plugin.shutdown()

    # Test middleware
    logger = StructuredLogger("middleware-test")
    middleware = LoggingMiddleware(logger)

    # Mock request object
    class MockRequest:
        def __init__(self, request_id: str):
            self.id = request_id

    # Mock handler
    def mock_handler(request: object) -> object:
        time.sleep(0.1)  # Simulate processing
        return FlextResult[dict].ok({"processed": True, "request_id": request.id})

    request = MockRequest("test_123")
    result = middleware.process(request, mock_handler)

    if result.success:
        print(f"✅ Middleware processed request: {result.value}")

    # Test observability
    obs = SimpleObservabilityCollector()

    # Record metrics
    obs.record_metric("requests_total", 1.0, {"method": "GET", "status": "200"})
    obs.record_metric("response_time_ms", 156.7, {"endpoint": "/api/users"})

    # Start trace
    trace_result = obs.start_trace("user_creation")
    if trace_result.success:
        trace_id = trace_result.value
        print(f"🔍 Started trace: {trace_id}")

    # Health check
    health = obs.health_check()
    if health.success:
        health_data = health.value
        print(f"✅ Observability health: {health_data['metrics_count']} metrics, {health_data['active_traces']} traces")

test_extensions_protocols()
```

---

## 🎯 Advanced Patterns and Best Practices

### 1. Protocol Composition Patterns

```python
# Composing multiple protocols for complex functionality
class EnterpriseUserService(
    FlextProtocols.Domain.Service,
    FlextProtocols.Infrastructure.Configurable,
    FlextProtocols.Extensions.Observability
):
    """User service composing multiple protocol interfaces."""

    def __init__(self, logger: FlextProtocols.Infrastructure.LoggerProtocol):
        self.logger = logger
        self.config: dict[str, object] = {}
        self.metrics: dict[str, dict] = {}
        self.traces: dict[str, dict] = {}
        self._running = False

    # Domain.Service implementation
    def __call__(self, *args: object, **kwargs: object) -> object:
        return self.process_user_operation(*args, **kwargs)

    def start(self) -> object:
        self.logger.info("Starting enterprise user service")
        self._running = True
        self.record_metric("service_starts", 1.0)
        return FlextResult[None].ok(None)

    def stop(self) -> object:
        self.logger.info("Stopping enterprise user service")
        self._running = False
        return FlextResult[None].ok(None)

    def health_check(self) -> object:
        if not self._running:
            return FlextResult[dict].fail("Service not running")

        return FlextResult[dict].ok({
            "status": "healthy",
            "running": self._running,
            "config_loaded": bool(self.config),
            "metrics_collected": len(self.metrics)
        })

    # Infrastructure.Configurable implementation
    def configure(self, config: dict[str, object]) -> object:
        self.config.update(config)
        self.logger.info("Service configured", config_keys=list(config.keys()))
        return FlextResult[None].ok(None)

    def get_config(self) -> dict[str, object]:
        return self.config.copy()

    # Extensions.Observability implementation
    def record_metric(self, name: str, value: float, tags: dict[str, str] | None = None) -> object:
        metric_key = f"{name}:{tags}" if tags else name
        self.metrics[metric_key] = {
            "name": name,
            "value": value,
            "tags": tags or {},
            "timestamp": time.time()
        }
        return FlextResult[None].ok(None)

    def start_trace(self, operation_name: str) -> object:
        trace_id = f"{operation_name}_{uuid.uuid4()}"
        self.traces[trace_id] = {
            "operation": operation_name,
            "start_time": time.time(),
            "status": "active"
        }
        return FlextResult[str].ok(trace_id)

    # Business logic
    def process_user_operation(self, *args: object, **kwargs: object) -> object:
        operation = kwargs.get("operation")

        # Start trace
        trace_result = self.start_trace(f"user_{operation}")

        if operation == "create":
            self.record_metric("user_operations", 1.0, {"type": "create"})
            # Implementation here

        return FlextResult[dict].ok({"operation": operation, "success": True})

# Test composed service
enterprise_service = EnterpriseUserService(StructuredLogger("enterprise"))

# Configure
enterprise_service.configure({"max_users": 1000, "timeout": 30})

# Start service
start_result = enterprise_service.start()
if start_result.success:
    # Test operations
    result = enterprise_service(operation="create", user_data={"name": "Test User"})

    # Check health
    health = enterprise_service.health_check()
    print(f"✅ Enterprise service health: {health.value}")

    enterprise_service.stop()
```

### 2. Runtime Protocol Validation

```python
def validate_protocol_implementation(obj: object, protocol_class: type) -> bool:
    """Validate object implements protocol at runtime."""

    if not hasattr(protocol_class, '__runtime_checkable__'):
        print(f"⚠️  Protocol {protocol_class.__name__} is not runtime checkable")
        return False

    if isinstance(obj, protocol_class):
        print(f"✅ Object implements {protocol_class.__name__} protocol")
        return True
    else:
        print(f"❌ Object does not implement {protocol_class.__name__} protocol")
        return False

# Test protocol validation
logger = StructuredLogger("test")
db_conn = DatabaseConnection("test://localhost")

# These should pass
validate_protocol_implementation(logger, FlextProtocols.Infrastructure.LoggerProtocol)
validate_protocol_implementation(db_conn, FlextProtocols.Infrastructure.Connection)

# This should fail
validate_protocol_implementation(logger, FlextProtocols.Domain.Service)
```

### 3. Performance Configuration

```python
# Configure protocols for different environments
def setup_production_protocols():
    """Setup protocols for production environment."""

    prod_config = FlextProtocolsConfig.create_environment_protocols_config("production")

    if prod_config.success:
        config = prod_config.value
        print(f"Production config: {config['protocol_level']}")

        # Apply high-performance optimizations
        perf_config = FlextProtocolsConfig.optimize_protocols_performance("high")

        if perf_config.success:
            optimized = perf_config.value
            print(f"Performance optimized: {optimized['performance_level']}")

def setup_development_protocols():
    """Setup protocols for development environment."""

    dev_config = FlextProtocolsConfig.create_environment_protocols_config("development")

    if dev_config.success:
        config = dev_config.value
        print(f"Development config: {config['protocol_level']}")

# Test environment configurations
setup_production_protocols()
setup_development_protocols()
```

---

This comprehensive implementation guide demonstrates how to effectively leverage FlextProtocols across all architectural layers, from basic protocol adoption to enterprise-grade contract architecture with performance optimization and runtime validation capabilities.
