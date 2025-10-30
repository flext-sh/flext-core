# Advanced Dependency Injection with FlextContainer

**Status**: Production Ready | **Version**: 0.9.9 | **Type-Safety**: Full Generic Support

FlextContainer is FLEXT's type-safe dependency injection container providing centralized service management, singleton pattern enforcement, and advanced DI features integrated with the FlextResult railway pattern.

## Core Concepts

### Service Locator Pattern

FlextContainer implements the Service Locator pattern (also called the registry pattern). Services are registered centrally and retrieved on demand:

```
┌─────────────────────┐
│  FlextContainer     │
│  (Global Singleton) │
├─────────────────────┤
│ Services:           │
│ - logger            │ → FlextLogger instance
│ - database          │ → DatabaseService instance
│ - config            │ → FlextConfig instance
│ - api_client        │ → APIClient factory
└─────────────────────┘
```

### Singleton Pattern

FlextContainer is itself a singleton - only one instance exists per application:

```python
from flext_core import FlextContainer

# Get the global container (same instance)
container1 = FlextContainer.get_global()
container2 = FlextContainer.get_global()

assert container1 is container2  # True - same instance
```

### Type-Safe Resolution (v0.9.9+)

Modern FLEXT provides **generic type preservation** for safe service retrieval:

```python
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()

# Type-safe retrieval - type checker knows the exact type
result = container.get_typed("logger", FlextLogger)

if result.is_success:
    logger: FlextLogger = result.unwrap()  # Type is known
    logger.info("Message")  # IDE autocomplete works
```

## Getting Started

### Global Container Access

```python
from flext_core import FlextContainer

# Get the global singleton container
container = FlextContainer.get_global()

# Every call returns the same instance
assert FlextContainer.get_global() is container
```

### Registering Services

#### Simple Instance Registration

```python
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()

# Create service instance
logger = FlextLogger(__name__)

# Register it (wrapped in FlextResult)
result = container.register("logger", logger)

if result.is_success:
    print("✅ Logger registered")
else:
    print(f"❌ Registration failed: {result.error}")
```

**Source**: `src/flext_core/container.py` - `register()` method

#### Factory Registration

Register a factory function that creates instances on-demand:

```python
from flext_core import FlextContainer

def create_database_connection():
    """Factory function - called each time service is requested."""
    return {
        "type": "postgresql",
        "connection_id": 123,
    }

container = FlextContainer.get_global()

# Register factory (creates new instance on each call)
result = container.register_factory("database", create_database_connection)

if result.is_success:
    print("✅ Database factory registered")
```

#### Safe Factory Registration

```python
from flext_core import FlextContainer, FlextResult

def create_service_that_might_fail():
    """Factory that can fail."""
    if condition_not_met:
        raise ValueError("Configuration invalid")
    return ServiceInstance()

def safe_factory() -> FlextResult[object]:
    """Wrap potentially-failing factory."""
    try:
        service = create_service_that_might_fail()
        return FlextResult.ok(service)
    except Exception as e:
        return FlextResult.fail(str(e))

container = FlextContainer.get_global()
result = container.register_factory("safe_service", safe_factory)
```

### Retrieving Services

#### Basic Retrieval

```python
from flext_core import FlextContainer

container = FlextContainer.get_global()

# Get service (returns FlextResult[object])
result = container.get("logger")

if result.is_success:
    logger = result.unwrap()
    logger.info("Logged message")
else:
    print(f"❌ Service not found: {result.error}")
```

#### Type-Safe Retrieval (Recommended)

```python
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()

# Get service with type information (returns FlextResult[FlextLogger])
result = container.get_typed("logger", FlextLogger)

if result.is_success:
    logger: FlextLogger = result.unwrap()  # Type checker knows exact type
    logger.info("Message")
```

## Real-World Patterns

### Pattern 1: Application Initialization

```python
from flext_core import FlextContainer, FlextResult, FlextConfig, FlextLogger

def initialize_application() -> FlextResult[None]:
    """Initialize all application services."""
    container = FlextContainer.get_global()

    # Load configuration
    config_result = (
        FlextResult.ok(None)
        .flat_map(lambda _: FlextConfig.load())
    )

    if config_result.is_failure:
        return config_result

    config = config_result.unwrap()

    # Register configuration
    register_result = container.register("config", config)
    if register_result.is_failure:
        return register_result

    # Register logger
    logger = FlextLogger(config.log_level)
    logger_result = container.register("logger", logger)
    if logger_result.is_failure:
        return logger_result

    # Register database with factory
    def create_db():
        return DatabaseConnection(config.database_url)

    db_result = container.register_factory("database", create_db)
    if db_result.is_failure:
        return db_result

    logger.info("✅ Application initialized successfully")
    return FlextResult[None].ok(None)

# Usage
app_init = initialize_application()
if app_init.is_failure:
    print(f"Failed to initialize app: {app_init.error}")
    exit(1)
```

### Pattern 2: Service Resolution Chain

```python
from flext_core import FlextContainer, FlextResult

class PaymentService:
    def __init__(self, database, logger, config):
        self.db = database
        self.logger = logger
        self.config = config

    def process_payment(self, amount: float) -> FlextResult[dict]:
        """Process payment with dependencies."""
        self.logger.info(f"Processing payment: {amount}")
        # Use self.db, self.config
        return FlextResult[dict].ok({"status": "complete"})

def resolve_payment_service() -> FlextResult[PaymentService]:
    """Resolve PaymentService with all dependencies."""
    container = FlextContainer.get_global()

    # Get database
    db_result = container.get("database")
    if db_result.is_failure:
        return FlextResult[PaymentService].fail(f"No database: {db_result.error}")

    # Get logger
    logger_result = container.get("logger")
    if logger_result.is_failure:
        return FlextResult[PaymentService].fail(f"No logger: {logger_result.error}")

    # Get config
    config_result = container.get("config")
    if config_result.is_failure:
        return FlextResult[PaymentService].fail(f"No config: {config_result.error}")

    # Construct service with resolved dependencies
    service = PaymentService(
        database=db_result.unwrap(),
        logger=logger_result.unwrap(),
        config=config_result.unwrap(),
    )

    return FlextResult[PaymentService].ok(service)

# Usage
service_result = resolve_payment_service()
if service_result.is_success:
    service = service_result.unwrap()
    result = service.process_payment(1000.00)
```

### Pattern 3: Lazy Service Initialization

```python
from flext_core import FlextContainer
import time

class ExpensiveService:
    def __init__(self):
        print("Initializing expensive service...")
        time.sleep(2)  # Simulate expensive initialization
        self.ready = True

    def do_work(self):
        return "Work completed"

def create_expensive_service():
    """Factory that creates expensive service."""
    return ExpensiveService()

container = FlextContainer.get_global()

# Register factory - service NOT created yet
print("Registering factory...")
container.register_factory("expensive", create_expensive_service)
print("Factory registered (service not created yet)")

# First access - service is created
print("Getting service...")
result1 = container.get("expensive")
print(f"First access: {result1.unwrap().do_work()}")

# Second access - same instance (singleton behavior)
print("Getting service again...")
result2 = container.get("expensive")
print(f"Second access: {result2.unwrap().do_work()}")

# Both are the same instance
assert result1.unwrap() is result2.unwrap()
```

### Pattern 4: Conditional Service Registration

```python
from flext_core import FlextContainer, FlextResult, FlextConfig

def setup_services_based_on_config() -> FlextResult[None]:
    """Register services conditionally based on configuration."""
    container = FlextContainer.get_global()

    # Load config
    config_result = FlextConfig.load()
    if config_result.is_failure:
        return config_result

    config = config_result.unwrap()

    # Conditional registration
    if config.debug:
        # In debug mode: use mock services
        container.register("cache", MockCache())
        container.register("email_service", DebugEmailService())
    else:
        # In production: use real services
        container.register("cache", RedisCache(config.redis_url))
        container.register("email_service", SendgridEmailService(config.api_key))

    return FlextResult[None].ok(None)
```

### Pattern 5: Service Lifecycle Management

```python
from flext_core import FlextContainer, FlextResult

class DatabaseConnection:
    def __init__(self, url: str):
        self.url = url
        self.connected = False

    def connect(self) -> FlextResult[None]:
        """Establish connection."""
        print(f"Connecting to {self.url}")
        self.connected = True
        return FlextResult[None].ok(None)

    def disconnect(self):
        """Close connection."""
        print("Disconnecting database")
        self.connected = False

def setup_database_lifecycle() -> FlextResult[None]:
    """Setup database with lifecycle management."""
    container = FlextContainer.get_global()
    config_result = FlextConfig.load()

    if config_result.is_failure:
        return config_result

    config = config_result.unwrap()

    # Create and register database
    db = DatabaseConnection(config.database_url)

    # Connect before registering
    connect_result = db.connect()
    if connect_result.is_failure:
        return connect_result

    # Register in container
    register_result = container.register("database", db)

    # Could register cleanup function for application shutdown
    # container.register("_cleanup_db", lambda: db.disconnect())

    return register_result

# Usage
setup_result = setup_database_lifecycle()
if setup_result.is_success:
    print("✅ Database setup complete")
```

## Advanced Features

### Batch Operations

```python
from flext_core import FlextContainer

container = FlextContainer.get_global()

# Batch register multiple services (stops at first failure)
result = container.batch_register([
    ("logger", FlextLogger()),
    ("database", DatabaseService()),
    ("cache", CacheService()),
])

if result.is_success:
    print("✅ All services registered")
else:
    print(f"❌ Batch failed: {result.error}")
```

### Fallback Resolution

```python
from flext_core import FlextContainer

container = FlextContainer.get_global()

# Try primary service, fallback to alternative
def resolve_with_fallback():
    primary = container.get("primary_service")
    if primary.is_success:
        return primary

    # If primary failed, try fallback
    fallback = container.get("fallback_service")
    return fallback

# Or using railway pattern
result = (
    FlextResult.ok(None)
    .flat_map(lambda _: container.get("primary_service"))
    .lash(lambda _: container.get("fallback_service"))
)
```

### Service Validation

```python
from flext_core import FlextContainer, FlextResult

def validate_all_services() -> FlextResult[None]:
    """Validate that all registered services work correctly."""
    container = FlextContainer.get_global()

    # Get and validate critical services
    services_to_validate = ["logger", "database", "cache"]

    for service_name in services_to_validate:
        service_result = container.get(service_name)

        if service_result.is_failure:
            return FlextResult[None].fail(
                f"Service validation failed: {service_name}",
                error_code="SERVICE_MISSING",
                error_data={"service": service_name},
            )

        # Could add service-specific validation here
        service = service_result.unwrap()
        if not validate_service(service):
            return FlextResult[None].fail(
                f"Service validation failed: {service_name}",
                error_code="SERVICE_INVALID",
            )

    return FlextResult[None].ok(None)
```

## Type Safety Best Practices

### Correct: Type-Safe Retrieval

```python
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()

# CORRECT - Type checker knows exact type
result: FlextResult[FlextLogger] = container.get_typed("logger", FlextLogger)

if result.is_success:
    logger: FlextLogger = result.unwrap()
    logger.info("Message")  # IDE knows all methods
```

### Avoid: Untyped Retrieval

```python
# WORKS but loses type information
result = container.get("logger")  # Returns FlextResult[object]

if result.is_success:
    logger = result.unwrap()  # Type is object
    # IDE can't help with autocomplete
```

### Correct: Semantic Type Aliases

```python
from flext_core import FlextTypes

# Use semantic types for consistency
ServiceName = FlextTypes.ServiceName
ServiceType = FlextTypes.ServiceType

def get_service(name: ServiceName, type_cls: type[ServiceType]) -> FlextResult[ServiceType]:
    container = FlextContainer.get_global()
    return container.get_typed(name, type_cls)
```

## Common Patterns

### Pattern: Dependency Injection in Handlers

```python
from flext_core import FlextContainer, FlextResult

class UserHandler:
    def __init__(self, container: FlextContainer):
        self.container = container

    def create_user(self, user_data: dict) -> FlextResult[dict]:
        """Create user using injected services."""
        # Get dependencies from container
        db_result = self.container.get("database")
        logger_result = self.container.get("logger")

        if db_result.is_failure or logger_result.is_failure:
            return FlextResult[dict].fail("Dependencies unavailable")

        db = db_result.unwrap()
        logger = logger_result.unwrap()

        # Use dependencies
        logger.info(f"Creating user: {user_data}")
        user = db.save_user(user_data)

        return FlextResult[dict].ok(user)

# Usage
container = FlextContainer.get_global()
handler = UserHandler(container)
result = handler.create_user({"name": "Alice"})
```

### Pattern: Testing with Mock Services

```python
from flext_core import FlextContainer
import unittest

class TestUserService(unittest.TestCase):
    def setUp(self):
        """Setup test container with mocks."""
        self.container = FlextContainer()  # Create test instance

        # Register mock services
        self.container.register("database", MockDatabase())
        self.container.register("logger", MockLogger())

    def test_user_creation(self):
        """Test with mocked services."""
        handler = UserHandler(self.container)
        result = handler.create_user({"name": "Bob"})

        assert result.is_success
        assert result.unwrap()["name"] == "Bob"
```

## Best Practices

### 1. Use Global Singleton

```python
# ✅ CORRECT - Always use global container
container = FlextContainer.get_global()
result = container.get("service")

# ❌ WRONG - Creating multiple containers
container1 = FlextContainer()
container2 = FlextContainer()  # Different instance!
```

### 2. Check Results

```python
# ✅ CORRECT - Always check FlextResult
result = container.get("service")
if result.is_success:
    service = result.unwrap()
else:
    print(f"Failed: {result.error}")

# ❌ WRONG - Assuming success
service = container.get("service").unwrap()  # May crash
```

### 3. Register Early

```python
# ✅ CORRECT - Register during initialization
def initialize():
    container = FlextContainer.get_global()
    container.register("logger", FlextLogger())
    container.register("config", FlextConfig.load().unwrap())

initialize()

# Later in code
container = FlextContainer.get_global()
logger_result = container.get("logger")

# ❌ WRONG - Registering late, missing dependencies
def some_random_function():
    container = FlextContainer.get_global()
    container.register("database", create_db())  # Too late!
```

### 4. Type-Safe Retrieval

```python
# ✅ CORRECT - Use get_typed for type safety
result: FlextResult[FlextLogger] = container.get_typed("logger", FlextLogger)

# ⚠️ OK but less safe - Basic retrieval
result: FlextResult[object] = container.get("logger")
```

## Architecture Integration

**Layer**: Layer 1 (Foundation)
**Used by**: Layers 2-4 (Domain, Application, Infrastructure)
**Dependencies**: FlextResult, FlextConfig, dependency-injector
**Ecosystem**: 32+ projects depend on FlextContainer

```
Layer 4: Services use container to resolve dependencies
    ↓
Layer 3: Handlers and use cases get services via container
    ↓
Layer 2: Domain services registered in container
    ↓
Layer 1: FlextContainer - core dependency injection
    ↓
Layer 0: FlextConstants, FlextTypes
```

## Key Takeaways

1. **Global Singleton**: Only one container instance per application
2. **Type-Safe**: Generic support preserves type information
3. **Railway Pattern**: All operations return FlextResult
4. **Flexible Registration**: Support for instances and factories
5. **Lifecycle Management**: Can manage service initialization and cleanup
6. **Testable**: Easy to substitute mock services for testing

## See Also

- [Railway-Oriented Programming](./railway-oriented-programming.md)
- [Architecture Overview](../architecture/overview.md)
- [API Reference: FlextContainer](../api-reference/foundation.md#flextcontainer)
- **FLEXT CLAUDE.md**: Development workflow and patterns

---

**Example from FLEXT Ecosystem**: See `src/flext_tests/test_container.py` for 180+ test cases demonstrating container patterns and edge cases.
