# Advanced Dependency Injection with FlextContainer

<!-- TOC START -->

- [Core Concepts](#core-concepts)
  - [Service Locator Pattern](#service-locator-pattern)
  - [Singleton Pattern](#singleton-pattern)
  - [Type-Safe Resolution (v0.10.0+)](#type-safe-resolution-v0100)
- [Getting Started](#getting-started)
  - [Global Container Access](#global-container-access)
  - [Registering Services](#registering-services)
  - [Retrieving Services](#retrieving-services)
- [Real-World Patterns](#real-world-patterns)
  - [Pattern 1: Application Initialization](#pattern-1-application-initialization)
  - [Pattern 2: Service Resolution Chain](#pattern-2-service-resolution-chain)
  - [Pattern 3: Lazy Service Initialization](#pattern-3-lazy-service-initialization)
  - [Pattern 4: Conditional Service Registration](#pattern-4-conditional-service-registration)
  - [Pattern 5: Service Lifecycle Management](#pattern-5-service-lifecycle-management)
- [Advanced Features](#advanced-features)
  - [Batch Operations](#batch-operations)
  - [Fallback Resolution](#fallback-resolution)
  - [Service Validation](#service-validation)
- [Type Safety Best Practices](#type-safety-best-practices)
  - [Correct: Type-Safe Retrieval](#correct-type-safe-retrieval)
  - [Avoid: Untyped Retrieval](#avoid-untyped-retrieval)
  - [Correct: Semantic Type Aliases](#correct-semantic-type-aliases)
- [Common Patterns](#common-patterns)
  - [Pattern: Dependency Injection in Handlers](#pattern-dependency-injection-in-handlers)
  - [Pattern: Testing with Mock Services](#pattern-testing-with-mock-services)
- [Best Practices](#best-practices)
  - [1. Use Global Singleton](#1-use-global-singleton)
  - [2. Check Results](#2-check-results)
  - [3. Register Early](#3-register-early)
  - [4. Type-Safe Retrieval](#4-type-safe-retrieval)
- [FlextDispatcher Reliability Settings](#flextdispatcher-reliability-settings)
  - [Configuration via FlextSettings](#configuration-via-flextsettings)
  - [Circuit Breaker Configuration](#circuit-breaker-configuration)
  - [Rate Limiting Configuration](#rate-limiting-configuration)
  - [Retry Policy Configuration](#retry-policy-configuration)
  - [Timeout Configuration](#timeout-configuration)
  - [Complete Reliability Configuration Example](#complete-reliability-configuration-example)
  - [Environment-Based Configuration](#environment-based-configuration)
  - [Monitoring Reliability Metrics](#monitoring-reliability-metrics)
- [Architecture Integration](#architecture-integration)
- [Key Takeaways](#key-takeaways)
- [Next Steps](#next-steps)
- [See Also](#see-also)

<!-- TOC END -->

**Status**: Production Ready | **Version**: 0.10.0 | **Type-Safety**: Full Generic Support

FlextContainer is FLEXT's type-safe dependency injection container providing centralized service management, singleton pattern enforcement, and advanced DI features integrated with the FlextResult railway pattern.

## Core Concepts

### Service Locator Pattern

FlextContainer implements the Service Locator pattern (also called the registry pattern). Services are registered centrally and retrieved on demand:

````text
┌─────────────────────┐
│  FlextContainer     │
│  (Global Singleton) │
├─────────────────────┤
│ Services:           │
│ - logger            │ → FlextLogger instance
│ - database          │ → DatabaseService instance
│ - config            │ → FlextSettings instance
│ - api_client        │ → APIClient factory
└─────────────────────┘
```text

### Singleton Pattern

FlextContainer is itself a singleton - only one instance exists per application:

```python
from flext_core import FlextContainer

# Get the global container (same instance)
container1 = FlextContainer.get_global()
container2 = FlextContainer.get_global()

assert container1 is container2  # True - same instance
```text

### Type-Safe Resolution (v0.10.0+)

Modern FLEXT provides **generic type preservation** for safe service retrieval:

```python
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()

# Type-safe retrieval - type checker knows the exact type
result = container.get_typed("logger", FlextLogger)

if result.is_success:
    logger: FlextLogger = result.value  # Type is known
    logger.info("Message")  # IDE autocomplete works
```text

## Getting Started

### Global Container Access

```python
from flext_core import FlextContainer

# Get the global singleton container
container = FlextContainer.get_global()

# Every call returns the same instance
assert FlextContainer.get_global() is container
```text

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
```text

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
```text

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
```text

### Retrieving Services

#### Basic Retrieval

```python
from flext_core import FlextContainer

container = FlextContainer.get_global()

# Get service (returns FlextResult[object])
result = container.get("logger")

if result.is_success:
    logger = result.value
    logger.info("Logged message")
else:
    print(f"❌ Service not found: {result.error}")
```text

#### Type-Safe Retrieval (Recommended)

```python
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()

# Get service with type information (returns FlextResult[FlextLogger])
result = container.get_typed("logger", FlextLogger)

if result.is_success:
    logger: FlextLogger = result.value  # Type checker knows exact type
    logger.info("Message")
```text

## Real-World Patterns

### Pattern 1: Application Initialization

```python
from flext_core import FlextContainer, FlextResult, FlextSettings, FlextLogger

def initialize_application() -> FlextResult[bool]:
    """Initialize all application services."""
    container = FlextContainer.get_global()

    # Load configuration
    config_result = (
        FlextResult[bool].ok(True)
        .flat_map(lambda _: FlextSettings.load())
    )

    if config_result.is_failure:
        return config_result

    config = config_result.value

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
    return FlextResult[bool].ok(True)

# Usage
app_init = initialize_application()
if app_init.is_failure:
    print(f"Failed to initialize app: {app_init.error}")
    exit(1)
```text

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
        database=db_result.value,
        logger=logger_result.value,
        config=config_result.value,
    )

    return FlextResult[PaymentService].ok(service)

# Usage
service_result = resolve_payment_service()
if service_result.is_success:
    service = service_result.value
    result = service.process_payment(1000.00)
```text

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
print(f"First access: {result1.value.do_work()}")

# Second access - same instance (singleton behavior)
print("Getting service again...")
result2 = container.get("expensive")
print(f"Second access: {result2.value.do_work()}")

# Both are the same instance
assert result1.value is result2.value
```text

### Pattern 4: Conditional Service Registration

```python
from flext_core import FlextContainer, FlextResult, FlextSettings

def setup_services_based_on_config() -> FlextResult[bool]:
    """Register services conditionally based on configuration."""
    container = FlextContainer.get_global()

    # Load config
    config_result = FlextSettings.load()
    if config_result.is_failure:
        return config_result

    config = config_result.value

    # Conditional registration
    if config.debug:
        # In debug mode: use mock services
        container.register("cache", MockCache())
        container.register("email_service", DebugEmailService())
    else:
        # In production: use real services
        container.register("cache", RedisCache(config.redis_url))
        container.register("email_service", SendgridEmailService(config.api_key))

    return FlextResult[bool].ok(True)
```text

### Pattern 5: Service Lifecycle Management

```python
from flext_core import FlextContainer, FlextResult

class DatabaseConnection:
    def __init__(self, url: str):
        self.url = url
        self.connected = False

    def connect(self) -> FlextResult[bool]:
        """Establish connection."""
        print(f"Connecting to {self.url}")
        self.connected = True
        return FlextResult[bool].ok(True)

    def disconnect(self):
        """Close connection."""
        print("Disconnecting database")
        self.connected = False

def setup_database_lifecycle() -> FlextResult[bool]:
    """Setup database with lifecycle management."""
    container = FlextContainer.get_global()
    config_result = FlextSettings.load()

    if config_result.is_failure:
        return config_result

    config = config_result.value

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
```text

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
```text

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
    FlextResult[bool].ok(True)
    .flat_map(lambda _: container.get("primary_service"))
    .lash(lambda _: container.get("fallback_service"))
)
```text

### Service Validation

```python
from flext_core import FlextContainer, FlextResult

def validate_all_services() -> FlextResult[bool]:
    """Validate that all registered services work correctly."""
    container = FlextContainer.get_global()

    # Get and validate critical services
    services_to_validate = ["logger", "database", "cache"]

    for service_name in services_to_validate:
        service_result = container.get(service_name)

        if service_result.is_failure:
            return FlextResult[bool].fail(
                f"Service validation failed: {service_name}",
                error_code="SERVICE_MISSING",
                error_data={"service": service_name},
            )

        # Could add service-specific validation here
        service = service_result.value
        if not validate_service(service):
            return FlextResult[bool].fail(
                f"Service validation failed: {service_name}",
                error_code="SERVICE_INVALID",
            )

    return FlextResult[bool].ok(True)
```text

## Type Safety Best Practices

### Correct: Type-Safe Retrieval

```python
from flext_core import FlextContainer, FlextLogger

container = FlextContainer.get_global()

# CORRECT - Type checker knows exact type
result: FlextResult[FlextLogger] = container.get_typed("logger", FlextLogger)

if result.is_success:
    logger: FlextLogger = result.value
    logger.info("Message")  # IDE knows all methods
```text

### Avoid: Untyped Retrieval

```python
# WORKS but loses type information
result = container.get("logger")  # Returns FlextResult[object]

if result.is_success:
    logger = result.value  # Type is object
    # IDE can't help with autocomplete
```text

### Correct: Semantic Type Aliases

```python
from flext_core import t

# Use semantic types for consistency
ServiceName = t.ServiceName
ServiceType = t.ServiceType

def get_service(name: ServiceName, type_cls: type[ServiceType]) -> FlextResult[ServiceType]:
    container = FlextContainer.get_global()
    return container.get_typed(name, type_cls)
```text

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

        db = db_result.value
        logger = logger_result.value

        # Use dependencies
        logger.info(f"Creating user: {user_data}")
        user = db.save_user(user_data)

        return FlextResult[dict].ok(user)

# Usage
container = FlextContainer.get_global()
handler = UserHandler(container)
result = handler.create_user({"name": "Alice"})
```text

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
        assert result.value["name"] == "Bob"
```text

## Best Practices

### 1. Use Global Singleton

```python
# ✅ CORRECT - Always use global container
container = FlextContainer.get_global()
result = container.get("service")

# ❌ WRONG - Creating multiple containers
container1 = FlextContainer()
container2 = FlextContainer()  # Different instance!
```text

### 2. Check Results

```python
# ✅ CORRECT - Always check FlextResult
result = container.get("service")
if result.is_success:
    service = result.value
else:
    print(f"Failed: {result.error}")

# ❌ WRONG - Assuming success
service = container.get("service").value  # May crash
```text

### 3. Register Early

```python
# ✅ CORRECT - Register during initialization
def initialize():
    container = FlextContainer.get_global()
    container.register("logger", FlextLogger())
    container.register("config", FlextSettings.load().value)

initialize()

# Later in code
container = FlextContainer.get_global()
logger_result = container.get("logger")

# ❌ WRONG - Registering late, missing dependencies
def some_random_function():
    container = FlextContainer.get_global()
    container.register("database", create_db())  # Too late!
```text

### 4. Type-Safe Retrieval

```python
# ✅ CORRECT - Use get_typed for type safety
result: FlextResult[FlextLogger] = container.get_typed("logger", FlextLogger)

# ⚠️ OK but less safe - Basic retrieval
result: FlextResult[object] = container.get("logger")
```text

## FlextDispatcher Reliability Settings

FlextDispatcher provides configurable reliability patterns including circuit breaker, rate limiting, retry policies, and timeout enforcement. These settings are configured via FlextSettings and apply to all dispatcher operations.

### Configuration via FlextSettings

```python
from flext_core import FlextSettings, FlextDispatcher

# Configure dispatcher reliability settings
class AppConfig(FlextSettings):
    """Application configuration with dispatcher reliability settings."""

    # Circuit Breaker Settings
    circuit_breaker_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_recovery_timeout: float = 60.0  # Seconds before retry
    circuit_breaker_success_threshold: int = 2  # Successes to close circuit

    # Rate Limiting Settings
    rate_limit_max_requests: int = 100  # Max requests per window
    rate_limit_window_seconds: float = 60.0  # Time window in seconds

    # Retry Policy Settings
    max_retry_attempts: int = 3  # Maximum retry attempts
    retry_delay: float = 1.0  # Base delay between retries (seconds)

    # Timeout Settings
    dispatcher_timeout_seconds: float = 30.0  # Default operation timeout
    enable_timeout_executor: bool = True  # Use executor for timeout enforcement
    executor_workers: int = 4  # Thread pool workers for timeout executor

# Initialize dispatcher with configuration
config = AppConfig()
dispatcher = FlextDispatcher()  # Uses config via FlextSettings singleton
```text

### Circuit Breaker Configuration

Circuit breaker prevents cascading failures by temporarily blocking requests when failures exceed threshold:

```python
from flext_core import FlextSettings, FlextDispatcher, r

class ConfigWithCircuitBreaker(FlextSettings):
    """Configuration with circuit breaker protection."""
    circuit_breaker_threshold: int = 5  # Open circuit after 5 failures
    circuit_breaker_recovery_timeout: float = 60.0  # Wait 60s before retry
    circuit_breaker_success_threshold: int = 2  # Close after 2 successes

config = ConfigWithCircuitBreaker()
dispatcher = FlextDispatcher()

# Dispatcher automatically applies circuit breaker to all operations
result = dispatcher.dispatch(CreateUserCommand(name="Alice"))
if result.is_failure and "circuit breaker" in result.error.lower():
    print("Circuit breaker is open - service temporarily unavailable")
```text

### Rate Limiting Configuration

Rate limiting prevents overload by restricting requests per time window:

```python
from flext_core import FlextSettings, FlextDispatcher

class ConfigWithRateLimiting(FlextSettings):
    """Configuration with rate limiting."""
    rate_limit_max_requests: int = 100  # Max 100 requests
    rate_limit_window_seconds: float = 60.0  # Per 60 seconds

config = ConfigWithRateLimiting()
dispatcher = FlextDispatcher()

# Dispatcher enforces rate limits automatically
for i in range(150):  # More than rate limit
    result = dispatcher.dispatch(GetUserQuery(user_id=str(i)))
    if result.is_failure and "rate limit" in result.error.lower():
        print(f"Rate limit exceeded at request {i}")
        break
```text

### Retry Policy Configuration

Retry policy automatically retries failed operations with configurable backoff:

```python
from flext_core import FlextSettings, FlextDispatcher

class ConfigWithRetry(FlextSettings):
    """Configuration with retry policy."""
    max_retry_attempts: int = 3  # Retry up to 3 times
    retry_delay: float = 1.0  # 1 second delay between retries

config = ConfigWithRetry()
dispatcher = FlextDispatcher()

# Dispatcher automatically retries on failure
result = dispatcher.dispatch(ProcessOrderCommand(order_id="123"))
# If first attempt fails, dispatcher automatically retries up to 3 times
# with 1 second delay between attempts
```text

### Timeout Configuration

Timeout enforcement prevents operations from hanging indefinitely:

```python
from flext_core import FlextSettings, FlextDispatcher

class ConfigWithTimeout(FlextSettings):
    """Configuration with timeout enforcement."""
    dispatcher_timeout_seconds: float = 30.0  # 30 second timeout
    enable_timeout_executor: bool = True  # Use executor for timeout
    executor_workers: int = 4  # Thread pool size

config = ConfigWithTimeout()
dispatcher = FlextDispatcher()

# Dispatcher enforces timeout on all operations
result = dispatcher.dispatch(LongRunningCommand(data=large_data))
if result.is_failure and "timeout" in result.error.lower():
    print("Operation exceeded timeout limit")

# Per-operation timeout override
result = dispatcher.dispatch(
    LongRunningCommand(data=large_data),
    timeout_override=60  # Override default timeout for this operation
)
```text

### Complete Reliability Configuration Example

```python
from flext_core import FlextSettings, FlextDispatcher, r

class ProductionConfig(FlextSettings):
    """Production configuration with comprehensive reliability settings."""

    # Circuit Breaker: Fail fast on repeated failures
    circuit_breaker_threshold: int = 10
    circuit_breaker_recovery_timeout: float = 120.0
    circuit_breaker_success_threshold: int = 3

    # Rate Limiting: Prevent overload
    rate_limit_max_requests: int = 1000
    rate_limit_window_seconds: float = 60.0

    # Retry Policy: Automatic recovery
    max_retry_attempts: int = 5
    retry_delay: float = 2.0

    # Timeout: Prevent hanging operations
    dispatcher_timeout_seconds: float = 45.0
    enable_timeout_executor: bool = True
    executor_workers: int = 8

# Initialize with production settings
config = ProductionConfig()
dispatcher = FlextDispatcher()

# All operations use these reliability settings
def process_with_reliability(command):
    """Process command with full reliability protection."""
    return dispatcher.dispatch(command)
    # Automatically includes:
    # - Circuit breaker (opens after 10 failures)
    # - Rate limiting (max 1000 requests/minute)
    # - Retry policy (up to 5 attempts with 2s delay)
    # - Timeout enforcement (45s limit)
```text

### Environment-Based Configuration

Configure reliability settings per environment:

```python
from flext_core import FlextSettings
import os

class Config(FlextSettings):
    """Environment-aware configuration."""

    @property
    def circuit_breaker_threshold(self) -> int:
        """Different thresholds per environment."""
        env = os.getenv("ENVIRONMENT", "development")
        if env == "production":
            return 10
        elif env == "staging":
            return 5
        else:
            return 2  # Development: fail fast

    @property
    def max_retry_attempts(self) -> int:
        """Different retry counts per environment."""
        env = os.getenv("ENVIRONMENT", "development")
        if env == "production":
            return 5
        else:
            return 2  # Development: fewer retries
```text

### Monitoring Reliability Metrics

Access dispatcher metrics to monitor reliability patterns:

```python
from flext_core import FlextDispatcher

dispatcher = FlextDispatcher()

# Execute operations
for _ in range(100):
    dispatcher.dispatch(SomeCommand())

# Check metrics
metrics = dispatcher.get_metrics()
print(f"Total executions: {metrics.get('total_executions', 0)}")
print(f"Circuit breaker opens: {metrics.get('circuit_breaker_opens', 0)}")
print(f"Rate limit hits: {metrics.get('rate_limit_hits', 0)}")
print(f"Retry attempts: {metrics.get('retry_attempts', 0)}")
print(f"Timeout executions: {metrics.get('timeout_executions', 0)}")
```text

## Architecture Integration

**Layer**: Layer 1 (Foundation)
**Used by**: Layers 2-4 (Domain, Application, Infrastructure)
**Dependencies**: FlextResult, FlextSettings, dependency-injector
**Ecosystem**: 32+ projects depend on FlextContainer

```text
Layer 4: Services use container to resolve dependencies
    ↓
Layer 3: Handlers and use cases get services via container
    ↓
Layer 2: Domain services registered in container
    ↓
Layer 1: FlextContainer - core dependency injection
    ↓
Layer 0: FlextConstants, t
```text

## Key Takeaways

1. **Global Singleton**: Only one container instance per application
1. **Type-Safe**: Generic support preserves type information
1. **Railway Pattern**: All operations return FlextResult
1. **Flexible Registration**: Support for instances and factories
1. **Lifecycle Management**: Can manage service initialization and cleanup
1. **Testable**: Easy to substitute mock services for testing

## Next Steps

1. **Reliability Settings**: Review dispatcher reliability configuration above
1. **Service Patterns**: Explore Service Patterns for service-level DI
1. **Railway Pattern**: See Railway-Oriented Programming for result handling
1. **Dispatcher**: Check API Reference: FlextDispatcher for complete API

## See Also

- Railway-Oriented Programming - Result handling with DI
- Service Patterns - Service-level dependency injection
- Architecture Overview - System architecture
- API Reference: FlextContainer - Complete container API
- API Reference: FlextDispatcher - Dispatcher reliability API
- **FLEXT CLAUDE.md**: Development workflow and patterns

______________________________________________________________________

**Example from FLEXT Ecosystem**: See `src/flext_tests/test_container.py` for 180+ test cases demonstrating container patterns and edge cases.
````
