# FLEXT-CORE INTEGRATION PATTERNS

## Universal FlextContainer & FlextContext Integration Guide

**Version**: 0.9.9 RC | **Status**: Implementation Guide | **Updated**: 2025-10-10 | **Phase 1**: Context Enrichment Completed

---

## 🎯 OVERVIEW

This document provides comprehensive guidance on using FlextContainer and FlextContext throughout flext-core via the **FlextMixins** pattern. This pattern provides automatic dependency injection, context management, structured logging, and performance tracking for all service classes.

### Key Benefits

- ✅ **Automatic DI**: Container access without manual instantiation
- ✅ **Auto Context**: Correlation IDs and operation tracking
- ✅ **Built-in Logging**: Structured logging with DI integration
- ✅ **Performance Metrics**: Automatic operation timing
- ✅ **Zero Boilerplate**: Infrastructure works transparently
- ✅ **100% ABI Compatible**: No breaking changes to existing code

### Phase 1 Context Enrichment (v0.9.9)

**Status**: ✅ **COMPLETED** - Major architectural enhancement providing zero-boilerplate context management for distributed tracing and audit trails.

#### New FlextService & FlextMixins Capabilities

Both `FlextService` and `FlextMixins` now provide automatic context management methods:

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

class PaymentService(FlextService[FlextTypes.Dict]):
    """Service with automatic context enrichment."""

    def process_payment(self, payment_id: str, amount: float, user_id: str) -> FlextResult[dict]:
        # Generate correlation ID for distributed tracing
        correlation_id = self._with_correlation_id()

        # Set user context for audit trail
        self._with_user_context(user_id, payment_id=payment_id)

        # Set operation context for tracking
        self._with_operation_context("process_payment", amount=amount)

        # All logs now include full context automatically
        self.logger.info("Processing payment", payment_id=payment_id, amount=amount)

        return FlextResult[dict].ok({"status": "completed", "correlation_id": correlation_id})
```

#### Available Context Methods

- `_with_correlation_id(correlation_id=None)` - Set/generate correlation ID for distributed tracing
- `_with_user_context(user_id, **user_data)` - Set user context for audit trails
- `_with_operation_context(operation_name, **data)` - Set operation context for tracking
- `_enrich_context(**context_data)` - Add custom metadata to logs
- `_clear_operation_context()` - Clean up operation context after completion

#### Direct FlextMixins Usage

For simpler services, you can inherit directly from `FlextMixins`:

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

class SimpleService(FlextMixins):
    """Service using FlextMixins directly."""

    def __init__(self):
        self._init_service("simple_service")
        # Automatic access to:
        # - self.container (global DI container)
        # - self.logger (structured logger with context)
        # - self._with_operation_context() (context management)
        # - self.track() (performance metrics)

    def process_data(self, data: dict) -> FlextResult[dict]:
        with self.track("process_data") as metrics:
            self.logger.info("Processing data", size=len(data))
            return FlextResult[dict].ok({"processed": True})
```

#### Complete Automation Helper

`execute_with_context_enrichment()` provides full automation:

```python
class OrderService(FlextService[Order]):
    def process_order(self, order_id: str, customer_id: str, correlation_id: str | None = None) -> FlextResult[Order]:
        return self.execute_with_context_enrichment(
            operation_name="process_order",
            correlation_id=correlation_id,
            user_id=customer_id,
            order_id=order_id,
        )
        # Automatically handles: correlation ID, user context, operation tracking, logging, cleanup
```

#### Benefits

- ✅ **Zero Boilerplate** - No manual context setup required
- ✅ **Distributed Tracing** - Automatic correlation ID generation
- ✅ **Audit Trail** - User context automatically captured
- ✅ **Ecosystem Ready** - Available to all 32+ dependent projects
- ✅ **Performance Tracking** - Operation lifecycle monitoring

See `examples/15_automation_showcase.py` for complete working examples.

---

## 🏗️ CORE PATTERN: FlextMixins

### The Complete Service Infrastructure

`FlextMixins` is the foundation class that provides complete infrastructure integration:

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

class MyService(FlextMixins):
    """Service with automatic infrastructure integration."""

    def __init__(self):
        # Initialize service with auto-registration
        self._init_service("my_service")
        # Now you have automatic access to:
        # - self.container (FlextContainer.get_global())
        # - self.context (FlextContext instance)
        # - self.logger (FlextLogger with DI)
        # - self.track() (performance metrics)

    def process_data(self, data: dict) -> FlextResult[dict]:
        """Process data with full infrastructure."""
        # Automatic performance tracking
        with self.track("process_data"):
            # Automatic context propagation
            self._propagate_context("process_data")

            # Automatic structured logging
            self.logger.info(
                "Processing data",
                extra={"size": len(data), "correlation_id": self._get_correlation_id()}
            )

            # Your business logic
            result = {"processed": True, "count": len(data)}
            return FlextResult[dict].ok(result)
```

### What You Get Automatically

#### 1. Container Access (FlextMixins)

```python
class MyService(FlextMixins):
    def __init__(self):
        self._init_service("my_service")

    def use_container(self):
        # Access global container automatically
        db_result = self.container.get_typed("database", DatabaseService)
        if db_result.is_success:
            db = db_result.unwrap()
            # Use database service

        # Register new services
        cache_result = self.container.register("cache", CacheService())
        if cache_result.is_success:
            self.logger.info("Cache service registered")
```

#### 2. Context Management (FlextMixins)

```python
class MyService(FlextMixins):
    def process_request(self, request_id: str, data: dict) -> FlextResult[dict]:
        # Propagate context for operation
        self._propagate_context("process_request")

        # Get correlation ID (auto-generated if not exists)
        correlation_id = self._get_correlation_id()

        # Set custom correlation ID
        self._set_correlation_id(f"req-{request_id}")

        # All logs will include correlation ID automatically
        self.logger.info("Processing request", extra={"request_id": request_id})

        return FlextResult[dict].ok({"status": "processed"})
```

#### 3. Structured Logging (FlextMixins)

```python
class MyService(FlextMixins):
    def complex_operation(self, data: dict) -> FlextResult[dict]:
        # Logger automatically configured with class name
        self.logger.info("Starting complex operation")

        # Context-aware logging (includes correlation ID)
        self._log_with_context(
            "info",
            "Processing phase 1",
            phase=1,
            data_size=len(data)
        )

        # Standard log levels available
        self.logger.debug("Detailed debug information")
        self.logger.warning("Warning message")
        self.logger.error("Error occurred")

        return FlextResult[dict].ok({"status": "completed"})
```

#### 4. Performance Tracking (FlextMixins)

```python
class MyService(FlextMixins):
    def timed_operation(self, data: dict) -> FlextResult[dict]:
        # Automatic performance tracking with context manager
        with self.track("timed_operation") as metrics:
            # Your operation here
            result = self._process_data(data)

            # Metrics automatically collected:
            # - Operation duration
            # - Correlation ID
            # - Operation name
            # - Timestamp

            return FlextResult[dict].ok(result)

    def manual_metrics(self):
        # Access metrics if needed
        start = time.perf_counter()
        # ... operation ...
        duration = time.perf_counter() - start
        self.logger.info(f"Operation took {duration:.2f}s")
```

---

## 🎨 USAGE PATTERNS BY MODULE TYPE

### Pattern 1: Simple Service

**Use Case**: Basic service with DI and logging needs

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

class UserService(FlextMixins):
    """Simple service for user operations."""

    def __init__(self):
        self._init_service("user_service")

    def create_user(self, name: str, email: str) -> FlextResult[dict]:
        """Create user with automatic logging and context."""
        with self.track("create_user"):
            self._propagate_context("create_user")

            self.logger.info(
                "Creating user",
                extra={"name": name, "email": email}
            )

            # Business logic
            user = {"id": "123", "name": name, "email": email}

            self.logger.info("User created successfully")
            return FlextResult[dict].ok(user)
```

### Pattern 2: CQRS Handler

**Use Case**: Command/Query handlers (already uses pattern!)

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

class CreateUserHandler(FlextHandlers[CreateUserCommand, User]):
    """Handler inherits FlextMixins automatically."""

    def __init__(self):
        config = FlextModels.Cqrs.Handler(
            handler_name="CreateUserHandler",
            handler_type="command"
        )
        super().__init__(config=config)
        # Automatically has: container, context, logger, metrics

    def handle(self, command: CreateUserCommand) -> FlextResult[User]:
        """Handle with automatic infrastructure."""
        # Logger, context, and metrics already configured!
        self.logger.info(f"Handling CreateUserCommand: {command.name}")

        # Business logic
        user = User(name=command.name, email=command.email)

        return FlextResult[User].ok(user)
```

### Pattern 3: Event Bus / Dispatcher

**Use Case**: Infrastructure services that route messages

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

class FlextBus(FlextMixins):
    """Event bus with automatic infrastructure."""

    def __init__(self):
        self._init_service("flext_bus")
        self._subscribers: dict[str, list[Callable]] = {}

    def publish(self, event_name: str, event_data: dict) -> FlextResult[None]:
        """Publish event with automatic tracking."""
        with self.track(f"publish_{event_name}"):
            self._propagate_context(f"publish_{event_name}")

            self.logger.info(
                f"Publishing event: {event_name}",
                extra={"event_data": event_data}
            )

            # Get subscribers from container or internal registry
            subscribers = self._subscribers.get(event_name, [])

            for subscriber in subscribers:
                try:
                    subscriber(event_data)
                except Exception as e:
                    self.logger.error(
                        f"Subscriber failed: {e}",
                        extra={"event_name": event_name}
                    )

            return FlextResult[None].ok(None)

    def subscribe(self, event_name: str, handler: Callable) -> FlextResult[None]:
        """Subscribe to event."""
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []

        self._subscribers[event_name].append(handler)
        self.logger.info(f"Handler subscribed to {event_name}")

        return FlextResult[None].ok(None)
```

### Pattern 4: Registry Service

**Use Case**: Service registration and discovery

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

class FlextRegistry(FlextMixins):
    """Registry service with automatic DI integration."""

    def __init__(self):
        self._init_service("flext_registry")

    def register_handler(
        self,
        handler_type: str,
        handler: object
    ) -> FlextResult[None]:
        """Register handler in both registry and container."""
        with self.track("register_handler"):
            self._propagate_context("register_handler")

            # Register in container for DI
            service_key = f"handler:{handler_type}"
            container_result = self.container.register(service_key, handler)

            if container_result.is_failure:
                self.logger.error(
                    f"Failed to register handler: {container_result.error}"
                )
                return FlextResult[None].fail(container_result.error)

            self.logger.info(
                f"Handler registered: {handler_type}",
                extra={"service_key": service_key}
            )

            return FlextResult[None].ok(None)

    def get_handler(self, handler_type: str) -> FlextResult[object]:
        """Get handler from container."""
        service_key = f"handler:{handler_type}"
        return self.container.get(service_key)
```

### Pattern 5: Processor / Pipeline

**Use Case**: Data processing pipelines

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

class FlextProcessors(FlextMixins):
    """Processing pipeline with automatic metrics."""

    def __init__(self):
        self._init_service("flext_processors")

    def process_pipeline(
        self,
        data: dict,
        steps: list[Callable[[dict], FlextResult[dict]]]
    ) -> FlextResult[dict]:
        """Execute processing pipeline with tracking."""
        with self.track("process_pipeline"):
            self._propagate_context("process_pipeline")

            self.logger.info(
                f"Starting pipeline with {len(steps)} steps",
                extra={"initial_data": data}
            )

            current_data = data

            for i, step in enumerate(steps):
                step_name = getattr(step, '__name__', f'step_{i}')

                self.logger.debug(f"Executing step: {step_name}")

                result = step(current_data)

                if result.is_failure:
                    self.logger.error(
                        f"Pipeline failed at step {step_name}: {result.error}"
                    )
                    return FlextResult[dict].fail(
                        f"Pipeline failed at {step_name}: {result.error}"
                    )

                current_data = result.unwrap()

            self.logger.info("Pipeline completed successfully")
            return FlextResult[dict].ok(current_data)
```

---

## 🔧 ADVANCED PATTERNS

### Pattern 6: Nested Service Composition

**Compose multiple services with shared infrastructure**:

```python
class UserRepository(FlextMixins):
    """Repository with database access."""

    def __init__(self):
        self._init_service("user_repository")

    def find_by_id(self, user_id: str) -> FlextResult[dict | None]:
        # Get database from container
        db_result = self.container.get("database")
        if db_result.is_failure:
            return FlextResult[dict | None].fail("Database not available")

        db = db_result.unwrap()
        # Query database...
        return FlextResult[dict | None].ok({"id": user_id, "name": "John"})


class UserService(FlextMixins):
    """Service using repository."""

    def __init__(self):
        self._init_service("user_service")

        # Get repository from container (DI)
        repo_result = self.container.get("user_repository")
        if repo_result.is_success:
            self._repository = repo_result.unwrap()
        else:
            # Create and register if not exists
            self._repository = UserRepository()
            self.container.register("user_repository", self._repository)

    def get_user(self, user_id: str) -> FlextResult[dict]:
        """Get user with automatic context propagation."""
        with self.track("get_user"):
            self._propagate_context("get_user")

            # Repository call will have same context
            return self._repository.find_by_id(user_id)
```

### Pattern 7: Testing with Infrastructure

**Test services with isolated infrastructure**:

```python
import pytest
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

class TestUserService:
    @pytest.fixture
    def isolated_service(self):
        """Create service with isolated container."""
        # Get fresh container for test
        container = FlextContainer.get_global()

        # Register test dependencies
        container.register("database", MockDatabase())

        # Create service
        service = UserService()

        yield service

        # Cleanup after test
        container.clear()

    def test_create_user(self, isolated_service):
        """Test with automatic infrastructure."""
        result = isolated_service.create_user("John", "john@example.com")

        assert result.is_success
        assert result.unwrap()["name"] == "John"
```

---

## 📋 MIGRATION GUIDE

### Migrating Existing Code to Use Pattern

**BEFORE** (manual infrastructure):

```python
class OldService:
    def __init__(self):
        self.logger = FlextLogger(__name__)
        self._container = FlextContainer.get_global()

    def process(self, data: dict) -> FlextResult[dict]:
        self.logger.info("Processing")
        # Manual context setup
        FlextContext.Request.set_operation_name("process")
        # Manual timing
        start = time.time()
        result = self._do_process(data)
        duration = time.time() - start
        self.logger.info(f"Took {duration}s")
        return result
```

**AFTER** (automatic infrastructure):

```python
class NewService(FlextMixins):
    def __init__(self):
        self._init_service("new_service")

    def process(self, data: dict) -> FlextResult[dict]:
        # Everything automatic!
        with self.track("process"):
            self._propagate_context("process")
            self.logger.info("Processing")
            return self._do_process(data)
```

### Backward Compatibility

**100% ABI Compatible** - Old code continues to work:

```python
# Old code still works
logger = FlextLogger(__name__)
container = FlextContainer.get_global()
context = FlextContext()

# New code uses pattern
class NewService(FlextMixins):
    pass  # Gets everything automatically
```

---

## 🎯 BEST PRACTICES

### DO ✅

1. **Inherit from FlextMixins** for all service classes
2. **Call \_init_service()** in **init** with service name
3. **Use \track()** for performance-critical operations
4. **Use \_propagate_context()** for operations that call other services
5. **Use self.logger** for all logging (automatic DI)
6. **Use self.container** for service discovery
7. **Return FlextResult** from all operations

### DON'T ❌

1. **Don't manually instantiate FlextLogger** - use self.logger
2. **Don't manually call FlextContainer.get_global()** - use self.container
3. **Don't manually manage context** - use \_propagate_context()
4. **Don't forget \_init_service()** - required for auto-registration
5. **Don't use print()** for logging - use self.logger
6. **Don't use try/except for business logic** - use FlextResult
7. **Don't create custom infrastructure** - use provided patterns

---

## 📚 REFERENCE

### FlextMixins API

```python
class Service(Container, Context, Logging, Metrics):
    """Complete service infrastructure."""

    # Container access
    @property
    def container() -> FlextContainer
    def _register_in_container(service_name: str) -> FlextResult[None]

    # Context management
    @property
    def context() -> object
    def _propagate_context(operation_name: str) -> None
    def _get_correlation_id() -> str | None
    def _set_correlation_id(correlation_id: str) -> None

    # Logging
    @property
    def logger() -> FlextLogger
    def _log_with_context(level: str, message: str, **extra: object) -> None

    # Metrics
    @contextmanager
    def track(operation_name: str) -> Iterator[FlextTypes.Dict]

    # Initialization
    def _init_service(service_name: str | None = None) -> None
```

### Complete Example

```python
from flext_core import FlextBus
from flext_core import FlextConfig
from flext_core import FlextConstants
from flext_core import FlextContainer
from flext_core import FlextContext
from flext_core import FlextDecorators
from flext_core import FlextDispatcher
from flext_core import FlextExceptions
from flext_core import FlextHandlers
from flext_core import FlextLogger
from flext_core import FlextMixins
from flext_core import FlextModels
from flext_core import FlextProcessors
from flext_core import FlextProtocols
from flext_core import FlextRegistry
from flext_core import FlextResult
from flext_core import FlextRuntime
from flext_core import FlextService
from flext_core import FlextTypes
from flext_core import FlextUtilities

class ComprehensiveService(FlextMixins):
    """Complete example using all infrastructure features."""

    def __init__(self, config: FlextConfig | None = None):
        # Initialize with auto-registration
        self._init_service("comprehensive_service")

        # Store config if provided
        self._config = config or FlextConfig.get_global_instance()

        # Register self in container for discovery
        self.container.register("comprehensive_service", self)

    def complete_workflow(
        self,
        data: dict,
        correlation_id: str | None = None
    ) -> FlextResult[dict]:
        """Complete workflow demonstrating all features."""

        # Set correlation ID if provided
        if correlation_id:
            self._set_correlation_id(correlation_id)

        # Track performance automatically
        with self.track("complete_workflow"):
            # Propagate context for this operation
            self._propagate_context("complete_workflow")

            # Structured logging with auto context
            self.logger.info(
                "Starting complete workflow",
                extra={
                    "data_size": len(data),
                    "correlation_id": self._get_correlation_id()
                }
            )

            # Get dependencies from container
            validator_result = self.container.get("validator")
            if validator_result.is_failure:
                return FlextResult[dict].fail("Validator not available")

            validator = validator_result.unwrap()

            # Validation step
            validation = validator.validate(data)
            if validation.is_failure:
                self.logger.error(f"Validation failed: {validation.error}")
                return FlextResult[dict].fail(validation.error)

            # Processing step
            processed = self._process_data(data)
            if processed.is_failure:
                return processed

            # Success
            result = processed.unwrap()
            self.logger.info(
                "Workflow completed successfully",
                extra={"result_keys": list(result.keys())}
            )

            return FlextResult[dict].ok(result)

    def _process_data(self, data: dict) -> FlextResult[dict]:
        """Internal processing with automatic context."""
        # Context automatically propagated from parent operation
        self.logger.debug("Processing data")

        # Business logic here
        processed = {"processed": True, **data}

        return FlextResult[dict].ok(processed)
```

---

## 🔗 SEE ALSO

- **FlextMixins API**: `src/flext_core/mixins.py`
- **FlextHandlers Pattern**: `src/flext_core/handlers.py`
- **FlextContainer**: `src/flext_core/container.py`
- **FlextContext**: `src/flext_core/context.py`
- **FLEXT Standards**: `../CLAUDE.md`

---

**Last Updated**: 2025-10-10 | **Version**: 0.9.9 RC | **Status**: Active | **Phase 1**: Context Enrichment Completed
