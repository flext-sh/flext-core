# Infrastructure Layer API Reference

This section covers the infrastructure layer classes that handle configuration, logging, context management, and external integrations.

## Configuration Management

### FlextConfig - Configuration System

Layered configuration system supporting multiple sources (environment variables, files, programmatic).

```python
from flext_core import FlextConfig

# Create configuration with multiple sources
config = FlextConfig(
    config_files=['config.toml', 'secrets.env'],
    overrides={'debug': True}
)

# Access configuration values
database_url = config.get('database.url')
debug_mode = config.get('debug', default=False)
api_key = config.get('api.key', required=True)

# Type-safe configuration
smtp_config = config.get_section('smtp')
smtp_host = smtp_config.get('host')
smtp_port = smtp_config.get('port', cast=int, default=587)
```

**Key Features:**

- Multiple configuration sources (files, env, programmatic)
- Type casting and validation
- Environment-specific configurations
- Secrets management and validation

## Logging and Observability

### FlextLogger - Structured Logging

Structured logging with correlation IDs and context propagation.

```python
from flext_core import FlextLogger

# Create logger with context
logger = FlextLogger(__name__)

# Basic logging
logger.info("Application started")
logger.warning("Deprecated feature used")
logger.error("Database connection failed", extra={"error_code": "DB001"})

# Structured logging with context
with logger.context(operation="user_creation", user_id="user_123"):
    logger.info("Creating user")
    # ... user creation logic
    logger.info("User created successfully")
```

**Key Features:**

- Structured logging with JSON output
- Context propagation across function calls
- Correlation ID tracking
- Sensitive data sanitization

### FlextContext - Context Management

Request and operation context with metadata and correlation tracking.

```python
from flext_core import FlextContext

# Create context
context = FlextContext.create(
    operation_id="op_123",
    user_id="user_456",
    metadata={"source": "api", "version": "1.0"}
)

# Access context information
operation_id = context.operation_id
user_id = context.user_id
metadata = context.metadata

# Context propagation
with context:
    # Context is available throughout this block
    logger = FlextLogger(__name__)
    logger.info("Processing request")  # Includes context info
```

## External Integrations

### FlextProtocols - Protocol Interfaces

Runtime-checkable protocols for external service integration.

```python
from flext_core import FlextProtocols
from typing import Protocol, runtime_checkable

@runtime_checkable
class DatabaseProtocol(FlextProtocols.BaseProtocol):
    """Protocol for database operations."""

    def connect(self) -> FlextResult[Connection]:
        ...

    def execute(self, query: str) -> FlextResult[Cursor]:
        ...

    def close(self) -> None:
        ...

@runtime_checkable
class EmailProtocol(FlextProtocols.BaseProtocol):
    """Protocol for email services."""

    def send_email(self, to: str, subject: str, body: str) -> FlextResult[bool]:
        ...

    def send_template(self, to: str, template: str, data: dict) -> FlextResult[bool]:
        ...
```

### FlextVersion - Version Management

Version information and compatibility checking.

```python
from flext_core import FlextVersion

# Version information
current_version = FlextVersion.current()
print(f"FLEXT-Core v{current_version}")

# Compatibility checking
is_compatible = FlextVersion.is_compatible("0.9.0", "0.9.9")
supports_feature = FlextVersion.supports_feature("async_handlers")
```

## Quality Metrics

| Module         | Coverage | Status       | Description                       |
| -------------- | -------- | ------------ | --------------------------------- |
| `config.py`    | 90%      | ✅ Stable    | Configuration management system   |
| `logging.py`   | 72%      | 🔄 Improving | Structured logging infrastructure |
| `context.py`   | 66%      | 🔄 Improving | Context tracking and propagation  |
| `protocols.py` | 99%      | ✅ Complete  | Runtime-checkable protocols       |
| `version.py`   | 100%     | ✅ Complete  | Version management                |

## Usage Examples

### Complete Infrastructure Setup

```python
from flext_core import (
    FlextConfig, FlextLogger, FlextContext,
    FlextContainer, FlextResult
)

# Configuration setup
config = FlextConfig.create(
    environment='production',
    config_files=['config.toml', 'secrets.env'],
    overrides={'log_level': 'INFO'}
)

# Logger setup
logger = FlextLogger("myapp")
logger.set_level(config.get('log_level'))

# Context setup
context = FlextContext.create(
    operation_id="api_request_123",
    user_id="user_456",
    metadata={"endpoint": "/api/users", "method": "POST"}
)

# Dependency injection setup
container = FlextContainer.get_global()

# Register infrastructure services
container.register("config", config)
container.register("logger", logger)
container.register("context", context)

# Application usage
def handle_api_request(request_data: dict) -> FlextResult[dict]:
    """Handle API request with full infrastructure support."""

    # Get dependencies
    config_result = container.get("config")
    logger_result = container.get("logger")

    if config_result.is_failure or logger_result.is_failure:
        return FlextResult[dict].fail("Infrastructure not available")

    config = config_result.unwrap()
    logger = logger_result.unwrap()

    # Log with context
    logger.info("Processing API request", extra={
        "endpoint": request_data.get("endpoint"),
        "user_id": request_data.get("user_id")
    })

    # Process request
    try:
        # Business logic here
        result = {"status": "success", "data": request_data}
        return FlextResult[dict].ok(result)

    except Exception as e:
        logger.error("Request processing failed", extra={"error": str(e)})
        return FlextResult[dict].fail("Internal server error")

# Usage
result = handle_api_request({
    "endpoint": "/api/users",
    "user_id": "user_456",
    "data": {"name": "Alice"}
})

if result.is_success:
    print("✅ Request processed successfully")
else:
    print(f"❌ Request failed: {result.error}")
```

### Configuration with Multiple Sources

```python
# config.toml
[database]
host = "localhost"
port = 5432
name = "myapp"

[api]
host = "0.0.0.0"
port = 8000
timeout = 30

# secrets.env
DATABASE_PASSWORD=secret_password
API_SECRET_KEY=super_secret_key

# Python code
config = FlextConfig.create(
    environment='production',
    config_files=['config.toml', 'secrets.env'],
    overrides={'database.pool_size': 10}
)

# Access configuration
db_config = config.get_section('database')
db_url = f"postgresql://{db_config.get('host')}:{db_config.get('port')}/{db_config.get('name')}"

api_config = config.get_section('api')
api_port = api_config.get('port', cast=int)
```

### Structured Logging with Correlation

```python
import asyncio
from flext_core import FlextLogger, FlextContext

logger = FlextLogger("payment_service")

async def process_payment(order_id: str, amount: float) -> FlextResult[str]:
    """Process payment with comprehensive logging."""

    # Create context for this operation
    context = FlextContext.create(
        operation_id=f"payment_{order_id}",
        metadata={"order_id": order_id, "amount": amount}
    )

    with context:
        logger.info("Starting payment processing")

        # Step 1: Validate payment
        with logger.context(step="validation"):
            if amount <= 0:
                logger.error("Invalid payment amount")
                return FlextResult[str].fail("Invalid amount")

        # Step 2: Authorize payment
        with logger.context(step="authorization"):
            auth_result = await authorize_payment(order_id, amount)
            if auth_result.is_failure:
                logger.error("Payment authorization failed", extra={
                    "reason": auth_result.error
                })
                return auth_result

        # Step 3: Capture payment
        with logger.context(step="capture"):
            capture_result = await capture_payment(order_id, amount)
            if capture_result.is_failure:
                logger.error("Payment capture failed", extra={
                    "reason": capture_result.error
                })
                return capture_result

        logger.info("Payment processed successfully")
        return FlextResult[str].ok(f"payment_{order_id}")

# Usage
result = asyncio.run(process_payment("order_123", 99.99))
```

This infrastructure layer provides a solid foundation for enterprise applications with proper configuration management, observability, and external service integration capabilities.
