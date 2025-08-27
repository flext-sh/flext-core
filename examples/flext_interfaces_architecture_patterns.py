#!/usr/bin/env python3
"""Architecture patterns with FlextInterfaces.

Demonstrates Clean Architecture, Domain-Driven Design,
and extensibility through protocols and abstract base classes.
    - Handler interfaces with middleware pipeline patterns
    - Repository interfaces with Unit of Work pattern
    - Plugin interfaces with extensibility and context management
    - Event interfaces with publish-subscribe patterns
    - Clean Architecture compliance through dependency inversion
    - Protocol-based structural typing for maximum flexibility

Key Components:
    - FlextValidator: Protocol for flexible validation implementations
    - FlextProtocols.Foundation.Validator: ABC for reusable validation rules
    - FlextService: ABC for service lifecycle management
    - FlextConfigurable: Protocol for configuration injection
    - FlextMessageHandler/FlextMiddleware: ABCs for message processing pipelines
    - FlextRepository/FlextUnitOfWork: ABCs for data access patterns
    - FlextProtocols.Plugin/FlextProtocols.PluginContext: ABCs and protocols for extensibility
    - FlextEventPublisher/FlextEventSubscriber: ABCs for event-driven patterns

This example shows real-world enterprise architecture scenarios
demonstrating the power and flexibility of the FlextInterfaces system.
"""

import time
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass

# =============================================================================
# LOCAL TYPE DEFINITIONS - For demonstration purposes
# =============================================================================
from typing import Protocol, TypeVar, cast

from structlog.stdlib import BoundLogger

from flext_core import (
    FlextConstants,
    FlextProtocols,
    FlextResult,
    FlextTypes,
    get_logger,
)

T = TypeVar("T")


class FlextPayload[T]:
    """Generic payload base class for demonstration."""

    def __init__(self, data: T) -> None:
        self.data = data


class FlextBaseHandler:
    """Base handler for demonstration."""


class FlextUnitOfWork:
    """Unit of work pattern for demonstration."""


class FlextLoggerProtocol(Protocol):
    """Logger protocol for demonstration."""

    def info(self, message: str) -> None: ...
    def error(self, message: str) -> None: ...
    def debug(self, message: str) -> None: ...
    def warning(self, message: str) -> None: ...
    def critical(self, message: str) -> None: ...
    def exception(self, message: str) -> None: ...
    def trace(self, message: str) -> None: ...


class FlextPluginContext:
    """Plugin context for demonstration."""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_service(self, _service_name: str) -> object:
        """Get service by name."""
        return object()

    def get_config(self, _key: str) -> object:
        """Get configuration value."""
        return object()

    def get_logger(self) -> FlextLoggerProtocol:
        """Get logger instance."""
        return cast("FlextLoggerProtocol", object())


class FlextMessageHandler:
    """Message handler for demonstration."""

    def handle(self, message: object) -> None:
        """Handle message."""


class FlextEvent:
    """Event base class for demonstration."""

    def __init__(self, event_id: str, timestamp: float) -> None:
        self.event_id = event_id
        self.timestamp = timestamp


# =============================================================================
# CONSTANTS - Using FlextConstants hierarchical patterns
# =============================================================================

# Network port validation constants using FlextConstants
MAX_TCP_PORT: FlextTypes.Core.Integer = FlextConstants.Network.MAX_PORT or 65535
MIN_AGE: FlextTypes.Core.Integer = 18  # Default age validation
MAX_AGE: FlextTypes.Core.Integer = 120  # Default max age validation

# Logger using centralized logging
logger = get_logger("flext.examples.interfaces")

# =============================================================================
# DOMAIN MODELS - Business entities for examples
# =============================================================================


@dataclass
class User:
    """User domain model using FlextTypes."""

    id: FlextTypes.Core.String
    name: FlextTypes.Core.String
    email: FlextTypes.Core.String
    age: FlextTypes.Core.Integer | None = None
    is_active: FlextTypes.Core.Boolean = True


@dataclass
class Product:
    """Product domain model using FlextTypes."""

    id: FlextTypes.Core.String
    name: FlextTypes.Core.String
    price: FlextTypes.Core.Float
    category: FlextTypes.Core.String
    stock: FlextTypes.Core.Integer = 0


@dataclass
class Order:
    """Order domain model using FlextTypes."""

    id: FlextTypes.Core.String
    user_id: FlextTypes.Core.String
    products: list[FlextTypes.Core.String]
    total: FlextTypes.Core.Float
    status: FlextTypes.Core.String = "pending"


# Domain Events
class UserCreatedEvent(FlextEvent):
    """Event indicating user was created."""

    def __init__(self, event_id: str, user_id: str, name: str, email: str) -> None:
        """Initialize UserCreatedEvent."""
        super().__init__(event_id, time.time())
        self.user_id = user_id
        self.name = name
        self.email = email
        self.event_type = "user_created"
        self.aggregate_id = user_id
        self.event_version = 1

    def to_dict(self) -> dict[str, object]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "event_version": self.event_version,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
        }


class OrderPlacedEvent(FlextEvent):
    """Event indicating order was placed."""

    def __init__(
        self, event_id: str, order_id: str, user_id: str, total: float
    ) -> None:
        """Initialize OrderPlacedEvent."""
        super().__init__(event_id, time.time())
        self.order_id = order_id
        self.user_id = user_id
        self.total = total
        self.event_type = "order_placed"
        self.aggregate_id = order_id
        self.event_version = 1

    def to_dict(self) -> dict[str, object]:
        """Convert event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "event_version": self.event_version,
            "timestamp": self.timestamp,
            "order_id": self.order_id,
            "user_id": self.user_id,
            "total": self.total,
        }


# =============================================================================
# VALIDATION INTERFACES IMPLEMENTATION
# =============================================================================


class EmailValidator:
    """Protocol-compliant email validator demonstrating FlextValidator."""

    def validate(self, value: object) -> FlextResult[object]:
        """Validate email format and normalize."""
        if not isinstance(value, str):
            return FlextResult[object].fail("Email must be a string")

        email = value.strip().lower()

        if not email:
            return FlextResult[object].fail("Email cannot be empty")

        if "@" not in email:
            return FlextResult[object].fail("Email must contain @ symbol")

        local, domain = email.split("@", 1)

        if not local or not domain:
            return FlextResult[object].fail("Email must have local and domain parts")

        if "." not in domain:
            return FlextResult[object].fail("Domain must contain at least one dot")

        return FlextResult[object].ok(email)


class AgeRangeRule:
    """Age validation rule demonstrating FlextProtocols.Foundation.Validator protocol."""

    def __init__(
        self,
        min_age: FlextTypes.Core.Integer = MIN_AGE,
        max_age: FlextTypes.Core.Integer = MAX_AGE,
    ) -> None:
        """Initialize AgeRangeRule.

        Args:
            min_age: Minimum age
            max_age: Maximum age

        """
        self.min_age = min_age
        self.max_age = max_age

    def validate(self, data: object) -> FlextResult[None]:
        """Validate age data."""
        if not isinstance(data, int):
            return FlextResult[None].fail("Age must be an integer")
        if not (self.min_age <= data <= self.max_age):
            return FlextResult[None].fail(
                f"Age must be between {self.min_age} and {self.max_age}"
            )
        return FlextResult[None].ok(None)

    def apply(self, value: object, field_name: str) -> FlextResult[object]:
        """Apply rule and return value when valid."""
        if not isinstance(value, int):
            return FlextResult[object].fail(f"{field_name} must be an integer")
        if not (self.min_age <= value <= self.max_age):
            return FlextResult[object].fail(
                f"{field_name} must be between {self.min_age} and {self.max_age}",
            )
        return FlextResult[object].ok(value)

    def get_error_message(self, field_name: str, value: object) -> str:
        """Provide rule error message."""
        del value  # Unused parameter
        return f"{field_name} must be between {self.min_age} and {self.max_age}"


class NonEmptyStringRule:
    """Non-empty string validation rule implementing FlextProtocols.Foundation.Validator protocol."""

    def validate(self, data: object) -> FlextResult[None]:
        """Validate non-empty string data."""
        if not isinstance(data, str) or len(data.strip()) == 0:
            return FlextResult[None].fail("Must be a non-empty string")
        return FlextResult[None].ok(None)

    def apply(self, value: object, field_name: str) -> FlextResult[object]:
        """Apply rule and return value when valid."""
        if not isinstance(value, str) or len(value.strip()) == 0:
            return FlextResult[object].fail(f"{field_name} must be a non-empty string")
        return FlextResult[object].ok(value)

    def get_error_message(self, field_name: str, value: object) -> str:
        """Provide rule error message."""
        del value  # Unused parameter
        return f"{field_name} must be a non-empty string"


class PositiveNumberRule:
    """Positive number validation rule implementing FlextProtocols.Foundation.Validator protocol."""

    def validate(self, data: object) -> FlextResult[None]:
        """Validate positive number data."""
        if not isinstance(data, (int, float)) or data <= 0:
            return FlextResult[None].fail("Must be a positive number")
        return FlextResult[None].ok(None)

    def apply(self, value: object, field_name: str) -> FlextResult[object]:
        """Apply rule and return value when valid."""
        if not isinstance(value, (int, float)) or value <= 0:
            return FlextResult[object].fail(f"{field_name} must be a positive number")
        return FlextResult[object].ok(value)

    def get_error_message(self, field_name: str, value: object) -> str:
        """Provide rule error message."""
        del value  # Unused parameter
        return f"{field_name} must be a positive number"


# =============================================================================
# SERVICE INTERFACES IMPLEMENTATION
# =============================================================================


class UserService:
    """User service implementing FlextService protocol for lifecycle management."""

    def __init__(self) -> None:
        """Initialize UserService."""
        self._users: dict[str, User] = {}
        self._is_running = False
        self._next_id = 1

    def __call__(self, *args: object, **kwargs: object) -> FlextResult[None]:  # noqa: ARG002
        """Callable interface for service invocation."""
        # Simple implementation for protocol compliance
        if not self._is_running:
            return FlextResult[None].fail("Service not started")
        return FlextResult[None].ok(None)

    def start(self) -> FlextResult[None]:
        """Start the user service."""
        if self._is_running:
            return FlextResult[None].fail("Service is already running")

        self._is_running = True
        return FlextResult[None].ok(None)

    def stop(self) -> FlextResult[None]:
        """Stop the user service."""
        if not self._is_running:
            return FlextResult[None].fail("Service is not running")

        self._is_running = False
        return FlextResult[None].ok(None)

    def health_check(self) -> FlextResult[FlextTypes.Core.Dict]:
        """Check user service health."""
        health_status: FlextTypes.Core.Dict = {
            "service": "UserService",
            "status": "healthy" if self._is_running else "stopped",
            "total_users": len(self._users),
            "uptime_status": "running" if self._is_running else "stopped",
        }
        return FlextResult[FlextTypes.Core.Dict].ok(health_status)

    def create_user(
        self,
        name: FlextTypes.Core.String,
        email: FlextTypes.Core.String,
        age: FlextTypes.Core.Integer | None = None,
    ) -> FlextResult[User]:
        """Create new user."""
        if not self._is_running:
            return FlextResult[User].fail("Service is not running")

        user_id = f"user_{self._next_id}"
        self._next_id += 1

        user = User(id=user_id, name=name, email=email, age=age)
        self._users[user_id] = user

        return FlextResult[User].ok(user)

    def get_user(self, user_id: FlextTypes.Core.String) -> FlextResult[User]:
        """Get user by ID."""
        if not self._is_running:
            return FlextResult[User].fail("Service is not running")

        if user_id not in self._users:
            return FlextResult[User].fail(f"User {user_id} not found")

        return FlextResult[User].ok(self._users[user_id])


class ConfigurableEmailService:
    """Email service demonstrating FlextConfigurable protocol."""

    def __init__(self) -> None:
        """Initialize ConfigurableEmailService."""
        self._smtp_host = "localhost"
        self._smtp_port = 587
        self._username = ""
        self._from_email = "noreply@example.com"
        self._configured = False

    def configure(self, settings: Mapping[str, object]) -> FlextResult[None]:
        """Configure email service with settings."""
        try:
            if "smtp_host" in settings:
                self._smtp_host = str(settings["smtp_host"])

            if "smtp_port" in settings:
                port = settings["smtp_port"]
                if isinstance(port, int) and 1 <= port <= MAX_TCP_PORT:
                    self._smtp_port = port
                else:
                    return FlextResult[None].fail("Invalid SMTP port")

            if "username" in settings:
                self._username = str(settings["username"])

            if "from_email" in settings:
                self._from_email = str(settings["from_email"])

            self._configured = True
            return FlextResult[None].ok(None)

        except (ValueError, TypeError, KeyError) as e:
            return FlextResult[None].fail(f"Configuration failed: {e}")

    def send_email(
        self,
        _to: FlextTypes.Core.String,
        _subject: FlextTypes.Core.String,
        _body: FlextTypes.Core.String,
    ) -> FlextResult[None]:
        """Send email (simulated)."""
        if not self._configured:
            return FlextResult[None].fail("Service not configured")

        return FlextResult[None].ok(None)


# =============================================================================
# HANDLER INTERFACES IMPLEMENTATION
# =============================================================================


class UserCommandHandler(FlextBaseHandler):
    """User command handler demonstrating FlextMessageHandler."""

    def __init__(self, user_service: UserService) -> None:
        """Initialize UserCommandHandler.

        Args:
            user_service: User service instance

        """
        self._user_service = user_service

    def can_handle(self, message_type: object) -> bool:
        """Check if can handle user-related messages."""
        return hasattr(message_type, "type") and str(
            getattr(message_type, "type", ""),
        ).startswith("user")

    @property
    def handler_name(self) -> str:
        """Handler name for registration."""
        return "user_command_handler"

    def handle_command(self, command: object) -> FlextResult[object]:
        """Handle command (abstract method implementation)."""
        return self.handle(command)

    def handle(self, request: object) -> FlextResult[object]:
        """Handle user commands."""
        if not hasattr(request, "type"):
            return FlextResult[object].fail("Message must have type attribute")

        message_type = request.type  # type: ignore[attr-defined]

        if message_type == "user_create":
            name = getattr(request, "name", "")
            email = getattr(request, "email", "")
            age = getattr(request, "age", None)

            result = self._user_service.create_user(name, email, age)
            return result.map(lambda user: user)

        if message_type == "user_get":
            user_id = getattr(request, "user_id", "")
            result = self._user_service.get_user(user_id)
            return result.map(lambda user: user)

        return FlextResult[object].fail(f"Unknown user command: {message_type}")


class LoggingMiddleware:
    """Logging middleware demonstrating FlextMiddleware."""

    def process(
        self,
        message: object,
        next_handler: "Callable[[object], FlextResult[object]]",
    ) -> FlextResult[object]:
        """Process message with logging."""
        getattr(message, "type", "unknown")

        # Process through next handler
        result = next_handler(message)

        if result.success:
            pass

        return result


class ValidationMiddleware:
    """Validation middleware with rule-based validation."""

    def __init__(self) -> None:
        """Initialize ValidationMiddleware."""
        self._validators: dict[
            str, list[FlextProtocols.Foundation.Validator[object]]
        ] = {}

    def process(
        self,
        message: object,
        next_handler: Callable[[object], FlextResult[object]],
    ) -> FlextResult[object]:
        """Process message with validation."""
        message_type = getattr(message, "type", "unknown")

        # Validate user creation
        if message_type == "user_create":
            name = getattr(message, "name", "")
            email = getattr(message, "email", "")
            age = getattr(message, "age", None)

            # Validate name
            # Apply a simple non-empty validation inline to satisfy protocol
            if not isinstance(name, str) or not name.strip():
                return FlextResult[object].fail(
                    "Name validation failed: Value must be a non-empty string",
                )

            # Validate email
            if not isinstance(email, str) or "@" not in email:
                return FlextResult[object].fail(
                    "Email validation failed: Email must contain @ symbol",
                )

            # Validate age if provided
            min_age, max_age = 18, 120
            if age is not None and (
                not isinstance(age, int) or not (min_age <= age <= max_age)
            ):
                return FlextResult[object].fail(
                    f"Age validation failed: Age must be between {min_age} and {max_age}",
                )

        # Continue to next handler
        return next_handler(message)


# =============================================================================
# REPOSITORY INTERFACES IMPLEMENTATION
# =============================================================================


class UserRepository:
    """User repository implementing FlextRepository[User] protocol."""

    def __init__(self) -> None:
        """Initialize UserRepository."""
        self._users: dict[str, User] = {}
        self._deleted_ids: set[str] = set()

    def find_by_id(self, entity_id: str) -> FlextResult[User]:
        """Find user by ID."""
        if entity_id in self._deleted_ids:
            return FlextResult[User].fail(f"User {entity_id} was deleted")

        if entity_id not in self._users:
            return FlextResult[User].fail(f"User {entity_id} not found")

        return FlextResult[User].ok(self._users[entity_id])

    def save(self, entity: User) -> FlextResult[User]:
        """Save user entity."""
        if entity.id in self._deleted_ids:
            return FlextResult[User].fail(f"Cannot save deleted user {entity.id}")

        self._users[entity.id] = entity
        return FlextResult[User].ok(entity)

    def get_by_id(self, entity_id: str) -> FlextResult[User | None]:
        """Get user by ID (implements abstract method)."""
        result = self.find_by_id(entity_id)
        if result.success:
            return FlextResult[User | None].ok(result.value)
        return FlextResult[User | None].ok(None)

    def find_all(self) -> FlextResult[list[User]]:
        """Find all users (implements abstract method)."""
        active_users = [
            user
            for user_id, user in self._users.items()
            if user_id not in self._deleted_ids
        ]
        return FlextResult[list[User]].ok(active_users)

    def delete(self, entity_id: str) -> FlextResult[None]:
        """Delete user by ID."""
        if entity_id not in self._users:
            return FlextResult[None].fail(f"User {entity_id} not found")

        del self._users[entity_id]
        self._deleted_ids.add(entity_id)
        return FlextResult[None].ok(None)


class DatabaseUnitOfWork(FlextUnitOfWork):
    """Database unit of work implementing FlextUnitOfWork protocol."""

    def __init__(self, user_repo: UserRepository) -> None:
        """Initialize DatabaseUnitOfWork.

        Args:
            user_repo: User repository instance

        """
        self._user_repo = user_repo
        self._committed = False
        self._rolled_back = False
        self._changes: list[tuple[str, object]] = []  # (operation, data)

    def add_change(self, operation: str, data: object) -> None:
        """Add change to unit of work."""
        self._changes.append((operation, data))

    def commit(self) -> FlextResult[None]:
        """Commit all changes."""
        if self._rolled_back:
            return FlextResult[None].fail("Unit of work was rolled back")

        if self._committed:
            return FlextResult[None].fail("Unit of work already committed")

        # Apply all changes
        for operation, data in self._changes:
            if operation == "save":
                if not isinstance(data, User):
                    return FlextResult[None].fail("Invalid data type for save")
                save_result = self._user_repo.save(data)
                if save_result.is_failure:
                    return FlextResult[None].fail(save_result.error or "save failed")
            elif operation == "delete":
                delete_result = self._user_repo.delete(str(data))
                if delete_result.is_failure:
                    return delete_result

        self._committed = True
        return FlextResult[None].ok(None)

    def rollback(self) -> FlextResult[None]:
        """Rollback all changes."""
        if self._committed:
            return FlextResult[None].fail("Cannot rollback committed unit of work")

        self._changes.clear()
        self._rolled_back = True
        return FlextResult[None].ok(None)

    def begin(self) -> FlextResult[None]:
        """Begin transaction (no-op for in-memory demo)."""
        self._committed = False
        self._rolled_back = False
        return FlextResult[None].ok(None)

    def __enter__(self) -> "DatabaseUnitOfWork":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit context with automatic rollback on error."""
        if exc_type is not None and not self._committed:
            self.rollback()


# =============================================================================
# PLUGIN INTERFACES IMPLEMENTATION
# =============================================================================


# Simple logger mock to satisfy BoundLogger interface
class MockLogger:
    """Mock logger that behaves like BoundLogger."""

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        """Log info message.

        Args:
            message: Message to log.
            *args: Positional arguments for formatting.
            **kwargs: Key-value pairs to append as structured context.

        """
        if args:
            message %= args
        if kwargs:
            # Append key=value pairs for kwargs to mimic structured logging
            context = " ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} {context}".strip()

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        """Log error message.

        Args:
            message: Message to log.
            *args: Positional arguments for formatting.
            **kwargs: Key-value pairs to append as structured context.

        """
        if args:
            message %= args
        if kwargs:
            context = " ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} {context}".strip()

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        """Log debug message.

        Args:
            message: Message to log.
            *args: Positional arguments for formatting.
            **kwargs: Key-value pairs to append as structured context.

        """
        if args:
            message %= args
        if kwargs:
            context = " ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} {context}".strip()

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        """Log warning message.

        Args:
            message: Message to log.
            *args: Positional arguments for formatting.
            **kwargs: Key-value pairs to append as structured context.

        """
        if args:
            message %= args
        if kwargs:
            context = " ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} {context}".strip()

    def critical(self, message: str, *args: object, **kwargs: object) -> None:
        """Log critical message.

        Args:
            message: Message to log.
            *args: Positional arguments for formatting.
            **kwargs: Key-value pairs to append as structured context.

        """
        if args:
            message %= args
        if kwargs:
            context = " ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} {context}".strip()

    # Mock the other methods that BoundLogger might have
    def bind(self, **_kwargs: object) -> "BoundLogger":
        """Bind additional context (mock implementation)."""
        # Return a new MockLogger to satisfy BoundLogger interface
        return cast("BoundLogger", MockLogger())

    def unbind(self, *_keys: str) -> "BoundLogger":
        """Unbind context keys (mock implementation)."""
        # Return a new MockLogger to satisfy BoundLogger interface
        return cast("BoundLogger", MockLogger())


class SimplePluginContext(FlextPluginContext):
    """Simple plugin context demonstrating FlextProtocols.PluginContext protocol."""

    def __init__(self, config: dict[str, object] | None = None) -> None:
        """Initialize SimplePluginContext.

        Args:
            config: Plugin configuration

        """
        self._config = config or {}
        self._services: dict[str, object] = {}
        self._logger = MockLogger()

    def get_logger(self) -> FlextLoggerProtocol:
        """Get logger for plugin (simplified)."""
        return cast("FlextLoggerProtocol", self._logger)

    def get_config(self, key: str = "") -> dict[str, object] | object:
        """Get plugin configuration."""
        if key:
            return self._config.get(key, {})
        return dict(self._config)

    def get_service(self, service_name: str) -> FlextResult[object]:
        """Get service by name."""
        if service_name not in self._services:
            return FlextResult[object].fail(f"Service {service_name} not found")

        return FlextResult[object].ok(self._services[service_name])

    def register_service(self, service_name: str, service: object) -> None:
        """Register service (helper method)."""
        self._services[service_name] = service


class EmailNotificationPlugin:
    """Email notification plugin implementing FlextPlugin protocol."""

    def __init__(self) -> None:
        """Initialize EmailNotificationPlugin."""
        self._email_service: ConfigurableEmailService | None = None
        self._initialized = False

    @property
    def name(self) -> str:
        """Plugin name."""
        return "email_notification"

    @property
    def version(self) -> str:
        """Plugin version."""
        return "0.9.0"

    def initialize(self, context: FlextPluginContext) -> FlextResult[None]:
        """Initialize plugin with context."""
        try:
            # Get email service from context
            email_service_result = context.get_service("email_service")
            if (
                isinstance(email_service_result, FlextResult)
                and email_service_result.is_failure
            ):
                return FlextResult[None].fail("Email service not available")

            service_data = (
                email_service_result.value
                if isinstance(email_service_result, FlextResult)
                else email_service_result
            )
            if isinstance(service_data, ConfigurableEmailService):
                self._email_service = service_data
            else:
                return FlextResult[None].fail("Invalid email service type")

            # Configure email service from plugin config
            config_raw = context.get_config("")
            config = config_raw if isinstance(config_raw, dict) else {}
            if isinstance(self._email_service, ConfigurableEmailService):
                config_result = self._email_service.configure(config)
                if config_result.is_failure:
                    return config_result

            self._initialized = True
            context.get_logger().info(
                f"Plugin initialized: {self.name} v{self.version}"
            )
            return FlextResult[None].ok(None)

        except (ValueError, TypeError, ImportError) as e:
            return FlextResult[None].fail(f"Plugin initialization failed: {e}")

    def get_info(self) -> dict[str, object]:
        """Get plugin information (implements abstract method)."""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Email notification plugin for demonstrations",
            "initialized": self._initialized,
            "email_service_available": self._email_service is not None,
        }

    def shutdown(self) -> FlextResult[None]:
        """Shutdown plugin cleanly."""
        if not self._initialized:
            return FlextResult[None].fail("Plugin not initialized")

        self._email_service = None
        self._initialized = False
        return FlextResult[None].ok(None)

    def send_welcome_email(self, user: User) -> FlextResult[None]:
        """Send welcome email to user."""
        if not self._initialized or not self._email_service:
            return FlextResult[None].fail("Plugin not properly initialized")

        return self._email_service.send_email(
            user.email,
            "Welcome!",
            f"Welcome to our platform, {user.name}!",
        )


class AuditLogPlugin:
    """Audit log plugin implementing FlextPlugin protocol."""

    def __init__(self) -> None:
        """Initialize AuditLogPlugin."""
        self._audit_log: list[dict[str, object]] = []
        self._initialized = False

    @property
    def name(self) -> str:
        """Plugin name."""
        return "audit_log"

    @property
    def version(self) -> str:
        """Plugin version."""
        return "0.9.0"

    def initialize(self, context: FlextPluginContext) -> FlextResult[None]:
        """Initialize plugin with context."""
        self._initialized = True
        context.get_logger().info(f"Plugin initialized: {self.name} v{self.version}")
        return FlextResult[None].ok(None)

    def get_info(self) -> dict[str, object]:
        """Get plugin information (implements abstract method)."""
        return {
            "name": self.name,
            "version": self.version,
            "description": "Audit logging plugin for demonstrations",
            "initialized": self._initialized,
            "log_entries": len(self._audit_log),
        }

    def shutdown(self) -> FlextResult[None]:
        """Shutdown plugin cleanly."""
        if not self._initialized:
            return FlextResult[None].fail("Plugin not initialized")

        return FlextResult[None].ok(None)

    def log_event(
        self,
        event_type: str,
        details: dict[str, object],
    ) -> FlextResult[None]:
        """Log audit event."""
        if not self._initialized:
            return FlextResult[None].fail("Plugin not initialized")

        audit_entry: dict[str, object] = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details,
        }

        self._audit_log.append(audit_entry)
        return FlextResult[None].ok(None)


# =============================================================================
# EVENT INTERFACES IMPLEMENTATION
# =============================================================================


class SimpleEventPublisher:
    """Simple event publisher demonstrating FlextEventPublisher."""

    def __init__(self) -> None:
        """Initialize SimpleEventPublisher."""
        self._subscribers: dict[
            type[object], list[FlextProtocols.Application.MessageHandler]
        ] = {}
        logger.info("SimpleEventPublisher initialized")

    def publish(self, event: FlextEvent) -> FlextResult[None]:
        """Publish event to subscribers."""
        event_type = type(event)

        if event_type not in self._subscribers:
            return FlextResult[None].ok(None)

        handlers = self._subscribers[event_type]
        failed_handlers = []

        for handler in handlers:
            try:
                result = handler.handle(message=event)
                # Handle FlextResult return types for compatibility
                if hasattr(result, "is_failure") and getattr(
                    result, "is_failure", False
                ):
                    error_msg = getattr(result, "error", "Unknown error")
                    failed_handlers.append(
                        f"{type(handler).__name__}: {error_msg}",
                    )
            except Exception as e:
                failed_handlers.append(
                    f"{type(handler).__name__}: {e!s}",
                )

        if failed_handlers:
            return FlextResult[None].fail(
                f"Some handlers failed: {'; '.join(failed_handlers)}",
            )

        return FlextResult[None].ok(None)

    def publish_batch(self, events: list[FlextEvent]) -> FlextResult[None]:
        """Publish a batch of events sequentially."""
        for event in events:
            result = self.publish(event)
            if result.is_failure:
                return result
        return FlextResult[None].ok(None)

    def add_subscriber(
        self,
        event_type: type[object],
        handler: FlextProtocols.Application.MessageHandler,
    ) -> None:
        """Add subscriber (helper method)."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)


class SimpleEventSubscriber:
    """Simple event subscriber demonstrating FlextEventSubscriber."""

    def __init__(self, publisher: SimpleEventPublisher) -> None:
        """Initialize SimpleEventSubscriber."""
        self._publisher = publisher
        self._subscriptions: dict[
            type[object], list[FlextProtocols.Application.MessageHandler]
        ] = {}

    def subscribe(
        self,
        event_type: type[object],
        handler: FlextProtocols.Application.MessageHandler,
    ) -> FlextResult[None]:
        """Subscribe to event type."""
        try:
            self._publisher.add_subscriber(event_type, handler)

            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []
            self._subscriptions[event_type].append(handler)

            return FlextResult[None].ok(None)

        except (ValueError, TypeError, KeyError) as e:
            return FlextResult[None].fail(f"Subscription failed: {e}")

    def unsubscribe(
        self,
        event_type: type[object],
        handler: FlextProtocols.Application.MessageHandler,
    ) -> FlextResult[None]:
        """Unsubscribe from event type."""
        try:
            if (
                event_type in self._subscriptions
                and handler in self._subscriptions[event_type]
            ):
                self._subscriptions[event_type].remove(handler)

            return FlextResult[None].ok(None)

        except (ValueError, TypeError, KeyError) as e:
            return FlextResult[None].fail(f"Unsubscription failed: {e}")

    # Implement protocol-required methods
    def handle_event(self, event: FlextEvent) -> FlextResult[None]:  # noqa: ARG002
        """Handle incoming event and return success when processed."""
        logger.info("Event handled", event_type=type(event).__name__)
        return FlextResult[None].ok(None)

    def can_handle(self, event_type: str) -> bool:
        """Return True when this subscriber can handle the given event type."""
        del event_type
        return True


class UserEventHandler:
    """User event handler for event system demonstration."""

    def __init__(self, audit_plugin: AuditLogPlugin) -> None:
        """Initialize UserEventHandler."""
        self._audit_plugin = audit_plugin

    def can_handle(self, message_type: type) -> bool:
        """Check if can handle user events."""
        try:
            return issubclass(message_type, (UserCreatedEvent, OrderPlacedEvent))
        except TypeError:
            return False

    @property
    def handler_name(self) -> str:
        """Handler name for registration."""
        return "user_event_handler"

    def handle(self, message: object) -> object:
        """Handle user events."""
        if isinstance(message, UserCreatedEvent):
            self._audit_plugin.log_event(
                "user_created",
                {
                    "user_id": message.user_id,
                    "name": message.name,
                    "email": message.email,
                },
            )
            return FlextResult[None].ok(None)

        if isinstance(message, OrderPlacedEvent):
            self._audit_plugin.log_event(
                "order_placed",
                {
                    "order_id": message.order_id,
                    "user_id": message.user_id,
                    "total": message.total,
                },
            )
            return FlextResult[None].ok(None)

        return FlextResult[None].fail("Unsupported message type")


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================


def demonstrate_validation_interfaces() -> None:
    """Demonstrate validation interfaces with protocols and ABCs."""
    # 1. Protocol-based validation

    email_validator = EmailValidator()

    # Test valid email
    result = email_validator.validate("  User@Example.COM  ")
    if result.success:
        pass

    # Test invalid email
    result = email_validator.validate("invalid-email")
    if result.success:
        pass

    # Runtime type checking

    # 2. Rule-based validation

    age_rule = AgeRangeRule(min_age=18, max_age=65)
    name_rule = NonEmptyStringRule()
    price_rule = PositiveNumberRule()

    # Test validation rules
    test_cases = [
        (age_rule, 25, "Valid age"),
        (age_rule, 16, "Too young"),
        (age_rule, 70, "Too old"),
        (name_rule, "John Doe", "Valid name"),
        (name_rule, "", "Empty name"),
        (name_rule, 123, "Non-string name"),
        (price_rule, 99.99, "Valid price"),
        (price_rule, -10, "Negative price"),
        (price_rule, "not a number", "Non-numeric price"),
    ]

    for rule, value, _description in test_cases:
        apply_result = rule.apply(value, "value")
        is_valid = apply_result.success
        "" if is_valid else f" - {rule.get_error_message('value', value)}"


def demonstrate_service_interfaces() -> None:
    """Demonstrate service interfaces with lifecycle and configuration."""
    # 1. Service lifecycle management

    user_service = UserService()

    # Test health check before start
    health = user_service.health_check()
    if health.success:
        pass

    # Start service
    result = user_service.start()
    if result.success:
        pass

    # Health check after start
    health = user_service.health_check()
    if health.success:
        pass

    # Use service
    user_result = user_service.create_user("Alice Johnson", "alice@example.com", 28)
    if user_result.success:
        user = user_result.value
        if user is not None:
            pass

    # Stop service
    result = user_service.stop()
    if result.success:
        pass

    # 2. Configurable service demonstration

    email_service = ConfigurableEmailService()

    # Test before configuration
    result = email_service.send_email("test@example.com", "Test", "Body")
    if result.is_failure:
        pass

    # Configure service
    config = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "myapp@example.com",
        "from_email": "noreply@myapp.com",
    }

    config_result = email_service.configure(config)
    if config_result.success:
        pass

    # Test after configuration
    result = email_service.send_email(
        "user@example.com",
        "Welcome!",
        "Welcome to our service!",
    )
    if result.success:
        pass

    # Test runtime protocol checking


def demonstrate_handler_interfaces() -> None:
    """Demonstrate handler and middleware interfaces."""
    # 1. Basic handler demonstration

    user_service = UserService()
    user_service.start()

    handler = UserCommandHandler(user_service)

    # Create mock message objects
    class MockMessage:
        def __init__(self, **kwargs: object) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    # Test create user command
    create_message = MockMessage(
        type="user_create",
        name="Bob Wilson",
        email="bob@example.com",
        age=35,
    )

    can_handle = handler.can_handle(create_message)

    if can_handle:
        result = handler.handle(create_message)
        if result.success:
            user = result.value
            if hasattr(user, "name") and hasattr(user, "id"):
                pass

    # 2. Middleware pipeline demonstration

    # Create middleware instances
    logging_middleware = LoggingMiddleware()
    validation_middleware = ValidationMiddleware()

    # Test valid message through pipeline
    valid_message = MockMessage(
        type="user_create",
        name="Carol Brown",
        email="carol@example.com",
        age=42,
    )

    # Process through validation middleware first, then logging
    result = validation_middleware.process(valid_message, handler.handle)
    if result.success:
        result = logging_middleware.process(valid_message, handler.handle)
        if result.success:
            user = result.value
            if hasattr(user, "name") and hasattr(user, "id"):
                pass

    # Test invalid message through pipeline
    invalid_message = MockMessage(
        type="user_create",
        name="",  # Invalid empty name
        email="david@example.com",
        age=25,
    )

    result = validation_middleware.process(invalid_message, handler.handle)
    if result.is_failure:
        pass

    user_service.stop()


def demonstrate_repository_interfaces() -> None:
    """Demonstrate repository and unit of work interfaces."""
    _print_repo_header()
    user_repo = UserRepository()
    _basic_repo_operations(user_repo)
    fresh_repo = UserRepository()
    _unit_of_work_success_flow(fresh_repo)
    _unit_of_work_failure_flow(fresh_repo)


def _print_repo_header() -> None:
    pass


def _basic_repo_operations(user_repo: UserRepository) -> None:
    users = [
        User("user_1", "Alice", "alice@example.com", 25),
        User("user_2", "Bob", "bob@example.com", 30),
        User("user_3", "Carol", "carol@example.com", 35),
    ]
    for user in users:
        save_result = user_repo.save(user)
        if save_result.success:
            pass
    for user_id in ["user_1", "user_999", "user_2"]:
        result = user_repo.find_by_id(user_id)
        if result.success:
            user_data = result.value
            if (
                user_data is not None
                and hasattr(user_data, "name")
                and hasattr(user_data, "id")
            ):
                pass
    delete_result = user_repo.delete("user_2")
    if delete_result.success:
        pass
    result = user_repo.find_by_id("user_2")
    if result.is_failure:
        pass


def _unit_of_work_success_flow(fresh_repo: UserRepository) -> None:
    with DatabaseUnitOfWork(fresh_repo) as uow:
        new_user = User("user_100", "Transaction User", "transaction@example.com", 40)
        uow.add_change("save", new_user)  # type: ignore[attr-defined]
        commit_result = uow.commit()
        if commit_result.success:
            pass
    result = fresh_repo.find_by_id("user_100")
    if result.success and result.value is not None:
        pass


def _unit_of_work_failure_flow(fresh_repo: UserRepository) -> None:
    def _simulate_transaction_error() -> None:
        msg = "Simulated transaction error"
        raise ValueError(msg)

    try:
        with DatabaseUnitOfWork(fresh_repo) as uow:
            failing_user = User("user_101", "Failing User", "fail@example.com", 50)
            uow.add_change("save", failing_user)  # type: ignore[attr-defined]
            _simulate_transaction_error()
    except ValueError:
        pass
    result = fresh_repo.find_by_id("user_101")
    if result.is_failure:
        pass


def demonstrate_plugin_interfaces() -> None:
    """Demonstrate plugin interfaces with extensibility."""
    # 1. Plugin context setup

    # Create context with services
    context = SimplePluginContext(
        {
            "smtp_host": "mail.example.com",
            "smtp_port": 587,
            "from_email": "plugins@example.com",
        },
    )

    # Register services in context
    email_service = ConfigurableEmailService()
    context.register_service("email_service", email_service)

    # 2. Plugin initialization and usage

    # Initialize email notification plugin
    email_plugin = EmailNotificationPlugin()

    init_result = email_plugin.initialize(context)
    if init_result.success:
        pass

    # Initialize audit log plugin
    audit_plugin = AuditLogPlugin()

    init_result = audit_plugin.initialize(context)
    if init_result.success:
        pass

    # 3. Plugin usage

    # Use email plugin
    test_user = User("plugin_user", "Plugin Test User", "plugintest@example.com", 30)
    email_result = email_plugin.send_welcome_email(test_user)
    if email_result.success:
        pass

    # Use audit plugin
    audit_result = audit_plugin.log_event(
        "plugin_test",
        {
            "user_id": test_user.id,
            "action": "welcome_email_sent",
        },
    )
    if audit_result.success:
        pass

    # 4. Plugin shutdown

    shutdown_result = email_plugin.shutdown()
    if shutdown_result.success:
        pass

    shutdown_result = audit_plugin.shutdown()
    if shutdown_result.success:
        pass


def demonstrate_event_interfaces() -> None:
    """Demonstrate event interfaces with publish-subscribe patterns."""
    # 1. Event system setup

    publisher = SimpleEventPublisher()
    subscriber = SimpleEventSubscriber(publisher)

    # Create audit plugin for event handling
    audit_plugin = AuditLogPlugin()
    context = SimplePluginContext({})
    audit_plugin.initialize(context)

    # Create event handler
    event_handler = UserEventHandler(audit_plugin)

    # 2. Event subscription

    # Subscribe to events
    result = subscriber.subscribe(UserCreatedEvent, event_handler)
    if result.success:
        pass

    result = subscriber.subscribe(OrderPlacedEvent, event_handler)
    if result.success:
        pass

    # 3. Event publishing

    # Publish user created event
    user_event = UserCreatedEvent(
        event_id="evt_user_001",
        user_id="event_user_1",
        name="Event User",
        email="eventuser@example.com",
    )

    result = publisher.publish(user_event)
    if result.success:
        pass

    # Publish order placed event
    order_event = OrderPlacedEvent(
        event_id="evt_order_001",
        order_id="event_order_1",
        user_id="event_user_1",
        total=129.99,
    )

    result = publisher.publish(order_event)
    if result.success:
        pass

    # Publish event with no subscribers
    @dataclass
    class UnknownEvent(FlextEvent):
        # Domain event fields (FlextDomainEvent protocol)
        event_id: str
        event_type: str
        aggregate_id: str
        event_version: int
        timestamp: str

        # Event-specific fields
        event_data: str

        def to_dict(self) -> dict[str, object]:
            """Convert event to dictionary."""
            return {
                "event_id": self.event_id,
                "event_type": self.event_type,
                "aggregate_id": self.aggregate_id,
                "event_version": self.event_version,
                "timestamp": self.timestamp,
                "data": self.event_data,
            }

    unknown_event = UnknownEvent(
        event_id="evt_001",
        event_type="unknown",
        aggregate_id="test",
        event_version=1,
        timestamp="2023-01-01T00:00:00Z",
        event_data="test data",
    )
    result = publisher.publish(unknown_event)
    if result.success:
        pass

    # 4. Event unsubscription

    result = subscriber.unsubscribe(UserCreatedEvent, event_handler)
    if result.success:
        pass

    # Publish event after unsubscription
    user_event2 = UserCreatedEvent(
        event_id="evt_user_002",
        user_id="event_user_2",
        name="Second Event User",
        email="eventuser2@example.com",
    )

    result = publisher.publish(user_event2)
    if result.success:
        pass


def main() -> None:
    """Execute all FlextInterfaces demonstrations."""
    try:
        demonstrate_validation_interfaces()
        demonstrate_service_interfaces()
        demonstrate_handler_interfaces()
        demonstrate_repository_interfaces()
        demonstrate_plugin_interfaces()
        demonstrate_event_interfaces()

    except (ValueError, TypeError, ImportError, AttributeError):
        traceback.print_exc()


if __name__ == "__main__":
    main()
