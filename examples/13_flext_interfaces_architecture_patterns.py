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
    - FlextValidationRule: ABC for reusable validation rules
    - FlextService: ABC for service lifecycle management
    - FlextConfigurable: Protocol for configuration injection
    - FlextHandler/FlextMiddleware: ABCs for message processing pipelines
    - FlextRepository/FlextUnitOfWork: ABCs for data access patterns
    - FlextPlugin/FlextPluginContext: ABCs and protocols for extensibility
    - FlextEventPublisher/FlextEventSubscriber: ABCs for event-driven patterns

This example shows real-world enterprise architecture scenarios
demonstrating the power and flexibility of the FlextInterfaces system.
"""

import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

from flext_core.protocols import (
    FlextConfigurable,
    FlextEventPublisher,
    FlextEventSubscriber,
    FlextHandler,
    FlextMiddleware,
    FlextPlugin,
    FlextPluginContext,
    FlextRepository,
    FlextService,
    FlextUnitOfWork,
    FlextValidationRule,
    FlextValidator,
)
from flext_core.result import FlextResult
from flext_core.typings import TAnyDict

# =============================================================================
# INTERFACE CONSTANTS - Network and system constraints
# =============================================================================

# Network port validation constants
MAX_TCP_PORT = 65535  # Maximum valid TCP port number

# =============================================================================
# DOMAIN MODELS - Business entities for examples
# =============================================================================


@dataclass
class User:
    """User domain model."""

    id: str
    name: str
    email: str
    age: int | None = None
    is_active: bool = True


@dataclass
class Product:
    """Product domain model."""

    id: str
    name: str
    price: float
    category: str
    stock: int = 0


@dataclass
class Order:
    """Order domain model."""

    id: str
    user_id: str
    products: list[str]
    total: float
    status: str = "pending"


# Domain Events
@dataclass
class UserCreatedEvent:
    """Event indicating user was created."""

    user_id: str
    name: str
    email: str
    timestamp: float


@dataclass
class OrderPlacedEvent:
    """Event indicating order was placed."""

    order_id: str
    user_id: str
    total: float
    timestamp: float


# =============================================================================
# VALIDATION INTERFACES IMPLEMENTATION
# =============================================================================


class EmailValidator:
    """Protocol-compliant email validator demonstrating FlextValidator."""

    def validate(self, value: object) -> FlextResult[object]:
        """Validate email format and normalize."""
        if not isinstance(value, str):
            return FlextResult.fail("Email must be a string")

        email = value.strip().lower()

        if not email:
            return FlextResult.fail("Email cannot be empty")

        if "@" not in email:
            return FlextResult.fail("Email must contain @ symbol")

        local, domain = email.split("@", 1)

        if not local or not domain:
            return FlextResult.fail("Email must have local and domain parts")

        if "." not in domain:
            return FlextResult.fail("Domain must contain at least one dot")

        return FlextResult.ok(email)


class AgeRangeRule(FlextValidationRule):
    """Age validation rule demonstrating FlextValidationRule."""

    def __init__(self, min_age: int = 18, max_age: int = 120) -> None:
        """Initialize AgeRangeRule.

        Args:
            min_age: Minimum age
            max_age: Maximum age

        """
        self.min_age = min_age
        self.max_age = max_age

    def check(self, value: object) -> bool:
        """Check if age is within valid range."""
        if not isinstance(value, int):
            return False
        return self.min_age <= value <= self.max_age

    def error_message(self) -> str:
        """Get age validation error message."""
        return f"Age must be between {self.min_age} and {self.max_age}"


class NonEmptyStringRule(FlextValidationRule):
    """Non-empty string validation rule."""

    def check(self, value: object) -> bool:
        """Check if value is non-empty string."""
        return isinstance(value, str) and len(value.strip()) > 0

    def error_message(self) -> str:
        """Get non-empty string error message."""
        return "Value must be a non-empty string"


class PositiveNumberRule(FlextValidationRule):
    """Positive number validation rule."""

    def check(self, value: object) -> bool:
        """Check if value is positive number."""
        return isinstance(value, int | float) and value > 0

    def error_message(self) -> str:
        """Get positive number error message."""
        return "Value must be a positive number"


# =============================================================================
# SERVICE INTERFACES IMPLEMENTATION
# =============================================================================


class UserService(FlextService):
    """User service demonstrating FlextService lifecycle management."""

    def __init__(self) -> None:
        """Initialize UserService."""
        self._users: dict[str, User] = {}
        self._is_running = False
        self._next_id = 1

    def start(self) -> FlextResult[None]:
        """Start the user service."""
        if self._is_running:
            return FlextResult.fail("Service is already running")

        self._is_running = True
        print("UserService started successfully")
        return FlextResult.ok(None)

    def stop(self) -> FlextResult[None]:
        """Stop the user service."""
        if not self._is_running:
            return FlextResult.fail("Service is not running")

        self._is_running = False
        print("UserService stopped successfully")
        return FlextResult.ok(None)

    def health_check(self) -> FlextResult[TAnyDict]:
        """Check user service health."""
        health_status: TAnyDict = {
            "service": "UserService",
            "status": "healthy" if self._is_running else "stopped",
            "total_users": len(self._users),
            "uptime_status": "running" if self._is_running else "stopped",
        }
        return FlextResult.ok(health_status)

    def create_user(
        self,
        name: str,
        email: str,
        age: int | None = None,
    ) -> FlextResult[User]:
        """Create new user."""
        if not self._is_running:
            return FlextResult.fail("Service is not running")

        user_id = f"user_{self._next_id}"
        self._next_id += 1

        user = User(id=user_id, name=name, email=email, age=age)
        self._users[user_id] = user

        return FlextResult.ok(user)

    def get_user(self, user_id: str) -> FlextResult[User]:
        """Get user by ID."""
        if not self._is_running:
            return FlextResult.fail("Service is not running")

        if user_id not in self._users:
            return FlextResult.fail(f"User {user_id} not found")

        return FlextResult.ok(self._users[user_id])


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
                    return FlextResult.fail("Invalid SMTP port")

            if "username" in settings:
                self._username = str(settings["username"])

            if "from_email" in settings:
                self._from_email = str(settings["from_email"])

            self._configured = True
            print(f"EmailService configured: {self._smtp_host}:{self._smtp_port}")
            return FlextResult.ok(None)

        except (ValueError, TypeError, KeyError) as e:
            return FlextResult.fail(f"Configuration failed: {e}")

    def send_email(self, to: str, subject: str, _body: str) -> FlextResult[None]:
        """Send email (simulated)."""
        if not self._configured:
            return FlextResult.fail("Service not configured")

        print(f"Sending email to {to}: {subject}")
        print(f"From: {self._from_email} via {self._smtp_host}:{self._smtp_port}")
        return FlextResult.ok(None)


# =============================================================================
# HANDLER INTERFACES IMPLEMENTATION
# =============================================================================


class UserCommandHandler(FlextHandler):
    """User command handler demonstrating FlextHandler."""

    def __init__(self, user_service: UserService) -> None:
        """Initialize UserCommandHandler.

        Args:
            user_service: User service instance

        """
        self._user_service = user_service

    def can_handle(self, message: object) -> bool:
        """Check if can handle user-related messages."""
        return hasattr(message, "type") and str(
            getattr(message, "type", ""),
        ).startswith("user")

    def handle(self, message: object) -> FlextResult[object]:
        """Handle user commands."""
        if not hasattr(message, "type"):
            return FlextResult.fail("Message must have type attribute")

        message_type = message.type

        if message_type == "user_create":
            name = getattr(message, "name", "")
            email = getattr(message, "email", "")
            age = getattr(message, "age", None)

            result = self._user_service.create_user(name, email, age)
            return result.map(lambda user: user)

        if message_type == "user_get":
            user_id = getattr(message, "user_id", "")
            result = self._user_service.get_user(user_id)
            return result.map(lambda user: user)

        return FlextResult.fail(f"Unknown user command: {message_type}")


class LoggingMiddleware(FlextMiddleware):
    """Logging middleware demonstrating FlextMiddleware."""

    def process(
        self,
        message: object,
        next_handler: FlextHandler,
    ) -> FlextResult[object]:
        """Process message with logging."""
        message_type = getattr(message, "type", "unknown")
        print(f"[MIDDLEWARE] Processing message: {message_type}")

        # Process through next handler
        result = next_handler.handle(message)

        if result.success:
            print(f"[MIDDLEWARE] Message {message_type} processed successfully")
        else:
            print(f"[MIDDLEWARE] Message {message_type} failed: {result.error}")

        return result


class ValidationMiddleware(FlextMiddleware):
    """Validation middleware with rule-based validation."""

    def __init__(self) -> None:
        """Initialize ValidationMiddleware."""
        self._validators: dict[str, list[FlextValidationRule]] = {
            "user_create": [
                NonEmptyStringRule(),  # For name and email
                AgeRangeRule(18, 120),  # For age if provided
            ],
        }

    def process(
        self,
        message: object,
        next_handler: FlextHandler,
    ) -> FlextResult[object]:
        """Process message with validation."""
        message_type = getattr(message, "type", "unknown")

        # Validate user creation
        if message_type == "user_create":
            name = getattr(message, "name", "")
            email = getattr(message, "email", "")
            age = getattr(message, "age", None)

            # Validate name
            name_rule = NonEmptyStringRule()
            if not name_rule.check(name):
                return FlextResult.fail(
                    f"Name validation failed: {name_rule.error_message()}",
                )

            # Validate email
            email_rule = NonEmptyStringRule()
            if not email_rule.check(email):
                return FlextResult.fail(
                    f"Email validation failed: {email_rule.error_message()}",
                )

            # Validate age if provided
            if age is not None:
                age_rule = AgeRangeRule()
                if not age_rule.check(age):
                    return FlextResult.fail(
                        f"Age validation failed: {age_rule.error_message()}",
                    )

        # Continue to next handler
        return next_handler.handle(message)


# =============================================================================
# REPOSITORY INTERFACES IMPLEMENTATION
# =============================================================================


class UserRepository(FlextRepository):
    """User repository demonstrating FlextRepository."""

    def __init__(self) -> None:
        """Initialize UserRepository."""
        self._users: dict[str, User] = {}
        self._deleted_ids: set[str] = set()

    def find_by_id(self, entity_id: str) -> FlextResult[object]:
        """Find user by ID."""
        if entity_id in self._deleted_ids:
            return FlextResult.fail(f"User {entity_id} was deleted")

        if entity_id not in self._users:
            return FlextResult.fail(f"User {entity_id} not found")

        return FlextResult.ok(self._users[entity_id])

    def save(self, entity: object) -> FlextResult[None]:
        """Save user entity."""
        if not isinstance(entity, User):
            return FlextResult.fail("Entity must be a User")

        if entity.id in self._deleted_ids:
            return FlextResult.fail(f"Cannot save deleted user {entity.id}")

        self._users[entity.id] = entity
        print(f"User {entity.id} saved to repository")
        return FlextResult.ok(None)

    def delete(self, entity_id: str) -> FlextResult[None]:
        """Delete user by ID."""
        if entity_id not in self._users:
            return FlextResult.fail(f"User {entity_id} not found")

        del self._users[entity_id]
        self._deleted_ids.add(entity_id)
        print(f"User {entity_id} deleted from repository")
        return FlextResult.ok(None)


class DatabaseUnitOfWork(FlextUnitOfWork):
    """Database unit of work demonstrating FlextUnitOfWork."""

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
            return FlextResult.fail("Unit of work was rolled back")

        if self._committed:
            return FlextResult.fail("Unit of work already committed")

        # Apply all changes
        for operation, data in self._changes:
            if operation == "save":
                result = self._user_repo.save(data)
                if result.is_failure:
                    return result
            elif operation == "delete":
                result = self._user_repo.delete(str(data))
                if result.is_failure:
                    return result

        self._committed = True
        print(f"Unit of work committed with {len(self._changes)} changes")
        return FlextResult.ok(None)

    def rollback(self) -> FlextResult[None]:
        """Rollback all changes."""
        if self._committed:
            return FlextResult.fail("Cannot rollback committed unit of work")

        self._changes.clear()
        self._rolled_back = True
        print("Unit of work rolled back")
        return FlextResult.ok(None)

    def __enter__(self) -> FlextUnitOfWork:
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

    def info(self, message: str, *args: object) -> None:
        """Log info message.

        Args:
            message: Message to log
            *args: Arguments to format message

        """
        if args:
            message %= args
        print(f"[INFO] {message}")

    def error(self, message: str, *args: object) -> None:
        """Log error message.

        Args:
            message: Message to log
            *args: Arguments to format message

        """
        if args:
            message %= args
        print(f"[ERROR] {message}")

    def debug(self, message: str, *args: object) -> None:
        """Log debug message.

        Args:
            message: Message to log
            *args: Arguments to format message

        """
        if args:
            message %= args
        print(f"[DEBUG] {message}")

    def warning(self, message: str, *args: object) -> None:
        """Log warning message.

        Args:
            message: Message to log
            *args: Arguments to format message

        """
        if args:
            message %= args
        print(f"[WARNING] {message}")

    def critical(self, message: str, *args: object) -> None:
        """Log critical message.

        Args:
            message: Message to log
            *args: Arguments to format message

        """
        if args:
            message %= args
        print(f"[CRITICAL] {message}")

    # Mock the other methods that BoundLogger might have
    def bind(self, **_kwargs: object) -> "BoundLogger":
        """Bind additional context (mock implementation)."""
        # Return a new MockLogger to satisfy BoundLogger interface
        return MockLogger()

    def unbind(self, *_keys: str) -> "BoundLogger":
        """Unbind context keys (mock implementation)."""
        # Return a new MockLogger to satisfy BoundLogger interface
        return MockLogger()


class SimplePluginContext:
    """Simple plugin context demonstrating FlextPluginContext protocol."""

    def __init__(self, config: dict[str, object] | None = None) -> None:
        """Initialize SimplePluginContext.

        Args:
            config: Plugin configuration

        """
        self._config = config or {}
        self._services: dict[str, object] = {}
        self._logger = MockLogger()

    @property
    def logger(self) -> "BoundLogger":
        """Get logger for plugin (simplified)."""
        # Return MockLogger which implements BoundLogger interface
        return self._logger

    @property
    def config(self) -> Mapping[str, object]:
        """Get plugin configuration."""
        return self._config

    def get_service(self, service_name: str) -> FlextResult[object]:
        """Get service by name."""
        if service_name not in self._services:
            return FlextResult.fail(f"Service {service_name} not found")

        return FlextResult.ok(self._services[service_name])

    def register_service(self, service_name: str, service: object) -> None:
        """Register service (helper method)."""
        self._services[service_name] = service


class EmailNotificationPlugin(FlextPlugin):
    """Email notification plugin demonstrating FlextPlugin."""

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
            if email_service_result.is_failure:
                return FlextResult.fail("Email service not available")

            service_data = email_service_result.data
            if isinstance(service_data, ConfigurableEmailService):
                self._email_service = service_data
            else:
                return FlextResult.fail("Invalid email service type")

            # Configure email service from plugin config
            config = dict(context.config)
            if isinstance(self._email_service, ConfigurableEmailService):
                config_result = self._email_service.configure(config)
                if config_result.is_failure:
                    return config_result

            self._initialized = True
            context.logger.info("Plugin %s v%s initialized", self.name, self.version)
            return FlextResult.ok(None)

        except (ValueError, TypeError, ImportError) as e:
            return FlextResult.fail(f"Plugin initialization failed: {e}")

    def shutdown(self) -> FlextResult[None]:
        """Shutdown plugin cleanly."""
        if not self._initialized:
            return FlextResult.fail("Plugin not initialized")

        self._email_service = None
        self._initialized = False
        print(f"Plugin {self.name} shut down")
        return FlextResult.ok(None)

    def send_welcome_email(self, user: User) -> FlextResult[None]:
        """Send welcome email to user."""
        if not self._initialized or not self._email_service:
            return FlextResult.fail("Plugin not properly initialized")

        return self._email_service.send_email(
            to=user.email,
            subject="Welcome!",
            _body=f"Welcome to our platform, {user.name}!",
        )


class AuditLogPlugin(FlextPlugin):
    """Audit log plugin demonstrating FlextPlugin."""

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
        context.logger.info("Plugin %s v%s initialized", self.name, self.version)
        return FlextResult.ok(None)

    def shutdown(self) -> FlextResult[None]:
        """Shutdown plugin cleanly."""
        if not self._initialized:
            return FlextResult.fail("Plugin not initialized")

        print(f"Plugin {self.name} shut down - {len(self._audit_log)} log entries")
        return FlextResult.ok(None)

    def log_event(
        self,
        event_type: str,
        details: dict[str, object],
    ) -> FlextResult[None]:
        """Log audit event."""
        if not self._initialized:
            return FlextResult.fail("Plugin not initialized")

        audit_entry = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details,
        }

        self._audit_log.append(audit_entry)
        print(f"Audit log: {event_type}")
        return FlextResult.ok(None)


# =============================================================================
# EVENT INTERFACES IMPLEMENTATION
# =============================================================================


class SimpleEventPublisher(FlextEventPublisher):
    """Simple event publisher demonstrating FlextEventPublisher."""

    def __init__(self) -> None:
        """Initialize SimpleEventPublisher."""
        self._subscribers: dict[type[object], list[FlextHandler]] = {}

    def publish(self, event: object) -> FlextResult[None]:
        """Publish event to subscribers."""
        event_type = type(event)

        if event_type not in self._subscribers:
            print(f"No subscribers for event {event_type.__name__}")
            return FlextResult.ok(None)

        handlers = self._subscribers[event_type]
        failed_handlers = []

        for handler in handlers:
            if handler.can_handle(event):
                result = handler.handle(event)
                if result.is_failure:
                    failed_handlers.append(
                        f"{handler.__class__.__name__}: {result.error}",
                    )

        if failed_handlers:
            return FlextResult.fail(
                f"Some handlers failed: {'; '.join(failed_handlers)}",
            )

        print(f"Event {event_type.__name__} published to {len(handlers)} handlers")
        return FlextResult.ok(None)

    def add_subscriber(self, event_type: type[object], handler: FlextHandler) -> None:
        """Add subscriber (helper method)."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)


class SimpleEventSubscriber(FlextEventSubscriber):
    """Simple event subscriber demonstrating FlextEventSubscriber."""

    def __init__(self, publisher: SimpleEventPublisher) -> None:
        """Initialize SimpleEventSubscriber."""
        self._publisher = publisher
        self._subscriptions: dict[type[object], list[FlextHandler]] = {}

    def subscribe(
        self,
        event_type: type[object],
        handler: FlextHandler,
    ) -> FlextResult[None]:
        """Subscribe to event type."""
        try:
            self._publisher.add_subscriber(event_type, handler)

            if event_type not in self._subscriptions:
                self._subscriptions[event_type] = []
            self._subscriptions[event_type].append(handler)

            print(
                f"Handler {handler.__class__.__name__} subscribed to "
                f"{event_type.__name__}",
            )
            return FlextResult.ok(None)

        except (ValueError, TypeError, KeyError) as e:
            return FlextResult.fail(f"Subscription failed: {e}")

    def unsubscribe(
        self,
        event_type: type[object],
        handler: FlextHandler,
    ) -> FlextResult[None]:
        """Unsubscribe from event type."""
        try:
            if (
                event_type in self._subscriptions
                and handler in self._subscriptions[event_type]
            ):
                self._subscriptions[event_type].remove(handler)
                print(
                    f"Handler {handler.__class__.__name__} unsubscribed from "
                    f"{event_type.__name__}",
                )

            return FlextResult.ok(None)

        except (ValueError, TypeError, KeyError) as e:
            return FlextResult.fail(f"Unsubscription failed: {e}")


class UserEventHandler(FlextHandler):
    """User event handler for event system demonstration."""

    def __init__(self, audit_plugin: AuditLogPlugin) -> None:
        """Initialize UserEventHandler."""
        self._audit_plugin = audit_plugin

    def can_handle(self, message: object) -> bool:
        """Check if can handle user events."""
        return isinstance(message, UserCreatedEvent | OrderPlacedEvent)

    def handle(self, message: object) -> FlextResult[object]:
        """Handle user events."""
        if isinstance(message, UserCreatedEvent):
            result = self._audit_plugin.log_event(
                "user_created",
                {
                    "user_id": message.user_id,
                    "name": message.name,
                    "email": message.email,
                },
            )
            return result.map(lambda _: None)

        if isinstance(message, OrderPlacedEvent):
            result = self._audit_plugin.log_event(
                "order_placed",
                {
                    "order_id": message.order_id,
                    "user_id": message.user_id,
                    "total": message.total,
                },
            )
            return result.map(lambda _: None)

        return FlextResult.fail(f"Unknown event type: {type(message)}")


# =============================================================================
# DEMONSTRATION FUNCTIONS
# =============================================================================


def demonstrate_validation_interfaces() -> None:
    """Demonstrate validation interfaces with protocols and ABCs."""
    print("\n" + "=" * 80)
    print("✅ VALIDATION INTERFACES - PROTOCOLS AND RULES")
    print("=" * 80)

    # 1. Protocol-based validation
    print("\n1. Protocol-based email validation:")

    email_validator = EmailValidator()

    # Test valid email
    result = email_validator.validate("  User@Example.COM  ")
    if result.success:
        print(f"✅ Valid email normalized: {result.data}")
    else:
        print(f"❌ Email validation failed: {result.error}")

    # Test invalid email
    result = email_validator.validate("invalid-email")
    if result.success:
        print(f"✅ Email validated: {result.data}")
    else:
        print(f"❌ Invalid email (expected): {result.error}")

    # Runtime type checking
    print(
        f"   Email validator is FlextValidator: "
        f"{isinstance(email_validator, FlextValidator)}",
    )

    # 2. Rule-based validation
    print("\n2. Rule-based validation with ABCs:")

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

    for rule, value, description in test_cases:
        is_valid = rule.check(value)
        status = "✅" if is_valid else "❌"
        error_msg = "" if is_valid else f" - {rule.error_message()}"
        print(f"   {status} {description}: {value}{error_msg}")

    print("✅ Validation interfaces demonstration completed")


def demonstrate_service_interfaces() -> None:
    """Demonstrate service interfaces with lifecycle and configuration."""
    print("\n" + "=" * 80)
    print("🔧 SERVICE INTERFACES - LIFECYCLE AND CONFIGURATION")
    print("=" * 80)

    # 1. Service lifecycle management
    print("\n1. Service lifecycle management:")

    user_service = UserService()

    # Test health check before start
    health = user_service.health_check()
    if health.success:
        print(f"   Health before start: {health.data}")

    # Start service
    result = user_service.start()
    if result.success:
        print("   ✅ Service started successfully")
    else:
        print(f"   ❌ Service start failed: {result.error}")

    # Health check after start
    health = user_service.health_check()
    if health.success:
        print(f"   Health after start: {health.data}")

    # Use service
    user_result = user_service.create_user("Alice Johnson", "alice@example.com", 28)
    if user_result.success:
        user = user_result.data
        if user is not None:
            print(f"   ✅ User created: {user.name} ({user.id})")

    # Stop service
    result = user_service.stop()
    if result.success:
        print("   ✅ Service stopped successfully")

    # 2. Configurable service demonstration
    print("\n2. Configurable service demonstration:")

    email_service = ConfigurableEmailService()

    # Test before configuration
    result = email_service.send_email("test@example.com", "Test", "Body")
    if result.is_failure:
        print(f"   ❌ Service not configured (expected): {result.error}")

    # Configure service
    config = {
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "username": "myapp@example.com",
        "from_email": "noreply@myapp.com",
    }

    config_result = email_service.configure(config)
    if config_result.success:
        print("   ✅ Email service configured successfully")
    else:
        print(f"   ❌ Configuration failed: {config_result.error}")

    # Test after configuration
    result = email_service.send_email(
        "user@example.com",
        "Welcome!",
        "Welcome to our service!",
    )
    if result.success:
        print("   ✅ Email sent successfully")
    else:
        print(f"   ❌ Email send failed: {result.error}")

    # Test runtime protocol checking
    print(
        f"   Email service is configurable: "
        f"{isinstance(email_service, FlextConfigurable)}",
    )

    print("✅ Service interfaces demonstration completed")


def demonstrate_handler_interfaces() -> None:
    """Demonstrate handler and middleware interfaces."""
    print("\n" + "=" * 80)
    print("⚙️ HANDLER INTERFACES - MIDDLEWARE PIPELINE")
    print("=" * 80)

    # 1. Basic handler demonstration
    print("\n1. Basic handler demonstration:")

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
    print(f"   Handler can handle create message: {can_handle}")

    if can_handle:
        result = handler.handle(create_message)
        if result.success:
            user = result.data
            if hasattr(user, "name") and hasattr(user, "id"):
                print(f"   ✅ User created via handler: {user.name} ({user.id})")
        else:
            print(f"   ❌ Handler failed: {result.error}")

    # 2. Middleware pipeline demonstration
    print("\n2. Middleware pipeline demonstration:")

    # Create middleware instances
    logging_middleware = LoggingMiddleware()
    validation_middleware = ValidationMiddleware()

    # Test valid message through pipeline
    print("   Testing valid message through pipeline:")
    valid_message = MockMessage(
        type="user_create",
        name="Carol Brown",
        email="carol@example.com",
        age=42,
    )

    # Process through validation middleware first, then logging
    result = validation_middleware.process(valid_message, handler)
    if result.success:
        result = logging_middleware.process(valid_message, handler)
        if result.success:
            user = result.data
            if hasattr(user, "name") and hasattr(user, "id"):
                print(f"   ✅ Pipeline success: {user.name} ({user.id})")

    # Test invalid message through pipeline
    print("   Testing invalid message through pipeline:")
    invalid_message = MockMessage(
        type="user_create",
        name="",  # Invalid empty name
        email="david@example.com",
        age=25,
    )

    result = validation_middleware.process(invalid_message, handler)
    if result.is_failure:
        print(f"   ❌ Validation middleware caught error (expected): {result.error}")

    user_service.stop()
    print("✅ Handler interfaces demonstration completed")


def demonstrate_repository_interfaces() -> None:
    """Demonstrate repository and unit of work interfaces."""
    print("\n" + "=" * 80)
    print("💾 REPOSITORY INTERFACES - DATA ACCESS PATTERNS")
    print("=" * 80)

    # 1. Basic repository operations
    print("\n1. Basic repository operations:")

    user_repo = UserRepository()

    # Create and save users
    users = [
        User("user_1", "Alice", "alice@example.com", 25),
        User("user_2", "Bob", "bob@example.com", 30),
        User("user_3", "Carol", "carol@example.com", 35),
    ]

    for user in users:
        save_result = user_repo.save(user)
        if save_result.success:
            print(f"   ✅ Saved user: {user.name}")
        else:
            print(f"   ❌ Save failed: {save_result.error}")

    # Find users
    print("   Finding users:")
    for user_id in ["user_1", "user_999", "user_2"]:
        result = user_repo.find_by_id(user_id)
        if result.success:
            user_data = result.data
            if (
                user_data is not None
                and hasattr(user_data, "name")
                and hasattr(user_data, "id")
            ):
                print(f"   ✅ Found: {user_data.name} ({user_data.id})")
        else:
            print(f"   ❌ Not found: {user_id} - {result.error}")

    # Delete user
    delete_result = user_repo.delete("user_2")
    if delete_result.success:
        print("   ✅ User deleted")

    # Try to find deleted user
    result = user_repo.find_by_id("user_2")
    if result.is_failure:
        print(f"   ❌ Deleted user not found (expected): {result.error}")

    # 2. Unit of Work pattern
    print("\n2. Unit of Work pattern:")

    fresh_repo = UserRepository()

    # Successful transaction
    print("   Successful transaction:")
    with DatabaseUnitOfWork(fresh_repo) as uow:
        new_user = User("user_100", "Transaction User", "transaction@example.com", 40)
        if hasattr(uow, "add_change"):
            uow.add_change("save", new_user)

        # Commit explicitly
        commit_result = uow.commit()
        if commit_result.success:
            print("   ✅ Transaction committed successfully")

    # Verify user was saved
    result = fresh_repo.find_by_id("user_100")
    if result.success:
        user_data = result.data
        if hasattr(user_data, "name"):
            print(f"   ✅ User persisted after commit: {user_data.name}")

    # Failed transaction (rollback)
    print("   Failed transaction with rollback:")

    def _simulate_transaction_error() -> None:
        """Simulate a transaction error to test rollback behavior."""
        msg = "Simulated transaction error"
        raise ValueError(msg)

    try:
        with DatabaseUnitOfWork(fresh_repo) as uow:
            failing_user = User("user_101", "Failing User", "fail@example.com", 50)
            if hasattr(uow, "add_change"):
                uow.add_change("save", failing_user)

            # Simulate error by raising exception
            _simulate_transaction_error()
    except ValueError as e:
        print(f"   ❌ Transaction failed (expected): {e}")

    # Verify user was not saved due to rollback
    result = fresh_repo.find_by_id("user_101")
    if result.is_failure:
        print("   ✅ User not persisted after rollback (expected)")

    print("✅ Repository interfaces demonstration completed")


def demonstrate_plugin_interfaces() -> None:
    """Demonstrate plugin interfaces with extensibility."""
    print("\n" + "=" * 80)
    print("🔌 PLUGIN INTERFACES - EXTENSIBILITY PATTERNS")
    print("=" * 80)

    # 1. Plugin context setup
    print("\n1. Plugin context and service setup:")

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

    print("   Plugin context created with email service")

    # 2. Plugin initialization and usage
    print("\n2. Plugin initialization and usage:")

    # Initialize email notification plugin
    email_plugin = EmailNotificationPlugin()
    print(f"   Plugin: {email_plugin.name} v{email_plugin.version}")

    init_result = email_plugin.initialize(context)
    if init_result.success:
        print("   ✅ Email notification plugin initialized")
    else:
        print(f"   ❌ Plugin initialization failed: {init_result.error}")

    # Initialize audit log plugin
    audit_plugin = AuditLogPlugin()
    print(f"   Plugin: {audit_plugin.name} v{audit_plugin.version}")

    init_result = audit_plugin.initialize(context)
    if init_result.success:
        print("   ✅ Audit log plugin initialized")

    # 3. Plugin usage
    print("\n3. Plugin usage:")

    # Use email plugin
    test_user = User("plugin_user", "Plugin Test User", "plugintest@example.com", 30)
    email_result = email_plugin.send_welcome_email(test_user)
    if email_result.success:
        print("   ✅ Welcome email sent via plugin")
    else:
        print(f"   ❌ Email send failed: {email_result.error}")

    # Use audit plugin
    audit_result = audit_plugin.log_event(
        "plugin_test",
        {
            "user_id": test_user.id,
            "action": "welcome_email_sent",
        },
    )
    if audit_result.success:
        print("   ✅ Event logged via audit plugin")

    # 4. Plugin shutdown
    print("\n4. Plugin shutdown:")

    shutdown_result = email_plugin.shutdown()
    if shutdown_result.success:
        print("   ✅ Email plugin shut down")

    shutdown_result = audit_plugin.shutdown()
    if shutdown_result.success:
        print("   ✅ Audit plugin shut down")

    print("✅ Plugin interfaces demonstration completed")


def demonstrate_event_interfaces() -> None:
    """Demonstrate event interfaces with publish-subscribe patterns."""
    print("\n" + "=" * 80)
    print("📡 EVENT INTERFACES - PUBLISH-SUBSCRIBE PATTERNS")
    print("=" * 80)

    # 1. Event system setup
    print("\n1. Event system setup:")

    publisher = SimpleEventPublisher()
    subscriber = SimpleEventSubscriber(publisher)

    # Create audit plugin for event handling
    audit_plugin = AuditLogPlugin()
    context = SimplePluginContext()
    audit_plugin.initialize(context)

    # Create event handler
    event_handler = UserEventHandler(audit_plugin)

    print("   Event publisher and subscriber created")

    # 2. Event subscription
    print("\n2. Event subscription:")

    # Subscribe to events
    result = subscriber.subscribe(UserCreatedEvent, event_handler)
    if result.success:
        print("   ✅ Subscribed to UserCreatedEvent")

    result = subscriber.subscribe(OrderPlacedEvent, event_handler)
    if result.success:
        print("   ✅ Subscribed to OrderPlacedEvent")

    # 3. Event publishing
    print("\n3. Event publishing:")

    # Publish user created event
    user_event = UserCreatedEvent(
        user_id="event_user_1",
        name="Event User",
        email="eventuser@example.com",
        timestamp=time.time(),
    )

    result = publisher.publish(user_event)
    if result.success:
        print("   ✅ UserCreatedEvent published successfully")
    else:
        print(f"   ❌ Event publish failed: {result.error}")

    # Publish order placed event
    order_event = OrderPlacedEvent(
        order_id="event_order_1",
        user_id="event_user_1",
        total=129.99,
        timestamp=time.time(),
    )

    result = publisher.publish(order_event)
    if result.success:
        print("   ✅ OrderPlacedEvent published successfully")
    else:
        print(f"   ❌ Event publish failed: {result.error}")

    # Publish event with no subscribers
    @dataclass
    class UnknownEvent:
        data: str

    unknown_event = UnknownEvent("test data")
    result = publisher.publish(unknown_event)
    if result.success:
        print("   ✅ Unknown event published (no subscribers)")

    # 4. Event unsubscription
    print("\n4. Event unsubscription:")

    result = subscriber.unsubscribe(UserCreatedEvent, event_handler)
    if result.success:
        print("   ✅ Unsubscribed from UserCreatedEvent")

    # Publish event after unsubscription
    user_event2 = UserCreatedEvent(
        user_id="event_user_2",
        name="Second Event User",
        email="eventuser2@example.com",
        timestamp=time.time(),
    )

    result = publisher.publish(user_event2)
    if result.success:
        print("   ✅ Event published after unsubscription (should have no effect)")

    print("✅ Event interfaces demonstration completed")


def main() -> None:
    """Execute all FlextInterfaces demonstrations."""
    print("🚀 FLEXT INTERFACES - ARCHITECTURE PATTERNS EXAMPLE")
    print("Demonstrating comprehensive interface patterns for enterprise architecture")

    try:
        demonstrate_validation_interfaces()
        demonstrate_service_interfaces()
        demonstrate_handler_interfaces()
        demonstrate_repository_interfaces()
        demonstrate_plugin_interfaces()
        demonstrate_event_interfaces()

        print("\n" + "=" * 80)
        print("✅ ALL FLEXT INTERFACES DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n📊 Summary of patterns demonstrated:")
        print("   ✅ Validation interfaces with protocols and rules")
        print("   🔧 Service interfaces with lifecycle management and configuration")
        print("   ⚙️ Handler interfaces with middleware pipeline patterns")
        print("   💾 Repository interfaces with Unit of Work pattern")
        print("   🔌 Plugin interfaces with extensibility and context management")
        print("   📡 Event interfaces with publish-subscribe patterns")
        print("\n💡 FlextInterfaces provides enterprise-grade architecture patterns")
        print(
            "   with Clean Architecture, DDD, and extensibility through protocols "
            "and ABCs!",
        )

    except (ValueError, TypeError, ImportError, AttributeError) as e:
        print(f"\n❌ Error during FlextInterfaces demonstration: {e}")

        traceback.print_exc()


if __name__ == "__main__":
    main()
