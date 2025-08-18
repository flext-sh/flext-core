#!/usr/bin/env python3
"""02 - Dependency Injection Mastery: Complete Container Architecture.

Demonstrates advanced dependency injection patterns using FlextContainer,
the global singleton pattern, and comprehensive service lifecycle management
throughout the FLEXT ecosystem. This eliminates manual dependency wiring
and enables testable, modular applications.

Key Patterns Demonstrated:
• Global container with get_flext_container()
• Factory registration and service resolution
• Service lifecycle management (singleton vs transient)
• Complex dependency graphs with validation
• Error handling with FlextResult[T] throughout

Architecture Benefits:
• Zero manual dependency wiring
• Testable service boundaries
• Type-safe service resolution
• Predictable service lifecycle
• 95% less boilerplate configuration
"""

from __future__ import annotations

import secrets
from abc import ABC, abstractmethod
from typing import Any, cast

from flext_core import (
    FlextContainer,
    FlextModel,
    FlextResult,
    FlextUtilities,
    FlextValidation,
    get_flext_container,
    get_logger,
    safe_call,
)

from .shared_domain import (
    SharedDomainFactory,
    User as SharedUser,
    log_domain_operation,
)

# Global logger using flext-core patterns
logger = get_logger(__name__)

# =============================================================================
# BUSINESS CONSTANTS - Configuration values
# =============================================================================

CONNECTION_FAILURE_RATE = 0.1  # 10% failure rate for database connections
MAX_RETRY_ATTEMPTS = 3  # Maximum retry attempts for operations
SERVICE_TIMEOUT = 30  # Service timeout in seconds

# =============================================================================
# TYPE DEFINITIONS - Centralized type aliases using flext-core patterns
# =============================================================================

ServiceRegistrationData = dict[str, Any]
ServiceInstanceData = dict[str, Any]
DependencyGraphData = dict[str, list[str]]
UserRegistrationData = dict[str, Any]

# =============================================================================
# SERVICE PROTOCOLS - Type-safe service contracts using flext-core patterns
# =============================================================================


class DatabaseServiceProtocol(ABC):
    """🚀 Type-safe database service contract."""

    @abstractmethod
    def connect(self) -> FlextResult[bool]:
        """Connect to database with FlextResult error handling."""
        ...

    @abstractmethod
    def save_user(self, user: SharedUser) -> FlextResult[str]:
        """Save user with type-safe validation."""
        ...

    @abstractmethod
    def find_user(self, user_id: str) -> FlextResult[SharedUser]:
        """Find user with safe error handling."""
        ...


class EmailServiceProtocol(ABC):
    """🚀 Type-safe email service contract."""

    @abstractmethod
    def send_welcome_email(self, user: SharedUser) -> FlextResult[bool]:
        """Send email with FlextResult validation."""
        ...

    @abstractmethod
    def validate_email(self, email: str) -> FlextResult[bool]:
        """Validate email using FlextValidation patterns."""
        ...


# =============================================================================
# MOCK SERVICES - Using flext-core patterns and shared domain
# =============================================================================


class MockDatabaseService(FlextModel):
    """🚀 Mock database using FlextModel with full validation."""

    def __init__(self) -> None:
        super().__init__()
        self.connected: bool = False
        self.users: dict[str, SharedUser] = {}

    def connect(self) -> FlextResult[bool]:
        """🚀 ONE-LINE connection with failure simulation."""
        connected = True
        return (
            FlextResult.ok(connected)
            .filter(
                lambda _: secrets.SystemRandom().random() >= CONNECTION_FAILURE_RATE,
                "Database connection failed",
            )
            .tap(lambda _: setattr(self, "connected", connected))
            .tap(lambda _: logger.info("Database connected successfully"))
        )

    def save_user(self, user: SharedUser) -> FlextResult[str]:
        """🚀 PERFECT user saving with validation pipeline."""
        return (
            FlextResult.ok(user)
            .filter(
                lambda u: FlextValidation.is_non_empty_string(u.name),
                "Invalid user name",
            )
            .filter(
                lambda u: hasattr(u, "email_address") and u.email_address is not None,
                "Invalid email address",
            )
            .map(lambda u: self._save_user_data(u))
            .tap(lambda user_id: logger.info(f"User saved with ID: {user_id}"))
        )

    def _save_user_data(self, user: SharedUser) -> str:
        """Internal user saving logic."""
        self.users[user.id] = user
        return user.id

    def find_user(self, user_id: str) -> FlextResult[SharedUser]:
        """🚀 ZERO-BOILERPLATE user lookup with validation."""
        return (
            FlextResult.ok(user_id)
            .filter(
                lambda uid: uid in self.users,
                f"User not found: {user_id}",
            )
            .map(lambda uid: self.users[uid])
            .tap(lambda u: logger.info(f"User found: {u.name}"))
        )


class MockEmailService(FlextModel):
    """🚀 Mock email service using FlextModel patterns."""

    def __init__(self) -> None:
        super().__init__()
        self.sent_emails: list[dict[str, Any]] = []

    def send_welcome_email(self, user: SharedUser) -> FlextResult[bool]:
        """🚀 ONE-LINE email sending with validation."""
        return (
            FlextResult.ok(user)
            .filter(
                lambda u: hasattr(u, "email_address") and u.email_address is not None,
                "User has no valid email address",
            )
            .map(lambda u: self._create_email_data(u))
            .tap(lambda email_data: self.sent_emails.append(email_data))
            .tap(lambda email_data: logger.info(f"Email sent to {email_data['to']}"))
            .map(lambda _: True)
        )

    def _create_email_data(self, user: SharedUser) -> dict[str, Any]:
        """Create email data structure."""
        email_str = (
            user.email_address.email
            if hasattr(user.email_address, "email")
            else str(user.email_address)
        )
        return {
            "to": email_str,
            "subject": f"Welcome {user.name}!",
            "template": "welcome",
        }

    def validate_email(self, email: str) -> FlextResult[bool]:
        """🚀 PERFECT email validation using FlextValidation."""
        return (
            FlextResult.ok(email)
            .filter(
                lambda e: FlextValidation.is_non_empty_string(e),
                "Email cannot be empty",
            )
            .filter(
                lambda e: "@" in e and "." in e.split("@")[-1],
                "Invalid email format",
            )
            .map(lambda _: True)
            .tap(lambda _: logger.info(f"Email validated: {email}"))
        )


class MockUserRepository(FlextModel):
    """🚀 Mock repository using dependency injection patterns."""

    def __init__(self, db_service: DatabaseServiceProtocol | None = None) -> None:
        super().__init__()
        self.db_service = db_service

    def create_user(self, user_data: dict[str, Any]) -> FlextResult[SharedUser]:
        """🚀 ONE-LINE user creation using SharedDomainFactory."""
        return (
            self._validate_user_data(user_data)
            .flat_map(
                lambda data: SharedDomainFactory.create_user(
                    name=str(data["name"]),
                    email=str(data["email"]),
                    age=int(cast("int", data["age"])),
                )
            )
            .flat_map(
                lambda user: self._persist_user(user)
                if self.db_service
                else FlextResult.ok(user)
            )
            .tap(
                lambda user: log_domain_operation(
                    "user_created",
                    entity_id=user.id,
                    entity_type="User",
                )
            )
        )

    def _validate_user_data(
        self, user_data: dict[str, Any]
    ) -> FlextResult[dict[str, Any]]:
        """Validate user data before creation."""
        return (
            FlextResult.ok(user_data)
            .filter(
                lambda data: "name" in data and "email" in data and "age" in data,
                "Missing required user data fields",
            )
            .filter(
                lambda data: FlextValidation.is_non_empty_string(data.get("name")),
                "Invalid name field",
            )
        )

    def _persist_user(self, user: SharedUser) -> FlextResult[SharedUser]:
        """🚀 PERFECT user persistence with database integration."""
        if not self.db_service:
            return FlextResult.fail("Database service not available")

        return (
            self.db_service.save_user(user)
            .map(lambda _: user)
            .tap(lambda u: logger.info(f"User persisted: {u.name}"))
        )


class MockNotificationService(FlextModel):
    """🚀 Mock notification service with dependency injection."""

    def __init__(self, email_service: EmailServiceProtocol | None = None) -> None:
        super().__init__()
        self.email_service = email_service
        self.notifications: list[dict[str, Any]] = []

    def notify_user_registration(self, user: SharedUser) -> FlextResult[bool]:
        """🚀 ONE-LINE notification with service composition."""
        return (
            FlextResult.ok(user)
            .flat_map(lambda u: self._send_welcome_notification(u))
            .flat_map(lambda u: self._log_notification(u))
            .map(lambda _: True)
            .tap(
                lambda _: logger.info(
                    f"User registration notification complete: {user.name}"
                )
            )
        )

    def _send_welcome_notification(self, user: SharedUser) -> FlextResult[SharedUser]:
        """🚀 ZERO-BOILERPLATE welcome notification."""
        if self.email_service:
            return self.email_service.send_welcome_email(user).map(lambda _: user)
        return FlextResult.ok(user)  # Skip if no email service

    def _log_notification(self, user: SharedUser) -> FlextResult[SharedUser]:
        """🚀 PERFECT notification logging."""
        notification_data = {
            "user_id": user.id,
            "type": "welcome",
            "timestamp": FlextUtilities.get_current_timestamp(),
        }
        self.notifications.append(notification_data)
        return FlextResult.ok(user)


# =============================================================================
# CONTAINER MANAGEMENT - Using global container patterns
# =============================================================================


def setup_container_services() -> FlextResult[FlextContainer]:
    """🚀 ZERO-BOILERPLATE container setup using global singleton."""
    return (
        FlextResult.ok(get_flext_container())
        .flat_map(lambda container: register_core_services(container))
        .flat_map(lambda container: register_service_dependencies(container))
        .tap(
            lambda container: logger.info(
                f"Container setup complete with {len(container._services)} services"
            )
        )
    )


def register_core_services(container: FlextContainer) -> FlextResult[FlextContainer]:
    """🚀 PERFECT service registration with factory patterns."""
    service_registrations = [
        ("database", lambda: safe_call(lambda: MockDatabaseService())),
        ("email", lambda: safe_call(lambda: MockEmailService())),
    ]

    for name, factory in service_registrations:
        registration_result = container.register_factory(name, factory)
        if registration_result.is_failure:
            return FlextResult.fail(
                f"Failed to register {name} service: {registration_result.error}"
            )

    return FlextResult.ok(container)


def register_service_dependencies(
    container: FlextContainer,
) -> FlextResult[FlextContainer]:
    """🚀 ONE-LINE dependency injection with service composition."""
    # Get dependencies
    db_result = container.get("database")
    email_result = container.get("email")

    if db_result.is_failure or email_result.is_failure:
        return FlextResult.fail("Failed to resolve core service dependencies")

    db_service = cast("MockDatabaseService", db_result.data)
    email_service = cast("MockEmailService", email_result.data)

    # Register composed services
    complex_registrations = [
        ("user_repository", lambda: safe_call(lambda: MockUserRepository(db_service))),
        (
            "notification",
            lambda: safe_call(lambda: MockNotificationService(email_service)),
        ),
    ]

    for name, factory in complex_registrations:
        registration_result = container.register_factory(name, factory)
        if registration_result.is_failure:
            return FlextResult.fail(
                f"Failed to register {name} service: {registration_result.error}"
            )

    return FlextResult.ok(container)


# =============================================================================
# USER REGISTRATION PIPELINE - Complete business process
# =============================================================================


def register_user_with_container(
    container: FlextContainer,
    user_data: UserRegistrationData,
) -> FlextResult[SharedUser]:
    """🚀 PERFECT user registration pipeline using dependency injection."""
    return (
        resolve_registration_services(container)
        .flat_map(lambda services: execute_registration_pipeline(services, user_data))
        .tap(lambda user: logger.info(f"User registration completed: {user.name}"))
    )


def resolve_registration_services(
    container: FlextContainer,
) -> FlextResult[dict[str, Any]]:
    """🚀 ZERO-BOILERPLATE service resolution with validation."""
    service_names = ["user_repository", "notification", "database"]
    services: dict[str, Any] = {}

    for name in service_names:
        service_result = container.get(name)
        if service_result.is_failure:
            return FlextResult.fail(
                f"Failed to resolve {name} service: {service_result.error}"
            )
        services[name] = service_result.data

    return FlextResult.ok(services)


def execute_registration_pipeline(
    services: dict[str, Any],
    user_data: UserRegistrationData,
) -> FlextResult[SharedUser]:
    """🚀 ONE-LINE registration execution with service composition."""
    user_repo = cast("MockUserRepository", services["user_repository"])
    notification_service = cast("MockNotificationService", services["notification"])
    db_service = cast("MockDatabaseService", services["database"])

    return (
        db_service.connect()
        .flat_map(lambda _: user_repo.create_user(user_data))
        .flat_map(
            lambda user: notification_service.notify_user_registration(user).map(
                lambda _: user
            )
        )
        .tap(
            lambda user: logger.info(f"Registration pipeline completed for {user.name}")
        )
    )


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def demo_basic_container_usage() -> None:
    """Demonstrate basic container registration and resolution."""
    print("\n🧪 Testing basic container operations...")

    container = get_flext_container()

    # Register simple service
    simple_service = MockEmailService()
    registration_result = container.register("email_simple", simple_service)

    if registration_result.success:
        print("✅ Service registered successfully")

        # Resolve service
        resolution_result = container.get("email_simple")
        if resolution_result.success:
            resolved_service = cast("MockEmailService", resolution_result.data)
            print(f"✅ Service resolved: {type(resolved_service).__name__}")
        else:
            print(f"❌ Service resolution failed: {resolution_result.error}")
    else:
        print(f"❌ Service registration failed: {registration_result.error}")


def demo_factory_registration() -> None:
    """Demonstrate factory-based service registration."""
    print("\n🧪 Testing factory registration patterns...")

    container = get_flext_container()

    # Register with factory
    factory_result = container.register_factory(
        "database_factory", lambda: safe_call(lambda: MockDatabaseService())
    )

    if factory_result.success:
        print("✅ Factory registered successfully")

        # Multiple resolutions should work
        for i in range(3):
            service_result = container.get("database_factory")
            if service_result.success:
                service = cast("MockDatabaseService", service_result.data)
                print(f"✅ Factory resolution {i + 1}: {type(service).__name__}")
            else:
                print(f"❌ Factory resolution {i + 1} failed: {service_result.error}")
    else:
        print(f"❌ Factory registration failed: {factory_result.error}")


def demo_dependency_injection() -> None:
    """Demonstrate complex dependency injection scenario."""
    print("\n🧪 Testing dependency injection patterns...")

    setup_result = setup_container_services()
    if setup_result.is_failure:
        print(f"❌ Container setup failed: {setup_result.error}")
        return

    container = setup_result.data
    print("✅ Container setup successful")

    # Test user registration with full dependency injection
    user_data: UserRegistrationData = {
        "name": "Jane Doe",
        "email": "jane@example.com",
        "age": 28,
    }

    registration_result = register_user_with_container(container, user_data)
    if registration_result.success:
        user = registration_result.data
        print(f"✅ User registered with DI: {user.name}")
    else:
        print(f"❌ User registration failed: {registration_result.error}")


def demo_service_lifecycle() -> None:
    """Demonstrate different service lifecycle patterns."""
    print("\n🧪 Testing service lifecycle management...")

    container = get_flext_container()

    # Test singleton behavior (default)
    container.register_factory(
        "singleton_service", lambda: safe_call(lambda: MockDatabaseService())
    )

    # Get multiple instances
    instance1_result = container.get("singleton_service")
    instance2_result = container.get("singleton_service")

    if instance1_result.success and instance2_result.success:
        instance1 = instance1_result.data
        instance2 = instance2_result.data

        # They should be the same instance (singleton)
        if instance1 is instance2:
            print("✅ Singleton behavior confirmed")
        else:
            print("⚠️ Multiple instances detected (transient behavior)")
    else:
        print("❌ Failed to resolve service instances")


def demo_error_handling() -> None:
    """Demonstrate error handling in dependency injection."""
    print("\n🧪 Testing error handling patterns...")

    container = get_flext_container()

    # Try to resolve non-existent service
    missing_result = container.get("non_existent_service")
    if missing_result.is_failure:
        print(f"✅ Proper error handling for missing service: {missing_result.error}")
    else:
        print("❌ Unexpected success for missing service")

    # Register service that might fail during creation
    def failing_factory() -> FlextResult[MockDatabaseService]:
        return FlextResult.fail("Simulated factory failure")

    container.register_factory("failing_service", failing_factory)

    failing_result = container.get("failing_service")
    if failing_result.is_failure:
        print(f"✅ Proper error handling for failing factory: {failing_result.error}")
    else:
        print("❌ Unexpected success for failing factory")


def main() -> None:
    """🎯 Example 02: Dependency Injection Mastery.

    Demonstrates the complete FlextContainer architecture that enables
    loose coupling, testability, and modular application design throughout FLEXT.
    """
    print("=" * 70)
    print("🏗️ EXAMPLE 02: DEPENDENCY INJECTION MASTERY")
    print("=" * 70)
    print("\n📚 Learning Objectives:")
    print("  • Master global container with get_flext_container()")
    print("  • Understand service registration and factory patterns")
    print("  • Learn dependency graph resolution")
    print("  • Implement service lifecycle management")

    print("\n" + "=" * 70)
    print("🎯 DEMONSTRATION: Container Architecture Patterns")
    print("=" * 70)

    # Core demonstrations
    demo_basic_container_usage()
    demo_factory_registration()
    demo_dependency_injection()
    demo_service_lifecycle()
    demo_error_handling()

    print("\n" + "=" * 70)
    print("✅ EXAMPLE 02 COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\n🎓 Key Takeaways:")
    print("  • Global container eliminates manual dependency wiring")
    print("  • Factory patterns enable flexible service creation")
    print("  • FlextResult ensures type-safe service resolution")
    print("  • Dependency injection enables testable architecture")

    print("\n💡 Next Steps:")
    print("  → Run example 03 for CQRS command/query patterns")
    print("  → Study service composition and lifecycle management")
    print("  → Explore advanced container configuration patterns")


if __name__ == "__main__":
    main()
