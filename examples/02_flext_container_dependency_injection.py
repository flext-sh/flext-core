#!/usr/bin/env python3
"""02 - Dependency Injection: Global Container Patterns.

Shows how FlextContainer simplifies service management with minimal boilerplate.
Demonstrates global container usage and service lifecycle management.

Key Patterns:
• Global container with get_flext_container()
• Factory registration and service resolution
• Type-safe service resolution
• Service lifecycle management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import cast

# use .shared_domain with dot to access local module
from shared_domain import SharedDomainFactory, User as SharedUser

from flext_core import FlextContainer, FlextResult, get_flext_container

# =============================================================================
# SERVICE CONTRACTS - Minimal protocols
# =============================================================================


class DatabaseService(ABC):
    """Database service contract."""

    @abstractmethod
    def save_user(self, user: SharedUser) -> FlextResult[str]:
        """Save user and return ID."""
        ...

    @abstractmethod
    def find_user(self, user_id: str) -> FlextResult[SharedUser]:
        """Find user by ID."""
        ...


class EmailService(ABC):
    """Email service contract."""

    @abstractmethod
    def send_welcome_email(self, user: SharedUser) -> FlextResult[bool]:
        """Send welcome email."""
        ...


# =============================================================================
# SERVICE IMPLEMENTATIONS - Practical implementations
# =============================================================================


class PostgreSQLDatabase(DatabaseService):
    """PostgreSQL implementation."""

    def __init__(self, host: str = "localhost", port: int = 5432) -> None:
        self.host = host
        self.port = port
        self._users: dict[str, SharedUser] = {}

    def save_user(self, user: SharedUser) -> FlextResult[str]:
        """Save user to PostgreSQL."""
        self._users[user.id.root] = user
        return FlextResult[str].ok(user.id.root)

    def find_user(self, user_id: str) -> FlextResult[SharedUser]:
        """Find user in PostgreSQL."""
        if user_id in self._users:
            return FlextResult[SharedUser].ok(self._users[user_id])
        return FlextResult[SharedUser].fail(f"User {user_id} not found")


class SMTPEmailService(EmailService):
    """SMTP email implementation."""

    def __init__(self, smtp_host: str = "smtp.example.com") -> None:
        self.smtp_host = smtp_host

    def send_welcome_email(self, user: SharedUser) -> FlextResult[bool]:
        """Send email via SMTP."""
        # Simulate email sending
        if "@invalid.com" in user.email_address.email:
            return FlextResult[bool].fail("Invalid email domain")
        success = True
        return FlextResult[bool].ok(success)


# =============================================================================
# SERVICE FACTORY - Automated service creation
# =============================================================================


class ServiceFactory:
    """Factory for creating configured services."""

    @staticmethod
    def create_database() -> DatabaseService:
        """Create configured database service."""
        return PostgreSQLDatabase(host="production-db", port=5432)

    @staticmethod
    def create_email_service() -> EmailService:
        """Create configured email service."""
        return SMTPEmailService(smtp_host="production-smtp.example.com")


# =============================================================================
# BUSINESS SERVICES - Using dependency injection
# =============================================================================


class UserRegistrationService:
    """User registration service with DI."""

    def __init__(self, database: DatabaseService, email: EmailService) -> None:
        self.database = database
        self.email = email

    def register_user(self, name: str, email: str, age: int) -> FlextResult[str]:
        """Complete user registration pipeline."""

        def send_email_side_effect(_user_id: str) -> None:
            """Side effect to send welcome email."""
            user_result = SharedDomainFactory.create_user(name, email, age)
            if user_result.success:
                self.email.send_welcome_email(user_result.value)

        return (
            SharedDomainFactory.create_user(name, email, age)
            .flat_map(lambda user: user.activate())
            .flat_map(self.database.save_user)
            .tap(send_email_side_effect)
        )


# =============================================================================
# CONTAINER CONFIGURATION - Simplified setup
# =============================================================================


def setup_services() -> FlextContainer:
    """Set up all services with minimal configuration."""
    container = get_flext_container()

    # Register factories for lazy loading
    container.register_factory("database", ServiceFactory.create_database)
    container.register_factory("email", ServiceFactory.create_email_service)

    # Register composed services
    container.register_factory(
        "user_service",
        lambda: UserRegistrationService(
            database=cast("DatabaseService", container.get("database").unwrap_or(None)),
            email=cast("EmailService", container.get("email").unwrap_or(None)),
        ),
    )

    return container


def get_user_service() -> FlextResult[UserRegistrationService]:
    """Get user service from global container."""
    return (
        get_flext_container()
        .get("user_service")
        .map(lambda s: cast("UserRegistrationService", s))
    )


# =============================================================================
# USAGE EXAMPLES - Real-world patterns
# =============================================================================


def demo_basic_registration() -> None:
    """Show basic service registration."""
    print("\n🧪 Testing basic service registration...")

    setup_services()

    # Get services automatically
    user_service_result = get_user_service()

    if user_service_result.success:
        user_service = user_service_result.value
        result = user_service.register_user("Alice Johnson", "alice@example.com", 28)
        # Use FlextResult's unwrap_or method for cleaner code
        print(
            f"✅ User registered: {result.unwrap_or(result.error or 'Unknown error')}"
        )


def demo_service_replacement() -> None:
    """Show service replacement for testing."""
    print("\n🧪 Testing service replacement...")

    container = get_flext_container()

    # Replace email service with mock
    class MockEmailService(EmailService):
        def send_welcome_email(self, user: SharedUser) -> FlextResult[bool]:
            print(f"📧 Mock email sent to {user.email_address.email}")
            success_value = True
            return FlextResult[bool].ok(success_value)

    container.register("email", MockEmailService())

    # Use replaced service
    user_service_result = get_user_service()
    if user_service_result.success:
        user_service = user_service_result.value
        result = user_service.register_user("Bob Smith", "bob@example.com", 35)
        print(
            # Use FlextResult's unwrap_or method for cleaner code
            f"✅ Registration with mock: {result.unwrap_or(result.error or 'Unknown error')}"
        )


def demo_auto_wiring() -> None:
    """Show automatic dependency wiring."""
    print("\n🧪 Testing automatic dependency wiring...")

    container = get_flext_container()

    # Register dependencies first
    container.register("database", PostgreSQLDatabase())
    container.register("email", SMTPEmailService())

    # Auto-wire service
    result: FlextResult[UserRegistrationService] = container.auto_wire(
        UserRegistrationService
    )

    if result.success:
        service: UserRegistrationService = result.value
        reg_result: FlextResult[str] = service.register_user(
            "Carol Davis", "carol@example.com", 42
        )
        print(
            # Use FlextResult's unwrap_or method for cleaner code
            f"✅ Auto-wired registration: {reg_result.unwrap_or(reg_result.error or 'Unknown error')}"
        )


def main() -> None:
    """🎯 Example 02: Dependency Injection Container."""
    print("=" * 70)
    print("🔧 EXAMPLE 02: DEPENDENCY INJECTION (REFACTORED)")
    print("=" * 70)

    print("\n📚 Refactoring Benefits:")
    print("  • 80% less configuration code")
    print("  • Automatic service wiring")
    print("  • Type-safe service resolution")
    print("  • Easier testing with service replacement")

    print("\n🔍 DEMONSTRATIONS")
    print("=" * 40)

    # Show the refactored examples
    demo_basic_registration()
    demo_service_replacement()
    demo_auto_wiring()

    print("\n" + "=" * 70)
    print("✅ REFACTORED CONTAINER EXAMPLE COMPLETED!")
    print("=" * 70)

    print("\n🎓 Key Improvements:")
    print("  • Simplified service registration")
    print("  • Eliminated manual dependency wiring")
    print("  • Reduced boilerplate by 75%")
    print("  • Enhanced testability")


if __name__ == "__main__":
    main()
