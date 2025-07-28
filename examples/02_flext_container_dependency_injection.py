#!/usr/bin/env python3
"""FLEXT Container - Enterprise Dependency Injection Example.

Demonstrates dependency injection using FlextContainer with
type safety, service lifecycle management, and real-world architectural patterns.

Features demonstrated:
- Type-safe service registration and retrieval
- Factory pattern for complex service creation
- Service lifecycle management
- Dependency resolution chains
- Configuration-driven container setup
- Service health monitoring
- Performance tracking integration
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

from flext_core import FlextResult
from flext_core.container import FlextContainer, ServiceKey, get_flext_container
from flext_core.utilities import FlextUtilities

# =============================================================================
# DOMAIN INTERFACES - Abstract service contracts
# =============================================================================


class DatabaseConnection(ABC):
    """Abstract database connection interface."""

    @abstractmethod
    def connect(self) -> FlextResult[bool]:
        """Establish database connection."""

    @abstractmethod
    def execute_query(self, query: str) -> FlextResult[list[dict[str, Any]]]:
        """Execute database query."""

    @abstractmethod
    def close(self) -> FlextResult[bool]:
        """Close database connection."""


class EmailService(ABC):
    """Abstract email service interface."""

    @abstractmethod
    def send_email(self, to: str, subject: str, body: str) -> FlextResult[str]:
        """Send email and return message ID."""


class UserRepository(ABC):
    """Abstract user repository interface."""

    @abstractmethod
    def create_user(self, user_data: dict[str, Any]) -> FlextResult[str]:
        """Create user and return user ID."""

    @abstractmethod
    def get_user(self, user_id: str) -> FlextResult[dict[str, Any]]:
        """Get user data by ID."""


class NotificationService(ABC):
    """Abstract notification service interface."""

    @abstractmethod
    def notify_user_created(self, user_id: str, email: str) -> FlextResult[bool]:
        """Send user creation notification."""


# =============================================================================
# CONCRETE IMPLEMENTATIONS - Real service implementations
# =============================================================================


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL database connection implementation."""

    def __init__(self, host: str, port: int, database: str) -> None:
        self.host = host
        self.port = port
        self.database = database
        self.connected = False
        self.connection_id = FlextUtilities.generate_entity_id()

        print(f"🔧 PostgreSQL connection created: {self.connection_id}")

    def connect(self) -> FlextResult[bool]:
        """Simulate database connection."""
        print(f"🔌 Connecting to PostgreSQL at {self.host}:{self.port}/{self.database}")

        # Simulate connection with potential failure
        if random.random() < 0.1:  # 10% failure rate
            return FlextResult.fail(f"Connection failed to {self.host}")

        self.connected = True
        print(f"✅ PostgreSQL connected: {self.connection_id}")
        return FlextResult.ok(True)

    def execute_query(self, query: str) -> FlextResult[list[dict[str, Any]]]:
        """Execute database query."""
        if not self.connected:
            return FlextResult.fail("Database not connected")

        print(f"📊 Executing query: {query[:50]}...")

        # Simulate query execution
        if "users" in query.lower():
            return FlextResult.ok(
                [
                    {"id": "1", "name": "John Doe", "email": "john@example.com"},
                    {"id": "2", "name": "Jane Smith", "email": "jane@example.com"},
                ],
            )

        return FlextResult.ok([])

    def close(self) -> FlextResult[bool]:
        """Close database connection."""
        if self.connected:
            self.connected = False
            print(f"🔌 PostgreSQL connection closed: {self.connection_id}")
        return FlextResult.ok(True)


class SMTPEmailService(EmailService):
    """SMTP email service implementation."""

    def __init__(self, smtp_host: str, smtp_port: int) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.service_id = FlextUtilities.generate_entity_id()

        print(f"📧 SMTP email service created: {self.service_id}")

    def send_email(self, to: str, subject: str, body: str) -> FlextResult[str]:
        """Send email via SMTP."""
        print(f"📧 Sending email to {to}: {subject}")

        # Simulate email sending
        if "@invalid.com" in to:
            return FlextResult.fail(f"Invalid email domain: {to}")

        message_id = FlextUtilities.generate_uuid()
        print(f"✅ Email sent successfully: {message_id}")
        return FlextResult.ok(message_id)


class DatabaseUserRepository(UserRepository):
    """Database-backed user repository."""

    def __init__(self, db_connection: DatabaseConnection) -> None:
        self.db_connection = db_connection
        self.repository_id = FlextUtilities.generate_entity_id()

        print(f"👤 User repository created: {self.repository_id}")

    def create_user(self, user_data: dict[str, Any]) -> FlextResult[str]:
        """Create user in database."""
        print(f"👤 Creating user: {user_data.get('name', 'Unknown')}")

        # Validate user data
        if not user_data.get("name") or not user_data.get("email"):
            return FlextResult.fail("Missing required user data")

        # Simulate database insert
        query = (
            "INSERT INTO users (name, email) "
            f"VALUES ('{user_data['name']}', '{user_data['email']}')"
        )
        query_result = self.db_connection.execute_query(query)

        if query_result.is_failure:
            return FlextResult.fail(f"Database insert failed: {query_result.error}")

        user_id = FlextUtilities.generate_entity_id()
        print(f"✅ User created with ID: {user_id}")
        return FlextResult.ok(user_id)

    def get_user(self, user_id: str) -> FlextResult[dict[str, Any]]:
        """Get user from database."""
        print(f"👤 Retrieving user: {user_id}")

        query = f"SELECT * FROM users WHERE id = '{user_id}'"
        result = self.db_connection.execute_query(query)

        if result.is_failure:
            return FlextResult.fail(f"Database query failed: {result.error}")

        users = result.data
        if not users:
            return FlextResult.fail(f"User not found: {user_id}")

        user_data = users[0]
        print(f"✅ User retrieved: {user_data}")
        return FlextResult.ok(user_data)


class EmailNotificationService(NotificationService):
    """Email-based notification service."""

    def __init__(
        self,
        email_service: EmailService,
        user_repository: UserRepository,
    ) -> None:
        self.email_service = email_service
        self.user_repository = user_repository
        self.service_id = FlextUtilities.generate_entity_id()

        print(f"🔔 Notification service created: {self.service_id}")

    def notify_user_created(self, user_id: str, email: str) -> FlextResult[bool]:
        """Send user creation notification."""
        print(f"🔔 Sending user creation notification: {user_id}")

        # Get user details
        user_result = self.user_repository.get_user(user_id)
        if user_result.is_failure:
            return FlextResult.fail(f"Failed to get user data: {user_result.error}")

        user_data = user_result.data

        # Send welcome email
        subject = f"Welcome to FLEXT, {user_data['name']}!"
        body = f"""
        Hello {user_data["name"]},

        Welcome to the FLEXT platform! Your account has been successfully created.

        User ID: {user_id}
        Email: {email}
        Created: {FlextUtilities.generate_iso_timestamp()}

        Best regards,
        FLEXT Team
        """

        email_result = self.email_service.send_email(email, subject, body)
        if email_result.is_failure:
            return FlextResult.fail(f"Failed to send email: {email_result.error}")

        print(f"✅ User creation notification sent: {email_result.data}")
        return FlextResult.ok(True)


# =============================================================================
# SERVICE FACTORIES - Complex service creation patterns
# =============================================================================


class DatabaseConnectionFactory:
    """Factory for creating database connections."""

    @staticmethod
    def create_postgresql_connection(
        config: dict[str, Any],
    ) -> FlextResult[DatabaseConnection]:
        """Create PostgreSQL connection from configuration."""
        print("🏭 Creating PostgreSQL connection from config")

        # Validate configuration
        required_keys = ["host", "port", "database"]
        for key in required_keys:
            if key not in config:
                return FlextResult.fail(f"Missing configuration key: {key}")

        try:
            connection = PostgreSQLConnection(
                host=config["host"],
                port=int(config["port"]),
                database=config["database"],
            )

            # Test connection
            connect_result = connection.connect()
            if connect_result.is_failure:
                return FlextResult.fail(
                    f"Connection test failed: {connect_result.error}",
                )

            print("✅ PostgreSQL connection factory successful")
            return FlextResult.ok(connection)

        except Exception as e:
            return FlextResult.fail(f"Factory creation failed: {e}")


class EmailServiceFactory:
    """Factory for creating email services."""

    @staticmethod
    def create_smtp_service(config: dict[str, Any]) -> FlextResult[EmailService]:
        """Create SMTP email service from configuration."""
        print("🏭 Creating SMTP email service from config")

        # Set defaults and validate
        smtp_host = config.get("smtp_host", "localhost")
        smtp_port = config.get("smtp_port", 587)

        try:
            service = SMTPEmailService(smtp_host, smtp_port)
            print("✅ SMTP email service factory successful")
            return FlextResult.ok(service)

        except Exception as e:
            return FlextResult.fail(f"Email service factory failed: {e}")


# =============================================================================
# APPLICATION SERVICE - High-level business operations
# =============================================================================


class UserManagementService:
    """High-level user management service orchestrating multiple dependencies."""

    def __init__(
        self,
        user_repository: UserRepository,
        notification_service: NotificationService,
    ) -> None:
        self.user_repository = user_repository
        self.notification_service = notification_service
        self.service_id = FlextUtilities.generate_entity_id()

        print(f"👥 User management service created: {self.service_id}")

    def register_user(self, user_data: dict[str, Any]) -> FlextResult[dict[str, Any]]:
        """Complete user registration workflow."""
        print("👥 Starting user registration workflow")

        # Create user
        user_result = self.user_repository.create_user(user_data)
        if user_result.is_failure:
            return FlextResult.fail(f"User creation failed: {user_result.error}")

        user_id = user_result.data

        # Send notification
        notification_result = self.notification_service.notify_user_created(
            user_id,
            user_data["email"],
        )
        if notification_result.is_failure:
            print(
                f"⚠️ Notification failed but user created: {notification_result.error}",
            )

        # Return complete registration data
        registration_data = {
            "user_id": user_id,
            "user_data": user_data,
            "notification_sent": notification_result.is_success,
            "created_at": FlextUtilities.generate_iso_timestamp(),
            "service_id": self.service_id,
        }

        print(f"✅ User registration completed: {user_id}")
        return FlextResult.ok(registration_data)


# =============================================================================
# CONTAINER CONFIGURATION - Enterprise service setup
# =============================================================================


def setup_production_container() -> FlextResult[FlextContainer]:
    """Setups production container with all services."""
    print("\n🏗️ Setting up production container...")

    container = FlextContainer()

    # Database configuration
    db_config = {
        "host": "prod-db.example.com",
        "port": 5432,
        "database": "flext_prod",
    }

    # Email configuration
    email_config = {
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
    }

    # Register database connection factory
    def db_factory():
        return DatabaseConnectionFactory.create_postgresql_connection(
            db_config,
        ).unwrap()

    register_result = container.register_factory("database_connection", db_factory)
    if register_result.is_failure:
        return FlextResult.fail(
            f"Database factory registration failed: {register_result.error}",
        )

    # Register email service factory
    def email_factory():
        return EmailServiceFactory.create_smtp_service(
            email_config,
        ).unwrap()

    register_result = container.register_factory("email_service", email_factory)
    if register_result.is_failure:
        return FlextResult.fail(
            f"Email factory registration failed: {register_result.error}",
        )

    # Register user repository (depends on database)
    def user_repository_factory() -> UserRepository:
        db_result = container.get("database_connection")
        if db_result.is_failure:
            msg = f"Database dependency failed: {db_result.error}"
            raise RuntimeError(msg)
        return DatabaseUserRepository(db_result.unwrap())

    register_result = container.register_factory(
        "user_repository",
        user_repository_factory,
    )
    if register_result.is_failure:
        return FlextResult.fail(
            f"User repository registration failed: {register_result.error}",
        )

    # Register notification service (depends on email and user repository)
    def notification_service_factory() -> NotificationService:
        email_result = container.get("email_service")
        repo_result = container.get("user_repository")

        if email_result.is_failure:
            msg = f"Email service dependency failed: {email_result.error}"
            raise RuntimeError(msg)
        if repo_result.is_failure:
            msg = f"User repository dependency failed: {repo_result.error}"
            raise RuntimeError(
                msg,
            )

        return EmailNotificationService(
            email_result.unwrap(),
            repo_result.unwrap(),
        )

    register_result = container.register_factory(
        "notification_service",
        notification_service_factory,
    )
    if register_result.is_failure:
        return FlextResult.fail(
            f"Notification service registration failed: {register_result.error}",
        )

    # Register user management service (high-level orchestrator)
    def user_management_factory() -> UserManagementService:
        repo_result = container.get("user_repository")
        notification_result = container.get("notification_service")

        if repo_result.is_failure:
            msg = "User repository dependency failed: %s"
            raise RuntimeError(
                msg,
                repo_result.error,
            )
        if notification_result.is_failure:
            msg = "Notification service dependency failed: %s"
            raise RuntimeError(
                msg,
                notification_result.error,
            )

        return UserManagementService(
            repo_result.unwrap(),
            notification_result.unwrap(),
        )

    register_result = container.register_factory(
        "user_management_service",
        user_management_factory,
    )
    if register_result.is_failure:
        return FlextResult.fail(
            f"User management service registration failed: {register_result.error}",
        )

    print("✅ Production container setup completed")
    print(f"📊 Registered services: {container.get_service_count()}")

    return FlextResult.ok(container)


def setup_test_container() -> FlextResult[FlextContainer]:
    """Setups test container with mock services."""
    print("\n🧪 Setting up test container...")

    container = FlextContainer()

    # Mock implementations for testing
    class MockDatabase(DatabaseConnection):
        def connect(self) -> FlextResult[bool]:
            return FlextResult.ok(True)

        def execute_query(self, query: str) -> FlextResult[list[dict[str, Any]]]:
            return FlextResult.ok([{"id": "test", "name": "Test User"}])

        def close(self) -> FlextResult[bool]:
            return FlextResult.ok(True)

    class MockEmailService(EmailService):
        def send_email(self, to: str, subject: str, body: str) -> FlextResult[str]:
            return FlextResult.ok("mock-message-id")

    # Register mock services
    container.register("database_connection", MockDatabase())
    container.register("email_service", MockEmailService())

    # Register real services that depend on mocks
    db_result = container.get("database_connection")
    container.register("user_repository", DatabaseUserRepository(db_result.unwrap()))

    email_result = container.get("email_service")
    repo_result = container.get("user_repository")
    container.register(
        "notification_service",
        EmailNotificationService(
            email_result.unwrap(),
            repo_result.unwrap(),
        ),
    )

    repo_result = container.get("user_repository")
    notification_result = container.get("notification_service")
    container.register(
        "user_management_service",
        UserManagementService(
            repo_result.unwrap(),
            notification_result.unwrap(),
        ),
    )

    print("✅ Test container setup completed")
    print(f"📊 Registered services: {container.get_service_count()}")

    return FlextResult.ok(container)


# =============================================================================
# SERVICE HEALTH MONITORING
# =============================================================================


def check_container_health(container: FlextContainer) -> FlextResult[dict[str, Any]]:
    """Check health of all services in container."""
    print("\n🏥 Checking container health...")

    health_status = {
        "overall": "healthy",
        "services": {},
        "checked_at": FlextUtilities.generate_iso_timestamp(),
    }

    service_names = container.get_service_names()
    healthy_count = 0

    for service_name in service_names:
        print(f"🔍 Checking {service_name}...")

        service_result = container.get(service_name)
        if service_result.is_failure:
            health_status["services"][service_name] = {
                "status": "failed",
                "error": service_result.error,
            }
            continue

        service = service_result.unwrap()

        # Check if service has health check method
        if hasattr(service, "connect"):
            # For database connections
            health_result = service.connect()
            if health_result.is_success:
                health_status["services"][service_name] = {"status": "healthy"}
                healthy_count += 1
            else:
                health_status["services"][service_name] = {
                    "status": "unhealthy",
                    "error": health_result.error,
                }
        else:
            # For other services, assume healthy if instantiated
            health_status["services"][service_name] = {"status": "healthy"}
            healthy_count += 1

    # Determine overall health
    total_services = len(service_names)
    if healthy_count == total_services:
        health_status["overall"] = "healthy"
    elif healthy_count > total_services / 2:
        health_status["overall"] = "degraded"
    else:
        health_status["overall"] = "unhealthy"

    health_status["healthy_services"] = healthy_count
    health_status["total_services"] = total_services
    health_status["health_percentage"] = (healthy_count / total_services) * 100

    print(f"🏥 Health check completed: {health_status['overall']}")
    print(
        f"📊 {healthy_count}/{total_services} services healthy"
        f" ({health_status['health_percentage']:.1f}%)",
    )

    return FlextResult.ok(health_status)


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


def main() -> None:
    """Run comprehensive FlextContainer demonstration."""
    print("=" * 80)
    print("🏗️ FLEXT CONTAINER - DEPENDENCY INJECTION DEMONSTRATION")
    print("=" * 80)

    # Example 1: Production container setup
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 1: Production Container Setup")
    print("=" * 60)

    prod_container_result = setup_production_container()
    if prod_container_result.is_failure:
        print(f"❌ Production container setup failed: {prod_container_result.error}")
        return

    prod_container = prod_container_result.unwrap()

    # Example 2: Service retrieval and usage
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 2: Service Retrieval and Usage")
    print("=" * 60)

    # Get user management service
    service_result = prod_container.get("user_management_service")
    if service_result.is_failure:
        print(f"❌ Failed to get user management service: {service_result.error}")
        return

    user_service = service_result.unwrap()

    # Register a new user
    new_user = {
        "name": "Alice Johnson",
        "email": "alice@example.com",
    }

    registration_result = user_service.register_user(new_user)
    if registration_result.is_success:
        print("✅ User registration successful!")
        print(f"📄 Registration data: {registration_result.data}")
    else:
        print(f"❌ User registration failed: {registration_result.error}")

    # Example 3: Type-safe service keys
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 3: Type-safe Service Keys")
    print("=" * 60)

    # Create typed service keys
    user_repo_key = ServiceKey[UserRepository]("user_repository")
    email_service_key = ServiceKey[EmailService]("email_service")

    # Get services with type safety
    repo_result = prod_container.get(user_repo_key.name)
    email_result = prod_container.get(email_service_key.name)

    if repo_result.is_success and email_result.is_success:
        print("✅ Type-safe service retrieval successful")
        print(f"📊 User repository: {type(repo_result.unwrap()).__name__}")
        print(f"📧 Email service: {type(email_result.unwrap()).__name__}")

    # Example 4: Container health monitoring
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 4: Container Health Monitoring")
    print("=" * 60)

    health_result = check_container_health(prod_container)
    if health_result.is_success:
        health_data = health_result.data
        print(f"🏥 Overall health: {health_data['overall']}")
        print(f"📊 Health percentage: {health_data['health_percentage']:.1f}%")

    # Example 5: Test container for unit testing
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 5: Test Container for Unit Testing")
    print("=" * 60)

    test_container_result = setup_test_container()
    if test_container_result.is_success:
        test_container = test_container_result.unwrap()

        # Test with mock services
        test_service_result = test_container.get("user_management_service")
        if test_service_result.is_success:
            test_service = test_service_result.unwrap()

            test_user = {
                "name": "Test User",
                "email": "test@example.com",
            }

            test_result = test_service.register_user(test_user)
            if test_result.is_success:
                print("✅ Test container user registration successful!")
                print("🧪 Using mock services for testing")

    # Example 6: Global container usage
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 6: Global Container Usage")
    print("=" * 60)

    # Get global container and configure it
    global_container = get_flext_container()

    # Register a simple service in global container
    simple_service = {
        "name": "Global Service",
        "version": "1.0.0",
        "created_at": FlextUtilities.generate_iso_timestamp(),
    }

    register_result = global_container.register("global_service", simple_service)
    if register_result.is_success:
        print("✅ Global service registered")

        # Retrieve from global container
        retrieved_result = global_container.get("global_service")
        if retrieved_result.is_success:
            print(f"📦 Global service retrieved: {retrieved_result.data}")

    # Example 7: Service information and introspection
    print("\n" + "=" * 60)
    print("📋 EXAMPLE 7: Service Information and Introspection")
    print("=" * 60)

    print("📊 Production container services:")
    services = prod_container.list_services()
    for name, service_type in services.items():
        info_result = prod_container.get_info(name)
        if info_result.is_success:
            info = info_result.data
            print(f"  🔧 {name}: {info['class']} ({service_type})")

    print("\n" + "=" * 80)
    print("🎉 FLEXT CONTAINER DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
