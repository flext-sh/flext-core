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
- Maximum type safety using flext_core.types
"""

from __future__ import annotations

import secrets
from abc import ABC, abstractmethod
from typing import cast

# Import shared domain models to eliminate duplication
from shared_domain import (
    SharedDomainFactory,
    User as SharedUser,
    log_domain_operation,
)

# Import additional flext-core patterns for enhanced functionality
from flext_core import (
    FlextContainer,
    FlextResult,
    FlextTypes,
    FlextUtilities,
    TAnyObject,
    TConfigDict,
    TEntityId,
    TErrorMessage,
    TLogMessage,
    TUserData,
    get_flext_container,
)

# Constants
CONNECTION_FAILURE_RATE = 0.1  # 10% failure rate for database connections


# =============================================================================
# ENHANCED DOMAIN MODELS - Using shared domain with DI patterns
# =============================================================================


# =============================================================================
# NO LOCAL DOMAIN MODELS - Use ONLY shared_domain.py models
# =============================================================================

# All domain functionality comes from shared_domain.py
# This eliminates ALL code duplication and uses standard SharedUser


# =============================================================================
# COMPLEXITY REDUCTION HELPERS - SOLID SRP: Eliminate repetitive patterns
# =============================================================================


class ServiceRegistrationHelper:
    """Helper to reduce repetitive service registration patterns - SOLID SRP."""

    @staticmethod
    def register_with_validation(
        container: FlextContainer,
        service_name: str,
        service_result: FlextResult[object],
    ) -> FlextResult[None]:
        """DRY Helper: Register service with validation - eliminates repetitive patterns."""
        if service_result.is_failure:
            return FlextResult.fail(
                f"{service_name} setup failed: {service_result.error}"
            )

        register_result = container.register(service_name, service_result.data)
        if register_result.is_failure:
            return FlextResult.fail(
                f"{service_name} registration failed: {register_result.error}"
            )

        return FlextResult.ok(None)


class ContainerSetupOrchestrator:
    """Strategy pattern: Orchestrate container setup - reduces complexity."""

    def __init__(self, container: FlextContainer, configurer: object) -> None:
        """Initialize with container and configurer."""
        self.container = container
        self.configurer = configurer
        self.registration_helper = ServiceRegistrationHelper()

    def setup_core_services(self) -> FlextResult[None]:
        """Setup all core services using strategy pattern - single return point."""
        service_configs = [
            ("DatabaseConnection", self.configurer.create_database_connection),
            ("EmailService", self.configurer.create_email_service),
            ("UserRepository", self.configurer.create_user_repository),
            ("NotificationService", self.configurer.create_notification_service),
        ]

        # Register all services using strategy pattern
        for service_name, service_factory in service_configs:
            service_result = service_factory()
            registration_result = self.registration_helper.register_with_validation(
                self.container, service_name, service_result
            )
            if registration_result.is_failure:
                return registration_result

        return FlextResult.ok(None)

    def setup_factories(self) -> FlextResult[None]:
        """Setup service factories - SOLID SRP."""

        def user_management_factory() -> UserManagementService:
            result = self.configurer.create_user_management_service()
            if result.is_failure:
                raise RuntimeError(
                    f"User management service creation failed: {result.error}"
                )
            return result.data

        factory_result = self.container.register_factory(
            "UserManagementService", user_management_factory
        )
        if factory_result.is_failure:
            return FlextResult.fail(
                f"Factory registration failed: {factory_result.error}"
            )

        return FlextResult.ok(None)


class DemonstrationFlowHelper:
    """Helper to reduce repetitive demonstration patterns - SOLID SRP."""

    @staticmethod
    def print_section_header(example_num: int, title: str) -> None:
        """DRY Helper: Print standardized section headers."""
        print("\n" + "=" * 60)
        print(f"📋 EXAMPLE {example_num}: {title}")
        print("=" * 60)

    @staticmethod
    def validate_prerequisite(condition: bool, error_message: str) -> FlextResult[None]:
        """DRY Helper: Validate prerequisites consistently."""
        if not condition:
            return FlextResult.fail(error_message)
        return FlextResult.ok(None)


# =============================================================================
# DOMAIN INTERFACES - Abstract service contracts
# =============================================================================


class DatabaseConnection(ABC):
    """Abstract database connection interface using flext_core.types."""

    @abstractmethod
    def connect(self) -> FlextResult[bool]:
        """Establish database connection."""

    @abstractmethod
    def execute_query(self, query: str) -> FlextResult[list[TAnyObject]]:
        """Execute database query using TAnyObject for results."""

    @abstractmethod
    def close(self) -> FlextResult[None]:
        """Close database connection."""


class EmailService(ABC):
    """Abstract email service interface using flext_core.types."""

    @abstractmethod
    def send_email(self, to: str, subject: str, body: str) -> FlextResult[str]:
        """Send email and return message ID."""


class UserRepository(ABC):
    """Abstract user repository interface using shared domain models."""

    @abstractmethod
    def create_user(self, user_data: TUserData) -> FlextResult[TEntityId]:
        """Create user using shared domain factory and return ID."""

    @abstractmethod
    def get_user(self, user_id: TEntityId) -> FlextResult[SharedUser]:
        """Get user entity by ID using shared domain models."""


class NotificationService(ABC):
    """Abstract notification service interface using enhanced user entities."""

    @abstractmethod
    def notify_user_created(self, user: SharedUser) -> FlextResult[None]:
        """Send user creation notification using shared user entity."""


# =============================================================================
# CONCRETE IMPLEMENTATIONS - Real service implementations
# =============================================================================


class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL database connection implementation using flext_core.types."""

    def __init__(self, host: str, port: int, database: str) -> None:
        """Initialize PostgreSQL connection with host, port and database."""
        self.host = host
        self.port = port
        self.database = database
        self.connected = False
        self.connection_id: TEntityId = FlextUtilities.generate_entity_id()

        log_message: TLogMessage = (
            f"🔧 PostgreSQL connection created: {self.connection_id}"
        )
        print(log_message)

    def connect(self) -> FlextResult[bool]:
        """Establish database connection."""
        log_message: TLogMessage = (
            f"🔌 Connecting to PostgreSQL: {self.host}:{self.port}/{self.database}"
        )
        print(log_message)

        # Simulate connection with potential failure
        if secrets.token_urlsafe(1) < CONNECTION_FAILURE_RATE:
            error_message: TErrorMessage = (
                f"Connection failed to {self.host}:{self.port}"
            )
            return FlextResult.fail(error_message)

        self.connected = True
        print(f"✅ Connected to PostgreSQL: {self.connection_id}")
        return FlextResult.ok(self.connected)

    def execute_query(self, query: str) -> FlextResult[list[TAnyObject]]:
        """Execute database query using TAnyObject for results."""
        if not self.connected:
            error_message: TErrorMessage = "Database not connected"
            return FlextResult.fail(error_message)

        log_message: TLogMessage = f"🔍 Executing query: {query[:50]}..."
        print(log_message)

        # Simulate query execution
        mock_results: list[TAnyObject] = [
            {"id": "1", "name": "John Doe", "email": "john@example.com"},
            {"id": "2", "name": "Jane Smith", "email": "jane@example.com"},
        ]

        print(f"✅ Query executed successfully: {len(mock_results)} rows")
        return FlextResult.ok(mock_results)

    def close(self) -> FlextResult[None]:
        """Close database connection."""
        if not self.connected:
            return FlextResult.ok(None)

        log_message: TLogMessage = (
            f"🔌 Closing PostgreSQL connection: {self.connection_id}"
        )
        print(log_message)
        self.connected = False
        print(f"✅ Connection closed: {self.connection_id}")
        return FlextResult.ok(None)


class SMTPEmailService(EmailService):
    """SMTP email service implementation using flext_core.types."""

    def __init__(self, smtp_host: str, smtp_port: int) -> None:
        """Initialize SMTP service with host and port."""
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.service_id: TEntityId = FlextUtilities.generate_entity_id()

        log_message: TLogMessage = (
            f"📧 SMTP service created: {self.smtp_host}:{self.smtp_port}"
        )
        print(log_message)

    def send_email(self, to: str, subject: str, body: str) -> FlextResult[str]:  # noqa: ARG002
        """Send email and return message ID."""
        log_message: TLogMessage = f"📧 Sending email to {to}: {subject}"
        print(log_message)

        # Simulate email sending
        message_id: str = FlextUtilities.generate_entity_id()
        print(f"✅ Email sent successfully: {message_id}")
        return FlextResult.ok(message_id)


class SharedDomainUserRepository(UserRepository):
    """User repository implementation using shared domain models."""

    def __init__(self, db_connection: DatabaseConnection) -> None:
        """Initialize repository with database connection."""
        self.db_connection = db_connection
        self.repository_id: TEntityId = FlextUtilities.generate_entity_id()

        log_message: TLogMessage = (
            f"🗄️ Shared domain user repository created: {self.repository_id}"
        )
        print(log_message)

    def create_user(self, user_data: TUserData) -> FlextResult[TEntityId]:
        """Create user using SharedDomainFactory."""
        log_message: TLogMessage = (
            f"👤 Creating enhanced user via repository: "
            f"{user_data.get('name', 'Unknown')}"
        )
        print(log_message)

        # Use SharedDomainFactory for robust user creation
        user_result = SharedDomainFactory.create_user(
            name=str(user_data.get("name", "")),
            email=str(user_data.get("email", "")),
            age=int(user_data.get("age", 0)),
        )

        if user_result.is_failure:
            return FlextResult.fail(f"User creation failed: {user_result.error}")

        user = user_result.data

        # Log domain operation using shared user
        log_domain_operation(
            "user_persisted_via_repository",
            "SharedUser",
            user.id,
            repository_id=self.repository_id,
            name=user.name,
            email=user.email_address.email,
        )

        print(f"✅ Shared user entity persisted: {user.id}")
        return FlextResult.ok(user.id)

    def get_user(self, user_id: TEntityId) -> FlextResult[SharedUser]:
        """Get user entity by ID using shared domain models."""
        log_message: TLogMessage = f"🔍 Getting shared user entity: {user_id}"
        print(log_message)

        # Simulate user entity retrieval using shared domain
        mock_user_result = SharedDomainFactory.create_user(
            name="Retrieved User",
            email="retrieved@example.com",
            age=30,
            id=user_id,
        )

        if mock_user_result.is_failure:
            return FlextResult.fail(
                f"Failed to retrieve user: {mock_user_result.error}",
            )

        user = mock_user_result.data
        print(f"✅ Shared user entity retrieved: {user_id}")
        return FlextResult.ok(user)


class EmailNotificationService(NotificationService):
    """Email notification service using enhanced user entities."""

    def __init__(
        self,
        email_service: EmailService,
        user_repository: UserRepository,
    ) -> None:
        """Initialize notification service."""
        self.email_service = email_service
        self.user_repository = user_repository
        self.service_id: TEntityId = FlextUtilities.generate_entity_id()

        log_message: TLogMessage = (
            f"🔔 Enhanced notification service created: {self.service_id}"
        )
        print(log_message)

    def notify_user_created(self, user: SharedUser) -> FlextResult[None]:
        """Send user creation notification using shared user entity."""
        log_message: TLogMessage = f"🔔 Sending notification for shared user: {user.id}"
        print(log_message)

        # Send welcome email using shared user
        email_result = self.email_service.send_email(
            to=user.email_address.email,
            subject="Welcome to our platform!",
            body=(
                f"Welcome {user.name}! Your account has been created "
                f"with ID: {user.id}."
            ),
        )

        if email_result.is_failure:
            error_message: TErrorMessage = (
                f"Failed to send notification: {email_result.error}"
            )
            return FlextResult.fail(error_message)

        print(f"✅ Shared user creation notification sent: {user.id}")
        return FlextResult.ok(None)


# =============================================================================
# FACTORY PATTERNS - Service creation with configuration
# =============================================================================


class DatabaseConnectionFactory:
    """Factory for creating database connections using flext_core.types."""

    @staticmethod
    def create_postgresql_connection(
        config: TConfigDict,
    ) -> FlextResult[DatabaseConnection]:
        """Create PostgreSQL connection using TConfigDict."""
        log_message: TLogMessage = (
            f"🏭 Creating PostgreSQL connection with config: {config}"
        )
        print(log_message)

        # Validate configuration using type guards
        if not FlextTypes.TypeGuards.is_dict_like(config):
            error_message: TErrorMessage = "Configuration must be a dictionary"
            return FlextResult.fail(error_message)

        required_keys = ["host", "port", "database"]
        for key in required_keys:
            if key not in config:
                error_message: TErrorMessage = f"Missing required config key: {key}"
                return FlextResult.fail(error_message)

        try:
            connection = PostgreSQLConnection(
                host=str(config["host"]),
                port=int(config["port"]),
                database=str(config["database"]),
            )
            print(
                f"✅ PostgreSQL connection factory created: {connection.connection_id}",
            )
            return FlextResult.ok(connection)
        except (TypeError, ValueError) as e:
            error_message: TErrorMessage = f"Invalid configuration: {e}"
            return FlextResult.fail(error_message)


class EmailServiceFactory:
    """Factory for creating email services using flext_core.types."""

    @staticmethod
    def create_smtp_service(config: TConfigDict) -> FlextResult[EmailService]:
        """Create SMTP service using TConfigDict."""
        log_message: TLogMessage = f"🏭 Creating SMTP service with config: {config}"
        print(log_message)

        # Validate configuration
        if not FlextTypes.TypeGuards.is_dict_like(config):
            error_message: TErrorMessage = "Configuration must be a dictionary"
            return FlextResult.fail(error_message)

        required_keys = ["smtp_host", "smtp_port"]
        for key in required_keys:
            if key not in config:
                error_message: TErrorMessage = f"Missing required config key: {key}"
                return FlextResult.fail(error_message)

        try:
            service = SMTPEmailService(
                smtp_host=str(config["smtp_host"]),
                smtp_port=int(config["smtp_port"]),
            )
            print(f"✅ SMTP service factory created: {service.service_id}")
            return FlextResult.ok(service)
        except (TypeError, ValueError) as e:
            error_message: TErrorMessage = f"Invalid configuration: {e}"
            return FlextResult.fail(error_message)


# =============================================================================
# BUSINESS LOGIC - Service orchestration
# =============================================================================


class UserManagementService:
    """User management service using shared domain models."""

    def __init__(
        self,
        user_repository: UserRepository,
        notification_service: NotificationService,
    ) -> None:
        """Initialize user management service."""
        self.user_repository = user_repository
        self.notification_service = notification_service
        self.service_id: TEntityId = FlextUtilities.generate_entity_id()

        log_message: TLogMessage = (
            f"👥 Enhanced user management service created: {self.service_id}"
        )
        print(log_message)

    def register_user(
        self,
        user_data: TUserData,
    ) -> FlextResult[TAnyObject]:
        """Register user using shared domain models."""
        log_message: TLogMessage = (
            f"👤 Registering enhanced user: {user_data.get('name', 'Unknown')}"
        )
        print(log_message)

        # Create user via repository
        create_result = self.user_repository.create_user(user_data)
        if create_result.is_failure:
            return FlextResult.fail(create_result.error)

        user_id = create_result.data

        # Get the enhanced user entity
        user_result = self.user_repository.get_user(user_id)
        if user_result.is_failure:
            return FlextResult.fail(user_result.error)

        user = user_result.data

        # Send notification using shared user
        notification_result = self.notification_service.notify_user_created(user)
        if notification_result.is_failure:
            # Log warning but don't fail the registration
            print(f"⚠️  Notification failed: {notification_result.error}")

        # Return registration result with shared user data
        registration_result: TAnyObject = {
            "user_id": user.id,
            "status": "registered",
            "email": user.email_address.email,
            "name": user.name,
            "created_at": str(user.created_at) if user.created_at else None,
            "version": user.version,
        }

        print(f"✅ Shared user registered successfully: {user.id}")
        return FlextResult.ok(registration_result)


# =============================================================================
# CONTAINER SETUP - Production and test configurations
# =============================================================================


# Separate service configuration classes for production setup
class ProductionServiceConfigurer:
    """Handles production service configuration following SOLID principles."""

    def __init__(self, container: FlextContainer) -> None:
        self.container = container

    def create_database_connection(self) -> FlextResult[DatabaseConnection]:
        """Create production database connection."""
        config: TConfigDict = {
            "host": "prod-db.example.com",
            "port": 5432,
            "database": "production_db",
        }
        return DatabaseConnectionFactory.create_postgresql_connection(config)

    def create_email_service(self) -> FlextResult[EmailService]:
        """Create production email service."""
        config: TConfigDict = {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
        }
        return EmailServiceFactory.create_smtp_service(config)

    def create_user_repository(self) -> FlextResult[UserRepository]:
        """Create user repository with database dependency."""
        db_result = self.container.get("DatabaseConnection")
        if db_result.is_failure:
            return FlextResult.fail(f"Database connection required: {db_result.error}")
        return FlextResult.ok(SharedDomainUserRepository(db_result.data))

    def create_notification_service(self) -> FlextResult[NotificationService]:
        """Create notification service with dependencies."""
        email_result = self.container.get("EmailService")
        if email_result.is_failure:
            return FlextResult.fail(f"Email service required: {email_result.error}")

        user_repo_result = self.container.get("UserRepository")
        if user_repo_result.is_failure:
            return FlextResult.fail(
                f"User repository required: {user_repo_result.error}"
            )

        return FlextResult.ok(
            EmailNotificationService(email_result.data, user_repo_result.data)
        )

    def create_user_management_service(self) -> FlextResult[UserManagementService]:
        """Create user management service with dependencies."""
        user_repo_result = self.container.get("UserRepository")
        if user_repo_result.is_failure:
            return FlextResult.fail(
                f"User repository required: {user_repo_result.error}"
            )

        notification_result = self.container.get("NotificationService")
        if notification_result.is_failure:
            return FlextResult.fail(
                f"Notification service required: {notification_result.error}"
            )

        return FlextResult.ok(
            UserManagementService(user_repo_result.data, notification_result.data)
        )


def setup_production_container() -> FlextResult[FlextContainer]:
    """Setup production container using Result pattern chaining - single return point."""
    log_message: TLogMessage = "🏭 Setting up production container..."
    print(log_message)

    container = get_flext_container()
    configurer = ProductionServiceConfigurer(container)
    orchestrator = ContainerSetupOrchestrator(container, configurer)

    # Use Result pattern chaining to eliminate multiple returns
    setup_result = (
        orchestrator.setup_core_services()
        .flat_map(lambda _: orchestrator.setup_factories())
        .map(lambda _: container)
    )

    if setup_result.is_success:
        print("✅ Production container setup completed")

    return setup_result


# Separate mock service classes for better organization
class MockDatabase(DatabaseConnection):
    """Mock database implementation for testing."""

    def connect(self) -> FlextResult[bool]:
        print("🧪 Mock database connected")
        return FlextResult.ok(True)  # noqa: FBT003

    def execute_query(self, _query: str) -> FlextResult[list[TAnyObject]]:
        mock_results: list[TAnyObject] = [{"id": "test", "name": "Test User"}]
        return FlextResult.ok(mock_results)

    def close(self) -> FlextResult[None]:
        print("🧪 Mock database closed")
        return FlextResult.ok(None)


class MockEmailService(EmailService):
    """Mock email service implementation for testing."""

    def send_email(self, _to: str, _subject: str, _body: str) -> FlextResult[str]:
        message_id: str = "mock_message_123"
        print(f"🧪 Mock email sent: {message_id}")
        return FlextResult.ok(message_id)


class MockUserRepository(UserRepository):
    """Mock user repository implementation for testing."""

    def create_user(self, user_data: TUserData) -> FlextResult[TEntityId]:  # noqa: ARG002
        user_id: TEntityId = "test_user_123"
        print(f"🧪 Mock user created: {user_id}")
        return FlextResult.ok(user_id)

    def get_user(self, user_id: TEntityId) -> FlextResult[SharedUser]:
        # Create mock shared user for testing
        mock_user_result = SharedDomainFactory.create_user(
            name="Test User",
            email="test@example.com",
            age=30,
            id=user_id,
        )

        if mock_user_result.is_failure:
            return FlextResult.fail(
                f"Failed to create mock user: {mock_user_result.error}",
            )

        user = mock_user_result.data
        print(f"🧪 Mock shared user retrieved: {user_id}")
        return FlextResult.ok(user)


class MockNotificationService(NotificationService):
    """Mock notification service implementation for testing."""

    def notify_user_created(self, user: SharedUser) -> FlextResult[None]:
        print(f"🧪 Mock notification sent for shared user: {user.id}")
        return FlextResult.ok(None)


class TestServiceConfigurer:
    """Handles test service configuration following SOLID principles."""

    def __init__(self, container: FlextContainer) -> None:
        self.container = container

    def register_mock_services(self) -> FlextResult[None]:
        """Register all mock services in the container."""
        try:
            self.container.register("DatabaseConnection", MockDatabase())
            self.container.register("EmailService", MockEmailService())
            self.container.register("UserRepository", MockUserRepository())
            self.container.register("NotificationService", MockNotificationService())
            return FlextResult.ok(None)
        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult.fail(f"Mock service registration failed: {e}")

    def register_user_management_factory(self) -> FlextResult[None]:
        """Register user management service factory."""

        def mock_user_management_factory() -> UserManagementService:
            user_repository = cast(
                "UserRepository", self.container.get("UserRepository").data
            )
            notification_service = cast(
                "NotificationService", self.container.get("NotificationService").data
            )
            return UserManagementService(user_repository, notification_service)

        try:
            self.container.register_factory(
                "UserManagementService", mock_user_management_factory
            )
            return FlextResult.ok(None)
        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult.fail(f"User management factory registration failed: {e}")


def setup_test_container() -> FlextResult[FlextContainer]:
    """Setup test container using SOLID principles and service configurer."""
    log_message: TLogMessage = "🧪 Setting up test container..."
    print(log_message)

    container = get_flext_container()
    configurer = TestServiceConfigurer(container)

    # Register mock services
    mock_result = configurer.register_mock_services()
    if mock_result.is_failure:
        return FlextResult.fail(f"Mock services setup failed: {mock_result.error}")

    # Register user management factory
    factory_result = configurer.register_user_management_factory()
    if factory_result.is_failure:
        return FlextResult.fail(
            f"User management factory setup failed: {factory_result.error}"
        )

    print("✅ Test container setup completed")
    return FlextResult.ok(container)


# =============================================================================
# HEALTH MONITORING - Service health checks
# =============================================================================


def check_container_health(container: FlextContainer) -> FlextResult[TAnyObject]:
    """Check container health using TAnyObject for health data."""
    log_message: TLogMessage = "🏥 Checking container health..."
    print(log_message)

    health_data: TAnyObject = {
        "container_id": FlextUtilities.generate_entity_id(),
        "timestamp": FlextUtilities.generate_iso_timestamp(),
        "services": {},
        "overall_status": "healthy",
    }

    # Check database connection
    try:
        db_connection = cast("DatabaseConnection", container.get(DatabaseConnection))
        connect_result = db_connection.connect()
        health_data["services"]["database"] = {
            "status": "healthy" if connect_result.is_success else "unhealthy",
            "error": connect_result.error if connect_result.is_failure else None,
        }
        if connect_result.is_success:
            db_connection.close()
    except (RuntimeError, ValueError, TypeError) as e:
        health_data["services"]["database"] = {
            "status": "error",
            "error": str(e),
        }

    # Check email service
    try:
        cast("EmailService", container.get(EmailService))
        health_data["services"]["email"] = {"status": "healthy"}
    except (RuntimeError, ValueError, TypeError) as e:
        health_data["services"]["email"] = {
            "status": "error",
            "error": str(e),
        }

    # Check user repository
    try:
        cast("UserRepository", container.get(UserRepository))
        health_data["services"]["user_repository"] = {"status": "healthy"}
    except (RuntimeError, ValueError, TypeError) as e:
        health_data["services"]["user_repository"] = {
            "status": "error",
            "error": str(e),
        }

    # Determine overall status
    unhealthy_services = [
        service
        for service in health_data["services"].values()
        if service["status"] != "healthy"
    ]
    if unhealthy_services:
        health_data["overall_status"] = "unhealthy"

    print(f"✅ Health check completed: {health_data['overall_status']}")
    return FlextResult.ok(health_data)


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


# Extract demonstration methods to reduce main complexity
class ContainerDemonstrator:
    """Handles container demonstration examples following SOLID principles - reduced complexity."""

    def __init__(self) -> None:
        self.test_container: FlextContainer | None = None
        self.prod_container: FlextContainer | None = None
        self.demo_helper = DemonstrationFlowHelper()

    def run_test_container_demo(self) -> FlextResult[None]:
        """Run test container setup and demonstration - uses helper for consistency."""
        # Use helper for standardized headers
        self.demo_helper.print_section_header(1, "Test Container Setup")

        test_container_result = setup_test_container()
        if test_container_result.is_failure:
            return FlextResult.fail(
                f"Test container setup failed: {test_container_result.error}"
            )

        self.test_container = test_container_result.data
        return FlextResult.ok(None)

    def _register_user_with_container(
        self, container: FlextContainer, user_data: TUserData, context_name: str
    ) -> FlextResult[None]:
        """DRY Helper: Register user with any container - eliminates code duplication."""
        try:
            user_service_result = container.get("UserManagementService")
            if user_service_result.is_failure:
                return FlextResult.fail(
                    f"Failed to get {context_name} user service: {user_service_result.error}"
                )

            user_service = cast("UserManagementService", user_service_result.data)
            registration_result = user_service.register_user(user_data)

            if registration_result.is_success:
                print(
                    f"✅ {context_name.title()} registration successful: {registration_result.data}"
                )
                return FlextResult.ok(None)
            return FlextResult.fail(
                f"{context_name.title()} registration failed: {registration_result.error}"
            )
        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult.fail(f"{context_name.title()} registration error: {e}")

    def run_user_registration_test(self) -> FlextResult[None]:
        """Run user registration test with test container - uses helper for consistency."""
        # Use helper for prerequisite validation
        prerequisite_result = self.demo_helper.validate_prerequisite(
            self.test_container is not None, "Test container not initialized"
        )
        if prerequisite_result.is_failure:
            return prerequisite_result

        # Use helper for standardized headers
        self.demo_helper.print_section_header(
            2, "User Registration with Test Container"
        )

        test_user_data: TUserData = {
            "name": "Test User",
            "email": "test@example.com",
            "age": 25,
        }

        return self._register_user_with_container(
            self.test_container, test_user_data, "test"
        )

    def run_health_check_demo(self) -> FlextResult[None]:
        """Run container health check demonstration - uses helper for consistency."""
        # Use helper for prerequisite validation
        prerequisite_result = self.demo_helper.validate_prerequisite(
            self.test_container is not None, "Test container not initialized"
        )
        if prerequisite_result.is_failure:
            return prerequisite_result

        # Use helper for standardized headers
        self.demo_helper.print_section_header(3, "Container Health Check")

        health_result = check_container_health(self.test_container)
        if health_result.is_success:
            health_data = health_result.data
            print(f"🏥 Container health: {health_data['overall_status']}")
            for service_name, service_health in health_data["services"].items():
                print(f"   {service_name}: {service_health['status']}")
            return FlextResult.ok(None)
        return FlextResult.fail(f"Health check failed: {health_result.error}")

    def run_production_container_demo(self) -> FlextResult[None]:
        """Run production container setup and demonstration - uses helper for consistency."""
        # Use helper for standardized headers
        self.demo_helper.print_section_header(4, "Production Container Setup")

        prod_container_result = setup_production_container()
        if prod_container_result.is_failure:
            return FlextResult.fail(
                f"Production container setup failed: {prod_container_result.error}"
            )

        self.prod_container = prod_container_result.data
        print("✅ Production container setup successful")

        # Test production user registration
        return self._test_production_user_registration()

    def _test_production_user_registration(self) -> FlextResult[None]:
        """Test user registration with production container - uses helper for consistency."""
        # Use helper for prerequisite validation
        prerequisite_result = self.demo_helper.validate_prerequisite(
            self.prod_container is not None, "Production container not initialized"
        )
        if prerequisite_result.is_failure:
            return prerequisite_result

        prod_user_data: TUserData = {
            "name": "Production User",
            "email": "prod@example.com",
            "age": 30,
        }

        return self._register_user_with_container(
            self.prod_container, prod_user_data, "production"
        )


def main() -> None:
    """Run comprehensive FlextContainer demonstration with SOLID principles."""
    print("=" * 80)
    print("🚀 FLEXT CONTAINER - DEPENDENCY INJECTION DEMONSTRATION")
    print("=" * 80)

    demonstrator = ContainerDemonstrator()

    # Run all demonstration steps
    steps = [
        demonstrator.run_test_container_demo,
        demonstrator.run_user_registration_test,
        demonstrator.run_health_check_demo,
        demonstrator.run_production_container_demo,
    ]

    for step in steps:
        result = step()
        if result.is_failure:
            print(f"❌ Demonstration step failed: {result.error}")
            return

    print("\n" + "=" * 80)
    print("🎉 FLEXT CONTAINER DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
