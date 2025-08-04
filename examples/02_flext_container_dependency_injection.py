#!/usr/bin/env python3
"""FLEXT Container Dependency Injection - Foundation Example 02.

Enterprise-grade dependency injection demonstration using FlextContainer
for type-safe service management across complex application architectures.

Module Role in Architecture:
    Examples Layer → Foundation Examples → Dependency Injection Implementation

    This example demonstrates essential patterns that enable:
    - Type-safe service registration used across 32 ecosystem projects
    - Service lifecycle management for enterprise applications
    - Factory patterns for complex service creation with dependencies
    - Configuration-driven container setup for environment flexibility

Dependency Injection Features:
    ✅ Type-Safe Registration: Service registration with full type validation
    ✅ Factory Patterns: Dynamic service creation with dependency resolution
    ✅ Service Lifecycle: Proper initialization and cleanup patterns
    ✅ Dependency Chains: Complex dependency graph resolution
    ✅ Configuration Integration: Environment-aware service configuration
    ✅ Health Monitoring: Service health validation and monitoring

Enterprise Applications:
    - Database connection management with connection pooling
    - Cache service configuration with TTL management
    - Logger service setup with structured logging
    - Authentication service with LDAP integration
    - Monitoring service with metrics collection

Real-World Usage Context:
    This container pattern is fundamental to all FLEXT services, providing
    reliable dependency management across distributed microservices and
    library integrations without tight coupling.

Architecture Benefits:
    - Inversion of Control: Decoupled service dependencies
    - Testability: Easy mocking and testing with container isolation
    - Configuration Flexibility: Environment-specific service setup
    - Performance Optimization: Lazy loading and singleton patterns

See Also:
    - src/flext_core/container.py: FlextContainer implementation
    - src/flext_core/config.py: Configuration management integration
    - examples/03_flext_commands_cqrs_pattern.py: Next architectural example
    - tests/unit/core/test_container.py: Comprehensive container tests

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

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
# SERVICE REGISTRATION STRATEGY - SOLID SRP: Single Responsibility
# =============================================================================


class ServiceRegistrationResult:
    """Value object for service registration results."""

    def __init__(self, *, success: bool, message: str, service_name: str) -> None:
        self.success = success
        self.message = message
        self.service_name = service_name


class ServiceRegistrationStrategy:
    """Strategy pattern for different service registration approaches - SOLID SRP."""

    def __init__(self, container: FlextContainer) -> None:
        self._container = container

    def register_service_with_validation(
        self,
        service_name: str,
        service_result: FlextResult[object],
    ) -> FlextResult[ServiceRegistrationResult]:
        """Register service with comprehensive validation using strategy pattern."""
        if service_result.is_failure:
            result = ServiceRegistrationResult(
                success=False,
                message=f"{service_name} creation failed: {service_result.error}",
                service_name=service_name,
            )
            return FlextResult.fail(result.message)

        register_result = self._container.register(service_name, service_result.data)
        if register_result.is_failure:
            result = ServiceRegistrationResult(
                success=False,
                message=f"{service_name} registration failed: {register_result.error}",
                service_name=service_name,
            )
            return FlextResult.fail(result.message)

        result = ServiceRegistrationResult(
            success=True,
            message=f"{service_name} registered successfully",
            service_name=service_name,
        )
        return FlextResult.ok(result)


class ServiceConfiguration:
    """Value object containing service configuration data."""

    def __init__(self, name: str, factory_method: callable) -> None:
        self.name = name
        self.factory_method = factory_method


class ContainerSetupOrchestrator:
    """Strategy pattern: Orchestrate container setup with reduced complexity."""

    def __init__(self, container: FlextContainer, configurer: object) -> None:
        """Initialize with container and configurer."""
        self._container = container
        self._configurer = configurer
        self._registration_strategy = ServiceRegistrationStrategy(container)

    def _get_core_service_configurations(self) -> list[ServiceConfiguration]:
        """Get core service configurations - SOLID OCP: Open for extension."""
        return [
            ServiceConfiguration(
                "DatabaseConnection",
                getattr(self._configurer, "create_database_connection", lambda: None),
            ),
            ServiceConfiguration(
                "EmailService",
                getattr(self._configurer, "create_email_service", lambda: None),
            ),
            ServiceConfiguration(
                "UserRepository",
                getattr(self._configurer, "create_user_repository", lambda: None),
            ),
            ServiceConfiguration(
                "NotificationService",
                getattr(self._configurer, "create_notification_service", lambda: None),
            ),
        ]

    def setup_core_services(self) -> FlextResult[None]:
        """Setup all core services using strategy pattern - single return point."""
        service_configs = self._get_core_service_configurations()

        for config in service_configs:
            service_result = config.factory_method()
            registration_result = (
                self._registration_strategy.register_service_with_validation(
                    config.name, service_result
                )
            )
            if registration_result.is_failure:
                return FlextResult.fail(registration_result.error)

        return FlextResult.ok(None)

    def setup_factories(self) -> FlextResult[None]:
        """Setup service factories using factory method pattern - SOLID SRP."""
        factory_creator = UserManagementServiceFactoryCreator(self._configurer)
        user_management_factory = factory_creator.create_factory()

        factory_result = self._container.register_factory(
            "UserManagementService", user_management_factory
        )
        if factory_result.is_failure:
            return FlextResult.fail(
                f"Factory registration failed: {factory_result.error}"
            )

        return FlextResult.ok(None)


class UserManagementServiceFactoryCreator:
    """Factory creator for UserManagementService - SOLID SRP."""

    def __init__(self, configurer: object) -> None:
        self._configurer = configurer

    def create_factory(self) -> callable:
        """Create factory function for UserManagementService."""

        def user_management_factory() -> UserManagementService:
            create_method = getattr(
                self._configurer, "create_user_management_service", None
            )
            if create_method is None:
                msg = "User management service factory not available"
                raise RuntimeError(msg)
            result = create_method()
            if result.is_failure:
                error_msg: str = (
                    f"User management service creation failed: {result.error}"
                )
                raise RuntimeError(error_msg)
            return result.data

        return user_management_factory


class DemonstrationSection:
    """Value object for demonstration section information."""

    def __init__(self, number: int, title: str) -> None:
        self.number = number
        self.title = title
        self.header_text = f"📋 EXAMPLE {number}: {title}"


class PrerequisiteValidation:
    """Value object for prerequisite validation result."""

    def __init__(self, *, is_valid: bool, error_message: str = "") -> None:
        self.is_valid = is_valid
        self.error_message = error_message


class DemonstrationFormatter:
    """Formatter for demonstration output using strategy pattern - SOLID SRP."""

    @staticmethod
    def format_section_header(section: DemonstrationSection) -> str:
        """Format standardized section headers."""
        separator = "=" * 60
        return f"\n{separator}\n{section.header_text}\n{separator}"

    @staticmethod
    def print_section_header(section: DemonstrationSection) -> None:
        """Print standardized section headers."""
        print(DemonstrationFormatter.format_section_header(section))


class PrerequisiteValidator:
    """Validator for demonstration prerequisites - SOLID SRP."""

    @staticmethod
    def validate_condition(
        *, condition: bool, error_message: str
    ) -> FlextResult[PrerequisiteValidation]:
        """Validate prerequisites with detailed result."""
        validation = PrerequisiteValidation(
            is_valid=condition, error_message=error_message if not condition else ""
        )

        if not condition:
            return FlextResult.fail(error_message)
        return FlextResult.ok(validation)


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
        assert user is not None

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
        assert user is not None

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


# =============================================================================
# SERVICE CONFIGURATION BASE CLASSES - SOLID Template Method Pattern
# =============================================================================


class ServiceConfigurationData:
    """Value object for service configuration data."""

    def __init__(self, **config_data: object) -> None:
        self.data = config_data

    def get(self, key: str, default: object = None) -> object:
        """Get configuration value with default."""
        return self.data.get(key, default)


class BaseServiceConfigurer:
    """Base class for service configurers using Template Method pattern - SOLID LSP."""

    def __init__(self, container: FlextContainer) -> None:
        self._container = container

    def _get_database_config(self) -> ServiceConfigurationData:
        """Template method: Get database configuration - override in subclasses."""
        msg = "Subclasses must implement _get_database_config"
        raise NotImplementedError(msg)

    def _get_email_config(self) -> ServiceConfigurationData:
        """Template method: Get email configuration - override in subclasses."""
        msg = "Subclasses must implement _get_email_config"
        raise NotImplementedError(msg)

    def create_database_connection(self) -> FlextResult[DatabaseConnection]:
        """Create database connection using template method pattern."""
        config = self._get_database_config()
        config_dict = cast("TConfigDict", config.data)
        return DatabaseConnectionFactory.create_postgresql_connection(config_dict)

    def create_email_service(self) -> FlextResult[EmailService]:
        """Create email service using template method pattern."""
        config = self._get_email_config()
        config_dict = cast("TConfigDict", config.data)
        return EmailServiceFactory.create_smtp_service(config_dict)

    def create_user_repository(self) -> FlextResult[UserRepository]:
        """Create user repository with database dependency - shared implementation."""
        db_result = self._container.get("DatabaseConnection")
        if db_result.is_failure:
            return FlextResult.fail(f"Database connection required: {db_result.error}")
        return FlextResult.ok(SharedDomainUserRepository(db_result.data))

    def create_notification_service(self) -> FlextResult[NotificationService]:
        """Create notification service with dependencies - shared implementation."""
        dependencies = self._resolve_notification_dependencies()
        if dependencies.is_failure:
            return dependencies

        email_service, user_repository = dependencies.data
        return FlextResult.ok(EmailNotificationService(email_service, user_repository))

    def create_user_management_service(self) -> FlextResult[UserManagementService]:
        """Create user management service with dependencies - shared implementation."""
        dependencies = self._resolve_user_management_dependencies()
        if dependencies.is_failure:
            return dependencies

        user_repository, notification_service = dependencies.data
        return FlextResult.ok(
            UserManagementService(user_repository, notification_service)
        )

    def _resolve_notification_dependencies(
        self,
    ) -> FlextResult[tuple[EmailService, UserRepository]]:
        """Resolve notification service dependencies."""
        email_result = self._container.get("EmailService")
        if email_result.is_failure:
            return FlextResult.fail(f"Email service required: {email_result.error}")

        user_repo_result = self._container.get("UserRepository")
        if user_repo_result.is_failure:
            return FlextResult.fail(
                f"User repository required: {user_repo_result.error}"
            )

        return FlextResult.ok((email_result.data, user_repo_result.data))

    def _resolve_user_management_dependencies(
        self,
    ) -> FlextResult[tuple[UserRepository, NotificationService]]:
        """Resolve user management service dependencies."""
        user_repo_result = self._container.get("UserRepository")
        if user_repo_result.is_failure:
            return FlextResult.fail(
                f"User repository required: {user_repo_result.error}"
            )

        notification_result = self._container.get("NotificationService")
        if notification_result.is_failure:
            return FlextResult.fail(
                f"Notification service required: {notification_result.error}"
            )

        return FlextResult.ok((user_repo_result.data, notification_result.data))


class ProductionServiceConfigurer(BaseServiceConfigurer):
    """Production service configurer using Template Method pattern - SOLID LSP."""

    def _get_database_config(self) -> ServiceConfigurationData:
        """Get production database configuration."""
        return ServiceConfigurationData(
            host="prod-db.example.com",
            port=5432,
            database="production_db",
        )

    def _get_email_config(self) -> ServiceConfigurationData:
        """Get production email configuration."""
        return ServiceConfigurationData(
            smtp_host="smtp.gmail.com",
            smtp_port=587,
        )


def setup_production_container() -> FlextResult[FlextContainer]:
    """Setup production container using Result pattern chaining.

    Single return point for consistent error handling.
    """
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

    if setup_result.success:
        print("✅ Production container setup completed")

    return setup_result


# Separate mock service classes for better organization
class MockDatabase(DatabaseConnection):
    """Mock database implementation for testing."""

    def connect(self) -> FlextResult[bool]:
        """Connect to the mock database.

        Returns:
            FlextResult containing True on successful connection.

        """
        print("🧪 Mock database connected")
        return FlextResult.ok(data=True)

    def execute_query(self, _query: str) -> FlextResult[list[TAnyObject]]:
        """Execute a query on the mock database.

        Args:
            _query: SQL query string (ignored in mock implementation).

        Returns:
            FlextResult containing mock query results.

        """
        mock_results: list[TAnyObject] = [{"id": "test", "name": "Test User"}]
        return FlextResult.ok(mock_results)

    def close(self) -> FlextResult[None]:
        """Close the mock database connection.

        Returns:
            FlextResult containing None on successful close.

        """
        print("🧪 Mock database closed")
        return FlextResult.ok(None)


class MockEmailService(EmailService):
    """Mock email service implementation for testing."""

    def send_email(self, _to: str, _subject: str, _body: str) -> FlextResult[str]:
        """Send a mock email.

        Args:
            _to: Email recipient (ignored in mock).
            _subject: Email subject (ignored in mock).
            _body: Email body (ignored in mock).

        Returns:
            FlextResult containing mock message ID.

        """
        message_id: str = "mock_message_123"
        print(f"🧪 Mock email sent: {message_id}")
        return FlextResult.ok(message_id)


class MockUserRepository(UserRepository):
    """Mock user repository implementation for testing."""

    def create_user(self, user_data: TUserData) -> FlextResult[TEntityId]:  # noqa: ARG002
        """Create a mock user.

        Args:
            user_data: User data for creation (ignored in mock).

        Returns:
            FlextResult containing mock user ID.

        """
        user_id: TEntityId = "test_user_123"
        print(f"🧪 Mock user created: {user_id}")
        return FlextResult.ok(user_id)

    def get_user(self, user_id: TEntityId) -> FlextResult[SharedUser]:
        """Get a mock user by ID.

        Args:
            user_id: ID of user to retrieve.

        Returns:
            FlextResult containing mock SharedUser.

        """
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
        """Send mock notification for user creation.

        Args:
            user: The SharedUser to notify about.

        Returns:
            FlextResult containing None on success.

        """
        print(f"🧪 Mock notification sent for shared user: {user.id}")
        return FlextResult.ok(None)


class TestServiceConfigurer:
    """Test service configurer using Strategy pattern for mock services - SOLID SRP."""

    def __init__(self, container: FlextContainer) -> None:
        self._container = container
        self._registration_strategy = ServiceRegistrationStrategy(container)

    def register_mock_services(self) -> FlextResult[None]:
        """Register all mock services using strategy pattern."""
        mock_services = self._create_mock_service_configurations()

        for service_name, service_instance in mock_services:
            service_result = FlextResult.ok(service_instance)
            registration_result = (
                self._registration_strategy.register_service_with_validation(
                    service_name, service_result
                )
            )
            if registration_result.is_failure:
                return FlextResult.fail(registration_result.error)

        return FlextResult.ok(None)

    def _create_mock_service_configurations(self) -> list[tuple[str, object]]:
        """Create mock service configurations - SOLID OCP: Open for extension."""
        return [
            ("DatabaseConnection", MockDatabase()),
            ("EmailService", MockEmailService()),
            ("UserRepository", MockUserRepository()),
            ("NotificationService", MockNotificationService()),
        ]

    def register_user_management_factory(self) -> FlextResult[None]:
        """Register user management service factory using factory creator."""
        factory_creator = MockUserManagementServiceFactoryCreator(self._container)
        user_management_factory = factory_creator.create_factory()

        try:
            self._container.register_factory(
                "UserManagementService", user_management_factory
            )
            return FlextResult.ok(None)
        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult.fail(f"User management factory registration failed: {e}")


class MockUserManagementServiceFactoryCreator:
    """Factory creator for mock UserManagementService - SOLID SRP."""

    def __init__(self, container: FlextContainer) -> None:
        self._container = container

    def create_factory(self) -> callable:
        """Create factory function for mock UserManagementService."""

        def mock_user_management_factory() -> UserManagementService:
            user_repository = cast(
                "UserRepository", self._container.get("UserRepository").data
            )
            notification_service = cast(
                "NotificationService", self._container.get("NotificationService").data
            )
            return UserManagementService(user_repository, notification_service)

        return mock_user_management_factory


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
        db_result = container.get("DatabaseConnection")
        if db_result.success:
            db_connection = cast("DatabaseConnection", db_result.data)
            connect_result = db_connection.connect()
            health_data["services"]["database"] = {
                "status": "healthy" if connect_result.success else "unhealthy",
                "error": connect_result.error if connect_result.is_failure else None,
            }
            if connect_result.success:
                db_connection.close()
        else:
            health_data["services"]["database"] = {
                "status": "unavailable",
                "error": f"Service not found: {db_result.error}",
            }
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
        for service in (
            health_data["services"].values()
            if isinstance(health_data.get("services"), dict)
            else []
        )
        if service["status"] != "healthy"
    ]
    if unhealthy_services:
        health_data["overall_status"] = "unhealthy"

    print(f"✅ Health check completed: {health_data['overall_status']}")
    return FlextResult.ok(health_data)


# =============================================================================
# DEMONSTRATION EXECUTION
# =============================================================================


# =============================================================================
# DEMONSTRATION ORCHESTRATION - SOLID Command Pattern
# =============================================================================


class DemonstrationCommand:
    """Base command for demonstration steps - SOLID Command pattern."""

    def execute(self) -> FlextResult[None]:
        """Execute demonstration command."""
        msg = "Subclasses must implement execute"
        raise NotImplementedError(msg)


class TestContainerSetupCommand(DemonstrationCommand):
    """Command for test container setup - SOLID SRP."""

    def __init__(self, demonstrator: ContainerDemonstrator) -> None:
        self._demonstrator = demonstrator
        self._section = DemonstrationSection(1, "Test Container Setup")

    def execute(self) -> FlextResult[None]:
        """Execute test container setup."""
        formatter = DemonstrationFormatter()
        formatter.print_section_header(self._section)

        test_container_result = setup_test_container()
        if test_container_result.is_failure:
            return FlextResult.fail(
                f"Test container setup failed: {test_container_result.error}"
            )

        self._demonstrator.test_container = test_container_result.data
        return FlextResult.ok(None)


class ContainerDemonstrator:
    """Orchestrates container demonstration using Command pattern."""

    def __init__(self) -> None:
        self.test_container: FlextContainer | None = None
        self.prod_container: FlextContainer | None = None
        self._formatter = DemonstrationFormatter()
        self._validator = PrerequisiteValidator()

    def run_test_container_demo(self) -> FlextResult[None]:
        """Run test container setup using command pattern."""
        command = TestContainerSetupCommand(self)
        return command.execute()

    def _register_user_with_container(
        self, container: FlextContainer, user_data: TUserData, context_name: str
    ) -> FlextResult[None]:
        """Register user with container using strategy pattern."""
        user_registration_strategy = UserRegistrationStrategy(container, context_name)
        return user_registration_strategy.register_user(user_data)

    def run_user_registration_test(self) -> FlextResult[None]:
        """Run user registration test using command pattern."""
        command = UserRegistrationTestCommand(
            self,
            self.test_container,
            "test",
            DemonstrationSection(2, "User Registration with Test Container"),
        )
        return command.execute()

    def run_health_check_demo(self) -> FlextResult[None]:
        """Run health check demo using command pattern."""
        command = HealthCheckDemoCommand(
            self,
            self.test_container,
            DemonstrationSection(3, "Health Check Demo"),
        )
        return command.execute()

    def run_production_container_demo(self) -> FlextResult[None]:
        """Run production container demo using command pattern."""
        command = ProductionContainerDemoCommand(
            self,
            DemonstrationSection(4, "Production Container Demo"),
        )
        return command.execute()


class UserRegistrationStrategy:
    """Strategy for user registration operations - SOLID SRP."""

    def __init__(self, container: FlextContainer, context_name: str) -> None:
        self._container = container
        self._context_name = context_name

    def register_user(self, user_data: TUserData) -> FlextResult[None]:
        """Register user using strategy pattern."""
        service_result = self._get_user_management_service()
        if service_result.is_failure:
            return service_result

        user_service = service_result.data
        assert user_service is not None
        registration_result = user_service.register_user(user_data)

        return self._handle_registration_result(registration_result)

    def _get_user_management_service(self) -> FlextResult[UserManagementService]:
        """Get user management service from container."""
        try:
            user_service_result = self._container.get("UserManagementService")
            if user_service_result.is_failure:
                return FlextResult.fail(
                    f"Failed to get {self._context_name} user service: "
                    f"{user_service_result.error}"
                )
            return FlextResult.ok(
                cast("UserManagementService", user_service_result.data)
            )
        except (RuntimeError, ValueError, TypeError) as e:
            return FlextResult.fail(f"{self._context_name.title()} service error: {e}")

    def _handle_registration_result(
        self, registration_result: FlextResult[TAnyObject]
    ) -> FlextResult[None]:
        """Handle user registration result."""
        if registration_result.success:
            print(
                f"✅ {self._context_name.title()} registration successful: "
                f"{registration_result.data}"
            )
            return FlextResult.ok(None)
        return FlextResult.fail(
            f"{self._context_name.title()} registration failed: "
            f"{registration_result.error}"
        )


class UserRegistrationTestCommand(DemonstrationCommand):
    """Command for user registration test - SOLID SRP."""

    def __init__(
        self,
        demonstrator: ContainerDemonstrator,
        container: FlextContainer | None,
        context_name: str,
        section: DemonstrationSection,
    ) -> None:
        self._demonstrator = demonstrator
        self._container = container
        self._context_name = context_name
        self._section = section

    def execute(self) -> FlextResult[None]:
        """Execute user registration test."""
        # Validate prerequisites
        prerequisite_result = PrerequisiteValidator.validate_condition(
            condition=self._container is not None,
            error_message="Test container not initialized",
        )
        if prerequisite_result.is_failure:
            return prerequisite_result

        # Print section header
        DemonstrationFormatter.print_section_header(self._section)

        # Create test data
        test_user_data: TUserData = {
            "name": "Test User",
            "email": "test@example.com",
            "age": 25,
        }

        # Register user
        return self._demonstrator._register_user_with_container(
            self._container, test_user_data, self._context_name
        )

    def run_health_check_demo(self) -> FlextResult[None]:
        """Run health check demo using command pattern."""
        command = HealthCheckDemoCommand(
            self, self.test_container, DemonstrationSection(3, "Container Health Check")
        )
        return command.execute()


class HealthCheckDemoCommand(DemonstrationCommand):
    """Command for health check demonstration - SOLID SRP."""

    def __init__(
        self,
        demonstrator: ContainerDemonstrator,
        container: FlextContainer | None,
        section: DemonstrationSection,
    ) -> None:
        self._demonstrator = demonstrator
        self._container = container
        self._section = section

    def execute(self) -> FlextResult[None]:
        """Execute health check demonstration."""
        # Validate prerequisites
        prerequisite_result = PrerequisiteValidator.validate_condition(
            condition=self._container is not None,
            error_message="Test container not initialized",
        )
        if prerequisite_result.is_failure:
            return prerequisite_result

        # Print section header
        DemonstrationFormatter.print_section_header(self._section)

        # Execute health check
        health_result = check_container_health(self._container)
        if health_result.success:
            self._display_health_results(health_result.data)
            return FlextResult.ok(None)
        return FlextResult.fail(f"Health check failed: {health_result.error}")

    def _display_health_results(self, health_data: TAnyObject) -> None:
        """Display health check results."""
        print(f"🏥 Container health: {health_data['overall_status']}")
        for service_name, service_health in (
            health_data["services"].items()
            if isinstance(health_data.get("services"), dict)
            else []
        ):
            print(f"   {service_name}: {service_health['status']}")

    def run_production_container_demo(self) -> FlextResult[None]:
        """Run production container demo using command pattern."""
        command = ProductionContainerDemoCommand(
            self, DemonstrationSection(4, "Production Container Setup")
        )
        return command.execute()


class ProductionContainerDemoCommand(DemonstrationCommand):
    """Command for production container demonstration - SOLID SRP."""

    def __init__(
        self, demonstrator: ContainerDemonstrator, section: DemonstrationSection
    ) -> None:
        self._demonstrator = demonstrator
        self._section = section

    def execute(self) -> FlextResult[None]:
        """Execute production container demonstration."""
        # Print section header
        DemonstrationFormatter.print_section_header(self._section)

        # Setup production container
        prod_container_result = setup_production_container()
        if prod_container_result.is_failure:
            return FlextResult.fail(
                f"Production container setup failed: {prod_container_result.error}"
            )

        self._demonstrator.prod_container = prod_container_result.data
        print("✅ Production container setup successful")

        # Test production user registration
        return self._test_production_user_registration()

    def _test_production_user_registration(self) -> FlextResult[None]:
        """Test user registration with production container."""
        # Validate prerequisites
        prerequisite_result = PrerequisiteValidator.validate_condition(
            condition=self._demonstrator.prod_container is not None,
            error_message="Production container not initialized",
        )
        if prerequisite_result.is_failure:
            return prerequisite_result

        # Create production user data
        prod_user_data: TUserData = {
            "name": "Production User",
            "email": "prod@example.com",
            "age": 30,
        }

        # Register user
        return self._demonstrator._register_user_with_container(
            self._demonstrator.prod_container, prod_user_data, "production"
        )


def main() -> None:
    """Run comprehensive FlextContainer demonstration using command pattern."""
    demonstration_orchestrator = DemonstrationOrchestrator()
    demonstration_orchestrator.run_complete_demonstration()


class DemonstrationOrchestrator:
    """Orchestrates complete demonstration using Command pattern - SOLID SRP."""

    def __init__(self) -> None:
        self._demonstrator = ContainerDemonstrator()

    def run_complete_demonstration(self) -> None:
        """Run complete demonstration with consistent error handling."""
        self._print_demonstration_header()

        demonstration_steps = self._create_demonstration_steps()

        for step in demonstration_steps:
            result = step()
            if result.is_failure:
                print(f"❌ Demonstration step failed: {result.error}")
                return

        self._print_demonstration_footer()

    def _print_demonstration_header(self) -> None:
        """Print demonstration header."""
        print("=" * 80)
        print("🚀 FLEXT CONTAINER - DEPENDENCY INJECTION DEMONSTRATION")
        print("=" * 80)

    def _print_demonstration_footer(self) -> None:
        """Print demonstration footer."""
        print("\n" + "=" * 80)
        print("🎉 FLEXT CONTAINER DEMONSTRATION COMPLETED")
        print("=" * 80)

    def _create_demonstration_steps(self) -> list[callable]:
        """Create demonstration steps - SOLID OCP: Open for extension."""
        return [
            self._demonstrator.run_test_container_demo,
            self._demonstrator.run_user_registration_test,
            self._demonstrator.run_health_check_demo,
            self._demonstrator.run_production_container_demo,
        ]


if __name__ == "__main__":
    main()
