"""Comprehensive tests for FlextContainer dependency injection system - Real functional testing."""

from __future__ import annotations

from typing import Any

import pytest

from flext_core import FlextContainer, FlextResult, get_flext_container
from flext_core.legacy import (
    FlextServiceKey,
    FlextServiceRegistrar,
    FlextServiceRetriever,
    GetServiceQuery,
    ListServicesQuery,
    RegisterFactoryCommand,
    RegisterServiceCommand,
    UnregisterServiceCommand,
)

# Test markers
pytestmark = [pytest.mark.unit, pytest.mark.core]


# =============================================================================
# TEST DOMAIN CLASSES - Real business objects for testing
# =============================================================================


class DatabaseService:
    """Real database service for testing."""

    def __init__(self, connection_string: str = "sqlite:///test.db") -> None:
        self.connection_string = connection_string
        self.connected = False

    def connect(self) -> FlextResult[None]:
        """Connect to database."""
        self.connected = True
        return FlextResult[None].ok(None)

    def execute_query(self, query: str, *args: object) -> FlextResult[list[dict[str, Any]]]:
        """Execute database query with parameters."""
        if not self.connected:
            return FlextResult[list[dict[str, Any]]].fail("Database not connected")
        params_str = ", ".join(str(arg) for arg in args) if args else ""
        result_msg = f"Executed: {query}" + (f" with params: [{params_str}]" if params_str else "")
        return FlextResult[list[dict[str, Any]]].ok([{"result": result_msg}])


class EmailService:
    """Real email service for testing."""

    def __init__(self, smtp_server: str = "localhost", port: int = 587) -> None:
        self.smtp_server = smtp_server
        self.port = port
        self.emails_sent: list[dict[str, Any]] = []

    def send_email(self, to: str, subject: str, body: str) -> FlextResult[str]:
        """Send email and return message ID."""
        email = {"to": to, "subject": subject, "body": body}
        self.emails_sent.append(email)
        message_id = f"msg_{len(self.emails_sent)}"
        return FlextResult[str].ok(message_id)


class UserService:
    """Real user service with dependency injection."""

    def __init__(self, database: DatabaseService, email: EmailService) -> None:
        self.database = database
        self.email = email

    def create_user(self, name: str, email: str) -> FlextResult[dict[str, Any]]:
        """Create user using injected dependencies."""
        # Use database to store user - safe for testing with mock database
        db_result = self.database.execute_query(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            name,
            email,
        )
        if db_result.is_failure:
            return FlextResult[dict[str, Any]].fail(
                f"Database error: {db_result.error}"
            )

        # Send welcome email
        email_result = self.email.send_email(
            email, "Welcome!", f"Hello {name}, welcome to our service!"
        )
        if email_result.is_failure:
            return FlextResult[dict[str, Any]].fail(
                f"Email error: {email_result.error}"
            )

        user = {"name": name, "email": email, "message_id": email_result.value}
        return FlextResult[dict[str, Any]].ok(user)


class ConfigService:
    """Configuration service that can be created from factory."""

    def __init__(self, environment: str = "test") -> None:
        self.environment = environment
        self.config = {
            "database_url": f"{environment}_db.sqlite",
            "email_enabled": environment != "test",
            "debug": environment == "development",
        }

    def get_config(self, key: str) -> str | bool | None:
        """Get configuration value."""
        return self.config.get(key)


# =============================================================================
# FLEXT SERVICE KEY TESTS - Type-safe service keys
# =============================================================================


class TestFlextServiceKeyFunctionality:
    """Test FlextServiceKey type-safe service key functionality."""

    def test_service_key_basic_creation(self) -> None:
        """Test FlextServiceKey basic creation and string behavior."""
        key = FlextServiceKey[DatabaseService]("database")

        assert str(key) == "database"
        assert key.name == "database"
        assert key == "database"  # String equality

    def test_service_key_type_subscription(self) -> None:
        """Test FlextServiceKey generic type subscription."""
        # Test with different types
        db_key = FlextServiceKey[DatabaseService]("db")
        email_key = FlextServiceKey[EmailService]("email")
        config_key = FlextServiceKey[ConfigService]("config")

        assert all(
            isinstance(key, FlextServiceKey) for key in [db_key, email_key, config_key]
        )
        assert str(db_key) == "db"
        assert str(email_key) == "email"
        assert str(config_key) == "config"

    def test_service_key_class_getitem(self) -> None:
        """Test FlextServiceKey __class_getitem__ for type subscription."""
        # Should return the same class regardless of generic argument
        subscripted_class = FlextServiceKey[DatabaseService]
        assert subscripted_class == FlextServiceKey

        # Should work with any type
        any_type_class = FlextServiceKey[Any]
        assert any_type_class == FlextServiceKey

    def test_service_key_string_operations(self) -> None:
        """Test FlextServiceKey string operations and compatibility."""
        key = FlextServiceKey[DatabaseService]("test_service")

        # String concatenation
        assert key + "_suffix" == "test_service_suffix"
        assert "prefix_" + key == "prefix_test_service"

        # String formatting
        assert f"Service: {key}" == "Service: test_service"

        # String methods
        assert key.upper() == "TEST_SERVICE"
        assert key.startswith("test")
        assert key.endswith("service")


# =============================================================================
# CONTAINER COMMANDS TESTS - CQRS pattern functionality
# =============================================================================


class TestContainerCommandsFunctionality:
    """Test container commands using CQRS patterns."""

    def test_register_service_command_creation(self) -> None:
        """Test RegisterServiceCommand creation and validation."""
        service = DatabaseService()
        command = RegisterServiceCommand.create("database", service)

        assert command.service_name == "database"
        assert command.service_instance is service
        assert command.command_type == "register_service"
        assert command.command_id is not None
        assert command.correlation_id is not None
        assert command.timestamp is not None

    def test_register_service_command_validation_success(self) -> None:
        """Test RegisterServiceCommand validation success."""
        service = DatabaseService()
        command = RegisterServiceCommand.create("valid_service", service)

        result = command.validate_command()
        assert result.is_success

    def test_register_service_command_validation_failure(self) -> None:
        """Test RegisterServiceCommand validation failure."""
        service = DatabaseService()
        command = RegisterServiceCommand.create("", service)  # Empty name

        result = command.validate_command()
        assert result.is_failure
        assert "Service name cannot be empty" in (result.error or "")

        # Whitespace-only name
        command_whitespace = RegisterServiceCommand.create("   ", service)
        result = command_whitespace.validate_command()
        assert result.is_failure

    def test_register_factory_command_creation(self) -> None:
        """Test RegisterFactoryCommand creation and validation."""
        factory = DatabaseService
        command = RegisterFactoryCommand.create("database_factory", factory)

        assert command.service_name == "database_factory"
        assert command.factory is factory
        assert command.command_type == "register_factory"
        assert callable(command.factory)

    def test_register_factory_command_validation_success(self) -> None:
        """Test RegisterFactoryCommand validation success."""
        factory = DatabaseService
        command = RegisterFactoryCommand.create("valid_factory", factory)

        result = command.validate_command()
        assert result.is_success

    def test_register_factory_command_validation_failures(self) -> None:
        """Test RegisterFactoryCommand validation failures."""
        factory = DatabaseService

        # Empty service name
        command = RegisterFactoryCommand.create("", factory)
        result = command.validate_command()
        assert result.is_failure

        # Non-callable factory
        non_callable_factory = "not_callable"
        command_bad_factory = RegisterFactoryCommand.create(
            "test", non_callable_factory
        )
        result = command_bad_factory.validate_command()
        assert result.is_failure
        assert "Factory must be callable" in (result.error or "")

    def test_unregister_service_command_functionality(self) -> None:
        """Test UnregisterServiceCommand creation and validation."""
        command = UnregisterServiceCommand.create("test_service")

        assert command.service_name == "test_service"
        assert command.command_type == "unregister_service"

        # Validation success
        result = command.validate_command()
        assert result.is_success

        # Validation failure
        empty_command = UnregisterServiceCommand.create("")
        result = empty_command.validate_command()
        assert result.is_failure

    def test_get_service_query_functionality(self) -> None:
        """Test GetServiceQuery creation and validation."""
        query = GetServiceQuery.create("database", "DatabaseService")

        assert query.service_name == "database"
        assert query.expected_type == "DatabaseService"
        assert query.query_type == "get_service"

        # Validation success
        result = query.validate_query()
        assert result.is_success

        # Validation failure
        empty_query = GetServiceQuery.create("")
        result = empty_query.validate_query()
        assert result.is_failure

    def test_list_services_query_functionality(self) -> None:
        """Test ListServicesQuery creation and defaults."""
        query = ListServicesQuery.create()

        assert query.include_factories is True
        assert query.service_type_filter is None
        assert query.query_type == "list_services"

        # With custom parameters
        custom_query = ListServicesQuery.create(
            include_factories=False, service_type_filter="Database"
        )
        assert custom_query.include_factories is False
        assert custom_query.service_type_filter == "Database"


# =============================================================================
# SERVICE REGISTRAR TESTS - Registration component functionality
# =============================================================================


class TestFlextServiceRegistrarFunctionality:
    """Test FlextServiceRegistrar registration functionality."""

    def test_service_registrar_initialization(self) -> None:
        """Test FlextServiceRegistrar initialization."""
        registrar = FlextServiceRegistrar()

        assert len(registrar.get_services_dict()) == 0
        assert len(registrar.get_factories_dict()) == 0
        assert registrar.get_service_count() == 0

    def test_register_service_success(self) -> None:
        """Test successful service registration."""
        registrar = FlextServiceRegistrar()
        service = DatabaseService("postgresql://test")

        result = registrar.register_service("database", service)
        assert result.is_success

        # Verify registration
        services = registrar.get_services_dict()
        assert "database" in services
        assert services["database"] is service
        assert registrar.get_service_count() == 1
        assert registrar.has_service("database")

    def test_register_service_validation_failures(self) -> None:
        """Test service registration validation failures."""
        registrar = FlextServiceRegistrar()
        service = DatabaseService()

        # Empty service name
        result = registrar.register_service("", service)
        assert result.is_failure
        assert "Service name cannot be empty" in (result.error or "")

        # Whitespace-only name
        result = registrar.register_service("   ", service)
        assert result.is_failure

    def test_register_service_replacement(self) -> None:
        """Test service registration replacement."""
        registrar = FlextServiceRegistrar()
        service1 = DatabaseService("sqlite:///db1.db")
        service2 = DatabaseService("sqlite:///db2.db")

        # Register first service
        result = registrar.register_service("database", service1)
        assert result.is_success

        # Replace with second service
        result = registrar.register_service("database", service2)
        assert result.is_success

        # Should have the second service
        services = registrar.get_services_dict()
        assert services["database"] is service2
        assert services["database"].connection_string == "sqlite:///db2.db"
        assert registrar.get_service_count() == 1

    def test_register_factory_success(self) -> None:
        """Test successful factory registration."""
        registrar = FlextServiceRegistrar()

        def factory() -> DatabaseService:
            return DatabaseService("factory_db")

        result = registrar.register_factory("database_factory", factory)
        assert result.is_success

        # Verify registration
        factories = registrar.get_factories_dict()
        assert "database_factory" in factories
        assert factories["database_factory"] is factory
        assert registrar.get_service_count() == 1
        assert registrar.has_service("database_factory")

    def test_register_factory_validation_failures(self) -> None:
        """Test factory registration validation failures."""
        registrar = FlextServiceRegistrar()

        # Non-callable factory
        result = registrar.register_factory("bad_factory", "not_callable")
        assert result.is_failure
        assert "Factory must be callable" in (result.error or "")

        # Factory with required parameters
        def factory_with_params(required_param: str) -> DatabaseService:
            return DatabaseService(required_param)

        result = registrar.register_factory("param_factory", factory_with_params)
        assert result.is_failure
        assert "must be callable without parameters" in (result.error or "")

    def test_register_factory_signature_inspection(self) -> None:
        """Test factory signature inspection for parameter requirements."""
        registrar = FlextServiceRegistrar()

        # Valid factory - no parameters
        def valid_factory() -> ConfigService:
            return ConfigService("production")

        result = registrar.register_factory("valid", valid_factory)
        assert result.is_success

        # Valid factory - optional parameters only
        def optional_factory(env: str = "test") -> ConfigService:
            return ConfigService(env)

        result = registrar.register_factory("optional", optional_factory)
        assert result.is_success

        # Invalid factory - required parameter
        def invalid_factory(required: str) -> ConfigService:
            return ConfigService(required)

        result = registrar.register_factory("invalid", invalid_factory)
        assert result.is_failure
        assert "requires 1 parameter(s)" in (result.error or "")

    def test_register_factory_replaces_service(self) -> None:
        """Test that registering factory replaces existing service."""
        registrar = FlextServiceRegistrar()
        service = DatabaseService()

        def factory() -> DatabaseService:
            return DatabaseService("factory_created")

        # Register service first
        result = registrar.register_service("database", service)
        assert result.is_success

        # Register factory with same name
        result = registrar.register_factory("database", factory)
        assert result.is_success

        # Service should be removed, factory should exist
        services = registrar.get_services_dict()
        factories = registrar.get_factories_dict()
        assert "database" not in services
        assert "database" in factories
        assert registrar.get_service_count() == 1

    def test_unregister_service_success(self) -> None:
        """Test successful service unregistration."""
        registrar = FlextServiceRegistrar()
        service = DatabaseService()

        # Register and unregister service
        registrar.register_service("database", service)
        result = registrar.unregister_service("database")
        assert result.is_success

        assert not registrar.has_service("database")
        assert registrar.get_service_count() == 0

    def test_unregister_factory_success(self) -> None:
        """Test successful factory unregistration."""
        registrar = FlextServiceRegistrar()
        factory = DatabaseService

        # Register and unregister factory
        registrar.register_factory("database", factory)
        result = registrar.unregister_service("database")
        assert result.is_success

        assert not registrar.has_service("database")
        assert registrar.get_service_count() == 0

    def test_unregister_nonexistent_service(self) -> None:
        """Test unregistering nonexistent service."""
        registrar = FlextServiceRegistrar()

        result = registrar.unregister_service("nonexistent")
        assert result.is_failure
        assert "not found" in (result.error or "")

    def test_clear_all_services(self) -> None:
        """Test clearing all services and factories."""
        registrar = FlextServiceRegistrar()

        # Register multiple services and factories
        registrar.register_service("service1", DatabaseService())
        registrar.register_service("service2", EmailService())
        registrar.register_factory("factory1", ConfigService)

        assert registrar.get_service_count() == 3

        result = registrar.clear_all()
        assert result.is_success
        assert registrar.get_service_count() == 0
        assert len(registrar.get_service_names()) == 0

    def test_service_utilities(self) -> None:
        """Test service utility methods."""
        registrar = FlextServiceRegistrar()

        # Register services
        registrar.register_service("db", DatabaseService())
        registrar.register_factory("email", EmailService)

        # Test utility methods
        names = registrar.get_service_names()
        assert "db" in names
        assert "email" in names
        assert len(names) == 2

        assert registrar.has_service("db")
        assert registrar.has_service("email")
        assert not registrar.has_service("nonexistent")

        assert registrar.get_service_count() == 2


# =============================================================================
# SERVICE RETRIEVER TESTS - Retrieval component functionality
# =============================================================================


class TestFlextServiceRetrieverFunctionality:
    """Test FlextServiceRetriever service retrieval functionality."""

    def test_service_retriever_initialization(self) -> None:
        """Test FlextServiceRetriever initialization."""
        services: dict[str, object] = {}
        factories: dict[str, object] = {}

        retriever = FlextServiceRetriever(services, factories)
        assert retriever is not None

    def test_get_service_from_registry(self) -> None:
        """Test retrieving service from service registry."""
        service = DatabaseService("test_db")
        services = {"database": service}
        factories: dict[str, object] = {}

        retriever = FlextServiceRetriever(services, factories)
        result = retriever.get_service("database")

        assert result.is_success
        assert result.value is service

    def test_get_service_from_factory(self) -> None:
        """Test retrieving service from factory."""

        def factory() -> ConfigService:
            return ConfigService("test_env")

        services: dict[str, object] = {}
        factories = {"config": factory}

        retriever = FlextServiceRetriever(services, factories)
        result = retriever.get_service("config")

        assert result.is_success
        assert isinstance(result.value, ConfigService)
        assert result.value.environment == "test_env"

    def test_get_service_factory_failure(self) -> None:
        """Test service retrieval when factory fails."""

        def failing_factory() -> ConfigService:
            msg = "Factory intentionally failed"
            raise ValueError(msg)

        services: dict[str, object] = {}
        factories = {"failing": failing_factory}

        retriever = FlextServiceRetriever(services, factories)
        result = retriever.get_service("failing")

        assert result.is_failure
        assert "failed" in (result.error or "")

    def test_get_nonexistent_service(self) -> None:
        """Test retrieving nonexistent service."""
        services: dict[str, object] = {}
        factories: dict[str, object] = {}

        retriever = FlextServiceRetriever(services, factories)
        result = retriever.get_service("nonexistent")

        assert result.is_failure
        assert "not found" in (result.error or "")

    def test_get_typed_service_success(self) -> None:
        """Test typed service retrieval success via FlextContainer."""
        container = FlextContainer()
        service = DatabaseService()

        # Register service
        reg_result = container.register("database", service)
        assert reg_result.is_success

        # Get typed service
        result = container.get_typed("database", DatabaseService)

        assert result.is_success
        assert result.value is service
        assert isinstance(result.value, DatabaseService)

    def test_get_typed_service_type_mismatch(self) -> None:
        """Test typed service retrieval with type mismatch via FlextContainer."""
        container = FlextContainer()
        service = DatabaseService()

        # Register service as database
        reg_result = container.register("database", service)
        assert reg_result.is_success

        # Try to get as EmailService (wrong type)
        result = container.get_typed("database", EmailService)

        assert result.is_failure
        # Check the actual error message from container
        assert any(
            word in (result.error or "").lower()
            for word in ["expected", "databaseservice", "emailservice"]
        )


# =============================================================================
# FLEXT CONTAINER INTEGRATION TESTS - Full container functionality
# =============================================================================


class TestFlextContainerIntegrationFunctionality:
    """Test FlextContainer full integration functionality."""

    def test_container_initialization(self) -> None:
        """Test FlextContainer initialization."""
        container = FlextContainer()

        assert container is not None
        assert container.get_service_count() == 0

    def test_container_service_registration_and_retrieval(self) -> None:
        """Test complete service registration and retrieval workflow."""
        container = FlextContainer()

        # Register services
        db_service = DatabaseService("postgresql://localhost/test")
        email_service = EmailService("smtp.test.com", 587)

        db_result = container.register("database", db_service)
        assert db_result.is_success

        email_result = container.register("email", email_service)
        assert email_result.is_success

        # Retrieve services
        db_retrieved = container.get("database")
        assert db_retrieved.is_success
        assert db_retrieved.value is db_service

        email_retrieved = container.get("email")
        assert email_retrieved.is_success
        assert email_retrieved.value is email_service

        # Verify container state
        assert container.get_service_count() == 2
        assert container.has("database")
        assert container.has("email")

    def test_container_factory_registration_and_instantiation(self) -> None:
        """Test factory registration and lazy instantiation."""
        container = FlextContainer()

        # Register factory
        def config_factory() -> ConfigService:
            return ConfigService("production")

        result = container.register_factory("config", config_factory)
        assert result.is_success

        # First retrieval should instantiate
        config1 = container.get("config")
        assert config1.is_success
        assert isinstance(config1.value, ConfigService)
        assert config1.value.environment == "production"

        # Second retrieval should return same instance (cached)
        config2 = container.get("config")
        assert config2.is_success
        assert config2.value is config1.value  # Same instance

    def test_container_get_or_create_functionality(self) -> None:
        """Test get_or_create method functionality."""
        container = FlextContainer()

        # Service doesn't exist, should create it
        def factory() -> DatabaseService:
            return DatabaseService("created_on_demand")

        result = container.get_or_create("database", factory)

        assert result.is_success
        assert isinstance(result.value, DatabaseService)
        assert result.value.connection_string == "created_on_demand"

        # Service exists now, should return existing
        result2 = container.get_or_create(
            "database", lambda: DatabaseService("should_not_create")
        )
        assert result2.is_success
        assert result2.value is result.value  # Same instance

    def test_container_auto_wire_functionality(self) -> None:
        """Test auto-wire dependency injection functionality."""
        container = FlextContainer()

        # Register dependencies first
        db_service = DatabaseService()
        db_service.connect()  # Ensure connected
        email_service = EmailService()

        container.register("database", db_service)
        container.register("email", email_service)

        # Auto-wire UserService (depends on database and email)
        result = container.auto_wire(UserService, "user_service")
        assert result.is_success

        user_service = result.value
        assert isinstance(user_service, UserService)
        assert user_service.database is db_service
        assert user_service.email is email_service

        # Test the auto-wired service functionality
        user_result = user_service.create_user("John Doe", "john@example.com")
        assert user_result.is_success

        user_data = user_result.value
        assert user_data["name"] == "John Doe"
        assert user_data["email"] == "john@example.com"
        assert "message_id" in user_data

    def test_container_auto_wire_missing_dependency(self) -> None:
        """Test auto-wire failure when dependency is missing."""
        container = FlextContainer()

        # Register only one of the required dependencies
        container.register("email", EmailService())

        # Auto-wire should fail due to missing database dependency
        result = container.auto_wire(UserService)
        assert result.is_failure
        assert "dependency 'database' not found" in (result.error or "").lower()

    def test_container_batch_registration_success(self) -> None:
        """Test successful batch registration."""
        container = FlextContainer()

        registrations = {
            "database": DatabaseService("batch_db"),
            "email": EmailService("batch_smtp"),
            "config_factory": lambda: ConfigService("batch_env"),
        }

        result = container.batch_register(registrations)
        assert result.is_success
        assert len(result.value) == 3
        assert "database" in result.value
        assert "email" in result.value
        assert "config_factory" in result.value

        # Verify all services are registered
        assert container.has("database")
        assert container.has("email")
        assert container.has("config_factory")
        assert container.get_service_count() == 3

    def test_container_batch_registration_replacement_behavior(self) -> None:
        """Test batch registration with service replacement behavior."""
        container = FlextContainer()

        # Pre-register a service
        original_service = DatabaseService("original")
        container.register("existing", original_service)
        assert container.get_service_count() == 1

        # Batch registration that includes replacement of existing service
        replacement_service = ConfigService("replacement")
        registrations = {
            "valid1": EmailService(),
            "existing": replacement_service,  # Should replace the existing service
            "valid2": DatabaseService("new_service"),
        }

        result = container.batch_register(registrations)
        assert result.is_success
        assert result.value == ["valid1", "existing", "valid2"]

        # All services should be registered/replaced
        assert container.get_service_count() == 3
        assert container.has("valid1")
        assert container.has("existing")
        assert container.has("valid2")

        # The existing service should be replaced
        existing_result = container.get("existing")
        assert existing_result.is_success
        assert isinstance(existing_result.value, ConfigService)
        assert existing_result.value is replacement_service

    def test_container_service_inspection_functionality(self) -> None:
        """Test container service inspection methods."""
        container = FlextContainer()

        # Register various services
        container.register("db", DatabaseService())
        container.register_factory("email", EmailService)
        container.register("config", ConfigService())

        # Test inspection methods
        names = container.get_service_names()
        assert len(names) == 3
        assert "db" in names
        assert "email" in names
        assert "config" in names

        assert container.get_service_count() == 3

        # Test service existence checks
        assert container.has("db")
        assert container.has("email")
        assert container.has("config")
        assert not container.has("nonexistent")

    def test_container_service_lifecycle_management(self) -> None:
        """Test complete service lifecycle management."""
        container = FlextContainer()

        # Register service
        service = DatabaseService("lifecycle_test")
        container.register("test_service", service)
        assert container.has("test_service")

        # Retrieve service
        retrieved = container.get("test_service")
        assert retrieved.is_success
        assert retrieved.value is service

        # Replace service
        new_service = DatabaseService("replaced")
        container.register("test_service", new_service)
        retrieved_replaced = container.get("test_service")
        assert retrieved_replaced.is_success
        assert retrieved_replaced.value is new_service
        assert retrieved_replaced.value is not service

        # Unregister service
        unregister_result = container.unregister("test_service")
        assert unregister_result.is_success
        assert not container.has("test_service")

        # Attempt to retrieve unregistered service
        after_unregister = container.get("test_service")
        assert after_unregister.is_failure

    def test_container_clear_all_functionality(self) -> None:
        """Test clearing all services and factories."""
        container = FlextContainer()

        # Register multiple services and factories
        container.register("service1", DatabaseService())
        container.register("service2", EmailService())
        container.register_factory("factory1", ConfigService)

        assert container.get_service_count() == 3

        # Clear all
        result = container.clear()
        assert result.is_success
        assert container.get_service_count() == 0
        assert len(container.get_service_names()) == 0


# =============================================================================
# GLOBAL CONTAINER TESTS - Singleton pattern functionality
# =============================================================================


class TestGlobalContainerFunctionality:
    """Test global container singleton functionality."""

    def test_get_flext_container_singleton(self) -> None:
        """Test get_flext_container returns singleton instance."""
        container1 = get_flext_container()
        container2 = get_flext_container()

        assert container1 is container2
        assert isinstance(container1, FlextContainer)

    def test_global_container_persistence(self) -> None:
        """Test global container persists services across calls."""
        # Clear any existing services first
        global_container = get_flext_container()
        global_container.clear()

        # Register service through first reference
        container1 = get_flext_container()
        service = DatabaseService("global_test")
        container1.register("global_db", service)

        # Retrieve service through second reference
        container2 = get_flext_container()
        retrieved = container2.get("global_db")

        assert retrieved.is_success
        assert retrieved.value is service

    def test_global_container_isolation_per_test(self) -> None:
        """Test that each test can use global container independently."""
        # Note: This test assumes test isolation is handled by test framework
        container = get_flext_container()

        # Register test-specific service
        test_service = EmailService("test_isolation")
        container.register("isolation_test", test_service)

        # Verify service exists
        retrieved = container.get("isolation_test")
        assert retrieved.is_success
        assert retrieved.value is test_service


# =============================================================================
# COMPLEX INTEGRATION SCENARIOS - Real-world usage patterns
# =============================================================================


class TestContainerComplexIntegrationScenarios:
    """Test complex real-world integration scenarios."""

    def test_layered_service_architecture(self) -> None:
        """Test layered service architecture with dependency chains."""
        container = FlextContainer()

        # Layer 1: Infrastructure services
        db = DatabaseService("postgresql://prod")
        db.connect()
        container.register("database", db)
        container.register("email", EmailService("prod.smtp.com"))

        # Layer 2: Domain services (depend on infrastructure)
        user_service = UserService(
            container.get("database").unwrap_or(db),
            container.get("email").unwrap_or(EmailService()),
        )
        container.register("user_service", user_service)

        # Layer 3: Application services
        def notification_factory() -> dict[str, Any]:
            user_svc = container.get("user_service").unwrap_or(user_service)
            return {"user_service": user_svc, "type": "notification"}

        container.register_factory("notification_service", notification_factory)

        # Test the complete architecture
        notification = container.get("notification_service")
        assert notification.is_success
        notification_data = notification.value
        assert "user_service" in notification_data
        assert notification_data["type"] == "notification"

    def test_service_replacement_and_migration(self) -> None:
        """Test service replacement patterns for migrations."""
        container = FlextContainer()

        # Original implementation
        old_db = DatabaseService("sqlite:///old.db")
        container.register("database", old_db)

        # Application using the service
        user_service = UserService(
            container.get("database").unwrap_or(old_db), EmailService()
        )

        # Migration: replace with new implementation
        new_db = DatabaseService("postgresql://new")
        new_db.connect()
        container.register("database", new_db)

        # New application instance should get new database
        new_user_service = UserService(
            container.get("database").unwrap_or(new_db), EmailService()
        )

        assert user_service.database is old_db
        assert new_user_service.database is new_db

    def test_conditional_service_registration(self) -> None:
        """Test conditional service registration patterns."""
        container = FlextContainer()

        environment = "production"

        # Conditional registration based on environment
        if environment == "production":
            container.register("database", DatabaseService("postgresql://prod"))
            container.register("email", EmailService("prod.smtp.com", 587))
        else:
            container.register("database", DatabaseService("sqlite:///test.db"))
            container.register_factory("email", lambda: EmailService("localhost", 1025))

        # Services should reflect production configuration
        db_service = container.get("database")
        assert db_service.is_success
        assert "postgresql" in db_service.value.connection_string

        email_service = container.get("email")
        assert email_service.is_success
        assert email_service.value.smtp_server == "prod.smtp.com"

    def test_service_composition_patterns(self) -> None:
        """Test service composition and aggregation patterns."""
        container = FlextContainer()

        # Register base services
        container.register("database", DatabaseService())
        container.register("email", EmailService())
        container.register("config", ConfigService("production"))

        # Composite service factory
        def application_context_factory() -> dict[str, Any]:
            return {
                "database": container.get("database").unwrap_or(DatabaseService()),
                "email": container.get("email").unwrap_or(EmailService()),
                "config": container.get("config").unwrap_or(ConfigService()),
                "environment": "production",
                "features": ["user_management", "email_notifications"],
            }

        container.register_factory("app_context", application_context_factory)

        # Test composite service
        app_context = container.get("app_context")
        assert app_context.is_success

        context = app_context.value
        assert "database" in context
        assert "email" in context
        assert "config" in context
        assert context["environment"] == "production"
        assert "user_management" in context["features"]

    def test_service_lifecycle_with_cleanup(self) -> None:
        """Test service lifecycle with proper cleanup patterns."""
        container = FlextContainer()

        # Register services with resources
        db_service = DatabaseService("postgresql://cleanup_test")
        db_service.connect()
        container.register("database", db_service)

        email_service = EmailService()
        container.register("email", email_service)

        # Verify services are working
        assert container.get("database").is_success
        assert container.get("email").is_success
        assert container.get_service_count() == 2

        # Simulate application shutdown - clear all services
        cleanup_result = container.clear()
        assert cleanup_result.is_success
        assert container.get_service_count() == 0

        # Services should no longer be available
        assert container.get("database").is_failure
        assert container.get("email").is_failure
