"""Extended comprehensive tests for FlextCore."""

import math
import threading
from enum import StrEnum

from pydantic import Field

from flext_core import FlextResult
from flext_core.container import FlextContainer
from flext_core.core import FlextCore
from flext_core.models import FlextModels


class TestStatus(StrEnum):
    """Test status enum for testing."""

    ACTIVE = "active"
    INACTIVE = "inactive"


class TestFlextValidateServiceName:
    """Test the standalone flext_validate_service_name function."""

    def test_valid_service_name(self) -> None:
        """Test validation with valid service names."""
        result = FlextContainer.flext_validate_service_name("database_service")
        assert result.success
        assert result.unwrap() is None

    def test_invalid_empty_service_name(self) -> None:
        """Test validation with empty service name."""
        result = FlextContainer.flext_validate_service_name("")
        assert result.failure
        assert "non-empty string" in result.error

    def test_invalid_none_service_name(self) -> None:
        """Test validation with None service name."""
        result = FlextContainer.flext_validate_service_name(None)
        assert result.failure
        assert "non-empty string" in result.error

    def test_invalid_non_string_service_name(self) -> None:
        """Test validation with non-string service name."""
        result = FlextContainer.flext_validate_service_name(123)
        assert result.failure
        assert "non-empty string" in result.error

    def test_invalid_whitespace_only_service_name(self) -> None:
        """Test validation with whitespace-only service name."""
        result = FlextContainer.flext_validate_service_name("   ")
        assert result.failure
        assert "cannot be only whitespace" in result.error


class TestFlextCoreExtendedValidation:
    """Extended tests for FlextCore validation methods."""

    def test_validate_email_valid(self) -> None:
        """Test email validation with valid emails."""
        core = FlextCore.get_instance()

        result = core.validate_email("user@example.com")
        assert result.success
        assert result.unwrap() == "user@example.com"

    def test_validate_email_invalid(self) -> None:
        """Test email validation with invalid emails."""
        core = FlextCore.get_instance()

        result = core.validate_email("invalid-email")
        assert result.failure
        assert "email pattern" in result.error

    def test_validate_email_empty(self) -> None:
        """Test email validation with empty string."""
        core = FlextCore.get_instance()

        result = core.validate_email("")
        assert result.failure

    def test_validate_string_field_valid(self) -> None:
        """Test string field validation with valid input."""
        core = FlextCore.get_instance()

        result = core.validate_string_field("test_string", "username")
        assert result.success
        assert result.unwrap() == "test_string"

    def test_validate_string_field_invalid_none(self) -> None:
        """Test string field validation with None input."""
        core = FlextCore.get_instance()

        result = core.validate_string_field(None, "username")
        assert result.failure
        assert "is not a valid string" in result.error

    def test_validate_string_field_invalid_number(self) -> None:
        """Test string field validation with numeric input."""
        core = FlextCore.get_instance()

        result = core.validate_string_field(123, "username")
        assert result.failure
        assert "is not a valid string" in result.error

    def test_validate_numeric_field_valid_int(self) -> None:
        """Test numeric field validation with valid integer."""
        core = FlextCore.get_instance()

        result = core.validate_numeric_field(42, "age")
        assert result.success
        assert result.unwrap() == 42

    def test_validate_numeric_field_valid_float(self) -> None:
        """Test numeric field validation with valid float."""
        core = FlextCore.get_instance()

        result = core.validate_numeric_field(math.pi, "price")
        assert result.success
        assert result.unwrap() == math.pi

    def test_validate_numeric_field_invalid_string(self) -> None:
        """Test numeric field validation with string input."""
        core = FlextCore.get_instance()

        result = core.validate_numeric_field("not-a-number", "age")
        assert result.failure
        assert "not numeric" in result.error

    def test_validate_user_data_valid(self) -> None:
        """Test user data validation with valid data."""
        core = FlextCore.get_instance()

        user_data = {"name": "John Doe", "email": "john@example.com", "age": 30}
        result = core.validate_user_data(user_data)
        assert result.success

    def test_validate_user_data_invalid_missing_fields(self) -> None:
        """Test user data validation with missing required fields."""
        core = FlextCore.get_instance()

        user_data = {"name": "John Doe"}  # Missing email and age
        result = core.validate_user_data(user_data)
        assert result.failure

    def test_validate_api_request_valid(self) -> None:
        """Test API request validation with valid request."""
        core = FlextCore.get_instance()

        request = {
            "action": "get_users",
            "version": "1.0",
            "method": "GET",
            "path": "/api/users",
            "headers": {"Content-Type": "application/json"},
        }
        result = core.validate_api_request(request)
        assert result.success

    def test_validate_api_request_invalid_missing_method(self) -> None:
        """Test API request validation with missing method."""
        core = FlextCore.get_instance()

        request = {"path": "/api/users"}  # Missing method
        result = core.validate_api_request(request)
        assert result.failure


class TestFlextCoreEntityCreation:
    """Extended tests for FlextCore entity creation methods."""

    def test_create_entity_success(self) -> None:
        """Test successful entity creation."""
        core = FlextCore.get_instance()

        # Use a simple model class from FlextModels
        class TestUser(FlextModels.Entity):
            name: str = Field(..., description="User name")
            email: str = Field(..., description="User email")

            def validate_business_rules(self) -> FlextResult[None]:
                """Validate business rules for the user."""
                return FlextResult[None].ok(None)

        result = core.create_entity(
            TestUser,
            id="user-123",
            name="John Doe",
            email="john@example.com",
        )
        assert result.success
        user = result.unwrap()
        assert user.name == "John Doe"
        assert user.email == "john@example.com"

    def test_create_entity_validation_error(self) -> None:
        """Test entity creation with validation error."""
        core = FlextCore.get_instance()

        class TestUser(FlextModels.Entity):
            name: str = Field(..., description="User name")
            email: str = Field(..., description="User email")

            def validate_business_rules(self) -> FlextResult[None]:
                """Validate business rules for the user."""
                return FlextResult[None].ok(None)

        # Missing required field 'name'
        result = core.create_entity(TestUser, email="john@example.com")
        assert result.failure

    def test_create_value_object_success(self) -> None:
        """Test successful value object creation."""
        core = FlextCore.get_instance()

        class TestEmail(FlextModels.Value):
            address: str = Field(..., description="Email address")

            def validate_business_rules(self) -> FlextResult[None]:
                """Validate business rules for the email."""
                return FlextResult[None].ok(None)

        result = core.create_value_object(TestEmail, address="test@example.com")
        assert result.success
        email = result.unwrap()
        assert email.address == "test@example.com"

    def test_create_domain_event_success(self) -> None:
        """Test successful domain event creation."""
        core = FlextCore.get_instance()

        class TestEvent(FlextModels.Event):
            event_type: str = Field(..., description="Event type")
            data: dict = Field(default_factory=dict, description="Event data")

        result = core.create_domain_event(
            event_type="UserCreated",
            aggregate_id="123",
            aggregate_type="User",
            data={"user_id": "123", "name": "John"},
            source_service="test_service",
        )
        assert result.success
        event = result.unwrap()
        assert event.event_type == "UserCreated"
        assert event.data["user_id"] == "123"

    def test_create_payload_success(self) -> None:
        """Test successful payload creation."""
        core = FlextCore.get_instance()

        result = core.create_payload(
            data={"user_id": "123", "name": "John"},
            message_type="UserCreated",
            source_service="test_service",
        )
        assert result.success
        payload = result.unwrap()
        assert "user_id" in payload.data
        assert payload.data["user_id"] == "123"


class TestFlextCoreUtilities:
    """Extended tests for FlextCore utility methods."""

    def test_generate_uuid(self) -> None:
        """Test UUID generation."""
        core = FlextCore.get_instance()

        uuid1 = core.generate_uuid()
        uuid2 = core.generate_uuid()

        assert uuid1 != uuid2
        assert len(str(uuid1)) == 36  # Standard UUID format
        assert "-" in str(uuid1)

    def test_generate_correlation_id(self) -> None:
        """Test correlation ID generation."""
        core = FlextCore.get_instance()

        id1 = core.generate_correlation_id()
        id2 = core.generate_correlation_id()

        assert id1 != id2
        assert isinstance(id1, str)
        assert len(id1) > 0

    def test_generate_entity_id(self) -> None:
        """Test entity ID generation."""
        core = FlextCore.get_instance()

        id1 = core.generate_entity_id()
        id2 = core.generate_entity_id()

        assert id1 != id2
        assert isinstance(id1, str)
        assert len(id1) > 0

    def test_format_duration_seconds(self) -> None:
        """Test duration formatting in seconds."""
        core = FlextCore.get_instance()

        result = core.format_duration(45.5)
        assert "45.5" in result or "45" in result
        assert "s" in result  # Should contain seconds abbreviation

    def test_format_duration_minutes(self) -> None:
        """Test duration formatting in minutes."""
        core = FlextCore.get_instance()

        result = core.format_duration(125.0)  # > 60 seconds
        assert "m" in result or "s" in result  # Should contain time unit abbreviation

    def test_clean_text_basic(self) -> None:
        """Test text cleaning with basic input."""
        core = FlextCore.get_instance()

        result = core.clean_text("  Hello World  ")
        assert result.strip() == "Hello World"

    def test_clean_text_special_characters(self) -> None:
        """Test text cleaning with special characters."""
        core = FlextCore.get_instance()

        result = core.clean_text("Hello\nWorld\t!")
        assert result is not None
        assert isinstance(result, str)

    def test_batch_process_small_list(self) -> None:
        """Test batch processing with small list."""
        core = FlextCore.get_instance()

        items = [1, 2, 3, 4, 5]
        batches = core.batch_process(items, batch_size=2)

        assert len(batches) == 3
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]
        assert batches[2] == [5]

    def test_batch_process_exact_size(self) -> None:
        """Test batch processing with exact batch size."""
        core = FlextCore.get_instance()

        items = [1, 2, 3, 4]
        batches = core.batch_process(items, batch_size=2)

        assert len(batches) == 2
        assert batches[0] == [1, 2]
        assert batches[1] == [3, 4]

    def test_batch_process_empty_list(self) -> None:
        """Test batch processing with empty list."""
        core = FlextCore.get_instance()

        items = []
        batches = core.batch_process(items, batch_size=2)

        assert len(batches) == 0

    def test_batch_process_default_size(self) -> None:
        """Test batch processing with default batch size."""
        core = FlextCore.get_instance()

        items = list(range(250))  # More than default batch size of 100
        batches = core.batch_process(items)

        assert len(batches) == 3  # 100 + 100 + 50
        assert len(batches[0]) == 100
        assert len(batches[1]) == 100
        assert len(batches[2]) == 50


class TestFlextCoreErrorCreation:
    """Extended tests for FlextCore error creation methods."""

    def test_create_validation_error_with_details(self) -> None:
        """Test validation error creation with details."""
        core = FlextCore.get_instance()

        error = core.create_validation_error(
            "Email validation failed",
            field="email",
            value="invalid-email",
            details={"reason": "Missing @ symbol"},
        )

        assert "Email validation failed" in str(error)
        assert hasattr(error, "field") or "email" in str(error)

    def test_create_validation_error_minimal(self) -> None:
        """Test validation error creation with minimal parameters."""
        core = FlextCore.get_instance()

        error = core.create_validation_error("Simple validation error")
        assert "Simple validation error" in str(error)

    def test_create_configuration_error_with_config(self) -> None:
        """Test configuration error creation with config details."""
        core = FlextCore.get_instance()

        error = core.create_configuration_error(
            "Database config invalid",
            config_key="database.host",
            config_value="",
            expected_type=str,
        )

        assert "Database config invalid" in str(error)

    def test_create_connection_error_with_details(self) -> None:
        """Test connection error creation with details."""
        core = FlextCore.get_instance()

        error = core.create_connection_error(
            "Failed to connect to database",
            host="localhost",
            port=5432,
            retry_count=3,
        )

        assert "Failed to connect to database" in str(error)


class TestFlextCoreTypeGuards:
    """Extended tests for FlextCore type guard methods."""

    def test_is_string_with_string(self) -> None:
        """Test string type guard with string input."""
        core = FlextCore.get_instance()

        result = core.is_string("hello world")
        assert result is True

    def test_is_string_with_non_string(self) -> None:
        """Test string type guard with non-string input."""
        core = FlextCore.get_instance()

        assert core.is_string(123) is False
        assert core.is_string([]) is False
        assert core.is_string({}) is False
        assert core.is_string(None) is False

    def test_is_dict_with_dict(self) -> None:
        """Test dict type guard with dict input."""
        core = FlextCore.get_instance()

        result = core.is_dict({"key": "value"})
        assert result is True

    def test_is_dict_with_non_dict(self) -> None:
        """Test dict type guard with non-dict input."""
        core = FlextCore.get_instance()

        assert core.is_dict("string") is False
        assert core.is_dict([]) is False
        assert core.is_dict(123) is False
        assert core.is_dict(None) is False

    def test_is_list_with_list(self) -> None:
        """Test list type guard with list input."""
        core = FlextCore.get_instance()

        result = core.is_list([1, 2, 3])
        assert result is True

    def test_is_list_with_non_list(self) -> None:
        """Test list type guard with non-list input."""
        core = FlextCore.get_instance()

        assert core.is_list("string") is False
        assert core.is_list({}) is False
        assert core.is_list(123) is False
        assert core.is_list(None) is False


class TestFlextCoreLogging:
    """Extended tests for FlextCore logging methods."""

    def test_log_info_basic(self) -> None:
        """Test basic info logging."""
        core = FlextCore.get_instance()

        # Should not raise any exceptions
        core.log_info("Test info message")

    def test_log_info_with_extra(self) -> None:
        """Test info logging with extra parameters."""
        core = FlextCore.get_instance()

        # Should not raise any exceptions
        core.log_info("User action", user_id="123", action="login")

    def test_log_error_basic(self) -> None:
        """Test basic error logging."""
        core = FlextCore.get_instance()

        # Should not raise any exceptions
        core.log_error("Test error message")

    def test_log_error_with_exception(self) -> None:
        """Test error logging with exception."""
        core = FlextCore.get_instance()

        def _raise_error() -> None:
            msg = "Test exception"
            raise ValueError(msg)

        try:
            _raise_error()
        except ValueError as e:
            # Should not raise any exceptions
            core.log_error("Error occurred", exception=str(e))

    def test_log_warning_basic(self) -> None:
        """Test basic warning logging."""
        core = FlextCore.get_instance()

        # Should not raise any exceptions
        core.log_warning("Test warning message")

    def test_log_warning_with_context(self) -> None:
        """Test warning logging with context."""
        core = FlextCore.get_instance()

        # Should not raise any exceptions
        core.log_warning("Deprecated feature used", feature="old_api", version="2.0")


class TestFlextCoreServiceManagement:
    """Extended tests for FlextCore service management methods."""

    def test_register_and_get_service_success(self) -> None:
        """Test successful service registration and retrieval."""
        core = FlextCore.get_instance()

        # Register a simple service
        class TestService:
            def __init__(self) -> None:
                self.name = "test_service"

        service = TestService()
        register_result = core.register_service("test_service", service)
        assert register_result.success

        # Retrieve the service
        get_result = core.get_service("test_service")
        assert get_result.success
        retrieved_service = get_result.unwrap()
        assert retrieved_service.name == "test_service"

    def test_get_service_not_found(self) -> None:
        """Test getting non-existent service."""
        core = FlextCore.get_instance()

        result = core.get_service("nonexistent_service")
        assert result.failure
        assert "not found" in result.error or "not registered" in result.error

    def test_register_service_invalid_name(self) -> None:
        """Test service registration with invalid name."""
        core = FlextCore.get_instance()

        service = object()
        result = core.register_service("", service)
        assert result.failure


class TestFlextCoreSystemConfiguration:
    """Extended tests for FlextCore system configuration methods."""

    def test_get_aggregates_config(self) -> None:
        """Test getting aggregates configuration."""
        core = FlextCore.get_instance()

        config_result = core.get_aggregates_config()
        assert config_result.success
        config = config_result.unwrap()
        assert config is not None
        assert isinstance(config, dict)

    def test_get_commands_config(self) -> None:
        """Test getting commands configuration."""
        core = FlextCore.get_instance()

        config_result = core.get_commands_config()
        assert config_result.success
        config = config_result.unwrap()
        assert config is not None
        assert isinstance(config, dict)

    def test_get_context_config(self) -> None:
        """Test getting context configuration."""
        core = FlextCore.get_instance()

        result = core.get_context_config()
        assert result.success
        config = result.unwrap()
        assert isinstance(config, dict)

    def test_get_decorators_config(self) -> None:
        """Test getting decorators configuration."""
        core = FlextCore.get_instance()

        result = core.get_decorators_config()
        assert result.success
        config = result.unwrap()
        assert isinstance(config, dict)

    def test_load_config_from_file_nonexistent(self) -> None:
        """Test loading config from nonexistent file."""
        core = FlextCore.get_instance()

        result = core.load_config_from_file("nonexistent_config.json")
        assert result.failure


class TestFlextCoreFieldOperations:
    """Extended tests for FlextCore field operations."""

    def test_create_boolean_field(self) -> None:
        """Test creating boolean field."""
        core = FlextCore.get_instance()

        field_result = core.fields.create_boolean_field(name="is_active", default=True)
        assert field_result.success
        field = field_result.unwrap()
        assert field is not None

    def test_validate_field_success(self) -> None:
        """Test successful field validation."""
        core = FlextCore.get_instance()

        result = core.validate_field("test_value", lambda x: isinstance(x, str))
        assert result.success
        assert result.unwrap() == "test_value"

    def test_validate_field_failure(self) -> None:
        """Test failed field validation."""
        core = FlextCore.get_instance()

        result = core.validate_field("test_value", lambda x: isinstance(x, int))
        assert result.failure


class TestFlextCoreAdvancedFeatures:
    """Extended tests for FlextCore advanced features and edge cases."""

    def test_singleton_pattern(self) -> None:
        """Test that FlextCore maintains singleton pattern."""
        core1 = FlextCore.get_instance()
        core2 = FlextCore.get_instance()

        assert core1 is core2

    def test_property_accessors(self) -> None:
        """Test that property accessors work correctly."""
        core = FlextCore.get_instance()

        # Test that properties return appropriate objects
        assert core.container is not None
        assert core.config is not None
        assert core.context is not None
        assert core.logger is not None
        # observability was removed from FlextCore

    def test_multiple_service_registration(self) -> None:
        """Test registering multiple services."""
        core = FlextCore.get_instance()

        services = {f"service_{i}": f"value_{i}" for i in range(5)}

        for name, service in services.items():
            result = core.register_service(name, service)
            assert result.success

        # Verify all services can be retrieved
        for name in services:
            result = core.get_service(name)
            assert result.success

    def test_concurrent_service_access(self) -> None:
        """Test concurrent service access (basic thread safety)."""
        core = FlextCore.get_instance()
        results = []

        def access_service() -> None:
            result = core.get_service("nonexistent_service_concurrent")
            results.append(result.failure)

        threads = [threading.Thread(target=access_service) for _ in range(10)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # All should fail consistently
        assert all(results)
        assert len(results) == 10

    def test_edge_case_validations(self) -> None:
        """Test edge cases in validation methods."""
        core = FlextCore.get_instance()

        # Test with various edge case inputs
        edge_cases = [None, "", 0, [], {}, False]

        for case in edge_cases:
            email_result = (
                core.validate_email(case)
                if isinstance(case, str)
                else FlextResult.fail("not string")
            )
            # Should handle gracefully
            assert email_result is not None

    def test_system_configurations_integration(self) -> None:
        """Test integration between different system configurations."""
        core = FlextCore.get_instance()

        # Test that multiple config systems can be accessed without conflicts
        aggregates_config = core.get_aggregates_config()
        commands_config = core.get_commands_config()
        context_result = core.get_context_config()
        decorators_result = core.get_decorators_config()

        assert aggregates_config is not None
        assert commands_config is not None
        assert context_result.success
        assert decorators_result.success

    def test_error_handling_consistency(self) -> None:
        """Test that error handling is consistent across methods."""
        core = FlextCore.get_instance()

        # Test various methods that should fail gracefully
        validation_error = core.create_validation_error("test")
        config_error = core.create_configuration_error("test")
        connection_error = core.create_connection_error("test")

        # All should create valid error objects
        assert validation_error is not None
        assert config_error is not None
        assert connection_error is not None

    def test_utility_methods_robustness(self) -> None:
        """Test utility methods with various inputs."""
        core = FlextCore.get_instance()

        # Test format_duration with edge cases
        assert core.format_duration(0) is not None
        assert core.format_duration(0.5) is not None
        assert core.format_duration(1000000) is not None

        # Test clean_text with edge cases
        assert core.clean_text("") is not None
        assert core.clean_text("   ") is not None

        # Test batch_process with edge cases
        assert core.batch_process([], 1) == []
        assert len(core.batch_process([1], 10)) == 1


class TestFlextCoreConfigurationProperties:
    """Test FlextCore configuration properties for database, security, and logging."""

    def test_database_config_none_when_empty(self) -> None:
        """Test database_config returns None when no config is set."""
        core = FlextCore.get_instance()
        
        # Clear any existing database config
        core._specialized_configs.pop("database_config", None)
        
        config = core.database_config
        assert config is None

    def test_database_config_returns_valid_config(self) -> None:
        """Test database_config returns valid config when set."""
        core = FlextCore.get_instance()
        
        # Create a valid database config
        db_config = FlextModels.DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
        )
        
        # Store in specialized configs
        core._specialized_configs["database_config"] = db_config
        
        # Test property
        result = core.database_config
        assert result is not None
        assert isinstance(result, FlextModels.DatabaseConfig)
        assert result.host == "localhost"
        assert result.port == 5432

    def test_database_config_none_for_wrong_type(self) -> None:
        """Test database_config returns None when wrong type is stored."""
        core = FlextCore.get_instance()
        
        # Store wrong type
        core._specialized_configs["database_config"] = "not a config object"
        
        config = core.database_config
        assert config is None

    def test_security_config_none_when_empty(self) -> None:
        """Test security_config returns None when no config is set."""
        core = FlextCore.get_instance()
        
        # Clear any existing security config
        core._specialized_configs.pop("security_config", None)
        
        config = core.security_config
        assert config is None

    def test_security_config_returns_valid_config(self) -> None:
        """Test security_config returns valid config when set."""
        core = FlextCore.get_instance()
        
        # Create a valid security config
        security_config = FlextModels.SecurityConfig(
            secret_key="Test_Secret_Key_12345678901234567890",
            jwt_secret="JWT_Secret_Key_12345678901234567890",
            encryption_key="Encryption_Key_12345678901234567890",
        )
        
        # Store in specialized configs
        core._specialized_configs["security_config"] = security_config
        
        # Test property
        result = core.security_config
        assert result is not None
        assert isinstance(result, FlextModels.SecurityConfig)
        assert result.secret_key == "Test_Secret_Key_12345678901234567890"
        assert result.session_timeout == 3600  # default

    def test_security_config_none_for_wrong_type(self) -> None:
        """Test security_config returns None when wrong type is stored."""
        core = FlextCore.get_instance()
        
        # Store wrong type
        core._specialized_configs["security_config"] = {"not": "a config object"}
        
        config = core.security_config
        assert config is None

    def test_logging_config_none_when_empty(self) -> None:
        """Test logging_config returns None when no config is set."""
        core = FlextCore.get_instance()
        
        # Clear any existing logging config
        core._specialized_configs.pop("logging_config", None)
        
        config = core.logging_config
        assert config is None

    def test_logging_config_returns_valid_config(self) -> None:
        """Test logging_config returns valid config when set."""
        core = FlextCore.get_instance()
        
        # Create a valid logging config
        logging_config = FlextModels.LoggingConfig(
            log_level="DEBUG",
            log_format="text",
            log_file="/tmp/test.log",
        )
        
        # Store in specialized configs
        core._specialized_configs["logging_config"] = logging_config
        
        # Test property
        result = core.logging_config
        assert result is not None
        assert isinstance(result, FlextModels.LoggingConfig)
        assert result.log_level == "DEBUG"
        assert result.log_format == "text"
        assert result.log_file == "/tmp/test.log"

    def test_logging_config_none_for_wrong_type(self) -> None:
        """Test logging_config returns None when wrong type is stored."""
        core = FlextCore.get_instance()
        
        # Store wrong type
        core._specialized_configs["logging_config"] = [1, 2, 3]
        
        config = core.logging_config
        assert config is None

    def test_all_config_properties_independent(self) -> None:
        """Test that all config properties work independently."""
        core = FlextCore.get_instance()
        
        # Create different configs
        db_config = FlextModels.DatabaseConfig(
            host="db.example.com",
            port=3306,
            database="prod_db",
            username="prod_user",
            password="prod_pass",
        )
        
        security_config = FlextModels.SecurityConfig(
            secret_key="Prod_Secret_Key_123456789012345678901234567890",
            jwt_secret="Prod_JWT_Secret_123456789012345678901234567890",
            encryption_key="Prod_Encryption_Key_123456789012345678901234567890",
        )
        
        logging_config = FlextModels.LoggingConfig(
            log_level="WARNING",
            log_format="json",
            max_file_size=20971520,  # 20MB
        )
        
        # Store all configs
        core._specialized_configs["database_config"] = db_config
        core._specialized_configs["security_config"] = security_config
        core._specialized_configs["logging_config"] = logging_config
        
        # Test all properties work independently
        db_result = core.database_config
        security_result = core.security_config
        logging_result = core.logging_config
        
        assert db_result is not None
        assert db_result.host == "db.example.com"
        
        assert security_result is not None
        assert security_result.secret_key == "Prod_Secret_Key_123456789012345678901234567890"
        
        assert logging_result is not None
        assert logging_result.log_level == "WARNING"
        
        # Test they don't interfere with each other
        assert db_result is not security_result
        assert security_result is not logging_result
        assert logging_result is not db_result
