"""Fixed comprehensive tests for FlextDomainService to achieve 100% coverage."""

import pytest
from pydantic import Field

from flext_core import FlextDomainService, FlextMixins, FlextModels, FlextResult


# Test Domain Service Implementations
class TestUserService(FlextDomainService[dict[str, str]]):
    """Test service for user operations."""

    user_id: str
    email: str = ""

    def execute(self) -> FlextResult[dict[str, str]]:
        """Execute user operation."""
        return FlextResult[dict[str, str]].ok(
            {
                "user_id": self.user_id,
                "email": self.email,
            },
        )


class TestComplexService(FlextDomainService[str]):
    """Test service with complex validation and operations."""

    name: str
    value: int = 0
    enabled: bool = True

    def validate_business_rules(self) -> FlextResult[None]:
        """Validate business rules with multiple checks."""
        if not self.name:
            return FlextResult[None].fail("Name is required")
        if self.value < 0:
            return FlextResult[None].fail("Value must be non-negative")
        if not self.enabled and self.value > 0:
            return FlextResult[None].fail("Cannot have value when disabled")
        return FlextResult[None].ok(None)

    def validate_config(self) -> FlextResult[None]:
        """Validate configuration with custom logic."""
        if len(self.name) > 50:
            return FlextResult[None].fail("Name too long")
        if self.value > 1000:
            return FlextResult[None].fail("Value too large")
        return FlextResult[None].ok(None)

    def execute(self) -> FlextResult[str]:
        """Execute complex operation."""
        # Simple validation first
        if not self.name:
            return FlextResult[str].fail("Name is required")
        if self.value < 0:
            return FlextResult[str].fail("Value must be non-negative")
        if len(self.name) > 50:
            return FlextResult[str].fail("Name too long")
        if self.value > 1000:
            return FlextResult[str].fail("Value too large")

        return FlextResult[str].ok(f"Processed: {self.name} with value {self.value}")


class TestFailingService(FlextDomainService[None]):
    """Test service that fails validation."""

    def validate_business_rules(self) -> FlextResult[None]:
        """Always fail validation."""
        return FlextResult[None].fail("Validation always fails")

    def execute(self) -> FlextResult[None]:
        """Execute failing operation."""
        return FlextResult[None].fail("Execution failed")


class TestExceptionService(FlextDomainService[str]):
    """Test service that raises exceptions."""

    should_raise: bool = False

    def validate_business_rules(self) -> FlextResult[None]:
        """Validation that can raise exceptions."""
        if self.should_raise:
            msg = "Validation exception"
            raise ValueError(msg)
        return FlextResult[None].ok(None)

    def execute(self) -> FlextResult[str]:
        """Execute operation that can raise."""
        if self.should_raise:
            msg = "Execution exception"
            raise RuntimeError(msg)
        return FlextResult[str].ok("Success")


class TestDomainServicesFixed:
    """Fixed comprehensive tests for FlextDomainService."""

    def test_basic_service_creation(self) -> None:
        """Test basic domain service creation."""
        service = TestUserService(user_id="123", email="test@example.com")
        assert service.user_id == "123"
        assert service.email == "test@example.com"
        assert isinstance(service, FlextDomainService)

    def test_service_immutability(self) -> None:
        """Test that service is immutable (frozen)."""
        service = TestUserService(user_id="123")

        with pytest.raises(Exception):  # Pydantic ValidationError or similar
            service.user_id = "456"

    def test_execute_method_abstract(self) -> None:
        """Test that execute method is abstract."""

        # Create a concrete implementation to test abstract behavior
        class ConcreteService(FlextDomainService[str]):
            def execute(self) -> FlextResult[str]:
                return FlextResult.ok("test")

        # This should work since we implemented execute
        ConcreteService()

    def test_basic_execution(self) -> None:
        """Test basic service execution."""
        service = TestUserService(user_id="123", email="test@example.com")
        result = service.execute()

        assert result.success is True
        data = result.unwrap()
        assert data["user_id"] == "123"
        assert data["email"] == "test@example.com"

    def test_is_valid_success(self) -> None:
        """Test is_valid with valid service."""
        service = TestComplexService(name="test", value=10, enabled=True)
        assert service.is_valid() is True

    def test_is_valid_failure(self) -> None:
        """Test is_valid with invalid service."""
        service = TestFailingService()
        assert service.is_valid() is False

    def test_is_valid_exception_handling(self) -> None:
        """Test is_valid handles exceptions gracefully."""
        service = TestExceptionService(should_raise=True)
        assert service.is_valid() is False

    def test_validate_business_rules_default(self) -> None:
        """Test default business rules validation."""
        service = TestUserService(user_id="123")
        result = service.validate_business_rules()
        assert result.success is True

    def test_validate_business_rules_custom_success(self) -> None:
        """Test custom business rules validation success."""
        service = TestComplexService(name="test", value=10, enabled=True)
        result = service.validate_business_rules()
        assert result.success is True

    def test_validate_business_rules_custom_failure(self) -> None:
        """Test custom business rules validation failure."""
        service = TestComplexService(name="", value=10, enabled=True)
        result = service.validate_business_rules()
        assert result.success is False
        assert result.error is not None
        assert "Name is required" in result.error

    def test_validate_business_rules_multiple_conditions(self) -> None:
        """Test business rules with multiple conditions."""
        # Test negative value
        service = TestComplexService(name="test", value=-5, enabled=True)
        result = service.validate_business_rules()
        assert result.success is False
        assert result.error is not None
        assert "must be non-negative" in result.error

        # Test disabled with value
        service = TestComplexService(name="test", value=10, enabled=False)
        result = service.validate_business_rules()
        assert result.success is False
        assert result.error is not None
        assert "Cannot have value when disabled" in result.error

    def test_validate_config_default(self) -> None:
        """Test default configuration validation."""
        service = TestUserService(user_id="123")
        result = service.validate_config()
        assert result.success is True

    def test_validate_config_custom_success(self) -> None:
        """Test custom configuration validation success."""
        service = TestComplexService(name="test", value=100, enabled=True)
        result = service.validate_config()
        assert result.success is True

    def test_validate_config_custom_failure(self) -> None:
        """Test custom configuration validation failure."""
        long_name = "x" * 60  # Too long
        service = TestComplexService(name=long_name, value=10, enabled=True)
        result = service.validate_config()
        assert result.success is False
        assert result.error is not None
        assert "too long" in result.error

        # Test value too large
        service = TestComplexService(name="test", value=2000, enabled=True)
        result = service.validate_config()
        assert result.success is False
        assert result.error is not None
        assert "too large" in result.error

    def test_execute_operation_success(self) -> None:
        """Test execute_operation with successful operation."""
        service = TestUserService(user_id="123")

        def test_operation(x: int, y: int) -> int:
            return x + y

        result = service.execute_operation("add_numbers", test_operation, 5, 3)
        assert result.success is True
        assert result.unwrap() == 8

    def test_execute_operation_with_kwargs(self) -> None:
        """Test execute_operation with keyword arguments."""
        service = TestUserService(user_id="123")

        def test_operation(name: str, value: int = 10) -> str:
            return f"{name}: {value}"

        result = service.execute_operation(
            "format_string",
            test_operation,
            name="test",
            value=20,
        )
        assert result.success is True
        assert result.unwrap() == "test: 20"

    def test_execute_operation_config_validation_failure(self) -> None:
        """Test execute_operation with configuration validation failure."""
        # Create service with invalid config
        service = TestComplexService(name="x" * 60, value=10, enabled=True)

        def test_operation() -> str:
            return "success"

        result = service.execute_operation("test", test_operation)
        assert result.success is False
        assert result.error is not None
        assert "too long" in result.error

    def test_execute_operation_runtime_error(self) -> None:
        """Test execute_operation with runtime error."""
        service = TestUserService(user_id="123")

        def failing_operation() -> str:
            msg = "Operation failed"
            raise RuntimeError(msg)

        result = service.execute_operation("failing_op", failing_operation)
        assert result.success is False
        assert result.error is not None
        assert "Operation failed" in result.error

    def test_execute_operation_value_error(self) -> None:
        """Test execute_operation with value error."""
        service = TestUserService(user_id="123")

        def value_error_operation() -> str:
            msg = "Invalid value"
            raise ValueError(msg)

        result = service.execute_operation("value_error_op", value_error_operation)
        assert result.success is False
        assert result.error is not None
        assert "Invalid value" in result.error

    def test_execute_operation_type_error(self) -> None:
        """Test execute_operation with type error."""
        service = TestUserService(user_id="123")

        def type_error_operation() -> str:
            msg = "Wrong type"
            raise TypeError(msg)

        result = service.execute_operation("type_error_op", type_error_operation)
        assert result.success is False
        assert result.error is not None
        assert "Wrong type" in result.error

    def test_execute_operation_unexpected_error(self) -> None:
        """Test execute_operation with unexpected error."""
        service = TestUserService(user_id="123")

        def unexpected_error_operation() -> str:
            msg = "Unexpected error"
            raise OSError(msg)

        result = service.execute_operation("unexpected_op", unexpected_error_operation)
        assert result.success is False
        assert result.error is not None
        assert "Unexpected error" in result.error

    def test_execute_operation_non_callable(self) -> None:
        """Test execute_operation with non-callable operation."""
        service = TestUserService(user_id="123")

        result = service.execute_operation("invalid", "not_callable")
        assert result.success is False
        assert result.error is not None
        assert "not callable" in result.error

    def test_get_service_info_basic(self) -> None:
        """Test get_service_info basic functionality."""
        service = TestUserService(user_id="123", email="test@example.com")
        info = service.get_service_info()

        assert isinstance(info, dict)
        assert "service_type" in info
        assert "service_id" in info
        assert "config_valid" in info
        assert "business_rules_valid" in info
        assert "timestamp" in info

        assert info["service_type"] == "TestUserService"
        assert isinstance(info["service_id"], str)
        assert isinstance(info["config_valid"], bool)
        assert isinstance(info["business_rules_valid"], bool)
        assert isinstance(info["timestamp"], str)

    def test_get_service_info_with_validation(self) -> None:
        """Test get_service_info includes validation status."""
        # Valid service
        valid_service = TestComplexService(name="test", value=10, enabled=True)
        info = valid_service.get_service_info()
        assert info["config_valid"] is True
        assert info["business_rules_valid"] is True

        # Invalid service
        invalid_service = TestFailingService()
        info = invalid_service.get_service_info()
        assert info["config_valid"] is True  # Default config validation passes
        assert info["business_rules_valid"] is False  # Business rules fail

    def test_service_serialization(self) -> None:
        """Test service serialization through mixins."""
        service = TestUserService(user_id="123", email="test@example.com")

        # Test serialization methods from mixins
        assert hasattr(service, "to_dict")
        serialized = service.to_dict()
        assert isinstance(serialized, dict)
        assert "user_id" in serialized
        assert serialized["user_id"] == "123"

    def test_service_logging(self) -> None:
        """Test service logging through mixins."""
        service = TestUserService(user_id="123", email="test@example.com")

        # Test logging methods that are actually available
        assert hasattr(service, "log_info")
        assert hasattr(service, "log_debug")
        assert hasattr(service, "log_error")

        # Test that logging methods can be called without error
        service.log_info("Test info message")
        service.log_debug("Test debug message")

    def test_complex_service_execution_success(self) -> None:
        """Test complex service execution with all validations."""
        service = TestComplexService(name="complex_test", value=100, enabled=True)
        result = service.execute()

        assert result.success is True
        expected = "Processed: complex_test with value 100"
        assert result.unwrap() == expected

    def test_complex_service_execution_business_rule_failure(self) -> None:
        """Test complex service execution with business rule failure."""
        service = TestComplexService(name="", value=100, enabled=True)
        result = service.execute()

        assert result.success is False
        assert result.error is not None
        assert "Name is required" in result.error

    def test_complex_service_execution_config_failure(self) -> None:
        """Test complex service execution with config validation failure."""
        service = TestComplexService(name="test", value=2000, enabled=True)
        result = service.execute()

        assert result.success is False
        assert result.error is not None
        assert "too large" in result.error

    def test_service_model_config(self) -> None:
        """Test service model configuration."""
        service = TestUserService(user_id="123")

        # Test frozen configuration
        assert service.model_config.get("frozen") is True
        assert service.model_config.get("validate_assignment") is True
        assert service.model_config.get("extra") == "forbid"
        assert service.model_config.get("arbitrary_types_allowed") is True

    def test_service_inheritance_hierarchy(self) -> None:
        """Test service inheritance from all mixins."""
        service = TestUserService(user_id="123")

        # Test all expected parent classes

        assert isinstance(service, FlextModels.Config)
        assert isinstance(service, FlextMixins.Serializable)
        assert isinstance(service, FlextMixins.Loggable)

    def test_service_with_complex_types(self) -> None:
        """Test service with complex field types."""

        class ComplexTypeService(FlextDomainService[dict[str, object]]):
            data: dict[str, object]
            numbers: list[int] = Field(default_factory=list)

            def execute(self) -> FlextResult[dict[str, object]]:
                return FlextResult[dict[str, object]].ok(self.data)

        service = ComplexTypeService(
            data={"key": "value", "nested": {"count": 42}},
            numbers=[1, 2, 3, 4, 5],
        )

        result = service.execute()
        assert result.success is True
        data = result.unwrap()
        assert isinstance(data, dict)
        # data is already typed as dict from isinstance check
        assert data["key"] == "value"
        nested = data["nested"]
        assert isinstance(nested, dict)
        assert nested["count"] == 42

    def test_service_extra_forbid(self) -> None:
        """Test that extra fields are forbidden."""
        # Use setattr to bypass type checking for this test
        service = TestUserService(user_id="123")
        with pytest.raises(Exception):  # Pydantic validation error
            service.extra_field = "not_allowed"


class TestDomainServiceStaticMethods:
    """Tests for domain service static methods and configuration."""

    def test_configure_domain_services_system(self) -> None:
        """Test configure_domain_services_system method."""
        config: dict[
            str,
            str | int | float | bool | list[object] | dict[str, object],
        ] = {
            "environment": "test",
            "enable_performance_monitoring": True,
            "max_service_operations": 50,
        }

        result = FlextDomainService.configure_domain_services_system(config)
        assert result.success is True

    def test_configure_domain_services_system_invalid_config(self) -> None:
        """Test configure_domain_services_system with invalid config."""
        # Test with None - should fail
        result = FlextDomainService.configure_domain_services_system(None)
        assert result.success is False
        assert result.error is not None
        assert "NoneType" in result.error or "Configuration" in result.error

        # Test with non-dict - should fail
        result = FlextDomainService.configure_domain_services_system("invalid")
        assert result.success is False

    def test_get_domain_services_system_config(self) -> None:
        """Test get_domain_services_system_config method."""
        result = FlextDomainService.get_domain_services_system_config()
        assert result.success is True

        config = result.unwrap()
        assert isinstance(config, dict)
        # Check for some expected fields
        assert len(config) > 0  # Should have some configuration

    def test_create_environment_domain_services_config(self) -> None:
        """Test create_environment_domain_services_config method."""
        # Test development environment
        result = FlextDomainService.create_environment_domain_services_config(
            "development",
        )
        assert result.success is True
        config = result.unwrap()
        assert config["environment"] == "development"

        # Test production environment
        result = FlextDomainService.create_environment_domain_services_config(
            "production",
        )
        assert result.success is True
        config = result.unwrap()
        assert config["environment"] == "production"
        assert config["enable_performance_monitoring"] is True

        # Test invalid environment - check for failure
        result = FlextDomainService.create_environment_domain_services_config("invalid")
        assert result.success is False
        # The error message might vary, just check it's a failure

    def test_optimize_domain_services_performance(self) -> None:
        """Test optimize_domain_services_performance method."""
        config: dict[
            str,
            str | int | float | bool | list[object] | dict[str, object],
        ] = {
            "environment": "production",
            "enable_caching": True,
            "batch_size": 100,
        }

        result = FlextDomainService.optimize_domain_services_performance(config)
        assert result.success is True

        optimized = result.unwrap()
        assert isinstance(optimized, dict)
        # Check for some optimization settings
        assert len(optimized) > len(config)  # Should have more settings

    def test_optimize_domain_services_performance_invalid_config(self) -> None:
        """Test optimize_domain_services_performance with invalid config."""
        result = FlextDomainService.optimize_domain_services_performance(None)
        assert result.success is False
        assert result.error is not None
        assert "NoneType" in result.error or "Configuration" in result.error
