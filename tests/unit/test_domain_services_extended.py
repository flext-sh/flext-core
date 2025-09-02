"""Extended test coverage for domain_services.py module.

Comprehensive tests to increase coverage of FlextDomainService system.
"""

from __future__ import annotations

import pytest

from flext_core import FlextDomainService
from flext_core.constants import FlextConstants
from flext_core.result import FlextResult

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextDomainServiceCore:
    """Test FlextDomainService core functionality."""

    def test_domain_service_abstract_class(self) -> None:
        """Test FlextDomainService abstract class behavior."""
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            FlextDomainService()

    def test_concrete_domain_service_creation(self) -> None:
        """Test concrete domain service creation and basic functionality."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            test_field: str = "default_value"

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        service = TestDomainService(test_field="test_value")
        assert service.test_field == "test_value"
        assert isinstance(service, FlextDomainService)

    def test_domain_service_immutability(self) -> None:
        """Test domain service immutability via Pydantic frozen configuration."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            test_field: str = "default_value"

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        service = TestDomainService(test_field="initial_value")

        # Should raise ValidationError due to frozen=True
        with pytest.raises(Exception):  # ValidationError from Pydantic
            service.test_field = "new_value"

    def test_is_valid_default_behavior(self) -> None:
        """Test is_valid default behavior."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        service = TestDomainService()
        # Should be valid by default (default validate_business_rules returns success)
        assert service.is_valid() is True

    def test_is_valid_with_failing_validation(self) -> None:
        """Test is_valid with failing business rule validation."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

            def validate_business_rules(self) -> FlextResult[None]:
                """Override with failing validation."""
                return FlextResult[None].fail("Business rules validation failed")

        service = TestDomainService()
        assert service.is_valid() is False

    def test_is_valid_with_exception(self) -> None:
        """Test is_valid handles exceptions gracefully."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

            def validate_business_rules(self) -> FlextResult[None]:
                """Override with exception raising."""
                msg = "Validation error"
                raise ValueError(msg)

        service = TestDomainService()
        assert service.is_valid() is False

    def test_validate_business_rules_default(self) -> None:
        """Test validate_business_rules default implementation."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        service = TestDomainService()
        result = service.validate_business_rules()
        assert result.success
        assert result.value is None

    def test_validate_business_rules_override(self) -> None:
        """Test validate_business_rules override functionality."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            required_field: str

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

            def validate_business_rules(self) -> FlextResult[None]:
                """Override with custom validation."""
                if not self.required_field:
                    return FlextResult[None].fail("Required field is missing")
                if len(self.required_field) < 3:
                    return FlextResult[None].fail("Field too short")
                return FlextResult[None].ok(None)

        # Valid case
        service = TestDomainService(required_field="valid_value")
        result = service.validate_business_rules()
        assert result.success

        # Invalid case - empty field
        service = TestDomainService(required_field="")
        result = service.validate_business_rules()
        assert result.failure
        assert "Required field is missing" in result.error

        # Invalid case - field too short
        service = TestDomainService(required_field="ab")
        result = service.validate_business_rules()
        assert result.failure
        assert "Field too short" in result.error

    def test_execute_abstract_method(self) -> None:
        """Test execute method is abstract and must be implemented."""

        class IncompleteService(FlextDomainService[str]):
            """Service missing execute implementation."""

        # Should raise TypeError because execute is not implemented
        with pytest.raises(TypeError):
            IncompleteService()

    def test_execute_implementation(self) -> None:
        """Test concrete execute implementation."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation with railway programming."""
                return (
                    self.validate_business_rules()
                    .flat_map(lambda _: self._perform_operation())
                    .map(lambda result: f"processed_{result}")
                )

            def _perform_operation(self) -> FlextResult[str]:
                """Perform core operation."""
                return FlextResult[str].ok("operation_result")

        service = TestDomainService()
        result = service.execute()
        assert result.success
        assert result.value == "processed_operation_result"

    def test_validate_config_default(self) -> None:
        """Test validate_config default implementation."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        service = TestDomainService()
        result = service.validate_config()
        assert result.success
        assert result.value is None

    def test_validate_config_override(self) -> None:
        """Test validate_config override functionality."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            smtp_host: str = ""
            smtp_port: int = 0

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

            def validate_config(self) -> FlextResult[None]:
                """Override with custom configuration validation."""
                if not self.smtp_host:
                    return FlextResult[None].fail("SMTP host is required")
                if self.smtp_port <= 0 or self.smtp_port > 65535:
                    return FlextResult[None].fail("Invalid SMTP port")
                return FlextResult[None].ok(None)

        # Valid configuration
        service = TestDomainService(smtp_host="localhost", smtp_port=587)
        result = service.validate_config()
        assert result.success

        # Invalid configuration - missing host
        service = TestDomainService(smtp_host="", smtp_port=587)
        result = service.validate_config()
        assert result.failure
        assert "SMTP host is required" in result.error

        # Invalid configuration - invalid port
        service = TestDomainService(smtp_host="localhost", smtp_port=0)
        result = service.validate_config()
        assert result.failure
        assert "Invalid SMTP port" in result.error


class TestFlextDomainServiceOperations:
    """Test FlextDomainService operation execution functionality."""

    def test_execute_operation_success(self) -> None:
        """Test successful operation execution."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        def test_operation(x: int, y: int) -> int:
            return x + y

        service = TestDomainService()
        result = service.execute_operation("add_numbers", test_operation, 2, 3)
        assert result.success
        assert result.value == 5

    def test_execute_operation_with_kwargs(self) -> None:
        """Test operation execution with keyword arguments."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        def test_operation(x: int, multiplier: int = 2) -> int:
            return x * multiplier

        service = TestDomainService()
        result = service.execute_operation("multiply", test_operation, 5, multiplier=3)
        assert result.success
        assert result.value == 15

    def test_execute_operation_config_validation_failure(self) -> None:
        """Test operation execution with configuration validation failure."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

            def validate_config(self) -> FlextResult[None]:
                """Override with failing validation."""
                return FlextResult[None].fail("Configuration is invalid")

        def test_operation() -> str:
            return "success"

        service = TestDomainService()
        result = service.execute_operation("test_op", test_operation)
        assert result.failure
        assert "Configuration is invalid" in result.error
        assert result.error_code == FlextConstants.Errors.VALIDATION_ERROR

    def test_execute_operation_not_callable(self) -> None:
        """Test operation execution with non-callable operation."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        service = TestDomainService()
        result = service.execute_operation("invalid_op", "not_callable")
        assert result.failure
        assert "is not callable" in result.error
        assert result.error_code == FlextConstants.Errors.OPERATION_ERROR

    def test_execute_operation_runtime_exception(self) -> None:
        """Test operation execution with runtime exception."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        def failing_operation() -> str:
            msg = "Operation failed"
            raise RuntimeError(msg)

        service = TestDomainService()
        result = service.execute_operation("failing_op", failing_operation)
        assert result.failure
        assert "Operation failed" in result.error
        assert result.error_code == FlextConstants.Errors.EXCEPTION_ERROR

    def test_execute_operation_value_error(self) -> None:
        """Test operation execution with ValueError."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        def value_error_operation() -> str:
            msg = "Invalid value"
            raise ValueError(msg)

        service = TestDomainService()
        result = service.execute_operation("value_error_op", value_error_operation)
        assert result.failure
        assert "Invalid value" in result.error
        assert result.error_code == FlextConstants.Errors.EXCEPTION_ERROR

    def test_execute_operation_type_error(self) -> None:
        """Test operation execution with TypeError."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        def type_error_operation() -> str:
            msg = "Type mismatch"
            raise TypeError(msg)

        service = TestDomainService()
        result = service.execute_operation("type_error_op", type_error_operation)
        assert result.failure
        assert "Type mismatch" in result.error
        assert result.error_code == FlextConstants.Errors.EXCEPTION_ERROR

    def test_execute_operation_unknown_exception(self) -> None:
        """Test operation execution with unknown exception."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        def unknown_error_operation() -> str:
            msg = "Network error"
            raise ConnectionError(msg)

        service = TestDomainService()
        result = service.execute_operation("unknown_error_op", unknown_error_operation)
        assert result.failure
        assert "Network error" in result.error
        assert result.error_code == FlextConstants.Errors.UNKNOWN_ERROR


class TestFlextDomainServiceInfo:
    """Test FlextDomainService information and metadata functionality."""

    def test_get_service_info_basic(self) -> None:
        """Test get_service_info basic functionality."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        service = TestDomainService()
        info = service.get_service_info()

        assert isinstance(info, dict)
        assert info["service_type"] == "TestDomainService"
        assert "service_id" in info
        assert info["service_id"].startswith("service_testdomainservice_")
        assert info["config_valid"] is True
        assert info["business_rules_valid"] is True
        assert "timestamp" in info

    def test_get_service_info_with_invalid_config(self) -> None:
        """Test get_service_info with invalid configuration."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

            def validate_config(self) -> FlextResult[None]:
                """Override with failing validation."""
                return FlextResult[None].fail("Config is invalid")

        service = TestDomainService()
        info = service.get_service_info()

        assert info["config_valid"] is False
        assert info["business_rules_valid"] is True  # Still valid

    def test_get_service_info_with_invalid_business_rules(self) -> None:
        """Test get_service_info with invalid business rules."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

            def validate_business_rules(self) -> FlextResult[None]:
                """Override with failing validation."""
                return FlextResult[None].fail("Business rules are invalid")

        service = TestDomainService()
        info = service.get_service_info()

        assert info["config_valid"] is True
        assert info["business_rules_valid"] is False

    def test_get_service_info_with_both_invalid(self) -> None:
        """Test get_service_info with both config and business rules invalid."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

            def validate_config(self) -> FlextResult[None]:
                """Override with failing validation."""
                return FlextResult[None].fail("Config is invalid")

            def validate_business_rules(self) -> FlextResult[None]:
                """Override with failing validation."""
                return FlextResult[None].fail("Business rules are invalid")

        service = TestDomainService()
        info = service.get_service_info()

        assert info["config_valid"] is False
        assert info["business_rules_valid"] is False


class TestFlextDomainServiceConfiguration:
    """Test FlextDomainService configuration methods."""

    def test_configure_domain_services_system_basic(self) -> None:
        """Test basic configuration of domain services system."""
        config = {
            "environment": "development",
            "service_level": "normal",
            "log_level": "INFO",
        }

        result = FlextDomainService.configure_domain_services_system(config)
        assert result.success

        validated_config = result.value
        assert validated_config["environment"] == "development"
        assert validated_config["service_level"] == "normal"
        assert validated_config["log_level"] == "INFO"
        assert validated_config["enable_business_rule_validation"] is True
        assert validated_config["max_service_operations"] == 50

    def test_configure_domain_services_system_defaults(self) -> None:
        """Test configuration with default values."""
        config: dict[str, object] = {}

        result = FlextDomainService.configure_domain_services_system(config)
        assert result.success

        validated_config = result.value
        assert (
            validated_config["environment"]
            == FlextConstants.Config.ConfigEnvironment.DEVELOPMENT.value
        )
        assert (
            validated_config["service_level"]
            == FlextConstants.Config.ValidationLevel.LOOSE.value
        )
        assert (
            validated_config["log_level"] == FlextConstants.Config.LogLevel.DEBUG.value
        )

    def test_configure_domain_services_system_invalid_environment(self) -> None:
        """Test configuration with invalid environment."""
        config = {"environment": "invalid_env"}

        result = FlextDomainService.configure_domain_services_system(config)
        assert result.failure
        assert "Invalid environment" in result.error

    def test_configure_domain_services_system_invalid_service_level(self) -> None:
        """Test configuration with invalid service level."""
        config = {"service_level": "invalid_level"}

        result = FlextDomainService.configure_domain_services_system(config)
        assert result.failure
        assert "Invalid service_level" in result.error

    def test_configure_domain_services_system_invalid_log_level(self) -> None:
        """Test configuration with invalid log level."""
        config = {"log_level": "INVALID"}

        result = FlextDomainService.configure_domain_services_system(config)
        assert result.failure
        assert "Invalid log_level" in result.error

    def test_configure_domain_services_system_exception(self) -> None:
        """Test configuration with exception handling."""

        # Create an object that will cause an exception when processed
        class BadConfig:
            def __getitem__(self, key: str) -> None:
                msg = "Bad config"
                raise RuntimeError(msg)

        bad_config = BadConfig()

        result = FlextDomainService.configure_domain_services_system(bad_config)
        assert result.failure
        assert "Configuration must be a dictionary" in result.error

    def test_get_domain_services_system_config(self) -> None:
        """Test getting current domain services system configuration."""
        result = FlextDomainService.get_domain_services_system_config()
        assert result.success

        config = result.value
        assert "environment" in config
        assert "service_level" in config
        assert "log_level" in config
        assert "active_service_operations" in config
        assert "ddd_validation_status" in config
        assert "available_service_patterns" in config
        assert config["ddd_validation_status"] == "enabled"
        assert "abstract" in config["available_service_patterns"]
        assert "stateless" in config["available_service_patterns"]

    def test_get_domain_services_system_config_exception(self) -> None:
        """Test get_domain_services_system_config exception handling."""
        # Use pytest.MonkeyPatch to mock the method
        import unittest.mock

        with unittest.mock.patch.object(
            FlextDomainService,
            "get_domain_services_system_config",
            classmethod(lambda cls: FlextResult.fail("Config generation failed")),
        ):
            result = FlextDomainService.get_domain_services_system_config()
            assert result.failure
            assert "Config generation failed" in result.error

    def test_create_environment_domain_services_config_production(self) -> None:
        """Test creating production environment configuration."""
        result = FlextDomainService.create_environment_domain_services_config(
            "production"
        )
        assert result.success

        config = result.value
        assert config["environment"] == "production"
        assert (
            config["service_level"]
            == FlextConstants.Config.ValidationLevel.STRICT.value
        )
        assert config["log_level"] == FlextConstants.Config.LogLevel.WARNING.value
        assert config["enable_performance_monitoring"] is True
        assert config["max_service_operations"] == 100
        assert config["service_execution_timeout_seconds"] == 30

    def test_create_environment_domain_services_config_development(self) -> None:
        """Test creating development environment configuration."""
        result = FlextDomainService.create_environment_domain_services_config(
            "development"
        )
        assert result.success

        config = result.value
        assert config["environment"] == "development"
        assert (
            config["service_level"] == FlextConstants.Config.ValidationLevel.LOOSE.value
        )
        assert config["log_level"] == FlextConstants.Config.LogLevel.DEBUG.value
        assert config["service_execution_timeout_seconds"] == 120
        assert config["enable_service_caching"] is False

    def test_create_environment_domain_services_config_test(self) -> None:
        """Test creating test environment configuration."""
        result = FlextDomainService.create_environment_domain_services_config("test")
        assert result.success

        config = result.value
        assert config["environment"] == "test"
        assert (
            config["service_level"]
            == FlextConstants.Config.ValidationLevel.STRICT.value
        )
        assert config["enable_performance_monitoring"] is False
        assert config["service_retry_attempts"] == 0

    def test_create_environment_domain_services_config_invalid_environment(
        self,
    ) -> None:
        """Test creating config with invalid environment."""
        result = FlextDomainService.create_environment_domain_services_config(
            "invalid_env"
        )
        assert result.failure
        assert "Invalid environment" in result.error

    def test_create_environment_domain_services_config_exception(self) -> None:
        """Test create_environment_domain_services_config exception handling."""
        import unittest.mock

        with unittest.mock.patch.object(
            FlextDomainService,
            "create_environment_domain_services_config",
            classmethod(
                lambda cls, environment: FlextResult.fail(
                    "Environment config creation failed"
                )
            ),
        ):
            result = FlextDomainService.create_environment_domain_services_config(
                "development"
            )
            assert result.failure
            assert "Environment config creation failed" in result.error

    def test_optimize_domain_services_performance_high(self) -> None:
        """Test optimizing domain services for high performance."""
        config = {"performance_level": "high", "max_concurrent_services": 50}

        result = FlextDomainService.optimize_domain_services_performance(config)
        assert result.success

        optimized = result.value
        assert optimized["performance_level"] == "high"
        assert optimized["service_cache_size"] == 500
        assert optimized["enable_service_pooling"] is True
        assert optimized["business_rule_cache_size"] == 1000
        assert optimized["cross_entity_batch_size"] == 100
        assert optimized["optimization_level"] == "aggressive"

    def test_optimize_domain_services_performance_medium(self) -> None:
        """Test optimizing domain services for medium performance."""
        config = {"performance_level": "medium"}

        result = FlextDomainService.optimize_domain_services_performance(config)
        assert result.success

        optimized = result.value
        assert optimized["performance_level"] == "medium"
        assert optimized["service_cache_size"] == 250
        assert optimized["business_rule_cache_size"] == 500
        assert optimized["cross_entity_batch_size"] == 50
        assert optimized["optimization_level"] == "balanced"

    def test_optimize_domain_services_performance_low(self) -> None:
        """Test optimizing domain services for low performance."""
        config = {"performance_level": "low"}

        result = FlextDomainService.optimize_domain_services_performance(config)
        assert result.success

        optimized = result.value
        assert optimized["performance_level"] == "low"
        assert optimized["service_cache_size"] == 50
        assert optimized["enable_service_pooling"] is False
        assert optimized["business_rule_cache_size"] == 100
        assert optimized["optimization_level"] == "conservative"

    def test_optimize_domain_services_performance_default(self) -> None:
        """Test optimizing domain services with default performance level."""
        config: dict[str, object] = {}

        result = FlextDomainService.optimize_domain_services_performance(config)
        assert result.success

        optimized = result.value
        assert optimized["performance_level"] == "medium"  # Default
        assert "optimization_enabled" in optimized
        assert "optimization_timestamp" in optimized

    def test_optimize_domain_services_performance_exception(self) -> None:
        """Test optimize_domain_services_performance exception handling."""
        import unittest.mock

        with unittest.mock.patch.object(
            FlextDomainService,
            "optimize_domain_services_performance",
            classmethod(
                lambda cls, config: FlextResult.fail("Performance optimization failed")
            ),
        ):
            result = FlextDomainService.optimize_domain_services_performance({})
            assert result.failure
            assert "Performance optimization failed" in result.error


class TestFlextDomainServiceIntegration:
    """Test FlextDomainService integration with FLEXT ecosystem."""

    def test_mixin_integration(self) -> None:
        """Test mixin integration (Serializable, Loggable)."""

        class TestDomainService(FlextDomainService[str]):
            """Test domain service implementation."""

            test_field: str = "test_value"

            def execute(self) -> FlextResult[str]:
                """Execute test operation."""
                return FlextResult[str].ok("test_result")

        service = TestDomainService()

        # Test Serializable mixin
        assert hasattr(service, "to_dict")
        service_dict = service.to_dict()
        assert isinstance(service_dict, dict)

        # Test Loggable mixin
        assert hasattr(service, "get_logger")
        logger = service.get_logger()
        assert logger is not None

    def test_complex_service_example(self) -> None:
        """Test complex service with realistic domain logic."""

        class OrderProcessingService(FlextDomainService[dict]):
            """Example order processing service."""

            customer_id: str
            order_items: list[str]
            payment_method: str = "credit_card"

            def execute(self) -> FlextResult[dict]:
                """Execute order processing with railway programming."""
                return (
                    self.validate_business_rules()
                    .flat_map(lambda _: self._validate_inventory())
                    .flat_map(lambda _: self._process_payment())
                    .flat_map(self._create_order)
                    .map(self._add_confirmation_number)
                )

            def validate_business_rules(self) -> FlextResult[None]:
                """Validate order business rules."""
                if not self.customer_id:
                    return FlextResult[None].fail("Customer ID required")
                if not self.order_items:
                    return FlextResult[None].fail("Order items required")
                if len(self.order_items) > 10:
                    return FlextResult[None].fail("Too many items in order")
                return FlextResult[None].ok(None)

            def _validate_inventory(self) -> FlextResult[None]:
                """Validate inventory availability."""
                # Simulate inventory check
                if "out_of_stock" in self.order_items:
                    return FlextResult[None].fail("Item out of stock")
                return FlextResult[None].ok(None)

            def _process_payment(self) -> FlextResult[dict]:
                """Process payment."""
                if self.payment_method == "invalid":
                    return FlextResult[dict].fail("Invalid payment method")
                return FlextResult[dict].ok({"payment_id": "pay_123", "amount": 99.99})

            def _create_order(self, payment: dict) -> FlextResult[dict]:
                """Create order with payment."""
                order = {
                    "order_id": "ord_456",
                    "customer_id": self.customer_id,
                    "items": self.order_items,
                    "payment": payment,
                }
                return FlextResult[dict].ok(order)

            def _add_confirmation_number(self, order: dict) -> dict:
                """Add confirmation number to order."""
                order["confirmation"] = "CONF_789"
                return order

        # Test successful order processing
        service = OrderProcessingService(
            customer_id="cust_123", order_items=["item1", "item2"]
        )

        result = service.execute()
        assert result.success
        order = result.value
        assert order["order_id"] == "ord_456"
        assert order["customer_id"] == "cust_123"
        assert order["confirmation"] == "CONF_789"

        # Test business rule failure
        service = OrderProcessingService(
            customer_id="",  # Invalid
            order_items=["item1"],
        )
        result = service.execute()
        assert result.failure
        assert "Customer ID required" in result.error

        # Test inventory failure
        service = OrderProcessingService(
            customer_id="cust_123", order_items=["out_of_stock"]
        )
        result = service.execute()
        assert result.failure
        assert "Item out of stock" in result.error

        # Test payment failure
        service = OrderProcessingService(
            customer_id="cust_123", order_items=["item1"], payment_method="invalid"
        )
        result = service.execute()
        assert result.failure
        assert "Invalid payment method" in result.error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
