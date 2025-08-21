"""Comprehensive tests for domain_services.py module.

This test suite provides complete coverage of the domain service system,
testing all aspects including abstract service base, validation, execution,
configuration, and integration with mixins to achieve near 100% coverage.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from flext_core import FlextDomainService, FlextResult

# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_TOTAL_PAGES = 8
EXPECTED_DATA_COUNT = 3

pytestmark = [pytest.mark.unit, pytest.mark.core]


# Create concrete test services for testing
class SampleCalculationService(FlextDomainService[float]):
    """Test calculation service for comprehensive testing."""

    value: float = 5.0
    multiplier: float = 2.0

    def execute(self) -> FlextResult[float]:
        """Execute calculation operation."""
        if self.value < 0:
            return FlextResult[float].fail("Value cannot be negative")
        result = self.value * self.multiplier
        return FlextResult[float].ok(result)

    def calculate_sum(self, a: int, b: int) -> int:
        """Calculate sum of two numbers."""
        return a + b

    def calculate_product(self, a: int, b: int) -> int:
        """Calculate product of two numbers."""
        return a * b


class SampleValidationService(FlextDomainService[str]):
    """Test validation service for comprehensive testing."""

    name: str = "test"
    min_length: int = 2

    def execute(self) -> FlextResult[str]:
        """Execute validation operation."""
        if len(self.name) < self.min_length:
            return FlextResult[str].fail("Name too short")
        return FlextResult[str].ok(self.name.upper())

    def validate_config(self) -> FlextResult[None]:
        """Validate service configuration."""
        if not self.name or not self.name.strip():
            return FlextResult[None].fail("Name cannot be empty")
        if self.min_length < 0:
            return FlextResult[None].fail("Min length cannot be negative")
        return FlextResult[None].ok(None)

    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        return "@" in email and "." in email

    def validate_age(self, age: int) -> bool:
        """Validate age range."""
        return 0 <= age <= 150


class SampleErrorService(FlextDomainService[None]):
    """Test service that always returns errors."""

    def execute(self) -> FlextResult[None]:
        """Execute operation that always fails."""
        return FlextResult[None].fail("Service always fails")

    def always_fail(self) -> FlextResult[None]:
        """Return failure for testing."""
        return FlextResult[None].fail("Service always fails")

    def fail_with_context(self, context: str) -> FlextResult[None]:
        """Fail with specific context."""
        return FlextResult[None].fail(f"Failed with context: {context}")


class SampleExceptionService(FlextDomainService[None]):
    """Test service that raises exceptions."""

    def execute(self) -> FlextResult[None]:
        """Execute operation that raises exception."""
        msg = "Service exception during execution"
        raise RuntimeError(msg)

    def raise_exception(self) -> None:
        """Raise an exception."""
        msg = "Service exception"
        raise RuntimeError(msg)

    def raise_custom_exception(self, message: str) -> None:
        """Raise custom exception."""
        raise ValueError(message)


class SampleConfigErrorService(FlextDomainService[None]):
    """Test service with configuration errors."""

    def execute(self) -> FlextResult[None]:
        """Execute operation with config error."""
        return FlextResult[None].fail("Configuration error in execution")

    def validate_config(self) -> FlextResult[None]:
        """Validate configuration (always fails for this test service)."""
        return FlextResult[None].fail("Configuration validation failed")


@pytest.mark.unit
class TestFlextDomainService:
    """Test FlextDomainService functionality."""

    def test_domain_service_creation_valid(self) -> None:
        """Test creating a valid domain service."""
        service = SampleCalculationService()

        if service.value != 5.0:
            raise AssertionError(f"Expected {5.0}, got {service.value}")
        assert service.multiplier == EXPECTED_BULK_SIZE
        assert isinstance(service, FlextDomainService)

    def test_domain_service_creation_with_defaults(self) -> None:
        """Test creating domain service with default values."""
        service = SampleCalculationService()

        if service.value != 5.0:
            raise AssertionError(f"Expected {5.0}, got {service.value}")
        assert service.multiplier == EXPECTED_BULK_SIZE  # Default value

    def test_domain_service_frozen_immutable(self) -> None:
        """Test domain service is frozen and immutable."""
        service = SampleCalculationService()

        with pytest.raises(ValueError, match=".*"):  # ValidationError or AttributeError
            service.value = 20.0

    def test_domain_service_extra_fields_forbidden(self) -> None:
        """Test domain service forbids extra fields."""
        with pytest.raises(ValueError, match=".*"):  # ValidationError
            SampleCalculationService.model_validate(
                {
                    "value": 10.0,
                    "extra_field": "not allowed",
                }
            )

    def test_execute_method_is_abstract(self) -> None:
        """Test that execute method is abstract."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            FlextDomainService()

    def test_execute_success(self) -> None:
        """Test successful service execution."""
        service = SampleCalculationService()

        result = service.execute()

        assert result.success
        if result.value != 10.0:
            raise AssertionError(f"Expected {10.0}, got {result.value}")

    def test_execute_business_logic_failure(self) -> None:
        """Test service execution with business logic failure."""
        service = SampleCalculationService.model_validate({"value": -5.0})

        result = service.execute()

        assert result.is_failure
        if "Value cannot be negative" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Value cannot be negative' in {result.error}",
            )

    def test_execute_validation_service_success(self) -> None:
        """Test validation service execution success."""
        service = SampleValidationService()

        result = service.execute()

        assert result.success
        if result.value != "TEST":
            raise AssertionError(f"Expected {'TEST'}, got {result.value}")

    def test_execute_validation_service_failure(self) -> None:
        """Test validation service execution failure."""
        service = SampleValidationService(name="a", min_length=5)

        result = service.execute()

        assert result.is_failure
        if "Name too short" not in (result.error or ""):
            raise AssertionError(f"Expected 'Name too short' in {result.error}")

    def test_validate_config_default_success(self) -> None:
        """Test default validate_config returns success."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        result = service.validate_config()

        assert result.success
        # FlextResult[None].ok(None) is valid - represents "success without data"
        assert result.value is None

    def test_validate_config_custom_success(self) -> None:
        """Test custom validate_config success."""
        service = SampleValidationService(name="test", min_length=2)

        result = service.validate_config()

        assert result.success

    def test_validate_config_custom_failure(self) -> None:
        """Test custom validate_config failure."""
        service = SampleValidationService(name="", min_length=1)

        result = service.validate_config()

        assert result.is_failure
        if "Name cannot be empty" not in (result.error or ""):
            raise AssertionError(f"Expected 'Name cannot be empty' in {result.error}")

    def test_validate_config_negative_min_length(self) -> None:
        """Test validate_config with negative min_length."""
        service = SampleValidationService(name="test", min_length=-1)

        result = service.validate_config()

        assert result.is_failure
        if "Min length cannot be negative" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Min length cannot be negative' in {result.error}",
            )

    def test_execute_operation_success(self) -> None:
        """Test execute_operation with successful operation."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        def test_operation(x: float, y: float) -> float:
            return x + y

        result = service.execute_operation("add", test_operation, 5.0, 3.0)

        assert result.success
        if result.value != EXPECTED_TOTAL_PAGES:
            raise AssertionError(f"Expected {8.0}, got {result.value}")

    def test_execute_operation_with_kwargs(self) -> None:
        """Test execute_operation with keyword arguments."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        def test_operation(x: float, multiplier: float = 1.0) -> float:
            return x * multiplier

        result = service.execute_operation(
            "multiply",
            test_operation,
            5.0,
            multiplier=3.0,
        )

        assert result.success
        if result.value != 15.0:
            raise AssertionError(f"Expected {15.0}, got {result.value}")

    def test_execute_operation_config_validation_failure(self) -> None:
        """Test execute_operation with config validation failure."""
        service = SampleConfigErrorService()

        def test_operation() -> str:
            return "should not execute"

        result = service.execute_operation("test", test_operation)

        assert result.is_failure
        if "Configuration validation failed" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Configuration validation failed' in {result.error}",
            )

    def test_execute_operation_config_validation_empty_error(self) -> None:
        """Test execute_operation with empty error from config validation."""

        # Create a service that returns empty error messages
        class EmptyErrorService(FlextDomainService[str]):
            def execute(self) -> FlextResult[str]:
                return FlextResult[str].ok("executed")

            def validate_config(self) -> FlextResult[None]:
                return FlextResult[None].fail("")  # Empty error message

        service = EmptyErrorService()

        def test_operation() -> str:
            return "should not execute"

        result = service.execute_operation("test", test_operation)

        assert result.is_failure
        # The error should indicate configuration validation failed
        if result.error not in {
            "Configuration validation failed",
            "Unknown error occurred",
            "",
        }:
            raise AssertionError(f"Expected {result.error} in {{}}")

    def test_execute_operation_exception_handling(self) -> None:
        """Test execute_operation with exception in operation."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        def failing_operation() -> None:
            msg = "Operation failed"
            raise ValueError(msg)

        result = service.execute_operation("failing", failing_operation)

        assert result.is_failure
        if "Operation failing failed" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Operation failing failed' in {result.error}",
            )

    def test_execute_operation_type_error_handling(self) -> None:
        """Test execute_operation with TypeError in operation."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        def type_error_operation() -> None:
            msg = "Type error"
            raise TypeError(msg)

        result = service.execute_operation("type_error", type_error_operation)

        assert result.is_failure
        if "Operation type_error failed" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Operation type_error failed' in {result.error}",
            )

    def test_execute_operation_runtime_error_handling(self) -> None:
        """Test execute_operation with RuntimeError in operation."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        def runtime_error_operation() -> None:
            msg = "Runtime error"
            raise RuntimeError(msg)

        result = service.execute_operation("runtime_error", runtime_error_operation)

        assert result.is_failure
        if "Operation runtime_error failed" not in (result.error or ""):
            raise AssertionError(
                f"Expected 'Operation runtime_error failed' in {result.error}",
            )

    def test_get_service_info_success(self) -> None:
        """Test get_service_info with valid service."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        info = service.get_service_info()

        if info["service_type"] != "SampleCalculationService":
            raise AssertionError(
                f"Expected {'SampleCalculationService'}, got {info['service_type']}",
            )
        if not (info["config_valid"]):
            raise AssertionError(f"Expected True, got {info['config_valid']}")

    def test_get_service_info_invalid_config(self) -> None:
        """Test get_service_info with invalid config."""
        service = SampleConfigErrorService()

        info = service.get_service_info()

        if info["service_type"] != "SampleConfigErrorService":
            raise AssertionError(
                f"Expected {'SampleConfigErrorService'}, got {info['service_type']}",
            )
        if info["config_valid"]:
            raise AssertionError(f"Expected False, got {info['config_valid']}")

    def test_get_service_info_validation_service(self) -> None:
        """Test get_service_info with validation service."""
        service = SampleValidationService(name="test", min_length=2)

        info = service.get_service_info()

        if info["service_type"] != "SampleValidationService":
            raise AssertionError(
                f"Expected {'SampleValidationService'}, got {info['service_type']}",
            )
        if not (info["config_valid"]):
            raise AssertionError(f"Expected True, got {info['config_valid']}")

    def test_get_service_info_invalid_validation_service(self) -> None:
        """Test get_service_info with invalid validation service."""
        service = SampleValidationService(name="", min_length=1)

        info = service.get_service_info()

        if info["service_type"] != "SampleValidationService":
            raise AssertionError(
                f"Expected {'SampleValidationService'}, got {info['service_type']}",
            )
        if info["config_valid"]:
            raise AssertionError(f"Expected False, got {info['config_valid']}")

    def test_inheritance_from_mixins(self) -> None:
        """Test that domain service inherits from mixins."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        # Should inherit from FlextValidatableMixin
        assert hasattr(service, "is_valid")

        # Should inherit from FlextSerializableMixin
        assert hasattr(service, "to_dict_basic")

    def test_mixin_validation_functionality(self) -> None:
        """Test mixin validation functionality works."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        # Test that validation properties exist
        assert hasattr(service, "is_valid")
        # Note: We're not testing the full mixin functionality here
        # as that's covered in the mixin tests

    def test_mixin_serialization_functionality(self) -> None:
        """Test mixin serialization functionality works."""
        service = SampleCalculationService.model_validate(
            {
                "value": 10.0,
                "multiplier": 3.0,
            }
        )

        # Test that serialization methods exist
        assert hasattr(service, "to_dict_basic")

        # Test basic serialization works
        result = service.to_dict_basic()
        assert isinstance(result, dict)

    def test_service_with_complex_data_types(self) -> None:
        """Test service with complex field types."""
        # Create a service that demonstrates complex usage
        service = SampleValidationService(name="complex_test")

        result = service.execute()
        assert result.success
        if result.value != "COMPLEX_TEST":
            raise AssertionError(f"Expected {'COMPLEX_TEST'}, got {result.value}")

    def test_error_service_execution(self) -> None:
        """Test service that always returns failure."""
        service = SampleErrorService()

        result = service.execute()

        assert result.is_failure
        if result.error != "Service always fails":
            raise AssertionError(
                f"Expected {'Service always fails'}, got {result.error}",
            )

    def test_error_service_with_default_message(self) -> None:
        """Test error service with default error message."""
        service = SampleErrorService()

        result = service.execute()

        assert result.is_failure
        if result.error != "Service always fails":
            raise AssertionError(
                f"Expected {'Service always fails'}, got {result.error}",
            )


@pytest.mark.unit
class TestDomainServiceIntegration:
    """Test domain service integration scenarios."""

    def test_service_composition_sequential(self) -> None:
        """Test sequential service execution composition."""
        calc_service = SampleCalculationService()
        validation_service = SampleValidationService()

        # Execute services in sequence
        calc_result = calc_service.execute()
        validation_result = validation_service.execute()

        assert calc_result.success
        if calc_result.value != 10.0:
            raise AssertionError(f"Expected {10.0}, got {calc_result.value}")
        assert validation_result.success
        if validation_result.value != "TEST":
            raise AssertionError(f"Expected {'TEST'}, got {validation_result.value}")

    def test_service_composition_with_failure(self) -> None:
        """Test service composition when one service fails."""
        calc_service = SampleCalculationService.model_validate(
            {"value": -5.0}
        )  # Will fail
        validation_service = SampleValidationService(name="test", min_length=2)

        calc_result = calc_service.execute()
        validation_result = validation_service.execute()

        assert calc_result.is_failure
        assert validation_result.success

    def test_service_chaining_with_results(self) -> None:
        """Test chaining services using results."""
        calc_service = SampleCalculationService.model_validate(
            {
                "value": 5.0,
                "multiplier": 2.0,
            }
        )

        calc_result = calc_service.execute()
        if calc_result.success:
            # Use calculation result in another service
            # Since calc_result comes from SampleCalculationService[float], data is guaranteed to be float
            data_value = float(calc_result.value)  # Type guard to ensure float
            validation_service = SampleValidationService(
                name=f"result_{int(data_value)}",
                min_length=5,
            )
            validation_result = validation_service.execute()

            assert validation_result.success
            if validation_result.value != "RESULT_10":
                raise AssertionError(
                    f"Expected {'RESULT_10'}, got {validation_result.value}",
                )

    def test_multiple_service_info_collection(self) -> None:
        """Test collecting service info from multiple services."""
        calc_service = SampleCalculationService.model_validate({"value": 10.0})
        validation_service = SampleValidationService(name="test", min_length=2)
        error_service = SampleErrorService()
        config_error_service = SampleConfigErrorService()

        service_infos = [
            calc_service.get_service_info(),
            validation_service.get_service_info(),
            error_service.get_service_info(),
            config_error_service.get_service_info(),
        ]

        if len(service_infos) != 4:
            raise AssertionError(f"Expected {4}, got {len(service_infos)}")
        assert service_infos[0]["service_type"] == "SampleCalculationService"
        if service_infos[1]["service_type"] != "SampleValidationService":
            raise AssertionError(
                f"Expected {'SampleValidationService'}, got {service_infos[1]['service_type']}",
            )
        assert service_infos[2]["service_type"] == "SampleErrorService"
        if service_infos[3]["service_type"] != "SampleConfigErrorService":
            raise AssertionError(
                f"Expected {'SampleConfigErrorService'}, got {service_infos[3]['service_type']}",
            )

        # Check config validity
        if not (service_infos[0]["config_valid"]):
            raise AssertionError(
                f"Expected True, got {service_infos[0]['config_valid']}",
            )
        assert service_infos[1]["config_valid"] is True
        assert service_infos[2]["config_valid"] is True  # ErrorService has valid config
        assert (
            service_infos[3]["config_valid"] is False
        )  # ConfigErrorService has invalid config

    def test_service_execute_operation_complex_workflow(self) -> None:
        """Test complex workflow using execute_operation."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        def complex_operation(base: float, operations: list[dict[str, float]]) -> float:
            result = base
            for op in operations:
                if "add" in op:
                    result += op["add"]
                elif "multiply" in op:
                    result *= op["multiply"]
            return result

        operations = [
            {"add": 5.0},
            {"multiply": 2.0},
            {"add": 3.0},
        ]

        result = service.execute_operation(
            "complex_workflow",
            complex_operation,
            service.value,
            operations,
        )

        assert result.success
        # (10 + 5) * 2 + 3 = 33
        if result.value != 33.0:
            raise AssertionError(f"Expected {33.0}, got {result.value}")


@pytest.mark.unit
class TestDomainServiceEdgeCases:
    """Test domain service edge cases and error conditions."""

    def test_service_with_empty_operation_name(self) -> None:
        """Test execute_operation with empty operation name."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        def test_operation() -> str:
            return "executed"

        result = service.execute_operation("", test_operation)

        assert result.success
        if result.value != "executed":
            raise AssertionError(f"Expected {'executed'}, got {result.value}")

    def test_service_operation_with_no_args(self) -> None:
        """Test execute_operation with no arguments."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        def no_args_operation() -> str:
            return "no args"

        result = service.execute_operation("no_args", no_args_operation)

        assert result.success
        if result.value != "no args":
            raise AssertionError(f"Expected {'no args'}, got {result.value}")

    def test_service_operation_returning_none(self) -> None:
        """Test execute_operation with operation returning None."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        def none_operation() -> None:
            return None

        result = service.execute_operation("none_op", none_operation)

        assert result.success
        # FlextResult[None].ok(None) is valid - represents "success without data"
        assert result.value is None

    def test_service_operation_returning_complex_object(self) -> None:
        """Test execute_operation with operation returning complex object."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        def complex_operation() -> dict[str, object]:
            return {
                "result": 42,
                "status": "success",
                "data": [1, 2, 3],
            }

        result = service.execute_operation("complex", complex_operation)

        assert result.success
        assert isinstance(result.value, dict)
        if (result.value or {})["result"] != 42:
            raise AssertionError(f"Expected {42}, got {(result.value or {})['result']}")
        assert (result.value or {})["status"] == "success"

    def test_service_with_zero_values(self) -> None:
        """Test service with zero values."""
        service = SampleCalculationService.model_validate(
            {
                "value": 0.0,
                "multiplier": 0.0,
            }
        )

        result = service.execute()

        assert result.success
        if result.value != 0.0:
            raise AssertionError(f"Expected {0.0}, got {result.value}")

    def test_service_with_large_values(self) -> None:
        """Test service with large values."""
        service = SampleCalculationService.model_validate(
            {
                "value": 1e10,
                "multiplier": 1e5,
            }
        )

        result = service.execute()

        assert result.success
        if result.value != 1e15:
            raise AssertionError(f"Expected {1e15}, got {result.value}")

    def test_validation_service_boundary_conditions(self) -> None:
        """Test validation service at boundary conditions."""
        # Test exactly at minimum length
        service = SampleValidationService()
        result = service.execute()
        assert result.success
        if result.value != "TEST":
            raise AssertionError(f"Expected {'TEST'}, got {result.value}")

        # Test just below minimum length
        service = SampleValidationService(name="a", min_length=2)
        result = service.execute()
        assert result.is_failure

    def test_service_info_with_invalid_config_service(self) -> None:
        """Test get_service_info with a service that has invalid config."""
        # Use the SampleConfigErrorService which has invalid config by design
        service = SampleConfigErrorService()

        info = service.get_service_info()

        if info["service_type"] != "SampleConfigErrorService":
            raise AssertionError(
                f"Expected {'SampleConfigErrorService'}, got {info['service_type']}",
            )
        # Should handle invalid config gracefully
        if info["config_valid"]:
            raise AssertionError(f"Expected False, got {info['config_valid']}")


@pytest.mark.unit
class TestDomainServiceTypes:
    """Test domain service type checking and validation."""

    def test_service_type_annotations(self) -> None:
        """Test that service type annotations are preserved."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        # Check that field annotations are preserved
        annotations = service.__annotations__
        if "value" not in annotations:
            raise AssertionError(f"Expected {'value'} in {annotations}")
        assert "multiplier" in annotations

    def test_service_field_types(self) -> None:
        """Test service field type validation."""
        # Valid types should work
        service = SampleCalculationService.model_validate(
            {
                "value": 10.0,
                "multiplier": 2.5,
            }
        )
        if service.value != 10.0:
            raise AssertionError(f"Expected {10.0}, got {service.value}")
        assert service.multiplier == 2.5

        # Test with integer (should be coerced to float)
        service = SampleCalculationService.model_validate(
            {
                "value": 10,
                "multiplier": 2,
            }
        )
        if service.value != 10.0:
            raise AssertionError(f"Expected {10.0}, got {service.value}")
        assert service.multiplier == 2.0

    def test_service_validation_with_string_field(self) -> None:
        """Test service with string field validation."""
        service = SampleValidationService(name="test_string", min_length=5)

        if service.name != "test_string":
            raise AssertionError(f"Expected {'test_string'}, got {service.name}")
        assert service.min_length == 5

        result = service.execute()
        assert result.success
        if result.value != "TEST_STRING":
            raise AssertionError(f"Expected {'TEST_STRING'}, got {result.value}")

    def test_service_model_config_attributes(self) -> None:
        """Test service model config attributes."""
        service = SampleCalculationService.model_validate({"value": 10.0})

        # Should be frozen (immutable)
        config = service.model_config
        frozen_value = config.get("frozen", False)
        if not frozen_value:
            raise AssertionError(f"Expected True, got {frozen_value}")
        assert config.get("validate_assignment", False) is True
        extra_value = config.get("extra", "allow")
        if extra_value != "forbid":
            raise AssertionError(f"Expected forbid, got {extra_value}")


@pytest.mark.unit
class TestDomainServiceRealFunctionIntegration:
    """Test domain service with real function integration patterns."""

    def test_service_with_real_dependencies(self) -> None:
        """Test service with real function dependencies."""
        service = SampleCalculationService()

        # Create a real operation function that processes the argument
        def real_operation(arg: str) -> int:
            """Real operation that converts string to integer value."""
            if arg == "test_arg":
                return 42
            return 0

        result = service.execute_operation(
            "real_operation",
            real_operation,
            "test_arg",
        )

        assert result.success
        if result.value != 42:
            raise AssertionError(f"Expected {42}, got {result.value}")
        # Test that the real function processes the argument correctly
        assert real_operation("test_arg") == 42

    def test_service_operation_callback_pattern(self) -> None:
        """Test service operation with callback pattern."""
        service = SampleCalculationService.model_validate({"value": 10.0})
        callback_results: list[str] = []

        def callback_operation(value: float, callback: Callable[[str], None]) -> float:
            callback(f"Processing {value}")
            callback("Processing complete")
            return value * 2

        def test_callback(message: str) -> None:
            callback_results.append(message)

        result = service.execute_operation(
            "callback_op",
            callback_operation,
            service.value,
            test_callback,
        )

        assert result.success
        if result.value != 20.0:
            raise AssertionError(f"Expected {20.0}, got {result.value}")
        assert len(callback_results) == EXPECTED_BULK_SIZE
        if "Processing 10.0" not in callback_results:
            raise AssertionError(f"Expected {'Processing 10.0'} in {callback_results}")
        assert "Processing complete" in callback_results

    def test_service_with_complex_validation_logic(self) -> None:
        """Test service with complex validation scenarios."""
        # Test service with multiple validation conditions
        service = SampleValidationService()

        config_result = service.validate_config()
        assert config_result.success

        execute_result = service.execute()
        assert execute_result.success
        if execute_result.value != "TEST":
            raise AssertionError(f"Expected {'TEST'}, got {execute_result.value}")

        # Test validation failure cascade
        service = SampleValidationService(name=" ", min_length=5)  # Whitespace only
        config_result = service.validate_config()
        assert config_result.is_failure
        if "Name cannot be empty" not in (config_result.error or ""):
            raise AssertionError(
                f"Expected {'Name cannot be empty'} in {config_result.error}",
            )
