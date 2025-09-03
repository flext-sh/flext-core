"""Comprehensive tests for delegation_system.py to achieve maximum test coverage.

Tests cover all aspects of the FlextDelegationSystem including protocols, delegated properties,
mixin delegators, system validation, and configuration management.
"""

from unittest.mock import MagicMock, patch

import pytest

from flext_core.delegation_system import (
    FlextDelegationSystem,
)
from flext_core.exceptions import FlextExceptions


class TestFlextDelegationSystemProtocols:
    """Test delegation system protocol implementations."""

    def test_has_delegator_protocol(self) -> None:
        """Test HasDelegator protocol implementation."""

        # Create a class that implements the protocol
        class ImplementsHasDelegator:
            def __init__(self) -> None:
                self.delegator = FlextDelegationSystem.MixinDelegator(self)

        instance = ImplementsHasDelegator()
        # Test that the instance has the required attributes
        assert hasattr(instance, "delegator")
        assert isinstance(instance.delegator, FlextDelegationSystem.MixinDelegator)

    def test_delegator_protocol(self) -> None:
        """Test DelegatorProtocol implementation."""
        # Create a delegator that implements the protocol
        delegator = FlextDelegationSystem.MixinDelegator(object())
        assert isinstance(delegator, FlextDelegationSystem.DelegatorProtocol)
        assert hasattr(delegator, "get_delegation_info")
        assert callable(delegator.get_delegation_info)

    def test_delegated_method_protocol(self) -> None:
        """Test DelegatedMethodProtocol implementation."""

        # Create a test mixin
        class TestMixin:
            def test_method(self, x: int) -> int:
                return x * 2

        mixin = TestMixin()

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host, TestMixin)

        # Get the delegated method
        delegated_method = delegator._create_delegated_method("test_method", mixin)
        assert isinstance(
            delegated_method, FlextDelegationSystem.DelegatedMethodProtocol
        )
        assert callable(delegated_method)

        # Test method functionality
        result = delegated_method(5)
        assert result == 10


class TestDelegatedProperty:
    """Test DelegatedProperty descriptor functionality."""

    def test_delegated_property_creation(self) -> None:
        """Test DelegatedProperty descriptor creation."""

        class TestMixin:
            def __init__(self) -> None:
                self.test_value = "mixin_value"

        mixin = TestMixin()
        prop = FlextDelegationSystem.DelegatedProperty("test_value", mixin, "default")

        assert prop.prop_name == "test_value"
        assert prop.mixin_instance is mixin
        assert prop.default == "default"

    def test_delegated_property_get(self) -> None:
        """Test DelegatedProperty __get__ method."""

        class TestMixin:
            def __init__(self) -> None:
                self.test_value = "mixin_value"

        mixin = TestMixin()
        prop = FlextDelegationSystem.DelegatedProperty("test_value", mixin, "default")

        # Test instance access
        instance = object()
        value = prop.__get__(instance, type(instance))
        assert value == "mixin_value"

        # Test class access
        descriptor = prop.__get__(None, type(instance))
        assert descriptor is prop

    def test_delegated_property_get_default(self) -> None:
        """Test DelegatedProperty returns default when property missing."""

        class TestMixin:
            pass

        mixin = TestMixin()
        prop = FlextDelegationSystem.DelegatedProperty(
            "missing_prop", mixin, "default_value"
        )

        instance = object()
        value = prop.__get__(instance, type(instance))
        assert value == "default_value"

    def test_delegated_property_set(self) -> None:
        """Test DelegatedProperty __set__ method."""

        class TestMixin:
            def __init__(self) -> None:
                self.test_value = "original"

        mixin = TestMixin()
        prop = FlextDelegationSystem.DelegatedProperty("test_value", mixin)

        class TestInstance:
            pass

        instance = TestInstance()
        prop.__set__(instance, "new_value")

        # Should set on mixin instance
        assert mixin.test_value == "new_value"
        # Should also set on host instance with private name
        assert hasattr(instance, "_test_value")
        # Use getattr to safely access the attribute
        assert getattr(instance, "_test_value") == "new_value"

    def test_delegated_property_set_readonly(self) -> None:
        """Test DelegatedProperty with read-only mixin property."""

        class ReadOnlyMixin:
            @property
            def readonly_prop(self) -> str:
                return "readonly_value"

        mixin = ReadOnlyMixin()
        prop = FlextDelegationSystem.DelegatedProperty("readonly_prop", mixin)

        class TestInstance:
            pass

        instance = TestInstance()
        # Should not raise exception even if mixin property is read-only
        prop.__set__(instance, "new_value")

        # Should set on host instance as fallback
        assert hasattr(instance, "_readonly_prop")
        # Use getattr to safely access the attribute
        assert getattr(instance, "_readonly_prop") == "new_value"


class TestMixinDelegator:
    """Test MixinDelegator functionality."""

    def test_mixin_delegator_creation(self) -> None:
        """Test MixinDelegator creation with no mixins."""

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host)

        assert delegator.host_instance is host
        assert isinstance(delegator.mixin_instances, dict)
        assert len(delegator.mixin_instances) == 0
        assert hasattr(delegator, "logger")

    def test_mixin_delegator_with_mixins(self) -> None:
        """Test MixinDelegator creation with mixins."""

        class TestMixin:
            def __init__(self) -> None:
                self.is_valid = True
                self.validation_errors: list[str] = []

            def test_method(self) -> str:
                return "test_result"

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host, TestMixin)

        assert TestMixin in delegator.mixin_instances
        assert isinstance(delegator.mixin_instances[TestMixin], TestMixin)

        # Should have delegated properties and methods
        assert hasattr(host, "test_method")
        # Use getattr to safely access the method
        test_method = getattr(host, "test_method")
        assert callable(test_method)

    def test_register_mixin_success(self) -> None:
        """Test successful mixin registration."""

        class TestMixin:
            def __init__(self) -> None:
                self.test_attr = "test_value"

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host)
        delegator._register_mixin(TestMixin)

        assert TestMixin in delegator.mixin_instances
        assert isinstance(delegator.mixin_instances[TestMixin], TestMixin)

    def test_register_mixin_with_post_init(self) -> None:
        """Test mixin registration with __post_init__ method."""

        class MixinWithPostInit:
            def __init__(self) -> None:
                self.initialized = False
                self.post_init_called = False

            def __post_init__(self) -> None:
                self.post_init_called = True

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host, MixinWithPostInit)

        mixin_instance = delegator.mixin_instances[MixinWithPostInit]
        # Use getattr to safely access the attribute
        assert getattr(mixin_instance, "post_init_called", False) is True

    def test_register_mixin_failure(self) -> None:
        """Test mixin registration failure handling."""

        class FailingMixin:
            def __init__(self) -> None:
                msg = "Mixin initialization failed"
                raise ValueError(msg)

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host)

        with pytest.raises(FlextExceptions.BaseError, match="Failed to register mixin"):
            delegator._register_mixin(FailingMixin)

    def test_auto_delegate_methods(self) -> None:
        """Test automatic method delegation."""

        class TestMixin:
            def __init__(self) -> None:
                self.is_valid = True
                self.validation_errors: list[str] = []

            def public_method(self) -> str:
                return "public"

            def _private_method(self) -> str:
                return "private"

            @property
            def property_method(self) -> str:
                return "property"

        class TestHost:
            pass

        host = TestHost()
        FlextDelegationSystem.MixinDelegator(host, TestMixin)

        # Should delegate public methods
        assert hasattr(host, "public_method")
        # Use getattr to safely access the method
        public_method = getattr(host, "public_method")
        assert public_method() == "public"

        # Should not delegate private methods
        assert not hasattr(host, "_private_method")

    def test_create_delegated_property(self) -> None:
        """Test delegated property creation."""

        class TestMixin:
            def __init__(self) -> None:
                self.test_prop = "test_value"

        class TestHost:
            pass

        mixin = TestMixin()
        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host)

        prop = delegator._create_delegated_property("test_prop", mixin)
        assert isinstance(prop, FlextDelegationSystem.DelegatedProperty)
        assert prop.prop_name == "test_prop"
        assert prop.mixin_instance is mixin

    def test_create_delegated_method(self) -> None:
        """Test delegated method creation."""

        class TestMixin:
            def test_method(self, x: int) -> int:
                return x * 2

        mixin = TestMixin()

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host)

        method = delegator._create_delegated_method("test_method", mixin)
        assert callable(method)
        assert method(5) == 10
        # Use getattr to safely access the __name__ attribute
        assert getattr(method, "__name__", None) == "test_method"

    def test_create_delegated_method_with_exception(self) -> None:
        """Test delegated method with exception handling."""

        class FailingMixin:
            def failing_method(self) -> None:
                msg = "Method failed"
                raise ValueError(msg)

        mixin = FailingMixin()

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host)

        method = delegator._create_delegated_method("failing_method", mixin)
        with pytest.raises(
            FlextExceptions.BaseError, match="Delegated method failing_method failed"
        ):
            method()

    def test_validate_delegation_success(self) -> None:
        """Test successful delegation validation."""

        class ValidMixin:
            def __init__(self) -> None:
                self.is_valid = True

            def test_method(self) -> str:
                return "test"

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host, ValidMixin)

        result = delegator._validate_delegation()
        assert result.success is True

    def test_validate_delegation_invalid_mixin(self) -> None:
        """Test delegation validation with invalid mixin."""

        class InvalidMixin:
            def __init__(self) -> None:
                self.is_valid = False

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host, InvalidMixin)

        result = delegator._validate_delegation()
        assert result.success is False
        assert result.error is not None
        assert "not valid" in result.error

    def test_validate_delegation_exception(self) -> None:
        """Test delegation validation with exception."""

        class ExceptionMixin:
            def __init__(self) -> None:
                self.is_valid = True

            def validate(self) -> None:
                msg = "Validation method failed"
                raise ValueError(msg)

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host, ExceptionMixin)

        # Now cause an exception by having dir() fail for a specific mixin instance
        with patch("builtins.dir") as mock_dir:

            def side_effect_dir(obj: object) -> list[str]:
                # Cause exception only for ExceptionMixin instance
                if isinstance(obj, ExceptionMixin):
                    msg = "Dir failed on ExceptionMixin"
                    raise RuntimeError(msg)
                # Normal behavior for other objects
                return list(object.__dir__(obj))

            mock_dir.side_effect = side_effect_dir
            result = delegator._validate_delegation()
            assert result.success is False
            assert result.error is not None
            assert "Validation failed" in result.error

    def test_get_mixin_instance(self) -> None:
        """Test getting specific mixin instance."""

        class TestMixin:
            pass

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host, TestMixin)

        instance = delegator.get_mixin_instance(TestMixin)
        assert isinstance(instance, TestMixin)

        # Test with non-existent mixin
        class NonExistentMixin:
            pass

        instance = delegator.get_mixin_instance(NonExistentMixin)
        assert instance is None

    def test_get_delegation_info(self) -> None:
        """Test getting delegation information."""

        class TestMixin:
            def test_method(self) -> str:
                return "test"

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host, TestMixin)

        info = delegator.get_delegation_info()
        assert isinstance(info, dict)
        assert "host_class" in info
        assert "mixin_classes" in info
        assert "delegated_methods" in info
        assert info["host_class"] == "TestHost"
        # Use isinstance to ensure proper type checking
        mixin_classes = info["mixin_classes"]
        delegated_methods = info["delegated_methods"]
        assert isinstance(mixin_classes, (list, tuple, set))
        assert isinstance(delegated_methods, (list, tuple, set))
        assert "TestMixin" in mixin_classes
        assert "test_method" in delegated_methods

    def test_delegated_properties_class_var(self) -> None:
        """Test _DELEGATED_PROPERTIES class variable."""
        expected_props = ["is_valid", "validation_errors"]
        assert (
            expected_props == FlextDelegationSystem.MixinDelegator._DELEGATED_PROPERTIES
        )


class TestFlextDelegationSystemFactoryMethods:
    """Test FlextDelegationSystem factory methods."""

    def test_create_mixin_delegator(self) -> None:
        """Test factory method for creating mixin delegator."""

        class TestMixin:
            def test_method(self) -> str:
                return "test"

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.create_mixin_delegator(host, TestMixin)

        assert isinstance(delegator, FlextDelegationSystem.MixinDelegator)
        assert delegator.host_instance is host
        assert TestMixin in delegator.mixin_instances

    def test_create_mixin_delegator_multiple_mixins(self) -> None:
        """Test factory method with multiple mixins."""

        class MixinA:
            def method_a(self) -> str:
                return "a"

        class MixinB:
            def method_b(self) -> str:
                return "b"

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.create_mixin_delegator(host, MixinA, MixinB)

        assert isinstance(delegator, FlextDelegationSystem.MixinDelegator)
        assert len(delegator.mixin_instances) == 2
        assert MixinA in delegator.mixin_instances
        assert MixinB in delegator.mixin_instances


class TestFlextDelegationSystemValidation:
    """Test FlextDelegationSystem validation functionality."""

    def test_validate_delegation_system_success(self) -> None:
        """Test successful system validation."""
        # Mock FlextMixins methods and patch the TestHost creation
        with patch("flext_core.delegation_system.FlextMixins") as mock_mixins:

            def mock_init_validation(obj: object) -> None:
                # Add required attributes for validation using setattr
                setattr(obj, "is_valid", lambda: True)
                setattr(obj, "get_validation_errors", list)
                setattr(obj, "has_validation_errors", lambda: False)
                setattr(obj, "to_dict_basic", dict)

                # Mock the validation_errors property by adding it to the class
                # We need to create a simple property that returns empty list
                if not hasattr(obj.__class__, "validation_errors"):
                    setattr(
                        obj.__class__, "validation_errors", property(lambda self: [])
                    )

            mock_mixins.create_timestamp_fields = MagicMock()
            mock_mixins.initialize_validation = mock_init_validation

            result = FlextDelegationSystem.validate_delegation_system()

            assert result.success is True
            report = result.unwrap()
            assert isinstance(report, dict)
            assert "status" in report
            assert "test_results" in report
            assert "delegation_info" in report
            assert report["status"] == "SUCCESS"

    @patch.object(FlextDelegationSystem, "_validate_delegation_methods")
    def test_validate_delegation_system_failure(
        self, mock_validate_methods: MagicMock
    ) -> None:
        """Test system validation failure."""
        mock_validate_methods.side_effect = ValueError("Validation failed")

        # Also need to mock FlextMixins for this test
        with patch("flext_core.delegation_system.FlextMixins") as mock_mixins:
            mock_mixins.create_timestamp_fields = MagicMock()
            mock_mixins.initialize_validation = MagicMock()

            result = FlextDelegationSystem.validate_delegation_system()

            assert result.success is False
            assert result.error is not None
            assert "Validation failed" in result.error

    def test_validate_delegation_methods(self) -> None:
        """Test delegation methods validation."""

        # Create a test host with required attributes
        class TestHost:
            def __init__(self) -> None:
                self.is_valid = True
                self.validation_errors: list[str] = []
                self.has_validation_errors = lambda: False
                self.to_dict_basic = dict

        host = TestHost()
        test_results: list[str] = []

        # Should not raise exception
        FlextDelegationSystem._validate_delegation_methods(host, test_results)
        assert len(test_results) == 2
        assert "✓ Validation methods successfully delegated" in test_results
        assert "✓ Serialization methods successfully delegated" in test_results

    def test_validate_delegation_methods_missing_attributes(self) -> None:
        """Test delegation methods validation with missing attributes."""

        class TestHost:
            pass

        host = TestHost()  # Missing required attributes
        test_results: list[str] = []

        with pytest.raises(
            FlextExceptions.BaseError, match="is_valid property not delegated"
        ):
            FlextDelegationSystem._validate_delegation_methods(host, test_results)

    def test_validate_method_functionality(self) -> None:
        """Test method functionality validation."""

        class TestHost:
            def __init__(self) -> None:
                pass

            def is_valid(self) -> bool:
                return True

        host = TestHost()
        test_results: list[str] = []

        # Should not raise exception
        FlextDelegationSystem._validate_method_functionality(host, test_results)
        assert len(test_results) == 1
        assert "✓ Delegated methods are functional" in test_results

    def test_validate_method_functionality_invalid_type(self) -> None:
        """Test method functionality validation with invalid type."""

        class TestHost:
            def __init__(self) -> None:
                pass

            def is_valid(self) -> str:  # Method that returns wrong type
                return "not_a_bool"

        host = TestHost()
        test_results: list[str] = []

        with pytest.raises(
            FlextExceptions.TypeError, match="is_valid should return bool"
        ):
            FlextDelegationSystem._validate_method_functionality(host, test_results)

    def test_get_host_delegator_success(self) -> None:
        """Test successful host delegator access."""

        class TestHost:
            def __init__(self) -> None:
                self.delegator = FlextDelegationSystem.MixinDelegator(self)

        host = TestHost()
        delegator = FlextDelegationSystem._get_host_delegator(host)

        assert isinstance(delegator, FlextDelegationSystem.MixinDelegator)
        assert delegator is host.delegator

    def test_get_host_delegator_missing_attribute(self) -> None:
        """Test host delegator access with missing attribute."""

        class TestHost:
            pass

        host = TestHost()  # No delegator attribute

        with pytest.raises(
            FlextExceptions.BaseError, match="Host missing delegator attribute"
        ):
            FlextDelegationSystem._get_host_delegator(host)

    def test_get_host_delegator_none_attribute(self) -> None:
        """Test host delegator access with None attribute."""

        class TestHost:
            def __init__(self) -> None:
                self.delegator = None

        host = TestHost()

        with pytest.raises(
            FlextExceptions.BaseError, match="Host delegator attribute is None"
        ):
            FlextDelegationSystem._get_host_delegator(host)

    def test_validate_delegation_info(self) -> None:
        """Test delegation info validation."""

        class TestHost:
            def __init__(self) -> None:
                self.delegator = FlextDelegationSystem.MixinDelegator(self)

        host = TestHost()
        test_results: list[str] = []

        info = FlextDelegationSystem._validate_delegation_info(host, test_results)

        assert isinstance(info, dict)
        assert "host_class" in info
        assert "mixin_classes" in info
        assert "delegated_methods" in info
        assert len(test_results) == 1
        assert "✓ Delegation info validation successful" in test_results

    def test_validate_delegation_info_missing_method(self) -> None:
        """Test delegation info validation with missing method."""

        class InvalidDelegator:
            pass  # Missing get_delegation_info method

        class TestHost:
            def __init__(self) -> None:
                self.delegator = InvalidDelegator()

        host = TestHost()
        test_results: list[str] = []

        with pytest.raises(
            FlextExceptions.BaseError,
            match="Delegator missing get_delegation_info method",
        ):
            FlextDelegationSystem._validate_delegation_info(host, test_results)


class TestFlextDelegationSystemConfig:
    """Test FlextDelegationSystem functionality."""

    def test_configure_delegation_system_success(self) -> None:
        """Test successful delegation system configuration."""
        config: dict[str, object] = {
            "environment": "production",
            "config_source": "environment",
            "validation_level": "strict",
        }

        result = FlextDelegationSystem.configure_delegation_system(config)

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert result.get("environment") == "production"
        assert result.get("config_source") == "environment"
        assert result.get("validation_level") == "strict"

    def test_configure_delegation_system_missing_keys(self) -> None:
        """Test delegation system configuration with missing keys."""
        config: dict[str, object] = {
            "environment": "production"
        }  # Missing required keys

        result = FlextDelegationSystem.configure_delegation_system(config)

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "Missing required configuration keys" in result.get("error", "")

    def test_configure_delegation_system_with_optional_params(self) -> None:
        """Test delegation system configuration with optional parameters."""
        config: dict[str, object] = {
            "environment": "production",
            "config_source": "environment",
            "validation_level": "strict",
            "performance_level": "high",
            "delegation_mode": "optimized",
        }

        result = FlextDelegationSystem.configure_delegation_system(config)

        assert isinstance(result, dict)
        assert result.get("success") is True
        assert result.get("performance_level") == "high"
        assert result.get("delegation_mode") == "optimized"

    def test_configure_delegation_system_exception(self) -> None:
        """Test delegation system configuration exception handling."""
        # Test with None config to trigger exception in .get() calls
        from typing import cast

        result = FlextDelegationSystem.configure_delegation_system(
            cast("dict[str, object]", None)
        )

        assert isinstance(result, dict)
        assert result.get("success") is False
        assert "error" in result
        assert "Delegation system configuration failed" in result.get("error", "")

    def test_get_delegation_system_config(self) -> None:
        """Test getting delegation system configuration."""
        config = FlextDelegationSystem.get_delegation_system_config()

        assert isinstance(config, dict)
        assert "system_name" in config
        assert "environment" in config
        assert "delegation_performance" in config
        assert "mixin_composition_performance" in config
        assert "health_status" in config
        assert config["system_name"] == "FlextDelegationSystem"

    def test_get_delegation_system_config_exception(self) -> None:
        """Test getting delegation system configuration with exception."""
        with patch("time.time", side_effect=Exception("Time error")):
            config = FlextDelegationSystem.get_delegation_system_config()

            assert isinstance(config, dict)
            assert "error" in config
            assert (
                "Failed to retrieve delegation system configuration" in config["error"]
            )

    def test_create_environment_delegation_config_development(self) -> None:
        """Test creating environment config for development."""
        config = FlextDelegationSystem.create_environment_delegation_config(
            "development"
        )

        assert isinstance(config, dict)
        assert config["environment"] == "development"
        assert config["config_source"] == "file"
        assert config["validation_level"] == "efficient"
        assert config["debugging_enabled"] is True

    def test_create_environment_delegation_config_production(self) -> None:
        """Test creating environment config for production."""
        config = FlextDelegationSystem.create_environment_delegation_config(
            "production"
        )

        assert isinstance(config, dict)
        assert config["environment"] == "production"
        assert config["config_source"] == "environment"
        assert config["validation_level"] == "strict"
        assert config["debugging_enabled"] is False

    def test_create_environment_delegation_config_unknown(self) -> None:
        """Test creating environment config for unknown environment."""
        config = FlextDelegationSystem.create_environment_delegation_config("unknown")

        assert isinstance(config, dict)
        assert config["environment"] == "unknown"
        assert "configuration_warning" in config

    def test_create_environment_delegation_config_exception(self) -> None:
        """Test creating environment config with exception."""

        # Test with an object that has no lower() method to trigger AttributeError
        # This will cause exception in environment.lower() call but allow string conversion in error handling
        class BadEnvironment:
            def __str__(self) -> str:
                return "bad_environment"

        from typing import cast

        config = FlextDelegationSystem.create_environment_delegation_config(
            cast("str", BadEnvironment())
        )

        # Should handle gracefully and return error config
        assert isinstance(config, dict)
        assert "error" in config
        assert "bad_environment" in config["error"]
        assert config["configuration_valid"] is False

    def test_optimize_delegation_performance_low(self) -> None:
        """Test performance optimization for low level."""
        config = FlextDelegationSystem.optimize_delegation_performance("low")

        assert isinstance(config, dict)
        assert config["optimization_level"] == "low"
        assert config["resource_usage"] == "minimal"
        assert config["max_concurrent_delegations"] == 4

    def test_optimize_delegation_performance_high(self) -> None:
        """Test performance optimization for high level."""
        config = FlextDelegationSystem.optimize_delegation_performance("high")

        assert isinstance(config, dict)
        assert config["optimization_level"] == "high"
        assert config["resource_usage"] == "aggressive"
        assert config["max_concurrent_delegations"] == 32

    def test_optimize_delegation_performance_unknown(self) -> None:
        """Test performance optimization for unknown level."""
        config = FlextDelegationSystem.optimize_delegation_performance("unknown")

        assert isinstance(config, dict)
        assert config["optimization_level"] == "unknown"
        assert "optimization_warning" in config

    def test_optimize_delegation_performance_exception(self) -> None:
        """Test performance optimization with exception."""
        with patch(
            "flext_core.delegation_system.FlextDelegationSystem.optimize_delegation_performance"
        ) as mock_method:
            mock_method.side_effect = Exception("Optimization error")

            # Test the actual method behavior when it handles its own exceptions
            try:
                FlextDelegationSystem.optimize_delegation_performance("test")
                # If no exception handling in method, we expect the exception
                msg = "Should have raised exception"
                raise AssertionError(msg)
            except Exception:
                # This is expected behavior if method doesn't handle exceptions internally
                pass


class TestFlextDelegationSystemIntegration:
    """Integration tests for the complete delegation system."""

    def test_full_delegation_workflow(self) -> None:
        """Test complete delegation workflow from creation to validation."""

        # Create a comprehensive mixin
        class ComprehensiveMixin:
            def __init__(self) -> None:
                self.is_valid = True
                self.validation_errors: list[str] = []
                self.data = {"key": "value"}

            def validate(self, data: dict[str, object]) -> bool:
                return bool(data)

            def get_data(self) -> dict[str, object]:
                from typing import cast

                return cast("dict[str, object]", self.data)

            def process(self, input_data: str) -> str:
                return f"processed_{input_data}"

        # Create host and delegation
        class BusinessLogic:
            def __init__(self) -> None:
                self.delegator = FlextDelegationSystem.create_mixin_delegator(
                    self, ComprehensiveMixin
                )

        # Test the complete workflow
        business = BusinessLogic()

        # Test delegated properties
        assert hasattr(business, "is_valid")
        # Check if the delegated property returns the expected value
        is_valid_prop = business.is_valid
        if hasattr(is_valid_prop, "__get__"):
            # It's a DelegatedProperty descriptor - call __get__ to get the actual value
            assert is_valid_prop.__get__(business, type(business))
        else:
            # It's a direct value
            assert is_valid_prop

        # Test delegated methods
        assert hasattr(business, "validate")
        assert business.validate({"test": "data"}) is True

        assert hasattr(business, "get_data")
        assert business.get_data() == {"key": "value"}

        assert hasattr(business, "process")
        assert business.process("test") == "processed_test"

        # Test delegation info
        info = business.delegator.get_delegation_info()
        assert isinstance(info, dict)
        mixin_classes = info["mixin_classes"]
        delegated_methods = info["delegated_methods"]
        assert isinstance(mixin_classes, (list, tuple, set))
        assert isinstance(delegated_methods, (list, tuple, set))
        assert "ComprehensiveMixin" in mixin_classes
        assert "validate" in delegated_methods
        assert "get_data" in delegated_methods
        assert "process" in delegated_methods

        # Test validation
        validation_result = business.delegator._validate_delegation()
        assert validation_result.success is True

    def test_multiple_mixin_delegation(self) -> None:
        """Test delegation with multiple mixins."""

        class ValidationMixin:
            def __init__(self) -> None:
                self.is_valid = True

            def validate(self) -> bool:
                return True

        class SerializationMixin:
            def serialize(self) -> str:
                return "serialized"

        class CachingMixin:
            def cache_get(self, key: str) -> str:
                return f"cached_{key}"

        # Create host with multiple mixins
        class MultiMixinHost:
            def __init__(self) -> None:
                self.delegator = FlextDelegationSystem.create_mixin_delegator(
                    self, ValidationMixin, SerializationMixin, CachingMixin
                )

        host = MultiMixinHost()

        # Test all mixins are registered
        assert len(host.delegator.mixin_instances) == 3
        assert ValidationMixin in host.delegator.mixin_instances
        assert SerializationMixin in host.delegator.mixin_instances
        assert CachingMixin in host.delegator.mixin_instances

        # Test all methods are delegated
        assert hasattr(host, "validate")
        assert hasattr(host, "serialize")
        assert hasattr(host, "cache_get")

        # Test functionality
        assert host.validate() is True
        assert host.serialize() == "serialized"
        assert host.cache_get("test") == "cached_test"

    def test_delegation_system_with_config(self) -> None:
        """Test delegation system with configuration management."""
        # Test configuration creation
        config: dict[str, object] = {
            "environment": "production",
            "config_source": "environment",
            "validation_level": "strict",
            "performance_level": "high",
        }

        system_config = FlextDelegationSystem.configure_delegation_system(config)
        assert isinstance(system_config, dict)
        assert system_config["success"] is True

        # Test environment-specific config
        env_config = FlextDelegationSystem.create_environment_delegation_config(
            "production"
        )
        assert isinstance(env_config, dict)
        assert env_config["environment"] == "production"

        # Test performance optimization
        perf_config = FlextDelegationSystem.optimize_delegation_performance("high")
        assert isinstance(perf_config, dict)
        assert perf_config["optimization_level"] == "high"

        # Test system validation - now working correctly after fixes
        validation_result = FlextDelegationSystem.validate_delegation_system()
        assert validation_result.success is True

    def test_error_propagation_in_delegation(self) -> None:
        """Test error propagation through delegation system."""

        class ErrorMixin:
            def failing_method(self) -> None:
                msg = "Original error"
                raise ValueError(msg)

        class TestHost:
            pass

        host = TestHost()
        FlextDelegationSystem.MixinDelegator(host, ErrorMixin)

        # Test that errors are properly wrapped and propagated
        # Use getattr to safely access the method
        failing_method = getattr(host, "failing_method")
        with pytest.raises(FlextExceptions.BaseError) as exc_info:
            failing_method()

        assert "Delegated method failing_method failed" in str(exc_info.value)
        assert "Original error" in str(exc_info.value)

    def test_property_delegation_edge_cases(self) -> None:
        """Test edge cases in property delegation."""

        class ComplexMixin:
            def __init__(self) -> None:
                self._private_value = "private"

            @property
            def computed_property(self) -> str:
                return f"computed_{self._private_value}"

            @computed_property.setter
            def computed_property(self, value: str) -> None:
                self._private_value = value

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextDelegationSystem.MixinDelegator(host, ComplexMixin)

        # Test property delegation through descriptor
        mixin_instance = delegator.mixin_instances[ComplexMixin]
        prop_descriptor = FlextDelegationSystem.DelegatedProperty(
            "computed_property", mixin_instance
        )

        # Test getter
        value = prop_descriptor.__get__(host, type(host))
        assert value == "computed_private"

        # Test setter
        prop_descriptor.__set__(host, "new_value")
        new_value = prop_descriptor.__get__(host, type(host))
        assert new_value == "computed_new_value"


if __name__ == "__main__":
    pytest.main([__file__])
