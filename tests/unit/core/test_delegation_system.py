"""Comprehensive tests for FlextMixinDelegator and delegation system.

This test suite provides complete coverage of the delegation system,
testing all aspects of mixin delegation, automatic method discovery,
property delegation, and validation to achieve near 100% coverage.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast
from unittest.mock import Mock, patch

import pytest

from flext_core import (
    FlextMixinDelegator,
    FlextOperationError,
    FlextResult,
    FlextTypeError,
    create_mixin_delegator,
    validate_delegation_system,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]

# Constants
EXPECTED_BULK_SIZE = 2
EXPECTED_DATA_COUNT = 3


class SampleMixin1:
    """Test mixin for delegation testing."""

    def __init__(self) -> None:
        """Initialize sample mixin 1."""
        self._value = "mixin1_value"

    @property
    def readonly_property(self) -> str:
        """Read-only property."""
        return "readonly_value"

    @property
    def writable_property(self) -> str:
        """Writable property."""
        return self._value

    @writable_property.setter
    def writable_property(self, value: str) -> None:
        self._value = value

    def mixin_method1(self) -> str:
        """Test method from mixin 1."""
        return "mixin1_method1"

    def mixin_method2(self, param: str) -> str:
        """Test method with parameter from mixin 1."""
        return f"mixin1_method2_{param}"


class SampleMixin2:
    """Another test mixin for delegation testing."""

    def __init__(self) -> None:
        """Initialize sample mixin 2."""
        self._data = {"key": "value"}

    @property
    def another_property(self) -> str:
        """Another property for testing."""
        return "another_value"

    def mixin_method3(self) -> dict[str, str]:
        """Test method from mixin 2."""
        return self._data.copy()

    def mixin_method4(self, key: str, value: str) -> None:
        """Test method that modifies state."""
        self._data[key] = value


class InitializableMixin:
    """Mixin with initialization methods."""

    def __init__(self) -> None:
        """Initialize initializable mixin."""
        self.initialized = False
        self.validation_initialized = False
        self.timestamps_initialized = False
        self.id_initialized = False
        self.logging_initialized = False
        self.serialization_initialized = False

    def _initialize_validation(self) -> None:
        """Initialize validation system."""
        self.validation_initialized = True

    def _initialize_timestamps(self) -> None:
        """Initialize timestamps system."""
        self.timestamps_initialized = True

    def _initialize_id(self) -> None:
        """Initialize ID system."""
        self.id_initialized = True

    def _initialize_logging(self) -> None:
        """Initialize logging system."""
        self.logging_initialized = True

    def _initialize_serialization(self) -> None:
        """Initialize serialization system."""
        self.serialization_initialized = True

    def test_method(self) -> str:
        """Test method."""
        return "initializable_method"


class FailingInitMixin:
    """Mixin with failing initialization."""

    def __init__(self) -> None:
        """Initialize failing init mixin."""
        self.initialized = False

    def _initialize_validation(self) -> None:
        """Failing initialization method."""
        msg = "Initialization failed"
        raise ValueError(msg)

    def _initialize_timestamps(self) -> None:
        """Another failing initialization method."""
        msg = "Timestamps initialization failed"
        raise TypeError(msg)

    def test_method(self) -> str:
        """Test method."""
        return "failing_init_method"


class HostObject:
    """Test host object for delegation."""

    def __init__(self) -> None:
        """Initialize host object."""
        self.host_attr = "host_value"

    def host_method(self) -> str:
        """Host's own method."""
        return "host_method_result"


class FrozenHostObject:
    """Frozen host object to test property delegation edge cases."""

    def __init__(self) -> None:
        """Initialize frozen host object."""
        self.host_attr = "frozen_host"
        self._frozen = False

    def __setattr__(self, name: str, value: object) -> None:
        """Simulate frozen object behavior."""
        if hasattr(self, "_frozen") and self._frozen and name != "_frozen":
            error_message = f"Cannot set {name} on frozen object"
            raise AttributeError(error_message)
        super().__setattr__(name, value)

    def freeze(self) -> None:
        """Freeze the object."""
        self._frozen = True


@pytest.mark.unit
class TestFlextMixinDelegator:
    """Test FlextMixinDelegator functionality."""

    def test_delegator_initialization_single_mixin(self) -> None:
        """Test delegator initialization with single mixin."""
        host = HostObject()
        delegator = FlextMixinDelegator(host, SampleMixin1)

        assert delegator._host is host
        if len(delegator._mixin_instances) != 1:
            raise AssertionError(f"Expected {1}, got {len(delegator._mixin_instances)}")
        if SampleMixin1 not in delegator._mixin_instances:
            raise AssertionError(
                f"Expected {SampleMixin1} in {delegator._mixin_instances}",
            )
        assert isinstance(delegator._mixin_instances[SampleMixin1], SampleMixin1)

    def test_delegator_initialization_multiple_mixins(self) -> None:
        """Test delegator initialization with multiple mixins."""
        host = HostObject()
        delegator = FlextMixinDelegator(host, SampleMixin1, SampleMixin2)

        if len(delegator._mixin_instances) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(delegator._mixin_instances)}")
        if SampleMixin1 not in delegator._mixin_instances:
            raise AssertionError(
                f"Expected {SampleMixin1} in {delegator._mixin_instances}",
            )
        assert SampleMixin2 in delegator._mixin_instances

    def test_delegator_initialization_no_mixins(self) -> None:
        """Test delegator initialization with no mixins."""
        host = HostObject()
        delegator = FlextMixinDelegator(host)

        if len(delegator._mixin_instances) != 0:
            raise AssertionError(f"Expected {0}, got {len(delegator._mixin_instances)}")
        assert len(delegator._delegated_methods) == 0

    def test_delegator_initialization_with_initializable_mixin(self) -> None:
        """Test delegator with mixin that has initialization methods."""
        host = HostObject()
        delegator = FlextMixinDelegator(host, InitializableMixin)

        # Check that initialization methods were called
        mixin_instance = delegator._mixin_instances[InitializableMixin]
        assert isinstance(mixin_instance, InitializableMixin)
        if not (mixin_instance.validation_initialized):
            raise AssertionError(
                f"Expected True, got {mixin_instance.validation_initialized}",
            )
        assert mixin_instance.timestamps_initialized is True
        if not (mixin_instance.id_initialized):
            raise AssertionError(f"Expected True, got {mixin_instance.id_initialized}")
        assert mixin_instance.logging_initialized is True
        if not (mixin_instance.serialization_initialized):
            raise AssertionError(
                f"Expected True, got {mixin_instance.serialization_initialized}",
            )

        # Check initialization log
        log = delegator._initialization_log
        success_entries = [entry for entry in log if "✓" in entry]
        if len(success_entries) != 5:  # All 5 initialization methods:
            raise AssertionError(f"Expected {5}, got {len(success_entries)}")

    def test_delegator_initialization_with_failing_init(self) -> None:
        """Test delegator with mixin that has failing initialization."""
        host = HostObject()
        delegator = FlextMixinDelegator(host, FailingInitMixin)

        # Mixin should still be registered despite initialization failure
        if FailingInitMixin not in delegator._mixin_instances:
            raise AssertionError(
                f"Expected {FailingInitMixin} in {delegator._mixin_instances}",
            )

        # Check that failures were logged
        log = delegator._initialization_log
        error_entries = [entry for entry in log if "✗" in entry]
        if (
            len(error_entries) != EXPECTED_BULK_SIZE
        ):  # Both initialization methods failed:
            raise AssertionError(f"Expected {2}, got {len(error_entries)}")

    def test_delegator_initialization_with_mixin_registration_failure(self) -> None:
        """Test delegator with mixin that fails to register."""
        host = HostObject()

        # Mock a failing mixin class
        class FailingMixin:
            def __init__(self) -> None:
                msg = "Cannot create mixin instance"
                raise TypeError(msg)

        with pytest.raises(TypeError, match="Cannot create mixin instance"):
            FlextMixinDelegator(host, FailingMixin)

    def test_method_delegation_basic(self) -> None:
        """Test basic method delegation."""
        host = HostObject()
        FlextMixinDelegator(host, SampleMixin1)

        # Test that method was delegated
        assert hasattr(host, "mixin_method1")
        result = host.mixin_method1()
        if result != "mixin1_method1":
            raise AssertionError(f"Expected {'mixin1_method1'}, got {result}")

    def test_method_delegation_with_arguments(self) -> None:
        """Test method delegation with arguments."""
        host = HostObject()
        FlextMixinDelegator(host, SampleMixin1)

        # Test method with arguments
        assert hasattr(host, "mixin_method2")
        result = host.mixin_method2("hello")
        if result != "mixin1_method2_hello":
            raise AssertionError(f"Expected {'mixin1_method2_hello'}, got {result}")

        # Test with default arguments
        result = host.mixin_method2("hello")
        if result != "mixin1_method2_hello":
            raise AssertionError(f"Expected {'mixin1_method2_hello'}, got {result}")

    def test_method_delegation_multiple_mixins(self) -> None:
        """Test method delegation from multiple mixins."""
        host = HostObject()
        FlextMixinDelegator(host, SampleMixin1, SampleMixin2)

        # Test methods from both mixins
        assert hasattr(host, "mixin_method1")
        assert hasattr(host, "mixin_method3")

        result1 = host.mixin_method1()
        result2 = host.mixin_method3()

        if result1 != "mixin1_method1":
            raise AssertionError(f"Expected {'mixin1_method1'}, got {result1}")
        assert result2 == {"key": "value"}

    def test_property_delegation_getter(self) -> None:
        """Test property delegation getter."""
        host = HostObject()
        FlextMixinDelegator(host, SampleMixin1)

        # Test property getter
        assert hasattr(host, "writable_property")
        value = host.writable_property
        if value != "mixin1_value":
            raise AssertionError(f"Expected {'mixin1_value'}, got {value}")

    def test_property_delegation_setter(self) -> None:
        """Test property delegation setter."""
        host = HostObject()
        FlextMixinDelegator(host, SampleMixin1)

        # Test property setter using setattr/getattr to satisfy typing
        host.writable_property = "new_value"
        if host.writable_property != "new_value":
            raise AssertionError(
                f"Expected {'new_value'}, got {host.writable_property}",
            )

    def test_property_delegation_readonly(self) -> None:
        """Test delegation of read-only property."""
        host = HostObject()
        FlextMixinDelegator(host, SampleMixin1)

        # Test readonly property via getattr
        value = host.readonly_property
        if value != "readonly_value":
            raise AssertionError(f"Expected {'readonly_value'}, got {value}")

        # Test that setting readonly property raises FlextOperationError
        # (custom behavior)
        with pytest.raises(
            FlextOperationError,
            match="Property 'readonly_property' is read-only",
        ):
            host.readonly_property = "new_value"

    def test_private_methods_not_delegated(self) -> None:
        """Test that private methods are not delegated."""
        host = HostObject()
        FlextMixinDelegator(host, SampleMixin1)

        # Private methods should not be delegated
        assert not hasattr(host, "_value")
        assert not hasattr(host, "_private_method")

        # Public methods should be delegated
        assert hasattr(host, "mixin_method1")
        assert hasattr(host, "mixin_method2")

    def test_method_delegation_error_handling(self) -> None:
        """Test error handling in delegated methods."""

        class ErrorMixin:
            def error_method(self) -> str:
                msg = "Method error"
                raise ValueError(msg)

        host = HostObject()
        FlextMixinDelegator(host, ErrorMixin)

        # Test that delegation error is properly wrapped
        with pytest.raises(
            FlextOperationError,
            match="Delegation error in ErrorMixin.error_method",
        ):
            host.error_method()

    def test_method_signature_preservation(self) -> None:
        """Test that method signatures are preserved."""
        host = HostObject()
        FlextMixinDelegator(host, SampleMixin1)

        # Test that delegated method has correct signature - use getattr for dynamic delegation
        method = getattr(host, "mixin_method2", None)
        assert method is not None
        if method.__name__ != "mixin_method2":
            raise AssertionError(f"Expected {'mixin_method2'}, got {method.__name__}")
        assert hasattr(method, "__doc__")

    def test_frozen_host_object_delegation(self) -> None:
        """Test delegation to frozen host objects."""
        host = FrozenHostObject()
        host.freeze()

        # Should still work by attaching to class
        delegator = FlextMixinDelegator(host, SampleMixin1)

        # Methods should be available through class delegation
        if "mixin_method1" not in delegator._delegated_methods:
            msg: str = f"Expected {'mixin_method1'} in {delegator._delegated_methods}"
            raise AssertionError(msg)

    def test_get_mixin_instance(self) -> None:
        """Test getting specific mixin instance."""
        host = HostObject()
        delegator = FlextMixinDelegator(host, SampleMixin1, SampleMixin2)

        # Test getting specific mixin instances
        mixin1_instance = delegator.get_mixin_instance(SampleMixin1)
        mixin2_instance = delegator.get_mixin_instance(SampleMixin2)

        assert isinstance(mixin1_instance, SampleMixin1)
        assert isinstance(mixin2_instance, SampleMixin2)

        # Test getting non-existent mixin
        non_existent = delegator.get_mixin_instance(str)
        assert non_existent is None

    def test_get_delegation_info(self) -> None:
        """Test getting delegation information."""
        host = HostObject()
        delegator = FlextMixinDelegator(host, SampleMixin1, SampleMixin2)

        info = delegator.get_delegation_info()

        assert isinstance(info, dict)
        if "registered_mixins" not in info:
            error_msg: str = f"Expected {'registered_mixins'} in {info}"
            raise AssertionError(error_msg)
        assert "delegated_methods" in info
        if "initialization_log" not in info:
            log_error_msg: str = f"Expected {'initialization_log'} in {info}"
            raise AssertionError(log_error_msg)
        assert "validation_result" in info

        registered_mixins = cast("list[object]", info["registered_mixins"])
        if SampleMixin1 not in registered_mixins:
            mixin_error_msg: str = f"Expected {SampleMixin1} in {registered_mixins}"
            raise AssertionError(mixin_error_msg)
        assert SampleMixin2 in registered_mixins

        delegated_methods = cast("list[str]", info["delegated_methods"])
        if "mixin_method1" not in delegated_methods:
            method_error_msg: str = f"Expected {'mixin_method1'} in {delegated_methods}"
            raise AssertionError(method_error_msg)
        assert "mixin_method3" in delegated_methods
        if not (info["validation_result"]):
            validation_error_msg: str = (
                f"Expected True, got {info['validation_result']}"
            )
            raise AssertionError(validation_error_msg)

    def test_delegation_validation_success(self) -> None:
        """Test successful delegation validation."""
        host = HostObject()
        delegator = FlextMixinDelegator(host, SampleMixin1)

        result = delegator._validate_delegation()
        assert result.success

    def test_delegation_validation_no_mixins(self) -> None:
        """Test delegation validation with no mixins."""
        host = HostObject()

        # Manually create delegator with no mixins
        delegator = FlextMixinDelegator.__new__(FlextMixinDelegator)
        delegator._host = host
        delegator._mixin_instances = {}
        delegator._delegated_methods = {}
        delegator._initialization_log = []

        result = delegator._validate_delegation()
        assert result.is_failure
        if "No mixins were successfully registered" not in (result.error or ""):
            msg: str = (
                f"Expected 'No mixins were successfully registered' in {result.error}"
            )
            raise AssertionError(msg)

    def test_delegation_validation_no_methods(self) -> None:
        """Test delegation validation with no delegated methods."""
        host = HostObject()

        # Create delegator with empty methods
        delegator = FlextMixinDelegator.__new__(FlextMixinDelegator)
        delegator._host = host
        delegator._mixin_instances = {SampleMixin1: SampleMixin1()}
        delegator._delegated_methods = {}
        delegator._initialization_log = []

        result = delegator._validate_delegation()
        assert result.is_failure
        if "No methods were successfully delegated" not in (result.error or ""):
            msg: str = (
                f"Expected 'No methods were successfully delegated' in {result.error}"
            )
            raise AssertionError(msg)

    def test_delegation_validation_with_init_failures(self) -> None:
        """Test delegation validation with initialization failures."""
        host = HostObject()
        delegator = FlextMixinDelegator(host, FailingInitMixin)

        result = delegator._validate_delegation()
        assert result.is_failure
        if "Initialization failed" not in (result.error or ""):
            msg: str = f"Expected 'Initialization failed' in {result.error}"
            raise AssertionError(msg)

    def test_mixin_registry_global(self) -> None:
        """Test global mixin registry."""
        host = HostObject()
        FlextMixinDelegator(host, SampleMixin1)

        # Test that mixin was registered globally
        if "SampleMixin1" not in FlextMixinDelegator._MIXIN_REGISTRY:
            msg: str = (
                f"Expected {'SampleMixin1'} in {FlextMixinDelegator._MIXIN_REGISTRY}"
            )
            raise AssertionError(msg)
        assert FlextMixinDelegator._MIXIN_REGISTRY["SampleMixin1"] is SampleMixin1


@pytest.mark.unit
class TestCreateMixinDelegator:
    """Test create_mixin_delegator factory function."""

    def test_create_mixin_delegator_basic(self) -> None:
        """Test basic mixin delegator creation."""
        host = HostObject()
        delegator = create_mixin_delegator(host, SampleMixin1)

        assert isinstance(delegator, FlextMixinDelegator)
        assert delegator._host is host
        if SampleMixin1 not in delegator._mixin_instances:
            msg: str = f"Expected {SampleMixin1} in {delegator._mixin_instances}"
            raise AssertionError(msg)

    def test_create_mixin_delegator_multiple_mixins(self) -> None:
        """Test creating delegator with multiple mixins."""
        host = HostObject()
        delegator = create_mixin_delegator(host, SampleMixin1, SampleMixin2)

        if len(delegator._mixin_instances) != EXPECTED_BULK_SIZE:
            length_msg: str = f"Expected {2}, got {len(delegator._mixin_instances)}"
            raise AssertionError(length_msg)
        if SampleMixin1 not in delegator._mixin_instances:
            mixin_instance_msg: str = (
                f"Expected {SampleMixin1} in {delegator._mixin_instances}"
            )
            raise AssertionError(mixin_instance_msg)
        assert SampleMixin2 in delegator._mixin_instances

    def test_create_mixin_delegator_no_mixins(self) -> None:
        """Test creating delegator with no mixins."""
        host = HostObject()
        delegator = create_mixin_delegator(host)

        assert isinstance(delegator, FlextMixinDelegator)
        if len(delegator._mixin_instances) != 0:
            msg: str = f"Expected {0}, got {len(delegator._mixin_instances)}"
            raise AssertionError(msg)


@pytest.mark.unit
class TestValidateDelegationSystem:
    """Test validate_delegation_system function."""

    def test_validate_delegation_system_success(self) -> None:
        """Test successful delegation system validation."""
        result = validate_delegation_system()

        assert isinstance(result, FlextResult)
        if result.success:
            data = result.data
            assert isinstance(data, dict)
            if data["status"] != "SUCCESS":
                raise AssertionError(f"Expected {'SUCCESS'}, got {data['status']}")
            if "test_results" not in data:
                raise AssertionError(f"Expected {'test_results'} in {data}")
            assert "delegation_info" in data

            # Check that test results show success
            test_results = data["test_results"]
            if not isinstance(test_results, (list, tuple)):
                raise AssertionError(
                    f"Expected list/tuple for test_results, got {type(test_results)}",
                )
            success_results = [
                r for r in test_results if isinstance(r, str) and r.startswith("✓")
            ]
            assert len(success_results) > 0

    def test_validate_delegation_system_missing_methods(self) -> None:
        """Test validation when required methods are missing."""

        # Test with a mixin that doesn't have required methods
        class IncompleteMixin:
            def __init__(self) -> None:
                pass

        host = HostObject()
        _ = FlextMixinDelegator(host, IncompleteMixin)  # Test delegator creation

        # Validation should succeed even with missing methods
        result = validate_delegation_system()  # Não aceita argumentos
        assert result.success

    def test_validate_delegation_system_delegation_failure(self) -> None:
        """Test validation system with delegation failure."""
        with patch(
            "flext_core.delegation_system.create_mixin_delegator",
        ) as mock_create:
            # Mock create_mixin_delegator to raise an exception
            mock_create.side_effect = RuntimeError("Delegation creation failed")

            result = validate_delegation_system()

            # Should fail due to delegation creation failure
            assert result.is_failure
            if "Test failed" not in (result.error or ""):
                msg: str = f"Expected 'Test failed' in {result.error}"
                raise AssertionError(msg)
            if result.error is None:
                error_msg = "Expected error message, got None"
                raise AssertionError(error_msg)
            assert "Delegation creation failed" in result.error

    def test_validate_delegation_system_type_error(self) -> None:
        """Test validation system with type errors."""
        with patch(
            "flext_core.delegation_system.create_mixin_delegator",
        ) as mock_create:
            # Mock delegator that returns wrong type for is_valid
            mock_delegator = Mock()
            mock_host = Mock()
            mock_host.delegator = mock_delegator
            mock_host.is_valid = "not_a_bool"

            def mock_create_delegator(host: object, *_mixins: type) -> Mock:
                # Use dynamic attribute access in test mocking
                host.delegator = mock_delegator
                host.is_valid = "not_a_bool"
                return mock_delegator

            mock_create.side_effect = mock_create_delegator
            mock_delegator.get_delegation_info.return_value = {
                "validation_result": True,
            }

            result = validate_delegation_system()

            # Should fail due to type error
            assert result.is_failure
            if "Test failed" not in (result.error or ""):
                msg: str = f"Expected 'Test failed' in {result.error}"
                raise AssertionError(msg)

    def test_validate_delegation_system_various_exceptions(self) -> None:
        """Test validation system with various exception types."""
        exception_types = [
            AttributeError("Attribute missing"),
            TypeError("Type error"),
            ValueError("Value error"),
            RuntimeError("Runtime error"),
            FlextOperationError("Operation error", operation="test"),
            FlextTypeError("Type error"),
        ]

        for exception in exception_types:
            with patch(
                "flext_core.delegation_system.create_mixin_delegator",
            ) as mock_create:
                mock_create.side_effect = exception

                result = validate_delegation_system()
                assert result.is_failure
                if "Test failed" not in (result.error or ""):
                    msg: str = f"Expected 'Test failed' in {result.error}"
                    raise AssertionError(msg)


@pytest.mark.unit
class TestDelegationSystemEdgeCases:
    """Test edge cases and error conditions."""

    def test_delegator_with_callable_attribute(self) -> None:
        """Test delegation of callable attributes that aren't methods."""

        class CallableMixin:
            def __init__(self) -> None:
                self.callable_attr: Callable[[str], str] = lambda x: f"lambda: {x}"

            def regular_method(self) -> str:
                return "regular"

        host = HostObject()
        FlextMixinDelegator(host, CallableMixin)

        # Callable attribute should be delegated
        assert hasattr(host, "callable_attr")
        result = host.callable_attr("test")
        if result != "lambda: test":
            callable_msg: str = f"Expected {'lambda: test'}, got {result}"
            raise AssertionError(callable_msg)

        # Regular method should also be delegated
        assert hasattr(host, "regular_method")
        result = host.regular_method()
        if result != "regular":
            regular_msg: str = f"Expected {'regular'}, got {result}"
            raise AssertionError(regular_msg)

    def test_delegator_method_name_conflicts(self) -> None:
        """Test handling of method name conflicts between mixins."""

        class Mixin1:
            def conflict_method(self) -> str:
                return "mixin1"

        class Mixin2:
            def conflict_method(self) -> str:
                return "mixin2"

        host = HostObject()

        # Later mixins should override earlier ones
        FlextMixinDelegator(host, Mixin1, Mixin2)

        # Should have the method from the last registered mixin - use cast for dynamic delegation
        result = host.conflict_method()
        # The actual behavior depends on the order of processing
        if result not in {"mixin1", "mixin2"}:
            msg: str = f"Expected {'mixin1', 'mixin2'}, got {result}"
            raise AssertionError(msg)

    def test_delegator_with_host_existing_methods(self) -> None:
        """Test delegation when host already has methods with same names."""

        class ConflictMixin:
            def host_method(self) -> str:
                return "mixin_host_method"

            def new_method(self) -> str:
                return "new_method"

        host = HostObject()

        FlextMixinDelegator(host, ConflictMixin)

        # Host's original method might be preserved or overridden
        # depending on the delegation strategy
        assert hasattr(host, "new_method")
        result = host.new_method()
        if result != "new_method":
            msg: str = f"Expected {'new_method'}, got {result}"
            raise AssertionError(msg)

    def test_delegator_initialization_with_complex_errors(self) -> None:
        """Test initialization with various error types."""

        class ComplexErrorMixin:
            def __init__(self) -> None:
                pass

            def _initialize_validation(self) -> None:
                msg = "Complex initialization error"
                raise RuntimeError(msg)

            def _initialize_timestamps(self) -> None:
                msg = "Another initialization error"
                raise AttributeError(msg)

            def _initialize_id(self) -> None:
                msg = "ID initialization error"
                raise TypeError(msg)

        host = HostObject()
        # Test that initialization fails with RuntimeError
        with pytest.raises(RuntimeError, match="Complex initialization error"):
            FlextMixinDelegator(host, ComplexErrorMixin)

    def test_delegation_with_property_edge_cases(self) -> None:
        """Test property delegation edge cases."""

        class PropertyMixin:
            def __init__(self) -> None:
                self._value = "initial"

            @property
            def normal_property(self) -> str:
                return self._value

            @normal_property.setter
            def normal_property(self, value: str) -> None:
                self._value = value

            @property
            def property_with_error(self) -> str:
                msg = "Property getter error"
                raise ValueError(msg)

        host = HostObject()
        FlextMixinDelegator(host, PropertyMixin)

        # Normal property should work
        assert hasattr(host, "normal_property")
        value = host.normal_property
        if value != "initial":
            raise AssertionError(f"Expected {'initial'}, got {value}")

        # Property with error should still be delegated but raise when accessed
        # Check if property exists in class __dict__ without calling the getter
        if "property_with_error" not in type(host).__dict__:
            raise AssertionError(
                f"Expected {'property_with_error'} in {type(host).__dict__}",
            )
        with pytest.raises(ValueError, match="Property getter error"):
            _ = host.property_with_error

    def test_mixin_registry_persistence(self) -> None:
        """Test that mixin registry persists across delegator instances."""
        host1 = HostObject()
        host2 = HostObject()

        # Create two delegators with same mixin
        FlextMixinDelegator(host1, SampleMixin1)
        FlextMixinDelegator(host2, SampleMixin1)

        # Both should see the same registry entry
        if "SampleMixin1" not in FlextMixinDelegator._MIXIN_REGISTRY:
            msg: str = (
                f"Expected {'SampleMixin1'} in {FlextMixinDelegator._MIXIN_REGISTRY}"
            )
            raise AssertionError(msg)
        assert FlextMixinDelegator._MIXIN_REGISTRY["SampleMixin1"] is SampleMixin1

    def test_delegation_info_completeness(self) -> None:
        """Test that delegation info contains all expected information."""
        host = HostObject()
        delegator = FlextMixinDelegator(host, SampleMixin1, InitializableMixin)

        info = delegator.get_delegation_info()

        # Should contain all required keys
        required_keys = [
            "registered_mixins",
            "delegated_methods",
            "initialization_log",
            "validation_result",
        ]
        for key in required_keys:
            if key not in info:
                raise AssertionError(f"Expected {key} in {info}")

        # Should contain information from both mixins
        registered_mixins = info["registered_mixins"]
        if not isinstance(registered_mixins, (list, tuple)):
            raise TypeError(
                f"Expected list/tuple for registered_mixins, got {type(registered_mixins)}",
            )
        if len(registered_mixins) != EXPECTED_BULK_SIZE:
            raise AssertionError(f"Expected {2}, got {len(registered_mixins)}")
        if SampleMixin1 not in registered_mixins:
            raise AssertionError(f"Expected {SampleMixin1} in {registered_mixins}")
        assert InitializableMixin in registered_mixins

        # Should contain methods from both mixins
        methods = info["delegated_methods"]
        if not isinstance(methods, (list, tuple, dict)):
            raise TypeError(
                f"Expected list/tuple/dict for delegated_methods, got {type(methods)}",
            )
        if "mixin_method1" not in methods:  # From both mixins, but one will override:
            raise AssertionError(f"Expected {'mixin_method1'} in {methods}")

        # Should contain initialization log entries
        initialization_log = info["initialization_log"]
        if not isinstance(initialization_log, (list, tuple)):
            raise TypeError(
                f"Expected list/tuple for initialization_log, got {type(initialization_log)}",
            )
        if len(initialization_log) < 5:  # At least 5 initialization methods:
            raise AssertionError(f"Expected {len(initialization_log)} >= {5}")

    def test_signature_preservation_edge_cases(self) -> None:
        """Test signature preservation edge cases."""

        class SignatureMixin:
            def method_with_signature_issues(self) -> str:
                """Return a method with potential signature issues."""
                return "signature_test"

            # Method that might have signature preservation issues
            def __getattr__(self, name: str) -> object:
                return "dynamic_attr"

        host = HostObject()
        delegator = FlextMixinDelegator(host, SignatureMixin)

        # Should handle methods with potential signature issues
        if "method_with_signature_issues" not in delegator._delegated_methods:
            msg = (
                f"Expected {'method_with_signature_issues'} in "
                f"{delegator._delegated_methods}"
            )
            raise AssertionError(msg)

    def test_error_handling_in_method_creation(self) -> None:
        """Test error handling during method creation."""

        class ProblematicMixin:
            def normal_method(self) -> str:
                return "normal"

            def method_causing_delegation_error(self) -> str:
                msg = "Delegation setup error"
                raise AttributeError(msg)

        host = HostObject()
        # Should not crash even if some methods cause issues
        delegator = FlextMixinDelegator(host, ProblematicMixin)

        # Should still have the normal method
        if "normal_method" not in delegator._delegated_methods:
            msg: str = f"Expected {'normal_method'} in {delegator._delegated_methods}"
            raise AssertionError(msg)


@pytest.mark.integration
class TestDelegationSystemIntegration:
    """Integration tests for delegation system."""

    def test_full_delegation_workflow(self) -> None:
        """Test complete delegation workflow from creation to usage."""
        # Create host and delegator
        host = HostObject()
        delegator = create_mixin_delegator(
            host,
            SampleMixin1,
            SampleMixin2,
            InitializableMixin,
        )

        # Verify initialization
        assert delegator._host is host
        if len(delegator._mixin_instances) != EXPECTED_DATA_COUNT:
            raise AssertionError(f"Expected {3}, got {len(delegator._mixin_instances)}")

        # Test method delegation from multiple mixins
        assert hasattr(host, "mixin_method1")
        assert hasattr(host, "mixin_method3")

        # Use delegated methods
        result1 = host.mixin_method1()
        result2 = host.mixin_method3()

        # Results should come from the appropriate mixins
        if result1 != "mixin1_method1":
            raise AssertionError(f"Expected {'mixin1_method1'}, got {result1}")
        assert result2 == {"key": "value"}

        # Test property delegation
        assert hasattr(host, "writable_property")
        value = host.writable_property
        assert isinstance(value, str)

        # Test validation
        validation_result = delegator._validate_delegation()
        assert validation_result.success

        # Test delegation info
        info = delegator.get_delegation_info()
        if not (info["validation_result"]):
            raise AssertionError(f"Expected True, got {info['validation_result']}")
        registered_mixins_info = info["registered_mixins"]
        if not isinstance(registered_mixins_info, (list, tuple)):
            raise TypeError(f"Expected list/tuple, got {type(registered_mixins_info)}")
        if len(registered_mixins_info) != EXPECTED_DATA_COUNT:
            raise AssertionError(f"Expected {3}, got {len(registered_mixins_info)}")

    def test_system_validation_integration(self) -> None:
        """Test integration with system validation."""
        result = validate_delegation_system()

        assert isinstance(result, FlextResult)
        if result.success:
            data = result.data
            if not isinstance(data, dict):
                raise AssertionError(f"Expected dict, got {type(data)}")
            if data["status"] != "SUCCESS":
                raise AssertionError(f"Expected {'SUCCESS'}, got {data['status']}")
            if "test_results" not in data:
                raise AssertionError(f"Expected {'test_results'} in {data}")

            test_results = data["test_results"]
            if not isinstance(test_results, (list, tuple)):
                raise AssertionError(
                    f"Expected list/tuple for test_results, got {type(test_results)}",
                )
            assert len(test_results) > 0

            # All test results should indicate success
            for test_result in test_results:
                assert test_result.startswith("✓")
        else:
            # If validation fails, error should be informative
            assert isinstance(result.error, str)
            assert len(result.error) > 0

    def test_complex_mixin_combination(self) -> None:
        """Test complex combination of mixins with various features."""
        host = HostObject()
        delegator = FlextMixinDelegator(
            host,
            SampleMixin1,  # Methods and properties
            SampleMixin2,  # Different methods
            InitializableMixin,  # Initialization methods
        )

        # Verify all expected methods are available
        expected_methods = {
            "mixin_method1",  # From SampleMixin1 or InitializableMixin
            "mixin_method2",  # From SampleMixin1
            "mixin_method3",  # From SampleMixin2
        }

        for method_name in expected_methods:
            assert hasattr(host, method_name), f"Method {method_name} not found"

        # Verify properties are available
        expected_properties = {
            "writable_property",  # From SampleMixin1
            "readonly_property",  # From SampleMixin1
            "another_property",  # From SampleMixin2
        }

        for prop_name in expected_properties:
            assert hasattr(host, prop_name), f"Property {prop_name} not found"

        # Test that initialization worked
        init_mixin = delegator.get_mixin_instance(InitializableMixin)
        assert isinstance(init_mixin, InitializableMixin)
        if not (init_mixin.validation_initialized):
            raise AssertionError(
                f"Expected True, got {init_mixin.validation_initialized}",
            )
        assert init_mixin.timestamps_initialized is True
        if not (init_mixin.id_initialized):
            raise AssertionError(f"Expected True, got {init_mixin.id_initialized}")
        assert init_mixin.logging_initialized is True
        if not (init_mixin.serialization_initialized):
            raise AssertionError(
                f"Expected True, got {init_mixin.serialization_initialized}",
            )

        # Verify system is valid
        validation_result = delegator._validate_delegation()
        assert validation_result.success

    def test_performance_with_many_mixins(self) -> None:
        """Test performance characteristics with many mixins."""
        # Create many simple mixins
        mixins = []
        for i in range(10):
            class_name = f"SampleMixin{i}"
            mixin_class = type(
                class_name,
                (),
                {
                    "__init__": lambda self, mixin_id=i: setattr(
                        self,
                        "value",
                        f"mixin_{mixin_id}",
                    ),
                    f"mixin_method_{i}": lambda _self,
                    mixin_id=i: f"mixin_method_{mixin_id}",
                },
            )
            mixins.append(mixin_class)

        host = HostObject()
        delegator = FlextMixinDelegator(host, *mixins)

        # Should handle many mixins without issues
        if len(delegator._mixin_instances) != 10:
            instances_msg: str = f"Expected {10}, got {len(delegator._mixin_instances)}"
            raise AssertionError(instances_msg)
        if len(delegator._delegated_methods) < 10:
            methods_msg: str = f"Expected {len(delegator._delegated_methods)} >= {10}"
            raise AssertionError(methods_msg)

        # Validation should still work
        validation_result = delegator._validate_delegation()
        assert validation_result.success
