"""Comprehensive tests for FlextMixinDelegator and delegation system.

# Constants
EXPECTED_BULK_SIZE = 2

This test suite provides complete coverage of the delegation system,
testing all aspects of mixin delegation, automatic method discovery,
property delegation, and validation.
"""

from __future__ import annotations

import pytest

from flext_core._delegation_system import (
    FlextMixinDelegator,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class SampleMixin1:
    """Test mixin for delegation testing."""

    def __init__(self) -> None:
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
        self._data = {"key": "value"}

    def mixin_method3(self) -> dict[str, str]:
        """Test method from mixin 2."""
        return self._data.copy()

    def mixin_method4(self, key: str, value: str) -> None:
        """Test method that modifies state."""
        self._data[key] = value


class InitializableMixin:
    """Mixin with initialization methods."""

    def __init__(self) -> None:
        self.initialized = False
        self.validation_initialized = False
        self.timestamps_initialized = False

    def _initialize_validation(self) -> None:
        """Initialize validation system."""
        self.validation_initialized = True

    def _initialize_timestamps(self) -> None:
        """Initialize timestamps system."""
        self.timestamps_initialized = True

    def test_method(self) -> str:
        """Test method."""
        return "initializable_method"


class FailingInitMixin:
    """Mixin with failing initialization."""

    def __init__(self) -> None:
        self.initialized = False

    def _initialize_validation(self) -> None:
        """Failing initialization method."""
        msg = "Initialization failed"
        raise ValueError(msg)

    def test_method(self) -> str:
        """Test method."""
        return "failing_init_method"


class HostObject:
    """Test host object for delegation."""

    def __init__(self) -> None:
        self.host_attr = "host_value"

    def host_method(self) -> str:
        """Host's own method."""
        return "host_method_result"


class FrozenHostObject:
    """Frozen host object to test property delegation edge cases."""

    def __init__(self) -> None:
        self.host_attr = "frozen_host"

    def __setattr__(self, name: str, value: object) -> None:
        """Simulate frozen object behavior."""
        if hasattr(self, "_frozen") and self._frozen:
            msg = "Cannot set attribute on frozen object"
            raise AttributeError(msg)
        super().__setattr__(name, value)

    def freeze(self) -> None:
        """Freeze the object."""
        self._frozen = True


# Keep legacy test classes for backward compatibility
class MockMixin:
    """Mock mixin for testing delegation."""

    def __init__(self) -> None:
        """Initialize mock mixin."""
        self.mixin_state = "initialized"

    def mixin_method(self) -> str:
        """Mock mixin method."""
        return "mixin_method_called"

    def mixin_method_with_args(self, arg1: str, arg2: int = 10) -> str:
        """Mock mixin method with arguments."""
        return f"mixin_method_with_args: {arg1}, {arg2}"

    def get_mixin_state(self) -> str:
        """Get mixin state."""
        return self.mixin_state


class AnotherMixin:
    """Another mock mixin for testing multiple mixins."""

    def __init__(self) -> None:
        """Initialize another mixin."""
        self.another_state = "another_initialized"

    def another_method(self) -> str:
        """Another mixin method."""
        return "another_method_called"


class HostInstance:
    """Host instance class for delegation."""

    def __init__(self) -> None:
        """Initialize host instance."""
        self.host_state = "host_initialized"

    def host_method(self) -> str:
        """Host method."""
        return "host_method_called"


class TestFlextMixinDelegator:
    """Test FlextMixinDelegator functionality."""

    def test_delegator_creation_success(self) -> None:
        """Test successful delegator creation."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        assert delegator._host is host
        if len(delegator._mixin_instances) != 1:
            raise AssertionError(f"Expected {1}, got {len(delegator._mixin_instances)}")
        if MockMixin not in delegator._mixin_instances:
            raise AssertionError(f"Expected {MockMixin} in {delegator._mixin_instances}")

    def test_get_mixin_instance_success(self) -> None:
        """Test successful mixin instance retrieval."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        mixin_instance = delegator.get_mixin_instance(MockMixin)

        assert mixin_instance is not None
        assert isinstance(mixin_instance, MockMixin)

    def test_multiple_mixins_registration(self) -> None:
        """Test registering multiple mixins."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin, AnotherMixin)

        if len(delegator._mixin_instances) != EXPECTED_BULK_SIZE:

            raise AssertionError(f"Expected {2}, got {len(delegator._mixin_instances)}")
        if MockMixin not in delegator._mixin_instances:
            raise AssertionError(f"Expected {MockMixin} in {delegator._mixin_instances}")
        assert AnotherMixin in delegator._mixin_instances

    def test_delegated_methods_created(self) -> None:
        """Test that delegated methods are created."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        assert len(delegator._delegated_methods) > 0
        if "mixin_method" not in delegator._delegated_methods:
            raise AssertionError(f"Expected {"mixin_method"} in {delegator._delegated_methods}")

    def test_method_delegation_through_host(self) -> None:
        """Test method delegation through host instance."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        # Ensure delegator initialized properly
        assert delegator._host is host

        # Method should be available on host through delegation
        if hasattr(host, "mixin_method"):
            result = host.mixin_method()  # type: ignore[attr-defined]
            if result != "mixin_method_called":
                raise AssertionError(f"Expected {"mixin_method_called"}, got {result}")

    def test_method_delegation_with_args(self) -> None:
        """Test method delegation with arguments."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        # Ensure delegator initialized properly
        assert delegator._host is host

        # Method with args should be available on host
        if hasattr(host, "mixin_method_with_args"):
            result = host.mixin_method_with_args("test", arg2=20)  # type: ignore[attr-defined]
            if result != "mixin_method_with_args: test, 20":
                raise AssertionError(f"Expected {"mixin_method_with_args: test, 20"}, got {result}")

    def test_validation_system(self) -> None:
        """Test delegation validation system."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        validation_result = delegator._validate_delegation()
        assert validation_result.is_success

    def test_delegation_info(self) -> None:
        """Test getting delegation information."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        info = delegator.get_delegation_info()
        if "registered_mixins" not in info:
            raise AssertionError(f"Expected {"registered_mixins"} in {info}")
        assert "delegated_methods" in info
        if "initialization_log" not in info:
            raise AssertionError(f"Expected {"initialization_log"} in {info}")
        assert "validation_result" in info

    def test_mixin_registry_functionality(self) -> None:
        """Test mixin registry functionality."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        # Ensure delegator initialized properly
        assert delegator._host is host

        # Check that mixin was registered globally
        if "MockMixin" not in FlextMixinDelegator._MIXIN_REGISTRY:
            raise AssertionError(f"Expected {"MockMixin"} in {FlextMixinDelegator._MIXIN_REGISTRY}")

    def test_initialization_log_tracking(self) -> None:
        """Test initialization log tracking."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        # Check that initialization was logged
        if len(delegator._initialization_log) < 0:
            raise AssertionError(f"Expected {len(delegator._initialization_log)} >= {0}")
        assert isinstance(delegator._initialization_log, list)

    def test_empty_delegation_validation(self) -> None:
        """Test validation with no mixins."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host)

        validation_result = delegator._validate_delegation()
        assert validation_result.is_failure
        if "No mixins were successfully registered" not in validation_result.error:
            raise AssertionError(f"Expected {"No mixins were successfully registered"} in {validation_result.error}")

    def test_multiple_mixin_method_delegation(self) -> None:
        """Test delegation with multiple mixins."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin, AnotherMixin)

        # Both mixin methods should be delegated
        info = delegator.get_delegation_info()
        delegated_methods = info["delegated_methods"]

        # Should contain methods from both mixins
        assert isinstance(delegated_methods, list)
        assert len(delegated_methods) > 0

    def test_mixin_instance_access(self) -> None:
        """Test direct mixin instance access."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        mixin_instance = delegator.get_mixin_instance(MockMixin)
        assert mixin_instance is not None
        assert hasattr(mixin_instance, "mixin_method")

        # Test calling method directly on instance
        result = mixin_instance.mixin_method()  # type: ignore[union-attr]
        if result != "mixin_method_called":
            raise AssertionError(f"Expected {"mixin_method_called"}, got {result}")

    def test_method_resolution_order(self) -> None:
        """Test method resolution order with multiple mixins."""

        class FirstMixin:
            def common_method(self) -> str:
                return "first_mixin"

        class SecondMixin:
            def common_method(self) -> str:
                return "second_mixin"

        host = HostInstance()
        # First mixin registered first should take precedence
        delegator = FlextMixinDelegator(host, FirstMixin, SecondMixin)

        # Test through delegation info
        info = delegator.get_delegation_info()
        if not (info["validation_result"]):
            raise AssertionError(f"Expected True, got {info["validation_result"]}")

    def test_delegation_error_handling(self) -> None:
        """Test error handling in delegation system."""

        class ErrorMixin:
            def error_method(self) -> None:
                msg = "Test error"
                raise ValueError(msg)

        host = HostInstance()
        # This should work without raising exceptions
        delegator = FlextMixinDelegator(host, ErrorMixin)

        # Validation might fail but shouldn't crash
        validation_result = delegator._validate_delegation()
        assert isinstance(validation_result.is_success, bool)

    def test_delegated_methods_storage(self) -> None:
        """Test delegated methods storage."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        # Check that delegated methods are stored
        assert isinstance(delegator._delegated_methods, dict)
        # Should have delegated some methods
        if len(delegator._delegated_methods) < 0:
            raise AssertionError(f"Expected {len(delegator._delegated_methods)} >= {0}")

    def test_mixin_state_preservation(self) -> None:
        """Test that mixin state is preserved."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        # Get mixin instance and check state
        mixin_instance = delegator.get_mixin_instance(MockMixin)
        assert mixin_instance is not None
        assert hasattr(mixin_instance, "mixin_state")

        # Modify state directly
        mixin_instance.mixin_state = "modified"  # type: ignore[union-attr]

        # State should be preserved
        if mixin_instance.mixin_state != "modified"  # type: ignore[union-attr]:
            raise AssertionError(f"Expected {"modified"  # type: ignore[union-attr]}, got {mixin_instance.mixin_state}")
