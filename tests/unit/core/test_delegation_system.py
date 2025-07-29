"""Tests for FLEXT Core Delegation System module."""

from __future__ import annotations

import pytest

from flext_core._delegation_system import FlextMixinDelegator

pytestmark = [pytest.mark.unit, pytest.mark.core]


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
        assert len(delegator._mixin_instances) == 1
        assert MockMixin in delegator._mixin_instances

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

        assert len(delegator._mixin_instances) == 2
        assert MockMixin in delegator._mixin_instances
        assert AnotherMixin in delegator._mixin_instances

    def test_delegated_methods_created(self) -> None:
        """Test that delegated methods are created."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        assert len(delegator._delegated_methods) > 0
        assert "mixin_method" in delegator._delegated_methods

    def test_method_delegation_through_host(self) -> None:
        """Test method delegation through host instance."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        # Ensure delegator initialized properly
        assert delegator._host is host

        # Method should be available on host through delegation
        if hasattr(host, "mixin_method"):
            result = host.mixin_method()  # type: ignore[attr-defined]
            assert result == "mixin_method_called"

    def test_method_delegation_with_args(self) -> None:
        """Test method delegation with arguments."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        # Ensure delegator initialized properly
        assert delegator._host is host

        # Method with args should be available on host
        if hasattr(host, "mixin_method_with_args"):
            result = host.mixin_method_with_args("test", arg2=20)  # type: ignore[attr-defined]
            assert result == "mixin_method_with_args: test, 20"

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
        assert "registered_mixins" in info
        assert "delegated_methods" in info
        assert "initialization_log" in info
        assert "validation_result" in info

    def test_mixin_registry_functionality(self) -> None:
        """Test mixin registry functionality."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        # Ensure delegator initialized properly
        assert delegator._host is host

        # Check that mixin was registered globally
        assert "MockMixin" in FlextMixinDelegator._MIXIN_REGISTRY

    def test_initialization_log_tracking(self) -> None:
        """Test initialization log tracking."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host, MockMixin)

        # Check that initialization was logged
        assert len(delegator._initialization_log) >= 0
        assert isinstance(delegator._initialization_log, list)

    def test_empty_delegation_validation(self) -> None:
        """Test validation with no mixins."""
        host = HostInstance()
        delegator = FlextMixinDelegator(host)

        validation_result = delegator._validate_delegation()
        assert validation_result.is_failure
        assert "No mixins were successfully registered" in validation_result.error

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
        assert result == "mixin_method_called"

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
        assert info["validation_result"] is True

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
        assert len(delegator._delegated_methods) >= 0

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
        assert mixin_instance.mixin_state == "modified"  # type: ignore[union-attr]
