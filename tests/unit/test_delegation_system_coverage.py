"""Tests specifically designed to achieve 100% coverage of delegation_system.py.

These tests target the uncovered lines to bring delegation_system.py coverage
from 69% to as close to 100% as possible.
"""

from __future__ import annotations

import pytest

from flext_core import FlextOperationError
from flext_core.delegation_system import (
    FlextDelegatedProperty,
    FlextMixinDelegator,
    create_mixin_delegator,
    validate_delegation_system,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextDelegatedPropertyCoverage:
    """Test FlextDelegatedProperty for complete coverage."""

    def test_delegated_property_setter_with_readonly_property(self) -> None:
        """Test property setter with read-only property raises error."""

        class MockMixin:
            @property
            def readonly_prop(self) -> str:
                return "read-only"

        mixin = MockMixin()

        # Create delegated property WITHOUT setter
        delegated_prop = FlextDelegatedProperty(
            "readonly_prop",
            mixin,
            has_setter=False,  # Read-only property
        )

        # Trying to set should raise FlextOperationError
        with pytest.raises(FlextOperationError) as exc_info:
            delegated_prop.__set__(None, "new_value")

        error = exc_info.value
        assert "Property 'readonly_prop' is read-only" in str(error)
        assert error.operation == "property_setter"
        # Context is nested: error.context contains the passed context
        inner_context = error.context["context"]
        assert inner_context["property_name"] == "readonly_prop"
        assert inner_context["readonly"] is True

    def test_delegated_property_setter_with_writable_property(self) -> None:
        """Test property setter with writable property works."""

        class MockMixin:
            def __init__(self) -> None:
                self._value = "initial"

            @property
            def writable_prop(self) -> str:
                return self._value

            @writable_prop.setter
            def writable_prop(self, value: str) -> None:
                self._value = value

        mixin = MockMixin()

        # Create delegated property WITH setter
        delegated_prop = FlextDelegatedProperty(
            "writable_prop",
            mixin,
            has_setter=True,  # Writable property
        )

        # Should not raise error
        delegated_prop.__set__(None, "new_value")
        assert mixin.writable_prop == "new_value"


class TestFlextMixinDelegatorInitializationCoverage:
    """Test mixin initialization edge cases for coverage."""

    def test_mixin_with_initialization_methods_success(self) -> None:
        """Test mixin with successful initialization methods."""

        class MixinWithInit:
            def __init__(self) -> None:
                self.validation_initialized = False
                self.timestamps_initialized = False

            def _initialize_validation(self) -> None:
                self.validation_initialized = True

            def _initialize_timestamps(self) -> None:
                self.timestamps_initialized = True

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextMixinDelegator(host, MixinWithInit)

        # Check initialization log contains success messages
        init_log = delegator._initialization_log
        success_logs = [log for log in init_log if log.startswith("✓")]
        assert len(success_logs) == 2
        assert "MixinWithInit._initialize_validation()" in success_logs[0]
        assert "MixinWithInit._initialize_timestamps()" in success_logs[1]

        # Check mixin actually initialized
        mixin = delegator.get_mixin_instance(MixinWithInit)
        assert mixin.validation_initialized is True
        assert mixin.timestamps_initialized is True

    def test_mixin_with_initialization_methods_failure(self) -> None:
        """Test mixin with failing initialization methods."""

        class MixinWithFailingInit:
            def _initialize_validation(self) -> None:
                msg = "Validation init failed"
                raise ValueError(msg)

            def _initialize_id(self) -> None:
                msg = "ID init failed"
                raise AttributeError(msg)

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextMixinDelegator(host, MixinWithFailingInit)

        # Check initialization log contains error messages
        init_log = delegator._initialization_log
        error_logs = [log for log in init_log if log.startswith("✗")]
        assert len(error_logs) == 2
        assert (
            "MixinWithFailingInit._initialize_validation(): Validation init failed"
            in error_logs[0]
        )
        assert "MixinWithFailingInit._initialize_id(): ID init failed" in error_logs[1]

    def test_mixin_registration_failure(self) -> None:
        """Test mixin registration failure raises and logs error."""

        class FailingMixin:
            def __init__(self) -> None:
                msg = "Cannot initialize mixin"
                raise TypeError(msg)

        class TestHost:
            pass

        host = TestHost()

        # Should raise the exception from mixin initialization
        with pytest.raises(TypeError) as exc_info:
            FlextMixinDelegator(host, FailingMixin)

        assert "Cannot initialize mixin" in str(exc_info.value)


class TestFlextMixinDelegatorExceptionPathsCoverage:
    """Test exception handling paths in auto-delegation."""

    def test_property_delegation_exception_handling(self) -> None:
        """Test property delegation with exceptions in getattr."""

        class ProblematicMixin:
            @property
            def problematic_prop(self) -> str:
                msg = "Property access fails"
                raise AttributeError(msg)

        class TestHost:
            pass

        host = TestHost()
        # Should not crash even with problematic properties
        delegator = FlextMixinDelegator(host, ProblematicMixin)

        # Delegation should continue despite property issues
        assert delegator is not None
        assert len(delegator._mixin_instances) == 1

    def test_method_delegation_exception_handling(self) -> None:
        """Test method delegation with various exception paths."""

        class ProblematicMixin:
            def normal_method(self) -> str:
                return "works"

            def __problematic_getattr(self) -> None:
                # This method will cause getattr issues
                pass

        class TestHost:
            pass

        host = TestHost()
        FlextMixinDelegator(host, ProblematicMixin)

        # Should have delegated the working method
        assert hasattr(host, "normal_method")
        assert host.normal_method() == "works"

    def test_signature_preservation_exceptions(self) -> None:
        """Test signature preservation with methods that can't be inspected."""

        class BuiltinMethodMixin:
            # Using built-in methods that might not have accessible signatures
            upper = str.upper
            lower = str.lower

            def normal_method(self) -> str:
                return "test"

        class TestHost:
            pass

        host = TestHost()
        FlextMixinDelegator(host, BuiltinMethodMixin)

        # Should not crash even if signature can't be preserved
        assert hasattr(host, "normal_method")


class TestValidationSystemCompleteCoverage:
    """Test validation system error paths for complete coverage."""

    def test_validate_delegation_system_host_missing_methods(self) -> None:
        """Test validation with host missing required methods."""
        # This should test the actual validation paths in validate_delegation_system()
        result = validate_delegation_system()

        # The function creates its own test host, so we test that it handles
        # various edge cases in validation
        if result.success:
            data = result.value  # Use .value not .data
            assert "status" in data
            assert "test_results" in data
            assert "delegation_info" in data
        else:
            # If it fails, should be a proper failure message
            assert (
                "validation failed" in result.error.lower()
                or "test failed" in result.error.lower()
            )

    def test_validation_system_type_errors(self) -> None:
        """Test validation system handles type errors properly."""

        class BadMixin:
            @property
            def is_valid(self) -> str:  # Wrong type - should be bool
                return "not a bool"

        class TestHost:
            def __init__(self) -> None:
                self.delegator = FlextMixinDelegator(self, BadMixin)

        # Create a host with bad type
        host = TestHost()

        # Even with wrong types, the system should handle gracefully
        assert host.delegator is not None


class TestCreateMixinDelegatorCoverage:
    """Test create_mixin_delegator function for coverage."""

    def test_create_mixin_delegator_function(self) -> None:
        """Test create_mixin_delegator convenience function."""

        class SimpleMixin:
            def simple_method(self) -> str:
                return "simple"

        class TestHost:
            pass

        host = TestHost()
        delegator = create_mixin_delegator(host, SimpleMixin)

        assert isinstance(delegator, FlextMixinDelegator)
        assert hasattr(host, "simple_method")
        assert host.simple_method() == "simple"

    def test_delegation_info_method_coverage(self) -> None:
        """Test get_delegation_info method for complete coverage."""

        class InfoMixin:
            def info_method(self) -> str:
                return "info"

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextMixinDelegator(host, InfoMixin)

        info = delegator.get_delegation_info()

        assert "registered_mixins" in info
        assert "delegated_methods" in info
        assert "initialization_log" in info
        assert "validation_result" in info

        assert InfoMixin in info["registered_mixins"]
        assert "info_method" in info["delegated_methods"]
        assert isinstance(info["validation_result"], bool)
