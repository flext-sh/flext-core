"""Simplified tests to achieve higher coverage of delegation_system.py.

Focus on covering the specific missing lines with simple, direct tests.
"""

from __future__ import annotations

import pytest

from flext_core import (
    FlextMixinDelegator,
    FlextOperationError,
    FlextTypeError,
    create_mixin_delegator,
)
from flext_core.delegation_system import (
    _validate_delegation_info,
    _validate_delegation_methods,
    _validate_method_functionality,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class ProblematicMixin:
    """Mixin that causes various delegation issues."""

    def __init__(self) -> None:
        # This will cause exception during some delegation scenarios
        pass

    @property
    def problematic_property(self) -> str:
        return "problematic"


class TestDelegationSystemCoverage:
    """Tests targeting uncovered lines in delegation_system.py."""

    def test_validation_function_missing_is_valid(self) -> None:
        """Test _validate_delegation_methods with missing is_valid (line 315)."""

        class HostWithoutIsValid:
            def __init__(self) -> None:
                self.validation_errors = []
                self.has_validation_errors = lambda: False
                self.to_dict_basic = dict
                self.delegator = create_mixin_delegator(self)

        host = HostWithoutIsValid()
        test_results: list[str] = []

        with pytest.raises(FlextOperationError) as exc_info:
            _validate_delegation_methods(host, test_results)

        assert "is_valid property not delegated" in str(exc_info.value)

    def test_validation_function_missing_validation_errors(self) -> None:
        """Test _validate_delegation_methods with missing validation_errors (line 317)."""

        class HostWithoutValidationErrors:
            def __init__(self) -> None:
                self.is_valid = True
                self.has_validation_errors = lambda: False
                self.to_dict_basic = dict
                self.delegator = create_mixin_delegator(self)

        host = HostWithoutValidationErrors()
        test_results: list[str] = []

        with pytest.raises(FlextOperationError) as exc_info:
            _validate_delegation_methods(host, test_results)

        assert "validation_errors property not delegated" in str(exc_info.value)

    def test_validation_function_missing_has_validation_errors(self) -> None:
        """Test _validate_delegation_methods with missing has_validation_errors (line 319)."""

        class HostWithoutHasValidationErrors:
            def __init__(self) -> None:
                self.is_valid = True
                self.validation_errors = []
                self.to_dict_basic = dict
                self.delegator = create_mixin_delegator(self)

        host = HostWithoutHasValidationErrors()
        test_results: list[str] = []

        with pytest.raises(FlextOperationError) as exc_info:
            _validate_delegation_methods(host, test_results)

        assert "has_validation_errors method not delegated" in str(exc_info.value)

    def test_validation_function_missing_to_dict_basic(self) -> None:
        """Test _validate_delegation_methods with missing to_dict_basic (line 324)."""

        class HostWithoutToDictBasic:
            def __init__(self) -> None:
                self.is_valid = True
                self.validation_errors = []
                self.has_validation_errors = lambda: False
                self.delegator = create_mixin_delegator(self)

        host = HostWithoutToDictBasic()
        test_results: list[str] = []

        with pytest.raises(FlextOperationError) as exc_info:
            _validate_delegation_methods(host, test_results)

        assert "to_dict_basic method not delegated" in str(exc_info.value)

    def test_method_functionality_wrong_type_is_valid(self) -> None:
        """Test _validate_method_functionality with wrong is_valid type (line 337)."""

        class HostWithWrongIsValidType:
            def __init__(self) -> None:
                self.is_valid = "not a bool"  # Should be bool
                self.delegator = create_mixin_delegator(self)

        host = HostWithWrongIsValidType()
        test_results: list[str] = []

        with pytest.raises(FlextTypeError) as exc_info:
            _validate_method_functionality(host, test_results)

        assert "is_valid should return bool" in str(exc_info.value)

    def test_delegation_info_missing_delegator(self) -> None:
        """Test _validate_delegation_info with missing delegator (line 352)."""

        class HostWithoutDelegator:
            pass

        host = HostWithoutDelegator()
        test_results: list[str] = []

        with pytest.raises(FlextOperationError) as exc_info:
            _validate_delegation_info(host, test_results)

        assert "Host must have delegator attribute" in str(exc_info.value)

    def test_delegation_info_missing_get_delegation_info(self) -> None:
        """Test _validate_delegation_info with missing get_delegation_info (line 358)."""

        class FakeDelegator:
            pass

        class HostWithBadDelegator:
            def __init__(self) -> None:
                self.delegator = FakeDelegator()

        host = HostWithBadDelegator()
        test_results: list[str] = []

        with pytest.raises(FlextOperationError) as exc_info:
            _validate_delegation_info(host, test_results)

        assert "Delegator must have get_delegation_info method" in str(exc_info.value)

    def test_delegation_info_validation_failure(self) -> None:
        """Test _validate_delegation_info with validation failure (line 367)."""

        class FailingDelegator:
            def get_delegation_info(self) -> dict[str, object]:
                return {"validation_result": False}  # Validation fails

        class HostWithFailingValidation:
            def __init__(self) -> None:
                self.delegator = FailingDelegator()

        host = HostWithFailingValidation()
        test_results: list[str] = []

        with pytest.raises(FlextOperationError) as exc_info:
            _validate_delegation_info(host, test_results)

        assert "Delegation validation should pass" in str(exc_info.value)

    def test_empty_mixin_handling(self) -> None:
        """Test empty mixin to verify no methods delegated."""

        class EmptyMixin:
            pass

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextMixinDelegator(host, EmptyMixin)

        # Should have no delegated methods
        assert len(delegator._delegated_methods) == 0
        assert EmptyMixin in delegator._mixin_instances

    def test_private_method_not_delegated(self) -> None:
        """Test that private methods are not delegated."""

        class PrivateMethodMixin:
            def _private_method(self) -> str:
                return "private"

            def __magic_method__(self) -> str:
                return "magic"

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextMixinDelegator(host, PrivateMethodMixin)

        # Private methods should not be delegated
        assert "_private_method" not in delegator._delegated_methods
        assert "__magic_method__" not in delegator._delegated_methods
        assert len(delegator._delegated_methods) == 0

    def test_mixin_with_init_methods_success(self) -> None:
        """Test mixin with successful initialization methods."""

        class InitializableMixin:
            def __init__(self) -> None:
                self._initialized = False

            def _initialize_validation(self) -> None:
                self._initialized = True

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextMixinDelegator(host, InitializableMixin)

        # Should have successful initialization in log
        init_logs = delegator._initialization_log
        success_logs = [log for log in init_logs if log.startswith("✓")]
        assert len(success_logs) > 0

    def test_mixin_with_init_methods_failure(self) -> None:
        """Test mixin with failing initialization methods."""

        class FailingInitMixin:
            def __init__(self) -> None:
                pass

            def _initialize_validation(self) -> None:
                raise ValueError("Initialization failed")

        class TestHost:
            pass

        host = TestHost()
        delegator = FlextMixinDelegator(host, FailingInitMixin)

        # Should have failure in initialization log
        init_logs = delegator._initialization_log
        failure_logs = [log for log in init_logs if log.startswith("✗")]
        assert len(failure_logs) > 0
        assert "Initialization failed" in " ".join(init_logs)


class TestValidateSystemExceptionHandling:
    """Test exception handling in validate_delegation_system."""

    def test_attribute_error_handling(self) -> None:
        """Test AttributeError handling in validate_delegation_system."""
        from unittest.mock import patch

        from flext_core.delegation_system import validate_delegation_system

        with patch("flext_core.delegation_system._validate_delegation_methods") as mock_validate:
            mock_validate.side_effect = AttributeError("Attribute error")

            result = validate_delegation_system()

            assert result.is_failure
            assert "Delegation system validation failed" in result.error
            assert "Attribute error" in result.error

    def test_type_error_handling(self) -> None:
        """Test TypeError handling in validate_delegation_system."""
        from unittest.mock import patch

        from flext_core.delegation_system import validate_delegation_system

        with patch("flext_core.delegation_system._validate_method_functionality") as mock_validate:
            mock_validate.side_effect = TypeError("Type error")

            result = validate_delegation_system()

            assert result.is_failure
            assert "Delegation system validation failed" in result.error
            assert "Type error" in result.error

    def test_value_error_handling(self) -> None:
        """Test ValueError handling in validate_delegation_system."""
        from unittest.mock import patch

        from flext_core.delegation_system import validate_delegation_system

        with patch("flext_core.delegation_system._validate_delegation_info") as mock_validate:
            mock_validate.side_effect = ValueError("Value error")

            result = validate_delegation_system()

            assert result.is_failure
            assert "Delegation system validation failed" in result.error
            assert "Value error" in result.error

    def test_runtime_error_handling(self) -> None:
        """Test RuntimeError handling in validate_delegation_system."""
        from unittest.mock import patch

        from flext_core.delegation_system import validate_delegation_system

        with patch("flext_core.delegation_system._validate_delegation_methods") as mock_validate:
            mock_validate.side_effect = RuntimeError("Runtime error")

            result = validate_delegation_system()

            assert result.is_failure
            assert "Delegation system validation failed" in result.error
            assert "Runtime error" in result.error

    def test_flext_operation_error_handling(self) -> None:
        """Test FlextOperationError handling in validate_delegation_system."""
        from unittest.mock import patch

        from flext_core.delegation_system import validate_delegation_system

        with patch("flext_core.delegation_system._validate_delegation_methods") as mock_validate:
            mock_validate.side_effect = FlextOperationError("Operation error", operation="test")

            result = validate_delegation_system()

            assert result.is_failure
            assert "Delegation system validation failed" in result.error
            assert "Operation error" in result.error

    def test_flext_type_error_handling(self) -> None:
        """Test FlextTypeError handling in validate_delegation_system."""
        from unittest.mock import patch

        from flext_core.delegation_system import validate_delegation_system

        with patch("flext_core.delegation_system._validate_method_functionality") as mock_validate:
            mock_validate.side_effect = FlextTypeError("Type error")

            result = validate_delegation_system()

            assert result.is_failure
            assert "Delegation system validation failed" in result.error
            assert "Type error" in result.error
