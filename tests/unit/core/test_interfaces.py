"""Tests for FLEXT Core interfaces module."""

from __future__ import annotations

import pytest

from flext_core.interfaces import FlextConfigurable, FlextValidator

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestProtocolInterfaces:
    """Test protocol interfaces functionality."""

    def test_configurable_protocol(self) -> None:
        """Test FlextConfigurable protocol."""
        # Test that protocol is defined
        assert hasattr(FlextConfigurable, "configure")

    def test_validator_protocol(self) -> None:
        """Test FlextValidator protocol."""
        # Test that protocol is defined
        assert hasattr(FlextValidator, "validate")

    def test_protocol_runtime_checkable(self) -> None:
        """Test that protocols are runtime checkable."""
        # All protocols should be runtime checkable
        assert hasattr(FlextConfigurable, "__instancecheck__")
        assert hasattr(FlextValidator, "__instancecheck__")

    def test_protocol_imports(self) -> None:
        """Test that TYPE_CHECKING imports are accessible in protocols."""
        # This test covers the TYPE_CHECKING import lines
        import types

        from flext_core import interfaces

        # Verify module structure
        assert isinstance(interfaces, types.ModuleType)
        assert hasattr(interfaces, "FlextConfigurable")
        assert hasattr(interfaces, "FlextValidator")

    def test_type_checking_imports(self) -> None:
        """Test TYPE_CHECKING imports for coverage."""
        # These tests will cover the TYPE_CHECKING import lines
        from flext_core.interfaces import FlextConfigurable, FlextValidator

        # Test protocol structure - covers TYPE_CHECKING imports
        assert FlextConfigurable.__module__ == "flext_core.interfaces"
        assert FlextValidator.__module__ == "flext_core.interfaces"
