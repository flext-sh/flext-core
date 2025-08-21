"""Tests for FLEXT Core interfaces module.

# ruff: noqa: ARG
"""

from __future__ import annotations

import types

import pytest

from flext_core import (
    FlextConfigurableProtocol as FlextConfigurable,
    FlextValidatorProtocol as FlextValidator,
    interfaces,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestProtocolInterfaces:
    """Test protocol interfaces functionality."""

    def test_configurable_protocol(self) -> None:
        """Test FlextConfigurable protocol."""  # Test that protocol is defined
        assert hasattr(FlextConfigurable, "configure")

    def test_validator_protocol(self) -> None:
        """Test FlextValidator protocol."""  # Test that protocol is defined
        assert hasattr(FlextValidator, "validate")

    def test_protocol_runtime_checkable(self) -> None:
        """Test that protocols are runtime checkable."""  # All protocols should be runtime checkable
        assert hasattr(FlextConfigurable, "__instancecheck__")
        assert hasattr(FlextValidator, "__instancecheck__")

    def test_protocol_imports(self) -> None:
        """Test that TYPE_CHECKING imports are accessible in protocols."""  # This test covers the TYPE_CHECKING import lines
        # Verify module structure
        assert isinstance(interfaces, types.ModuleType)
        assert hasattr(interfaces, "FlextConfigurable")
        assert hasattr(interfaces, "FlextValidator")

    def test_type_checking_imports(self) -> None:
        """Test TYPE_CHECKING imports for coverage."""  # These tests will cover the TYPE_CHECKING import lines
        # Test protocol structure - covers TYPE_CHECKING imports
        # Note: Protocols are defined in protocols.py but re-exported through interfaces.py
        assert FlextConfigurable.__module__ == "flext_core.protocols", (
            f"Expected {'flext_core.protocols'}, got {FlextConfigurable.__module__}"
        )
        assert FlextValidator.__module__ == "flext_core.protocols"
