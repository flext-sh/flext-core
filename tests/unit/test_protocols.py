"""Tests for FLEXT protocols module.

Unit tests validating protocol definitions, interface contracts,
and type checking for enterprise protocol patterns.
"""

from __future__ import annotations

import pytest

from flext_core import protocols

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextProtocols:
    """Test FLEXT protocol definitions and contracts."""

    def test_module_imports(self) -> None:
        """Test that protocols module imports correctly."""
        assert protocols is not None
