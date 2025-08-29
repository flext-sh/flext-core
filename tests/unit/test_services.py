"""Tests for FLEXT services module.

Unit tests validating service layer abstractions, service patterns,
and dependency injection for enterprise service architecture.
"""

from __future__ import annotations

import pytest

from flext_core import services

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestFlextServices:
    """Test FLEXT service layer abstractions."""

    def test_module_imports(self) -> None:
        """Test that services module imports correctly."""
        assert services is not None
