"""Tests for flext_infra.core module initialization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import flext_infra.core as core_module
import pytest
from flext_infra.core import FlextInfraBaseMkValidator


def test_core_getattr_raises_attribute_error() -> None:
    """Test that accessing nonexistent attribute raises AttributeError (line 55)."""
    with pytest.raises(AttributeError):
        core_module.nonexistent_xyz_attribute


def test_core_dir_returns_all_exports() -> None:
    """Test that dir() returns all exported attributes."""
    exports = dir(core_module)
    assert "FlextInfraBaseMkValidator" in exports
    assert "FlextInfraInventoryService" in exports
    assert "FlextInfraSkillValidator" in exports
    assert "FlextInfraStubSupplyChain" in exports
    assert "FlextInfraTextPatternScanner" in exports
    assert "main" in exports


def test_core_lazy_imports_work() -> None:
    """Test that lazy imports work correctly."""
    assert FlextInfraBaseMkValidator is not None
    assert hasattr(FlextInfraBaseMkValidator, "validate")


__all__ = []
