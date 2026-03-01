"""Tests for flext_infra.codegen module initialization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import flext_infra.codegen as codegen_module
import pytest
from flext_infra.codegen import FlextInfraCodegenLazyInit


def test_codegen_getattr_raises_attribute_error() -> None:
    """Test that accessing nonexistent attribute raises AttributeError (line 36)."""
    with pytest.raises(AttributeError):
        codegen_module.nonexistent_xyz_attribute


def test_codegen_dir_returns_all_exports() -> None:
    """Test that dir() returns all exported attributes."""
    exports = dir(codegen_module)
    assert "FlextInfraCodegenLazyInit" in exports


def test_codegen_lazy_imports_work() -> None:
    """Test that lazy imports work correctly."""
    assert FlextInfraCodegenLazyInit is not None
    assert hasattr(FlextInfraCodegenLazyInit, "run")


__all__ = []
