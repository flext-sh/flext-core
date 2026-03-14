"""Tests for flext_infra.codegen module initialization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

import flext_infra.codegen as codegen_module
from flext_infra.codegen import FlextInfraCodegenLazyInit
from flext_tests import tm


def test_codegen_getattr_raises_attribute_error() -> None:
    """Test that accessing nonexistent attribute raises AttributeError."""
    with pytest.raises(AttributeError):
        codegen_module.nonexistent_xyz_attribute


def test_codegen_dir_returns_all_exports() -> None:
    """Test that dir() returns all exported attributes."""
    exports = dir(codegen_module)
    tm.that("FlextInfraCodegenLazyInit" in exports, eq=True)


def test_codegen_lazy_imports_work() -> None:
    """Test that lazy imports work correctly."""
    tm.that(type(FlextInfraCodegenLazyInit).__name__, eq="type")
    tm.that(hasattr(FlextInfraCodegenLazyInit, "run"), eq=True)


__all__: list[str] = []
