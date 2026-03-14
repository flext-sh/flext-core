"""Tests for flext_infra.core module initialization.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

import flext_infra.core as core_module
from flext_infra.core import FlextInfraBaseMkValidator
from flext_tests import tm


class TestCoreModuleInit:
    """Test core module lazy loading and exports."""

    def test_core_getattr_raises_attribute_error(self) -> None:
        """Test that accessing nonexistent attribute raises AttributeError."""
        with pytest.raises(AttributeError):
            core_module.nonexistent_xyz_attribute

    def test_core_dir_returns_all_exports(self) -> None:
        """Test that dir() returns all exported attributes."""
        exports = dir(core_module)
        tm.that("FlextInfraBaseMkValidator" in exports, eq=True)
        tm.that("FlextInfraInventoryService" in exports, eq=True)
        tm.that("FlextInfraSkillValidator" in exports, eq=True)
        tm.that("FlextInfraStubSupplyChain" in exports, eq=True)
        tm.that("FlextInfraTextPatternScanner" in exports, eq=True)

    def test_core_lazy_imports_work(self) -> None:
        """Test that lazy imports resolve to real classes."""
        tm.that(FlextInfraBaseMkValidator, none=False)
        tm.that(hasattr(FlextInfraBaseMkValidator, "validate"), eq=True)


__all__: list[str] = []
