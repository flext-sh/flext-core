"""Tests for flext_infra.docs module initialization.

Tests lazy loading and __getattr__ fallthrough behavior.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

import flext_infra.docs as docs_module
from flext_infra.docs import (
    FlextInfraDocAuditor,
    FlextInfraDocBuilder,
    FlextInfraDocFixer,
    FlextInfraDocGenerator,
    FlextInfraDocValidator,
)
from flext_tests import tm


class TestFlextInfraDocs:
    """Tests for flext_infra.docs module."""

    def test_getattr_raises_attribute_error_for_unknown_symbol(self) -> None:
        """Test __getattr__ raises AttributeError for unknown attributes."""
        with pytest.raises(AttributeError):
            _ = docs_module.nonexistent_symbol_xyz

    def test_lazy_import_builder(self) -> None:
        """Test lazy import of FlextInfraDocBuilder."""
        tm.that(docs_module.FlextInfraDocBuilder is FlextInfraDocBuilder, eq=True)

    def test_lazy_import_fixer(self) -> None:
        """Test lazy import of FlextInfraDocFixer."""
        tm.that(docs_module.FlextInfraDocFixer is FlextInfraDocFixer, eq=True)

    def test_lazy_import_generator(self) -> None:
        """Test lazy import of FlextInfraDocGenerator."""
        tm.that(docs_module.FlextInfraDocGenerator is FlextInfraDocGenerator, eq=True)

    def test_lazy_import_validator(self) -> None:
        """Test lazy import of FlextInfraDocValidator."""
        tm.that(docs_module.FlextInfraDocValidator is FlextInfraDocValidator, eq=True)

    def test_lazy_import_auditor(self) -> None:
        """Test lazy import of FlextInfraDocAuditor."""
        tm.that(docs_module.FlextInfraDocAuditor is FlextInfraDocAuditor, eq=True)

    def test_dir_returns_all_exports(self) -> None:
        """Test dir() returns all exported symbols."""
        exports = dir(docs_module)
        tm.that("FlextInfraDocBuilder" in exports, eq=True)
        tm.that("FlextInfraDocFixer" in exports, eq=True)
        tm.that("FlextInfraDocGenerator" in exports, eq=True)
        tm.that("FlextInfraDocValidator" in exports, eq=True)
        tm.that("FlextInfraDocAuditor" in exports, eq=True)
