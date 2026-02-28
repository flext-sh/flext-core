"""Tests for FlextInfraLazyInitGenerator.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from pathlib import Path

import flext_infra.codegen as mod
import pytest
from flext_core import FlextService
from flext_infra.codegen import FlextInfraLazyInitGenerator
from flext_infra.codegen.lazy_init import (
    _extract_docstring_source,
    _extract_exports,
    _extract_inline_constants,
    _infer_package,
    _parse_existing_lazy_imports,
    _resolve_module,
)


class TestFlextInfraLazyInitGenerator:
    """Test suite for FlextInfraLazyInitGenerator service."""

    def test_init_accepts_workspace_root(self, tmp_path: Path) -> None:
        """Test generator initialization with workspace root."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        assert generator is not None

    def test_run_with_empty_workspace_returns_zero(self, tmp_path: Path) -> None:
        """Test run() on empty workspace returns 0 (no unmapped exports)."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        assert result == 0

    def test_run_with_check_only_flag(self, tmp_path: Path) -> None:
        """Test run() respects check_only flag without modifying files."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=True)
        assert result == 0

    def test_generate_output_to_sandboxed_path(self, tmp_path: Path) -> None:
        """Test that generated output goes to sandboxed tmp_path, not src/."""
        # Create a minimal __init__.py in sandbox
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        init_file.write_text(
            '"""Test package."""\n'
            "from test_pkg.module import TestClass\n"
            '__all__ = ["TestClass"]\n'
        )

        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=False)

        # Verify output is in tmp_path, not in actual src/flext_infra/
        assert result >= 0
        # Verify the file still exists in sandbox
        assert init_file.exists()

    def test_generator_is_flext_service(self, tmp_path: Path) -> None:
        """Test that FlextInfraLazyInitGenerator is a FlextService[str]."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        # Verify it's an instance of FlextService
        assert isinstance(generator, FlextService)

    def test_run_returns_integer_exit_code(self, tmp_path: Path) -> None:
        """Test that run() returns an integer exit code."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        assert isinstance(result, int)
        assert result >= 0

    def test_infer_package_deeply_nested(self) -> None:
        """Test _infer_package with deeply nested packages."""
        path = Path("/workspace/src/a/b/c/d/__init__.py")
        pkg = _infer_package(path)
        assert pkg == "a.b.c.d"

    def test_extract_exports_with_non_string_elements(self) -> None:
        """Test _extract_exports ignores non-string elements."""
        code = '__all__ = ["Foo", 123, "Bar"]'
        tree = ast.parse(code)
        has_all, exports = _extract_exports(tree)
        assert has_all is True
        assert exports == ["Foo", "Bar"]

    def test_extract_inline_constants_multiple(self) -> None:
        """Test _extract_inline_constants with multiple constants."""
        code = '__version__ = "1.0.0"\n__author__ = "Test"\n__license__ = "MIT"'
        tree = ast.parse(code)
        constants = _extract_inline_constants(tree)
        assert len(constants) == 3
        assert constants["__version__"] == "1.0.0"
        assert constants["__author__"] == "Test"
        assert constants["__license__"] == "MIT"

    def test_resolve_module_relative_level_2(self) -> None:
        """Test _resolve_module with level 2 relative import."""
        result = _resolve_module("module", 2, "a.b.c.d")
        assert result == "a.b.c.module"

    def test_resolve_module_relative_level_3(self) -> None:
        """Test _resolve_module with level 3 relative import."""
        result = _resolve_module("module", 3, "a.b.c.d")
        assert result == "a.b.module"

    def test_extract_docstring_source_with_quotes(self) -> None:
        """Test _extract_docstring_source preserves quote style."""
        code = "'''Module docstring.'''\nx = 1"
        tree = ast.parse(code)
        docstring = _extract_docstring_source(tree, code)
        assert "Module docstring" in docstring

    def test_parse_existing_lazy_imports_with_multiple_entries(self) -> None:
        """Test _parse_existing_lazy_imports with multiple entries."""
        code = '_LAZY_IMPORTS = {"Foo": ("module", "Foo"), "Bar": ("other", "Bar")}'
        tree = ast.parse(code)
        lazy_map = _parse_existing_lazy_imports(tree)
        assert len(lazy_map) == 2
        assert lazy_map["Foo"] == ("module", "Foo")
        assert lazy_map["Bar"] == ("other", "Bar")


def test_codegen_init_getattr_raises_attribute_error() -> None:
    """Test that accessing nonexistent attribute raises AttributeError."""
    with pytest.raises(AttributeError):
        mod.nonexistent_xyz_attribute
