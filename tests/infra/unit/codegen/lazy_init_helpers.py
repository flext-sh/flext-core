"""Tests for lazy_init helper functions: package inference, docstrings, exports.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

from flext_infra.codegen._utilities import FlextInfraUtilitiesCodegen
from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
from flext_tests import tm

_read_existing_docstring: Callable[[Path], str] = getattr(
    FlextInfraCodegenLazyInit, "_read_existing_docstring"
)
_build_sibling_export_index: Callable[[Path, str], dict[str, tuple[str, str]]] = (
    getattr(FlextInfraCodegenLazyInit, "_build_sibling_export_index")
)


class TestInferPackage:
    """Test infer_package function."""

    def test_src_path(self) -> None:
        """Test inference from src/ path."""
        path = Path("/workspace/src/test_pkg/__init__.py")
        tm.that(FlextInfraUtilitiesCodegen.infer_package(path), eq="test_pkg")

    def test_deeply_nested_src_path(self) -> None:
        """Test inference from deeply nested src/ path."""
        path = Path("/workspace/src/a/b/c/d/__init__.py")
        tm.that(FlextInfraUtilitiesCodegen.infer_package(path), eq="a.b.c.d")

    def test_tests_path(self) -> None:
        """Test inference from tests/ path."""
        path = Path("/workspace/tests/unit/__init__.py")
        tm.that(FlextInfraUtilitiesCodegen.infer_package(path), eq="tests.unit")

    def test_without_src_directory(self) -> None:
        """Test when path doesn't contain /src/."""
        path = Path("/workspace/lib/test/__init__.py")
        tm.that(FlextInfraUtilitiesCodegen.infer_package(path), eq="")


class TestReadExistingDocstring:
    """Test _read_existing_docstring function."""

    def test_with_docstring(self, tmp_path: Path) -> None:
        """Test extracting docstring from existing __init__.py."""
        init_file = tmp_path / "__init__.py"
        init_file.write_text('"""Package docstring."""\nx = 1\n')
        result = _read_existing_docstring(init_file)
        tm.that(result, contains="Package docstring")

    def test_without_docstring(self, tmp_path: Path) -> None:
        """Test returns empty when no docstring exists."""
        init_file = tmp_path / "__init__.py"
        init_file.write_text("x = 1\ny = 2\n")
        result = _read_existing_docstring(init_file)
        tm.that(result, eq="")

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Test returns empty when file doesn't exist."""
        init_file = tmp_path / "__init__.py"
        result = _read_existing_docstring(init_file)
        tm.that(result, eq="")

    def test_with_syntax_error(self, tmp_path: Path) -> None:
        """Test returns empty on syntax error."""
        init_file = tmp_path / "__init__.py"
        init_file.write_text("invalid syntax ][")
        result = _read_existing_docstring(init_file)
        tm.that(result, eq="")

    def test_with_single_quotes(self, tmp_path: Path) -> None:
        """Test preserves single-quote docstring style."""
        init_file = tmp_path / "__init__.py"
        init_file.write_text("'''Module docstring.'''\nx = 1\n")
        result = _read_existing_docstring(init_file)
        tm.that(result, contains="Module docstring")


class TestBuildSiblingExportIndex:
    """Test _build_sibling_export_index function."""

    def test_with_all_exports(self, tmp_path: Path) -> None:
        """Test scanning sibling files with __all__."""
        (tmp_path / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["Foo", "Bar"]\n\nclass Foo: pass\nclass Bar: pass\n',
        )
        index = _build_sibling_export_index(tmp_path, "test_pkg")
        tm.that(index, contains="Foo")
        tm.that(index, contains="Bar")
        tm.that(index["Foo"], eq=("test_pkg.models", "Foo"))

    def test_without_all_falls_back_to_ast(self, tmp_path: Path) -> None:
        """Test scanning sibling files without __all__ uses AST."""
        (tmp_path / "service.py").write_text(
            "class PublicService:\n    pass\n\ndef public_func():\n    pass\n",
        )
        index = _build_sibling_export_index(tmp_path, "test_pkg")
        tm.that(index, contains="PublicService")
        tm.that(index, contains="public_func")

    def test_skips_init_and_main(self, tmp_path: Path) -> None:
        """Test that __init__.py and __main__.py are skipped."""
        (tmp_path / "__init__.py").write_text('__all__ = ["Init"]\n')
        (tmp_path / "__main__.py").write_text("def main(): pass\n")
        (tmp_path / "models.py").write_text(
            '__all__ = ["Model"]\nclass Model: pass\n',
        )
        index = _build_sibling_export_index(tmp_path, "test_pkg")
        tm.that(index, excludes="Init")
        tm.that(index, excludes="main")
        tm.that(index, contains="Model")

    def test_skips_private_files(self, tmp_path: Path) -> None:
        """Test that _private.py files are skipped."""
        (tmp_path / "_internal.py").write_text("class Internal: pass\n")
        (tmp_path / "public.py").write_text("class Public: pass\n")
        index = _build_sibling_export_index(tmp_path, "test_pkg")
        tm.that(index, excludes="Internal")
        tm.that(index, contains="Public")

    def test_skips_version_file(self, tmp_path: Path) -> None:
        """Test that __version__.py is skipped (handled separately)."""
        (tmp_path / "__version__.py").write_text('__version__ = "1.0.0"\n')
        (tmp_path / "models.py").write_text(
            '__all__ = ["Model"]\nclass Model: pass\n',
        )
        index = _build_sibling_export_index(tmp_path, "test_pkg")
        tm.that(index, excludes="__version__")
        tm.that(index, contains="Model")

    def test_handles_syntax_error_gracefully(self, tmp_path: Path) -> None:
        """Test that syntax errors in sibling files are skipped."""
        (tmp_path / "broken.py").write_text("def broken(][: pass\n")
        (tmp_path / "good.py").write_text(
            '__all__ = ["Good"]\nclass Good: pass\n',
        )
        index = _build_sibling_export_index(tmp_path, "test_pkg")
        tm.that(index, contains="Good")


class TestExtractExports:
    """Test extract_exports function."""

    def test_with_list_all(self) -> None:
        """Test __all__ as list."""
        code = '__all__ = ["Foo", "Bar"]'
        tree = ast.parse(code)
        has_all, exports = FlextInfraUtilitiesCodegen.extract_exports(tree)
        tm.that(has_all, eq=True)
        tm.that(exports, eq=["Foo", "Bar"])

    def test_with_tuple_all(self) -> None:
        """Test __all__ as tuple."""
        code = '__all__ = ("Foo", "Bar")'
        tree = ast.parse(code)
        has_all, exports = FlextInfraUtilitiesCodegen.extract_exports(tree)
        tm.that(has_all, eq=True)
        tm.that(exports, eq=["Foo", "Bar"])

    def test_with_non_string_elements(self) -> None:
        """Test ignores non-string elements."""
        code = '__all__ = ["Foo", 123, "Bar"]'
        tree = ast.parse(code)
        has_all, exports = FlextInfraUtilitiesCodegen.extract_exports(tree)
        tm.that(has_all, eq=True)
        tm.that(exports, eq=["Foo", "Bar"])

    def test_without_all(self) -> None:
        """Test when __all__ is missing."""
        code = "x = 1"
        tree = ast.parse(code)
        has_all, exports = FlextInfraUtilitiesCodegen.extract_exports(tree)
        tm.that(has_all, eq=False)
        tm.that(exports, eq=[])
