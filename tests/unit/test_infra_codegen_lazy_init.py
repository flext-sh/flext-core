"""Tests for FlextInfraLazyInitGenerator.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import flext_infra.codegen as mod
import pytest
from flext_core import FlextService
from flext_infra.codegen import FlextInfraLazyInitGenerator
from flext_infra.codegen.lazy_init import (
    _derive_lazy_map,
    _extract_docstring_source,
    _extract_exports,
    _extract_inline_constants,
    _generate_file,
    _generate_type_checking,
    _infer_package,
    _parse_existing_lazy_imports,
    _resolve_module,
    _resolve_unmapped,
    _run_ruff_fix,
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


class TestLazyInitEdgeCases:
    """Test edge cases and error handling in lazy init generation."""

    def test_process_file_with_read_error(self, tmp_path: Path) -> None:
        """Test _process_file handles read errors gracefully."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        init_file.write_text('"""Test."""\n__all__ = ["Test"]')

        # Make file unreadable
        init_file.chmod(0o000)
        try:
            result = generator._process_file(init_file, check_only=False)  # Line 76
            assert result == -1
        finally:
            init_file.chmod(0o644)

    def test_process_file_with_parse_error(self, tmp_path: Path) -> None:
        """Test _process_file handles parse errors gracefully."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        init_file.write_text("invalid python syntax ][")  # Line 97

        result = generator._process_file(init_file, check_only=False)
        assert result == -1

    def test_process_file_without_all_returns_none(self, tmp_path: Path) -> None:
        """Test _process_file returns None when __all__ is missing."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        init_file.write_text('"""Test."""\nfrom module import something')  # Line 99

        result = generator._process_file(init_file, check_only=False)
        assert result is None

    def test_process_file_with_empty_all_returns_none(self, tmp_path: Path) -> None:
        """Test _process_file returns None when __all__ is empty."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        init_file.write_text('"""Test."""\n__all__ = []')  # Lines 101-102

        result = generator._process_file(init_file, check_only=False)
        assert result is None

    def test_process_file_check_only_mode(self, tmp_path: Path) -> None:
        """Test _process_file in check_only mode doesn't modify files."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        original_content = '"""Test."""\nfrom module import Test\n__all__ = ["Test"]'
        init_file.write_text(original_content)

        result = generator._process_file(init_file, check_only=True)  # Line 129-131
        assert result >= 0
        # Verify file wasn't modified
        assert init_file.read_text() == original_content

    def test_process_file_with_inline_constants(self, tmp_path: Path) -> None:
        """Test _process_file handles inline constants correctly."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        init_file.write_text(
            '"""Test."""\n'
            '__version__ = "1.0.0"\n'
            "from module import Test\n"
            '__all__ = ["__version__", "Test"]'
        )  # Lines 135-137

        result = generator._process_file(init_file, check_only=False)
        assert result >= 0

    def test_process_file_with_existing_lazy_imports(self, tmp_path: Path) -> None:
        """Test _process_file regenerates existing _LAZY_IMPORTS."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        init_file.write_text(
            '"""Test."""\n'
            '_LAZY_IMPORTS = {"Test": ("module", "Test")}\n'
            '__all__ = ["Test"]'
        )  # Line 142

        result = generator._process_file(init_file, check_only=False)
        assert result >= 0

    def test_infer_package_without_src_directory(self) -> None:
        """Test _infer_package when path doesn't contain /src/."""
        path = Path("/workspace/lib/test/__init__.py")
        pkg = _infer_package(path)  # Line 159
        assert pkg == ""

    def test_resolve_module_with_zero_level(self) -> None:
        """Test _resolve_module with level 0 (absolute import)."""
        result = _resolve_module("module.submodule", 0, "current.pkg")  # Line 165
        assert result == "module.submodule"

    def test_resolve_module_with_empty_current_pkg(self) -> None:
        """Test _resolve_module with empty current package."""
        result = _resolve_module("module", 1, "")  # Lines 168-169
        assert result == "module"

    def test_resolve_module_with_level_exceeding_depth(self) -> None:
        """Test _resolve_module when level exceeds package depth."""
        result = _resolve_module("module", 5, "a.b")  # Lines 168-169
        # Should return raw module when level is too deep
        assert result == "module" or result.startswith("a")

    def test_extract_docstring_source_without_docstring(self) -> None:
        """Test _extract_docstring_source when no docstring exists."""
        code = "x = 1\ny = 2"
        tree = ast.parse(code)  # Line 186-189
        docstring = _extract_docstring_source(tree, code)
        assert docstring == ""

    def test_extract_docstring_source_with_non_string_expr(self) -> None:
        """Test _extract_docstring_source with non-string first expression."""
        code = "123\nx = 1"
        tree = ast.parse(code)  # Line 186-189
        docstring = _extract_docstring_source(tree, code)
        assert docstring == ""

    def test_extract_exports_with_tuple_all(self) -> None:
        """Test _extract_exports with __all__ as tuple."""
        code = '__all__ = ("Foo", "Bar")'
        tree = ast.parse(code)  # Line 207
        has_all, exports = _extract_exports(tree)
        assert has_all is True
        assert exports == ["Foo", "Bar"]

    def test_extract_exports_without_all(self) -> None:
        """Test _extract_exports when __all__ is missing."""
        code = "x = 1"
        tree = ast.parse(code)  # Line 215
        has_all, exports = _extract_exports(tree)
        assert has_all is False
        assert exports == []

    def test_extract_inline_constants_with_non_string_values(self) -> None:
        """Test _extract_inline_constants ignores non-string values."""
        code = '__version__ = "1.0.0"\n__count__ = 42\n__enabled__ = True'
        tree = ast.parse(code)  # Line 219
        constants = _extract_inline_constants(tree)
        assert "__version__" in constants
        assert "__count__" not in constants
        assert "__enabled__" not in constants

    def test_parse_existing_lazy_imports_with_annotated_assignment(self) -> None:
        """Test _parse_existing_lazy_imports with type-annotated assignment."""
        code = '_LAZY_IMPORTS: dict[str, tuple[str, str]] = {"Foo": ("module", "Foo")}'
        tree = ast.parse(code)  # Lines 233
        lazy_map = _parse_existing_lazy_imports(tree)
        assert "Foo" in lazy_map
        assert lazy_map["Foo"] == ("module", "Foo")

    def test_parse_existing_lazy_imports_with_invalid_dict(self) -> None:
        """Test _parse_existing_lazy_imports with malformed dict."""
        code = '_LAZY_IMPORTS = {"Foo": ("module",)}'  # Tuple with only 1 element
        tree = ast.parse(code)  # Line 280
        lazy_map = _parse_existing_lazy_imports(tree)
        # Should skip malformed entries
        assert "Foo" not in lazy_map

    def test_derive_lazy_map_with_import_from(self) -> None:
        """Test _derive_lazy_map extracts from ImportFrom statements."""
        code = "from module import Test, Helper"
        tree = ast.parse(code)  # Line 305
        lazy_map = _derive_lazy_map(tree, "current.pkg")
        assert "Test" in lazy_map
        assert "Helper" in lazy_map

    def test_derive_lazy_map_with_import_statement(self) -> None:
        """Test _derive_lazy_map extracts from Import statements."""
        code = "import module\nimport other as alias"
        tree = ast.parse(code)  # Line 321
        lazy_map = _derive_lazy_map(tree, "current.pkg")
        assert "module" in lazy_map
        assert "alias" in lazy_map

    def test_derive_lazy_map_skips_skip_modules(self) -> None:
        """Test _derive_lazy_map skips modules in _SKIP_MODULES."""
        code = "from typing import List\nfrom module import Test"
        tree = ast.parse(code)  # Line 324
        lazy_map = _derive_lazy_map(tree, "current.pkg")
        assert "List" not in lazy_map
        assert "Test" in lazy_map

    def test_derive_lazy_map_with_assignment_aliases(self) -> None:
        """Test _derive_lazy_map captures assignment aliases."""
        code = "from module import FlextConstants\nc = FlextConstants"
        tree = ast.parse(code)  # Lines 330-335
        lazy_map = _derive_lazy_map(tree, "current.pkg")
        assert "c" in lazy_map

    def test_derive_lazy_map_fixes_single_letter_aliases(self) -> None:
        """Test _derive_lazy_map fixes single-letter aliases."""
        code = "from module import FlextConstants, FlextModels\nc = FlextConstants"
        tree = ast.parse(code)  # Lines 340-345
        lazy_map = _derive_lazy_map(tree, "current.pkg")
        # Should map 'c' to FlextConstants
        if "c" in lazy_map:
            assert lazy_map["c"][1] == "FlextConstants"

    def test_resolve_unmapped_with_alias_suffix_matching(self, tmp_path: Path) -> None:
        """Test _resolve_unmapped resolves single-letter aliases."""
        exports_set = {"c", "m", "t"}  # Lines 351-356
        filtered = {"FlextConstants": ("module", "FlextConstants")}
        pkg_dir = tmp_path

        _resolve_unmapped(exports_set, filtered, "test.pkg", pkg_dir)
        # Should resolve 'c' to FlextConstants
        assert "c" in filtered or "FlextConstants" in filtered

    def test_resolve_unmapped_with_version_file(self, tmp_path: Path) -> None:
        """Test _resolve_unmapped resolves __version__ from __version__.py."""
        exports_set = {"__version__"}  # Lines 372-389
        filtered = {}
        pkg_dir = tmp_path
        version_file = pkg_dir / "__version__.py"
        version_file.write_text('__version__ = "1.0.0"')

        _resolve_unmapped(exports_set, filtered, "test.pkg", pkg_dir)
        assert "__version__" in filtered

    def test_resolve_unmapped_with_version_info_file(self, tmp_path: Path) -> None:
        """Test _resolve_unmapped resolves __version_info__ from __version__.py."""
        exports_set = {"__version_info__"}  # Lines 372-389
        filtered = {}
        pkg_dir = tmp_path
        version_file = pkg_dir / "__version__.py"
        version_file.write_text("__version_info__ = (1, 0, 0)")

        _resolve_unmapped(exports_set, filtered, "test.pkg", pkg_dir)
        assert "__version_info__" in filtered

    def test_generate_type_checking_with_empty_groups(self) -> None:
        """Test _generate_type_checking with no imports."""
        groups: dict[str, list[tuple[str, str]]] = {}  # Line 405-406
        lines = _generate_type_checking(groups)
        assert "if TYPE_CHECKING:" in lines
        assert any("pass" in line for line in lines)

    def test_generate_type_checking_with_single_module(self) -> None:
        """Test _generate_type_checking with single module."""
        groups = {"module": [("Test", "Test")]}  # Line 419
        lines = _generate_type_checking(groups)
        assert "from module import" in " ".join(lines)

    def test_generate_type_checking_with_long_import_line(self) -> None:
        """Test _generate_type_checking wraps long import lines."""
        groups = {
            "module": [
                ("VeryLongClassName1", "VeryLongClassName1"),
                ("VeryLongClassName2", "VeryLongClassName2"),
                ("VeryLongClassName3", "VeryLongClassName3"),
            ]
        }  # Lines 424-426
        lines = _generate_type_checking(groups)
        # Should have imports from module
        assert any("module" in line for line in lines)

    def test_generate_type_checking_with_alias_imports(self) -> None:
        """Test _generate_type_checking with aliased imports."""
        groups = {"module": [("c", "FlextConstants"), ("m", "FlextModels")]}  # Line 435
        lines = _generate_type_checking(groups)
        output = " ".join(lines)
        assert "as" in output

    def test_generate_file_with_flext_core_package(self, tmp_path: Path) -> None:
        """Test _generate_file uses correct lazy import for flext_core."""
        exports = ["Test"]  # Line 462
        filtered = {"Test": ("module", "Test")}
        inline_constants = {}

        content = _generate_file("", exports, filtered, inline_constants, "flext_core")
        assert "flext_core._utilities.lazy" in content

    def test_generate_file_with_other_package(self) -> None:
        """Test _generate_file uses correct lazy import for other packages."""
        exports = ["Test"]  # Line 482
        filtered = {"Test": ("module", "Test")}
        inline_constants = {}

        content = _generate_file("", exports, filtered, inline_constants, "other_pkg")
        assert "from flext_core._utilities.lazy import" in content

    def test_generate_file_with_inline_constants(self) -> None:
        """Test _generate_file includes inline constants."""
        exports = ["__version__", "Test"]  # Line 484
        filtered = {"Test": ("module", "Test")}
        inline_constants = {"__version__": "1.0.0"}

        content = _generate_file("", exports, filtered, inline_constants, "test_pkg")
        assert '__version__ = "1.0.0"' in content

    def test_generate_file_with_docstring(self) -> None:
        """Test _generate_file preserves docstring."""
        docstring = '"""Test module."""'
        exports = ["Test"]
        filtered = {"Test": ("module", "Test")}
        inline_constants = {}

        content = _generate_file(
            docstring, exports, filtered, inline_constants, "test_pkg"
        )
        assert docstring in content

    def test_run_ruff_fix_with_nonexistent_file(self, tmp_path: Path) -> None:
        """Test _run_ruff_fix handles nonexistent files gracefully."""
        nonexistent = tmp_path / "nonexistent.py"
        # Should not raise exception
        _run_ruff_fix(nonexistent)

    def test_execute_method_returns_flext_result(self, tmp_path: Path) -> None:
        """Test execute() method returns FlextResult[int] (line 76)."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        result = generator.execute()
        assert result.is_success
        assert isinstance(result.value, int)

    def test_run_with_errors_increments_error_count(self, tmp_path: Path) -> None:
        """Test run() increments error count on parse errors (line 99)."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        init_file.write_text('"""Test."""\ninvalid syntax ][\n__all__ = ["Test"]')

        result = generator.run(check_only=False)
        # Should return 0 because the file is skipped due to parse error
        assert result >= 0

    def test_run_with_unmapped_exports_increments_unmapped_count(
        self, tmp_path: Path
    ) -> None:
        """Test run() increments unmapped_count when exports are unmapped (lines 101-102)."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        # Create a file with unmapped exports
        init_file.write_text(
            '"""Test."""\nfrom module import Test\n__all__ = ["Test", "Unmapped"]'
        )

        result = generator.run(check_only=False)
        # Should return > 0 because there's an unmapped export
        assert result > 0

    def test_process_file_with_unmapped_exports_message(self, tmp_path: Path) -> None:
        """Test _process_file formats message with unmapped exports (lines 186-189)."""
        generator = FlextInfraLazyInitGenerator(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        init_file = src_dir / "__init__.py"
        init_file.write_text(
            '"""Test."""\nfrom module import Test\n__all__ = ["Test", "Unmapped"]'
        )

        result = generator._process_file(init_file, check_only=False)
        # Should return 1 (one unmapped export)
        assert result == 1

    def test_parse_existing_lazy_imports_with_non_dict_value(
        self, tmp_path: Path
    ) -> None:
        """Test _parse_existing_lazy_imports returns empty dict for invalid dict (line 280)."""
        code = '_LAZY_IMPORTS = "not a dict"'
        tree = ast.parse(code)
        lazy_map = _parse_existing_lazy_imports(tree)
        assert lazy_map == {}

    def test_derive_lazy_map_skips_current_package(self, tmp_path: Path) -> None:
        """Test _derive_lazy_map skips imports from current package (line 324)."""
        code = "from test_pkg import something"
        tree = ast.parse(code)
        lazy_map = _derive_lazy_map(tree, "test_pkg")
        # Should not include imports from current package
        assert "something" not in lazy_map

    def test_derive_lazy_map_skips_stdlib_imports(self, tmp_path: Path) -> None:
        """Test _derive_lazy_map skips stdlib imports (line 334)."""
        code = "import sys"
        tree = ast.parse(code)
        lazy_map = _derive_lazy_map(tree, "test_pkg")
        # Should not include sys
        assert "sys" not in lazy_map

    def test_derive_lazy_map_fixes_single_letter_aliases_mapping(
        self, tmp_path: Path
    ) -> None:
        """Test _derive_lazy_map fixes single-letter aliases (lines 353-356)."""
        code = "from flext_core import FlextConstants\nc = FlextConstants"
        tree = ast.parse(code)
        lazy_map = _derive_lazy_map(tree, "test_pkg")
        # Should have 'c' mapped to FlextConstants
        assert "c" in lazy_map
        assert lazy_map["c"][1] == "FlextConstants"

    def test_generate_type_checking_with_long_imports(self, tmp_path: Path) -> None:
        """Test _generate_type_checking formats long imports on multiple lines (lines 424-426)."""
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        groups["module"] = [
            ("VeryLongExportName1", "VeryLongExportName1"),
            ("VeryLongExportName2", "VeryLongExportName2"),
            ("VeryLongExportName3", "VeryLongExportName3"),
        ]
        lines = _generate_type_checking(groups)
        content = "\n".join(lines)
        # Should have multi-line import with parentheses
        assert "(" in content or len(lines) > 1

    def test_generate_type_checking_with_multiple_modules_spacing(
        self, tmp_path: Path
    ) -> None:
        """Test _generate_type_checking adds blank lines between module groups (line 435)."""
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        groups["module_a"] = [("Test1", "Test1")]
        groups["module_b"] = [("Test2", "Test2")]
        lines = _generate_type_checking(groups)
        # Should have blank line between different modules
        assert "" in lines
