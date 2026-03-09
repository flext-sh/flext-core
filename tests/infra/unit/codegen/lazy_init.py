"""Tests for FlextInfraCodegenLazyInit.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

import flext_infra.codegen as mod
from flext_core import FlextService
from flext_infra.codegen import FlextInfraCodegenLazyInit
from flext_infra.codegen.lazy_init import (
    _build_sibling_export_index,
    _extract_exports,
    _extract_inline_constants,
    _extract_version_exports,
    _generate_file,
    _generate_type_checking,
    _infer_package,
    _merge_child_exports,
    _read_existing_docstring,
    _resolve_aliases,
    _run_ruff_fix,
    _scan_ast_public_defs,
    _should_bubble_up,
)
from flext_tests import tm


class TestFlextInfraCodegenLazyInit:
    """Test suite for FlextInfraCodegenLazyInit service."""

    def test_init_accepts_workspace_root(self, tmp_path: Path) -> None:
        """Test generator initialization with workspace root."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        tm.that(generator, none=False)

    def test_run_with_empty_workspace_returns_zero(self, tmp_path: Path) -> None:
        """Test run() on empty workspace returns 0 (no errors)."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        tm.that(result, eq=0)

    def test_run_with_check_only_flag(self, tmp_path: Path) -> None:
        """Test run() respects check_only flag without modifying files."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=True)
        tm.that(result, eq=0)

    def test_generator_is_flext_service(self, tmp_path: Path) -> None:
        """Test that FlextInfraCodegenLazyInit is a FlextService."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        tm.that(generator, is_=FlextService)

    def test_run_returns_integer_exit_code(self, tmp_path: Path) -> None:
        """Test that run() returns an integer exit code."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        tm.that(result, is_=int)
        tm.that(result, gte=0)

    def test_execute_method_returns_flext_result(self, tmp_path: Path) -> None:
        """Test execute() method returns FlextResult[int]."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.execute()
        tm.ok(result)
        tm.that(result.value, is_type=int)

    def test_generate_from_sibling_files(self, tmp_path: Path) -> None:
        """Test that generator discovers exports from sibling .py files."""
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        # Create sibling .py file with __all__
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["TestModel"]\n\nclass TestModel:\n    pass\n',
        )
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        tm.that(result, eq=0)
        init_file = src_dir / "__init__.py"
        tm.that(init_file.exists(), eq=True)
        content = init_file.read_text()
        tm.that(content, contains="TestModel")
        tm.that(content, contains="test_pkg.models")

    def test_generate_bottom_up(self, tmp_path: Path) -> None:
        """Test that subdirectory exports bubble up to parent."""
        src_dir = tmp_path / "src" / "pkg"
        sub_dir = src_dir / "sub"
        sub_dir.mkdir(parents=True)
        # Subdirectory has a module
        (sub_dir / "service.py").write_text(
            '"""Service."""\n\n__all__ = ["SubService"]\n\nclass SubService:\n    pass\n',
        )
        # Parent has its own module
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["PkgModel"]\n\nclass PkgModel:\n    pass\n',
        )
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        result = generator.run(check_only=False)
        tm.that(result, eq=0)
        # Child __init__.py should have SubService
        child_init = sub_dir / "__init__.py"
        tm.that(child_init.exists(), eq=True)
        tm.that(child_init.read_text(), contains="SubService")
        # Parent __init__.py should have both
        parent_init = src_dir / "__init__.py"
        tm.that(parent_init.exists(), eq=True)
        parent_content = parent_init.read_text()
        tm.that(parent_content, contains="PkgModel")
        tm.that(parent_content, contains="SubService")

    def test_generate_preserves_existing_docstring(self, tmp_path: Path) -> None:
        """Test that existing docstring is preserved in regenerated file."""
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        # Create existing __init__.py with docstring
        (src_dir / "__init__.py").write_text(
            '"""My custom package docstring."""\n\n__all__ = []\n',
        )
        # Create sibling .py file
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["Foo"]\n\nclass Foo:\n    pass\n',
        )
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        generator.run(check_only=False)
        content = (src_dir / "__init__.py").read_text()
        tm.that(content, contains="My custom package docstring")


class TestInferPackage:
    """Test _infer_package function."""

    def test_src_path(self) -> None:
        """Test inference from src/ path."""
        path = Path("/workspace/src/test_pkg/__init__.py")
        tm.that(_infer_package(path), eq="test_pkg")

    def test_deeply_nested_src_path(self) -> None:
        """Test inference from deeply nested src/ path."""
        path = Path("/workspace/src/a/b/c/d/__init__.py")
        tm.that(_infer_package(path), eq="a.b.c.d")

    def test_tests_path(self) -> None:
        """Test inference from tests/ path."""
        path = Path("/workspace/tests/unit/__init__.py")
        tm.that(_infer_package(path), eq="tests.unit")

    def test_without_src_directory(self) -> None:
        """Test when path doesn't contain /src/."""
        path = Path("/workspace/lib/test/__init__.py")
        tm.that(_infer_package(path), eq="")


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
    """Test _extract_exports function."""

    def test_with_list_all(self) -> None:
        """Test __all__ as list."""
        code = '__all__ = ["Foo", "Bar"]'
        tree = ast.parse(code)
        has_all, exports = _extract_exports(tree)
        tm.that(has_all, eq=True)
        tm.that(exports, eq=["Foo", "Bar"])

    def test_with_tuple_all(self) -> None:
        """Test __all__ as tuple."""
        code = '__all__ = ("Foo", "Bar")'
        tree = ast.parse(code)
        has_all, exports = _extract_exports(tree)
        tm.that(has_all, eq=True)
        tm.that(exports, eq=["Foo", "Bar"])

    def test_with_non_string_elements(self) -> None:
        """Test ignores non-string elements."""
        code = '__all__ = ["Foo", 123, "Bar"]'
        tree = ast.parse(code)
        has_all, exports = _extract_exports(tree)
        tm.that(has_all, eq=True)
        tm.that(exports, eq=["Foo", "Bar"])

    def test_without_all(self) -> None:
        """Test when __all__ is missing."""
        code = "x = 1"
        tree = ast.parse(code)
        has_all, exports = _extract_exports(tree)
        tm.that(has_all, eq=False)
        tm.that(exports, eq=[])


class TestScanAstPublicDefs:
    """Test _scan_ast_public_defs function."""

    def test_finds_classes(self) -> None:
        """Test scanning finds public classes."""
        tree = ast.parse("class PublicClass:\n    pass\n")
        index: dict[str, tuple[str, str]] = {}
        _scan_ast_public_defs(tree, "mod", index)
        tm.that(index, contains="PublicClass")

    def test_skips_private(self) -> None:
        """Test scanning skips private names."""
        tree = ast.parse("class _PrivateClass:\n    pass\n")
        index: dict[str, tuple[str, str]] = {}
        _scan_ast_public_defs(tree, "mod", index)
        tm.that(index, excludes="_PrivateClass")

    def test_finds_functions(self) -> None:
        """Test scanning finds public functions."""
        tree = ast.parse("def public_func():\n    pass\n")
        index: dict[str, tuple[str, str]] = {}
        _scan_ast_public_defs(tree, "mod", index)
        tm.that(index, contains="public_func")

    def test_finds_assignments(self) -> None:
        """Test scanning finds public assignments."""
        tree = ast.parse("MY_CONST = 42\n")
        index: dict[str, tuple[str, str]] = {}
        _scan_ast_public_defs(tree, "mod", index)
        tm.that(index, contains="MY_CONST")


class TestExtractInlineConstants:
    """Test _extract_inline_constants function."""

    def test_multiple_constants(self) -> None:
        """Test extracting multiple string constants."""
        code = '__version__ = "1.0.0"\n__author__ = "Test"\n__license__ = "MIT"'
        tree = ast.parse(code)
        constants = _extract_inline_constants(tree)
        tm.that(len(constants), eq=3)
        tm.that(constants["__version__"], eq="1.0.0")

    def test_ignores_non_string_values(self) -> None:
        """Test ignores non-string constant values."""
        code = '__version__ = "1.0.0"\n__count__ = 42\n__enabled__ = True'
        tree = ast.parse(code)
        constants = _extract_inline_constants(tree)
        tm.that(constants, contains="__version__")
        tm.that(constants, excludes="__count__")


class TestShouldBubbleUp:
    """Test _should_bubble_up function."""

    def test_public_class_name(self) -> None:
        """Test that public class names bubble up."""
        tm.that(_should_bubble_up("FlextInfraModels"), eq=True)

    def test_private_name_filtered(self) -> None:
        """Test that private names are filtered."""
        tm.that(_should_bubble_up("_internal"), eq=False)

    def test_main_filtered(self) -> None:
        """Test that 'main' entry point is filtered."""
        tm.that(_should_bubble_up("main"), eq=False)

    def test_all_caps_filtered(self) -> None:
        """Test that ALL_CAPS constants are filtered."""
        tm.that(_should_bubble_up("BLUE"), eq=False)
        tm.that(_should_bubble_up("SYM_ARROW"), eq=False)

    def test_singleton_name_passes(self) -> None:
        """Test that lowercase singleton names pass."""
        tm.that(_should_bubble_up("output"), eq=True)

    def test_single_letter_alias_passes(self) -> None:
        """Test that single-letter aliases pass."""
        tm.that(_should_bubble_up("c"), eq=True)
        tm.that(_should_bubble_up("e"), eq=True)


class TestMergeChildExports:
    """Test _merge_child_exports function."""

    def test_merges_child_exports(self, tmp_path: Path) -> None:
        """Test that child exports are merged into parent."""
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        lazy_map: dict[str, tuple[str, str]] = {}
        dir_exports = {
            str(sub_dir): {
                "SubService": ("pkg.sub.service", "SubService"),
            },
        }
        _merge_child_exports(tmp_path, lazy_map, dir_exports)
        tm.that(lazy_map, contains="SubService")
        tm.that(lazy_map["SubService"], eq=("pkg.sub.service", "SubService"))

    def test_sibling_exports_take_precedence(self, tmp_path: Path) -> None:
        """Test that existing sibling exports are NOT overwritten."""
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        lazy_map: dict[str, tuple[str, str]] = {
            "Model": ("pkg.models", "Model"),
        }
        dir_exports = {
            str(sub_dir): {
                "Model": ("pkg.sub.models", "Model"),
            },
        }
        _merge_child_exports(tmp_path, lazy_map, dir_exports)
        # Sibling wins
        tm.that(lazy_map["Model"], eq=("pkg.models", "Model"))

    def test_filters_all_caps(self, tmp_path: Path) -> None:
        """Test that ALL_CAPS constants don't bubble up."""
        sub_dir = tmp_path / "sub"
        sub_dir.mkdir()
        lazy_map: dict[str, tuple[str, str]] = {}
        dir_exports = {
            str(sub_dir): {
                "BLUE": ("pkg.sub.colors", "BLUE"),
                "Service": ("pkg.sub.service", "Service"),
            },
        }
        _merge_child_exports(tmp_path, lazy_map, dir_exports)
        tm.that(lazy_map, excludes="BLUE")
        tm.that(lazy_map, contains="Service")


class TestExtractVersionExports:
    """Test _extract_version_exports function."""

    def test_extracts_string_constants(self, tmp_path: Path) -> None:
        """Test extracting __version__ as inline constant."""
        (tmp_path / "__version__.py").write_text('__version__ = "1.0.0"\n')
        inline, _ = _extract_version_exports(tmp_path, "test_pkg")
        tm.that(inline, contains="__version__")
        tm.that(inline["__version__"], eq="1.0.0")

    def test_extracts_non_string_as_lazy(self, tmp_path: Path) -> None:
        """Test extracting __version_info__ as lazy import."""
        (tmp_path / "__version__.py").write_text(
            '__version__ = "1.0.0"\n__version_info__ = (1, 0, 0)\n',
        )
        inline, lazy = _extract_version_exports(tmp_path, "test_pkg")
        tm.that(inline, contains="__version__")
        tm.that(lazy, contains="__version_info__")
        tm.that(
            lazy["__version_info__"],
            eq=("test_pkg.__version__", "__version_info__"),
        )

    def test_no_version_file(self, tmp_path: Path) -> None:
        """Test returns empty when __version__.py doesn't exist."""
        inline, lazy = _extract_version_exports(tmp_path, "test_pkg")
        tm.that(inline, eq={})
        tm.that(lazy, eq={})


class TestResolveAliases:
    """Test _resolve_aliases function."""

    def test_resolves_c_alias(self) -> None:
        """Test resolving 'c' alias to Constants class."""
        lazy_map: dict[str, tuple[str, str]] = {
            "FlextConstants": ("pkg.constants", "FlextConstants"),
        }
        _resolve_aliases(lazy_map)
        tm.that(lazy_map, contains="c")
        tm.that(lazy_map["c"], eq=("pkg.constants", "FlextConstants"))

    def test_does_not_overwrite_existing(self) -> None:
        """Test that existing alias is not overwritten."""
        lazy_map: dict[str, tuple[str, str]] = {
            "c": ("pkg.custom", "CustomConst"),
            "FlextConstants": ("pkg.constants", "FlextConstants"),
        }
        _resolve_aliases(lazy_map)
        # Should keep existing mapping
        tm.that(lazy_map["c"], eq=("pkg.custom", "CustomConst"))

    def test_resolves_multiple_aliases(self) -> None:
        """Test resolving multiple aliases at once."""
        lazy_map: dict[str, tuple[str, str]] = {
            "FlextConstants": ("pkg.constants", "FlextConstants"),
            "FlextModels": ("pkg.models", "FlextModels"),
            "FlextTypes": ("pkg.typings", "FlextTypes"),
        }
        _resolve_aliases(lazy_map)
        tm.that(lazy_map, contains="c")
        tm.that(lazy_map, contains="m")
        tm.that(lazy_map, contains="t")


class TestGenerateTypeChecking:
    """Test _generate_type_checking function."""

    def test_with_empty_groups(self) -> None:
        """Test with no imports."""
        groups: dict[str, list[tuple[str, str]]] = {}
        lines = _generate_type_checking(groups)
        tm.that(lines, contains="if TYPE_CHECKING:")
        tm.that(any("pass" in line for line in lines), eq=True)

    def test_with_single_module(self) -> None:
        """Test with single module."""
        groups = {"module": [("Test", "Test")]}
        lines = _generate_type_checking(groups)
        tm.that(" ".join(lines), contains="from module import")

    def test_with_aliased_imports(self) -> None:
        """Test with aliased imports."""
        groups = {"module": [("c", "FlextConstants"), ("m", "FlextModels")]}
        lines = _generate_type_checking(groups)
        joined = " ".join(lines)
        tm.that(joined, contains="as")

    def test_with_long_import_line(self) -> None:
        """Test wraps long import lines."""
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        groups["module"] = [
            ("VeryLongClassName1", "VeryLongClassName1"),
            ("VeryLongClassName2", "VeryLongClassName2"),
            ("VeryLongClassName3", "VeryLongClassName3"),
        ]
        lines = _generate_type_checking(groups)
        tm.that(any("module" in line for line in lines), eq=True)

    def test_with_multiple_modules_spacing(self) -> None:
        """Test blank lines between different top-level package groups."""
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        groups["alpha_pkg.module"] = [("Test1", "Test1")]
        groups["beta_pkg.module"] = [("Test2", "Test2")]
        lines = _generate_type_checking(groups)
        tm.that(lines, contains="")


class TestGenerateFile:
    """Test _generate_file function."""

    def test_with_flext_core_package(self) -> None:
        """Test uses correct lazy import for flext_core."""
        exports = ["Test"]
        filtered = {"Test": ("module", "Test")}
        inline_constants: dict[str, str] = {}
        content = _generate_file("", exports, filtered, inline_constants, "flext_core")
        tm.that(content, contains="flext_core._utilities.lazy")

    def test_with_other_package(self) -> None:
        """Test uses correct lazy import for non-core packages."""
        exports = ["Test"]
        filtered = {"Test": ("module", "Test")}
        inline_constants: dict[str, str] = {}
        content = _generate_file("", exports, filtered, inline_constants, "other_pkg")
        tm.that(content, contains="from flext_core.lazy import")

    def test_with_inline_constants(self) -> None:
        """Test includes inline constants."""
        exports = ["__version__", "Test"]
        filtered = {"Test": ("module", "Test")}
        inline_constants = {"__version__": "1.0.0"}
        content = _generate_file("", exports, filtered, inline_constants, "test_pkg")
        tm.that(content, contains='__version__ = "1.0.0"')

    def test_with_docstring(self) -> None:
        """Test preserves docstring."""
        docstring = '"""Test module."""'
        exports = ["Test"]
        filtered = {"Test": ("module", "Test")}
        inline_constants: dict[str, str] = {}
        content = _generate_file(
            docstring,
            exports,
            filtered,
            inline_constants,
            "test_pkg",
        )
        tm.that(content, contains=docstring)

    def test_has_autogen_header(self) -> None:
        """Test generated file starts with autogen header."""
        exports = ["Test"]
        filtered = {"Test": ("module", "Test")}
        inline_constants: dict[str, str] = {}
        content = _generate_file("", exports, filtered, inline_constants, "test_pkg")
        tm.that(content, contains="AUTO-GENERATED")

    def test_has_all_list(self) -> None:
        """Test generated file has __all__ list."""
        exports = ["Alpha", "Beta"]
        filtered = {"Alpha": ("mod", "Alpha"), "Beta": ("mod", "Beta")}
        inline_constants: dict[str, str] = {}
        content = _generate_file("", exports, filtered, inline_constants, "test_pkg")
        tm.that(content, contains="__all__")
        tm.that(content, contains='"Alpha"')
        tm.that(content, contains='"Beta"')


class TestRunRuffFix:
    """Test _run_ruff_fix function."""

    def test_with_nonexistent_file(self, tmp_path: Path) -> None:
        """Test handles nonexistent files gracefully."""
        nonexistent = tmp_path / "nonexistent.py"
        _run_ruff_fix(nonexistent)  # Should not raise


def test_codegen_init_getattr_raises_attribute_error() -> None:
    """Test that accessing nonexistent attribute raises AttributeError."""
    with pytest.raises(AttributeError):
        mod.nonexistent_xyz_attribute


class TestProcessDirectory:
    """Test the _process_directory method (integration-level)."""

    def test_generates_init_from_sibling_files(self, tmp_path: Path) -> None:
        """Test _process_directory generates __init__.py from siblings."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["TestModel"]\n\nclass TestModel:\n    pass\n',
        )
        dir_exports: dict[str, dict[str, tuple[str, str]]] = {}
        result, exports = generator._process_directory(
            src_dir,
            check_only=False,
            dir_exports=dir_exports,
        )
        tm.that(result, eq=0)
        tm.that(exports, contains="TestModel")
        init_content = (src_dir / "__init__.py").read_text()
        tm.that(init_content, contains="TestModel")

    def test_check_only_does_not_write(self, tmp_path: Path) -> None:
        """Test _process_directory in check_only mode doesn't write files."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["TestModel"]\n\nclass TestModel:\n    pass\n',
        )
        dir_exports: dict[str, dict[str, tuple[str, str]]] = {}
        result, exports = generator._process_directory(
            src_dir,
            check_only=True,
            dir_exports=dir_exports,
        )
        tm.that(result, eq=0)
        tm.that(exports, contains="TestModel")
        # __init__.py should NOT have been created
        tm.that((src_dir / "__init__.py").exists(), eq=False)

    def test_skips_directory_without_package(self, tmp_path: Path) -> None:
        """Test _process_directory skips dirs that can't infer package."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        random_dir = tmp_path / "random"
        random_dir.mkdir()
        (random_dir / "models.py").write_text("class Model: pass\n")
        dir_exports: dict[str, dict[str, tuple[str, str]]] = {}
        result, exports = generator._process_directory(
            random_dir,
            check_only=False,
            dir_exports=dir_exports,
        )
        tm.that(result, eq=None)
        tm.that(exports, eq={})

    def test_includes_child_exports(self, tmp_path: Path) -> None:
        """Test _process_directory includes child subdirectory exports."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "pkg"
        sub_dir = src_dir / "sub"
        sub_dir.mkdir(parents=True)
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["ParentModel"]\n\nclass ParentModel:\n    pass\n',
        )
        dir_exports = {
            str(sub_dir): {
                "ChildService": ("pkg.sub.service", "ChildService"),
            },
        }
        result, exports = generator._process_directory(
            src_dir,
            check_only=False,
            dir_exports=dir_exports,
        )
        tm.that(result, eq=0)
        tm.that(exports, contains="ParentModel")
        tm.that(exports, contains="ChildService")

    def test_handles_version_file(self, tmp_path: Path) -> None:
        """Test _process_directory handles __version__.py correctly."""
        generator = FlextInfraCodegenLazyInit(workspace_root=tmp_path)
        src_dir = tmp_path / "src" / "test_pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "models.py").write_text(
            '"""Models."""\n\n__all__ = ["Model"]\n\nclass Model:\n    pass\n',
        )
        (src_dir / "__version__.py").write_text(
            '__version__ = "1.0.0"\n__version_info__ = (1, 0, 0)\n',
        )
        dir_exports: dict[str, dict[str, tuple[str, str]]] = {}
        result, _ = generator._process_directory(
            src_dir,
            check_only=False,
            dir_exports=dir_exports,
        )
        tm.that(result, eq=0)
        content = (src_dir / "__init__.py").read_text()
        tm.that(content, contains='__version__ = "1.0.0"')
        tm.that(content, contains="__version_info__")
