"""Tests for lazy_init transform functions: AST scanning, filtering, merging.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
from pathlib import Path

from flext_infra.codegen._utilities import FlextInfraUtilitiesCodegen
from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
from flext_tests import tm

_scan_ast_public_defs: Callable[[ast.Module, str, dict[str, tuple[str, str]]], None] = (
    getattr(FlextInfraCodegenLazyInit, "_scan_ast_public_defs")
)
_should_bubble_up: Callable[[str], bool] = getattr(
    FlextInfraCodegenLazyInit, "_should_bubble_up"
)
_merge_child_exports: Callable[
    [Path, dict[str, tuple[str, str]], Mapping[str, dict[str, tuple[str, str]]]], None
] = getattr(FlextInfraCodegenLazyInit, "_merge_child_exports")
_extract_version_exports: Callable[
    [Path, str], tuple[dict[str, str], dict[str, tuple[str, str]]]
] = getattr(FlextInfraCodegenLazyInit, "_extract_version_exports")


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
    """Test extract_inline_constants function."""

    def test_multiple_constants(self) -> None:
        """Test extracting multiple string constants."""
        code = '__version__ = "1.0.0"\n__author__ = "Test"\n__license__ = "MIT"'
        tree = ast.parse(code)
        constants = FlextInfraUtilitiesCodegen.extract_inline_constants(tree)
        tm.that(len(constants), eq=3)
        tm.that(constants["__version__"], eq="1.0.0")

    def test_ignores_non_string_values(self) -> None:
        """Test ignores non-string constant values."""
        code = '__version__ = "1.0.0"\n__count__ = 42\n__enabled__ = True'
        tree = ast.parse(code)
        constants = FlextInfraUtilitiesCodegen.extract_inline_constants(tree)
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
