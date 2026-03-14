"""Tests for lazy_init code generation: aliases, type checking, file output.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

import flext_infra.codegen as mod
from flext_infra.codegen._utilities import FlextInfraUtilitiesCodegen
from flext_infra.codegen.lazy_init import FlextInfraCodegenLazyInit
from flext_tests import tm

_resolve_aliases: Callable[[dict[str, tuple[str, str]]], None] = getattr(
    FlextInfraCodegenLazyInit, "_resolve_aliases"
)
_generate_file: Callable[
    [str, list[str], Mapping[str, tuple[str, str]], Mapping[str, str], str], str
] = getattr(FlextInfraCodegenLazyInit, "_generate_file")
_run_ruff_fix: Callable[[Path], None] = getattr(
    FlextInfraCodegenLazyInit, "_run_ruff_fix"
)


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
    """Test generate_type_checking function."""

    def test_with_empty_groups(self) -> None:
        """Test with no imports."""
        groups: dict[str, list[tuple[str, str]]] = {}
        lines = FlextInfraUtilitiesCodegen.generate_type_checking(groups)
        tm.that(lines, contains="if TYPE_CHECKING:")
        tm.that(any("pass" in line for line in lines), eq=True)

    def test_with_single_module(self) -> None:
        """Test with single module."""
        groups = {"module": [("Test", "Test")]}
        lines = FlextInfraUtilitiesCodegen.generate_type_checking(groups)
        tm.that(" ".join(lines), contains="from module import")

    def test_with_aliased_imports(self) -> None:
        """Test with aliased imports."""
        groups = {"module": [("c", "FlextConstants"), ("m", "FlextModels")]}
        lines = FlextInfraUtilitiesCodegen.generate_type_checking(groups)
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
        lines = FlextInfraUtilitiesCodegen.generate_type_checking(groups)
        tm.that(any("module" in line for line in lines), eq=True)

    def test_with_multiple_modules_spacing(self) -> None:
        """Test blank lines between different top-level package groups."""
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        groups["alpha_pkg.module"] = [("Test1", "Test1")]
        groups["beta_pkg.module"] = [("Test2", "Test2")]
        lines = FlextInfraUtilitiesCodegen.generate_type_checking(groups)
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
