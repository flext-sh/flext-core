"""Lazy import paths follow Python's package-relative import contract.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib

import pytest

from flext_core.lazy import merge_lazy_imports, normalize_lazy_imports
from tests import tm


class TestsFlextCoreLazyRelativePackages:
    """Resolve real standard-library modules through the public lazy mapping API."""

    @staticmethod
    @pytest.mark.parametrize(
        ("package", "relative", "absolute"),
        [
            ("xml.etree", "..dom", "xml.dom"),
            ("xml.dom", ".minidom", "xml.dom.minidom"),
            ("xml.dom", ".", "xml.dom"),
            ("xml.dom", "..", "xml"),
        ],
    )
    def test_string_targets_resolve_to_the_real_module(
        package: str,
        relative: str,
        absolute: str,
    ) -> None:
        """Both sibling and current-package paths identify importable modules."""
        normalized = normalize_lazy_imports(package, {"module": relative})
        target = normalized["module"]
        tm.that(target, is_=str)
        if isinstance(target, str):
            tm.that(
                importlib.import_module(target),
                where=lambda module: module is importlib.import_module(absolute),
            )

    @staticmethod
    def test_symbol_target_resolves_parent_package() -> None:
        """A symbol mapping preserves its attribute while resolving its module."""
        normalized = normalize_lazy_imports("xml.etree", {"Node": ("..dom", "Node")})
        target = normalized["Node"]
        tm.that(target, is_=tuple)
        module_name, attribute = target[0], target[1]
        module = importlib.import_module(module_name)
        expected = importlib.import_module("xml.dom")
        tm.that(getattr(module, attribute) is expected.Node, truth=True)

    @staticmethod
    def test_relative_path_cannot_escape_the_top_level_package() -> None:
        """An invalid parent traversal fails instead of manufacturing a module name."""
        with pytest.raises(ImportError):
            normalize_lazy_imports("xml", {"module": "..dom"})

    @staticmethod
    @pytest.mark.parametrize(
        ("package", "relative"),
        [
            ("xml.etree", "..dom"),
            ("xml.dom", ".minidom"),
            ("xml.dom", "."),
            ("xml.dom", ".."),
        ],
    )
    def test_merge_loads_relative_child_packages(
        package: str,
        relative: str,
    ) -> None:
        """Real child packages without lazy maps preserve the local exports."""
        merged = merge_lazy_imports(
            (relative,),
            {"Node": ("xml.dom", "Node")},
            module_name=package,
        )
        tm.that(merged, eq={"Node": ("xml.dom", "Node")})

    @staticmethod
    def test_merge_child_cannot_escape_the_top_level_package() -> None:
        """Child discovery rejects traversal beyond the containing package."""
        with pytest.raises(ImportError):
            merge_lazy_imports(("..dom",), {}, module_name="xml")
