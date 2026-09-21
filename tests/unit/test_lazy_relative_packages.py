"""Lazy import paths follow Python's package-relative import contract."""

from __future__ import annotations

import importlib

import pytest

from flext_core.lazy import merge_lazy_imports, normalize_lazy_imports


class TestsFlextCoreLazyRelativePackages:
    """Resolve real standard-library modules through the public lazy mapping API."""

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
        self, package: str, relative: str, absolute: str
    ) -> None:
        """Both sibling and current-package paths identify importable modules."""
        normalized = normalize_lazy_imports(package, {"module": relative})
        target = normalized["module"]
        assert isinstance(target, str)
        assert importlib.import_module(target) is importlib.import_module(absolute)

    def test_symbol_target_resolves_parent_package(self) -> None:
        """A symbol mapping preserves its attribute while resolving its module."""
        normalized = normalize_lazy_imports("xml.etree", {"Node": ("..dom", "Node")})
        target = normalized["Node"]
        assert isinstance(target, tuple)
        module_name, attribute = target
        module = importlib.import_module(module_name)
        expected = importlib.import_module("xml.dom")
        assert getattr(module, attribute) is expected.Node

    def test_relative_path_cannot_escape_the_top_level_package(self) -> None:
        """An invalid parent traversal fails instead of manufacturing a module name."""
        with pytest.raises(ImportError):
            normalize_lazy_imports("xml", {"module": "..dom"})

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
        self, package: str, relative: str
    ) -> None:
        """Real child packages without lazy maps preserve the local exports."""
        merged = merge_lazy_imports(
            (relative,), {"Node": ("xml.dom", "Node")}, module_name=package
        )
        assert merged == {"Node": ("xml.dom", "Node")}

    def test_merge_child_cannot_escape_the_top_level_package(self) -> None:
        """Child discovery rejects traversal beyond the containing package."""
        with pytest.raises(ImportError):
            merge_lazy_imports(("..dom",), {}, module_name="xml")
