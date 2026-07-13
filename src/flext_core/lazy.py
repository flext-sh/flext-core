"""PEP 562 lazy export helpers."""

from __future__ import annotations

# mro-wkii.17.26 (codex): private package inits are inert; import the leaf owner.
from flext_core._lazy_parts.flextlazy_part_02 import FlextLazy, FlextLazyAttribute

lazy = FlextLazy()
"""Shared ``FlextLazy`` singleton used by package-level lazy exports."""
build_lazy_import_map = lazy.build_map
"""Convenience alias for building flat lazy import maps."""
lazy_getattr = lazy.get
lazy_attribute = lazy.attribute
cleanup_submodule_namespace = lazy.cleanup
normalize_lazy_imports = lazy.normalize_map
merge_lazy_imports = lazy.merge
install_lazy_exports = lazy.install

__all__ = (
    "FlextLazy",
    "FlextLazyAttribute",
    "build_lazy_import_map",
    "cleanup_submodule_namespace",
    "install_lazy_exports",
    "lazy",
    "lazy_attribute",
    "lazy_getattr",
    "merge_lazy_imports",
    "normalize_lazy_imports",
)
