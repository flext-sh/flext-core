"""PEP 562 lazy export helpers."""

from __future__ import annotations

from ._lazy_parts import FlextLazy

lazy = FlextLazy()
"""Shared ``FlextLazy`` singleton used by package-level lazy exports."""
build_lazy_import_map = lazy.build_map
"""Convenience alias for building flat lazy import maps."""
lazy_getattr = lazy.get
cleanup_submodule_namespace = lazy.cleanup
normalize_lazy_imports = lazy.normalize_map
merge_lazy_imports = lazy.merge
install_lazy_exports = lazy.install

__all__ = (
    "FlextLazy",
    "build_lazy_import_map",
    "cleanup_submodule_namespace",
    "install_lazy_exports",
    "lazy",
    "lazy_getattr",
    "merge_lazy_imports",
    "normalize_lazy_imports",
)
