"""PEP 562 lazy export helpers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from ._lazy_parts.flextlazy_part_02 import FlextLazy, FlextLazyMember

lazy = FlextLazy()
"""Shared ``FlextLazy`` singleton used by package-level lazy exports."""
build_lazy_import_map = lazy.build_map
"""Convenience alias for building flat lazy import maps."""
lazy_getattr = lazy.get
lazy_member = lazy.member
resolve_lazy_members = lazy.resolve_members
cleanup_submodule_namespace = lazy.cleanup
normalize_lazy_imports = lazy.normalize_map
merge_lazy_imports = lazy.merge
install_lazy_exports = lazy.install

__all__ = (
    "FlextLazy",
    "FlextLazyMember",
    "build_lazy_import_map",
    "cleanup_submodule_namespace",
    "install_lazy_exports",
    "lazy",
    "lazy_getattr",
    "lazy_member",
    "merge_lazy_imports",
    "normalize_lazy_imports",
    "resolve_lazy_members",
)
