"""PEP 562 lazy export helpers.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._lazy_parts.flextlazy_attribute import FlextLazyAttribute
from flext_core._lazy_parts.flextlazy_member import FlextLazyMember
from flext_core._lazy_parts.flextlazy_part_02 import FlextLazy

lazy = FlextLazy()
"""Shared ``FlextLazy`` singleton used by package-level lazy exports."""
build_lazy_import_map = lazy.build_map
"""Convenience alias for building flat lazy import maps."""
lazy_getattr = lazy.get
lazy_member = lazy.member
"""Convenience alias for declaring one lazy member descriptor."""
resolve_lazy_members = lazy.resolve_members
"""Convenience alias for resolving lazy member names."""
cleanup_submodule_namespace = lazy.cleanup
normalize_lazy_imports = lazy.normalize_map
merge_lazy_imports = lazy.merge
install_lazy_exports = lazy.install
"""Convenience alias for installing lazy exports into a module."""

__all__ = (
    "FlextLazy",
    "FlextLazyAttribute",
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
