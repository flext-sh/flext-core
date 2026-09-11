# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Tests. Utilities. Files package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from types import MappingProxyType

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from . import _comparison_parts as _comparison_parts
    from ._assertions import FlextTestsFilesAssertionsMixin
    from ._batch import FlextTestsFilesBatchMixin
    from ._comparison import FlextTestsFilesComparisonMixin
    from ._contexts import FlextTestsFilesContextsMixin
    from ._creation import FlextTestsFilesCreationMixin
    from ._info import FlextTestsFilesInfoMixin
    from ._lifecycle import FlextTestsFilesLifecycleMixin
    from ._reading import FlextTestsFilesReadingMixin
__all__: tuple[str, ...] = (
    "FlextTestsFilesAssertionsMixin",
    "FlextTestsFilesBatchMixin",
    "FlextTestsFilesComparisonMixin",
    "FlextTestsFilesContextsMixin",
    "FlextTestsFilesCreationMixin",
    "FlextTestsFilesInfoMixin",
    "FlextTestsFilesLifecycleMixin",
    "FlextTestsFilesReadingMixin",
    "_comparison_parts",
)

_LAZY_IMPORTS = MappingProxyType(
    build_lazy_import_map(
        MappingProxyType({
            "._assertions": ("FlextTestsFilesAssertionsMixin",),
            "._batch": ("FlextTestsFilesBatchMixin",),
            "._comparison": ("FlextTestsFilesComparisonMixin",),
            "._comparison_parts": ("_comparison_parts",),
            "._contexts": ("FlextTestsFilesContextsMixin",),
            "._creation": ("FlextTestsFilesCreationMixin",),
            "._info": ("FlextTestsFilesInfoMixin",),
            "._lifecycle": ("FlextTestsFilesLifecycleMixin",),
            "._reading": ("FlextTestsFilesReadingMixin",),
        }),
        alias_groups=MappingProxyType({}),
        sort_keys=False,
    )
)

install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=__all__)
