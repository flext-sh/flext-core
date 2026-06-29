# AUTO-GENERATED FILE — Regenerate with: make gen
"""Shared Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from examples._shared_parts.shared_part_01 import (
        ExamplesFlextSharedBase as ExamplesFlextSharedBase,
    )
    from examples._shared_parts.shared_part_02 import (
        ExamplesFlextShared as ExamplesFlextShared,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".shared_part_01": ("ExamplesFlextSharedBase",),
        ".shared_part_02": ("ExamplesFlextShared",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
