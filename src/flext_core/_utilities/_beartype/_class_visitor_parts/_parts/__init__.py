# AUTO-GENERATED FILE — Regenerate with: make gen
"""Parts package."""

from __future__ import annotations

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".class_visitor_part_02_01": ("class_visitor_part_02_01",),
        ".class_visitor_part_02_02": ("class_visitor_part_02_02",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
