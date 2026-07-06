# AUTO-GENERATED FILE — Regenerate with: make gen
"""Result Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._result_parts.behavior import FlextResultBehaviorMixin
    from flext_core._result_parts.composition import FlextResultCompositionMixin
    from flext_core._result_parts.construction import FlextResultConstructionMixin
    from flext_core._result_parts.transforms import FlextResultTransformsMixin
    from flext_core._result_parts.unwrap import FlextResultUnwrapMixin
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".behavior": ("FlextResultBehaviorMixin",),
        ".composition": ("FlextResultCompositionMixin",),
        ".construction": ("FlextResultConstructionMixin",),
        ".transforms": ("FlextResultTransformsMixin",),
        ".unwrap": ("FlextResultUnwrapMixin",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
