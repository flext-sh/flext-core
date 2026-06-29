# AUTO-GENERATED FILE — Regenerate with: make gen
"""Root Exports Parts package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if _t.TYPE_CHECKING:
    from flext_core._root_exports_parts.lazy_core import (
        ROOT_LAZY_CORE as ROOT_LAZY_CORE,
    )
    from flext_core._root_exports_parts.lazy_facades import (
        ROOT_LAZY_FACADES as ROOT_LAZY_FACADES,
    )
    from flext_core._root_exports_parts.lazy_utilities import (
        ROOT_LAZY_UTILITIES as ROOT_LAZY_UTILITIES,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".lazy_core": ("ROOT_LAZY_CORE",),
        ".lazy_facades": ("ROOT_LAZY_FACADES",),
        ".lazy_utilities": ("ROOT_LAZY_UTILITIES",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
