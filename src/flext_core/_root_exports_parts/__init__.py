# AUTO-GENERATED FILE — Regenerate with: make gen
"""Root Exports Parts package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core._root_exports_parts.all_names import ROOT_ALL as ROOT_ALL
    from flext_core._root_exports_parts.exclude_names import (
        ROOT_EXCLUDE_NAMES as ROOT_EXCLUDE_NAMES,
    )
    from flext_core._root_exports_parts.lazy_core import (
        ROOT_LAZY_CORE as ROOT_LAZY_CORE,
    )
    from flext_core._root_exports_parts.lazy_facades import (
        ROOT_LAZY_FACADES as ROOT_LAZY_FACADES,
    )
    from flext_core._root_exports_parts.lazy_utilities import (
        ROOT_LAZY_UTILITIES as ROOT_LAZY_UTILITIES,
    )
    from flext_core._root_exports_parts.metadata_names import (
        ROOT_METADATA_NAMES as ROOT_METADATA_NAMES,
    )
    from flext_core._root_exports_parts.typing_only_names import (
        ROOT_TYPING_ONLY_NAMES as ROOT_TYPING_ONLY_NAMES,
    )
_LAZY_IMPORTS = build_lazy_import_map(
    {
        ".all_names": ("ROOT_ALL",),
        ".exclude_names": ("ROOT_EXCLUDE_NAMES",),
        ".lazy_core": ("ROOT_LAZY_CORE",),
        ".lazy_facades": ("ROOT_LAZY_FACADES",),
        ".lazy_utilities": ("ROOT_LAZY_UTILITIES",),
        ".metadata_names": ("ROOT_METADATA_NAMES",),
        ".typing_only_names": ("ROOT_TYPING_ONLY_NAMES",),
    },
)


install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    publish_all=False,
)
