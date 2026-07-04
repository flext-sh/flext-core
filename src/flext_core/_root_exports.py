"""Generated root lazy export configuration."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from ._root_exports_parts.all_names import ROOT_ALL
from ._root_exports_parts.exclude_names import ROOT_EXCLUDE_NAMES
from ._root_exports_parts.lazy_core import ROOT_LAZY_CORE
from ._root_exports_parts.lazy_facades import ROOT_LAZY_FACADES
from ._root_exports_parts.lazy_utilities import ROOT_LAZY_UTILITIES
from ._root_exports_parts.metadata_names import ROOT_METADATA_NAMES
from ._root_exports_parts.typing_only_names import ROOT_TYPING_ONLY_NAMES

if TYPE_CHECKING:
    from collections.abc import Mapping

ROOT_PACKAGE_MODULES: Final[tuple[str, ...]] = ()
ROOT_LAZY_MODULES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    **ROOT_LAZY_CORE,
    **ROOT_LAZY_UTILITIES,
    **ROOT_LAZY_FACADES,
})


__all__: list[str] = [
    "ROOT_ALL",
    "ROOT_EXCLUDE_NAMES",
    "ROOT_LAZY_MODULES",
    "ROOT_METADATA_NAMES",
    "ROOT_PACKAGE_MODULES",
    "ROOT_TYPING_ONLY_NAMES",
]
