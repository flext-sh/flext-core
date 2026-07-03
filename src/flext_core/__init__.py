# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core package."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.__version__ import (
    __author__,
    __author_email__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
    __version_info__,
)
from flext_core._root_exports import (
    ROOT_ALL,
    ROOT_LAZY_MODULES,
    ROOT_METADATA_NAMES,
)
from flext_core.lazy import build_lazy_import_map, install_lazy_exports

if TYPE_CHECKING:
    from flext_core.constants import FlextConstants as FlextConstants, c as c
    from flext_core.container import FlextContainer as FlextContainer
    from flext_core.context import FlextContext as FlextContext
    from flext_core.decorators import FlextDecorators as FlextDecorators, d as d
    from flext_core.dispatcher import FlextDispatcher as FlextDispatcher
    from flext_core.exceptions import FlextExceptions as FlextExceptions, e as e
    from flext_core.handlers import FlextHandlers as FlextHandlers, h as h
    from flext_core.lazy import FlextLazy as FlextLazy
    from flext_core.loggings import FlextLogger as FlextLogger
    from flext_core.mixins import FlextMixins as FlextMixins, x as x
    from flext_core.models import FlextModels as FlextModels, m as m
    from flext_core.protocols import FlextProtocols as FlextProtocols, p as p
    from flext_core.registry import FlextRegistry as FlextRegistry
    from flext_core.result import FlextResult as FlextResult, r as r
    from flext_core.runtime import FlextRuntime as FlextRuntime
    from flext_core.service import FlextService as FlextService, s as s
    from flext_core.settings import FlextSettings as FlextSettings
    from flext_core.typings import FlextTypes as FlextTypes, t as t
    from flext_core.utilities import FlextUtilities as FlextUtilities, u as u

_LAZY_IMPORTS = build_lazy_import_map(ROOT_LAZY_MODULES, sort_keys=False)
_PUBLISHED_NAMES = frozenset({*_LAZY_IMPORTS, *ROOT_METADATA_NAMES})
_PUBLIC_NAMES = frozenset(ROOT_ALL)
_ROOT_EXPORTS_DRIFT_ERROR = "flext_core root exports drift from ROOT_ALL"
if not _PUBLIC_NAMES <= _PUBLISHED_NAMES:
    raise RuntimeError(_ROOT_EXPORTS_DRIFT_ERROR)

install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    public_exports=ROOT_ALL,
)
