"""Generated root lazy export map: lazy_facades."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

ROOT_LAZY_FACADES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    ".constants": ("FlextConstants", "c"),
    ".container": ("FlextContainer",),
    ".context": ("FlextContext",),
    ".decorators": ("FlextDecorators", "d"),
    ".dispatcher": ("FlextDispatcher",),
    ".exceptions": ("FlextExceptions", "e"),
    ".handlers": ("FlextHandlers", "h"),
    ".lazy": ("FlextLazy", "build_lazy_import_map", "lazy", "normalize_lazy_imports"),
    ".loggings": ("FlextUtilitiesLogging",),
    ".mixins": ("FlextMixins", "x"),
    ".models": ("FlextModels", "m"),
    ".protocols": ("FlextProtocols", "p"),
    ".registry": ("FlextRegistry",),
    ".result": ("FlextResult", "r"),
    ".runtime": ("FlextRuntime",),
    ".service": ("FlextService", "s"),
    ".settings": ("FlextSettings",),
    ".typings": ("FlextTypes", "t"),
    ".utilities": ("FlextUtilities", "u"),
})

__all__: list[str] = ["ROOT_LAZY_FACADES"]
