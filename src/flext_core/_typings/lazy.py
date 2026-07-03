"""Lazy export typing contracts for ``flext_core.lazy`` internals."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from types import ModuleType

from .base import FlextTypingBase as t

type FlextLazyModuleGlobalValue = (
    t.JsonValue
    | t.LazyImportMap
    | t.StrSequence
    | ModuleType
    | type
    | Callable[..., FlextLazyModuleGlobalValue]
    | Callable[..., t.JsonValue | t.StrSequence | ModuleType | type | None]
    | None
)


class FlextTypesLazy:
    """Typing namespace for package-level lazy export internals."""

    type ModuleGlobalValue = FlextLazyModuleGlobalValue
    type ModuleGlobals = MutableMapping[str, FlextLazyModuleGlobalValue]


__all__: list[str] = ["FlextTypesLazy"]
