"""Lazy export typing contracts for ``flext_core.lazy`` internals.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from types import ModuleType

from flext_core._typings.base import FlextTypingBase

type FlextLazyModuleGlobalValue = (
    FlextTypingBase.JsonValue
    | FlextTypingBase.LazyImportMap
    | FlextTypingBase.StrSequence
    | ModuleType
    | type
    | Callable[
        ...,
        FlextTypingBase.JsonValue
        | FlextTypingBase.StrSequence
        | ModuleType
        | type
        | None,
    ]
    | None
)


class FlextTypesLazy:
    """Typing namespace for package-level lazy export internals."""

    type ModuleGlobalValue = FlextLazyModuleGlobalValue
    type ModuleGlobals = MutableMapping[
        str,
        FlextLazyModuleGlobalValue | Callable[..., FlextLazyModuleGlobalValue],
    ]


__all__: list[str] = ["FlextTypesLazy"]
