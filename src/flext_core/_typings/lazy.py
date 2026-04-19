"""Lazy export typing contracts for ``flext_core.lazy`` internals."""

from __future__ import annotations

from collections.abc import Mapping


class FlextTypingLazy:
    """Dedicated typing namespace for lazy export machinery."""

    type LazyImportEntry = str | tuple[str, str]
    type LazyImportMap = Mapping[str, LazyImportEntry]
    type NormalizedMapCacheKey = tuple[str, int]
    type InstallCacheSignature = tuple[int, int, int, bool]
