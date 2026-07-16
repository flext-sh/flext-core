"""Lazy export typing contracts for ``flext_core.lazy`` internals."""

from __future__ import annotations

from types import ModuleType


class FlextTypesLazy:
    """Typing namespace for package-level lazy export internals."""

    # NOTE (multi-agent, mro-0ftd.3.3.1): keep the low-tier export value finite;
    # structural model and callback capabilities belong to the protocols facet.
    type LazyModule = ModuleType


__all__: list[str] = ["FlextTypesLazy"]
