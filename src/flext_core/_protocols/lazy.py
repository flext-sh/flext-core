"""Structural contracts for PEP 562 lazy exports."""

from __future__ import annotations

from typing import Protocol

from flext_core._protocols.base import FlextProtocolsBase as p
from flext_core._typings.lazy import FlextTypesLazy as t


class FlextProtocolsLazy:
    """Finite contracts for resolved exports and installed module hooks."""

    # NOTE (multi-agent, mro-0ftd.3.3.1): separate resolved symbols from the
    # two exact PEP 562 callbacks so no callable or value aliases recurse.
    type ResolvedExport = t.LazyModule | p.Model | p.ModuleOwned

    class GetattrHook(Protocol):
        """Module ``__getattr__`` callback contract."""

        def __call__(self, name: str, /) -> FlextProtocolsLazy.ResolvedExport: ...

    class DirHook(Protocol):
        """Module ``__dir__`` callback contract."""

        def __call__(self) -> list[str]: ...

    type NamespaceValue = ResolvedExport | GetattrHook | DirHook | tuple[str, ...]

    class Namespace(Protocol):
        """Mutable operations required from a module globals namespace."""

        def __setitem__(
            self, key: str, value: FlextProtocolsLazy.NamespaceValue, /
        ) -> None: ...

        def __contains__(self, key: str, /) -> bool: ...

        def __delitem__(self, key: str, /) -> None: ...


__all__: list[str] = ["FlextProtocolsLazy"]
