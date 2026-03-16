"""FlextProtocolsContext - context and bootstrap protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import ModuleType
from typing import TYPE_CHECKING, Protocol, Self, overload, runtime_checkable

from pydantic import BaseModel
from pydantic_settings import BaseSettings

from flext_core import t

if TYPE_CHECKING:
    from flext_core import r


class FlextProtocolsContext:
    """Protocols for context and runtime bootstrap options."""

    @runtime_checkable
    class Context(Protocol):
        """Context protocol for type safety without circular imports.

        Defined in protocols.py to keep all protocol definitions together.
        Full context protocol p.Context extends this minimal interface.

        Methods use generic return types (Any) for structural compatibility
        with p.Context which uses ResultLike[T] (also covariant with Any).
        """

        def clone(self) -> Self:
            """Clone context for isolated execution."""
            ...

        def get(self, key: str, scope: str = ...) -> r[t.Container | BaseModel]:
            """Get a context value. Returns Result-like object."""
            ...

        @overload
        def set(
            self, key_or_data: str, value: t.Container | BaseModel, *, scope: str = ...
        ) -> r[bool]: ...

        @overload
        def set(
            self,
            key_or_data: t.ConfigMap,
            value: None = ...,
            *,
            scope: str = ...,
        ) -> r[bool]: ...

        def set(
            self,
            key_or_data: str | t.ConfigMap,
            value: t.Container | BaseModel | None = ...,
            *,
            scope: str = ...,
        ) -> r[bool]:
            """Set a context value. Returns Result-like object."""
            ...

    @runtime_checkable
    class RuntimeBootstrapOptions(Protocol):
        """Runtime bootstrap options for service initialization."""

        config_type: type[BaseSettings] | None
        config_overrides: Mapping[str, t.Scalar] | None
        context: FlextProtocolsContext.Context | None
        subproject: str | None
        services: Mapping[str, object] | None
        factories: Mapping[str, Callable[..., object]] | None
        resources: Mapping[str, Callable[..., object]] | None
        container_overrides: Mapping[str, t.Scalar] | None
        wire_modules: Sequence[ModuleType | str] | None
        wire_packages: Sequence[str] | None
        wire_classes: Sequence[type] | None


__all__ = ["FlextProtocolsContext"]
