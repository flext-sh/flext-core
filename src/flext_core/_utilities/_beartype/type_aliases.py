"""Evaluate aliases after classifying source-proven static-only dependencies.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import ModuleType
from typing import Annotated, TypeAliasType, get_args, get_origin

from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._protocols.base import FlextProtocolsBase as p
from flext_core._utilities._beartype.module_source import (
    FlextUtilitiesBeartypeModuleSource,
)


class FlextUtilitiesBeartypeTypeAliases:
    """Own typed alias resolution; unexpected evaluation errors escape unchanged."""

    @staticmethod
    def resolve(
        alias: TypeAliasType,
        *,
        owner: ModuleType | type | None = None,
    ) -> me.ResolvedAlias | me.DeferredAlias:
        if owner is not None:
            deferred = FlextUtilitiesBeartypeModuleSource.deferred(alias, owner=owner)
            if deferred is not None:
                return deferred
        return me.ResolvedAlias(value=alias.__value__)

    @classmethod
    def deferred(  # ruff: ignore[too-many-arguments] -- public resolver surface mirroring the deferred-alias probe options; the keyword flags are the stable API.
        cls,
        hint: p.AttributeProbe,
        *,
        recursive: bool = False,
        unwrap_annotated: bool = False,
        inspect_origin: bool = False,
        owner: ModuleType | type | None = None,
        seen: set[int] | None = None,
    ) -> tuple[me.DeferredAlias, ...]:
        """Collect only aliases the requesting predicate actually evaluates.

        Returns:
            The resulting ``tuple[me.DeferredAlias, ...]``.

        """
        visited = set() if seen is None else seen
        if id(hint) in visited:
            return ()
        visited.add(id(hint))
        if isinstance(hint, TypeAliasType):
            resolution = cls.resolve(hint, owner=owner)
            if isinstance(resolution, me.DeferredAlias):
                return (resolution,)
            return cls.deferred(
                resolution.value,
                recursive=recursive,
                unwrap_annotated=unwrap_annotated,
                inspect_origin=inspect_origin,
                owner=owner,
                seen=visited,
            )
        origin = get_origin(hint)
        if hint is type or origin is type:
            return ()
        args = get_args(hint)
        if unwrap_annotated and origin is Annotated and args:
            return cls.deferred(
                args[0],
                unwrap_annotated=True,
                inspect_origin=inspect_origin,
                owner=owner,
                seen=visited,
            )
        if inspect_origin and isinstance(origin, TypeAliasType):
            return cls.deferred(origin, owner=owner, seen=visited)
        if not recursive:
            return ()
        return tuple(
            deferred
            for child in args
            for deferred in cls.deferred(
                child,
                recursive=True,
                owner=owner,
                seen=visited,
            )
        )
