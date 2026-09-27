"""Evaluate aliases after classifying source-proven static-only dependencies."""

from __future__ import annotations

from types import ModuleType
from typing import Annotated, TypeAliasType, get_args, get_origin

from ..._models.enforcement import FlextModelsEnforcement as me
from ..._protocols.base import FlextProtocolsBase as p
from .module_source import FlextUtilitiesBeartypeModuleSource


class FlextUtilitiesBeartypeTypeAliases:
    """Own typed alias resolution; unexpected evaluation errors escape unchanged."""

    @staticmethod
    def resolve(
        alias: TypeAliasType, *, owner: ModuleType | type | None = None
    ) -> me.ResolvedAlias | me.DeferredAlias:
        if owner is not None:
            deferred = FlextUtilitiesBeartypeModuleSource.deferred(alias, owner=owner)
            if deferred is not None:
                return deferred
        return me.ResolvedAlias(value=alias.__value__)

    @classmethod
    def deferred(
        cls,
        hint: p.AttributeProbe,
        *,
        recursive: bool = False,
        unwrap_annotated: bool = False,
        inspect_origin: bool = False,
        owner: ModuleType | type | None = None,
        seen: set[int] | None = None,
    ) -> tuple[me.DeferredAlias, ...]:
        """Collect only aliases the requesting predicate actually evaluates."""
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
                child, recursive=True, owner=owner, seen=visited
            )
        )
