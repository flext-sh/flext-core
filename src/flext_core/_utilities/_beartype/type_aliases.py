"""Evaluate aliases after classifying source-proven static-only dependencies.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import ModuleType
from typing import Annotated, TypeAliasType, get_args, get_origin

from flext_core._models import FlextModelsEnforcement
from flext_core._protocols import FlextProtocolsBase
from flext_core._typings.base import FlextTypingBase
from flext_core._utilities import FlextUtilitiesBeartypeModuleSource


class FlextUtilitiesBeartypeTypeAliases:
    """Own typed alias resolution; unexpected evaluation errors escape unchanged."""

    @staticmethod
    def resolve(
        alias: TypeAliasType,
        *,
        owner: ModuleType | type | None = None,
    ) -> FlextModelsEnforcement.ResolvedAlias | FlextModelsEnforcement.DeferredAlias:
        if owner is not None:
            deferred = FlextUtilitiesBeartypeModuleSource.deferred(alias, owner=owner)
            if deferred is not None:
                return deferred
        return FlextModelsEnforcement.ResolvedAlias(value=alias.__value__)

    @classmethod
    def deferred(
        cls,
        hint: FlextTypingBase.TypeFormSpecifier
        | FlextProtocolsBase.AttributeProbe
        | None,
        *,
        recursive: bool = False,
        owner: ModuleType | type | None = None,
    ) -> tuple[FlextModelsEnforcement.DeferredAlias, ...]:
        """Collect only aliases the requesting predicate actually evaluates.

        Returns:
            The resulting ``tuple[me.DeferredAlias, ...]``.

        """
        return cls._deferred_scan(hint, (recursive, False, False), owner)

    @classmethod
    def deferred_annotated(
        cls,
        hint: FlextTypingBase.TypeFormSpecifier
        | FlextProtocolsBase.AttributeProbe
        | None,
        *,
        owner: ModuleType | type | None = None,
    ) -> tuple[FlextModelsEnforcement.DeferredAlias, ...]:
        """Collect deferred aliases unwrapping ``Annotated`` and its origins.

        Returns:
            The resulting ``tuple[me.DeferredAlias, ...]``.

        """
        return cls._deferred_scan(hint, (False, True, True), owner)

    @classmethod
    def _deferred_scan(
        cls,
        hint: FlextTypingBase.TypeFormSpecifier
        | FlextProtocolsBase.AttributeProbe
        | None,
        scan: tuple[bool, bool, bool],
        owner: ModuleType | type | None,
    ) -> tuple[FlextModelsEnforcement.DeferredAlias, ...]:
        """Walk the hint graph iteratively collecting deferred aliases.

        ``scan`` carries ``(recursive, unwrap_annotated, inspect_origin)``: the
        flag set the recursive formulation passed down, so ordering and
        semantics are preserved without growing the signature.

        Returns:
            The resulting ``tuple[me.DeferredAlias, ...]``.

        """
        recursive, unwrap_annotated, inspect_origin = scan
        visited: set[int] = set()
        pending: list[
            tuple[
                FlextTypingBase.TypeFormSpecifier
                | FlextProtocolsBase.AttributeProbe
                | None,
                bool,
                bool,
                bool,
            ]
        ] = [
            (hint, recursive, unwrap_annotated, inspect_origin),
        ]
        deferred_aliases: list[FlextModelsEnforcement.DeferredAlias] = []
        while pending:
            node, recurse, unwrap, inspect_o = pending.pop(0)
            if id(node) in visited:
                continue
            visited.add(id(node))
            if isinstance(node, TypeAliasType):
                resolution = cls.resolve(node, owner=owner)
                if isinstance(resolution, FlextModelsEnforcement.DeferredAlias):
                    deferred_aliases.append(resolution)
                    continue
                pending.insert(0, (resolution.value, recurse, unwrap, inspect_o))
                continue
            origin = get_origin(node)
            if node is type or origin is type:
                continue
            args = get_args(node)
            if unwrap and origin is Annotated and args:
                pending.insert(0, (args[0], False, True, inspect_o))
                continue
            if inspect_o and isinstance(origin, TypeAliasType):
                pending.insert(0, (origin, False, False, False))
                continue
            if not recurse:
                continue
            pending[:0] = [(child, True, False, False) for child in args]
        return tuple(deferred_aliases)
