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
    def _deferred_step(
        cls,
        node: FlextTypingBase.TypeFormSpecifier
        | FlextProtocolsBase.AttributeProbe
        | None,
        state: tuple[bool, bool, bool],
        owner: ModuleType | type | None,
    ) -> tuple[
        FlextModelsEnforcement.DeferredAlias | None,
        list[
            tuple[
                FlextTypingBase.TypeFormSpecifier
                | FlextProtocolsBase.AttributeProbe
                | None,
                tuple[bool, bool, bool],
            ]
        ],
    ]:
        """Classify one worklist node into a deferred hit or its successors.

        ``state`` carries ``(recursive, unwrap_annotated, inspect_origin)``.

        Returns:
            The resulting ``(deferred | None, successors)`` pair.

        """
        recurse, unwrap, inspect_o = state
        if isinstance(node, TypeAliasType):
            resolution = cls.resolve(node, owner=owner)
            return (
                (resolution, [])
                if isinstance(resolution, FlextModelsEnforcement.DeferredAlias)
                else (None, [(resolution.value, state)])
            )
        origin = get_origin(node)
        if node is type or origin is type:
            return None, []
        successors: list[
            tuple[
                FlextTypingBase.TypeFormSpecifier
                | FlextProtocolsBase.AttributeProbe
                | None,
                tuple[bool, bool, bool],
            ]
        ] = []
        if unwrap and origin is Annotated and get_args(node):
            successors = [(get_args(node)[0], (False, True, inspect_o))]
        elif inspect_o and isinstance(origin, TypeAliasType):
            successors = [(origin, (False, False, False))]
        elif recurse:
            successors = [(child, (True, False, False)) for child in get_args(node)]
        return None, successors

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
        visited: set[int] = set()
        pending: list[
            tuple[
                FlextTypingBase.TypeFormSpecifier
                | FlextProtocolsBase.AttributeProbe
                | None,
                tuple[bool, bool, bool],
            ]
        ] = [(hint, scan)]
        deferred_aliases: list[FlextModelsEnforcement.DeferredAlias] = []
        while pending:
            node, state = pending.pop(0)
            if id(node) in visited:
                continue
            visited.add(id(node))
            resolution, successors = cls._deferred_step(node, state, owner)
            if resolution is not None:
                deferred_aliases.append(resolution)
            pending[:0] = successors
        return tuple(deferred_aliases)
