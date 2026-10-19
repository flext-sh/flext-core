"""Evaluate aliases after classifying source-proven static-only dependencies.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import builtins
from collections import ChainMap
from types import ModuleType
from typing import Annotated, ForwardRef, TypeAliasType, get_args, get_origin

from pydantic._internal._model_construction import unpack_lenient_weakvaluedict
from pydantic._internal._namespace_utils import NamespacesTuple, NsResolver

from flext_core._models import FlextModelsEnforcement
from flext_core._protocols import FlextProtocolsBase
from flext_core._typings import FlextTypingBase
from flext_core._utilities._beartype import FlextUtilitiesBeartypeModuleSource


class FlextUtilitiesBeartypeTypeAliases:
    """Own typed alias resolution; unexpected evaluation errors escape unchanged."""

    @staticmethod
    def declared_origin(
        hint: str | ForwardRef,
        *,
        owner: ModuleType | type | None,
    ) -> type | TypeAliasType:
        """Look up the outer contract without evaluating its element expression."""
        if owner is None:
            msg = f"Declaring namespace required for annotation {hint!r}"
            raise ValueError(msg)
        resolver = NsResolver(
            parent_namespace=unpack_lenient_weakvaluedict(
                getattr(owner, "__pydantic_parent_namespace__", None),
            ),
        )
        if isinstance(owner, ModuleType):
            namespaces = NamespacesTuple(vars(owner), {})
        else:
            with resolver.push(owner):
                namespaces = resolver.types_namespace
        namespace = ChainMap(namespaces.locals, namespaces.globals, vars(builtins))
        node = ast.parse(
            hint.__forward_arg__ if isinstance(hint, ForwardRef) else hint,
            mode="eval",
        ).body
        while True:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node = ast.parse(node.value, mode="eval").body
                continue
            expression = node.value if isinstance(node, ast.Subscript) else node
            attributes: list[str] = []
            while isinstance(expression, ast.Attribute):
                attributes.insert(0, expression.attr)
                expression = expression.value
            if not isinstance(expression, ast.Name):
                msg = f"Annotation has no named outer contract: {hint!r}"
                raise TypeError(msg)
            binding: FlextProtocolsBase.AttributeProbe = namespace[expression.id]
            for attribute in attributes:
                binding = getattr(binding, attribute)
            if binding is Annotated and isinstance(node, ast.Subscript):
                if not isinstance(node.slice, ast.Tuple):
                    msg = f"Annotated contract requires metadata: {hint!r}"
                    raise TypeError(msg)
                node = node.slice.elts[0]
                continue
            if isinstance(binding, (type, TypeAliasType)):
                return binding
            origin = get_origin(binding)
            if isinstance(origin, type):
                return origin
            msg = f"Annotation outer binding is not a type or alias: {hint!r}"
            raise TypeError(msg)

    @staticmethod
    def resolve(
        alias: TypeAliasType,
        *,
        owner: ModuleType | type | None = None,
    ) -> FlextModelsEnforcement.ResolvedAlias | FlextModelsEnforcement.DeferredAlias:
        if owner is not None:
            deferred = FlextUtilitiesBeartypeModuleSource.deferred_alias(
                alias, owner=owner
            )
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


__all__: tuple[str, ...] = ("FlextUtilitiesBeartypeTypeAliases",)
