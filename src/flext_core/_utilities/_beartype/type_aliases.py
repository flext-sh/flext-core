"""Evaluate aliases after classifying source-proven static-only dependencies.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import builtins
import inspect
from types import ModuleType, UnionType
from typing import (
    Annotated,
    ForwardRef,
    Optional,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel

from flext_core._models import FlextModelsEnforcement
from flext_core._protocols import FlextProtocolsBase
from flext_core._typings import FlextTypingBase
from flext_core._utilities._beartype import FlextUtilitiesBeartypeModuleSource


class FlextUtilitiesBeartypeTypeAliases:
    """Own typed alias resolution; unexpected evaluation errors escape unchanged."""

    @staticmethod
    def _declaring_namespace(
        owner: ModuleType | type,
    ) -> FlextTypingBase.MappingKV[str, FlextProtocolsBase.AttributeProbe]:
        """Read public module and class declarations without private model state.

        Returns:
            The declaring namespace, including enclosing class declarations.

        Raises:
            TypeError: If the owner has no inspectable defining module.

        """
        namespace: dict[str, FlextProtocolsBase.AttributeProbe] = dict(vars(builtins))
        if isinstance(owner, ModuleType):
            namespace.update(vars(owner))
            return namespace
        module = inspect.getmodule(owner)
        if module is None:
            msg = f"Annotation owner has no defining module: {owner!r}"
            raise TypeError(msg)
        namespace.update(vars(module))
        enclosing: FlextProtocolsBase.AttributeProbe = module
        for name in owner.__qualname__.split(".")[:-1]:
            if name == "<locals>":
                break
            enclosing = getattr(enclosing, name)
            if isinstance(enclosing, type):
                namespace.update(vars(enclosing))
        namespace.update(vars(owner))
        namespace[owner.__name__] = owner
        return namespace

    @staticmethod
    def _model_annotation_origin(
        hint: str | ForwardRef,
        owner: ModuleType | type,
    ) -> tuple[bool, type | TypeAliasType | None]:
        """Use Pydantic's public resolved fields for captured local declarations.

        Returns:
            Whether a resolved field matched, and its exact outer contract.

        """
        if not isinstance(owner, type) or not issubclass(owner, BaseModel):
            return False, None
        source = hint.__forward_arg__ if isinstance(hint, ForwardRef) else hint
        declarations = inspect.get_annotations(owner, eval_str=False)
        for name, field in owner.model_fields.items():
            if declarations.get(name) != source or isinstance(
                field.annotation, (str, ForwardRef)
            ):
                continue
            origin = get_origin(field.annotation)
            if origin in {Union, UnionType}:
                return True, None
            return True, FlextUtilitiesBeartypeTypeAliases._runtime_origin(
                field.annotation, hint
            )
        return False, None

    @staticmethod
    def _outer_binding(
        expression: ast.expr,
        namespace: FlextTypingBase.MappingKV[str, FlextProtocolsBase.AttributeProbe],
        hint: str | ForwardRef,
    ) -> FlextProtocolsBase.AttributeProbe:
        """Resolve one dotted outer binding without evaluating the annotation.

        Returns:
            The binding selected from the declaring namespace.

        Raises:
            TypeError: If the expression has no named outer contract.

        """
        attributes: list[str] = []
        while isinstance(expression, ast.Attribute):
            attributes.insert(0, expression.attr)
            expression = expression.value
        if not isinstance(expression, ast.Name):
            msg = f"Annotation has no named outer contract: {hint!r}"
            raise TypeError(msg)
        binding = namespace[expression.id]
        for attribute in attributes:
            binding = getattr(binding, attribute)
        return binding

    @staticmethod
    def _runtime_origin(
        binding: FlextTypingBase.TypeFormSpecifier
        | FlextProtocolsBase.AttributeProbe
        | None,
        hint: str | ForwardRef,
    ) -> type | TypeAliasType:
        """Require an exact runtime type or alias for the outer binding.

        Returns:
            The runtime type or its declared alias.

        Raises:
            TypeError: If the binding has no exact runtime origin.

        """
        if isinstance(binding, (type, TypeAliasType)):
            return binding
        origin = get_origin(binding)
        if isinstance(origin, type):
            return origin
        msg = f"Annotation outer binding is not a type or alias: {hint!r}"
        raise TypeError(msg)

    @staticmethod
    def declared_origin(
        hint: str | ForwardRef,
        *,
        owner: ModuleType | type | None,
    ) -> type | TypeAliasType | None:
        """Look up the outer contract; unions are not exact mutable origins.

        Returns:
            The exact outer type or alias, or None for a union.

        Raises:
            TypeError: If an Annotated contract does not declare metadata.
            ValueError: If the declaring namespace is absent.

        """
        if owner is None:
            msg = f"Declaring namespace required for annotation {hint!r}"
            raise ValueError(msg)
        matched, model_origin = (
            FlextUtilitiesBeartypeTypeAliases._model_annotation_origin(hint, owner)
        )
        if matched:
            return model_origin
        namespace = FlextUtilitiesBeartypeTypeAliases._declaring_namespace(owner)
        node = ast.parse(
            hint.__forward_arg__ if isinstance(hint, ForwardRef) else hint,
            mode="eval",
        ).body
        while True:
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                node = ast.parse(node.value, mode="eval").body
                continue
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
                return None
            expression = node.value if isinstance(node, ast.Subscript) else node
            binding = FlextUtilitiesBeartypeTypeAliases._outer_binding(
                expression, namespace, hint
            )
            if isinstance(node, ast.Subscript) and (
                binding is Optional or binding is Union
            ):
                return None
            if binding is Annotated and isinstance(node, ast.Subscript):
                if not isinstance(node.slice, ast.Tuple):
                    msg = f"Annotated contract requires metadata: {hint!r}"
                    raise TypeError(msg)
                node = node.slice.elts[0]
                continue
            return FlextUtilitiesBeartypeTypeAliases._runtime_origin(binding, hint)

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
