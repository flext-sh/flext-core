"""Type and module introspection helpers — annotation inspection + bytecode analysis.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import builtins
import inspect
import types as _types_mod
from collections.abc import Callable, MutableMapping, MutableSequence, MutableSet
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    ForwardRef,
    TypeAliasType,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel

from flext_core import c
from flext_core._utilities._beartype._helpers_parts import helpers_part_02

# Import directly from base modules to avoid a circular load through the public
# flext_core facade while this module is still being initialized.

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsBase
    from flext_core._typings import FlextTypingBase


class FlextUtilitiesBeartypeHelpers(helpers_part_02.FlextUtilitiesBeartypeHelpers):
    @staticmethod
    def module_filename_for(module: _types_mod.ModuleType) -> str | None:
        filename = getattr(module, "__file__", None)
        return filename if isinstance(filename, str) else None

    @staticmethod
    def object_module_for(
        obj: FlextProtocolsBase.AttributeProbe,
    ) -> _types_mod.ModuleType | None:
        module = inspect.getmodule(obj)
        return module if isinstance(module, _types_mod.ModuleType) else None

    @staticmethod
    def object_module_name_for(obj: FlextProtocolsBase.AttributeProbe) -> str | None:
        module = inspect.getmodule(obj)
        name = getattr(module, "__name__", None)
        return name if isinstance(name, str) else None

    @staticmethod
    def count_union_members(hint: FlextTypingBase.TypeHintSpecifier | None) -> int:
        h = FlextUtilitiesBeartypeHelpers
        h2 = h.unwrap_type_alias(hint)
        if h2 is None or get_origin(h2) not in {UnionType, Union}:
            return 0
        return sum(1 for a in get_args(h2) if a is not type(None))

    @staticmethod
    def matches_str_none_union(hint: FlextTypingBase.TypeFormSpecifier | None) -> bool:
        h = FlextUtilitiesBeartypeHelpers
        h2 = h.unwrap_type_alias(hint)
        if h2 is None or get_origin(h2) not in {UnionType, Union}:
            return False
        return str in (a := get_args(h2)) and type(None) in a

    @staticmethod
    def alias_contains_any(
        alias_value: FlextTypingBase.TypeFormSpecifier | None,
        *,
        owner: _types_mod.ModuleType | type | None = None,
    ) -> bool:
        h = FlextUtilitiesBeartypeHelpers
        return h.contains_any_recursive(alias_value, seen=set(), owner=owner)

    @staticmethod
    def mutable_kind(value: FlextProtocolsBase.AttributeProbe) -> str | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if isinstance(value, kind):
                return kind.__name__
        return None

    @staticmethod
    def mutable_default_factory_kind(
        factory: type | Callable[..., FlextProtocolsBase.AttributeProbe] | None,
    ) -> type | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if factory is kind or get_origin(factory) is kind:
                return kind
        return None

    @staticmethod
    def allows_mutable_default_factory(
        hint: FlextTypingBase.TypeFormSpecifier | None,
        factory: type | Callable[..., FlextProtocolsBase.AttributeProbe] | None,
        *,
        owner: type | None = None,
    ) -> bool:
        h = FlextUtilitiesBeartypeHelpers
        mk = h.mutable_default_factory_kind(factory)
        kinds: FlextTypingBase.MappingKV[type, type] = {
            list: MutableSequence,
            dict: MutableMapping,
            set: MutableSet,
        }
        exp = kinds.get(mk) if mk is not None else None
        if exp is None:
            return False
        norm = h.unwrap_annotated(hint)
        if isinstance(norm, (str, ForwardRef)):
            if owner is None:
                return False
            text = norm if isinstance(norm, str) else norm.__forward_arg__
            outer = FlextUtilitiesBeartypeHelpers._annotation_outer_type(
                text, FlextUtilitiesBeartypeHelpers._annotation_namespace(owner)
            )
            outer_origin = get_origin(outer)
            if isinstance(outer_origin, (type, TypeAliasType)):
                norm = outer_origin
            elif isinstance(outer, (type, TypeAliasType)):
                norm = outer
            else:
                return False
        origin = get_origin(norm)
        if isinstance(origin, TypeAliasType):
            norm = origin
        resolved = h.unwrap_annotated(h.unwrap_type_alias(norm))
        return (get_origin(resolved) or resolved) is exp

    @staticmethod
    def _annotation_namespace(
        owner: type,
    ) -> FlextTypingBase.MutableMappingKV[str, FlextProtocolsBase.AttributeProbe]:
        """Use live module and class bindings without interpreting private weak refs.

        Returns:
            Proven bindings; unresolved lexical shadowing remains unavailable.

        """
        namespace: FlextTypingBase.MutableMappingKV[
            str, FlextProtocolsBase.AttributeProbe
        ] = dict(vars(builtins))
        module = inspect.getmodule(owner)
        if module is not None:
            namespace.update(vars(module))
        if issubclass(owner, BaseModel):
            parent = owner.__pydantic_parent_namespace__
            if parent is not None:
                for name in parent:
                    namespace[name] = None
        for base in reversed(owner.__mro__):
            namespace.update(vars(base))
        return namespace

    @staticmethod
    def _annotation_reference(
        expression: ast.expr,
        namespace: FlextTypingBase.MappingKV[str, FlextProtocolsBase.AttributeProbe],
    ) -> FlextProtocolsBase.AttributeProbe | None:
        """Resolve only declared names and attributes, never executable expressions.

        Returns:
            The actual namespace binding, or None when it cannot be authenticated.

        """
        path: list[str] = []
        while isinstance(expression, ast.Attribute):
            path.insert(0, expression.attr)
            expression = expression.value
        if not isinstance(expression, ast.Name):
            return None
        value = namespace.get(expression.id)
        for name in path:
            scopes = value.__mro__ if isinstance(value, type) else (value,)
            value = next(
                (
                    vars(scope)[name]
                    for scope in scopes
                    if isinstance(scope, (type, _types_mod.ModuleType))
                    and name in vars(scope)
                ),
                None,
            )
        return value

    @staticmethod
    def _annotation_outer_type(
        source: str,
        namespace: FlextTypingBase.MappingKV[str, FlextProtocolsBase.AttributeProbe],
    ) -> FlextProtocolsBase.AttributeProbe | None:
        """Inspect the outer collection without evaluating unresolved item types.

        Returns:
            The actual outer type or alias, with Annotated metadata peeled away.

        """
        expression = ast.parse(source, mode="eval").body
        while isinstance(expression, ast.Constant) and isinstance(
            expression.value, str
        ):
            expression = ast.parse(expression.value, mode="eval").body
        while isinstance(expression, ast.Subscript):
            outer = FlextUtilitiesBeartypeHelpers._annotation_reference(
                expression.value, namespace
            )
            if outer is not Annotated:
                return outer
            if not isinstance(expression.slice, ast.Tuple) or not expression.slice.elts:
                return None
            expression = expression.slice.elts[0]
        return FlextUtilitiesBeartypeHelpers._annotation_reference(
            expression, namespace
        )

    @staticmethod
    def has_relaxed_extra_base(target: type) -> bool:
        return any(
            b.__name__ in c.ENFORCEMENT_RELAXED_EXTRA_BASES for b in target.__mro__
        )


__all__: list[str] = ["FlextUtilitiesBeartypeHelpers"]
