"""Union-arm declaration analysis sidecar for the field visitor.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from types import UnionType
from typing import Annotated, TypeAliasType, Union, get_args, get_origin

from flext_core import t
from flext_core._utilities._beartype import FlextUtilitiesBeartypeHelpers


class FlextUtilitiesBeartypeFieldUnionScan:
    """Count declared union arms without expanding type aliases."""

    @classmethod
    def _ast_union_members(cls, node: ast.expr) -> tuple[ast.expr, ...]:
        """Return only the top-level members written in a union expression.

        Returns:
            Only the top-level members written in a union expression.

        """
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return (
                *cls._ast_union_members(node.left),
                *cls._ast_union_members(node.right),
            )
        if isinstance(node, ast.Subscript) and (
            (isinstance(node.value, ast.Name) and node.value.id == "Union")
            or (isinstance(node.value, ast.Attribute) and node.value.attr == "Union")
        ):
            if isinstance(node.slice, ast.Tuple):
                return tuple(node.slice.elts)
            return (node.slice,)
        return (node,)

    @classmethod
    def _string_union_members(cls, declared: str) -> int:
        """Count union arms inside a string annotation.

        Returns:
            The resulting ``int``.

        """
        unwrapped = FlextUtilitiesBeartypeHelpers.unwrap_annotated(declared)
        if not isinstance(unwrapped, str):
            return 0
        try:
            expression = ast.parse(unwrapped, mode="eval").body
        except SyntaxError:
            return 0
        members = cls._ast_union_members(expression)
        if len(members) == 1:
            return 0
        return sum(
            1
            for member in members
            if not (isinstance(member, ast.Name) and member.id == "None")
            and not (isinstance(member, ast.Constant) and member.value is None)
        )

    @classmethod
    def _declared_union_members(cls, annotation: t.TypeHintSpecifier | None) -> int:
        """Count union arms from declaration syntax without expanding aliases.

        Returns:
            The resulting ``int``.

        """
        declared = annotation
        if isinstance(declared, str):
            return cls._string_union_members(declared)
        if get_origin(declared) is Annotated:
            args = get_args(declared)
            declared = args[0] if args else declared
        if isinstance(declared, TypeAliasType) or get_origin(declared) not in {
            UnionType,
            Union,
        }:
            return 0
        return sum(1 for member in get_args(declared) if member is not type(None))


__all__: list[str] = ["FlextUtilitiesBeartypeFieldUnionScan"]
