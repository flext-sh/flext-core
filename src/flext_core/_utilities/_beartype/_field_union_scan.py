"""Union-arm declaration analysis sidecar for the field visitor.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
from types import UnionType
from typing import Annotated, TypeAliasType, Union, get_args, get_origin

from flext_core._utilities import FlextUtilitiesBeartypeHelpers


def ast_union_members(node: ast.expr) -> tuple[ast.expr, ...]:
    """Return only the top-level members written in a union expression.

    Returns:
        Only the top-level members written in a union expression.

    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return (*ast_union_members(node.left), *ast_union_members(node.right))
    if isinstance(node, ast.Subscript) and (
        (isinstance(node.value, ast.Name) and node.value.id == "Union")
        or (isinstance(node.value, ast.Attribute) and node.value.attr == "Union")
    ):
        if isinstance(node.slice, ast.Tuple):
            return tuple(node.slice.elts)
        return (node.slice,)
    return (node,)


def string_union_members(declared: str) -> int:
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
    members = ast_union_members(expression)
    if len(members) == 1:
        return 0
    return sum(
        1
        for member in members
        if not (isinstance(member, ast.Name) and member.id == "None")
        and not (isinstance(member, ast.Constant) and member.value is None)
    )


def declared_union_members(annotation: object | None) -> int:
    """Count union arms from declaration syntax without expanding aliases.

    Returns:
        The resulting ``int``.

    """
    declared = annotation
    if isinstance(declared, str):
        return string_union_members(declared)
    if get_origin(declared) is Annotated:
        args = get_args(declared)
        declared = args[0] if args else declared
    if isinstance(declared, TypeAliasType) or get_origin(declared) not in {
        UnionType,
        Union,
    }:
        return 0
    return sum(1 for member in get_args(declared) if member is not type(None))


__all__: list[str] = ["declared_union_members"]
