"""Deferred type-alias proof sidecar for module-source inspection.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import builtins
import inspect
import sys
import textwrap
from types import ModuleType
from typing import TypeAliasType

from flext_core import t
from flext_core._models import FlextModelsEnforcement
from flext_core._utilities._beartype import (
    bound_names,
    declarations,
    parse_module,
    scope_nodes,
)


def _flag_guarded(
    condition: ast.expr,
    flag_imports: t.MappingKV[str, ast.ImportFrom],
    stable: frozenset[str],
) -> bool:
    """Prove one ``if`` guards a ``TYPE_CHECKING`` flag import.

    Returns:
        The resulting ``bool``.

    """
    return (
        isinstance(condition, ast.Name)
        and condition.id in flag_imports
        and condition.id in stable
    )


def _module_guarded(
    condition: ast.expr,
    module_imports: t.MappingKV[str, ast.Import],
    stable: frozenset[str],
) -> bool:
    """Prove one ``if`` guards a ``typing`` module ``TYPE_CHECKING`` block.

    Returns:
        The resulting ``bool``.

    """
    if not isinstance(condition, ast.Attribute) or condition.attr != "TYPE_CHECKING":
        return False
    if not isinstance(condition.value, ast.Name):
        return False
    return condition.value.id in module_imports and condition.value.id in stable


def guarded_imports(scopes: tuple[ast.Module | ast.ClassDef, ...]) -> set[str]:
    """Collect names imported behind provably-stable ``TYPE_CHECKING`` guards.

    Returns:
        The resulting ``set[str]``.

    """
    nodes = tuple(node for scope in scopes for node in scope_nodes(scope))
    flag_imports: t.MutableMappingKV[str, ast.ImportFrom] = {
        item.asname or item.name: node
        for node in nodes
        if isinstance(node, ast.ImportFrom) and node.module == "typing"
        for item in node.names
        if item.name == "TYPE_CHECKING"
    }
    module_imports: t.MutableMappingKV[str, ast.Import] = {
        item.asname or item.name: node
        for node in nodes
        if isinstance(node, ast.Import)
        for item in node.names
        if item.name == "typing"
    }
    markers = {**flag_imports, **module_imports}
    stable = frozenset(
        name
        for name, declaration in markers.items()
        if not any(
            name in bound_names(node) and node is not declaration for node in nodes
        )
    )
    imports: list[ast.Import | ast.ImportFrom] = []
    for node in nodes:
        if not isinstance(node, ast.If):
            continue
        condition = node.test
        if _flag_guarded(condition, flag_imports, stable) or _module_guarded(
            condition,
            module_imports,
            stable,
        ):
            imports.extend(
                item
                for item in node.body
                if isinstance(item, (ast.Import, ast.ImportFrom))
            )
    names = {name for node in imports for name in bound_names(node)}
    return {
        name
        for name in names
        if not any(name in bound_names(node) and node not in imports for node in nodes)
    }


def _deferred_module(
    alias: TypeAliasType, owner: ModuleType | type
) -> ModuleType | None:
    """Return the loaded defining module of ``alias`` when provable.

    Returns:
        The resulting ``ModuleType | None``.

    """
    if vars(owner).get(alias.__name__) is not alias:
        return None
    module_name = alias.__module__
    module = sys.modules.get(module_name) if module_name is not None else None
    if module is None or not isinstance(vars(module).get("__file__"), str):
        return None
    return module


def _declaration_entry(
    alias: TypeAliasType,
    owner: ModuleType | type,
    source_tree: ast.Module,
) -> tuple[ast.TypeAlias, tuple[ast.Module | ast.ClassDef, ...]] | None:
    """Resolve the unique declaring entry of ``alias`` in the owner source.

    Returns:
        The resulting ``(declaration, scopes) | None``.

    Raises:
        ValueError: If Ambiguous type alias declaration; or if Ambiguous type alias
            binding.

    """
    candidates = tuple(
        entry
        for entry in declarations(source_tree)
        if entry[0][-1:] == (alias.__name__,)
        and (
            isinstance(owner, ModuleType)
            or entry[0][-2:] == (owner.__name__, alias.__name__)
        )
    )
    if not candidates:
        return None
    if len(candidates) != 1:
        msg = f"Ambiguous type alias declaration: {alias.__module__}.{alias.__name__}"
        raise ValueError(msg)
    path, declaration, scopes = candidates[0]
    if any(
        alias.__name__ in bound_names(node) and node is not declaration.name
        for node in scope_nodes(scopes[-1])
    ):
        msg = f"Ambiguous type alias binding: {alias.__module__}.{'.'.join(path)}"
        raise ValueError(msg)
    return declaration, scopes


def _unavailable_names(
    declaration: ast.TypeAlias,
    available: set[str],
) -> set[str]:
    """Collect loaded names the alias value consumes but cannot resolve.

    Returns:
        The resulting ``set[str]``.

    """
    return {
        node.id
        for node in ast.walk(declaration.value)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in available
    }


def deferred_alias(
    alias: TypeAliasType,
    *,
    owner: ModuleType | type,
) -> FlextModelsEnforcement.DeferredAlias | None:
    """Prove deferral from the explicitly supplied declaring owner.

    Ambiguous declaration or binding failures propagate from the declaration
    resolver.

    Returns:
        The resulting ``me.DeferredAlias | None``.

    """
    module = _deferred_module(alias, owner)
    if module is None:
        return None
    source_lines, source_start = inspect.getsourcelines(owner)
    source_tree = ast.parse(
        textwrap.dedent("".join(source_lines)),
        filename=inspect.getfile(owner),
    )
    entry = _declaration_entry(alias, owner, source_tree)
    if entry is None:
        return None
    declaration, scopes = entry
    available = set(vars(builtins)) | set(vars(module))
    available.update(vars(owner))
    available.update(item.__name__ for item in getattr(owner, "__type_params__", ()))
    available.update(item.__name__ for item in alias.__type_params__)
    missing = _unavailable_names(declaration, available)
    guarded = guarded_imports((parse_module(module),)) | guarded_imports(scopes)
    if not missing or not missing <= guarded:
        return None
    return FlextModelsEnforcement.DeferredAlias(
        module=module.__name__,
        qualname=f"{owner.__qualname__}.{alias.__name__}"
        if isinstance(owner, type)
        else alias.__name__,
        file_path=inspect.getfile(module),
        line_number=source_start + declaration.lineno - 1,
        unavailable_imports=tuple(sorted(missing)),
    )


__all__: list[str] = ["deferred_alias", "guarded_imports"]
