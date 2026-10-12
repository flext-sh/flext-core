"""Owner-provenance analysis sidecar for module-source bindings.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast

from flext_core._utilities._beartype import bound_names


def import_roots(node: ast.Import | ast.ImportFrom) -> frozenset[str]:
    """Return the absolute root packages an import statement binds through.

    Relative imports carry no provable absolute root and fail closed.

    Returns:
        The resulting ``frozenset[str]``.

    """
    if isinstance(node, ast.ImportFrom):
        if node.level or node.module is None:
            return frozenset()
        return frozenset({node.module.split(".")[0]})
    return frozenset(item.name.split(".")[0] for item in node.names)


def owner_derived(tree: ast.Module | None, name: str, owner_root: str) -> bool:
    """Prove a module-level binding chain is rooted in owner-project imports.

    Follows ``name =`` assignments and attribute chains transitively to
    the import that rooted the value; only imports from the owner
    project's own root prove facade provenance. Calls (dynamic
    imports), relative imports, and unreadable source fail closed, so
    direct and dynamic acquisition of an owned library keeps violating.

    Returns:
        The resulting ``bool``.

    """
    if tree is None:
        return False
    bindings = module_bindings(tree)
    seen: set[str] = set()
    current: str | None = name
    while current is not None and current not in seen:
        seen.add(current)
        node = bindings.get(current)
        if node is None:
            return False
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return import_roots(node) == frozenset({owner_root})
        if not isinstance(node, ast.Assign):
            return False
        current = assign_root_name(node.value)
    return False


def module_bindings(tree: ast.Module) -> dict[str, ast.stmt]:
    """Index module-level import and assignment bindings by bound name.

    Returns:
        The resulting ``dict[str, ast.stmt]``.

    """
    bindings: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            bound = tuple(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound = bound_names(node)
        else:
            continue
        for bound_name in bound:
            bindings[bound_name] = node
    return bindings


def assign_root_name(value: ast.expr) -> str | None:
    """Resolve the name rooted by one assignment value, if provable.

    Returns:
        The resulting ``str | None``.

    """
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        root: ast.expr = value
        while isinstance(root, ast.Attribute):
            root = root.value
        return root.id if isinstance(root, ast.Name) else None
    return None


__all__: list[str] = ["owner_derived"]
