"""Source-AST utilities shared by module-source provenance and deferral.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import functools
import inspect
from collections.abc import Iterator
from types import ModuleType


class FlextUtilitiesBeartypeModuleSourceScan:
    """Read module source as AST and walk its lexical scopes and declarations."""

    @staticmethod
    @functools.cache
    def parse_module(module: ModuleType) -> ast.Module:
        """Read the defining source of one module as an AST.

        Returns:
            The resulting ``ast.Module``.

        """
        return ast.parse(inspect.getsource(module), filename=inspect.getfile(module))

    @staticmethod
    def _bound_names(node: ast.AST) -> tuple[str, ...]:
        """Return the names one AST statement binds.

        Returns:
            The resulting ``tuple[str, ...]``.

        """
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return tuple(
                item.asname
                or (
                    item.name.split(".")[0]
                    if isinstance(node, ast.Import)
                    else item.name
                )
                for item in node.names
            )
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            return (node.id,)
        if isinstance(node, ast.arg):
            return (node.arg,)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return (node.name,)
        return ()

    @staticmethod
    @functools.lru_cache(maxsize=8192)
    def _scope_node_tuple(node: ast.AST) -> tuple[ast.AST, ...]:
        """Cache the descendants reachable without entering a nested scope.

        Returns:
            The descendant nodes in traversal order.

        """
        result: list[ast.AST] = []
        for child in ast.iter_child_nodes(node):
            result.append(child)
            if not isinstance(
                child,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
            ):
                result.extend(
                    FlextUtilitiesBeartypeModuleSourceScan._scope_node_tuple(child)
                )
        return tuple(result)

    @staticmethod
    def _declarations(
        node: ast.AST,
        scopes: tuple[
            ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, ...
        ] = (),
        path: tuple[str, ...] = (),
    ) -> Iterator[
        tuple[
            tuple[str, ...],
            ast.TypeAlias,
            tuple[
                ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, ...
            ],
        ]
    ]:
        """Yield every type-alias declaration with its lexical scopes and qualified path.

        Yields:
            Each ``(path, declaration, scopes)`` triple.

        """
        if isinstance(node, ast.Lambda):
            return
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            scopes = (*scopes, node)
        if isinstance(node, ast.ClassDef):
            path = (*path, node.name)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            path = (*path, node.name, "<locals>")
        if isinstance(node, ast.TypeAlias):
            yield (*path, node.name.id), node, scopes
            return
        for child in ast.iter_child_nodes(node):
            yield from FlextUtilitiesBeartypeModuleSourceScan._declarations(
                child, scopes, path
            )

    @staticmethod
    @functools.cache
    def _cached_declarations(
        node: ast.AST,
    ) -> tuple[
        tuple[
            tuple[str, ...],
            ast.TypeAlias,
            tuple[
                ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, ...
            ],
        ],
        ...,
    ]:
        """Cache the full declaration traversal for one AST root.

        Returns:
            The resulting cached declaration tuple.

        """
        return tuple(FlextUtilitiesBeartypeModuleSourceScan._declarations(node))

    @staticmethod
    def _export_names(tree: ast.Module) -> frozenset[str]:
        """Collect names a module publishes through ``__all__`` assignments.

        Returns:
            The resulting ``frozenset[str]``.

        """
        exported: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                targets: tuple[ast.expr, ...] = tuple(node.targets)
            elif isinstance(node, ast.AugAssign):
                targets = (node.target,)
            else:
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in targets
            ):
                continue
            if not isinstance(value := node.value, (ast.List, ast.Tuple)):
                continue
            exported.update(
                item.value
                for item in value.elts
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
        return frozenset(exported)

    @staticmethod
    def _loads(tree: ast.Module, name: str) -> bool:
        """Prove ``name`` is consumed anywhere in the tree (Load context).

        Returns:
            The resulting ``bool``.

        """
        return any(
            isinstance(node, ast.Name)
            and node.id == name
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(tree)
        )


__all__: list[str] = ["FlextUtilitiesBeartypeModuleSourceScan"]
