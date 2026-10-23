"""Deferred type-alias proof sidecar for module-source inspection.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import builtins
import functools
import inspect
import sys
from types import ModuleType
from typing import TypeAliasType

from flext_core import t
from flext_core._models import FlextModelsEnforcement
from flext_core._utilities._beartype import FlextUtilitiesBeartypeModuleSourceProvenance


class FlextUtilitiesBeartypeModuleSourceDeferred(
    FlextUtilitiesBeartypeModuleSourceProvenance
):
    """Prove a type alias is deferred behind provably-stable ``TYPE_CHECKING`` imports."""

    @staticmethod
    @functools.lru_cache(maxsize=65536)
    def _scope_bindings(scope: ast.AST) -> t.MappingKV[str, tuple[ast.AST, ...]]:
        """Index each binding once in its lexical scope, retaining duplicate owners.

        Returns:
            The nodes that bind each name without entering a nested scope.

        """
        source = FlextUtilitiesBeartypeModuleSourceDeferred
        bindings: t.MutableMappingKV[str, list[ast.AST]] = {}
        for node in source._scope_node_tuple(scope):
            for name in source._bound_names(node):
                bindings.setdefault(name, []).append(node)
        return {name: tuple(nodes) for name, nodes in bindings.items()}

    @staticmethod
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

    @staticmethod
    def _module_guarded(
        condition: ast.expr,
        module_imports: t.MappingKV[str, ast.Import],
        stable: frozenset[str],
    ) -> bool:
        """Prove one ``if`` guards a ``typing`` module ``TYPE_CHECKING`` block.

        Returns:
            The resulting ``bool``.

        """
        if (
            not isinstance(condition, ast.Attribute)
            or condition.attr != "TYPE_CHECKING"
        ):
            return False
        if not isinstance(condition.value, ast.Name):
            return False
        return condition.value.id in module_imports and condition.value.id in stable

    @classmethod
    def _guarded_imports(
        cls,
        scopes: tuple[
            ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, ...
        ],
    ) -> set[str]:
        """Collect names imported behind provably-stable ``TYPE_CHECKING`` guards.

        Returns:
            The resulting ``set[str]``.

        """
        nodes = tuple(node for scope in scopes for node in cls._scope_node_tuple(scope))
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
                name in cls._bound_names(node) and node is not declaration
                for node in nodes
            )
        )
        imports: list[ast.Import | ast.ImportFrom] = []
        for node in nodes:
            if not isinstance(node, ast.If):
                continue
            condition = node.test
            if cls._flag_guarded(
                condition, flag_imports, stable
            ) or cls._module_guarded(
                condition,
                module_imports,
                stable,
            ):
                imports.extend(
                    item
                    for item in node.body
                    if isinstance(item, (ast.Import, ast.ImportFrom))
                )
        names = {name for node in imports for name in cls._bound_names(node)}
        return {
            name
            for name in names
            if not any(
                name in cls._bound_names(node) and node not in imports for node in nodes
            )
        }

    @staticmethod
    def _deferred_module(
        alias: TypeAliasType,
        path: tuple[str, ...] | None = None,
        *,
        owner: ModuleType | type | None = None,
    ) -> tuple[ModuleType, tuple[type, ...]] | None:
        """Authenticate the native module and an optional lexical declaration path.

        Returns:
            The defining module and its proven class owners, or None.

        """
        module_name = alias.__module__
        module = sys.modules.get(module_name) if module_name is not None else None
        if module is None or not isinstance(vars(module).get("__file__"), str):
            return None
        if path is not None and "<locals>" in path:
            if (
                not isinstance(owner, type)
                or owner.__module__ != module.__name__
                or owner.__qualname__ != ".".join(path[:-1])
                or vars(owner).get(path[-1]) is not alias
            ):
                return None
            return module, (owner,)
        owners: list[type] = []
        scope: ModuleType | type = module
        if path is not None:
            for position, name in enumerate(path[:-1], start=1):
                value = vars(scope).get(name)
                if (
                    not isinstance(value, type)
                    or value.__module__ != module.__name__
                    or value.__qualname__ != ".".join(path[:position])
                ):
                    return None
                owners.append(value)
                scope = value
            if vars(scope).get(path[-1]) is not alias:
                return None
        return module, tuple(owners)

    @classmethod
    def _declaration_entry(
        cls,
        alias: TypeAliasType,
        module: ModuleType,
        source_tree: ast.Module,
        owner: ModuleType | type,
    ) -> (
        tuple[
            ast.TypeAlias,
            tuple[
                ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, ...
            ],
            tuple[ModuleType | type, ...],
        ]
        | None
    ):
        """Resolve the unique declaring entry of ``alias`` in the owner source.

        Returns:
            The resulting ``(declaration, scopes) | None``.

        Raises:
            ValueError: If Ambiguous type alias declaration; or if Ambiguous type alias
                binding.

        """
        candidates = tuple(
            (entry, context)
            for entry in cls._cached_declarations(source_tree)
            if entry[0][-1:] == (alias.__name__,)
            and (context := cls._deferred_module(alias, entry[0], owner=owner))
            is not None
            and context[0] is module
            and (
                "<locals>" not in entry[0]
                or (
                    isinstance(entry[2][-1], ast.ClassDef)
                    and vars(context[1][-1]).get("__firstlineno__")
                    == min((
                        entry[2][-1].lineno,
                        *(
                            decorator.lineno
                            for decorator in entry[2][-1].decorator_list
                        ),
                    ))
                )
            )
        )
        if not candidates:
            return None
        if len(candidates) != 1:
            msg = (
                f"Ambiguous type alias declaration: {alias.__module__}.{alias.__name__}"
            )
            raise ValueError(msg)
        (path, declaration, scopes), (declaring_module, class_owners) = candidates[0]
        owners: tuple[ModuleType | type, ...] = (declaring_module, *class_owners)
        if any(
            node is not binding
            for binding, scope in zip(
                (*scopes[1:], declaration.name), scopes, strict=True
            )
            for name in cls._bound_names(binding)
            for node in cls._scope_bindings(scope).get(name, ())
        ):
            msg = f"Ambiguous type alias binding: {alias.__module__}.{'.'.join(path)}"
            raise ValueError(msg)
        return declaration, scopes, owners

    @staticmethod
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

    @classmethod
    def deferred_alias(
        cls,
        alias: TypeAliasType,
        *,
        owner: ModuleType | type,
    ) -> FlextModelsEnforcement.DeferredAlias | None:
        """Prove deferral from the alias's authenticated lexical declaration.

        The requesting owner may consume an alias declared in another module;
        a function-local class alias requires its exact declaring owner.
        Ambiguous declaration or binding failures propagate from the declaration
        resolver.

        Returns:
            The resulting ``me.DeferredAlias | None``.

        """
        context = cls._deferred_module(alias)
        if context is None:
            return None
        module = context[0]
        source_tree = cls.parse_module(module)
        entry = cls._declaration_entry(alias, module, source_tree, owner)
        if entry is None:
            return None
        declaration, scopes, owners = entry
        owner = owners[-1]
        available = set(vars(builtins)) | set(vars(module))
        for lexical_owner in owners[1:]:
            available.update(vars(lexical_owner))
            available.update(
                item.__name__ for item in vars(lexical_owner).get("__type_params__", ())
            )
        available.update(item.__name__ for item in alias.__type_params__)
        missing = cls._unavailable_names(declaration, available)
        if not missing:
            return None
        guarded = cls._guarded_imports(scopes)
        if not missing <= guarded:
            return None
        return FlextModelsEnforcement.DeferredAlias(
            module=module.__name__,
            qualname=f"{owner.__qualname__}.{alias.__name__}"
            if isinstance(owner, type)
            else alias.__name__,
            file_path=inspect.getfile(module),
            line_number=declaration.lineno,
            unavailable_imports=tuple(sorted(missing)),
        )


__all__: list[str] = ["FlextUtilitiesBeartypeModuleSourceDeferred"]
