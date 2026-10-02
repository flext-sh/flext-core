"""Source-backed discovery shared by runtime module and alias inspection.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import builtins
import inspect
import sys
import textwrap
from collections.abc import Iterator
from types import ModuleType
from typing import TypeAliasType, runtime_checkable

from flext_core._models.enforcement import FlextModelsEnforcement as me


class FlextUtilitiesBeartypeModuleSource:
    """Prove a lazy alias's source declaration and unavailable guarded imports."""

    @staticmethod
    def parse(module: ModuleType) -> ast.Module:
        """Read the defining source without normalizing inspection failures.

        Returns:
            The resulting ``ast.Module``.

        """
        return ast.parse(inspect.getsource(module), filename=inspect.getfile(module))

    @staticmethod
    def declares_runtime_checkable(target: type) -> bool:
        """Prove from its source that ``target`` is decorated by ``runtime_checkable``.

        Each decorator name or dotted path is resolved in the defining module's
        namespace, so ``runtime_checkable``, ``typing.runtime_checkable`` and any
        import alias of it are recognized by identity.

        Returns:
            The resulting ``bool``.

        Raises:
            TypeError: If Class source does not open with its declaration.

        """
        source = textwrap.dedent(inspect.getsource(target))
        declaration = ast.parse(source, filename=inspect.getfile(target)).body[0]
        if not isinstance(declaration, ast.ClassDef):
            msg = f"Class source does not open with its declaration: {target!r}"
            raise TypeError(msg)
        namespace = vars(sys.modules[target.__module__])
        for expression in declaration.decorator_list:
            path: list[str] = []
            node = expression
            while isinstance(node, ast.Attribute):
                path.insert(0, node.attr)
                node = node.value
            if not isinstance(node, ast.Name) or node.id not in namespace:
                continue
            resolved = namespace[node.id]
            for attribute in path:
                resolved = getattr(resolved, attribute)
            if resolved is runtime_checkable:
                return True
        return False

    @classmethod
    def _declarations(
        cls,
        node: ast.AST,
        scopes: tuple[ast.Module | ast.ClassDef, ...] = (),
        path: tuple[str, ...] = (),
    ) -> Iterator[
        tuple[tuple[str, ...], ast.TypeAlias, tuple[ast.Module | ast.ClassDef, ...]]
    ]:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if isinstance(node, (ast.Module, ast.ClassDef)):
            scopes = (*scopes, node)
        if isinstance(node, ast.ClassDef):
            path = (*path, node.name)
        if isinstance(node, ast.TypeAlias):
            yield (*path, node.name.id), node, scopes
            return
        for child in ast.iter_child_nodes(node):
            yield from cls._declarations(child, scopes, path)

    @classmethod
    def _scope_nodes(cls, node: ast.AST) -> Iterator[ast.AST]:
        for child in ast.iter_child_nodes(node):
            yield child
            if not isinstance(
                child,
                (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda),
            ):
                yield from cls._scope_nodes(child)

    @staticmethod
    def _bound_names(node: ast.AST) -> tuple[str, ...]:
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
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            return (node.name,)
        return ()

    @staticmethod
    def export_names(tree: ast.Module) -> frozenset[str]:
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
    def loads(tree: ast.Module, name: str) -> bool:
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

    @classmethod
    def internal_dependency(cls, tree: ast.Module | None, name: str) -> bool:
        """Prove a module-level alias serves its own module, not consumers.

        An alias consumed by in-module declarations (base classes,
        annotations, calls) and absent from ``__all__`` is a dependency of
        the module, never a backwards-compat export. Unreadable source
        fails closed so the alias stays flagged.

        Returns:
            The resulting ``bool``.

        """
        if tree is None:
            return False
        return name not in cls.export_names(tree) and cls.loads(tree, name)

    @staticmethod
    def _import_roots(node: ast.Import | ast.ImportFrom) -> frozenset[str]:
        """Absolute root packages an import statement binds through.

        Relative imports carry no provable absolute root and fail closed.

        Returns:
            The resulting ``frozenset[str]``.

        """
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module is None:
                return frozenset()
            return frozenset({node.module.split(".")[0]})
        return frozenset(item.name.split(".")[0] for item in node.names)

    @classmethod
    def owner_derived(cls, tree: ast.Module | None, name: str, owner_root: str) -> bool:
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
        bindings: dict[str, ast.stmt] = {}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                bound = tuple(
                    target.id for target in node.targets if isinstance(target, ast.Name)
                )
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                bound = cls._bound_names(node)
            else:
                continue
            for bound_name in bound:
                bindings[bound_name] = node
        seen: set[str] = set()
        current: str | None = name
        while current is not None and current not in seen:
            seen.add(current)
            node = bindings.get(current)
            if node is None:
                return False
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return cls._import_roots(node) == frozenset({owner_root})
            if not isinstance(node, ast.Assign):
                return False
            value = node.value
            if isinstance(value, ast.Name):
                current = value.id
            elif isinstance(value, ast.Attribute):
                root: ast.expr = value
                while isinstance(root, ast.Attribute):
                    root = root.value
                current = root.id if isinstance(root, ast.Name) else None
            else:
                return False
        return False

    @classmethod
    def _guarded_imports(
        cls,
        scopes: tuple[ast.Module | ast.ClassDef, ...],
    ) -> set[str]:
        nodes = tuple(node for scope in scopes for node in cls._scope_nodes(scope))
        flag_imports = {
            item.asname or item.name: node
            for node in nodes
            if isinstance(node, ast.ImportFrom) and node.module == "typing"
            for item in node.names
            if item.name == "TYPE_CHECKING"
        }
        module_imports = {
            item.asname or item.name: node
            for node in nodes
            if isinstance(node, ast.Import)
            for item in node.names
            if item.name == "typing"
        }
        markers = flag_imports | module_imports
        stable = {
            name
            for name, declaration in markers.items()
            if not any(
                name in cls._bound_names(node) and node is not declaration
                for node in nodes
            )
        }
        imports: list[ast.Import | ast.ImportFrom] = []
        for node in nodes:
            if not isinstance(node, ast.If):
                continue
            condition = node.test
            guarded = (
                isinstance(condition, ast.Name)
                and condition.id in flag_imports
                and condition.id in stable
            ) or (
                isinstance(condition, ast.Attribute)
                and condition.attr == "TYPE_CHECKING"
                and isinstance(condition.value, ast.Name)
                and condition.value.id in module_imports
                and condition.value.id in stable
            )
            if guarded:
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

    @classmethod
    def deferred(
        cls,
        alias: TypeAliasType,
        *,
        owner: ModuleType | type,
    ) -> me.DeferredAlias | None:
        """Prove deferral from the explicitly supplied declaring owner.

        Returns:
            The resulting ``me.DeferredAlias | None``.

        Raises:
            ValueError: If Ambiguous type alias declaration; or if Ambiguous type alias
                binding.

        """
        if vars(owner).get(alias.__name__) is not alias:
            return None
        module_name = alias.__module__
        module = sys.modules.get(module_name) if module_name is not None else None
        if module is None:
            return None
        if not isinstance(vars(module).get("__file__"), str):
            return None
        source_lines, source_start = inspect.getsourcelines(owner)
        source = textwrap.dedent("".join(source_lines))
        source_tree = ast.parse(source, filename=inspect.getfile(owner))
        candidates = tuple(
            entry
            for entry in cls._declarations(source_tree)
            if entry[0][-1:] == (alias.__name__,)
            and (
                isinstance(owner, ModuleType)
                or entry[0][-2:] == (owner.__name__, alias.__name__)
            )
        )
        if not candidates:
            return None
        if len(candidates) != 1:
            msg = (
                f"Ambiguous type alias declaration: {alias.__module__}.{alias.__name__}"
            )
            raise ValueError(msg)
        path, declaration, scopes = candidates[0]
        if any(
            alias.__name__ in cls._bound_names(node) and node is not declaration.name
            for node in cls._scope_nodes(scopes[-1])
        ):
            msg = f"Ambiguous type alias binding: {alias.__module__}.{'.'.join(path)}"
            raise ValueError(msg)
        available = set(vars(builtins)) | set(vars(module))
        available.update(vars(owner))
        available.update(
            item.__name__ for item in getattr(owner, "__type_params__", ())
        )
        available.update(item.__name__ for item in alias.__type_params__)
        missing = {
            node.id
            for node in ast.walk(declaration.value)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in available
        }
        guarded = cls._guarded_imports((cls.parse(module),)) | cls._guarded_imports(
            scopes,
        )
        if not missing or not missing <= guarded:
            return None
        return me.DeferredAlias(
            module=module.__name__,
            qualname=f"{owner.__qualname__}.{alias.__name__}"
            if isinstance(owner, type)
            else alias.__name__,
            file_path=inspect.getfile(module),
            line_number=source_start + declaration.lineno - 1,
            unavailable_imports=tuple(sorted(missing)),
        )
