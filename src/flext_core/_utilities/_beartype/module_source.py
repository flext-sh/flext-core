"""Source-backed discovery shared by runtime module and alias inspection."""

from __future__ import annotations

import ast
import builtins
import inspect
import sys
import textwrap
from collections.abc import Iterator
from types import ModuleType
from typing import TypeAliasType

from ..._models.enforcement import FlextModelsEnforcement as me


class FlextUtilitiesBeartypeModuleSource:
    """Prove a lazy alias's source declaration and unavailable guarded imports."""

    @staticmethod
    def parse(module: ModuleType) -> ast.Module:
        """Read the defining source without normalizing inspection failures."""
        return ast.parse(inspect.getsource(module), filename=inspect.getfile(module))

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
                child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
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

    @classmethod
    def _guarded_imports(
        cls, scopes: tuple[ast.Module | ast.ClassDef, ...]
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
        cls, alias: TypeAliasType, *, owner: ModuleType | type
    ) -> me.DeferredAlias | None:
        """Prove deferral from the explicitly supplied declaring owner."""
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
            scopes
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
