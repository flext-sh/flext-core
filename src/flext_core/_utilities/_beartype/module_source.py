"""Source-backed discovery shared by runtime module and alias inspection.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import sys
from typing import runtime_checkable

from flext_core._utilities import FlextUtilitiesBeartypeModuleSourceDeferred


class FlextUtilitiesBeartypeModuleSource(FlextUtilitiesBeartypeModuleSourceDeferred):
    """Prove a lazy alias's source declaration and unavailable guarded imports."""

    @classmethod
    def declares_runtime_checkable(cls, target: type) -> bool:
        """Prove from its source that ``target`` is decorated by ``runtime_checkable``.

        Each decorator name or dotted path is resolved in the defining module's
        namespace, so ``runtime_checkable``, ``typing.runtime_checkable`` and any
        import alias of it are recognized by identity.

        Returns:
            The resulting ``bool``.

        Raises:
            TypeError: If Class source does not open with its declaration.

        """
        declaration: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        declaration = cls.parse_module(sys.modules[target.__module__])
        for name in target.__qualname__.split("."):
            if name == "<locals>":
                continue
            matches = [
                node
                for node in declaration.body
                if isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
                and node.name == name
            ]
            if len(matches) != 1:
                msg = f"Class source does not uniquely declare {target!r}"
                raise TypeError(msg)
            declaration = matches[0]
        if not isinstance(declaration, ast.ClassDef):
            msg = f"Class source does not open with its declaration: {target!r}"
            raise TypeError(msg)
        namespace = vars(sys.modules[target.__module__])
        for expression in declaration.decorator_list:
            path: list[str] = []
            node: ast.expr = expression
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
    def internal_dependency(
        cls,
        tree: ast.Module | None,
        name: str,
    ) -> bool:
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
        return name not in cls._export_names(tree) and cls._loads(tree, name)


__all__: list[str] = ["FlextUtilitiesBeartypeModuleSource"]
