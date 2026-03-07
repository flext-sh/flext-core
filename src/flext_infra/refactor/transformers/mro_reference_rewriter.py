"""AST transformer for MRO reference rewrites."""

from __future__ import annotations

import ast
from typing import override

from flext_infra import c, m


class FlextInfraRefactorMROReferenceRewriter(ast.NodeTransformer):
    """Rewrite AST references to moved constants using canonical alias `c`."""

    def __init__(
        self,
        *,
        imported_symbols: dict[str, m.Infra.Refactor.MROImportedSymbol],
        module_aliases: dict[str, str],
        module_facades: dict[str, str],
        moved_index: dict[str, dict[str, str]],
    ) -> None:
        """Initialize with symbol mappings for rewriting."""
        self._imported_symbols = imported_symbols
        self._module_aliases = module_aliases
        self._module_facades = module_facades
        self._moved_index = moved_index
        self.replacements = 0

    @override
    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Store):
            return node
        imported = self._imported_symbols.get(node.id)
        if imported is None:
            return node
        self.replacements += 1
        return ast.copy_location(
            ast.Attribute(
                value=ast.Name(
                    id=c.Infra.Refactor.DEFAULT_FACADE_ALIAS,
                    ctx=ast.Load(),
                ),
                attr=imported.symbol,
                ctx=node.ctx,
            ),
            node,
        )

    @override
    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        rewritten = self.generic_visit(node)
        if not isinstance(rewritten, ast.Attribute):
            return rewritten
        if not isinstance(rewritten.value, ast.Name):
            return rewritten
        module_name = self._module_aliases.get(rewritten.value.id)
        if module_name is None:
            module_name = self._module_facades.get(rewritten.value.id)
        if module_name is None:
            return rewritten
        symbol_map = self._moved_index.get(module_name)
        if symbol_map is None:
            return rewritten
        new_symbol = symbol_map.get(rewritten.attr)
        if new_symbol is None:
            return rewritten
        self.replacements += 1
        return ast.copy_location(
            ast.Attribute(
                value=ast.Name(
                    id=c.Infra.Refactor.DEFAULT_FACADE_ALIAS,
                    ctx=ast.Load(),
                ),
                attr=new_symbol,
                ctx=rewritten.ctx,
            ),
            rewritten,
        )


__all__ = ["FlextInfraRefactorMROReferenceRewriter"]
