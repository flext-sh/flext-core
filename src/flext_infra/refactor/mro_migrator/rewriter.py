"""Workspace reference rewrite phase for migrate-to-mro."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from flext_infra import FlextInfraDiscoveryService, c


@dataclass(frozen=True)
class MRORewriteResult:
    file: str
    replacements: int


@dataclass(frozen=True)
class _ImportedSymbol:
    symbol: str
    facade_alias: str


class _AstReferenceRewriter(ast.NodeTransformer):
    def __init__(
        self,
        *,
        imported_symbols: dict[str, _ImportedSymbol],
        module_aliases: dict[str, str],
        moved_index: dict[str, dict[str, str]],
    ) -> None:
        self._imported_symbols = imported_symbols
        self._module_aliases = module_aliases
        self._moved_index = moved_index
        self.replacements = 0

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Store):
            return node
        imported = self._imported_symbols.get(node.id)
        if imported is None:
            return node
        self.replacements += 1
        return ast.copy_location(
            ast.Attribute(
                value=ast.Name(id=imported.facade_alias, ctx=ast.Load()),
                attr=imported.symbol,
                ctx=node.ctx,
            ),
            node,
        )

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        rewritten = self.generic_visit(node)
        if not isinstance(rewritten, ast.Attribute):
            return rewritten
        if not isinstance(rewritten.value, ast.Name):
            return rewritten
        module_name = self._module_aliases.get(rewritten.value.id)
        if module_name is None:
            return rewritten
        symbol_map = self._moved_index.get(module_name)
        if symbol_map is None:
            return rewritten
        facade_alias = symbol_map.get(rewritten.attr)
        if facade_alias is None:
            return rewritten
        self.replacements += 1
        return ast.copy_location(
            ast.Attribute(
                value=ast.Name(id=facade_alias, ctx=ast.Load()),
                attr=rewritten.attr,
                ctx=rewritten.ctx,
            ),
            rewritten,
        )


class FlextInfraRefactorMRORewriter:
    @classmethod
    def rewrite_workspace(
        cls,
        *,
        workspace_root: Path,
        moved_index: dict[str, dict[str, str]],
        apply_changes: bool,
    ) -> list[MRORewriteResult]:
        results: list[MRORewriteResult] = []
        for file_path in cls._iter_workspace_python_files(
            workspace_root=workspace_root
        ):
            rewritten = cls.rewrite_file(
                file_path=file_path,
                moved_index=moved_index,
                apply_changes=apply_changes,
            )
            if rewritten is not None and rewritten.replacements > 0:
                results.append(rewritten)
        return results

    @staticmethod
    def rewrite_file(
        *,
        file_path: Path,
        moved_index: dict[str, dict[str, str]],
        apply_changes: bool,
    ) -> MRORewriteResult | None:
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return None

        imported_symbols: dict[str, _ImportedSymbol] = {}
        module_aliases: dict[str, str] = {}
        facade_imports_needed: set[tuple[str, str]] = set()

        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module in moved_index:
                if any(alias.name == "*" for alias in stmt.names):
                    continue
                kept_names: list[ast.alias] = []
                for alias in stmt.names:
                    symbol_map = moved_index[stmt.module]
                    if alias.name in symbol_map:
                        imported_symbols[alias.asname or alias.name] = _ImportedSymbol(
                            symbol=alias.name,
                            facade_alias=symbol_map[alias.name],
                        )
                        facade_imports_needed.add((stmt.module, symbol_map[alias.name]))
                        continue
                    kept_names.append(alias)
                stmt.names = kept_names

            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name in moved_index:
                        module_aliases[alias.asname or alias.name] = alias.name

        rewriter = _AstReferenceRewriter(
            imported_symbols=imported_symbols,
            module_aliases=module_aliases,
            moved_index=moved_index,
        )
        rewritten = rewriter.visit(tree)
        if not isinstance(rewritten, ast.Module):
            return None

        rewritten.body = [
            stmt
            for stmt in rewritten.body
            if not (
                isinstance(stmt, ast.ImportFrom)
                and stmt.module in moved_index
                and len(stmt.names) == 0
            )
        ]

        existing_imports = {
            (stmt.module, alias.name)
            for stmt in rewritten.body
            if isinstance(stmt, ast.ImportFrom) and stmt.module is not None
            for alias in stmt.names
            if alias.name != "*"
        }
        imports_to_add = sorted(facade_imports_needed - existing_imports)
        if len(imports_to_add) > 0:
            insert_at = FlextInfraRefactorMRORewriter._import_insertion_index(
                module=rewritten
            )
            for offset, (module_name, alias_name) in enumerate(imports_to_add):
                rewritten.body.insert(
                    insert_at + offset,
                    ast.ImportFrom(
                        module=module_name,
                        names=[ast.alias(name=alias_name)],
                        level=0,
                    ),
                )

        if rewriter.replacements == 0 and len(imports_to_add) == 0:
            return None

        rendered = ast.unparse(ast.fix_missing_locations(rewritten))
        if apply_changes and rendered != source:
            file_path.write_text(f"{rendered}\n", encoding=c.Infra.Encoding.DEFAULT)
        return MRORewriteResult(file=str(file_path), replacements=rewriter.replacements)

    @staticmethod
    def _iter_workspace_python_files(*, workspace_root: Path) -> list[Path]:
        discovery = FlextInfraDiscoveryService()
        projects = discovery.discover_projects(workspace_root)
        roots: list[Path] = []
        if projects.is_success:
            roots = [project.path for project in projects.unwrap()]
        if (
            len(roots) == 0
            and (workspace_root / c.Infra.Paths.DEFAULT_SRC_DIR).is_dir()
        ):
            roots = [workspace_root]

        files: list[Path] = []
        for project_root in roots:
            src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
            tests_dir = project_root / c.Infra.Directories.TESTS
            if src_dir.is_dir():
                files.extend(sorted(src_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)))
            if tests_dir.is_dir():
                files.extend(sorted(tests_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)))
        return files

    @staticmethod
    def _import_insertion_index(*, module: ast.Module) -> int:
        insert_at = 0
        for index, stmt in enumerate(module.body):
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                insert_at = index + 1
                continue
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                insert_at = index + 1
        return insert_at


__all__ = ["FlextInfraRefactorMRORewriter", "MRORewriteResult"]
