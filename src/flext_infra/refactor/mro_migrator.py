"""LibCST-driven migration of module constants into MRO constants facades."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import libcst as cst

from flext_infra import FlextInfraDiscoveryService, c

_CONSTANT_PATTERN = re.compile(r"^_*[A-Z][A-Z0-9_]*$")


@dataclass(frozen=True)
class MROConstantCandidate:
    symbol: str
    line: int


@dataclass(frozen=True)
class MROFileScan:
    """Scan result for one constants module candidate file."""

    file: str
    module: str
    constants_class: str
    candidates: tuple[MROConstantCandidate, ...]


@dataclass(frozen=True)
class MROFileMigration:
    """Transformation outcome for a single constants module."""

    file: str
    module: str
    moved_symbols: tuple[str, ...]
    created_classes: tuple[str, ...]


@dataclass(frozen=True)
class MRORewriteResult:
    """Reference rewrite summary for one Python file."""

    file: str
    replacements: int


@dataclass(frozen=True)
class _ImportedSymbol:
    symbol: str


class MROMigrationScanner:
    """Discover constants.py files with loose Final declarations."""

    @classmethod
    def scan_workspace(
        cls,
        *,
        workspace_root: Path,
        target: str,
    ) -> tuple[list[MROFileScan], int]:
        """Scan workspace and return candidate files with counts."""
        if target not in {"constants", "all"}:
            return [], 0

        results: list[MROFileScan] = []
        scanned = 0
        for project_root in cls._project_roots(workspace_root=workspace_root):
            for file_path in cls._iter_constants_files(project_root=project_root):
                scanned += 1
                result = cls.scan_file(file_path=file_path, project_root=project_root)
                if result is None or len(result.candidates) == 0:
                    continue
                results.append(result)
        return results, scanned

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        project_root: Path,
    ) -> MROFileScan | None:
        """Scan a constants module for module-level Final constants."""
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return None

        constants_class = cls._first_constants_class_name(tree=tree)
        module = cls._module_path(file_path=file_path, project_root=project_root)
        candidates: list[MROConstantCandidate] = []
        for stmt in tree.body:
            if not isinstance(stmt, ast.AnnAssign):
                continue
            if not isinstance(stmt.target, ast.Name):
                continue
            if not _CONSTANT_PATTERN.match(stmt.target.id):
                continue
            if not cls._is_final_annotation(annotation=stmt.annotation):
                continue
            candidates.append(
                MROConstantCandidate(
                    symbol=stmt.target.id,
                    line=stmt.lineno,
                )
            )

        return MROFileScan(
            file=str(file_path),
            module=module,
            constants_class=constants_class,
            candidates=tuple(candidates),
        )

    @staticmethod
    def _project_roots(*, workspace_root: Path) -> list[Path]:
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
        return roots

    @staticmethod
    def _iter_constants_files(*, project_root: Path) -> list[Path]:
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return []
        return sorted(src_dir.rglob("constants.py"))

    @staticmethod
    def _module_path(*, file_path: Path, project_root: Path) -> str:
        rel = file_path.relative_to(project_root)
        parts = [part for part in rel.with_suffix("").parts if part != "src"]
        return ".".join(parts)

    @staticmethod
    def _first_constants_class_name(*, tree: ast.Module) -> str:
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and stmt.name.endswith("Constants"):
                return stmt.name
        return ""

    @staticmethod
    def _is_final_annotation(*, annotation: ast.expr) -> bool:
        if isinstance(annotation, ast.Name):
            return annotation.id == "Final"
        if isinstance(annotation, ast.Attribute):
            return annotation.attr == "Final"
        if isinstance(annotation, ast.Subscript):
            base = annotation.value
            if isinstance(base, ast.Name):
                return base.id == "Final"
            if isinstance(base, ast.Attribute):
                return base.attr == "Final"
        return False


class MROMigrationTransformer:
    """Move module-level constants into the constants facade class."""

    @staticmethod
    def migrate_file(
        *,
        scan_result: MROFileScan,
    ) -> tuple[str, MROFileMigration, dict[str, str]]:
        """Transform a candidate file and return code plus symbol map."""
        source = Path(scan_result.file).read_text(encoding=c.Infra.Encoding.DEFAULT)
        module = cst.parse_module(source)

        candidate_symbols = {candidate.symbol for candidate in scan_result.candidates}
        moved_statements: list[tuple[str, cst.AnnAssign]] = []
        retained_module_body: list[cst.CSTNode] = []

        for stmt in module.body:
            moved = MROMigrationTransformer._extract_moved_statement(
                statement=stmt,
                candidate_symbols=candidate_symbols,
            )
            if moved is None:
                retained_module_body.append(stmt)
                continue
            moved_statements.append(moved)

        if len(moved_statements) == 0:
            return (
                source,
                MROFileMigration(
                    file=scan_result.file,
                    module=scan_result.module,
                    moved_symbols=(),
                    created_classes=(),
                ),
                {},
            )

        moved_by_symbol = dict(moved_statements)
        ordered_symbols = [symbol for symbol, _ in moved_statements]

        transformed_body: list[cst.CSTNode] = []
        symbol_map: dict[str, str] = {}
        class_name = scan_result.constants_class or "FlextConstants"
        class_found = False

        for stmt in retained_module_body:
            if isinstance(stmt, cst.ClassDef) and stmt.name.value == class_name:
                class_found = True
                transformed_class, class_symbol_map = (
                    MROMigrationTransformer._migrate_constants_class(
                        class_def=stmt,
                        moved_by_symbol=moved_by_symbol,
                        ordered_symbols=ordered_symbols,
                    )
                )
                transformed_body.append(transformed_class)
                symbol_map.update(class_symbol_map)
                continue
            transformed_body.append(stmt)

        created_classes: tuple[str, ...] = ()
        if not class_found:
            new_class, class_symbol_map = (
                MROMigrationTransformer._create_constants_class(
                    class_name=class_name,
                    moved_by_symbol=moved_by_symbol,
                    ordered_symbols=ordered_symbols,
                )
            )
            transformed_body.append(new_class)
            symbol_map.update(class_symbol_map)
            created_classes = (class_name,)

        updated_module = module.with_changes(body=tuple(transformed_body))
        migration = MROFileMigration(
            file=scan_result.file,
            module=scan_result.module,
            moved_symbols=tuple(ordered_symbols),
            created_classes=created_classes,
        )
        return updated_module.code, migration, symbol_map

    @staticmethod
    def _extract_moved_statement(
        *,
        statement: cst.CSTNode,
        candidate_symbols: set[str],
    ) -> tuple[str, cst.AnnAssign] | None:
        if not isinstance(statement, cst.SimpleStatementLine):
            return None
        if len(statement.body) != 1:
            return None
        first_stmt = statement.body[0]
        if not isinstance(first_stmt, cst.AnnAssign):
            return None
        if not isinstance(first_stmt.target, cst.Name):
            return None
        symbol = first_stmt.target.value
        if symbol not in candidate_symbols:
            return None
        return symbol, first_stmt

    @staticmethod
    def _migrate_constants_class(
        *,
        class_def: cst.ClassDef,
        moved_by_symbol: dict[str, cst.AnnAssign],
        ordered_symbols: list[str],
    ) -> tuple[cst.ClassDef, dict[str, str]]:
        retained_class_body: list[cst.CSTNode] = []
        alias_by_symbol: dict[str, str] = {}

        for statement in class_def.body.body:
            alias = MROMigrationTransformer._extract_alias_assignment(
                statement=statement
            )
            if alias is not None and alias[1] in moved_by_symbol:
                alias_by_symbol[alias[1]] = alias[0]
                continue
            retained_class_body.append(statement)

        symbol_map: dict[str, str] = {}
        added_targets: set[str] = set()
        moved_lines: list[cst.CSTNode] = []
        for symbol in ordered_symbols:
            target = alias_by_symbol.get(
                symbol
            ) or MROMigrationTransformer._default_target(symbol=symbol)
            if target in added_targets:
                continue
            added_targets.add(target)
            symbol_map[symbol] = target
            moved_lines.append(
                cst.SimpleStatementLine(
                    body=[
                        moved_by_symbol[symbol].with_changes(
                            target=cst.Name(target),
                        )
                    ]
                )
            )

        cleaned_body = [
            statement
            for statement in retained_class_body
            if not (
                len(moved_lines) > 0
                and isinstance(statement, cst.SimpleStatementLine)
                and len(statement.body) == 1
                and isinstance(statement.body[0], cst.Pass)
            )
        ]

        final_nodes: list[cst.CSTNode] = [*cleaned_body, *moved_lines]
        final_body = [
            statement
            for statement in final_nodes
            if isinstance(statement, cst.BaseStatement)
        ]
        if len(final_body) == 0:
            final_body = [cst.SimpleStatementLine(body=[cst.Pass()])]

        return (
            class_def.with_changes(
                body=class_def.body.with_changes(body=tuple(final_body))
            ),
            symbol_map,
        )

    @staticmethod
    def _create_constants_class(
        *,
        class_name: str,
        moved_by_symbol: dict[str, cst.AnnAssign],
        ordered_symbols: list[str],
    ) -> tuple[cst.ClassDef, dict[str, str]]:
        class_template = cst.parse_statement(f"class {class_name}:\n    pass\n")
        if not isinstance(class_template, cst.ClassDef):
            msg = f"unable to create class {class_name}"
            raise TypeError(msg)

        class_body: list[cst.BaseStatement] = []
        symbol_map: dict[str, str] = {}
        for symbol in ordered_symbols:
            target = MROMigrationTransformer._default_target(symbol=symbol)
            symbol_map[symbol] = target
            class_body.append(
                cst.SimpleStatementLine(
                    body=[moved_by_symbol[symbol].with_changes(target=cst.Name(target))]
                )
            )

        return (
            class_template.with_changes(
                body=class_template.body.with_changes(body=tuple(class_body))
            ),
            symbol_map,
        )

    @staticmethod
    def _extract_alias_assignment(*, statement: cst.CSTNode) -> tuple[str, str] | None:
        if not isinstance(statement, cst.SimpleStatementLine):
            return None
        if len(statement.body) != 1:
            return None
        assign = statement.body[0]
        if not isinstance(assign, cst.Assign):
            return None
        if len(assign.targets) != 1:
            return None
        if not isinstance(assign.targets[0].target, cst.Name):
            return None
        if not isinstance(assign.value, cst.Name):
            return None
        return assign.targets[0].target.value, assign.value.value

    @staticmethod
    def _default_target(*, symbol: str) -> str:
        stripped = symbol.lstrip("_")
        return stripped or symbol


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
                value=ast.Name(id="c", ctx=ast.Load()),
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
        new_symbol = symbol_map.get(rewritten.attr)
        if new_symbol is None:
            return rewritten
        self.replacements += 1
        return ast.copy_location(
            ast.Attribute(
                value=ast.Name(id="c", ctx=ast.Load()),
                attr=new_symbol,
                ctx=rewritten.ctx,
            ),
            rewritten,
        )


class MROImportRewriter:
    """Rewrite imports and references to use constants facade alias `c`."""

    @classmethod
    def rewrite_workspace(
        cls,
        *,
        workspace_root: Path,
        moved_index: dict[str, dict[str, str]],
        apply_changes: bool,
    ) -> list[MRORewriteResult]:
        """Rewrite references across all project Python files."""
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
        """Rewrite one file according to moved constant symbol mappings."""
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
                    new_symbol = symbol_map.get(alias.name)
                    if new_symbol is None:
                        kept_names.append(alias)
                        continue
                    imported_symbols[alias.asname or alias.name] = _ImportedSymbol(
                        symbol=new_symbol,
                    )
                    facade_imports_needed.add((stmt.module, "c"))
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
            insert_at = MROImportRewriter._import_insertion_index(module=rewritten)
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


class MROMigrationValidator:
    """Re-scan workspace to confirm remaining loose constants count."""

    @classmethod
    def validate(cls, *, workspace_root: Path, target: str) -> tuple[int, int]:
        """Return NS-001 style remaining-violations metrics."""
        file_results, _ = MROMigrationScanner.scan_workspace(
            workspace_root=workspace_root,
            target=target,
        )
        remaining = sum(len(item.candidates) for item in file_results)
        return remaining, 0


__all__ = [
    "MROFileMigration",
    "MROFileScan",
    "MROImportRewriter",
    "MROMigrationScanner",
    "MROMigrationTransformer",
    "MROMigrationValidator",
    "MRORewriteResult",
]
