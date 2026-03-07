"""LibCST-driven migration of module constants into MRO constants facades."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import override

import libcst as cst

from flext_infra import c, m
from flext_infra._utilities.refactor import FlextInfraUtilitiesRefactor

_CONSTANT_PATTERN = re.compile(r"^_*[A-Z][A-Z0-9_]*$")
_MRO_TARGETS: frozenset[str] = frozenset({"constants", "all"})
_CONSTANTS_FILE_GLOB = "constants.py"
_CONSTANTS_CLASS_SUFFIX = "Constants"
_FINAL_ANNOTATION_NAME = "Final"
_DEFAULT_CONSTANTS_CLASS = "FlextConstants"
_DEFAULT_FACADE_ALIAS = "c"
_RUNTIME_ALIAS_NAMES: frozenset[str] = frozenset({
    "c",
    "m",
    "r",
    "t",
    "u",
    "p",
    "d",
    "e",
    "h",
    "s",
    "x",
})


class FlextInfraRefactorMROMigrationScanner:
    """Discover constants.py files with loose Final declarations."""

    @classmethod
    def scan_workspace(
        cls,
        *,
        workspace_root: Path,
        target: str,
    ) -> tuple[list[m.Infra.Refactor.MROFileScan], int]:
        """Scan workspace and return candidate files with counts."""
        if target not in _MRO_TARGETS:
            return [], 0

        results: list[m.Infra.Refactor.MROFileScan] = []
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
    ) -> m.Infra.Refactor.MROFileScan | None:
        """Scan a constants module for module-level Final constants."""
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return None

        constants_class = cls._first_constants_class_name(tree=tree)
        module = cls._module_path(file_path=file_path, project_root=project_root)
        candidates: list[m.Infra.Refactor.MROConstantCandidate] = []
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
                m.Infra.Refactor.MROConstantCandidate(
                    symbol=stmt.target.id,
                    line=stmt.lineno,
                )
            )

        return m.Infra.Refactor.MROFileScan(
            file=str(file_path),
            module=module,
            constants_class=constants_class,
            candidates=tuple(candidates),
        )

    @staticmethod
    def _project_roots(*, workspace_root: Path) -> list[Path]:
        return FlextInfraUtilitiesRefactor.discover_project_roots(
            workspace_root=workspace_root
        )

    @staticmethod
    def _iter_constants_files(*, project_root: Path) -> list[Path]:
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return []
        return sorted(src_dir.rglob(_CONSTANTS_FILE_GLOB))

    @staticmethod
    def _module_path(*, file_path: Path, project_root: Path) -> str:
        return FlextInfraUtilitiesRefactor.module_path(
            file_path=file_path,
            project_root=project_root,
        )

    @staticmethod
    def _first_constants_class_name(*, tree: ast.Module) -> str:
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and stmt.name.endswith(
                _CONSTANTS_CLASS_SUFFIX
            ):
                return stmt.name
        return ""

    @staticmethod
    def _is_final_annotation(*, annotation: ast.expr) -> bool:
        final_name = _FINAL_ANNOTATION_NAME
        if isinstance(annotation, ast.Name):
            return annotation.id == final_name
        if isinstance(annotation, ast.Attribute):
            return annotation.attr == final_name
        if isinstance(annotation, ast.Subscript):
            base = annotation.value
            if isinstance(base, ast.Name):
                return base.id == final_name
            if isinstance(base, ast.Attribute):
                return base.attr == final_name
        return False


class FlextInfraRefactorMROMigrationTransformer:
    """Move module-level constants into the constants facade class."""

    @staticmethod
    def migrate_file(
        *,
        scan_result: m.Infra.Refactor.MROFileScan,
    ) -> tuple[str, m.Infra.Refactor.MROFileMigration, dict[str, str]]:
        """Transform a candidate file and return code plus symbol map."""
        source = Path(scan_result.file).read_text(encoding=c.Infra.Encoding.DEFAULT)
        module = cst.parse_module(source)

        candidate_symbols = {candidate.symbol for candidate in scan_result.candidates}
        moved_statements: list[tuple[str, cst.AnnAssign]] = []
        retained_module_body: list[cst.CSTNode] = []

        for stmt in module.body:
            moved = FlextInfraRefactorMROMigrationTransformer._extract_moved_statement(
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
                m.Infra.Refactor.MROFileMigration(
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
        class_name = scan_result.constants_class or _DEFAULT_CONSTANTS_CLASS
        class_found = False

        for stmt in retained_module_body:
            if isinstance(stmt, cst.ClassDef) and stmt.name.value == class_name:
                class_found = True
                transformed_class, class_symbol_map = (
                    FlextInfraRefactorMROMigrationTransformer._migrate_constants_class(
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
                FlextInfraRefactorMROMigrationTransformer._create_constants_class(
                    class_name=class_name,
                    moved_by_symbol=moved_by_symbol,
                    ordered_symbols=ordered_symbols,
                )
            )
            transformed_body.append(new_class)
            symbol_map.update(class_symbol_map)
            created_classes = (class_name,)

        updated_module = module.with_changes(body=tuple(transformed_body))

        replacement_values: dict[str, cst.BaseExpression] = {}
        for symbol in ordered_symbols:
            if not symbol.startswith("_"):
                continue
            value = moved_by_symbol[symbol].value
            if value is None:
                continue
            replacement_values[symbol] = value

        replacement_transformer = _ReplacementTransformer(
            replacement_values=replacement_values
        )
        updated_module = updated_module.visit(replacement_transformer)

        migration = m.Infra.Refactor.MROFileMigration(
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

        alias_replacement_values: dict[str, cst.BaseExpression] = {}
        for statement in class_def.body.body:
            alias = FlextInfraRefactorMROMigrationTransformer._extract_alias_assignment(
                statement=statement
            )
            if alias is not None and alias[1] in moved_by_symbol:
                alias_by_symbol[alias[1]] = alias[0]
                private_value = moved_by_symbol[alias[1]].value
                if private_value is not None:
                    alias_replacement_values[alias[0]] = private_value
                continue
            retained_class_body.append(statement)

        symbol_map: dict[str, str] = {}
        added_targets: set[str] = set()
        moved_lines: list[cst.CSTNode] = []
        for symbol in ordered_symbols:
            target = alias_by_symbol.get(
                symbol
            ) or FlextInfraRefactorMROMigrationTransformer._default_target(
                symbol=symbol
            )
            if target in added_targets:
                continue
            added_targets.add(target)
            symbol_map[symbol] = target
            replacement_value = alias_replacement_values.get(target)
            if replacement_value is not None:
                moved_stmt = moved_by_symbol[symbol].with_changes(
                    target=cst.Name(target),
                    value=replacement_value,
                )
            else:
                moved_stmt = moved_by_symbol[symbol].with_changes(
                    target=cst.Name(target),
                )
            moved_lines.append(cst.SimpleStatementLine(body=[moved_stmt]))

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
            target = FlextInfraRefactorMROMigrationTransformer._default_target(
                symbol=symbol
            )
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
        if isinstance(assign, cst.AnnAssign):
            if not isinstance(assign.target, cst.Name):
                return None
            if not isinstance(assign.value, cst.Name):
                return None
            return assign.target.value, assign.value.value
        if isinstance(assign, cst.Assign):
            if len(assign.targets) != 1:
                return None
            if not isinstance(assign.targets[0].target, cst.Name):
                return None
            if not isinstance(assign.value, cst.Name):
                return None
            return assign.targets[0].target.value, assign.value.value
        return None

    @staticmethod
    def _default_target(*, symbol: str) -> str:
        stripped = symbol.lstrip("_")
        return stripped or symbol


class _AstReferenceRewriter(ast.NodeTransformer):
    def __init__(
        self,
        *,
        imported_symbols: dict[str, m.Infra.Refactor.MROImportedSymbol],
        module_aliases: dict[str, str],
        module_facades: dict[str, str],
        moved_index: dict[str, dict[str, str]],
    ) -> None:
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
                value=ast.Name(id=imported.facade_name, ctx=ast.Load()),
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
            return rewritten
        symbol_map = self._moved_index.get(module_name)
        if symbol_map is None:
            return rewritten
        new_symbol = symbol_map.get(rewritten.attr)
        if new_symbol is None:
            return rewritten
        facade_name = self._module_facades.get(module_name, _DEFAULT_FACADE_ALIAS)
        self.replacements += 1
        return ast.copy_location(
            ast.Attribute(
                value=ast.Name(id=facade_name, ctx=ast.Load()),
                attr=new_symbol,
                ctx=rewritten.ctx,
            ),
            rewritten,
        )


class FlextInfraRefactorMROImportRewriter:
    """Rewrite imports and references to use the local facade alias."""

    @classmethod
    def rewrite_workspace(
        cls,
        *,
        workspace_root: Path,
        moved_index: dict[str, dict[str, str]],
        apply_changes: bool,
    ) -> list[m.Infra.Refactor.MRORewriteResult]:
        """Rewrite references across all project Python files."""
        results: list[m.Infra.Refactor.MRORewriteResult] = []
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
    ) -> m.Infra.Refactor.MRORewriteResult | None:
        """Rewrite one file according to moved constant symbol mappings."""
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return None

        imported_symbols: dict[str, m.Infra.Refactor.MROImportedSymbol] = {}
        module_aliases: dict[str, str] = {}
        module_facades: dict[str, m.Infra.Refactor.MROFacadeImport] = (
            FlextInfraRefactorMROImportRewriter._discover_module_facades(
                tree=tree,
                moved_index=moved_index,
            )
        )
        facade_imports_needed: set[str] = set()
        facade_import_objects: dict[str, m.Infra.Refactor.MROFacadeImport] = {}

        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom):
                module_name = stmt.module
                if module_name is None or module_name not in moved_index:
                    continue
                if any(alias.name == "*" for alias in stmt.names):
                    continue
                kept_names: list[ast.alias] = []
                for alias in stmt.names:
                    symbol_map = moved_index[module_name]
                    new_symbol = symbol_map.get(alias.name)
                    if new_symbol is None:
                        kept_names.append(alias)
                        continue
                    module_facade = module_facades.get(module_name)
                    if module_facade is None:
                        module_facade = m.Infra.Refactor.MROFacadeImport(
                            module=module_name,
                            import_name=_DEFAULT_FACADE_ALIAS,
                            as_name=None,
                        )
                        module_facades[module_name] = module_facade
                    imported_symbols[alias.asname or alias.name] = (
                        m.Infra.Refactor.MROImportedSymbol(
                            symbol=new_symbol,
                            facade_name=module_facade.as_name
                            or module_facade.import_name,
                        )
                    )
                    facade_key = f"{module_facade.module}:{module_facade.import_name}:{module_facade.as_name or ''}"
                    facade_imports_needed.add(facade_key)
                    facade_import_objects[facade_key] = module_facade
                stmt.names = kept_names

            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name in moved_index:
                        module_aliases[alias.asname or alias.name] = alias.name

        rewriter = _AstReferenceRewriter(
            imported_symbols=imported_symbols,
            module_aliases=module_aliases,
            module_facades={
                module: facade.as_name or facade.import_name
                for module, facade in module_facades.items()
            },
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

        existing_imports: set[str] = set()
        for stmt in rewritten.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module is not None:
                for alias in stmt.names:
                    if alias.name != "*":
                        key = f"{stmt.module}:{alias.name}:{alias.asname or ''}"
                        existing_imports.add(key)

        imports_to_add = sorted(
            facade_imports_needed - existing_imports,
        )
        if len(imports_to_add) > 0:
            insert_at = FlextInfraRefactorMROImportRewriter._import_insertion_index(
                module=rewritten
            )
            for offset, facade_key in enumerate(imports_to_add):
                facade_import = facade_import_objects[facade_key]
                rewritten.body.insert(
                    insert_at + offset,
                    ast.ImportFrom(
                        module=facade_import.module,
                        names=[
                            ast.alias(
                                name=facade_import.import_name,
                                asname=facade_import.as_name,
                            )
                        ],
                        level=0,
                    ),
                )

        if rewriter.replacements == 0 and len(imports_to_add) == 0:
            return None

        rendered = ast.unparse(ast.fix_missing_locations(rewritten))
        if apply_changes and rendered != source:
            file_path.write_text(f"{rendered}\n", encoding=c.Infra.Encoding.DEFAULT)
        return m.Infra.Refactor.MRORewriteResult(
            file=str(file_path),
            replacements=rewriter.replacements,
        )

    @staticmethod
    def _iter_workspace_python_files(*, workspace_root: Path) -> list[Path]:
        return FlextInfraUtilitiesRefactor.iter_python_files(
            workspace_root=workspace_root,
        )

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

    @staticmethod
    def _discover_module_facades(
        *,
        tree: ast.Module,
        moved_index: dict[str, dict[str, str]],
    ) -> dict[str, m.Infra.Refactor.MROFacadeImport]:
        module_facades: dict[str, m.Infra.Refactor.MROFacadeImport] = {}
        runtime_aliases = _RUNTIME_ALIAS_NAMES
        for stmt in tree.body:
            if not isinstance(stmt, ast.ImportFrom):
                continue
            module_name = stmt.module
            if module_name is None or module_name not in moved_index:
                continue
            if stmt.level != 0:
                continue
            for alias in stmt.names:
                if alias.name not in runtime_aliases:
                    continue
                module_facades[module_name] = m.Infra.Refactor.MROFacadeImport(
                    module=module_name,
                    import_name=alias.name,
                    as_name=alias.asname,
                )
                break
        return module_facades


class _ReplacementTransformer(cst.CSTTransformer):
    """Replace Name nodes that reference private constants with their literal values."""

    def __init__(self, *, replacement_values: dict[str, cst.BaseExpression]) -> None:
        self.replacement_values = replacement_values

    @override
    def leave_Name(
        self, original_node: cst.Name, updated_node: cst.Name
    ) -> cst.BaseExpression:
        if original_node.value in self.replacement_values:
            return self.replacement_values[original_node.value]
        return updated_node


class FlextInfraRefactorMROMigrationValidator:
    """Re-scan workspace to confirm remaining loose constants count."""

    @classmethod
    def validate(cls, *, workspace_root: Path, target: str) -> tuple[int, int]:
        """Return NS-001 style remaining-violations metrics."""
        file_results, _ = FlextInfraRefactorMROMigrationScanner.scan_workspace(
            workspace_root=workspace_root,
            target=target,
        )
        remaining = sum(len(item.candidates) for item in file_results)
        return remaining, 0


__all__ = [
    "FlextInfraRefactorMROImportRewriter",
    "FlextInfraRefactorMROMigrationScanner",
    "FlextInfraRefactorMROMigrationTransformer",
    "FlextInfraRefactorMROMigrationValidator",
]
