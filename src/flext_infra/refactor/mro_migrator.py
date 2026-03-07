"""LibCST-driven migration of module constants into MRO constants facades."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import libcst as cst

from flext_infra import c, m, u
from flext_infra.refactor.transformers.mro_private_inline import (
    FlextInfraRefactorMROPrivateInlineTransformer,
)
from flext_infra.refactor.transformers.mro_reference_rewriter import (
    FlextInfraRefactorMROReferenceRewriter,
)

CONSTANT_PATTERN = re.compile(r"^_?[A-Z][A-Z0-9_]*$")


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
        if target not in c.Infra.Refactor.MRO_TARGETS:
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
            if not CONSTANT_PATTERN.match(stmt.target.id):
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
        return u.Infra.Refactor.discover_project_roots(workspace_root=workspace_root)

    @staticmethod
    def _iter_constants_files(*, project_root: Path) -> list[Path]:
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return []
        return sorted(src_dir.rglob(c.Infra.Refactor.CONSTANTS_FILE_GLOB))

    @staticmethod
    def _module_path(*, file_path: Path, project_root: Path) -> str:
        return u.Infra.Refactor.module_path(
            file_path=file_path,
            project_root=project_root,
        )

    @staticmethod
    def _first_constants_class_name(*, tree: ast.Module) -> str:
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and stmt.name.endswith(
                c.Infra.Refactor.CONSTANTS_CLASS_SUFFIX
            ):
                return stmt.name
        return ""

    @staticmethod
    def _is_final_annotation(*, annotation: ast.expr) -> bool:
        final_name = c.Infra.Refactor.FINAL_ANNOTATION_NAME
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
        class_name = (
            scan_result.constants_class or c.Infra.Refactor.DEFAULT_CONSTANTS_CLASS
        )
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

        inline_transformer = FlextInfraRefactorMROPrivateInlineTransformer(
            replacement_values=replacement_values
        )
        updated_module = updated_module.visit(inline_transformer)

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
        class_template = cst.ClassDef(
            name=cst.Name(class_name),
            body=cst.IndentedBlock(body=()),
        )

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
        facade_aliases: dict[str, str] = {}
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
                    if alias.name == c.Infra.Refactor.DEFAULT_FACADE_ALIAS:
                        facade_local_name = alias.asname or alias.name
                        facade_aliases[facade_local_name] = module_name
                        facade_import = m.Infra.Refactor.MROFacadeImport(
                            module=module_name,
                            import_name=c.Infra.Refactor.DEFAULT_FACADE_ALIAS,
                            as_name=None,
                        )
                        facade_key = (
                            f"{facade_import.module}:"
                            f"{facade_import.import_name}:"
                            f"{facade_import.as_name or ''}"
                        )
                        facade_imports_needed.add(facade_key)
                        facade_import_objects[facade_key] = facade_import
                        continue

                    symbol_map = moved_index[module_name]
                    new_symbol = symbol_map.get(alias.name)
                    if new_symbol is None:
                        kept_names.append(alias)
                        continue
                    imported_symbols[alias.asname or alias.name] = (
                        m.Infra.Refactor.MROImportedSymbol(
                            symbol=new_symbol,
                            facade_name=c.Infra.Refactor.DEFAULT_FACADE_ALIAS,
                        )
                    )
                    facade_import = m.Infra.Refactor.MROFacadeImport(
                        module=module_name,
                        import_name=c.Infra.Refactor.DEFAULT_FACADE_ALIAS,
                        as_name=None,
                    )
                    facade_key = (
                        f"{facade_import.module}:"
                        f"{facade_import.import_name}:"
                        f"{facade_import.as_name or ''}"
                    )
                    facade_imports_needed.add(facade_key)
                    facade_import_objects[facade_key] = facade_import
                stmt.names = kept_names

            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    if alias.name in moved_index:
                        module_aliases[alias.asname or alias.name] = alias.name

        rewriter = FlextInfraRefactorMROReferenceRewriter(
            imported_symbols=imported_symbols,
            module_aliases=module_aliases,
            module_facades=facade_aliases,
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
        return u.Infra.Refactor.iter_python_files(
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
