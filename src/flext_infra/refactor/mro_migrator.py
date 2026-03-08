"""LibCST-driven migration of module constants into MRO constants facades."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
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
TYPE_CANDIDATE_PATTERN = re.compile(r"^_?[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class _MROTargetSpec:
    family_alias: str
    file_names: frozenset[str]
    package_directory: str
    class_suffix: str


def _new_symbol_candidate(
    *, symbol: str, line: int, kind: str = "constant"
) -> m.Infra.Refactor.MROSymbolCandidate:
    return m.Infra.Refactor.MROSymbolCandidate(
        symbol=symbol,
        line=line,
        kind=kind,
        class_name="",
        facade_name="",
    )


def _new_scan_report(
    *,
    file: str,
    module: str,
    constants_class: str,
    facade_alias: str,
    candidates: tuple[m.Infra.Refactor.MROSymbolCandidate, ...],
) -> m.Infra.Refactor.MROScanReport:
    return m.Infra.Refactor.MROScanReport(
        file=file,
        module=module,
        constants_class=constants_class,
        facade_alias=facade_alias,
        candidates=candidates,
    )


def _new_file_migration(
    *,
    file: str,
    module: str,
    moved_symbols: tuple[str, ...],
    created_classes: tuple[str, ...],
) -> m.Infra.Refactor.MROFileMigration:
    return m.Infra.Refactor.MROFileMigration(
        file=file,
        module=module,
        moved_symbols=moved_symbols,
        created_classes=created_classes,
    )


def _new_import_rewrite(
    *, module: str, import_name: str, as_name: str | None, symbol: str, facade_name: str
) -> m.Infra.Refactor.MROImportRewrite:
    return m.Infra.Refactor.MROImportRewrite(
        module=module,
        import_name=import_name,
        as_name=as_name,
        symbol=symbol,
        facade_name=facade_name,
    )


def _new_rewrite_result(
    *, file: str, replacements: int
) -> m.Infra.Refactor.MRORewriteResult:
    return m.Infra.Refactor.MRORewriteResult(file=file, replacements=replacements)


class FlextInfraRefactorMROMigrationScanner:
    """Discover constants.py files with loose Final declarations."""

    @classmethod
    def scan_workspace(
        cls, *, workspace_root: Path, target: str
    ) -> tuple[list[m.Infra.Refactor.MROScanReport], int]:
        """Scan workspace and return candidate files with counts."""
        if target not in c.Infra.Refactor.MRO_TARGETS:
            return ([], 0)
        results: list[m.Infra.Refactor.MROScanReport] = []
        scanned = 0
        target_specs = cls._target_specs(target=target)
        for project_root in cls._project_roots(workspace_root=workspace_root):
            for target_spec in target_specs:
                for file_path in cls._iter_target_files(
                    project_root=project_root,
                    target_spec=target_spec,
                ):
                    scanned += 1
                    result = cls.scan_file(
                        file_path=file_path,
                        project_root=project_root,
                        target_spec=target_spec,
                    )
                    if result is None or len(result.candidates) == 0:
                        continue
                    results.append(result)
        return (results, scanned)

    @classmethod
    def scan_file(
        cls,
        *,
        file_path: Path,
        project_root: Path,
        target_spec: _MROTargetSpec,
    ) -> m.Infra.Refactor.MROScanReport | None:
        """Scan a constants module for module-level Final constants."""
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return None
        constants_class = cls._facade_class_name(tree=tree, target_spec=target_spec)
        if not constants_class:
            return None
        module = cls._module_path(file_path=file_path, project_root=project_root)
        candidates: list[m.Infra.Refactor.MROSymbolCandidate] = []
        for stmt in tree.body:
            candidate = cls._candidate_from_statement(
                stmt=stmt, target_spec=target_spec
            )
            if candidate is not None:
                candidates.append(candidate)
        return _new_scan_report(
            file=str(file_path),
            module=module,
            constants_class=constants_class,
            facade_alias=target_spec.family_alias,
            candidates=tuple(candidates),
        )

    @staticmethod
    def _candidate_from_statement(
        *, stmt: ast.stmt, target_spec: _MROTargetSpec
    ) -> m.Infra.Refactor.MROSymbolCandidate | None:
        if target_spec.family_alias == "t":
            return (
                FlextInfraRefactorMROMigrationScanner._typing_candidate_from_statement(
                    stmt=stmt
                )
            )
        if target_spec.family_alias == "p":
            return FlextInfraRefactorMROMigrationScanner._protocol_candidate_from_statement(
                stmt=stmt
            )
        if isinstance(stmt, ast.AnnAssign):
            if not isinstance(stmt.target, ast.Name):
                return None
            if not CONSTANT_PATTERN.match(stmt.target.id):
                return None
            if not FlextInfraRefactorMROMigrationScanner._is_final_annotation(
                annotation=stmt.annotation
            ):
                return None
            return _new_symbol_candidate(symbol=stmt.target.id, line=stmt.lineno)
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                return None
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                return None
            if not CONSTANT_PATTERN.match(target.id):
                return None
            return _new_symbol_candidate(symbol=target.id, line=stmt.lineno)
        return None

    @staticmethod
    def _project_roots(*, workspace_root: Path) -> list[Path]:
        return u.Infra.Refactor.discover_project_roots(workspace_root=workspace_root)

    @staticmethod
    def _target_specs(*, target: str) -> tuple[_MROTargetSpec, ...]:
        ref_c: type[c.Infra.Refactor] = c.Infra.Refactor
        constants_spec = _MROTargetSpec(
            family_alias="c",
            file_names=ref_c.MRO_CONSTANTS_FILE_NAMES,
            package_directory=ref_c.MRO_CONSTANTS_DIRECTORY,
            class_suffix=ref_c.CONSTANTS_CLASS_SUFFIX,
        )
        typings_spec = _MROTargetSpec(
            family_alias="t",
            file_names=ref_c.MRO_TYPINGS_FILE_NAMES,
            package_directory=ref_c.MRO_TYPINGS_DIRECTORY,
            class_suffix="Types",
        )
        protocols_spec = _MROTargetSpec(
            family_alias="p",
            file_names=ref_c.MRO_PROTOCOLS_FILE_NAMES,
            package_directory=ref_c.MRO_PROTOCOLS_DIRECTORY,
            class_suffix="Protocols",
        )
        models_spec = _MROTargetSpec(
            family_alias="m",
            file_names=ref_c.MRO_MODELS_FILE_NAMES,
            package_directory=ref_c.MRO_MODELS_DIRECTORY,
            class_suffix="Models",
        )
        utilities_spec = _MROTargetSpec(
            family_alias="u",
            file_names=ref_c.MRO_UTILITIES_FILE_NAMES,
            package_directory=ref_c.MRO_UTILITIES_DIRECTORY,
            class_suffix="Utilities",
        )
        if target == "constants":
            return (constants_spec,)
        if target == "typings":
            return (typings_spec,)
        if target == "protocols":
            return (protocols_spec,)
        if target == "models":
            return (models_spec,)
        if target == "utilities":
            return (utilities_spec,)
        return (
            constants_spec,
            typings_spec,
            protocols_spec,
            models_spec,
            utilities_spec,
        )

    @staticmethod
    def _iter_constants_files(*, project_root: Path) -> list[Path]:
        ref_c: type[c.Infra.Refactor] = c.Infra.Refactor
        constants_spec = _MROTargetSpec(
            family_alias="c",
            file_names=ref_c.MRO_CONSTANTS_FILE_NAMES,
            package_directory=ref_c.MRO_CONSTANTS_DIRECTORY,
            class_suffix=ref_c.CONSTANTS_CLASS_SUFFIX,
        )
        return FlextInfraRefactorMROMigrationScanner._iter_target_files(
            project_root=project_root,
            target_spec=constants_spec,
        )

    @staticmethod
    def _iter_target_files(
        *, project_root: Path, target_spec: _MROTargetSpec
    ) -> list[Path]:
        ref_c: type[c.Infra.Refactor] = c.Infra.Refactor
        candidates: set[Path] = set()
        for directory_name in ref_c.MRO_SCAN_DIRECTORIES:
            root: Path = project_root / directory_name
            if not root.is_dir():
                continue
            for file_path in root.rglob(c.Infra.Extensions.PYTHON_GLOB):
                if file_path.name in target_spec.file_names:
                    candidates.add(file_path)
                    continue
                if target_spec.package_directory in file_path.parts:
                    candidates.add(file_path)
        return sorted(candidates)

    @staticmethod
    def _module_path(*, file_path: Path, project_root: Path) -> str:
        return u.Infra.Refactor.module_path(
            file_path=file_path, project_root=project_root
        )

    @staticmethod
    def _facade_class_name(*, tree: ast.Module, target_spec: _MROTargetSpec) -> str:
        for stmt in tree.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if (
                not isinstance(target, ast.Name)
                or target.id != target_spec.family_alias
            ):
                continue
            if not isinstance(stmt.value, ast.Name):
                continue
            class_name = stmt.value.id
            if class_name.endswith(target_spec.class_suffix):
                return class_name
        for stmt in tree.body:
            if not isinstance(stmt, ast.ClassDef):
                continue
            if stmt.name.endswith(target_spec.class_suffix):
                return stmt.name
        return ""

    @staticmethod
    def _typing_candidate_from_statement(
        *, stmt: ast.stmt
    ) -> m.Infra.Refactor.MROSymbolCandidate | None:
        if isinstance(stmt, ast.TypeAlias):
            symbol = stmt.name.id
            if TYPE_CANDIDATE_PATTERN.match(symbol) is None:
                return None
            return _new_symbol_candidate(
                symbol=symbol,
                line=stmt.lineno,
                kind="typealias",
            )
        if isinstance(stmt, ast.AnnAssign):
            if not isinstance(stmt.target, ast.Name):
                return None
            symbol = stmt.target.id
            if TYPE_CANDIDATE_PATTERN.match(symbol) is None:
                return None
            if not FlextInfraRefactorMROMigrationScanner._is_type_alias_annotation(
                annotation=stmt.annotation
            ):
                return None
            return _new_symbol_candidate(
                symbol=symbol,
                line=stmt.lineno,
                kind="typealias",
            )
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                return None
            symbol = stmt.targets[0].id
            if TYPE_CANDIDATE_PATTERN.match(symbol) is None:
                return None
            if not FlextInfraRefactorMROMigrationScanner._is_typing_factory_call(
                expr=stmt.value
            ):
                return None
            return _new_symbol_candidate(
                symbol=symbol,
                line=stmt.lineno,
                kind="typevar",
            )
        return None

    @staticmethod
    def _protocol_candidate_from_statement(
        *, stmt: ast.stmt
    ) -> m.Infra.Refactor.MROSymbolCandidate | None:
        if not isinstance(stmt, ast.ClassDef):
            return None
        has_protocol_base = False
        for base_expr in stmt.bases:
            if isinstance(base_expr, ast.Name) and base_expr.id == "Protocol":
                has_protocol_base = True
                break
            if isinstance(base_expr, ast.Attribute) and base_expr.attr == "Protocol":
                has_protocol_base = True
                break
            if isinstance(base_expr, ast.Subscript):
                root_expr = base_expr.value
                if isinstance(root_expr, ast.Name) and root_expr.id == "Protocol":
                    has_protocol_base = True
                    break
                if (
                    isinstance(root_expr, ast.Attribute)
                    and root_expr.attr == "Protocol"
                ):
                    has_protocol_base = True
                    break
        if not has_protocol_base:
            return None
        return _new_symbol_candidate(
            symbol=stmt.name,
            line=stmt.lineno,
            kind="protocol",
        )

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

    @staticmethod
    def _is_type_alias_annotation(*, annotation: ast.expr) -> bool:
        alias_name = "TypeAlias"
        if isinstance(annotation, ast.Name):
            return annotation.id == alias_name
        if isinstance(annotation, ast.Attribute):
            return annotation.attr == alias_name
        if isinstance(annotation, ast.Subscript):
            base = annotation.value
            if isinstance(base, ast.Name):
                return base.id == alias_name
            if isinstance(base, ast.Attribute):
                return base.attr == alias_name
        return False

    @staticmethod
    def _is_typing_factory_call(*, expr: ast.expr) -> bool:
        if not isinstance(expr, ast.Call):
            return False
        func = expr.func
        if isinstance(func, ast.Name):
            return func.id in {"TypeVar", "ParamSpec", "TypeVarTuple", "NewType"}
        if isinstance(func, ast.Attribute):
            return func.attr in {"TypeVar", "ParamSpec", "TypeVarTuple", "NewType"}
        return False


class FlextInfraRefactorMROMigrationTransformer:
    """Move module-level constants into the constants facade class."""

    @staticmethod
    def migrate_file(
        *, scan_result: m.Infra.Refactor.MROScanReport
    ) -> tuple[str, m.Infra.Refactor.MROFileMigration, dict[str, str]]:
        """Transform a candidate file and return code plus symbol map."""
        source = Path(scan_result.file).read_text(encoding=c.Infra.Encoding.DEFAULT)
        module = cst.parse_module(source)
        candidate_symbols = {candidate.symbol for candidate in scan_result.candidates}
        moved_statements: list[tuple[str, cst.CSTNode]] = []
        retained_module_body: list[cst.CSTNode] = []
        for stmt in module.body:
            moved = FlextInfraRefactorMROMigrationTransformer._extract_moved_statement(
                statement=stmt, candidate_symbols=candidate_symbols
            )
            if moved is None:
                retained_module_body.append(stmt)
                continue
            moved_statements.append(moved)
        if len(moved_statements) == 0:
            return (
                source,
                _new_file_migration(
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
        for retained_stmt in retained_module_body:
            if (
                isinstance(retained_stmt, cst.ClassDef)
                and retained_stmt.name.value == class_name
            ):
                class_found = True
                transformed_class, class_symbol_map = (
                    FlextInfraRefactorMROMigrationTransformer._migrate_constants_class(
                        class_def=retained_stmt,
                        moved_by_symbol=moved_by_symbol,
                        ordered_symbols=ordered_symbols,
                    )
                )
                transformed_body.append(transformed_class)
                symbol_map.update(class_symbol_map)
                continue
            transformed_body.append(retained_stmt)
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
            value = FlextInfraRefactorMROMigrationTransformer._statement_value(
                statement=moved_by_symbol[symbol]
            )
            if value is None:
                continue
            replacement_values[symbol] = value
        inline_transformer = FlextInfraRefactorMROPrivateInlineTransformer(
            replacement_values=replacement_values
        )
        updated_module = updated_module.visit(inline_transformer)
        migration = _new_file_migration(
            file=scan_result.file,
            module=scan_result.module,
            moved_symbols=tuple(ordered_symbols),
            created_classes=created_classes,
        )
        return (updated_module.code, migration, symbol_map)

    @staticmethod
    def _extract_moved_statement(
        *, statement: cst.CSTNode, candidate_symbols: set[str]
    ) -> tuple[str, cst.CSTNode] | None:
        if isinstance(statement, cst.ClassDef):
            symbol = statement.name.value
            if symbol in candidate_symbols:
                return (symbol, statement)
            return None
        if not isinstance(statement, cst.SimpleStatementLine):
            return None
        if len(statement.body) != 1:
            return None
        first_stmt = statement.body[0]
        if isinstance(first_stmt, cst.AnnAssign):
            if not isinstance(first_stmt.target, cst.Name):
                return None
            symbol = first_stmt.target.value
        elif isinstance(first_stmt, cst.Assign):
            if len(first_stmt.targets) != 1:
                return None
            assign_target = first_stmt.targets[0].target
            if not isinstance(assign_target, cst.Name):
                return None
            symbol = assign_target.value
        else:
            return None
        if symbol not in candidate_symbols:
            return None
        return (symbol, first_stmt)

    @staticmethod
    def _migrate_constants_class(
        *,
        class_def: cst.ClassDef,
        moved_by_symbol: dict[str, cst.CSTNode],
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
                private_value = (
                    FlextInfraRefactorMROMigrationTransformer._statement_value(
                        statement=moved_by_symbol[alias[1]]
                    )
                )
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
            moved_node = FlextInfraRefactorMROMigrationTransformer._retarget_statement(
                statement=moved_by_symbol[symbol],
                target_name=target,
                replacement_value=replacement_value,
            )
            moved_lines.append(moved_node)
        cleaned_body = [
            statement
            for statement in retained_class_body
            if not (
                len(moved_lines) > 0
                and isinstance(statement, cst.SimpleStatementLine)
                and (len(statement.body) == 1)
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
        moved_by_symbol: dict[str, cst.CSTNode],
        ordered_symbols: list[str],
    ) -> tuple[cst.ClassDef, dict[str, str]]:
        class_template = cst.ClassDef(
            name=cst.Name(class_name), body=cst.IndentedBlock(body=())
        )
        class_body: list[cst.BaseStatement] = []
        symbol_map: dict[str, str] = {}
        for symbol in ordered_symbols:
            target = FlextInfraRefactorMROMigrationTransformer._default_target(
                symbol=symbol
            )
            symbol_map[symbol] = target
            moved_node = FlextInfraRefactorMROMigrationTransformer._retarget_statement(
                statement=moved_by_symbol[symbol],
                target_name=target,
                replacement_value=None,
            )
            if isinstance(moved_node, cst.BaseStatement):
                class_body.append(moved_node)
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
            return (assign.target.value, assign.value.value)
        if isinstance(assign, cst.Assign):
            if len(assign.targets) != 1:
                return None
            if not isinstance(assign.targets[0].target, cst.Name):
                return None
            if not isinstance(assign.value, cst.Name):
                return None
            return (assign.targets[0].target.value, assign.value.value)
        return None

    @staticmethod
    def _default_target(*, symbol: str) -> str:
        stripped = symbol.lstrip("_")
        return stripped or symbol

    @staticmethod
    def _statement_value(*, statement: cst.CSTNode) -> cst.BaseExpression | None:
        if isinstance(statement, cst.AnnAssign):
            return statement.value
        if isinstance(statement, cst.Assign):
            return statement.value
        return None

    @staticmethod
    def _retarget_statement(
        *,
        statement: cst.CSTNode,
        target_name: str,
        replacement_value: cst.BaseExpression | None,
    ) -> cst.CSTNode:
        if isinstance(statement, cst.ClassDef):
            if statement.name.value == target_name:
                return statement
            return statement.with_changes(name=cst.Name(target_name))
        if isinstance(statement, cst.AnnAssign):
            if replacement_value is not None:
                return cst.SimpleStatementLine(
                    body=[
                        statement.with_changes(
                            target=cst.Name(target_name), value=replacement_value
                        )
                    ]
                )
            return cst.SimpleStatementLine(
                body=[statement.with_changes(target=cst.Name(target_name))]
            )
        if isinstance(statement, cst.Assign):
            assign_value = replacement_value or statement.value
            return cst.SimpleStatementLine(
                body=[
                    statement.with_changes(
                        targets=(cst.AssignTarget(target=cst.Name(target_name)),),
                        value=assign_value,
                    )
                ]
            )
        msg = "unsupported constant statement type"
        raise ValueError(msg)


class FlextInfraRefactorMROImportRewriter:
    """Rewrite imports and references to use the local facade alias."""

    @classmethod
    def rewrite_workspace(
        cls,
        *,
        workspace_root: Path,
        moved_index: dict[str, dict[str, str]],
        module_facade_aliases: dict[str, str],
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
                module_facade_aliases=module_facade_aliases,
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
        module_facade_aliases: dict[str, str],
        apply_changes: bool,
    ) -> m.Infra.Refactor.MRORewriteResult | None:
        """Rewrite one file according to moved constant symbol mappings."""
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = ast.parse(source)
        except (OSError, SyntaxError):
            return None
        imported_symbols: dict[str, m.Infra.Refactor.MROImportRewrite] = {}
        module_aliases: dict[str, str] = {}
        facade_aliases: dict[str, str] = {}
        module_facade_alias: dict[str, str] = {}
        facade_imports_needed: set[str] = set()
        facade_import_objects: dict[str, m.Infra.Refactor.MROImportRewrite] = {}
        for stmt in tree.body:
            if isinstance(stmt, ast.ImportFrom):
                module_name = stmt.module
                if module_name is None or module_name not in moved_index:
                    continue
                if any(alias.name == "*" for alias in stmt.names):
                    continue
                kept_names: list[ast.alias] = []
                for alias in stmt.names:
                    default_facade_alias = module_facade_aliases.get(
                        module_name, c.Infra.Refactor.DEFAULT_FACADE_ALIAS
                    )
                    if alias.name == default_facade_alias:
                        facade_local_name = default_facade_alias
                        facade_aliases[facade_local_name] = module_name
                        module_facade_alias[module_name] = facade_local_name
                        facade_import = _new_import_rewrite(
                            module=module_name,
                            import_name=default_facade_alias,
                            as_name=None,
                            symbol="",
                            facade_name=facade_local_name,
                        )
                        facade_key = f"{facade_import.module}:{facade_import.import_name}:{facade_import.as_name or ''}"
                        facade_imports_needed.add(facade_key)
                        facade_import_objects[facade_key] = facade_import
                        if alias.asname is None or alias.asname == default_facade_alias:
                            kept_names.append(ast.alias(name=default_facade_alias))
                        continue
                    symbol_map = moved_index[module_name]
                    new_symbol = symbol_map.get(alias.name)
                    if new_symbol is None:
                        kept_names.append(alias)
                        continue
                    imported_symbols[alias.asname or alias.name] = _new_import_rewrite(
                        module=module_name,
                        import_name=default_facade_alias,
                        as_name=None,
                        symbol=new_symbol,
                        facade_name=module_facade_alias.get(
                            module_name, default_facade_alias
                        ),
                    )
                    facade_import = _new_import_rewrite(
                        module=module_name,
                        import_name=default_facade_alias,
                        as_name=None
                        if module_name not in module_facade_alias
                        else (
                            None
                            if module_facade_alias[module_name] == default_facade_alias
                            else module_facade_alias[module_name]
                        ),
                        symbol="",
                        facade_name=module_facade_alias.get(
                            module_name, default_facade_alias
                        ),
                    )
                    facade_key = f"{facade_import.module}:{facade_import.import_name}:{facade_import.as_name or ''}"
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
                and (len(stmt.names) == 0)
            )
        ]
        existing_imports: set[str] = set()
        for stmt in rewritten.body:
            if isinstance(stmt, ast.ImportFrom) and stmt.module is not None:
                for alias in stmt.names:
                    if alias.name != "*":
                        key = f"{stmt.module}:{alias.name}:{alias.asname or ''}"
                        existing_imports.add(key)
        imports_to_add = sorted(facade_imports_needed - existing_imports)
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
            _ = file_path.write_text(f"{rendered}\n", encoding=c.Infra.Encoding.DEFAULT)
        return _new_rewrite_result(
            file=str(file_path),
            replacements=rewriter.replacements,
        )

    @staticmethod
    def _iter_workspace_python_files(*, workspace_root: Path) -> list[Path]:
        return u.Infra.Refactor.iter_python_files(workspace_root=workspace_root)

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
            workspace_root=workspace_root, target=target
        )
        remaining = sum(len(item.candidates) for item in file_results)
        return (remaining, 0)


__all__ = [
    "FlextInfraRefactorMROImportRewriter",
    "FlextInfraRefactorMROMigrationScanner",
    "FlextInfraRefactorMROMigrationTransformer",
    "FlextInfraRefactorMROMigrationValidator",
]
