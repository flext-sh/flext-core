"""Violation analysis helpers for flext_infra.refactor."""

from __future__ import annotations

import ast
import operator
import sys
from collections import Counter
from collections.abc import Mapping
from operator import itemgetter
from pathlib import Path
from typing import override

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_core import r
from flext_infra import FlextInfraRefactorLooseClassScanner, c, m, u

type _ClassNestingMappingIndex = dict[
    tuple[str, str], m.Infra.Refactor.ClassNestingMapping
]


class ImportDependencyCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.local_to_import: dict[str, str] = {}

    @override
    def visit_Import(self, node: cst.Import) -> None:
        for raw_alias in node.names:
            imported = u.Infra.dotted_name(raw_alias.name)
            if not imported:
                continue
            local_name = u.Infra.asname_to_local(raw_alias.asname)
            if local_name is None:
                local_name = imported.split(".", maxsplit=1)[0]
            self.local_to_import[local_name] = imported

    @override
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if isinstance(node.names, cst.ImportStar):
            return
        if node.module is None:
            return
        module_name = u.Infra.dotted_name(node.module)
        if not module_name:
            return
        for raw_alias in node.names:
            if not isinstance(raw_alias.name, cst.Name):
                continue
            imported_name = raw_alias.name.value
            if imported_name == "*":
                continue
            local_name = imported_name
            local_name_from_alias = u.Infra.asname_to_local(raw_alias.asname)
            if local_name_from_alias is not None:
                local_name = local_name_from_alias
            self.local_to_import[local_name] = f"{module_name}.{imported_name}"


class FunctionDependencyCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    @override
    def visit_Name(self, node: cst.Name) -> None:
        self.names.add(node.value)


class FlextInfraRefactorPydanticCentralizerAnalysis:
    _MODEL_BASES: tuple[str, ...] = (
        "BaseModel",
        "RootModel",
        "TypedDict",
        "ArbitraryTypesModel",
        "FrozenModel",
        "FrozenStrictModel",
    )
    _TYPED_DICT_MIN_ARGS: int = 2

    @staticmethod
    def _class_base_names(node: ast.ClassDef) -> set[str]:
        names: set[str] = set()
        for base in node.bases:
            if isinstance(base, ast.Name):
                names.add(base.id)
            elif isinstance(base, ast.Attribute):
                names.add(base.attr)
                root = ast.unparse(base)
                if root:
                    names.add(root)
        return names

    @staticmethod
    def _is_model_like_base_name(base_name: str) -> bool:
        if base_name in FlextInfraRefactorPydanticCentralizerAnalysis._MODEL_BASES:
            return True
        return bool(base_name.startswith("FlextModels."))

    @staticmethod
    def is_top_level_model_class(node: ast.stmt) -> bool:
        if not isinstance(node, ast.ClassDef):
            return False
        base_names = FlextInfraRefactorPydanticCentralizerAnalysis._class_base_names(
            node
        )
        return any(
            FlextInfraRefactorPydanticCentralizerAnalysis._is_model_like_base_name(
                base_name,
            )
            for base_name in base_names
        )

    @staticmethod
    def _is_typings_scope(file_path: Path) -> bool:
        posix = file_path.as_posix()
        return posix.endswith(("/typings.py", "/_typings.py")) or "/typings/" in posix

    @staticmethod
    def _is_dict_like_expr(expr: str) -> bool:
        return any(
            marker in expr for marker in ("dict[", "Mapping[", "MutableMapping[")
        )

    @staticmethod
    def _is_dict_like_alias(
        node: ast.stmt,
        source: str,
        *,
        file_path: Path,
    ) -> m.Infra.Refactor.AliasMove | None:
        keys = (
            "dict",
            "payload",
            "schema",
            "entry",
            "config",
            "metadata",
            "fixture",
            "case",
        )
        is_typings_scope = (
            FlextInfraRefactorPydanticCentralizerAnalysis._is_typings_scope(
                file_path,
            )
        )
        match node:
            case ast.TypeAlias():
                alias_name = node.name.id
                if (not is_typings_scope) and (
                    not any(token in alias_name.lower() for token in keys)
                ):
                    return None
                expr = ast.get_source_segment(source, node.value)
                if expr is None:
                    return None
                if not FlextInfraRefactorPydanticCentralizerAnalysis._is_dict_like_expr(
                    expr,
                ):
                    return None
                return m.Infra.Refactor.AliasMove(
                    name=alias_name,
                    start=node.lineno,
                    end=node.end_lineno or node.lineno,
                    alias_expr=expr,
                )
            case ast.AnnAssign():
                if not isinstance(node.target, ast.Name):
                    return None
                alias_name = node.target.id
                if (not is_typings_scope) and (
                    not any(token in alias_name.lower() for token in keys)
                ):
                    return None
                if node.value is None:
                    return None
                expr = ast.get_source_segment(source, node.value)
                if expr is None:
                    return None
                if not FlextInfraRefactorPydanticCentralizerAnalysis._is_dict_like_expr(
                    expr,
                ):
                    return None
                annotation = ast.get_source_segment(source, node.annotation) or ""
                if ("TypeAlias" not in annotation) and (not is_typings_scope):
                    return None
                return m.Infra.Refactor.AliasMove(
                    name=alias_name,
                    start=node.lineno,
                    end=node.end_lineno or node.lineno,
                    alias_expr=expr,
                )
            case ast.Assign():
                if len(node.targets) != 1:
                    return None
                if not isinstance(node.targets[0], ast.Name):
                    return None
                alias_name = node.targets[0].id
                if (not is_typings_scope) and (
                    not any(token in alias_name.lower() for token in keys)
                ):
                    return None
                expr = ast.get_source_segment(source, node.value)
                if expr is None:
                    return None
                if not FlextInfraRefactorPydanticCentralizerAnalysis._is_dict_like_expr(
                    expr,
                ):
                    return None
                return m.Infra.Refactor.AliasMove(
                    name=alias_name,
                    start=node.lineno,
                    end=node.end_lineno or node.lineno,
                    alias_expr=expr,
                )
            case _:
                return None

    @staticmethod
    def _typed_dict_factory_model(
        node: ast.Assign,
    ) -> m.Infra.Refactor.ClassMove | None:
        if len(node.targets) != 1:
            return None
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            return None
        if not isinstance(node.value, ast.Call):
            return None
        func = node.value.func
        is_typed_dict_factory = False
        if isinstance(func, ast.Name):
            is_typed_dict_factory = func.id == "TypedDict"
        elif isinstance(func, ast.Attribute):
            is_typed_dict_factory = func.attr == "TypedDict"
        if not is_typed_dict_factory:
            return None
        if (
            len(node.value.args)
            < FlextInfraRefactorPydanticCentralizerAnalysis._TYPED_DICT_MIN_ARGS
        ):
            return None
        field_map_arg = node.value.args[1]
        if not isinstance(field_map_arg, ast.Dict):
            return None
        field_lines: list[str] = []
        total_false = False
        for kw in node.value.keywords:
            if (
                kw.arg == "total"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is False
            ):
                total_false = True
        for key_node, value_node in zip(
            field_map_arg.keys,
            field_map_arg.values,
            strict=True,
        ):
            if not isinstance(key_node, ast.Constant):
                continue
            key_value = key_node.value
            if not isinstance(key_value, str):
                continue
            annotation = ast.unparse(value_node)
            if total_false:
                field_lines.append(
                    f"    {key_value}: Annotated[{annotation} | None, Field(default=None)]",
                )
            else:
                field_lines.append(f"    {key_value}: {annotation}")
        if len(field_lines) == 0:
            field_lines.append("    pass")
        rendered_fields = "\n".join(field_lines)
        rendered_class = (
            f"class {target.id}(BaseModel):\n"
            '    model_config = ConfigDict(extra="forbid")\n'
            f"{rendered_fields}\n"
        )
        return m.Infra.Refactor.ClassMove(
            name=target.id,
            start=node.lineno,
            end=node.end_lineno or node.lineno,
            source=rendered_class,
            kind="typed_dict_factory",
        )

    @staticmethod
    def _typed_dict_total_false(node: ast.ClassDef) -> bool:
        for keyword in node.keywords:
            if keyword.arg == "total" and isinstance(keyword.value, ast.Constant):
                return bool(keyword.value.value) is False
        return False

    @staticmethod
    def _build_model_from_typed_dict(node: ast.ClassDef, source: str) -> str:
        total_false = (
            FlextInfraRefactorPydanticCentralizerAnalysis._typed_dict_total_false(
                node,
            )
        )
        fields: list[str] = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                ann = ast.get_source_segment(source, stmt.annotation)
                if ann is None:
                    continue
                if total_false:
                    fields.append(
                        f"    {stmt.target.id}: Annotated[{ann} | None, Field(default=None)]",
                    )
                else:
                    fields.append(f"    {stmt.target.id}: {ann}")
        if len(fields) == 0:
            fields.append("    pass")
        body = "\n".join(fields)
        return f'class {node.name}(BaseModel):\n    model_config = ConfigDict(extra="forbid")\n{body}\n'

    @staticmethod
    def collect_moves(
        file_path: Path,
    ) -> tuple[list[m.Infra.Refactor.ClassMove], list[m.Infra.Refactor.AliasMove]]:
        source = file_path.read_text(encoding="utf-8")
        # given source text is needed for ast.get_source_segment below
        tree = u.Infra.parse_ast_from_source(source)
        if tree is None:
            msg = "Failed to parse source"
            raise SyntaxError(msg)
        lines = source.splitlines()
        class_moves: list[m.Infra.Refactor.ClassMove] = []
        alias_moves: list[m.Infra.Refactor.AliasMove] = []
        for stmt in tree.body:
            typed_dict_factory_move = (
                FlextInfraRefactorPydanticCentralizerAnalysis._typed_dict_factory_model(
                    stmt
                )
                if isinstance(stmt, ast.Assign)
                else None
            )
            if typed_dict_factory_move is not None:
                class_moves.append(typed_dict_factory_move)
                continue
            if FlextInfraRefactorPydanticCentralizerAnalysis.is_top_level_model_class(
                stmt
            ):
                if not isinstance(stmt, ast.ClassDef):
                    continue
                start = stmt.lineno
                end = stmt.end_lineno or stmt.lineno
                snippet = "\n".join(lines[start - 1 : end])
                base_names = (
                    FlextInfraRefactorPydanticCentralizerAnalysis._class_base_names(
                        stmt
                    )
                )
                kind = "typed_dict" if "TypedDict" in base_names else "base_model"
                if kind == "typed_dict":
                    snippet = FlextInfraRefactorPydanticCentralizerAnalysis._build_model_from_typed_dict(
                        stmt,
                        source,
                    )
                class_moves.append(
                    m.Infra.Refactor.ClassMove(
                        name=stmt.name,
                        start=start,
                        end=end,
                        source=snippet,
                        kind=kind,
                    ),
                )
                continue
            alias_move = (
                FlextInfraRefactorPydanticCentralizerAnalysis._is_dict_like_alias(
                    stmt,
                    source,
                    file_path=file_path,
                )
            )
            if alias_move is not None:
                alias_moves.append(alias_move)
        return (class_moves, alias_moves)

    @staticmethod
    def collect_moves_safe(
        file_path: Path,
        *,
        failure_stats: m.Infra.Refactor.CentralizerFailureStats,
    ) -> (
        tuple[list[m.Infra.Refactor.ClassMove], list[m.Infra.Refactor.AliasMove]] | None
    ):
        try:
            return FlextInfraRefactorPydanticCentralizerAnalysis.collect_moves(
                file_path
            )
        except SyntaxError:
            failure_stats.parse_syntax_errors += 1
            return None
        except UnicodeDecodeError:
            failure_stats.parse_encoding_errors += 1
            return None
        except OSError:
            failure_stats.parse_io_errors += 1
            return None

    @staticmethod
    def scan_file_violations(file_path: Path) -> tuple[int, int]:
        try:
            source = file_path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return (0, 0)
        # given source text is reused for dict-like alias detection
        tree = u.Infra.parse_ast_from_source(source)
        if tree is None:
            return (0, 0)
        model_class_count = 0
        dict_alias_count = 0
        for stmt in tree.body:
            if FlextInfraRefactorPydanticCentralizerAnalysis.is_top_level_model_class(
                stmt,
            ):
                model_class_count += 1
                continue
            if (
                FlextInfraRefactorPydanticCentralizerAnalysis._is_dict_like_alias(
                    stmt,
                    source,
                    file_path=file_path,
                )
                is not None
            ):
                dict_alias_count += 1
        return (model_class_count, dict_alias_count)

    @staticmethod
    def rewrite_source(
        file_path: Path,
        class_moves: list[m.Infra.Refactor.ClassMove],
        alias_moves: list[m.Infra.Refactor.AliasMove],
        *,
        import_statement: str,
    ) -> str:
        source = file_path.read_text(encoding="utf-8")
        lines = source.splitlines()
        ranges = sorted(
            [(m.start, m.end) for m in class_moves]
            + [(a.start, a.end) for a in alias_moves],
            key=operator.itemgetter(0),
            reverse=True,
        )
        for start, end in ranges:
            del lines[start - 1 : end]
        updated = "\n".join(lines)
        moved_names = [m.name for m in class_moves] + [a.name for a in alias_moves]
        if moved_names:
            updated = FlextInfraRefactorPydanticCentralizerAnalysis.insert_import(
                updated,
                import_statement,
            )
        if source.endswith("\n") and (not updated.endswith("\n")):
            updated += "\n"
        return updated

    @staticmethod
    def insert_import(source: str, import_stmt: str) -> str:
        return u.Infra.insert_import_statement(source, import_stmt)


class FlextInfraRefactorClassNestingAnalyzer:
    """Analyze files for class nesting violations using YAML mapping rules."""

    @classmethod
    def analyze_files(cls, files: list[Path]) -> m.Infra.Refactor.ClassNestingReport:
        """Return aggregate and per-file class-nesting violation counts."""
        if not files:
            return m.Infra.Refactor.ClassNestingReport(
                violations_count=0,
                confidence_counts={},
                violations=[],
                per_file_counts={},
            )
        grouped_targets = cls._group_targets_by_project_root(files)
        if not grouped_targets:
            return m.Infra.Refactor.ClassNestingReport(
                violations_count=0,
                confidence_counts={},
                violations=[],
                per_file_counts={},
            )
        scanner = FlextInfraRefactorLooseClassScanner()
        mapping_result = cls._load_mapping_index()
        mapping_index: _ClassNestingMappingIndex = (
            mapping_result.value if mapping_result.is_success else {}
        )
        confidence_counts: Counter[str] = Counter()
        per_file_counts: Counter[str] = Counter()
        violations: list[m.Infra.Refactor.ClassNestingViolation] = []
        for project_root, target_files in grouped_targets.items():
            scan_result = scanner.scan(project_root)
            if scan_result.is_failure:
                continue
            try:
                parsed_violations: list[m.Infra.Refactor.LooseClassViolation] = (
                    TypeAdapter(
                        list[m.Infra.Refactor.LooseClassViolation],
                    ).validate_python(
                        scan_result.value.get(c.Infra.ReportKeys.VIOLATIONS, [])
                    )
                )
            except ValidationError:
                continue
            for parsed_violation in parsed_violations:
                normalized_file = cls._normalize_module_path(parsed_violation.file)
                if target_files and normalized_file not in target_files:
                    continue
                line = parsed_violation.line if parsed_violation.line > 0 else 1
                confidence = parsed_violation.confidence or c.Infra.Severity.LOW
                target_namespace = ""
                rewrite_scope = c.Infra.ReportKeys.FILE
                mapped_entry = mapping_index.get((
                    normalized_file,
                    parsed_violation.class_name,
                ))
                if mapped_entry is not None:
                    target_namespace = mapped_entry.target_namespace
                    confidence = mapped_entry.confidence
                    rewrite_scope = cls._normalize_rewrite_scope(
                        mapped_entry.rewrite_scope,
                    )
                elif parsed_violation.expected_prefix:
                    target_namespace = parsed_violation.expected_prefix
                violations.append(
                    m.Infra.Refactor.ClassNestingViolation(
                        file=normalized_file,
                        line=line,
                        class_name=parsed_violation.class_name,
                        target_namespace=target_namespace,
                        confidence=confidence,
                        rewrite_scope=rewrite_scope,
                    ),
                )
                confidence_counts[confidence] += 1
                per_file_counts[normalized_file] += 1
        return m.Infra.Refactor.ClassNestingReport(
            violations_count=len(violations),
            confidence_counts=dict(confidence_counts.items()),
            violations=violations,
            per_file_counts=dict(per_file_counts.items()),
        )

    @classmethod
    def _group_targets_by_project_root(cls, files: list[Path]) -> dict[Path, set[str]]:
        grouped: dict[Path, set[str]] = {}
        for file_path in files:
            project_root = cls._find_project_root(file_path)
            if project_root is None:
                continue
            module_path = cls._module_path_for_file(file_path, project_root)
            if module_path is None:
                continue
            grouped.setdefault(project_root, set()).add(module_path)
        return grouped

    @classmethod
    def _find_project_root(cls, file_path: Path) -> Path | None:
        resolved = file_path.resolve()
        for parent in (resolved.parent, *resolved.parents):
            src_dir = parent / c.Infra.Paths.DEFAULT_SRC_DIR
            if not src_dir.is_dir():
                continue
            try:
                resolved.relative_to(src_dir.resolve())
                return parent
            except ValueError:
                continue
        return None

    @classmethod
    def _module_path_for_file(cls, file_path: Path, project_root: Path) -> str | None:
        src_dir = (project_root / c.Infra.Paths.DEFAULT_SRC_DIR).resolve()
        resolved = file_path.resolve()
        try:
            relative = resolved.relative_to(src_dir)
        except ValueError:
            return None
        return relative.as_posix()

    @classmethod
    def _load_mapping_index(cls) -> r[_ClassNestingMappingIndex]:
        mapping_path = (
            Path(__file__).resolve().parent / c.Infra.Refactor.MAPPINGS_RELATIVE_PATH
        )
        try:
            typed_doc = u.Infra.safe_load_yaml(mapping_path)
        except (OSError, TypeError) as exc:
            return r[_ClassNestingMappingIndex].fail(str(exc))
        raw_nesting = typed_doc.get(c.Infra.ReportKeys.CLASS_NESTING)
        if not isinstance(raw_nesting, list):
            return r[_ClassNestingMappingIndex].ok({})
        try:
            entries = TypeAdapter(
                list[m.Infra.Refactor.ClassNestingMapping],
            ).validate_python(raw_nesting)
        except ValidationError as exc:
            return r[_ClassNestingMappingIndex].fail(str(exc))
        index: _ClassNestingMappingIndex = {}
        for entry in entries:
            scope = cls._normalize_rewrite_scope(entry.rewrite_scope)
            norm = cls._normalize_module_path(entry.current_file)
            index[norm, entry.loose_name] = m.Infra.Refactor.ClassNestingMapping(
                loose_name=entry.loose_name,
                current_file=entry.current_file,
                target_namespace=entry.target_namespace,
                confidence=entry.confidence,
                rewrite_scope=scope,
                target_name=entry.target_name,
                reason=entry.reason,
            )
        return r[_ClassNestingMappingIndex].ok(index)

    @classmethod
    def _normalize_module_path(cls, raw_path: str) -> str:
        normalized = raw_path.replace("\\", "/")
        path = Path(normalized)
        parts = path.parts
        if c.Infra.Paths.DEFAULT_SRC_DIR in parts:
            src_index = parts.index(c.Infra.Paths.DEFAULT_SRC_DIR)
            suffix = parts[src_index + 1 :]
            if suffix:
                return Path(*suffix).as_posix()
        return path.as_posix().lstrip("./")

    @classmethod
    def _normalize_rewrite_scope(cls, raw_scope: str | None) -> str:
        if not isinstance(raw_scope, str):
            return c.Infra.ReportKeys.FILE
        candidate = raw_scope.strip().lower()
        if candidate in {
            c.Infra.ReportKeys.FILE,
            c.Infra.Toml.PROJECT,
            c.Infra.ReportKeys.WORKSPACE,
        }:
            return candidate
        return c.Infra.ReportKeys.FILE


class FlextInfraRefactorViolationAnalyzer:
    """Scan files and aggregate massive pattern violations."""

    @classmethod
    def analyze_files(
        cls,
        files: list[Path],
    ) -> m.Infra.Refactor.ViolationAnalysisReport:
        """Return aggregate and per-file violation counts."""
        totals: Counter[str] = Counter()
        per_file: dict[str, dict[str, int]] = {}
        helper_suggestions: list[m.Infra.Refactor.HelperClassification] = []
        helper_totals: Counter[str] = Counter()
        helper_manual_review: list[m.Infra.Refactor.HelperClassification] = []
        for file_path in files:
            try:
                content = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            except (OSError, UnicodeDecodeError):
                continue
            helper_analysis = cls._analyze_file_helpers(
                file_path=file_path,
            )
            helper_suggestions.extend(helper_analysis.suggestions)
            helper_totals.update(helper_analysis.totals)
            helper_manual_review.extend(helper_analysis.manual_review)
            file_counts: dict[str, int] = {}
            for name, pattern in c.Infra.Refactor.VIOLATION_PATTERNS.items():
                count = len(pattern.findall(content))
                if count <= 0:
                    continue
                totals[name] += count
                file_counts[name] = count
            if file_counts:
                per_file[str(file_path)] = file_counts
        class_nesting = FlextInfraRefactorClassNestingAnalyzer.analyze_files(files)
        class_nesting_count = class_nesting.violations_count
        if class_nesting_count > 0:
            totals[c.Infra.ReportKeys.CLASS_NESTING] += class_nesting_count
        for raw_file, raw_count in class_nesting.per_file_counts.items():
            counts = per_file.setdefault(raw_file, {})
            counts[c.Infra.ReportKeys.CLASS_NESTING] = raw_count
        ranked_files: list[tuple[str, int, dict[str, int]]] = []
        for file_name, counts in per_file.items():
            ranked_files.append((file_name, sum(counts.values()), counts))
        ranked_files.sort(key=itemgetter(1), reverse=True)
        hottest_files = [
            m.Infra.Refactor.ViolationAnalysisReport.TopFileSection(
                file=file_name,
                total=total,
                counts=counts,
            )
            for file_name, total, counts in ranked_files[:25]
        ]
        helper_report = m.Infra.Refactor.HelperClassificationReport(
            totals=dict(helper_totals.items()),
            suggestions=helper_suggestions,
            manual_review=helper_manual_review,
        )
        return m.Infra.Refactor.ViolationAnalysisReport(
            totals=dict(totals.items()),
            files=per_file,
            top_files=hottest_files,
            files_scanned=len(files),
            helper_classification=helper_report,
            class_nesting=class_nesting,
        )

    @classmethod
    def _analyze_file_helpers(
        cls,
        *,
        file_path: Path,
    ) -> m.Infra.Refactor.HelperFileAnalysis:
        suggestions: list[m.Infra.Refactor.HelperClassification] = []
        totals: Counter[str] = Counter()
        manual_review: list[m.Infra.Refactor.HelperClassification] = []
        module = u.Infra.parse_module_cst(file_path)
        if module is None:
            return m.Infra.Refactor.HelperFileAnalysis(
                suggestions=suggestions,
                totals=dict(totals.items()),
                manual_review=manual_review,
            )
        import_collector = ImportDependencyCollector()
        _ = module.visit(import_collector)
        for stmt in module.body:
            if not isinstance(stmt, cst.FunctionDef):
                continue
            classification = cls._classify_helper_function(
                file_path=file_path,
                function=stmt,
                local_to_import=import_collector.local_to_import,
            )
            suggestions.append(classification)
            category = classification.category
            totals[category] += 1
            if classification.manual_review:
                manual_review.append(classification)
        return m.Infra.Refactor.HelperFileAnalysis(
            suggestions=suggestions,
            totals=dict(totals.items()),
            manual_review=manual_review,
        )

    @classmethod
    def _classify_helper_function(
        cls,
        *,
        file_path: Path,
        function: cst.FunctionDef,
        local_to_import: Mapping[str, str],
    ) -> m.Infra.Refactor.HelperClassification:
        dependency_collector = FunctionDependencyCollector()
        function.visit(dependency_collector)
        dependencies: set[str] = set()
        for name in dependency_collector.names:
            imported = local_to_import.get(name)
            if imported is not None:
                dependencies.add(imported)
        decorator_dependencies: set[str] = set()
        for decorator in function.decorators:
            decorator_root = u.Infra.root_name(decorator.decorator)
            if not decorator_root:
                continue
            imported = local_to_import.get(decorator_root)
            if imported is not None:
                decorator_dependencies.add(imported)
        dependencies.update(decorator_dependencies)
        matched_categories = cls._match_categories(
            dependencies=dependencies,
            has_decorators=bool(function.decorators),
        )
        category, manual, reason = cls._resolve_category(
            dependencies=dependencies,
            matched_categories=matched_categories,
        )
        namespace_root = c.Infra.Refactor.NAMESPACE_PREFIXES[category]
        return m.Infra.Refactor.HelperClassification(
            file=str(file_path),
            function=function.name.value,
            category=category,
            target_namespace=f"{namespace_root}.{function.name.value}",
            dependencies=sorted(dependencies),
            manual_review=manual,
            review_reason=reason,
        )

    @classmethod
    def _match_categories(
        cls,
        *,
        dependencies: set[str],
        has_decorators: bool,
    ) -> set[str]:
        matched: set[str] = set()
        for dependency in dependencies:
            lowered = dependency.lower()
            if any(token in lowered for token in c.Infra.Refactor.MODEL_TOKENS):
                matched.add("models")
            if any(token in lowered for token in c.Infra.Refactor.DECORATOR_TOKENS):
                matched.add("decorators")
            if any(token in lowered for token in c.Infra.Refactor.DISPATCHER_TOKENS):
                matched.add("dispatcher")
        if has_decorators:
            matched.add("decorators")
        return matched

    @classmethod
    def _resolve_category(
        cls,
        *,
        dependencies: set[str],
        matched_categories: set[str],
    ) -> tuple[str, bool, str]:
        if len(matched_categories) > 1:
            ordered = [
                category
                for category in c.Infra.Refactor.CLASSIFICATION_PRIORITY
                if category in matched_categories
            ]
            cats = ", ".join(sorted(matched_categories))
            return (ordered[0], True, f"Cross-cutting concerns detected: {cats}")
        if len(matched_categories) == 1:
            category = next(iter(matched_categories))
            return (category, False, "")
        if cls._is_pure_utility_dependencies(dependencies):
            return ("utility", False, "")
        return (
            "utility",
            True,
            "External dependencies outside helper taxonomy; manual review required",
        )

    @classmethod
    def _is_pure_utility_dependencies(cls, dependencies: set[str]) -> bool:
        if not dependencies:
            return True
        for dependency in dependencies:
            root = dependency.split(".", maxsplit=1)[0]
            if root in sys.stdlib_module_names:
                continue
            if root in {"typing", "collections", "dataclasses", "pathlib"}:
                continue
            if root == "builtins":
                continue
            return False
        return True


__all__ = [
    "FlextInfraRefactorClassNestingAnalyzer",
    "FlextInfraRefactorViolationAnalyzer",
]
