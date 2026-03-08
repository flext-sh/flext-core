"""Violation analysis helpers for flext_infra.refactor."""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Mapping
from operator import itemgetter
from pathlib import Path
from typing import override

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_core import r
from flext_infra import c, m, u
from flext_infra.refactor.scanner import FlextInfraRefactorLooseClassScanner


class ImportDependencyCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.local_to_import: dict[str, str] = {}

    @override
    def visit_Import(self, node: cst.Import) -> None:
        for raw_alias in node.names:
            imported = u.Infra.Refactor.dotted_name(raw_alias.name)
            if not imported:
                continue
            local_name = u.Infra.Refactor.asname_to_local(raw_alias.asname)
            if local_name is None:
                local_name = imported.split(".", maxsplit=1)[0]
            self.local_to_import[local_name] = imported

    @override
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if isinstance(node.names, cst.ImportStar):
            return
        if node.module is None:
            return
        module_name = u.Infra.Refactor.dotted_name(node.module)
        if not module_name:
            return
        for raw_alias in node.names:
            if not isinstance(raw_alias.name, cst.Name):
                continue
            imported_name = raw_alias.name.value
            if imported_name == "*":
                continue
            local_name = imported_name
            local_name_from_alias = u.Infra.Refactor.asname_to_local(raw_alias.asname)
            if local_name_from_alias is not None:
                local_name = local_name_from_alias
            self.local_to_import[local_name] = f"{module_name}.{imported_name}"


class FunctionDependencyCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    @override
    def visit_Name(self, node: cst.Name) -> None:
        self.names.add(node.value)


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
        empty: dict[tuple[str, str], m.Infra.Refactor.ClassNestingMapping] = {}
        mapping_index = mapping_result.value if mapping_result.is_success else empty
        confidence_counts: Counter[str] = Counter()
        per_file_counts: Counter[str] = Counter()
        violations: list[m.Infra.Refactor.ClassNestingViolation] = []
        for project_root, target_files in grouped_targets.items():
            scan_result = scanner.scan(project_root)
            if scan_result.is_failure:
                continue
            raw_violations = scan_result.value.get(c.Infra.ReportKeys.VIOLATIONS, [])
            if not isinstance(raw_violations, list):
                continue
            parsed_violations: list[m.Infra.Refactor.LooseClassViolation] = [
                v
                for v in raw_violations
                if isinstance(v, m.Infra.Refactor.LooseClassViolation)
            ]
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
            confidence_counts=dict(confidence_counts),
            violations=violations,
            per_file_counts=dict(per_file_counts),
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
    def _load_mapping_index(
        cls,
    ) -> r[dict[tuple[str, str], m.Infra.Refactor.ClassNestingMapping]]:
        mapping_path = (
            Path(__file__).resolve().parent / c.Infra.Refactor.MAPPINGS_RELATIVE_PATH
        )
        try:
            typed_doc = u.Infra.Yaml.safe_load_yaml(mapping_path)
        except (OSError, TypeError) as exc:
            return r[dict[tuple[str, str], m.Infra.Refactor.ClassNestingMapping]].fail(
                str(exc),
            )
        raw_nesting = typed_doc.get(c.Infra.ReportKeys.CLASS_NESTING)
        if not isinstance(raw_nesting, list):
            return r[dict[tuple[str, str], m.Infra.Refactor.ClassNestingMapping]].ok({})
        try:
            entries = TypeAdapter(
                list[m.Infra.Refactor.ClassNestingMapping],
            ).validate_python(raw_nesting)
        except ValidationError as exc:
            return r[dict[tuple[str, str], m.Infra.Refactor.ClassNestingMapping]].fail(
                str(exc),
            )
        index: dict[tuple[str, str], m.Infra.Refactor.ClassNestingMapping] = {}
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
        return r[dict[tuple[str, str], m.Infra.Refactor.ClassNestingMapping]].ok(index)

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
        cls, files: list[Path],
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
                file_path=file_path, content=content,
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
                file=file_name, total=total, counts=counts,
            )
            for file_name, total, counts in ranked_files[:25]
        ]
        helper_report = m.Infra.Refactor.HelperClassificationReport(
            totals=dict(helper_totals),
            suggestions=helper_suggestions,
            manual_review=helper_manual_review,
        )
        return m.Infra.Refactor.ViolationAnalysisReport(
            totals=dict(totals),
            files=per_file,
            top_files=hottest_files,
            files_scanned=len(files),
            helper_classification=helper_report,
            class_nesting=class_nesting,
        )

    @classmethod
    def _analyze_file_helpers(
        cls, *, file_path: Path, content: str,
    ) -> m.Infra.Refactor.HelperFileAnalysis:
        suggestions: list[m.Infra.Refactor.HelperClassification] = []
        totals: Counter[str] = Counter()
        manual_review: list[m.Infra.Refactor.HelperClassification] = []
        try:
            module = cst.parse_module(content)
        except cst.ParserSyntaxError:
            return m.Infra.Refactor.HelperFileAnalysis(
                suggestions=suggestions,
                totals=dict(totals),
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
            suggestions=suggestions, totals=dict(totals), manual_review=manual_review,
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
            decorator_root = u.Infra.Refactor.root_name(decorator.decorator)
            if not decorator_root:
                continue
            imported = local_to_import.get(decorator_root)
            if imported is not None:
                decorator_dependencies.add(imported)
        dependencies.update(decorator_dependencies)
        matched_categories = cls._match_categories(
            dependencies=dependencies, has_decorators=bool(function.decorators),
        )
        category, manual, reason = cls._resolve_category(
            dependencies=dependencies, matched_categories=matched_categories,
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
        cls, *, dependencies: set[str], has_decorators: bool,
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
        cls, *, dependencies: set[str], matched_categories: set[str],
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
