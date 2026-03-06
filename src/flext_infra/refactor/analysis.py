"""Violation analysis helpers for flext_infra.refactor."""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from operator import itemgetter
from pathlib import Path
from typing import override

import libcst as cst
import yaml
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from flext_infra import c
from flext_infra._utilities.refactor import FlextInfraUtilitiesRefactor
from flext_infra.refactor.scanner import FlextInfraRefactorLooseClassScanner


def _dotted_name(expr: cst.BaseExpression) -> str:
    """Extract dotted name; delegates to ``u.Infra.Refactor``."""
    return FlextInfraUtilitiesRefactor.dotted_name(expr)


def _root_name(expr: cst.BaseExpression) -> str:
    """Extract root name; delegates to ``u.Infra.Refactor``."""
    return FlextInfraUtilitiesRefactor.root_name(expr)


class _HelperClassification(BaseModel):
    file: str
    function: str
    category: str
    target_namespace: str
    dependencies: list[str] = Field(default_factory=list)
    manual_review: bool = False
    review_reason: str = ""


class _HelperFileAnalysis(BaseModel):
    suggestions: Sequence[object] = Field(default_factory=list)
    totals: dict[str, int] = Field(default_factory=dict)
    manual_review: Sequence[object] = Field(default_factory=list)


class _ClassNestingMappingRecord(BaseModel):
    loose_name: str
    current_file: str
    target_namespace: str
    confidence: str
    rewrite_scope: str | None = None


class _ClassNestingMappingDocument(BaseModel):
    class_nesting: Sequence[object] = Field(default_factory=list)


class _LooseClassViolationModel(BaseModel):
    file: str
    line: int = 1
    class_name: str
    confidence: str = "low"
    expected_prefix: str = ""


class _ClassNestingMappingEntryModel(BaseModel):
    target_namespace: str
    confidence: str
    rewrite_scope: str


class _ClassNestingViolationModel(BaseModel):
    file: str
    line: int
    class_name: str
    target_namespace: str = ""
    confidence: str = "low"
    rewrite_scope: str = "file"


class _ClassNestingReportModel(BaseModel):
    violations_count: int = 0
    confidence_counts: dict[str, int] = Field(default_factory=dict)
    violations: Sequence[object] = Field(default_factory=list)
    per_file_counts: dict[str, int] = Field(default_factory=dict)


class _HelperClassificationReportModel(BaseModel):
    totals: dict[str, int] = Field(default_factory=dict)
    suggestions: Sequence[object] = Field(default_factory=list)
    manual_review: Sequence[object] = Field(default_factory=list)


class _ViolationTopFileModel(BaseModel):
    file: str
    total: int
    counts: dict[str, int]


class _ViolationAnalysisReportModel(BaseModel):
    totals: dict[str, int] = Field(default_factory=dict)
    files: dict[str, dict[str, int]] = Field(default_factory=dict)
    top_files: tuple[dict[str, object], ...] = Field(default_factory=tuple)
    files_scanned: int = 0
    helper_classification: _HelperClassificationReportModel
    class_nesting: _ClassNestingReportModel


ViolationAnalysisReport = _ViolationAnalysisReportModel


def _asname_to_local(asname: cst.AsName | None) -> str | None:
    """Extract local alias; delegates to ``u.Infra.Refactor``."""
    return FlextInfraUtilitiesRefactor.asname_to_local(asname)


class _ImportDependencyCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.local_to_import: dict[str, str] = {}

    @override
    def visit_Import(self, node: cst.Import) -> None:
        for raw_alias in node.names:
            imported = _dotted_name(raw_alias.name)
            if not imported:
                continue
            local_name = _asname_to_local(raw_alias.asname)
            if local_name is None:
                local_name = imported.split(".", maxsplit=1)[0]
            self.local_to_import[local_name] = imported

    @override
    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if isinstance(node.names, cst.ImportStar):
            return
        if node.module is None:
            return
        module_name = _dotted_name(node.module)
        if not module_name:
            return
        for raw_alias in node.names:
            if not isinstance(raw_alias.name, cst.Name):
                continue
            imported_name = raw_alias.name.value
            if imported_name == "*":
                continue
            local_name = imported_name
            local_name_from_alias = _asname_to_local(raw_alias.asname)
            if local_name_from_alias is not None:
                local_name = local_name_from_alias
            self.local_to_import[local_name] = f"{module_name}.{imported_name}"


class _FunctionDependencyCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    @override
    def visit_Name(self, node: cst.Name) -> None:
        self.names.add(node.value)


class FlextInfraRefactorClassNestingAnalyzer:
    """Analyze files for class nesting violations using YAML mapping rules."""

    @classmethod
    def analyze_files(cls, files: list[Path]) -> _ClassNestingReportModel:
        """Return aggregate and per-file class-nesting violation counts."""
        if not files:
            return _ClassNestingReportModel(
                violations_count=0,
                confidence_counts={},
                violations=[],
                per_file_counts={},
            )

        grouped_targets = cls._group_targets_by_project_root(files)
        if not grouped_targets:
            return _ClassNestingReportModel(
                violations_count=0,
                confidence_counts={},
                violations=[],
                per_file_counts={},
            )

        scanner = FlextInfraRefactorLooseClassScanner()
        mapping_index = cls._load_mapping_index()
        confidence_counts: Counter[str] = Counter()
        per_file_counts: Counter[str] = Counter()
        violations: list[_ClassNestingViolationModel] = []

        for project_root, target_files in grouped_targets.items():
            scan_result = scanner.scan(project_root)
            raw_violations = scan_result.get("violations", [])
            try:
                parsed_violations = TypeAdapter(
                    list[_LooseClassViolationModel]
                ).validate_python(raw_violations)
            except ValidationError:
                continue

            for parsed_violation in parsed_violations:
                normalized_file = cls._normalize_module_path(parsed_violation.file)
                if target_files and normalized_file not in target_files:
                    continue

                line = parsed_violation.line if parsed_violation.line > 0 else 1
                confidence = parsed_violation.confidence or "low"

                target_namespace = ""
                rewrite_scope = "file"
                mapped_entry = mapping_index.get((
                    normalized_file,
                    parsed_violation.class_name,
                ))
                if mapped_entry is not None:
                    target_namespace = mapped_entry.target_namespace
                    confidence = mapped_entry.confidence
                    rewrite_scope = mapped_entry.rewrite_scope
                elif parsed_violation.expected_prefix:
                    target_namespace = parsed_violation.expected_prefix

                violations.append(
                    _ClassNestingViolationModel(
                        file=normalized_file,
                        line=line,
                        class_name=parsed_violation.class_name,
                        target_namespace=target_namespace,
                        confidence=confidence,
                        rewrite_scope=rewrite_scope,
                    )
                )
                confidence_counts[confidence] += 1
                per_file_counts[normalized_file] += 1

        return _ClassNestingReportModel(
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
    ) -> Mapping[tuple[str, str], _ClassNestingMappingEntryModel]:
        mapping_path = (
            Path(__file__).resolve().parent / c.Infra.Refactor.MAPPINGS_RELATIVE_PATH
        )
        try:
            raw_content = mapping_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            parsed = yaml.safe_load(raw_content)
        except (OSError, yaml.YAMLError):
            return {}

        try:
            doc = _ClassNestingMappingDocument.model_validate(parsed)
        except ValidationError:
            return {}

        index: dict[tuple[str, str], _ClassNestingMappingEntryModel] = {}
        for raw_entry in doc.class_nesting:
            try:
                entry = _ClassNestingMappingRecord.model_validate(raw_entry)
            except ValidationError:
                continue
            rewrite_scope = cls._normalize_rewrite_scope(entry.rewrite_scope)
            normalized_file = cls._normalize_module_path(entry.current_file)
            index[normalized_file, entry.loose_name] = _ClassNestingMappingEntryModel(
                target_namespace=entry.target_namespace,
                confidence=entry.confidence,
                rewrite_scope=rewrite_scope,
            )

        return index

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
    def _normalize_rewrite_scope(cls, raw_scope: object) -> str:
        if not isinstance(raw_scope, str):
            return "file"
        candidate = raw_scope.strip().lower()
        if candidate in {"file", c.Infra.Toml.PROJECT, "workspace"}:
            return candidate
        return "file"


class FlextInfraRefactorViolationAnalyzer:
    """Scan files and aggregate massive pattern violations."""

    @classmethod
    def analyze_files(cls, files: list[Path]) -> _ViolationAnalysisReportModel:
        """Return aggregate and per-file violation counts."""
        totals: Counter[str] = Counter()
        per_file: dict[str, dict[str, int]] = {}
        helper_suggestions: list[object] = []
        helper_totals: Counter[str] = Counter()
        helper_manual_review: list[object] = []

        for file_path in files:
            try:
                content = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            except (OSError, UnicodeDecodeError):
                continue

            helper_analysis = cls._analyze_file_helpers(
                file_path=file_path, content=content
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
            totals["class_nesting"] += class_nesting_count

        for raw_file, raw_count in class_nesting.per_file_counts.items():
            counts = per_file.setdefault(raw_file, {})
            counts["class_nesting"] = raw_count

        ranked_files: list[tuple[str, int, dict[str, int]]] = []
        for file_name, counts in per_file.items():
            ranked_files.append((file_name, sum(counts.values()), counts))
        ranked_files.sort(key=itemgetter(1), reverse=True)

        hottest_files = [
            _ViolationTopFileModel(
                file=file_name,
                total=total,
                counts=counts,
            )
            for file_name, total, counts in ranked_files[:25]
        ]

        helper_report = _HelperClassificationReportModel(
            totals=dict(helper_totals),
            suggestions=helper_suggestions,
            manual_review=helper_manual_review,
        )

        return _ViolationAnalysisReportModel(
            totals=dict(totals),
            files=per_file,
            top_files=tuple(item.model_dump() for item in hottest_files),
            files_scanned=len(files),
            helper_classification=helper_report,
            class_nesting=class_nesting,
        )

    @classmethod
    def _analyze_file_helpers(
        cls,
        *,
        file_path: Path,
        content: str,
    ) -> _HelperFileAnalysis:
        suggestions: list[_HelperClassification] = []
        totals: Counter[str] = Counter()
        manual_review: list[_HelperClassification] = []

        try:
            module = cst.parse_module(content)
        except cst.ParserSyntaxError:
            return _HelperFileAnalysis(
                suggestions=suggestions,
                totals=dict(totals),
                manual_review=manual_review,
            )

        import_collector = _ImportDependencyCollector()
        module.visit(import_collector)

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

        return _HelperFileAnalysis(
            suggestions=suggestions,
            totals=dict(totals),
            manual_review=manual_review,
        )

    @classmethod
    def _classify_helper_function(
        cls,
        *,
        file_path: Path,
        function: cst.FunctionDef,
        local_to_import: Mapping[str, str],
    ) -> _HelperClassification:
        dependency_collector = _FunctionDependencyCollector()
        function.visit(dependency_collector)

        dependencies: set[str] = set()
        for name in dependency_collector.names:
            imported = local_to_import.get(name)
            if imported is not None:
                dependencies.add(imported)

        decorator_dependencies: set[str] = set()
        for decorator in function.decorators:
            decorator_root = _root_name(decorator.decorator)
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

        return _HelperClassification(
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
            return (
                ordered[0],
                True,
                f"Cross-cutting concerns detected: {', '.join(sorted(matched_categories))}",
            )

        if len(matched_categories) == 1:
            category = next(iter(matched_categories))
            return category, False, ""

        if cls._is_pure_utility_dependencies(dependencies):
            return "utility", False, ""

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
    "ViolationAnalysisReport",
]
