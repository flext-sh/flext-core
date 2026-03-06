"""Violation analysis helpers for flext_infra.refactor."""

from __future__ import annotations

import re
import sys
from collections import Counter
from collections.abc import Mapping
from operator import itemgetter
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, TypedDict, cast, override

import libcst as cst
import yaml

from flext_infra.refactor.scanner import FlextInfraRefactorLooseClassScanner


def _dotted_name(expr: cst.BaseExpression) -> str:
    if isinstance(expr, cst.Name):
        return expr.value
    if isinstance(expr, cst.Attribute):
        root = _dotted_name(expr.value)
        if not root:
            return ""
        return f"{root}.{expr.attr.value}"
    return ""


def _root_name(expr: cst.BaseExpression) -> str:
    if isinstance(expr, cst.Name):
        return expr.value
    if isinstance(expr, cst.Attribute):
        return _root_name(expr.value)
    if isinstance(expr, cst.Call):
        return _root_name(expr.func)
    return ""


class _HelperFileAnalysis(TypedDict):
    suggestions: list[dict[str, object]]
    totals: Counter[str]
    manual_review: list[dict[str, object]]


class _ClassNestingMappingEntry(TypedDict):
    target_namespace: str
    confidence: str
    rewrite_scope: str


class _ClassNestingViolation(TypedDict):
    file: str
    line: int
    class_name: str
    target_namespace: str
    confidence: str
    rewrite_scope: str


def _asname_to_local(asname: cst.AsName | None) -> str | None:
    if asname is None:
        return None
    if isinstance(asname.name, cst.Name):
        return asname.name.value
    return None


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
    _MAPPINGS_RELATIVE_PATH: ClassVar[Path] = (
        Path("rules") / "class-nesting-mappings.yml"
    )

    @classmethod
    def analyze_files(cls, files: list[Path]) -> Mapping[str, object]:
        if not files:
            return {
                "violations_count": 0,
                "confidence_counts": {},
                "violations": [],
                "per_file_counts": {},
            }

        grouped_targets = cls._group_targets_by_project_root(files)
        if not grouped_targets:
            return {
                "violations_count": 0,
                "confidence_counts": {},
                "violations": [],
                "per_file_counts": {},
            }

        scanner = FlextInfraRefactorLooseClassScanner()
        mapping_index = cls._load_mapping_index()
        confidence_counts: Counter[str] = Counter()
        per_file_counts: Counter[str] = Counter()
        violations: list[_ClassNestingViolation] = []

        for project_root, target_files in grouped_targets.items():
            scan_result = scanner.scan(project_root)
            raw_violations = scan_result.get("violations", [])
            if not isinstance(raw_violations, list):
                continue

            for raw_violation in cast("list[object]", raw_violations):
                if not isinstance(raw_violation, dict):
                    continue
                typed_violation = cast("dict[object, object]", raw_violation)

                raw_file = typed_violation.get("file")
                raw_line = typed_violation.get("line")
                raw_class_name = typed_violation.get("class_name")
                raw_confidence = typed_violation.get("confidence")
                raw_expected_prefix = typed_violation.get("expected_prefix")

                if not isinstance(raw_file, str):
                    continue
                if not isinstance(raw_class_name, str):
                    continue

                normalized_file = cls._normalize_module_path(raw_file)
                if target_files and normalized_file not in target_files:
                    continue

                line = 1
                if isinstance(raw_line, int) and raw_line > 0:
                    line = raw_line

                confidence = "low"
                if isinstance(raw_confidence, str) and raw_confidence:
                    confidence = raw_confidence

                target_namespace = ""
                rewrite_scope = "file"
                mapped_entry = mapping_index.get((normalized_file, raw_class_name))
                if mapped_entry is not None:
                    target_namespace = mapped_entry["target_namespace"]
                    confidence = mapped_entry["confidence"]
                    rewrite_scope = mapped_entry["rewrite_scope"]
                elif isinstance(raw_expected_prefix, str):
                    target_namespace = raw_expected_prefix

                violations.append({
                    "file": normalized_file,
                    "line": line,
                    "class_name": raw_class_name,
                    "target_namespace": target_namespace,
                    "confidence": confidence,
                    "rewrite_scope": rewrite_scope,
                })
                confidence_counts[confidence] += 1
                per_file_counts[normalized_file] += 1

        return {
            "violations_count": len(violations),
            "confidence_counts": dict(confidence_counts),
            "violations": violations,
            "per_file_counts": dict(per_file_counts),
        }

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
            src_dir = parent / "src"
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
        src_dir = (project_root / "src").resolve()
        resolved = file_path.resolve()
        try:
            relative = resolved.relative_to(src_dir)
        except ValueError:
            return None
        return relative.as_posix()

    @classmethod
    def _load_mapping_index(cls) -> Mapping[tuple[str, str], _ClassNestingMappingEntry]:
        mapping_path = Path(__file__).resolve().parent / cls._MAPPINGS_RELATIVE_PATH
        try:
            raw_content = mapping_path.read_text(encoding="utf-8")
            parsed = cast("object", yaml.safe_load(raw_content))
        except (OSError, yaml.YAMLError):
            return {}

        if not isinstance(parsed, dict):
            return {}

        parsed_dict = cast("dict[object, object]", parsed)
        raw_entries = parsed_dict.get("class_nesting", [])
        if not isinstance(raw_entries, list):
            return {}

        index: dict[tuple[str, str], _ClassNestingMappingEntry] = {}
        for raw_entry in cast("list[object]", raw_entries):
            if not isinstance(raw_entry, dict):
                continue
            typed_entry = cast("dict[object, object]", raw_entry)
            loose_name = typed_entry.get("loose_name")
            current_file = typed_entry.get("current_file")
            target_namespace = typed_entry.get("target_namespace")
            confidence = typed_entry.get("confidence")
            rewrite_scope_raw = typed_entry.get("rewrite_scope")
            if not isinstance(loose_name, str):
                continue
            if not isinstance(current_file, str):
                continue
            if not isinstance(target_namespace, str):
                continue
            if not isinstance(confidence, str):
                continue
            rewrite_scope = cls._normalize_rewrite_scope(rewrite_scope_raw)

            normalized_file = cls._normalize_module_path(current_file)
            index[normalized_file, loose_name] = {
                "target_namespace": target_namespace,
                "confidence": confidence,
                "rewrite_scope": rewrite_scope,
            }

        return index

    @classmethod
    def _normalize_module_path(cls, raw_path: str) -> str:
        normalized = raw_path.replace("\\", "/")
        path = Path(normalized)
        parts = path.parts
        if "src" in parts:
            src_index = parts.index("src")
            suffix = parts[src_index + 1 :]
            if suffix:
                return Path(*suffix).as_posix()
        return path.as_posix().lstrip("./")

    @classmethod
    def _normalize_rewrite_scope(cls, raw_scope: object) -> str:
        if not isinstance(raw_scope, str):
            return "file"
        candidate = raw_scope.strip().lower()
        if candidate in {"file", "project", "workspace"}:
            return candidate
        return "file"


class FlextInfraRefactorViolationAnalyzer:
    """Scan files and aggregate massive pattern violations."""

    _PATTERNS: ClassVar[Mapping[str, re.Pattern[str]]] = MappingProxyType({
        "container_invariance": re.compile(
            r"\bdict\s*\[\s*str\s*,\s*t\.(?:Container|ContainerValue)\s*\]"
        ),
        "redundant_cast": re.compile(r"\bcast\s*\(\s*[\"'][^\"']+[\"']\s*,"),
        "direct_submodule_import": re.compile(
            r"\bfrom\s+flext_core\.[\w\.]+\s+import\b"
        ),
        "legacy_typing_mapping": re.compile(
            r"\bfrom\s+typing\s+import\s+.*\bMapping\b"
        ),
        "runtime_alias_violation": re.compile(
            r"\bfrom\s+flext_core\s+import\s+(?!.*\b(?:c|m|r|t|u|p|d|e|h|s|x)\b).*"
        ),
    })
    _MODEL_TOKENS: ClassVar[tuple[str, ...]] = (
        "model",
        "schema",
        "entity",
        "pydantic",
        "dataclass",
    )
    _DECORATOR_TOKENS: ClassVar[tuple[str, ...]] = (
        "decorator",
        "inject",
        "provide",
    )
    _DISPATCHER_TOKENS: ClassVar[tuple[str, ...]] = (
        "dispatcher",
        "dispatch",
        "command",
        "query",
        "event",
    )
    _NAMESPACE_PREFIXES: ClassVar[Mapping[str, str]] = MappingProxyType({
        "utility": "FlextUtilities",
        "models": "FlextModels",
        "decorators": "FlextDecorators",
        "dispatcher": "FlextDispatcher",
    })
    _CLASSIFICATION_PRIORITY: ClassVar[tuple[str, ...]] = (
        "dispatcher",
        "decorators",
        "models",
        "utility",
    )

    @classmethod
    def analyze_files(cls, files: list[Path]) -> Mapping[str, object]:
        """Return aggregate and per-file violation counts."""
        totals: Counter[str] = Counter()
        per_file: dict[str, dict[str, int]] = {}
        helper_suggestions: list[dict[str, object]] = []
        helper_totals: Counter[str] = Counter()
        helper_manual_review: list[dict[str, object]] = []

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue

            helper_analysis = cls._analyze_file_helpers(
                file_path=file_path, content=content
            )
            helper_suggestions.extend(helper_analysis["suggestions"])
            helper_totals.update(helper_analysis["totals"])
            helper_manual_review.extend(helper_analysis["manual_review"])

            file_counts: dict[str, int] = {}
            for name, pattern in cls._PATTERNS.items():
                count = len(pattern.findall(content))
                if count <= 0:
                    continue
                totals[name] += count
                file_counts[name] = count

            if file_counts:
                per_file[str(file_path)] = file_counts

        class_nesting = FlextInfraRefactorClassNestingAnalyzer.analyze_files(files)
        class_nesting_count_obj = class_nesting.get("violations_count", 0)
        class_nesting_count = (
            class_nesting_count_obj if isinstance(class_nesting_count_obj, int) else 0
        )
        if class_nesting_count > 0:
            totals["class_nesting"] += class_nesting_count

        class_nesting_per_file_obj = class_nesting.get("per_file_counts", {})
        if isinstance(class_nesting_per_file_obj, dict):
            typed_class_nesting_per_file = cast(
                "dict[object, object]", class_nesting_per_file_obj
            )
            for raw_file, raw_count in typed_class_nesting_per_file.items():
                if not isinstance(raw_file, str):
                    continue
                if not isinstance(raw_count, int):
                    continue
                counts = per_file.setdefault(raw_file, {})
                counts["class_nesting"] = raw_count

        ranked_files: list[tuple[str, int, dict[str, int]]] = []
        for file_name, counts in per_file.items():
            ranked_files.append((file_name, sum(counts.values()), counts))
        ranked_files.sort(key=itemgetter(1), reverse=True)

        hottest_files = [
            {
                "file": file_name,
                "total": total,
                "counts": counts,
            }
            for file_name, total, counts in ranked_files[:25]
        ]

        return {
            "totals": dict(totals),
            "files": per_file,
            "top_files": hottest_files,
            "files_scanned": len(files),
            "helper_classification": {
                "totals": dict(helper_totals),
                "suggestions": helper_suggestions,
                "manual_review": helper_manual_review,
            },
            "class_nesting": class_nesting,
        }

    @classmethod
    def _analyze_file_helpers(
        cls,
        *,
        file_path: Path,
        content: str,
    ) -> _HelperFileAnalysis:
        suggestions: list[dict[str, object]] = []
        totals: Counter[str] = Counter()
        manual_review: list[dict[str, object]] = []

        try:
            module = cst.parse_module(content)
        except cst.ParserSyntaxError:
            return {
                "suggestions": suggestions,
                "totals": totals,
                "manual_review": manual_review,
            }

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
            category = cast("str", classification["category"])
            totals[category] += 1
            if bool(classification["manual_review"]):
                manual_review.append(classification)

        return {
            "suggestions": suggestions,
            "totals": totals,
            "manual_review": manual_review,
        }

    @classmethod
    def _classify_helper_function(
        cls,
        *,
        file_path: Path,
        function: cst.FunctionDef,
        local_to_import: Mapping[str, str],
    ) -> dict[str, object]:
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
        namespace_root = cls._NAMESPACE_PREFIXES[category]

        return {
            "file": str(file_path),
            "function": function.name.value,
            "category": category,
            "target_namespace": f"{namespace_root}.{function.name.value}",
            "dependencies": sorted(dependencies),
            "manual_review": manual,
            "review_reason": reason,
        }

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
            if any(token in lowered for token in cls._MODEL_TOKENS):
                matched.add("models")
            if any(token in lowered for token in cls._DECORATOR_TOKENS):
                matched.add("decorators")
            if any(token in lowered for token in cls._DISPATCHER_TOKENS):
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
                for category in cls._CLASSIFICATION_PRIORITY
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
]
