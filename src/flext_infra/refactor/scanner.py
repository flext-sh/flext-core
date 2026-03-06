"""Loose class detection and scanning for flext-infra refactor."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast, override

import libcst as cst

from flext_infra import FlextInfraCommandRunner, c, m


class _TopLevelClassCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self._depth = 0
        self.classes: list[m.Infra.Refactor.ClassOccurrence] = []

    @override
    def visit_ClassDef(self, node: cst.ClassDef) -> None:
        is_top_level = self._depth == 0
        self.classes.append(
            m.Infra.Refactor.ClassOccurrence(
                name=node.name.value,
                line=0,
                is_top_level=is_top_level,
            )
        )
        self._depth += 1

    @override
    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        _ = original_node
        self._depth -= 1


class FlextInfraRefactorLooseClassScanner:
    """Scan a project tree and report top-level classes lacking namespace prefixes."""

    def scan(self, project_root: Path) -> Mapping[str, object]:
        """Scan *project_root*/src and return a violation report dict."""
        python_files = self._discover_python_files(project_root)
        ast_grep_index = self._scan_with_ast_grep(project_root)

        violations: list[m.Infra.Refactor.LooseClassViolation] = []
        targets_found = dict.fromkeys(c.Infra.Refactor.REQUIRED_CLASS_TARGETS, False)
        classes_scanned = 0

        for file_path in python_files:
            class_occurrences = self._scan_file_with_libcst(file_path)
            classes_scanned += len(class_occurrences)
            ast_grep_hits = ast_grep_index.get(file_path, {})

            rel_path = self._relative_module_path(project_root, file_path)
            if rel_path is None:
                continue

            for occurrence in class_occurrences:
                violation = self._build_violation(rel_path, occurrence, ast_grep_hits)
                if violation is None:
                    continue
                violations.append(violation)
                if violation.class_name in targets_found:
                    targets_found[violation.class_name] = True

        counters = Counter(item.confidence for item in violations)

        return {
            "rule": "class_nesting",
            "files_scanned": len(python_files),
            "classes_scanned": classes_scanned,
            "violations_count": len(violations),
            "confidence_counts": dict(counters),
            "required_targets": targets_found,
            "violations": [item.model_dump() for item in violations],
        }

    def _build_violation(
        self,
        rel_path: Path,
        occurrence: m.Infra.Refactor.ClassOccurrence,
        ast_grep_hits: Mapping[str, int],
    ) -> m.Infra.Refactor.LooseClassViolation | None:
        if not occurrence.is_top_level:
            return None

        expected_prefix = self._expected_prefix_for_module(rel_path)
        if expected_prefix and occurrence.name.startswith(expected_prefix):
            return None

        confidence = self._confidence_from_location(rel_path)
        score = c.Infra.Refactor.CONFIDENCE_TO_SCORE[confidence]
        line = occurrence.line
        if occurrence.name in ast_grep_hits:
            score = min(score + 0.02, 0.99)
            line = ast_grep_hits[occurrence.name]

        reason = (
            "top_level_class_in_private_directory"
            if self._has_private_directory(rel_path)
            else "top_level_class_without_namespace_prefix"
        )

        return m.Infra.Refactor.LooseClassViolation(
            file=rel_path.as_posix(),
            line=max(line, 1),
            class_name=occurrence.name,
            expected_prefix=expected_prefix,
            rule="class_nesting",
            reason=reason,
            confidence=confidence,
            score=round(score, 2),
        )

    def _confidence_from_location(self, rel_path: Path) -> str:
        parent_parts = rel_path.parent.parts[1:]
        if any(part.startswith("_") for part in parent_parts):
            return "high"
        if parent_parts:
            return "medium"
        return "low"

    def _discover_python_files(self, project_root: Path) -> list[Path]:
        src_dir = project_root / "src"
        if not src_dir.is_dir():
            return []

        files: list[Path] = []
        for file_path in sorted(src_dir.rglob("*.py")):
            if file_path.name.startswith("__") and file_path.name != "__init__.py":
                continue
            if "__pycache__" in file_path.parts:
                continue
            files.append(file_path)
        return files

    _MIN_PATH_DEPTH = 2

    def _expected_prefix_for_module(self, rel_path: Path) -> str:
        parts = rel_path.parts
        if len(parts) < self._MIN_PATH_DEPTH:
            return ""

        project_part = parts[0]
        project_prefix = self._pascal_case(project_part.split("_", maxsplit=1)[0])
        directories = "".join(self._pascal_case(part) for part in parts[1:-1])
        module = self._pascal_case(rel_path.stem)
        return f"{project_prefix}{directories}{module}"

    def _has_private_directory(self, rel_path: Path) -> bool:
        return any(part.startswith("_") for part in rel_path.parent.parts[1:])

    def _pascal_case(self, value: str) -> str:
        normalized = c.Infra.Refactor.CLASS_PATTERN.sub(" ", value.replace("_", " "))
        return "".join(word.capitalize() for word in normalized.split())

    def _relative_module_path(self, project_root: Path, file_path: Path) -> Path | None:
        src_dir = project_root / "src"
        try:
            return file_path.relative_to(src_dir)
        except ValueError:
            return None

    def _scan_file_with_libcst(
        self,
        file_path: Path,
    ) -> list[m.Infra.Refactor.ClassOccurrence]:
        try:
            source = file_path.read_text(encoding="utf-8")
            module = cst.parse_module(source)
            collector = _TopLevelClassCollector()
            module.visit(collector)
            return collector.classes
        except (OSError, UnicodeDecodeError, cst.ParserSyntaxError):
            return []

    def _scan_with_ast_grep(
        self, project_root: Path
    ) -> Mapping[Path, Mapping[str, int]]:
        cmd = [
            "sg",
            "--pattern",
            "class $NAME",
            "--lang",
            "python",
            "--json",
            str(project_root / "src"),
        ]
        runner = FlextInfraCommandRunner()
        result = runner.capture(cmd)
        if result.is_failure:
            return {}

        payload = result.value.strip()
        if not payload:
            return {}

        try:
            entries_raw = json.loads(payload)
        except json.JSONDecodeError:
            return {}

        if not isinstance(entries_raw, list):
            return {}

        index: dict[Path, dict[str, int]] = {}
        for item in cast("list[object]", entries_raw):
            if not isinstance(item, dict):
                continue
            entry = cast("dict[str, object]", item)
            file_raw = entry.get("file")
            meta_raw = entry.get("metaVariables")
            if not isinstance(file_raw, str) or not isinstance(meta_raw, dict):
                continue
            meta = cast("dict[str, object]", meta_raw)
            single_raw = meta.get("single")
            if not isinstance(single_raw, dict):
                continue
            single = cast("dict[str, object]", single_raw)
            name_raw = single.get("NAME")
            if not isinstance(name_raw, dict):
                continue
            name_entry = cast("dict[str, object]", name_raw)
            class_name = name_entry.get("text")
            if not isinstance(class_name, str):
                continue

            line = 1
            range_raw = entry.get("range")
            if isinstance(range_raw, dict):
                range_dict = cast("dict[str, object]", range_raw)
                start_raw = range_dict.get("start")
                if isinstance(start_raw, dict):
                    start_dict = cast("dict[str, object]", start_raw)
                    line_raw = start_dict.get("line")
                    if isinstance(line_raw, int) and line_raw > 0:
                        line = line_raw

            file_path = Path(file_raw)
            if not file_path.is_absolute():
                file_path = (project_root / file_path).resolve()
            names = index.setdefault(file_path, {})
            if class_name not in names:
                names[class_name] = line

        return index


__all__ = [
    "FlextInfraRefactorLooseClassScanner",
]
