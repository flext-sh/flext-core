"""Loose class detection and scanning for flext-infra refactor."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import override

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_infra import FlextInfraCommandRunner, c, t
from flext_infra.models import FlextInfraModels as m


class TopLevelClassCollector(cst.CSTVisitor):
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

    def scan(self, project_root: Path) -> t.ConfigurationMapping:
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
            "rule": c.Infra.ReportKeys.CLASS_NESTING,
            "files_scanned": len(python_files),
            "classes_scanned": classes_scanned,
            c.Infra.ReportKeys.VIOLATIONS_COUNT: len(violations),
            "confidence_counts": dict(counters),
            "required_targets": targets_found,
            c.Infra.ReportKeys.VIOLATIONS: [item.model_dump() for item in violations],
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
            rule=c.Infra.ReportKeys.CLASS_NESTING,
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
        return c.Infra.Severity.LOW

    def _discover_python_files(self, project_root: Path) -> list[Path]:
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src_dir.is_dir():
            return []

        files: list[Path] = []
        for file_path in sorted(src_dir.rglob(c.Infra.Extensions.PYTHON_GLOB)):
            if (
                file_path.name.startswith("__")
                and file_path.name != c.Infra.Files.INIT_PY
            ):
                continue
            if "__pycache__" in file_path.parts:
                continue
            files.append(file_path)
        return files

    def _expected_prefix_for_module(self, rel_path: Path) -> str:
        parts = rel_path.parts
        if len(parts) < c.Infra.Refactor.MIN_PATH_DEPTH:
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
        src_dir = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        try:
            return file_path.relative_to(src_dir)
        except ValueError:
            return None

    def _scan_file_with_libcst(
        self,
        file_path: Path,
    ) -> list[m.Infra.Refactor.ClassOccurrence]:
        try:
            source = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            module = cst.parse_module(source)
            collector = TopLevelClassCollector()
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
            c.Infra.Toml.PYTHON,
            "--json",
            str(project_root / c.Infra.Paths.DEFAULT_SRC_DIR),
        ]
        runner = FlextInfraCommandRunner()
        result = runner.capture(cmd)
        if result.is_failure:
            return {}

        payload = result.value.strip()
        if not payload:
            return {}

        try:
            entries = TypeAdapter(list[m.Infra.Refactor.AstGrepEntry]).validate_json(
                payload
            )
        except ValidationError:
            return {}

        index: dict[Path, dict[str, int]] = {}
        for entry in entries:
            name_entry = entry.meta_variables.single.get("NAME")
            if name_entry is None:
                continue
            class_name = name_entry.text

            line = 1
            if entry.range is not None and entry.range.start is not None:
                line_raw = entry.range.start.line
                if line_raw is not None and line_raw > 0:
                    line = line_raw

            file_path = Path(entry.file)
            if not file_path.is_absolute():
                file_path = (project_root / file_path).resolve()
            names = index.setdefault(file_path, {})
            if class_name not in names:
                names[class_name] = line

        return index


__all__ = [
    "FlextInfraRefactorLooseClassScanner",
]
