"""Loose class detection and scanning for flext-infra refactor."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import override

import libcst as cst
from pydantic import TypeAdapter, ValidationError

from flext_core import r
from flext_infra import c, m, t, u


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

    def scan(self, project_root: Path) -> r[t.ConfigurationMapping]:
        """Scan *project_root*/src and return a violation report dict."""
        files_result = self._discover_python_files(project_root)
        if files_result.is_failure:
            return r[t.ConfigurationMapping].fail(
                files_result.error or "discovery failed"
            )

        discovered_files: list[Path] = files_result.value

        grep_result = self._scan_with_ast_grep(project_root)
        grep_index: dict[Path, dict[str, int]] = (
            grep_result.value if grep_result.is_success else {}
        )

        violations: list[m.Infra.Refactor.LooseClassViolation] = []
        targets_found = dict.fromkeys(c.Infra.Refactor.REQUIRED_CLASS_TARGETS, False)
        classes_scanned = 0

        for fp in discovered_files:
            parsed = self._scan_file_with_libcst(fp)
            if parsed.is_failure:
                continue
            occurrences: list[m.Infra.Refactor.ClassOccurrence] = parsed.value
            classes_scanned += len(occurrences)

            rel = self._relative_module_path(project_root, fp)
            if rel.is_failure:
                continue
            rel_path: Path = rel.value

            for occ in occurrences:
                viol = self._build_violation(rel_path, occ, grep_index.get(fp, {}))
                if viol is None:
                    continue
                violations.append(viol)
                if viol.class_name in targets_found:
                    targets_found[viol.class_name] = True

        counters = Counter(v.confidence for v in violations)
        return r[t.ConfigurationMapping].ok({
            "rule": c.Infra.ReportKeys.CLASS_NESTING,
            "files_scanned": len(discovered_files),
            "classes_scanned": classes_scanned,
            c.Infra.ReportKeys.VIOLATIONS_COUNT: len(violations),
            "confidence_counts": dict(counters),
            "required_targets": targets_found,
            c.Infra.ReportKeys.VIOLATIONS: violations,
        })

    # ── private helpers ─────────────────────────────────────────

    def _build_violation(
        self,
        rel_path: Path,
        occ: m.Infra.Refactor.ClassOccurrence,
        grep_hits: Mapping[str, int],
    ) -> m.Infra.Refactor.LooseClassViolation | None:
        if not occ.is_top_level:
            return None
        prefix = self._expected_prefix_for_module(rel_path)
        if prefix and occ.name.startswith(prefix):
            return None

        confidence = self._confidence_from_location(rel_path)
        score = c.Infra.Refactor.CONFIDENCE_TO_SCORE[confidence]
        line = occ.line
        if occ.name in grep_hits:
            score = min(score + 0.02, 0.99)
            line = grep_hits[occ.name]

        return m.Infra.Refactor.LooseClassViolation(
            file=rel_path.as_posix(),
            line=max(line, 1),
            class_name=occ.name,
            expected_prefix=prefix,
            rule=c.Infra.ReportKeys.CLASS_NESTING,
            reason=(
                "top_level_class_in_private_directory"
                if self._has_private_directory(rel_path)
                else "top_level_class_without_namespace_prefix"
            ),
            confidence=confidence,
            score=round(score, 2),
        )

    def _confidence_from_location(self, rel_path: Path) -> str:
        parts = rel_path.parent.parts[1:]
        if any(p.startswith("_") for p in parts):
            return "high"
        return "medium" if parts else c.Infra.Severity.LOW

    def _discover_python_files(self, project_root: Path) -> r[list[Path]]:
        src = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src.is_dir():
            return r[list[Path]].fail(f"src not found: {src}")
        return r[list[Path]].ok([
            fp
            for fp in sorted(src.rglob(c.Infra.Extensions.PYTHON_GLOB))
            if "__pycache__" not in fp.parts
            and not (fp.name.startswith("__") and fp.name != c.Infra.Files.INIT_PY)
        ])

    def _expected_prefix_for_module(self, rel_path: Path) -> str:
        parts = rel_path.parts
        if len(parts) < c.Infra.Refactor.MIN_PATH_DEPTH:
            return ""
        pc = self._pascal_case
        proj = pc(parts[0].split("_", maxsplit=1)[0])
        dirs = "".join(pc(p) for p in parts[1:-1])
        return f"{proj}{dirs}{pc(rel_path.stem)}"

    def _has_private_directory(self, rel_path: Path) -> bool:
        return any(p.startswith("_") for p in rel_path.parent.parts[1:])

    def _pascal_case(self, value: str) -> str:
        norm = c.Infra.Refactor.CLASS_PATTERN.sub(" ", value.replace("_", " "))
        return "".join(w.capitalize() for w in norm.split())

    def _relative_module_path(self, project_root: Path, file_path: Path) -> r[Path]:
        src = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        try:
            return r[Path].ok(file_path.relative_to(src))
        except ValueError as exc:
            return r[Path].fail(str(exc))

    def _scan_file_with_libcst(
        self, file_path: Path
    ) -> r[list[m.Infra.Refactor.ClassOccurrence]]:
        try:
            src = file_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            tree = cst.parse_module(src)
            col = TopLevelClassCollector()
            tree.visit(col)
            return r[list[m.Infra.Refactor.ClassOccurrence]].ok(col.classes)
        except (OSError, UnicodeDecodeError, cst.ParserSyntaxError) as exc:
            return r[list[m.Infra.Refactor.ClassOccurrence]].fail(f"{file_path}: {exc}")

    def _scan_with_ast_grep(self, project_root: Path) -> r[dict[Path, dict[str, int]]]:
        cmd = [
            "sg",
            "--pattern",
            "class $NAME",
            "--lang",
            c.Infra.Toml.PYTHON,
            "--json",
            str(project_root / c.Infra.Paths.DEFAULT_SRC_DIR),
        ]
        capture = u.Infra.Refactor.capture_output(cmd)
        if capture.is_failure:
            return r[dict[Path, dict[str, int]]].fail(
                capture.error or "ast-grep failed"
            )
        if not capture.value:
            return r[dict[Path, dict[str, int]]].ok({})

        try:
            entries = TypeAdapter(
                list[m.Infra.Refactor.AstGrepMatchEnvelope]
            ).validate_json(capture.value)
        except ValidationError as exc:
            return r[dict[Path, dict[str, int]]].fail(str(exc))

        idx: dict[Path, dict[str, int]] = {}
        for entry in entries:
            name = entry.symbol_name
            if name is None:
                continue
            line = 1
            if entry.start_line is not None and entry.start_line > 0:
                line = entry.start_line
            fp = Path(entry.file)
            if not fp.is_absolute():
                fp = (project_root / fp).resolve()
            idx.setdefault(fp, {}).setdefault(name, line)
        return r[dict[Path, dict[str, int]]].ok(idx)


__all__ = [
    "FlextInfraRefactorLooseClassScanner",
]
