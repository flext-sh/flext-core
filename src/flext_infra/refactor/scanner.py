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

type RConfigMapping = r[t.Infra.ContainerDict]
type RListPath = r[list[Path]]
type RPath = r[Path]
type RListClassOccurrence = r[list[m.Infra.Refactor.ClassOccurrence]]
type RDictPathGrep = r[dict[Path, dict[str, int]]]


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
            ),
        )
        self._depth += 1

    @override
    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        _ = original_node
        self._depth -= 1


class FlextInfraRefactorLooseClassScanner:
    """Scan a project tree and report top-level classes lacking namespace prefixes."""

    def scan(self, project_root: Path) -> RConfigMapping:
        """Scan *project_root*/src and return a violation report dict."""
        files_result = self._discover_python_files(project_root)
        if files_result.is_failure:
            out: RConfigMapping = r[t.Infra.ContainerDict].fail(
                files_result.error or "discovery failed",
            )
            return out
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
        violations_infra: list[t.Infra.InfraValue] = [
            v.model_dump() for v in violations
        ]
        confidence_counts: dict[str, t.Infra.InfraValue] = dict(counters)
        required_targets_infra: dict[str, t.Infra.InfraValue] = dict(targets_found)
        result_dict: t.Infra.ContainerDict = {
            "rule": c.Infra.ReportKeys.CLASS_NESTING,
            "files_scanned": len(discovered_files),
            "classes_scanned": classes_scanned,
            c.Infra.ReportKeys.VIOLATIONS_COUNT: len(violations),
            "confidence_counts": confidence_counts,
            "required_targets": required_targets_infra,
            c.Infra.ReportKeys.VIOLATIONS: violations_infra,
        }
        out2: RConfigMapping = r[t.Infra.ContainerDict].ok(result_dict)
        return out2

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
            reason="top_level_class_in_private_directory"
            if self._has_private_directory(rel_path)
            else "top_level_class_without_namespace_prefix",
            confidence=confidence,
            score=round(score, 2),
        )

    def _confidence_from_location(self, rel_path: Path) -> str:
        parts = rel_path.parent.parts[1:]
        if any(p.startswith("_") for p in parts):
            return "high"
        return "medium" if parts else c.Infra.Severity.LOW

    def _discover_python_files(self, project_root: Path) -> RListPath:
        src = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        if not src.is_dir():
            out: RListPath = r[list[Path]].fail(f"src not found: {src}")
            return out
        file_list: list[Path] = [
            fp
            for fp in u.Infra.iter_directory_python_files(src)
            if not (fp.name.startswith("__") and fp.name != c.Infra.Files.INIT_PY)
        ]
        out2: RListPath = r[list[Path]].ok(file_list)
        return out2

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

    def _relative_module_path(self, project_root: Path, file_path: Path) -> RPath:
        src = project_root / c.Infra.Paths.DEFAULT_SRC_DIR
        try:
            rel: Path = file_path.relative_to(src)
            out: RPath = r[Path].ok(rel)
            return out
        except ValueError as exc:
            out2: RPath = r[Path].fail(str(exc))
            return out2

    def _scan_file_with_libcst(
        self,
        file_path: Path,
    ) -> RListClassOccurrence:
        tree = u.Infra.parse_module_cst(file_path)
        if tree is None:
            out: RListClassOccurrence = r[list[m.Infra.Refactor.ClassOccurrence]].fail(
                f"{file_path}: parse_failed"
            )
            return out
        col = TopLevelClassCollector()
        tree.visit(col)
        out2: RListClassOccurrence = r[list[m.Infra.Refactor.ClassOccurrence]].ok(
            col.classes,
        )
        return out2

    def _scan_with_ast_grep(self, project_root: Path) -> RDictPathGrep:
        cmd = [
            "sg",
            "--pattern",
            "class $NAME",
            "--lang",
            c.Infra.Toml.PYTHON,
            "--json",
            str(project_root / c.Infra.Paths.DEFAULT_SRC_DIR),
        ]
        capture = u.Infra.capture_output(cmd)
        if capture.is_failure:
            out: RDictPathGrep = r[dict[Path, dict[str, int]]].fail(
                capture.error or "ast-grep failed",
            )
            return out
        if not capture.value:
            out2: RDictPathGrep = r[dict[Path, dict[str, int]]].ok({})
            return out2
        try:
            json_raw: str | bytes | bytearray = capture.value
            entries = TypeAdapter(
                list[m.Infra.Refactor.AstGrepMatchEnvelope],
            ).validate_json(json_raw)
        except ValidationError as exc:
            out3: RDictPathGrep = r[dict[Path, dict[str, int]]].fail(str(exc))
            return out3
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
        out4: RDictPathGrep = r[dict[Path, dict[str, int]]].ok(idx)
        return out4


__all__ = ["FlextInfraRefactorLooseClassScanner"]
