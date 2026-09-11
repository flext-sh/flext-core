"""Class-nesting analysis for refactor violation reporting."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, m, t, u
from flext_infra.refactor.scanner import FlextInfraRefactorLooseClassScanner

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from flext_infra import p


class FlextInfraRefactorClassNestingAnalyzer:
    """Detect class nesting violations and report FLEXT hierarchy issues."""

    @classmethod
    def analyze_files(cls, files: t.SequenceOf[Path]) -> m.Infra.ClassNestingReport:
        """Analyze files and return aggregated class-nesting violations."""
        if not files:
            return m.Infra.ClassNestingReport(
                violations_count=0,
                confidence_counts={},
                violations=(),
                per_file_counts={},
            )
        grouped_targets = cls._group_targets_by_project_root(files)
        if not grouped_targets:
            return m.Infra.ClassNestingReport(
                violations_count=0,
                confidence_counts={},
                violations=(),
                per_file_counts={},
            )
        scanner = FlextInfraRefactorLooseClassScanner()
        mapping_result = cls._load_mapping_index()
        mapping_index: t.MappingKV[t.StrPair, m.Infra.ClassNestingMapping] = (
            mapping_result.unwrap() if mapping_result.success else {}
        )
        confidence_counts: Counter[str] = Counter()
        per_file_counts: Counter[str] = Counter()
        violations: t.MutableSequenceOf[m.Infra.ClassNestingViolation] = []
        for project_root, target_files in grouped_targets.items():
            scan_result = scanner.scan(project_root)
            if scan_result.failure:
                continue
            try:
                typed_items = t.Infra.CONTAINER_DICT_SEQ_ADAPTER.validate_python(
                    scan_result.value.get(c.Infra.RK_VIOLATIONS, [])
                )
                parsed_violations: t.SequenceOf[m.Infra.LooseClassViolation] = [
                    m.Infra.LooseClassViolation.model_validate(item)
                    for item in typed_items
                ]
            except c.ValidationError:
                continue
            for parsed_violation in parsed_violations:
                normalized_file = u.Infra.normalize_module_path(parsed_violation.file)
                if target_files and normalized_file not in target_files:
                    continue
                line = parsed_violation.line if parsed_violation.line > 0 else 1
                confidence = parsed_violation.confidence or c.Infra.SeverityLevel.LOW
                target_namespace = ""
                rewrite_scope = c.Infra.RK_FILE
                mapping_key: tuple[str, str] = (
                    normalized_file,
                    parsed_violation.class_name,
                )
                mapped_entry = mapping_index.get(mapping_key)
                if mapped_entry is not None:
                    target_namespace = mapped_entry.target_namespace
                    confidence = mapped_entry.confidence
                    rewrite_scope = cls._normalize_rewrite_scope(
                        mapped_entry.rewrite_scope
                    )
                elif parsed_violation.expected_prefix:
                    target_namespace = parsed_violation.expected_prefix
                violations.append(
                    m.Infra.ClassNestingViolation(
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
        return m.Infra.ClassNestingReport(
            violations_count=len(violations),
            confidence_counts=dict(confidence_counts),
            violations=tuple(violations),
            per_file_counts=dict(per_file_counts),
        )

    @classmethod
    def _group_targets_by_project_root(
        cls, files: t.SequenceOf[Path]
    ) -> t.MappingKV[Path, t.Infra.StrSet]:
        """Group targets by project root."""
        grouped: MutableMapping[Path, t.Infra.StrSet] = {}
        for file_path in files:
            project_root = u.Infra.resolve_project_root(file_path)
            if project_root is None:
                continue
            module_path = cls._module_path_for_file(file_path, project_root)
            if module_path is None:
                continue
            grouped.setdefault(project_root, set()).add(module_path)
        return grouped

    @classmethod
    def _module_path_for_file(cls, file_path: Path, project_root: Path) -> str | None:
        """Return the module path for a file."""
        src_dir = (project_root / c.Infra.DEFAULT_SRC_DIR).resolve()
        resolved = file_path.resolve()
        try:
            relative = resolved.relative_to(src_dir)
        except ValueError:
            return None
        return relative.as_posix()

    @classmethod
    def _load_mapping_index(
        cls,
    ) -> p.Result[t.MappingKV[t.StrPair, m.Infra.ClassNestingMapping]]:
        """Load mapping index."""
        mapping_path = Path(__file__).resolve().parent / c.Infra.MAPPINGS_RELATIVE_PATH
        try:
            typed_doc = u.Cli.yaml_load_mapping(mapping_path)
        except c.EXC_OS_TYPE as exc:
            return r[t.MappingKV[t.Pair[str, str], m.Infra.ClassNestingMapping]].fail(
                str(exc)
            )
        raw_nesting = typed_doc.get(c.Infra.RK_CLASS_NESTING)
        if not isinstance(raw_nesting, list):
            return r[t.MappingKV[t.Pair[str, str], m.Infra.ClassNestingMapping]].ok({})
        try:
            typed_items = t.Infra.CONTAINER_DICT_SEQ_ADAPTER.validate_python(
                raw_nesting
            )
            entries: t.SequenceOf[m.Infra.ClassNestingMapping] = [
                m.Infra.ClassNestingMapping.model_validate(item) for item in typed_items
            ]
        except c.ValidationError as exc:
            return r[t.MappingKV[t.Pair[str, str], m.Infra.ClassNestingMapping]].fail(
                str(exc)
            )
        index: MutableMapping[t.StrPair, m.Infra.ClassNestingMapping] = {}
        for entry in entries:
            scope = cls._normalize_rewrite_scope(entry.rewrite_scope)
            norm = u.Infra.normalize_module_path(entry.current_file)
            index[norm, entry.loose_name] = m.Infra.ClassNestingMapping(
                loose_name=entry.loose_name,
                current_file=entry.current_file,
                target_namespace=entry.target_namespace,
                confidence=entry.confidence,
                rewrite_scope=scope,
                target_name=entry.target_name,
                reason=entry.reason,
            )
        return r[t.MappingKV[t.Pair[str, str], m.Infra.ClassNestingMapping]].ok(index)

    @classmethod
    def _normalize_rewrite_scope(cls, raw_scope: str | None) -> str:
        """Normalize rewrite scope."""
        default_scope: str = c.Infra.RK_FILE
        if not isinstance(raw_scope, str):
            return default_scope
        candidate: str = u.norm_str(raw_scope, case="lower")
        if candidate in {c.Infra.RK_FILE, c.Infra.PROJECT, c.Infra.RK_WORKSPACE}:
            return candidate
        return default_scope


__all__: list[str] = ["FlextInfraRefactorClassNestingAnalyzer"]
