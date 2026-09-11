"""Violation analysis and helper classification for refactor reports.

Uses rope-based analysis and regex source scanning instead of CST visitors.
"""

from __future__ import annotations

from collections import Counter
from operator import itemgetter
from typing import TYPE_CHECKING

from flext_infra import c, m, u
from flext_infra.refactor._violation_helper_classifier import (
    FlextInfraRefactorViolationHelperClassifierMixin,
)
from flext_infra.refactor.class_nesting_analyzer import (
    FlextInfraRefactorClassNestingAnalyzer,
)
from flext_infra.transformers.violation_census_visitor import (
    FlextInfraViolationCensusVisitor,
)

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path

    from flext_infra import t


class FlextInfraRefactorViolationAnalyzer(
    FlextInfraRefactorViolationHelperClassifierMixin
):
    """Analyzer for refactor violation metrics across source files."""

    @classmethod
    def analyze_files(
        cls, files: t.SequenceOf[Path]
    ) -> m.Infra.ViolationAnalysisReport:
        """Analyze files and return aggregated violation and helper metrics."""
        totals: Counter[str] = Counter()
        per_file: MutableMapping[str, t.MutableIntMapping] = {}
        helper_suggestions: t.MutableSequenceOf[m.Infra.HelperClassification] = []
        helper_totals: Counter[str] = Counter()
        helper_manual_review: t.MutableSequenceOf[m.Infra.HelperClassification] = []
        for file_path in files:
            read = u.Cli.files_read_text(file_path)
            if read.failure:
                continue
            content = read.value
            helper_analysis = cls._analyze_file_helpers(
                file_path=file_path, content=content
            )
            helper_suggestions.extend(helper_analysis.suggestions)
            helper_totals.update(helper_analysis.totals)
            helper_manual_review.extend(helper_analysis.manual_review)
            file_counts: t.MutableIntMapping = {}
            violation_visitor = FlextInfraViolationCensusVisitor(file_path=file_path)
            violation_visitor.scan_source(content)
            for record in violation_visitor.records:
                kind = str(record.get("kind", ""))
                if kind:
                    totals[kind] += 1
                    file_counts[kind] = file_counts.get(kind, 0) + 1
            if file_counts:
                per_file[str(file_path)] = file_counts
        class_nesting = FlextInfraRefactorClassNestingAnalyzer.analyze_files(files)
        class_nesting_count = class_nesting.violations_count
        if class_nesting_count > 0:
            totals[c.Infra.RK_CLASS_NESTING] += class_nesting_count
        for raw_file, raw_count in class_nesting.per_file_counts.items():
            counts = per_file.setdefault(raw_file, {})
            counts[c.Infra.RK_CLASS_NESTING] = raw_count
        ranked_files: t.SequenceOf[t.Triple[str, int, t.IntMapping]] = [
            (file_name, sum(counts.values()), counts)
            for file_name, counts in per_file.items()
        ]
        ranked_sorted = sorted(ranked_files, key=itemgetter(1), reverse=True)
        hottest_files = [
            m.Infra.ViolationTopFileSection(
                file=file_name, total=total, counts={**counts}
            )
            for file_name, total, counts in ranked_sorted[:25]
        ]
        helper_report = m.Infra.HelperClassificationReport(
            totals=dict(helper_totals),
            suggestions=tuple(helper_suggestions),
            manual_review=tuple(helper_manual_review),
        )
        return m.Infra.ViolationAnalysisReport(
            totals=dict(totals),
            files={k: {**v} for k, v in per_file.items()},
            top_files=tuple(hottest_files),
            files_scanned=len(files),
            helper_classification=helper_report,
            class_nesting=class_nesting,
        )


__all__: list[str] = ["FlextInfraRefactorViolationAnalyzer"]
