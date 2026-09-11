"""Census dry-run preview validation of removal candidates — extracted concern."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import u

if TYPE_CHECKING:
    from flext_infra import m, p, t

_log = u.fetch_logger(__name__)


class FlextInfraRefactorCensusValidateMixin:
    """Filter removal candidates through dry-run gates, surfacing rejections.

    Parent of FlextInfraRefactorCensusCollectMixin (its ``_assemble_report``
    calls ``_validated_project_reports``); borrows root + dry-run flags + the
    raw-violation builder from the facade and sibling mixins via FLEXT.
    """

    if TYPE_CHECKING:
        dry_run: bool

        @property
        def fail_fast(self) -> bool: ...

        @property
        def root(self) -> Path: ...

        @property
        def dry_run_gate_names(self) -> t.StrSequence: ...
        @staticmethod
        def _raw_violation(
            *,
            project: str,
            object_name: str,
            object_kind: str,
            kind: str,
            file_path: Path,
            line: int,
            description: str,
            fixable: bool = False,
            fix_action: str = "",
        ) -> m.Infra.Census.Violation: ...

    def _validated_project_reports(
        self,
        rope: p.Infra.RopeWorkspaceDsl,
        project_reports: tuple[m.Infra.Census.ProjectReport, ...],
    ) -> tuple[m.Infra.Census.ProjectReport, ...]:
        """Keep only removal candidates that pass the configured dry-run gates.

        Gate rejections are surfaced as explicit ``preview_rejected``
        violations so the census still completes with actionable output
        instead of aborting on the first rejected candidate.
        """
        validated_reports: list[m.Infra.Census.ProjectReport] = []
        # Preview writes are restored before the next candidate, so one shared
        # source cache stays valid for the entire dry-run validation pass.
        source_cache: dict[Path, str] = {}
        for report in project_reports:
            if not report.removal_candidates:
                validated_reports.append(report)
                continue
            validated_candidates_list: list[m.Infra.Census.RemovalCandidate] = []
            validated_violations = list(report.violations)
            for candidate in report.removal_candidates:
                preview_result = u.Infra.preview_simple_removal_candidate(
                    rope,
                    self.root,
                    candidate,
                    gates=self.dry_run_gate_names,
                    source_cache=source_cache,
                )
                if preview_result.failure:
                    msg = preview_result.error or (
                        "simple removal preview failed for "
                        f"{candidate.file_path}:{candidate.line} {candidate.object_name}"
                    )
                    if self.fail_fast:
                        raise RuntimeError(msg)
                    _log.warning(
                        "census_preview_candidate_rejected",
                        candidate=candidate.file_path,
                        object_name=candidate.object_name,
                        error=msg,
                    )
                    validated_violations.append(
                        self._raw_violation(
                            project=report.project,
                            object_name=candidate.object_name,
                            object_kind=candidate.object_kind,
                            kind="preview_rejected",
                            file_path=Path(candidate.file_path),
                            line=candidate.line,
                            description=msg,
                        )
                    )
                    continue
                if preview_result.unwrap_or(False):
                    validated_candidates_list.append(candidate)
            validated_candidates = tuple(validated_candidates_list)
            validated_reports.append(
                report.model_copy(
                    update={
                        "violations": tuple(validated_violations),
                        "violations_total": len(validated_violations),
                        "removal_candidate_count": len(validated_candidates),
                        "removal_candidates": validated_candidates,
                    }
                )
            )
        return tuple(validated_reports)


__all__: list[str] = ["FlextInfraRefactorCensusValidateMixin"]
