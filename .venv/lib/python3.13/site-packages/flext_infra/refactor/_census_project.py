"""Census per-project report assembly + rope-stage failure handling."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from flext_infra import m, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t

_log = u.fetch_logger(__name__)


class FlextInfraRefactorCensusProjectMixin:
    """Build one project report (violations + removal candidates) for census.

    Composed into FlextInfraRefactorCensus via inheritance; borrows the
    rule-inclusion + object-classification helpers from sibling mixins via FLEXT.
    """

    if TYPE_CHECKING:

        @property
        def fail_fast(self) -> bool: ...

        @staticmethod
        def _include_rule(
            rule: str,
            *,
            rule_names: t.StrSequence | None,
            selected_rules: frozenset[str] | None = None,
        ) -> bool: ...
        @staticmethod
        def _is_unused(item: m.Infra.Census.Object) -> bool: ...
        @staticmethod
        def _object_key(item: m.Infra.Census.Object) -> str: ...
        @staticmethod
        def _violation(
            item: m.Infra.Census.Object,
            *,
            kind: str,
            description: str,
            fixable: bool = False,
            fix_action: str = "",
        ) -> m.Infra.Census.Violation: ...
        @classmethod
        def _removal_candidate(
            cls, item: m.Infra.Census.Object, *, include_unused: bool
        ) -> m.Infra.Census.RemovalCandidate | None: ...

    def _handle_rope_stage_failure(
        self, *, file_path: Path, stage: str, exc: BaseException
    ) -> None:
        """Handle rope stage failure."""
        error = f"{type(exc).__name__}: {exc}"
        _log.warning(
            "census_rope_stage_failed",
            stage=stage,
            file_path=str(file_path),
            error=error,
        )
        if self.fail_fast:
            msg = f"census rope {stage} failed for {file_path}: {error}"
            raise RuntimeError(msg) from exc

    def _project_report(
        self,
        project: str,
        *,
        objects: tuple[m.Infra.Census.Object, ...],
        seed_violations: tuple[m.Infra.Census.Violation, ...],
        fixes: tuple[m.Infra.Census.Fix, ...],
        duplicate_keys: frozenset[str],
        rule_names: t.StrSequence | None,
        selected_rules: frozenset[str] | None = None,
    ) -> m.Infra.Census.ProjectReport:
        """Project report."""
        violations = list(seed_violations)
        if selected_rules is None and rule_names:
            selected_rules = frozenset(rule_names)
        include_unused = self._include_rule(
            "unused", rule_names=rule_names, selected_rules=selected_rules
        )
        include_duplicate = self._include_rule(
            "duplicate", rule_names=rule_names, selected_rules=selected_rules
        )
        include_wrong_tier = self._include_rule(
            "wrong_tier", rule_names=rule_names, selected_rules=selected_rules
        )
        unused_count = 0
        removal_candidates: list[m.Infra.Census.RemovalCandidate] = []
        for item in objects:
            is_unused = self._is_unused(item)
            if include_duplicate and self._object_key(item) in duplicate_keys:
                violations.append(
                    self._violation(
                        item,
                        kind="duplicate",
                        description="Duplicate definition in workspace",
                    )
                )
            if is_unused and include_unused:
                unused_count += 1
                violations.append(
                    self._violation(
                        item,
                        kind="unused",
                        description="Object has no non-definition references",
                    )
                )
            if (
                include_wrong_tier
                and item.expected_tier
                and item.actual_tier
                and item.expected_tier != item.actual_tier
            ):
                violations.append(
                    self._violation(
                        item,
                        kind="wrong_tier",
                        description=f"Expected tier '{item.expected_tier}' but found '{item.actual_tier}'",
                    )
                )
            candidate = self._removal_candidate(item, include_unused=include_unused)
            if candidate is not None:
                removal_candidates.append(candidate)
        return m.Infra.Census.ProjectReport(
            project=project,
            objects=objects,
            objects_total=len(objects),
            objects_by_kind=dict(Counter(item.kind for item in objects)),
            violations=tuple(violations),
            fixes=fixes,
            violations_total=len(violations),
            fixes_applied=sum(1 for fix in fixes if fix.applied),
            unused_count=unused_count,
            removal_candidate_count=len(removal_candidates),
            removal_candidates=tuple(removal_candidates),
        )


__all__: list[str] = ["FlextInfraRefactorCensusProjectMixin"]
