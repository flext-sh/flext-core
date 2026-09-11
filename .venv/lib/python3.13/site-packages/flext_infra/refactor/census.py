"""Workspace-wide Rope-only census orchestration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated, override

from flext_cli import cli
from flext_core import r
from flext_infra import c, m, p, t, u
from flext_infra.base_selection import FlextInfraProjectSelectionServiceBase
from flext_infra.refactor._census_apply import FlextInfraRefactorCensusApplyMixin
from flext_infra.refactor._census_collect import FlextInfraRefactorCensusCollectMixin
from flext_infra.refactor._census_collect_helpers import (
    FlextInfraRefactorCensusCollectHelpersMixin,
)
from flext_infra.refactor._census_filters import FlextInfraRefactorCensusFiltersMixin
from flext_infra.refactor._census_inventory import (
    FlextInfraRefactorCensusInventoryMixin,
)
from flext_infra.refactor._census_objects import FlextInfraRefactorCensusObjectsMixin
from flext_infra.refactor._census_project import FlextInfraRefactorCensusProjectMixin
from flext_infra.refactor._census_render import FlextInfraRefactorCensusRenderMixin
from flext_infra.refactor._census_rules_alias import (
    FlextInfraRefactorCensusRulesAliasMixin,
)
from flext_infra.refactor._census_rules_dispatch import (
    FlextInfraRefactorCensusRulesDispatchMixin,
)
from flext_infra.refactor._census_rules_struct import (
    FlextInfraRefactorCensusRulesStructMixin,
)
from flext_infra.refactor._census_symbols import FlextInfraRefactorCensusSymbolsMixin
from flext_infra.refactor._census_validate import FlextInfraRefactorCensusValidateMixin
from flext_infra.workspace.rope import FlextInfraRopeWorkspace


class FlextInfraRefactorCensus(
    FlextInfraProjectSelectionServiceBase[m.Infra.Census.WorkspaceReport],
    FlextInfraRefactorCensusApplyMixin,
    FlextInfraRefactorCensusCollectMixin,
    FlextInfraRefactorCensusCollectHelpersMixin,
    FlextInfraRefactorCensusFiltersMixin,
    FlextInfraRefactorCensusInventoryMixin,
    FlextInfraRefactorCensusObjectsMixin,
    FlextInfraRefactorCensusProjectMixin,
    FlextInfraRefactorCensusRenderMixin,
    FlextInfraRefactorCensusRulesAliasMixin,
    FlextInfraRefactorCensusRulesDispatchMixin,
    FlextInfraRefactorCensusRulesStructMixin,
    FlextInfraRefactorCensusSymbolsMixin,
    FlextInfraRefactorCensusValidateMixin,
):
    """Generalized Rope-only census service for Python objects across the workspace."""

    json_output: Annotated[
        str | None, m.Field(description="Path to write JSON report")
    ] = None
    impact_map_output: Annotated[
        str | None, m.Field(description="Path to write dry-run impact map JSON")
    ] = None
    kinds: Annotated[
        t.StrSequence | None,
        m.Field(description="Optional symbol-kind filters; repeat --kinds NAME"),
    ] = None
    rules: Annotated[
        t.StrSequence | None,
        m.Field(description="Optional violation-rule filters; repeat --rules NAME"),
    ] = None
    families: Annotated[
        t.StrSequence | None,
        m.Field(
            description="Optional namespace-family filters; repeat --families NAME"
        ),
    ] = None
    include_local_scopes: Annotated[
        bool, m.Field(description="Include locals, parameters, and nested scopes")
    ] = True

    @property
    def json_output_path(self) -> Path | None:
        """Resolved JSON export path when provided."""
        path: Path | None = u.Infra.normalize_optional_path(self.json_output)
        return path

    @property
    def impact_map_output_path(self) -> Path | None:
        """Resolved impact-map export path when provided."""
        path: Path | None = u.Infra.normalize_optional_path(self.impact_map_output)
        return path

    @property
    def kind_names(self) -> t.StrSequence | None:
        """Normalized symbol-kind filters."""
        return u.Infra.normalize_sequence_values(self.kinds)

    @property
    def rule_names(self) -> t.StrSequence | None:
        """Normalized violation-rule filters."""
        return u.Infra.normalize_sequence_values(self.rules)

    @property
    def family_names(self) -> t.StrSequence | None:
        """Normalized family filters."""
        return u.Infra.normalize_sequence_values(self.families)

    @property
    @override
    def dry_run_gate_names(self) -> t.StrSequence:
        """Per-candidate gate set (``lint`` + ``pyrefly``).

        Mypy and pyright perform transitive module analysis that flags
        ``__init__.py`` lazy-import references to just-removed symbols
        *before* ``FlextInfraCodegenLazyInit`` regenerates them at the
        end of ``_apply_supported_fixes``. That would roll back every
        safe candidate. The per-candidate gate therefore uses the two
        fast tools that validate the actual file being modified
        (``ruff`` E/F + ``pyrefly``); the session-level regen run + a
        final ``ruff format/fix`` on every touched file guarantees
        consistency afterwards.
        """
        return (c.Infra.LINT, c.Infra.PYREFLY)

    def _rope_root_for_selection(self) -> Path | None:
        """Return a project-scoped Rope root when exactly one project is selected.

        Workspace-wide scans (zero or many projects) keep the canonical workspace
        root so cross-project rules such as duplicate detection remain accurate.
        """
        names: t.StrSequence | None = self.project_names
        if names is None or len(names) != 1:
            return None
        project_name: str = names[0]
        project_path: Path = self.root / project_name
        if project_path.is_dir():
            return project_path
        return None

    def _execution_reports(
        self,
    ) -> tuple[m.Infra.Census.WorkspaceReport, m.Infra.Census.WorkspaceReport | None]:
        """Collect the final report and the pre-apply impact-map report."""
        started = time.monotonic()
        applied = frozenset[str]()
        impact_map_report: m.Infra.Census.WorkspaceReport | None = None
        rope_root = self._rope_root_for_selection()
        with FlextInfraRopeWorkspace.open_workspace(
            self.root, rope_repository_root=rope_root
        ) as rope:

            def collect(applied: frozenset[str]) -> m.Infra.Census.WorkspaceReport:
                return self._collect_report(
                    rope,
                    project_names=self.project_names,
                    kind_names=self.kind_names,
                    family_names=self.family_names,
                    rule_names=self.rule_names,
                    include_local_scopes=self.include_local_scopes,
                    applied=applied,
                )

            report = collect(applied)
            impact_map_report = report
            effective_dry_run: bool = self.effective_dry_run
            if self.apply_changes and not effective_dry_run:
                applied = self._apply_supported_fixes(rope, report)
                if applied:
                    rope.reload()
                    report = collect(applied)
        finalized_report = report.model_copy(
            update={"scan_duration_seconds": time.monotonic() - started}
        )
        return finalized_report, impact_map_report

    def build_report(self) -> m.Infra.Census.WorkspaceReport:
        """Build the canonical workspace census report without CLI side effects."""
        report, _ = self._execution_reports()
        return report

    @override
    def execute(self) -> p.Result[m.Infra.Census.WorkspaceReport]:
        """Execute the census with one shared Rope session."""
        report, impact_map_report = self._execution_reports()
        cli.display_text(self.render_text(report))
        if self.json_output_path is not None:
            u.Infra.export_pydantic_json(report, self.json_output_path)
            u.Cli.info(f"JSON report exported to: {self.json_output_path}")
        if self.impact_map_output_path is not None:
            impact_result = u.Infra.write_impact_map(
                self._impact_map_results(impact_map_report or report),
                self.impact_map_output_path,
            )
            if impact_result.failure:
                return r[m.Infra.Census.WorkspaceReport].fail(
                    impact_result.error or "impact map write failed"
                )
            u.Cli.info(f"Impact map exported to: {self.impact_map_output_path}")
        return r[m.Infra.Census.WorkspaceReport].ok(report)


__all__: list[str] = ["FlextInfraRefactorCensus"]
