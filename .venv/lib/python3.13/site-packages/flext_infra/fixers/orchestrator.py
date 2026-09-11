"""Catalog-driven enforcement fix orchestrator.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Annotated, ClassVar, override

from flext_core import r
from flext_infra import c, m, p, t, u
from flext_infra._enforcement.engine import FlextInfraEnforcementEngine
from flext_infra.base_selection import FlextInfraProjectSelectionServiceBase
from flext_infra.fixers.gate_fixer import FlextInfraGateFixerAdapter
from flext_infra.fixers.manual_fixer import FlextInfraManualFixerAdapter
from flext_infra.fixers.rope_fixer import FlextInfraRopeFixerAdapter
from flext_infra.fixers.transformer_fixer import FlextInfraTransformerFixerAdapter

if TYPE_CHECKING:
    from pathlib import Path

    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra.fixers.base import FlextInfraFixerAdapter


class FlextInfraEnforcementFixerOrchestrator(
    FlextInfraProjectSelectionServiceBase[str]
):
    """Apply automatic fixes for enforcement-catalog rules across workspace projects.

    The orchestrator reads the canonical enforcement catalog from flext-core,
    keeps only rules that declare a ``fix_action``, collects violations for
    those rules using the same validators/detectors that the pytest dispatcher
    uses, and routes each violation to the appropriate adapter.
    """

    _ADAPTER_CLASSES: ClassVar[tuple[type[FlextInfraFixerAdapter], ...]] = (
        FlextInfraGateFixerAdapter,
        FlextInfraManualFixerAdapter,
        FlextInfraRopeFixerAdapter,
        FlextInfraTransformerFixerAdapter,
    )

    apply: Annotated[
        bool, m.Field(description="Apply fixes instead of dry-run preview")
    ] = False
    rules: Annotated[
        tuple[str, ...], m.Field(description="Enforcement rule IDs to fix")
    ] = ()
    safe_only: Annotated[
        bool, m.Field(description="Only apply fixes marked safe in the catalog")
    ] = True
    check_after: Annotated[
        bool, m.Field(description="Re-run the corresponding check after fixing")
    ] = True

    @classmethod
    def execute_payload(cls, params: m.Infra.FixEnforcementCommand) -> p.Result[str]:
        """Execute enforcement fixes from the canonical CLI payload."""
        instance = cls(
            repository_root=params.workspace_path,
            selected_projects=params.projects,
            apply=params.apply,
            rules=tuple(params.rules),
            safe_only=params.safe_only,
            check_after=params.check_after,
        )
        return instance.execute()

    @override
    def execute(self) -> p.Result[str]:
        """Run the enforcement fix pipeline and return a human-readable report."""
        try:
            selected_rules = self._selected_rules()
        except ValueError as exc:
            return r[str].fail(str(exc))
        if not selected_rules:
            return r[str].ok("No fixable enforcement rules selected.")
        projects = self._resolve_projects()
        if projects.failure:
            return r[str].fail(projects.error or "unable to resolve projects")
        all_results: list[m.Infra.ProjectFixResult] = []
        for project in projects.value:
            project_result = self._fix_project(project, selected_rules)
            all_results.extend(project_result)
        report = self._render_report(all_results)
        if any(result.failed for result in all_results):
            return r[str].fail(report)
        return r[str].ok(report)

    def _selected_rules(
        self, catalog: me.EnforcementCatalog | None = None
    ) -> tuple[me.EnforcementRuleSpec, ...]:
        """Return enabled rules with fix actions matching the CLI filter.

        In the default mode (no explicit ``--rules``), adapterless fix actions
        stay selected so the routing phase emits a failed fix instead of
        silently hiding an unsupported catalog contract. When ``--rules`` is
        used, every requested rule must be enabled, declare a fix action, and
        have an available adapter; otherwise a ``ValueError`` is raised so the
        caller can surface a clear failure.
        """
        catalog = catalog or FlextInfraEnforcementEngine.canonical_catalog()
        adapterless = tuple(
            rule.id
            for rule in catalog.enabled_rules()
            if rule.fix_action is not None and not self._has_adapter(rule)
        )
        return FlextInfraEnforcementEngine.selected_rules(
            catalog=catalog,
            wanted=self.rules,
            safe_only=self.safe_only,
            adapterless=adapterless,
        )

    def _has_adapter(self, rule: me.EnforcementRuleSpec) -> bool:
        """Return whether ``rule`` has a registered fixer adapter."""
        fix_action = rule.fix_action
        if fix_action is None:
            return False
        return self._adapter_for(fix_action) is not None

    def _resolve_projects(self) -> p.Result[t.SequenceOf[p.Infra.ProjectInfo]]:
        """Resolve the project list from CLI selection or workspace discovery."""
        projects_result = u.Infra.projects(self.repository_root)
        if projects_result.failure:
            return r[t.SequenceOf[p.Infra.ProjectInfo]].fail(
                projects_result.error or "workspace discovery failed"
            )
        discovered = tuple(projects_result.unwrap())
        selected_projects: t.StrSequence = (
            self.project_names if self.project_names is not None else ()
        )
        scope: frozenset[str] = frozenset(selected_projects)
        available_names = {
            name
            for project in discovered
            for name in (project.name, project.path.name)
            if name
        }
        missing = scope - available_names
        if missing:
            return r[t.SequenceOf[p.Infra.ProjectInfo]].fail(
                f"Requested projects were not discovered: {', '.join(sorted(missing))}"
            )
        selected = (
            tuple(p for p in discovered if p.name in scope or p.path.name in scope)
            if scope
            else discovered
        )
        return r[t.SequenceOf[p.Infra.ProjectInfo]].ok(selected)

    def _fix_project(
        self, project: p.Infra.ProjectInfo, rules: t.SequenceOf[me.EnforcementRuleSpec]
    ) -> t.SequenceOf[m.Infra.ProjectFixResult]:
        """Collect violations and apply fixes for one project."""
        project_dir = project.path
        results: list[m.Infra.ProjectFixResult] = []
        by_adapter, unhandled = self._group_by_adapter(rules)
        if unhandled:
            results.append(
                m.Infra.ProjectFixResult(
                    project=project_dir.name,
                    failed=tuple(
                        self._collection_failure(
                            project_dir,
                            rule,
                            "no registered fixer adapter for "
                            f"{rule.fix_action.kind if rule.fix_action else 'none'}:"
                            f"{rule.fix_action.target if rule.fix_action else 'none'}",
                        )
                        for rule in unhandled
                    ),
                )
            )
            if self.fail_fast:
                return tuple(results)
        for adapter_cls, adapter_rules in by_adapter.items():
            adapter = self._instantiate_adapter(adapter_cls)
            violations, failures = self._collect_violations(
                project_dir=project_dir, rules=adapter_rules
            )
            if failures:
                results.append(
                    m.Infra.ProjectFixResult(
                        project=project_dir.name, failed=tuple(failures)
                    )
                )
                if self.fail_fast:
                    return tuple(results)
            if not violations:
                continue
            try:
                result = adapter.fix_project(
                    project_dir, violations, self._command_ctx()
                )
            except c.EXC_BROAD_RUNTIME as exc:
                rule_id = violations[0][0].id
                results.append(
                    m.Infra.ProjectFixResult(
                        project=project_dir.name,
                        failed=(
                            m.Infra.FailedFix(
                                rule_id=rule_id,
                                file_path=str(project_dir),
                                error=(
                                    f"{adapter_cls.__name__} failed: "
                                    f"{type(exc).__name__}: {exc}"
                                ),
                            ),
                        ),
                    )
                )
                if self.fail_fast:
                    return tuple(results)
                continue
            results.append(result)
            if result.failed and self.fail_fast:
                return tuple(results)
        return tuple(results)

    def _group_by_adapter(
        self, rules: t.SequenceOf[me.EnforcementRuleSpec]
    ) -> tuple[
        dict[type[FlextInfraFixerAdapter], list[me.EnforcementRuleSpec]],
        tuple[me.EnforcementRuleSpec, ...],
    ]:
        """Group rules by the adapter that can handle their fix_action."""
        grouped: dict[type[FlextInfraFixerAdapter], list[me.EnforcementRuleSpec]] = (
            defaultdict(list)
        )
        unhandled: list[me.EnforcementRuleSpec] = []
        for rule in rules:
            fix_action = rule.fix_action
            if fix_action is None:
                unhandled.append(rule)
                continue
            adapter_cls = self._adapter_for(fix_action)
            if adapter_cls is not None:
                grouped[adapter_cls].append(rule)
                continue
            unhandled.append(rule)
        return grouped, tuple(unhandled)

    def _adapter_for(
        self, fix_action: me.EnforcementFixAction
    ) -> type[FlextInfraFixerAdapter] | None:
        """Return the first adapter class that accepts ``fix_action``."""
        for adapter_cls in self._ADAPTER_CLASSES:
            adapter = adapter_cls.__new__(adapter_cls)
            if adapter.can_fix(fix_action):
                return adapter_cls
        return None

    def _instantiate_adapter(
        self, adapter_cls: type[FlextInfraFixerAdapter]
    ) -> FlextInfraFixerAdapter:
        """Create an adapter instance, injecting repository root."""
        return adapter_cls(self.repository_root)

    def _engine(self) -> FlextInfraEnforcementEngine:
        """Build the shared enforcement engine for this workspace."""
        return FlextInfraEnforcementEngine(self.repository_root)

    def _collect_violations(
        self, project_dir: Path, rules: t.SequenceOf[me.EnforcementRuleSpec]
    ) -> tuple[
        list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]], list[m.Infra.FailedFix]
    ]:
        """Collect violations for ``rules`` inside ``project_dir``."""
        evaluation = self._engine().collect_project(project_dir, rules)
        return evaluation.violations, evaluation.failures

    def _collect_tests_validator_violations(
        self, project_dir: Path, rule: me.EnforcementRuleSpec
    ) -> tuple[
        list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]], list[m.Infra.FailedFix]
    ]:
        """Run the flext-tests validator method for ``rule``."""
        return self._engine().collect_tests_validator(project_dir, rule)

    def _collect_python_file_violations(
        self, project_dir: Path, rule: me.EnforcementRuleSpec
    ) -> tuple[
        list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]], list[m.Infra.FailedFix]
    ]:
        """Return one probe per Python file for transformer-backed detector rules.

        Runtime detector rules backed by a deterministic source
        transformer are applied project-wide; the transformer itself decides
        whether each file needs a change. This keeps the orchestrator from
        silently skipping rules whose detectors do not emit per-file probes.
        """
        return self._engine().collect_python_file_probes(project_dir, rule)

    def _collect_declarative_violations(
        self, project_dir: Path, rules: t.SequenceOf[me.EnforcementRuleSpec]
    ) -> tuple[
        list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]], list[m.Infra.FailedFix]
    ]:
        """Return concrete probes by running the declarative engine per file.

        Unlike the generic per-file probe, this asks the declarative engine
        to actually inspect each file and emit violation probes with line
        numbers and metadata. All ``rules`` are evaluated inside a single rope
        project to avoid repeated project open/close overhead.
        """
        return self._engine().collect_declarative(project_dir, rules)

    @staticmethod
    def _stub_file_paths(project_dir: Path) -> tuple[Path, ...]:
        """Return source stub files while respecting canonical excluded dirs."""
        return FlextInfraEnforcementEngine.stub_file_paths(project_dir)

    def _collect_project_violations(
        self, project_dir: Path, rule: me.EnforcementRuleSpec
    ) -> list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]]:
        """Return one project-level probe for gate-backed fixes."""
        return FlextInfraEnforcementEngine.collect_project_probe(project_dir, rule)

    @staticmethod
    def _probe_for_path(path: Path) -> p.AttributeProbe:
        """Build the minimal structural probe consumed by fixer adapters."""
        return FlextInfraEnforcementEngine.probe_for_path(path)

    @staticmethod
    def _collection_failure(
        project_dir: Path, rule: me.EnforcementRuleSpec, message: str
    ) -> m.Infra.FailedFix:
        """Build a failed-fix record for collection/routing errors."""
        return FlextInfraEnforcementEngine.collection_failure(
            project_dir, rule, message
        )

    def _command_ctx(self) -> m.Infra.FixEnforcementCommand:
        """Build a command context for adapters from the service fields.

        Dry-run (``apply=False``) must never trigger post-fix checks, because
        check-after gates may rewrite files in place and restore originals.
        Forcing ``check_after=False`` keeps the run 100% read-only regardless
        of the CLI default or any future check-after implementation.
        """
        return m.Infra.FixEnforcementCommand(
            workspace=str(self.repository_root),
            projects=self.project_names,
            apply=self.apply,
            rules=self.rules,
            safe_only=self.safe_only,
            check_after=self.check_after and self.apply,
            fail_fast=self.fail_fast,
        )

    @staticmethod
    def _render_report(results: t.SequenceOf[m.Infra.ProjectFixResult]) -> str:
        """Render a concise text report from all project results."""
        total_fixed = sum(len(r.fixed) for r in results)
        total_previewed = sum(len(r.previewed) for r in results)
        total_skipped = sum(len(r.skipped) for r in results)
        total_failed = sum(len(r.failed) for r in results)
        project_count = len({r.project for r in results})
        lines = [
            "Enforcement fix report:",
            f"  projects: {project_count}",
            f"  fixed: {total_fixed}",
            f"  previewed: {total_previewed}",
            f"  skipped: {total_skipped}",
            f"  failed: {total_failed}",
        ]
        for result in results:
            if not (
                result.fixed or result.previewed or result.skipped or result.failed
            ):
                continue
            interesting_skipped = [
                skipped
                for skipped in result.skipped
                if skipped.reason != "no changes produced"
            ]
            lines.append(f"  {result.project}:")
            lines.extend(
                f"    + {fixed.rule_id}: {fixed.file_path}" for fixed in result.fixed
            )
            lines.extend(
                f"    ? {preview.rule_id}: {preview.file_path}"
                for preview in result.previewed
            )
            lines.extend(
                f"    ~ {skipped.rule_id}: {skipped.reason}"
                for skipped in interesting_skipped
            )
            lines.extend(
                f"    ! {failed.rule_id}: {failed.error}" for failed in result.failed
            )
        return "\n".join(lines)


__all__: list[str] = ["FlextInfraEnforcementFixerOrchestrator"]
