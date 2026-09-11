"""Data-driven skill validation service.

Validates workspace skills against rules.yml-based policy gates,
supporting AST-grep and custom rule types with baseline comparison.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, override

from flext_core import r
from flext_infra import c, m, t, u
from flext_infra.base import s
from flext_infra.validate._skill_rule_runner import FlextInfraSkillRuleRunnerMixin

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraSkillValidator(s[bool], FlextInfraSkillRuleRunnerMixin):
    """Validates workspace skills using rules.yml policy gates.

    Supports AST-grep rules, custom validator scripts, and baseline
    comparison with per-group and total strategies.
    """

    skill: Annotated[str, m.Field(description="Skill folder name")]
    mode: Annotated[
        c.Infra.OperationMode,
        m.Field(description="Validation mode (baseline or strict)"),
    ] = c.Infra.OperationMode.BASELINE

    @staticmethod
    def _render_template(repository_root: Path, template: str, skill: str) -> Path:
        """Render a skill path template."""
        rendered = template.replace("{skill}", skill)
        candidate = Path(rendered)
        if candidate.is_absolute():
            return candidate
        return (repository_root / candidate).resolve()

    def _apply_baseline_comparison(
        self,
        rules: t.MappingKV[str, t.Infra.InfraValue],
        root: Path,
        skill_name: str,
        counts: t.IntMapping,
        total: int,
    ) -> bool:
        """Compare counts against the baseline file and return pass/fail."""
        baseline_obj = u.Cli.json_deep_mapping(
            rules, c.Infra.OperationMode.BASELINE.value
        )
        if not baseline_obj:
            return True
        strategy = str(baseline_obj.get("strategy", c.Infra.RK_TOTAL))
        baseline_path = self._render_template(
            root,
            str(baseline_obj.get(c.Infra.RK_FILE, c.Infra.BASELINE_DEFAULT)),
            skill_name,
        )
        if not baseline_path.exists():
            return True
        bl_data_result = u.Cli.json_read(baseline_path)
        if bl_data_result.failure:
            return True
        bl_data = u.Cli.json_as_mapping(bl_data_result.value)
        bl_counts_raw_map = u.Cli.json_deep_mapping(bl_data, "counts")
        bl_counts: t.MutableIntMapping = {}
        for key_obj, val_obj in bl_counts_raw_map.items():
            if isinstance(val_obj, int):
                bl_counts[key_obj] = val_obj
        if strategy == c.Infra.RK_TOTAL:
            baseline_allows_total: bool = total <= sum(bl_counts.values())
            return baseline_allows_total
        within_limits: bool = all(
            counts.get(g, 0) <= bl_counts.get(g, 0)
            for g in set(counts) | set(bl_counts)
        )
        return within_limits

    def build_report(
        self,
        repository_root: Path,
        skill_name: str,
        *,
        mode: c.Infra.OperationMode = c.Infra.OperationMode.BASELINE,
        _project_filter: t.StrSequence | None = None,
    ) -> p.Result[m.Infra.ValidationReport]:
        """Validate a single skill across workspace projects.

        Args:
            repository_root: Root directory of the workspace.
            skill_name: Name of the skill folder to validate.
            mode: Validation mode ("baseline" or "strict").

        Returns:
            r with ValidationReport.

        """
        try:
            return self._build_skill_report(repository_root, skill_name, mode)
        except c.EXC_OS_RUNTIME_TYPE as exc:
            return r[m.Infra.ValidationReport].fail_op("skill validation", exc)

    @staticmethod
    def _missing_rules_report(skill_name: str) -> m.Infra.ValidationReport:
        """Build the report returned when a skill lacks rules.yml."""
        return m.Infra.ValidationReport(
            passed=False,
            violations=[f"rules.yml not found for skill '{skill_name}'"],
            summary=f"no rules.yml for {skill_name}",
        )

    @staticmethod
    def _scan_globs(
        scan_targets: t.MappingKV[str, t.Infra.InfraValue],
    ) -> tuple[t.StrSequence, t.StrSequence]:
        """Return include and exclude glob lists from rules.yml scan targets."""
        include_globs = u.Infra.string_list(
            scan_targets.get("include", ["**/*.py"])
        ) or ["**/*"]
        exclude_globs = u.Infra.string_list(scan_targets.get(c.Infra.EXCLUDE, []))
        return include_globs, exclude_globs

    @staticmethod
    def _rules_list(
        rules: t.MappingKV[str, t.Infra.InfraValue],
    ) -> p.Result[t.JsonList]:
        """Validate the rules.yml rules payload."""
        rules_list_obj = rules.get(c.Infra.RK_RULES, [])
        if not isinstance(rules_list_obj, list):
            return r[t.JsonList].fail("rules must be a list")
        return r[t.JsonList].ok(t.Cli.JSON_LIST_ADAPTER.validate_python(rules_list_obj))

    def _evaluate_rules(
        self, context: m.Infra.SkillRuleEvaluationContext
    ) -> tuple[t.IntMapping, t.StrSequence]:
        """Evaluate skill validation rules and return counts plus violations."""
        counts: t.MutableIntMapping = {}
        violations: t.MutableSequenceOf[str] = []
        for rule_obj_raw in context.rules_list:
            rule_obj = u.Cli.json_as_mapping(rule_obj_raw)
            if not rule_obj:
                continue
            self._evaluate_single_rule(
                rule_obj,
                context.skill_dir,
                context.root,
                context.mode,
                context.include_globs,
                context.exclude_globs,
                counts,
                violations,
            )
        return counts, tuple(violations)

    def _skill_report_model(
        self, context: m.Infra.SkillReportContext
    ) -> m.Infra.ValidationReport:
        """Build the canonical skill validation report model."""
        total = sum(context.counts.values())
        passed = (
            total == 0
            if context.mode == c.Infra.OperationMode.STRICT
            else self._apply_baseline_comparison(
                context.rules, context.root, context.skill_name, context.counts, total
            )
        )
        summary = (
            f"{context.skill_name}: {total} violations, "
            f"{('PASS' if passed else 'FAIL')}"
        )
        return m.Infra.ValidationReport(
            passed=passed, violations=context.violations, summary=summary
        )

    def _build_skill_report(
        self, repository_root: Path, skill_name: str, mode: c.Infra.OperationMode
    ) -> p.Result[m.Infra.ValidationReport]:
        """Build a skill validation report after path resolution."""
        root = repository_root.resolve()
        skills_dir = root / c.Infra.SKILLS_DIR
        rules_path = skills_dir / skill_name / "rules.yml"
        if not rules_path.exists():
            return r[m.Infra.ValidationReport].ok(
                self._missing_rules_report(skill_name)
            )
        rules = u.Cli.yaml_load_mapping(rules_path)
        scan_targets_raw = rules.get("scan_targets", {})
        scan_targets = u.Cli.json_as_mapping(scan_targets_raw)
        if not scan_targets and scan_targets_raw not in ({}, None):
            return r[m.Infra.ValidationReport].fail(
                f"scan_targets must be a mapping: {rules_path}"
            )
        include_globs, exclude_globs = self._scan_globs(scan_targets)
        rules_list_result = self._rules_list(rules)
        if rules_list_result.failure:
            return r[m.Infra.ValidationReport].fail(
                rules_list_result.error or "rules must be a list"
            )
        counts, violations = self._evaluate_rules(
            m.Infra.SkillRuleEvaluationContext(
                rules_list=rules_list_result.value,
                skill_dir=skills_dir / skill_name,
                root=root,
                mode=mode,
                include_globs=include_globs,
                exclude_globs=exclude_globs,
            )
        )
        return r[m.Infra.ValidationReport].ok(
            self._skill_report_model(
                m.Infra.SkillReportContext(
                    rules=rules,
                    root=root,
                    skill_name=skill_name,
                    mode=mode,
                    counts=counts,
                    violations=violations,
                )
            )
        )

    @override
    def execute(self) -> p.Result[bool]:
        """Execute the skill-validation CLI flow."""
        report_result = self.build_report(
            self.repository_root, self.skill, mode=self.mode
        )
        if report_result.failure:
            return r[bool].fail(report_result.error or "skill validation failed")
        report = report_result.unwrap()
        return r[bool].ok(True) if report.passed else r[bool].fail(report.summary)


__all__: list[str] = ["FlextInfraSkillValidator"]
