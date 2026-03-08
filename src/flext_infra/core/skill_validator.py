"""Data-driven skill validation service.

Validates workspace skills against rules.yml-based policy gates,
supporting AST-grep and custom rule types with baseline comparison.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from flext_core import r
from flext_infra import (
    FlextInfraCommandRunner,
    FlextInfraJsonService,
    FlextInfraTomlService,
    c,
    m,
    p,
    t,
)
from flext_infra._utilities.yaml import FlextInfraUtilitiesYaml


def _safe_load_yaml(path: Path) -> Mapping[str, t.ContainerValue]:
    """Load YAML file safely; delegates to ``u.Infra.Yaml``."""
    return FlextInfraUtilitiesYaml.safe_load_yaml(path)


def _normalize_string_list(value: t.ContainerValue, field: str) -> list[str]:
    """Validate and normalize a list[str] config field; delegates to ``u.Infra.Yaml``."""
    return FlextInfraUtilitiesYaml.normalize_string_list(value, field)


class FlextInfraSkillValidator:
    """Validates workspace skills using rules.yml policy gates.

    Supports AST-grep rules, custom validator scripts, and baseline
    comparison with per-group and total strategies.
    """

    def __init__(self) -> None:
        """Initialize the skill validator."""
        self._json = FlextInfraJsonService()
        self._runner: p.Infra.CommandRunner = FlextInfraCommandRunner()
        self._toml = FlextInfraTomlService()
        self._git_cache: MutableMapping[str, tuple[float, list[str]]] = {}

    @staticmethod
    def _render_template(root: Path, template: str, skill: str) -> Path:
        """Render a skill path template."""
        rendered = template.replace("{skill}", skill)
        candidate = Path(rendered)
        if candidate.is_absolute():
            return candidate
        return (root / candidate).resolve()

    def validate(
        self,
        workspace_root: Path,
        skill_name: str,
        *,
        mode: str = c.Infra.Modes.BASELINE,
        _project_filter: list[str] | None = None,
    ) -> r[m.Infra.Core.ValidationReport]:
        """Validate a single skill across workspace projects.

        Args:
            workspace_root: Root directory of the workspace.
            skill_name: Name of the skill folder to validate.
            mode: Validation mode ("baseline" or "strict").

        Returns:
            r with ValidationReport.

        """
        try:
            root = workspace_root.resolve()
            skills_dir = root / c.Infra.Core.SKILLS_DIR
            rules_path = skills_dir / skill_name / "rules.yml"
            if not rules_path.exists():
                return r[m.Infra.Core.ValidationReport].ok(
                    m.Infra.Core.ValidationReport(
                        passed=False,
                        violations=[f"rules.yml not found for skill '{skill_name}'"],
                        summary=f"no rules.yml for {skill_name}",
                    )
                )
            rules = _safe_load_yaml(rules_path)
            scan_targets = rules.get("scan_targets", {}) or {}
            if not isinstance(scan_targets, dict):
                return r[m.Infra.Core.ValidationReport].fail(
                    f"scan_targets must be a mapping: {rules_path}"
                )
            include_globs = _normalize_string_list(
                scan_targets.get("include", ["**/*.py"]), "scan_targets.include"
            ) or ["**/*"]
            exclude_globs = _normalize_string_list(
                scan_targets.get(c.Infra.Toml.EXCLUDE, []), "scan_targets.exclude"
            )
            rules_list = rules.get(c.Infra.ReportKeys.RULES, []) or []
            if not isinstance(rules_list, list):
                return r[m.Infra.Core.ValidationReport].fail("rules must be a list")
            counts: MutableMapping[str, int] = {}
            violations: list[str] = []
            for rule_obj in rules_list:
                if not isinstance(rule_obj, dict):
                    continue
                rule_id = str(rule_obj.get(c.Infra.ReportKeys.ID, "")).strip()
                rule_type = str(rule_obj.get("type", "")).strip()
                group = (
                    str(rule_obj.get(c.Infra.Toml.GROUP, rule_id)).strip() or rule_id
                )
                if rule_type == "ast-grep":
                    count = self._run_ast_grep_count(
                        rule_obj,
                        skills_dir / skill_name,
                        root,
                        include_globs,
                        exclude_globs,
                    )
                    counts[group] = counts.get(group, 0) + count
                    if count > 0:
                        violations.append(f"[{rule_id}] {count} ast-grep matches")
                elif rule_type == "custom":
                    count = self._run_custom_count(
                        rule_obj, skills_dir / skill_name, root, mode
                    )
                    counts[group] = counts.get(group, 0) + count
                    if count > 0:
                        violations.append(f"[{rule_id}] {count} custom violations")
            total = sum(counts.values())
            passed = total == 0 if mode == c.Infra.Modes.STRICT else True
            if mode != c.Infra.Modes.STRICT:
                baseline_obj = rules.get(c.Infra.Modes.BASELINE, {}) or {}
                if isinstance(baseline_obj, dict):
                    strategy = str(
                        baseline_obj.get("strategy", c.Infra.ReportKeys.TOTAL)
                    )
                    baseline_path = self._render_template(
                        root,
                        str(
                            baseline_obj.get(
                                c.Infra.ReportKeys.FILE, c.Infra.Core.BASELINE_DEFAULT
                            )
                        ),
                        skill_name,
                    )
                    if baseline_path.exists():
                        bl_data_result = self._json.read(baseline_path)
                        if bl_data_result.is_success:
                            bl_data = bl_data_result.value
                            bl_counts_raw = bl_data.get("counts", {})
                            if isinstance(bl_counts_raw, dict):
                                bl_counts = {
                                    str(k): int(v)
                                    for k, v in bl_counts_raw.items()
                                    if isinstance(v, int)
                                }
                                if strategy == c.Infra.ReportKeys.TOTAL:
                                    passed = total <= sum(bl_counts.values())
                                else:
                                    passed = all(
                                        counts.get(g, 0) <= bl_counts.get(g, 0)
                                        for g in set(counts) | set(bl_counts)
                                    )
            summary = (
                f"{skill_name}: {total} violations, {('PASS' if passed else 'FAIL')}"
            )
            return r[m.Infra.Core.ValidationReport].ok(
                m.Infra.Core.ValidationReport(
                    passed=passed, violations=violations, summary=summary
                )
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return r[m.Infra.Core.ValidationReport].fail(
                f"skill validation failed: {exc}"
            )

    def _run_ast_grep_count(
        self,
        rule: Mapping[str, t.ContainerValue],
        skill_dir: Path,
        project_path: Path,
        include_globs: list[str],
        exclude_globs: list[str],
    ) -> int:
        """Run an ast-grep rule and return match count."""
        rule_file_raw = str(rule.get(c.Infra.ReportKeys.FILE, "")).strip()
        if not rule_file_raw:
            return 0
        rule_file = Path(rule_file_raw)
        if not rule_file.is_absolute():
            rule_file = (skill_dir / rule_file_raw).resolve()
        if not rule_file.exists():
            return 0
        cmd = [
            c.Infra.Cli.SG,
            c.Infra.Cli.SgCmd.SCAN,
            "--rule",
            str(rule_file),
            "--json=stream",
        ]
        for pat in include_globs:
            cmd.extend(["--globs", pat])
        for pat in exclude_globs:
            cmd.extend(["--globs", f"!{pat}"])
        cmd.append(str(project_path))
        result_wrapper = self._runner.run_raw(
            cmd, cwd=project_path, timeout=c.Infra.Timeouts.DEFAULT
        )
        if result_wrapper.is_failure:
            return 0
        result: p.Infra.CommandOutput = result_wrapper.value
        if result.exit_code not in {0, 1}:
            return 0
        count = 0
        for raw_line in (result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parsed_line_result = self._json.parse(line)
            if parsed_line_result.is_success:
                count += 1
        return count

    def _run_custom_count(
        self,
        rule: Mapping[str, t.ContainerValue],
        skill_dir: Path,
        project_path: Path,
        mode: str,
    ) -> int:
        """Run a custom rule script and return violation count."""
        script_raw = str(rule.get("script", "")).strip()
        if not script_raw:
            return 0
        script = Path(script_raw)
        if not script.is_absolute():
            script = (skill_dir / script_raw).resolve()
        if not script.exists():
            return 0
        cmd: list[str] = (
            [sys.executable, str(script)]
            if script.suffix == c.Infra.Extensions.PYTHON
            else [str(script)]
        )
        cmd.extend(["--root", str(project_path)])
        if bool(rule.get("pass_mode")):
            cmd.extend(["--mode", mode])
        result_wrapper = self._runner.run_raw(
            cmd, cwd=project_path, timeout=c.Infra.Timeouts.DEFAULT
        )
        if result_wrapper.is_failure:
            return 0
        result: p.Infra.CommandOutput = result_wrapper.value
        count = 0
        for raw_line in (result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload_result = self._json.parse(line)
            if payload_result.is_success and isinstance(payload_result.value, dict):
                payload = payload_result.value
                maybe = payload.get("violation_count", payload.get("count", 0))
                if isinstance(maybe, int):
                    count += maybe
        if result.exit_code == 1:
            count = max(count, 1)
        return count


__all__ = ["FlextInfraSkillValidator"]
