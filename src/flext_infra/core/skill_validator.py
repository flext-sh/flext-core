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

from pydantic import JsonValue, TypeAdapter, ValidationError

from flext_core import r
from flext_infra import (
    c,
    m,
    p,
    t,
    u,
)
from flext_infra._utilities.yaml import FlextInfraUtilitiesYaml


class FlextInfraSkillValidator:
    """Validates workspace skills using rules.yml policy gates.

    Supports AST-grep rules, custom validator scripts, and baseline
    comparison with per-group and total strategies.
    """

    def __init__(self) -> None:
        """Initialize the skill validator."""
        self._git_cache: MutableMapping[str, tuple[float, list[str]]] = {}

    @staticmethod
    def _render_template(workspace_root: Path, template: str, skill: str) -> Path:
        """Render a skill path template."""
        rendered = template.replace("{skill}", skill)
        candidate = Path(rendered)
        if candidate.is_absolute():
            return candidate
        return (workspace_root / candidate).resolve()

    @staticmethod
    def _safe_load_yaml(path: Path) -> Mapping[str, t.Infra.InfraValue]:
        """Load YAML file safely; delegates to ``u.Infra``."""
        return FlextInfraUtilitiesYaml.safe_load_yaml(path)

    @staticmethod
    def _normalize_string_list(
        value: t.Infra.InfraValue | None, field: str
    ) -> list[str]:
        """Validate and normalize a list[str] config field; delegates to ``u.Infra``."""
        if value is None:
            return []
        if isinstance(value, list):
            try:
                typed_items: list[JsonValue] = TypeAdapter(
                    list[JsonValue]
                ).validate_python(value)
            except ValidationError as exc:
                msg = f"{field} must be list[str]: {exc}"
                raise TypeError(msg) from exc
            out: list[str] = []
            for item in typed_items:
                if not isinstance(item, str):
                    msg = f"{field} must be list[str]"
                    raise TypeError(msg)
                out.append(item)
            return out
        msg = f"{field} must be list[str]"
        raise TypeError(msg)

    @staticmethod
    def _normalize_str_object_mapping(
        value: Mapping[str, JsonValue] | JsonValue,
    ) -> dict[str, t.Infra.InfraValue]:
        try:
            adapter: TypeAdapter[dict[str, t.Infra.InfraValue]] = TypeAdapter(
                dict[str, t.Infra.InfraValue]
            )
            return adapter.validate_python(value)
        except ValidationError:
            return {}

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
                    ),
                )
            rules = self._safe_load_yaml(rules_path)
            scan_targets_raw = rules.get("scan_targets", {})
            scan_targets = self._normalize_str_object_mapping(scan_targets_raw)
            if not scan_targets and scan_targets_raw not in ({}, None):
                return r[m.Infra.Core.ValidationReport].fail(
                    f"scan_targets must be a mapping: {rules_path}",
                )
            include_globs = self._normalize_string_list(
                scan_targets.get("include", ["**/*.py"]),
                "scan_targets.include",
            ) or ["**/*"]
            exclude_globs = self._normalize_string_list(
                scan_targets.get(c.Infra.Toml.EXCLUDE, []),
                "scan_targets.exclude",
            )
            rules_list_obj = rules.get(c.Infra.ReportKeys.RULES, [])
            if not isinstance(rules_list_obj, list):
                return r[m.Infra.Core.ValidationReport].fail("rules must be a list")
            rules_list: list[JsonValue] = TypeAdapter(
                list[JsonValue],
            ).validate_python(rules_list_obj)
            counts: MutableMapping[str, int] = {}
            violations: list[str] = []
            for rule_obj_raw in rules_list:
                rule_obj = self._normalize_str_object_mapping(rule_obj_raw)
                if not rule_obj:
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
                        rule_obj,
                        skills_dir / skill_name,
                        root,
                        mode,
                    )
                    counts[group] = counts.get(group, 0) + count
                    if count > 0:
                        violations.append(f"[{rule_id}] {count} custom violations")
            total = sum(counts.values())
            passed = total == 0 if mode == c.Infra.Modes.STRICT else True
            if mode != c.Infra.Modes.STRICT:
                baseline_obj = self._normalize_str_object_mapping(
                    rules.get(c.Infra.Modes.BASELINE, {}),
                )
                if baseline_obj:
                    strategy = str(
                        baseline_obj.get("strategy", c.Infra.ReportKeys.TOTAL),
                    )
                    baseline_path = self._render_template(
                        root,
                        str(
                            baseline_obj.get(
                                c.Infra.ReportKeys.FILE,
                                c.Infra.Core.BASELINE_DEFAULT,
                            ),
                        ),
                        skill_name,
                    )
                    if baseline_path.exists():
                        bl_data_result = u.Infra.read_json(baseline_path)
                        if bl_data_result.is_success:
                            bl_data = self._normalize_str_object_mapping(
                                bl_data_result.value
                            )
                            bl_counts_raw_map = self._normalize_str_object_mapping(
                                bl_data.get("counts", {}),
                            )
                            bl_counts: dict[str, int] = {}
                            for key_obj, val_obj in bl_counts_raw_map.items():
                                if isinstance(val_obj, int):
                                    bl_counts[str(key_obj)] = int(val_obj)
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
                    passed=passed,
                    violations=violations,
                    summary=summary,
                ),
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return r[m.Infra.Core.ValidationReport].fail(
                f"skill validation failed: {exc}",
            )

    def _run_ast_grep_count(
        self,
        rule: Mapping[str, t.Infra.InfraValue],
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
        result_wrapper = u.Infra.run_raw(
            cmd,
            cwd=project_path,
            timeout=c.Infra.Timeouts.DEFAULT,
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
            parsed_line_result = u.Infra.parse(line)
            if parsed_line_result.is_success:
                count += 1
        return count

    def _run_custom_count(
        self,
        rule: Mapping[str, t.Infra.InfraValue],
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
        cmd.extend(["--workspace", str(project_path)])
        if bool(rule.get("pass_mode")):
            cmd.extend(["--mode", mode])
        result_wrapper = u.Infra.run_raw(
            cmd,
            cwd=project_path,
            timeout=c.Infra.Timeouts.DEFAULT,
        )
        if result_wrapper.is_failure:
            return 0
        result: p.Infra.CommandOutput = result_wrapper.value
        count = 0
        for raw_line in (result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload_result = u.Infra.parse(line)
            if payload_result.is_success and isinstance(payload_result.value, dict):
                payload = payload_result.value
                maybe = payload.get("violation_count", payload.get("count", 0))
                if isinstance(maybe, int):
                    count += maybe
        if result.exit_code == 1:
            count = max(count, 1)
        return count


__all__ = ["FlextInfraSkillValidator"]
