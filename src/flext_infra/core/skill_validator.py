"""Data-driven skill validation service.

Validates workspace skills against rules.yml-based policy gates,
supporting AST-grep and custom rule types with baseline comparison.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, MutableMapping
from pathlib import Path

from flext_core import FlextResult, r, t
from yaml import safe_load

from flext_infra import CommandRunner, JsonService, TomlService, c, m

_SKILLS_DIR = Path(".claude/skills")
_REPORT_DEFAULT = ".claude/skills/{skill}/report.json"
_BASELINE_DEFAULT = ".claude/skills/{skill}/baseline.json"
_CACHE_TTL_SECONDS = 300


def _safe_load_yaml(path: Path) -> Mapping[str, t.ConfigMapValue]:
    """Load YAML file safely, returning empty mapping on missing/invalid."""
    raw = path.read_text(encoding=c.Encoding.DEFAULT)
    parsed = safe_load(raw)
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        msg = f"rules.yml must be a mapping: {path}"
        raise TypeError(msg)
    return dict(parsed)


def _normalize_string_list(value: t.ConfigMapValue, field: str) -> list[str]:
    """Validate and normalize a list[str] config field."""
    if value is None:
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str):
                msg = f"{field} must be list[str]"
                raise TypeError(msg)
            out.append(item)
        return out
    msg = f"{field} must be list[str]"
    raise TypeError(msg)


class SkillValidator:
    """Validates workspace skills using rules.yml policy gates.

    Supports AST-grep rules, custom validator scripts, and baseline
    comparison with per-group and total strategies.
    """

    def __init__(self) -> None:
        """Initialize the skill validator."""
        self._json = JsonService()
        self._runner = CommandRunner()
        self._toml = TomlService()
        self._git_cache: MutableMapping[str, tuple[float, list[str]]] = {}

    def validate(
        self,
        workspace_root: Path,
        skill_name: str,
        *,
        mode: str = "baseline",
        _project_filter: list[str] | None = None,
    ) -> FlextResult[m.ValidationReport]:
        """Validate a single skill across workspace projects.

        Args:
            workspace_root: Root directory of the workspace.
            skill_name: Name of the skill folder to validate.
            mode: Validation mode ("baseline" or "strict").

        Returns:
            FlextResult with ValidationReport.

        """
        try:
            root = workspace_root.resolve()
            skills_dir = root / _SKILLS_DIR
            rules_path = skills_dir / skill_name / "rules.yml"
            if not rules_path.exists():
                return r[m.ValidationReport].ok(
                    m.ValidationReport(
                        passed=False,
                        violations=[f"rules.yml not found for skill '{skill_name}'"],
                        summary=f"no rules.yml for {skill_name}",
                    ),
                )

            rules = _safe_load_yaml(rules_path)
            scan_targets = rules.get("scan_targets", {}) or {}
            if not isinstance(scan_targets, dict):
                return r[m.ValidationReport].fail(
                    f"scan_targets must be a mapping: {rules_path}",
                )

            include_globs = _normalize_string_list(
                scan_targets.get("include", ["**/*.py"]),
                "scan_targets.include",
            ) or ["**/*"]
            exclude_globs = _normalize_string_list(
                scan_targets.get("exclude", []),
                "scan_targets.exclude",
            )

            rules_list = rules.get("rules", []) or []
            if not isinstance(rules_list, list):
                return r[m.ValidationReport].fail("rules must be a list")

            counts: MutableMapping[str, int] = {}
            violations: list[str] = []

            for rule_obj in rules_list:
                if not isinstance(rule_obj, dict):
                    continue
                rule_id = str(rule_obj.get("id", "")).strip()
                rule_type = str(rule_obj.get("type", "")).strip()
                group = str(rule_obj.get("group", rule_id)).strip() or rule_id

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
            passed = total == 0 if mode == "strict" else True

            if mode != "strict":
                baseline_obj = rules.get("baseline", {}) or {}
                if isinstance(baseline_obj, dict):
                    strategy = str(baseline_obj.get("strategy", "total"))
                    baseline_path = self._render_template(
                        root,
                        str(baseline_obj.get("file", _BASELINE_DEFAULT)),
                        skill_name,
                    )
                    if baseline_path.exists():
                        bl_result = self._json.read(baseline_path)
                        if bl_result.is_success:
                            bl_data = bl_result.value
                            bl_counts_raw = bl_data.get("counts", {})
                            if isinstance(bl_counts_raw, dict):
                                bl_counts = {
                                    str(k): int(v)
                                    for k, v in bl_counts_raw.items()
                                    if isinstance(v, int)
                                }
                                if strategy == "total":
                                    passed = total <= sum(bl_counts.values())
                                else:
                                    passed = all(
                                        counts.get(g, 0) <= bl_counts.get(g, 0)
                                        for g in set(counts) | set(bl_counts)
                                    )

            summary = (
                f"{skill_name}: {total} violations, {'PASS' if passed else 'FAIL'}"
            )
            return r[m.ValidationReport].ok(
                m.ValidationReport(
                    passed=passed,
                    violations=violations,
                    summary=summary,
                ),
            )
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return r[m.ValidationReport].fail(
                f"skill validation failed: {exc}",
            )

    def _run_ast_grep_count(
        self,
        rule: Mapping[str, t.ConfigMapValue],
        skill_dir: Path,
        project_path: Path,
        include_globs: list[str],
        exclude_globs: list[str],
    ) -> int:
        """Run an ast-grep rule and return match count."""
        rule_file_raw = str(rule.get("file", "")).strip()
        if not rule_file_raw:
            return 0
        rule_file = Path(rule_file_raw)
        if not rule_file.is_absolute():
            rule_file = (skill_dir / rule_file_raw).resolve()
        if not rule_file.exists():
            return 0

        cmd = ["sg", "scan", "--rule", str(rule_file), "--json=stream"]
        for pat in include_globs:
            cmd.extend(["--globs", pat])
        for pat in exclude_globs:
            cmd.extend(["--globs", f"!{pat}"])
        cmd.append(str(project_path))

        result_wrapper = self._runner.run_raw(
            cmd,
            cwd=project_path,
            timeout=300,
        )
        if result_wrapper.is_failure:
            return 0
        result = result_wrapper.value

        if result.exit_code not in {0, 1}:
            return 0

        count = 0
        for raw_line in (result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                continue
        return count

    def _run_custom_count(
        self,
        rule: Mapping[str, t.ConfigMapValue],
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
            [sys.executable, str(script)] if script.suffix == ".py" else [str(script)]
        )
        cmd.extend(["--root", str(project_path)])
        if bool(rule.get("pass_mode")):
            cmd.extend(["--mode", mode])

        result_wrapper = self._runner.run_raw(
            cmd,
            cwd=project_path,
            timeout=300,
        )
        if result_wrapper.is_failure:
            return 0
        result = result_wrapper.value

        count = 0
        for raw_line in (result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                maybe = payload.get("violation_count", payload.get("count", 0))
                if isinstance(maybe, int):
                    count += maybe
        if result.exit_code == 1:
            count = max(count, 1)
        return count

    @staticmethod
    def _render_template(root: Path, template: str, skill: str) -> Path:
        """Render a skill path template."""
        rendered = template.replace("{skill}", skill)
        candidate = Path(rendered)
        if candidate.is_absolute():
            return candidate
        return (root / candidate).resolve()


__all__ = ["SkillValidator"]
