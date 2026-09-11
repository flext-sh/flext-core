"""Skill validation per-rule execution + violation counting — extracted concern."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, u

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraSkillRuleRunnerMixin:
    """Run one rule (ast-grep or custom script) and count its violations.

    Composed into FlextInfraSkillValidator via inheritance; self-contained
    (all inputs passed as arguments, no validator state).
    """

    def _evaluate_single_rule(
        self,
        rule_obj: t.MappingKV[str, t.Infra.InfraValue],
        skill_dir: Path,
        root: Path,
        mode: c.Infra.OperationMode,
        include_globs: t.StrSequence,
        exclude_globs: t.StrSequence,
        counts: t.MutableIntMapping,
        violations: t.MutableSequenceOf[str],
    ) -> None:
        """Evaluate one rule entry and accumulate counts/violations."""
        rule_id = u.Cli.json_get_str_key(rule_obj, c.Infra.RK_ID)
        rule_type = u.Cli.json_get_str_key(rule_obj, "type")
        group = (
            u.Cli.json_get_str_key(rule_obj, c.Infra.GROUP, default=rule_id) or rule_id
        )
        match rule_type:
            case "ast-grep":
                count = self._run_ast_grep_count(
                    rule_obj, skill_dir, root, include_globs, exclude_globs
                )
            case "custom":
                count = self._run_custom_count(rule_obj, skill_dir, root, mode)
            case _:
                return
        counts[group] = counts.get(group, 0) + count
        if count > 0:
            label = (
                "ast-grep matches" if rule_type == "ast-grep" else "custom violations"
            )
            violations.append(f"[{rule_id}] {count} {label}")

    def _run_ast_grep_count(
        self,
        rule: t.MappingKV[str, t.Infra.InfraValue],
        skill_dir: Path,
        project_path: Path,
        include_globs: t.StrSequence,
        exclude_globs: t.StrSequence,
    ) -> int:
        """Run an ast-grep rule and return match count."""
        rule_file_raw = u.Cli.json_get_str_key(rule, c.Infra.RK_FILE)
        if not rule_file_raw:
            msg = "ast-grep rule must declare a non-empty file"
            raise RuntimeError(msg)
        rule_file = Path(rule_file_raw)
        if not rule_file.is_absolute():
            rule_file = (skill_dir / rule_file_raw).resolve()
        if not rule_file.exists():
            msg = f"ast-grep rule file does not exist: {rule_file}"
            raise RuntimeError(msg)
        cmd = [c.Infra.SG, c.Infra.SCAN, "--rule", str(rule_file), "--json=stream"]
        for pat in include_globs:
            cmd.extend(["--globs", pat])
        for pat in exclude_globs:
            cmd.extend(["--globs", f"!{pat}"])
        cmd.append(str(project_path))
        result_wrapper = u.Cli.run_raw(
            cmd, cwd=project_path, timeout=c.Infra.TIMEOUT_DEFAULT
        )
        if result_wrapper.failure:
            msg = result_wrapper.error or "ast-grep execution failed"
            raise RuntimeError(msg)
        result: p.Cli.CommandOutput = result_wrapper.value
        if result.exit_code not in {0, 1}:
            detail = (result.stderr or result.stdout).strip() or "no diagnostics"
            msg = f"ast-grep exited with code {result.exit_code}: {detail}"
            raise RuntimeError(msg)
        count = 0
        for raw_line in (result.stdout or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parsed_line_result = u.Cli.json_parse(line)
            if parsed_line_result.success:
                count += 1
        return count

    @staticmethod
    def _parse_violation_count(stdout: str) -> int:
        """Parse violation count from JSON-line stdout of a custom script."""
        count = 0
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            payload_result = u.Cli.json_parse(line)
            if payload_result.success and isinstance(payload_result.value, Mapping):
                payload = payload_result.value
                maybe = payload.get("violation_count", payload.get("count", 0))
                if isinstance(maybe, int):
                    count += maybe
        return count

    def _run_custom_count(
        self,
        rule: t.MappingKV[str, t.Infra.InfraValue],
        skill_dir: Path,
        project_path: Path,
        mode: c.Infra.OperationMode,
    ) -> int:
        """Run a custom rule script and return violation count."""
        script_raw = u.Cli.json_get_str_key(rule, "script")
        if not script_raw:
            msg = "custom rule must declare a non-empty script"
            raise RuntimeError(msg)
        script = Path(script_raw)
        if not script.is_absolute():
            script = (skill_dir / script_raw).resolve()
        if not script.exists():
            msg = f"custom rule script does not exist: {script}"
            raise RuntimeError(msg)
        cmd: t.MutableSequenceOf[str] = (
            [sys.executable, str(script)]
            if script.suffix == c.Infra.EXT_PYTHON
            else [str(script)]
        )
        cmd.extend(["--workspace", str(project_path)])
        if bool(rule.get("pass_mode")):
            cmd.extend(["--mode", mode.value])
        result_wrapper = u.Cli.run_raw(
            cmd, cwd=project_path, timeout=c.Infra.TIMEOUT_DEFAULT
        )
        if result_wrapper.failure:
            msg = result_wrapper.error or "custom rule execution failed"
            raise RuntimeError(msg)
        result: p.Cli.CommandOutput = result_wrapper.value
        if result.exit_code not in {0, 1}:
            detail = (result.stderr or result.stdout).strip() or "no diagnostics"
            msg = f"custom rule exited with code {result.exit_code}: {detail}"
            raise RuntimeError(msg)
        count = self._parse_violation_count(result.stdout or "")
        if result.exit_code == 1:
            count = max(count, 1)
        return count


__all__: list[str] = ["FlextInfraSkillRuleRunnerMixin"]
