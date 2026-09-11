"""Cohesive external-tool-runner mixin (deptry, mypy stubs, pip-check) for detection."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, t, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraDependencyDetectionRunnersMixin:
    """Mixin holding raw-run contract plus external analysis-tool runners."""

    if TYPE_CHECKING:
        # Conversion helper provided by the concrete analyzer; declared for static
        # resolution only (runtime impl lives on the concrete via FLEXT).
        def _to_toml_config(
            self, payload: t.MappingKV[str, t.Infra.InfraValue]
        ) -> t.JsonMapping: ...

    def _read_plain(self, path: Path) -> p.Result[t.JsonMapping]:
        """Read plain; concrete analyzer supplies the real reader."""
        _ = path
        msg = "_read_plain must be implemented by the concrete analyzer"
        raise NotImplementedError(msg)

    def _run_raw(
        self,
        cmd: t.StrSequence,
        *,
        cwd: Path | None = None,
        timeout: int | None = None,
        env: t.StrMapping | None = None,
    ) -> p.Result[p.Cli.CommandOutput]:
        """Run raw command; concrete analyzer supplies the real runner."""
        _ = cmd, cwd, timeout, env
        msg = "_run_raw must be implemented by the concrete analyzer"
        raise NotImplementedError(msg)

    def run_deptry(
        self,
        project_path: Path,
        venv_bin: Path,
        *,
        config_path: Path | None = None,
        json_output_path: Path | None = None,
        extend_exclude: t.StrSequence | None = None,
    ) -> p.Result[t.Pair[t.SequenceOf[t.JsonMapping], int]]:
        """Run deptry analysis on a project and parse JSON output."""
        settings = config_path or project_path / c.Infra.PYPROJECT_FILENAME
        if not settings.exists():
            return r[t.Pair[t.SequenceOf[t.JsonMapping], int]].ok(([], 0))
        out_file = json_output_path or project_path / ".deptry-report.json"
        cmd: t.MutableSequenceOf[str] = [
            str(venv_bin / c.Infra.DEPTRY),
            ".",
            "--config",
            str(settings),
            "--json-output",
            str(out_file),
            "--no-ansi",
        ]
        if extend_exclude:
            for excluded in extend_exclude:
                cmd.extend(["--extend-exclude", excluded])
        result = self._run_raw(cmd, cwd=project_path, timeout=c.Infra.TIMEOUT_MEDIUM)
        if result.failure:
            return r[t.Pair[t.SequenceOf[t.JsonMapping], int]].fail(
                result.error or "deptry execution failed"
            )
        issues: t.SequenceOf[t.JsonMapping] = []
        if out_file.exists():
            loaded_result = u.Cli.files_read_json(out_file)
            if loaded_result.failure:
                return r[t.Pair[t.SequenceOf[t.JsonMapping], int]].fail(
                    loaded_result.error or "deptry JSON output unreadable/invalid"
                )
            if isinstance(loaded_result.value, list):
                normalized_issues: t.MutableSequenceOf[t.JsonMapping] = []
                for item in loaded_result.value:
                    if not isinstance(item, Mapping):
                        continue
                    try:
                        typed_item = t.Infra.INFRA_MAPPING_ADAPTER.validate_python(item)
                    except c.ValidationError:
                        continue
                    converted_issue = self._to_toml_config(typed_item)
                    if len(converted_issue) == len(typed_item):
                        normalized_issues.append(converted_issue)
                issues = normalized_issues
            if json_output_path is None:
                try:
                    out_file.unlink()
                except OSError as exc:
                    return r[t.Pair[t.SequenceOf[t.JsonMapping], int]].fail(
                        f"failed to cleanup deptry temp output: {exc}"
                    )
        cmd_result: p.Cli.CommandOutput = result.value
        return r[t.Pair[t.SequenceOf[t.JsonMapping], int]].ok((
            issues,
            cmd_result.exit_code,
        ))

    def run_mypy_stub_hints(
        self, project_path: Path
    ) -> p.Result[t.Pair[t.StrSequence, t.StrSequence]]:
        """Run mypy via the command runner to detect missing stubs and hint packages."""
        cmd = u.Infra.mypy_limited_command((
            sys.executable,
            "-m",
            c.Infra.MYPY,
            c.Infra.DEFAULT_SRC_DIR,
            "--config-file",
            c.Infra.PYPROJECT_FILENAME,
            "--no-error-summary",
        ))
        result = self._run_raw(
            cmd, cwd=project_path, timeout=u.Infra.mypy_runner_timeout()
        )
        if result.failure:
            return r[t.Pair[t.StrSequence, t.StrSequence]].fail(
                u.Infra.mypy_launch_failure_diagnostic(
                    result.error or "Mypy process launch failed"
                )
            )
        command_output: p.Cli.CommandOutput = result.value
        if resource_diagnostic := u.Infra.mypy_failure_diagnostic(command_output):
            return r[t.Pair[t.StrSequence, t.StrSequence]].fail(resource_diagnostic)
        output = f"{command_output.stdout}\n{command_output.stderr}"
        hinted = {
            match.group(1).strip()
            for match in c.Infra.MYPY_HINT_RE.finditer(output)
            if match.group(1).strip()
        }
        missing = {
            match.group(1).strip()
            for match in c.Infra.MYPY_STUB_RE.finditer(output)
            if match.group(1).strip()
        }
        return r[t.Pair[t.StrSequence, t.StrSequence]].ok((
            sorted(hinted),
            sorted(missing),
        ))

    def run_pip_check(
        self, repository_root: Path, venv_bin: Path
    ) -> p.Result[t.Pair[t.StrSequence, int]]:
        """Run pip check to detect dependency conflicts in workspace."""
        pip = venv_bin / "pip"
        if not pip.exists():
            return r[t.Pair[t.StrSequence, int]].ok(([], 0))
        env = {"VIRTUAL_ENV": str(venv_bin.parent)}
        result = self._run_raw(
            [str(pip), c.Infra.VERB_CHECK],
            cwd=repository_root,
            timeout=c.Infra.TIMEOUT_SHORT,
            env=env,
        )
        if result.failure:
            return r[t.Pair[t.StrSequence, int]].fail(
                result.error or "pip check failed"
            )
        cmd_result: p.Cli.CommandOutput = result.value
        output = cmd_result.stdout
        lines = output.strip().splitlines() if output else []
        return r[t.Pair[t.StrSequence, int]].ok((lines, cmd_result.exit_code))


__all__: list[str] = ["FlextInfraDependencyDetectionRunnersMixin"]
