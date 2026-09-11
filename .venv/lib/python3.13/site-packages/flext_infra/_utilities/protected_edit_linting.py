"""Lint snapshot support for protected edit workflows."""

from __future__ import annotations

import concurrent.futures
import hashlib
from collections.abc import MutableMapping
from pathlib import Path
from typing import ClassVar

from flext_cli import u
from flext_infra._utilities.discovery import FlextInfraUtilitiesDiscovery
from flext_infra._utilities.resource_limits import FlextInfraUtilitiesResourceLimits
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t


class FlextInfraUtilitiesProtectedEditLinting:
    """Shared linting and path helpers for protected edit workflows."""

    _SNAPSHOT_MAX_WORKERS: ClassVar[int] = 4

    @staticmethod
    def _workspace_tool_command(workspace: Path, tool_name: str) -> t.StrSequence:
        """Resolve one tool against the workspace venv before falling back to PATH."""
        tool_path = (workspace.resolve() / c.Infra.VENV_BIN_REL / tool_name).resolve()
        if tool_path.is_file():
            return (str(tool_path),)
        return (tool_name,)

    @staticmethod
    def _normalize_lint_line(line: str) -> str:
        """Normalize lint line."""
        if c.Infra.CODE_FRAME_RE.match(line) or c.Infra.CODE_FRAME_BODY_RE.match(line):
            return ""

        def normalize_unused_import(match: t.Infra.RegexMatch) -> str:
            imported_name = match.group(1).rsplit(".", maxsplit=1)[-1]
            return f"`{imported_name}` imported but unused"

        normalized_line: str = c.Infra.LINE_COL_RE.sub("", line)
        normalized_without_unused_imports: str = c.Infra.UNUSED_IMPORT_RE.sub(
            normalize_unused_import, normalized_line
        )
        if c.Infra.LINT_SUMMARY_RE.match(normalized_without_unused_imports):
            return ""
        return normalized_without_unused_imports.strip()

    @staticmethod
    def _selected_lint_tools(
        gates: t.StrSequence | None = None,
    ) -> t.StrSequencePairTuple:
        """Return the selected lint tools."""
        env_gates = (
            u.Cli.process_env().get(c.Infra.ENV_VAR_LINT_SNAPSHOT_GATES, "").strip()
        )
        resolved_gates = gates or tuple(
            gate.strip()
            for gate in (
                env_gates.split(",")
                if env_gates
                else c.Infra.SAFE_EXECUTION_DEFAULT_GATES.split(",")
            )
            if gate.strip()
        )
        gate_names = {gate.strip().lower() for gate in resolved_gates if gate.strip()}
        selected = [
            (tool, tmpl)
            for tool, tmpl in c.Infra.LINT_TOOLS
            if gate_names.intersection({"lint" if tool == "ruff" else tool, tool})
        ]
        return tuple(selected) or c.Infra.LINT_TOOLS

    @classmethod
    def selected_lint_tool_names(
        cls, gates: t.StrSequence | None = None
    ) -> t.StrSequence:
        """Return the canonical lint tool names selected for a gate set."""
        return tuple(tool for tool, _ in cls._selected_lint_tools(gates))

    @classmethod
    def ruff_fix_files(cls, paths: t.SequenceOf[Path], workspace: Path) -> None:
        """Run ``ruff check --fix`` on *paths* with snapshot-identical resolution.

        Same venv-resolved binary and same working directory as the lint
        snapshots, so the fix pass and the before/after diff always resolve
        the SAME Ruff configuration for the same file.
        """
        for py_file in paths:
            _ = u.Cli.run_checked(
                [
                    *cls._workspace_tool_command(workspace, "ruff"),
                    c.Infra.CHECK,
                    "--fix",
                    str(py_file),
                ],
                cwd=cls._command_cwd(py_file, workspace),
                env=cls._command_env(),
            )

    @staticmethod
    def _relative_path(py_file: Path, workspace: Path) -> Path:
        """Relative path."""
        try:
            return py_file.relative_to(workspace)
        except ValueError:
            return py_file

    @staticmethod
    def _command_cwd(py_file: Path, workspace: Path) -> Path:
        """Command cwd."""
        resolved_workspace = workspace.resolve()
        project_root: Path | None = FlextInfraUtilitiesDiscovery.project_root(py_file)
        if project_root is None:
            return resolved_workspace
        return project_root

    _COMMAND_ENV_REMOVE_KEYS: ClassVar[t.StrSequence] = (
        c.Infra.ORCHESTRATOR_ENV_PYTHONPATH,
        c.Infra.ENV_VAR_FORCE_COLOR,
    )

    @staticmethod
    def _command_env() -> t.StrMapping:
        """Return subprocess env overrides for lint and validation commands.

        ``run_raw`` merges this mapping over ``os.environ``, so keys can only be
        dropped by passing ``_COMMAND_ENV_REMOVE_KEYS`` as ``remove_env_keys``.
        """
        return {c.Infra.ORCHESTRATOR_ENV_NO_COLOR: "1"}

    @classmethod
    def _new_file_lint_baseline(
        cls, py_file: Path, workspace: Path, *, gates: t.StrSequence | None = None
    ) -> t.Infra.LintSnapshot:
        """Compute the lint baseline for a new file."""
        py_file.parent.mkdir(parents=True, exist_ok=True)
        py_file.write_text(
            f"{c.Infra.FUTURE_ANNOTATIONS}\n", encoding=c.Cli.ENCODING_DEFAULT
        )
        try:
            return cls.lint_snapshot(py_file, workspace, gates=gates)
        finally:
            if py_file.exists():
                py_file.unlink()

    _snapshot_cache: ClassVar[
        MutableMapping[tuple[str, str, t.StrSequence], t.Infra.LintSnapshot]
    ] = {}

    @classmethod
    def clear_snapshot_cache(cls) -> None:
        """Reset the content-hash keyed lint snapshot cache."""
        cls._snapshot_cache.clear()

    @staticmethod
    def _lint_snapshot_cache_key(
        py_file: Path, gate_key: t.StrSequence
    ) -> tuple[str, str, t.StrSequence] | None:
        """Lint snapshot cache key."""
        try:
            raw_bytes = py_file.read_bytes()
        except OSError:
            return None
        return (str(py_file.resolve()), hashlib.sha256(raw_bytes).hexdigest(), gate_key)

    @classmethod
    def _execute_selected_lint_tools(
        cls, py_file: Path, workspace: Path, selected_tools: t.StrSequencePairTuple
    ) -> t.Infra.LintSnapshot:
        """Execute selected lint tools."""
        command_cwd = cls._command_cwd(py_file, workspace)
        command_env = cls._command_env()
        gate_timeout = max(5, min(15, c.Infra.TIMEOUT_SHORT))

        # flext-38p39: every gate is an independent subprocess -- _run_lint_gate
        # builds its own command and returns a value, touching no shared state.
        # Running any one of them ahead of the pool made a snapshot cost that
        # gate's full wall clock PLUS the slowest of the rest, instead of just
        # the slowest. Measured: 8 snapshots x 4 gates at 0.297s each.
        results: list[m.Infra.LintGateResult] = []
        if not selected_tools:
            return cls._lint_snapshot_from_results(())

        timeout_budget = max(1, gate_timeout + 10)
        pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(cls._SNAPSHOT_MAX_WORKERS, len(selected_tools)))
        )
        futures_by_tool = {
            pool.submit(
                cls._run_lint_gate,
                py_file=py_file,
                workspace=workspace,
                command_cwd=command_cwd,
                command_env=command_env,
                gate_timeout=gate_timeout,
                tool_name=tool,
                template=tmpl,
            ): tool
            for tool, tmpl in selected_tools
        }
        try:
            done, not_done = concurrent.futures.wait(
                tuple(futures_by_tool), timeout=timeout_budget
            )
            results.extend(future.result() for future in done)
            for future in not_done:
                tool_name = futures_by_tool[future]
                _ = future.cancel()
                results.append(
                    m.Infra.LintGateResult(
                        tool_name=tool_name,
                        errors=(
                            f"timeout {timeout_budget}s: lint gate '{tool_name}' did not finish",
                        ),
                    )
                )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return cls._lint_snapshot_from_results(tuple(results))

    @classmethod
    def _run_lint_gate(
        cls,
        *,
        py_file: Path,
        workspace: Path,
        command_cwd: Path,
        command_env: t.StrMapping,
        gate_timeout: int,
        tool_name: str,
        template: t.StrSequence,
    ) -> m.Infra.LintGateResult:
        """Run one lint gate and return a validated result model."""
        command: t.StrSequence = (
            *cls._workspace_tool_command(workspace, template[0]),
            *(item.replace("{file}", str(py_file)) for item in template[1:]),
        )
        if (
            tool_name == c.Infra.PYREFLY
            and (project_config := command_cwd / c.Infra.PYPROJECT_FILENAME).is_file()
        ):
            command = (*command, "--config", str(project_config))
        cmd = (
            FlextInfraUtilitiesResourceLimits.mypy_limited_command(command)
            if tool_name == c.Infra.MYPY
            else command
        )
        run_result = u.Cli.run_raw(
            cmd,
            cwd=command_cwd,
            env=command_env,
            remove_env_keys=cls._COMMAND_ENV_REMOVE_KEYS,
            timeout=FlextInfraUtilitiesResourceLimits.mypy_runner_timeout()
            if tool_name == c.Infra.MYPY
            else gate_timeout,
        )
        if run_result.failure:
            error = run_result.error or f"{tool_name} failed"
            gate_errors: t.StrSequence = (
                FlextInfraUtilitiesResourceLimits.mypy_launch_failure_diagnostic(error)
                if tool_name == c.Infra.MYPY
                else error,
            )
        elif run_result.success and run_result.value.exit_code != 0:
            resource_diagnostic = (
                FlextInfraUtilitiesResourceLimits.mypy_failure_diagnostic(
                    run_result.value
                )
                if tool_name == c.Infra.MYPY
                else None
            )
            output = (
                resource_diagnostic
                or (run_result.value.stdout + run_result.value.stderr).strip()
            )
            gate_errors = tuple(line for line in output.splitlines() if line.strip())
        else:
            gate_errors = ()
        return m.Infra.LintGateResult(tool_name=tool_name, errors=gate_errors)

    @staticmethod
    def _lint_snapshot_from_results(
        results: t.SequenceOf[m.Infra.LintGateResult],
    ) -> t.Infra.LintSnapshot:
        """Convert validated lint gate results into the public snapshot contract."""
        return {result.tool_name: result.errors for result in results if result.errors}

    @classmethod
    def lint_snapshot(
        cls, py_file: Path, workspace: Path, *, gates: t.StrSequence | None = None
    ) -> t.Infra.LintSnapshot:
        """Run selected lint tools on *py_file*, concurrent and content-cached."""
        selected_tools = cls._selected_lint_tools(gates)
        if not selected_tools:
            return {}

        gate_key = tuple(tool for tool, _ in selected_tools)
        cache_key = cls._lint_snapshot_cache_key(py_file, gate_key)
        if (
            cache_key is not None
            and (cached := cls._snapshot_cache.get(cache_key)) is not None
        ):
            return cached

        result = cls._execute_selected_lint_tools(
            py_file=py_file, workspace=workspace, selected_tools=selected_tools
        )
        if cache_key is not None:
            cls._snapshot_cache[cache_key] = dict(result)
        return result

    @classmethod
    def lint_snapshots(
        cls,
        paths: t.SequenceOf[Path],
        workspace: Path,
        *,
        gates: t.StrSequence | None = None,
    ) -> MutableMapping[Path, t.Infra.LintSnapshot]:
        """Run lint snapshots for multiple files concurrently."""
        ordered_paths = tuple(paths)
        if not ordered_paths:
            return {}
        if len(ordered_paths) == 1:
            path = ordered_paths[0]
            return {path: cls.lint_snapshot(path, workspace, gates=gates)}

        snapshots_by_path: MutableMapping[Path, t.Infra.LintSnapshot] = {}
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(cls._SNAPSHOT_MAX_WORKERS, len(ordered_paths)))
        ) as pool:
            futures_by_path = {
                pool.submit(cls.lint_snapshot, path, workspace, gates=gates): path
                for path in ordered_paths
            }
            for future in concurrent.futures.as_completed(futures_by_path):
                snapshots_by_path[futures_by_path[future]] = future.result()

        return {path: snapshots_by_path[path] for path in ordered_paths}

    @staticmethod
    def lint_new_errors(
        before: t.Infra.LintSnapshot, after: t.Infra.LintSnapshot
    ) -> t.Infra.LintSnapshot:
        """Return only lint errors introduced relative to *before*."""
        return {
            tool: added
            for tool, lines in after.items()
            if (
                added := [
                    line
                    for line in lines
                    if (
                        normalized
                        := FlextInfraUtilitiesProtectedEditLinting._normalize_lint_line(
                            line
                        )
                    )
                    and normalized
                    not in {
                        FlextInfraUtilitiesProtectedEditLinting._normalize_lint_line(
                            item
                        )
                        for item in before.get(tool, [])
                        if FlextInfraUtilitiesProtectedEditLinting._normalize_lint_line(
                            item
                        )
                    }
                ]
            )
        }

    @staticmethod
    def preview_source_lint(
        py_file: Path,
        workspace: Path,
        *,
        updated_source: str,
        gates: t.StrSequence | None = None,
    ) -> tuple[t.Infra.LintSnapshot, t.Infra.LintSnapshot]:
        """Preview lint output for ``updated_source`` while restoring the file."""
        original_source = py_file.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        before = FlextInfraUtilitiesProtectedEditLinting.lint_snapshot(
            py_file, workspace, gates=gates
        )
        if updated_source == original_source:
            return before, before
        py_file.write_text(updated_source, encoding=c.Cli.ENCODING_DEFAULT)
        try:
            after = FlextInfraUtilitiesProtectedEditLinting.lint_snapshot(
                py_file, workspace, gates=gates
            )
        finally:
            py_file.write_text(original_source, encoding=c.Cli.ENCODING_DEFAULT)
        return before, after


__all__: list[str] = ["FlextInfraUtilitiesProtectedEditLinting"]
