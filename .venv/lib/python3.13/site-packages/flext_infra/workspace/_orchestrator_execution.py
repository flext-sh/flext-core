"""Workspace orchestration execution behavior for CLI services.

Executes per-project make calls, progress reporting, and error summarization.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, m, t, u

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraWorkspaceOrchestratorExecutionMixin:
    """Project orchestration execution logic."""

    @staticmethod
    def _project_child_env() -> t.StrMapping:
        """Return child process env overrides for project make execution."""
        inherited = u.Cli.process_env()
        path = inherited.get(c.Infra.ORCHESTRATOR_ENV_PATH, "")
        blocked_path_entries = frozenset(
            entry
            for entry in (inherited.get(c.Infra.ORCHESTRATOR_ENV_MISE_SHIMS, ""),)
            if entry
        )
        path_entries = tuple(
            entry
            for entry in path.split(c.Infra.ORCHESTRATOR_ENV_PATH_SEPARATOR)
            if entry and entry not in blocked_path_entries
        )
        env: dict[str, str] = {c.Infra.ORCHESTRATOR_ENV_NO_COLOR: "1"}
        if path_entries:
            env[c.Infra.ORCHESTRATOR_ENV_PATH] = (
                c.Infra.ORCHESTRATOR_ENV_PATH_SEPARATOR.join(path_entries)
            )
        return env

    def _execute_project(
        self, project: str, verb: str, idx: int, *, make_args: t.StrSequence
    ) -> t.Pair[p.Cli.CommandOutput, bool]:
        """Run one project and return ``(output, succeeded)``."""
        output_result = self._run_project(project, verb, idx, make_args=make_args)
        if output_result.failure:
            return (
                m.Cli.CommandOutput(
                    stdout="",
                    stderr=output_result.error or "project execution failed",
                    exit_code=1,
                    duration=0.0,
                ),
                False,
            )
        cmd_output: p.Cli.CommandOutput = output_result.value
        return (cmd_output, cmd_output.exit_code == 0)

    @staticmethod
    def _exit_classification(exit_code: int) -> str:
        """Name the process outcome behind a non-zero exit status."""
        if exit_code == c.Infra.PROCESS_TIMEOUT_EXIT_CODE:
            return " timeout"
        if exit_code > c.Infra.PROCESS_SIGNAL_EXIT_OFFSET:
            return f" signal={exit_code - c.Infra.PROCESS_SIGNAL_EXIT_OFFSET}"
        return ""

    @staticmethod
    def _collect_failures(
        projects: t.StrSequence, results: t.SequenceOf[p.Cli.CommandOutput]
    ) -> t.SequenceOf[t.Triple[str, int, Path]]:
        """Collect failing projects with parsed error counters."""
        failures: t.MutableSequenceOf[t.Triple[str, int, Path]] = []
        for proj_name, cmd_result in zip(projects, results, strict=False):
            if cmd_result.exit_code != 0:
                log_file = (
                    Path(cmd_result.stdout)
                    if cmd_result.stdout
                    else Path(f"{proj_name}.log")
                )
                err_count, _ = u.Infra.extract_errors(log_file)
                failures.append((proj_name, err_count, log_file))
        return failures

    @staticmethod
    def _project_log_filename(project: str) -> str:
        """Return a report filename for project targets that may contain path parts."""
        project_path = Path(project)
        excluded_parts = frozenset({project_path.anchor, ".", "..", ""})
        safe_stem = "__".join(
            part for part in project_path.parts if part not in excluded_parts
        )
        return f"{safe_stem or 'workspace'}.log"

    def orchestrate(
        self,
        projects: t.StrSequence,
        verb: str,
        *,
        fail_fast: bool = False,
        make_args: t.StrSequence = (),
    ) -> p.Result[t.SequenceOf[p.Cli.CommandOutput]]:
        """Execute ``make <verb>`` across projects and return collected outputs."""
        u.Cli.header("Workspace Orchestration")
        try:
            return self._orchestrate_checked(
                projects, verb, fail_fast=fail_fast, make_args=make_args
            )
        except c.EXC_OS_RUNTIME_TYPE as exc:
            # flext-wkii.17 (Codex): keep type-only protocols out of runtime factories.
            return r.fail_op("Orchestration", exc)

    @staticmethod
    def _failure_summary(
        verb: str, failures: t.SequenceOf[t.Triple[str, int, Path]]
    ) -> None:
        """Print compact failure summary for orchestrated projects."""
        if not failures:
            return
        u.Cli.error(f"{verb} failures: {len(failures)} project(s)")
        for project, error_count, log_path in failures:
            u.Cli.error(f"- {project}: {error_count} errors ({log_path})")

    def _orchestrate_checked(
        self,
        projects: t.StrSequence,
        verb: str,
        *,
        fail_fast: bool,
        make_args: t.StrSequence,
    ) -> p.Result[t.SequenceOf[p.Cli.CommandOutput]]:
        """Execute a validated orchestration run with progress accounting."""
        allowed_verbs = c.Infra.ORCHESTRATED_PROJECT_VERBS
        if verb not in allowed_verbs:
            allowed = ", ".join(allowed_verbs)
            return r.fail(f"unsupported orchestrate verb '{verb}' (allowed: {allowed})")
        effective_make_args = self._normalize_fail_fast_make_args(
            make_args, fail_fast=fail_fast
        )
        results: t.MutableSequenceOf[p.Cli.CommandOutput] = []
        total = len(projects)
        success = 0
        failed = 0
        skipped = 0
        started_total = time.monotonic()
        # flext-9v0d: emit a deterministic, machine-parseable orchestration report
        # so a caller can attribute every project outcome and the child exit code.
        u.Cli.emit_raw(
            f"scope={c.Infra.RK_WORKSPACE} verb={verb} "
            f"projects={','.join(projects)}"
            + (
                f" gates={self._gates_of(make_args)}"
                if self._gates_of(make_args)
                else ""
            )
            + "\n"
        )
        for idx, project in enumerate(projects, start=1):
            u.Cli.emit_raw(f"[{idx}/{total}] START {project} {verb}\n")
            cmd_output, succeeded = self._execute_project(
                project, verb, idx, make_args=effective_make_args
            )
            results.append(cmd_output)
            state = "PASS" if succeeded else "FAIL"
            u.Cli.emit_raw(
                f"[{idx}/{total}] {state} {project} {verb} "
                f"exit={cmd_output.exit_code} duration={cmd_output.duration:.2f}s\n"
            )
            if succeeded:
                success += 1
            else:
                failed += 1
                if fail_fast:
                    skipped = total - idx
                    break
        elapsed_total = time.monotonic() - started_total
        failed_project = next(
            (
                project
                for project, output in zip(projects, results, strict=False)
                if output.exit_code != 0
            ),
            "",
        )
        exit_code = next(
            (output.exit_code for output in results if output.exit_code != 0), 0
        )
        u.Cli.emit_raw(
            f"summary scope={c.Infra.RK_WORKSPACE} verb={verb} total={total} "
            f"passed={success} failed={failed} skipped={skipped} exit={exit_code}\n"
        )
        _ = elapsed_total
        if failed > 0:
            failures = self._collect_failures(projects, results)
            self._failure_summary(verb, failures)
            return r.fail(
                f"orchestration completed with failures: {failed} "
                f"(first failure {failed_project} exit={exit_code}"
                f"{self._exit_classification(exit_code)})"
            )
        return r.ok(results)

    def _run_project(
        self, project: str, verb: str, _index: int, *, make_args: t.StrSequence
    ) -> p.Result[p.Cli.CommandOutput]:
        """Execute make verb for one project and capture output path/metrics."""
        log_path = u.Cli.resolve_report_path(
            Path.cwd(), c.Infra.RK_WORKSPACE, verb, self._project_log_filename(project)
        )
        _ = u.Cli.ensure_dir(log_path.parent)
        started = time.monotonic()
        proc_result = u.Cli.run_to_file(
            [c.Infra.MAKE, "-C", project, verb, *make_args],
            log_path,
            env=self._project_child_env(),
            remove_env_keys=c.Infra.ORCHESTRATOR_REMOVE_ENV_KEYS,
        )
        process_failure = (
            u.Infra.append_process_failure(
                log_path, proc_result.error or "project process lifecycle failed"
            )
            if proc_result.failure
            else ""
        )
        return_code: int = proc_result.unwrap() if proc_result.success else 1
        # flext-9v0d: GNU make exits 2 for any failed recipe, so recover the
        # child's real exit code from make's own error line in the log.
        if return_code != 0:
            child_code = u.Infra.extract_make_child_exit_code(log_path)
            if child_code is not None:
                return_code = child_code
        stderr = "" if proc_result.success else proc_result.error or ""
        elapsed = time.monotonic() - started
        if return_code == 0:
            u.Cli.info(f"  ✓ {project} completed in {int(elapsed)}s  ({log_path})")
        else:
            error_count, error_lines = u.Infra.extract_errors(log_path)
            displayed_errors = list(error_lines)
            if process_failure and process_failure not in displayed_errors:
                displayed_errors.insert(0, process_failure)
                error_count = max(error_count, len(displayed_errors))
            u.Cli.project_failure(
                m.Infra.ProjectFailureInfo(
                    project=project,
                    elapsed=elapsed,
                    log_path=log_path,
                    error_count=error_count,
                    errors=displayed_errors,
                )
            )
            if displayed_errors:
                stderr = "\n".join(displayed_errors)
        return r[m.Cli.CommandOutput].ok(
            m.Cli.CommandOutput(
                stdout=str(log_path),
                stderr=stderr,
                exit_code=return_code,
                duration=round(elapsed, 2),
            )
        )

    @staticmethod
    def _normalize_fail_fast_make_args(
        make_args: t.StrSequence, *, fail_fast: bool
    ) -> t.StrSequence:
        """Propagate fail-fast intent to make command invocation."""
        if not fail_fast:
            return make_args
        if any(make_arg.startswith("FAIL_FAST=") for make_arg in make_args):
            return make_args
        return (*make_args, "FAIL_FAST=1")

    @staticmethod
    def _gates_of(make_args: t.StrSequence) -> str:
        """Return the gate selection carried by make arguments, if declared."""
        prefix: str = f"{c.Infra.CHECK_GATES_VARIABLE}="
        for make_arg in make_args:
            if make_arg.startswith(prefix):
                gates: str = make_arg[len(prefix) :]
                return gates
        return ""


__all__: list[str] = ["FlextInfraWorkspaceOrchestratorExecutionMixin"]
