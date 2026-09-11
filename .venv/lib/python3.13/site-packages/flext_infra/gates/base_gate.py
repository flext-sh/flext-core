"""Shared gate template abstraction for workspace quality checks."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c, m, u

if TYPE_CHECKING:
    from flext_infra import p, t


class FlextInfraGate:
    """Abstract template implementing common check/fix execution flow for gates."""

    gate_id: ClassVar[str] = ""
    gate_name: ClassVar[str] = ""
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = ""
    tool_url: ClassVar[str] = ""

    def __init__(
        self, repository_root: Path, *, runner: p.Cli.CommandRunner | None = None
    ) -> None:
        """Bind repository root and optional command runner override."""
        self._repository_root = repository_root
        self._runner = runner

    @staticmethod
    def _python_module_command(module: str, *args: str) -> t.StrSequence:
        """Canonical venv-anchored tool invocation.

        Every linter/type-checker runs through the workspace interpreter
        (``sys.executable -m <module>``) so tool resolution is bound to the
        active ``.venv`` and never depends on ``PATH`` ordering or an external
        mise/system shim. This is the single source for building a Python
        module command shared by all gates.
        """
        return (sys.executable, "-m", module, *args)

    @staticmethod
    def _python_console_script_command(tool: str, *args: str) -> t.StrSequence:
        """Invoke a uv-managed console script from the active interpreter directory."""
        return (str(Path(sys.executable).with_name(tool)), *args)

    # ------------------------------------------------------------------
    # Template method: check
    # ------------------------------------------------------------------

    def check(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> m.Infra.GateExecution:
        """Template method: timing + dirs + skip + run + parse + result."""
        started = time.monotonic()
        check_dirs = self._get_check_dirs(project_dir, ctx)
        if not check_dirs:
            return self._skip_result(project_dir, started)
        return self._execute_check_command(project_dir, ctx, check_dirs, started)

    def check_files(
        self, files: t.SequenceOf[Path], project_dir: Path, ctx: m.Infra.GateContext
    ) -> m.Infra.GateExecution:
        """Check specific files instead of whole directory.

        Passes file paths directly to the tool CLI for scoped validation.
        Falls back to directory check if no files provided.
        """
        if not files:
            return self.check(project_dir, ctx)
        started = time.monotonic()
        file_strs = [str(f.relative_to(project_dir)) for f in files if f.exists()]
        if not file_strs:
            return self._skip_result(project_dir, started)
        return self._execute_check_command(project_dir, ctx, file_strs, started)

    def _execute_check_command(
        self,
        project_dir: Path,
        ctx: m.Infra.GateContext,
        targets: t.StrSequence,
        started: float,
    ) -> m.Infra.GateExecution:
        """Build, run, and parse the check command — shared by ``check`` and ``check_files``."""
        cmd = self._build_check_command(project_dir, ctx, targets)
        result = self._run(
            cmd,
            project_dir,
            timeout=self._check_timeout(project_dir, ctx),
            env=self._check_env(project_dir, ctx),
        )
        passed, issues = self._parse_check_output(result, project_dir, ctx)
        return self._build_check_gate_execution(
            project_dir,
            passed=passed,
            issues=issues,
            raw_output=self._raw_output(result),
            started=started,
            ctx=ctx,
        )

    def _build_check_gate_execution(
        self,
        project_dir: Path,
        *,
        passed: bool,
        issues: t.SequenceOf[m.Infra.Issue],
        raw_output: str,
        started: float,
        ctx: m.Infra.GateContext | None = None,
        errors: t.StrSequence | None = None,
    ) -> m.Infra.GateExecution:
        """Assemble a gate execution from parsed check output.

        When ``ctx.gate_mode == "warn"`` the gate reports issues but is
        marked passed so advisory enforcement gates do not fail the check
        pipeline. ``errors`` overrides the default issue-derived report
        lines (fix paths report applied changes there).
        """
        if ctx is not None and getattr(ctx, "gate_mode", None) == "warn" and not passed:
            warn_issues = [
                issue.model_copy(update={"severity": "WARNING"})
                if hasattr(issue, "model_copy")
                else issue
                for issue in issues
            ]
            return m.Infra.GateExecution(
                result=m.Infra.GateResult(
                    gate=self.gate_id,
                    project=project_dir.name,
                    passed=True,
                    errors=[],
                    duration=round(time.monotonic() - started, 3),
                ),
                issues=tuple(warn_issues),
                raw_output=raw_output,
            )
        return m.Infra.GateExecution(
            result=m.Infra.GateResult(
                gate=self.gate_id,
                project=project_dir.name,
                passed=passed,
                errors=(
                    list(errors)
                    if errors is not None
                    else [issue.formatted for issue in issues]
                ),
                duration=round(time.monotonic() - started, 3),
            ),
            issues=tuple(issues),
            raw_output=raw_output,
        )

    def _build_project_error_gate_result(
        self,
        project_dir: Path,
        *,
        passed: bool,
        errors: t.SequenceOf[str],
        started: float,
        ctx: m.Infra.GateContext,
    ) -> m.Infra.GateExecution:
        """Build a gate result from project-level error strings (no per-file issues)."""
        issues = [
            m.Infra.Issue(
                file=str(project_dir),
                line=1,
                column=1,
                code=self.gate_id,
                message=error,
                severity="ERROR",
            )
            for error in errors
        ]
        return self._build_check_gate_execution(
            project_dir,
            passed=passed,
            issues=issues,
            raw_output="\n".join(errors),
            started=started,
            ctx=ctx,
        )

    def _build_single_issue_result(
        self,
        project_dir: Path,
        file_path: Path,
        message: str,
        *,
        passed: bool,
        started: float,
        ctx: m.Infra.GateContext,
    ) -> m.Infra.GateExecution:
        """Build a gate execution from a single issue (scan-failure / fix-failure)."""
        issue = m.Infra.Issue(
            file=str(file_path),
            line=1,
            column=1,
            code=self.gate_id,
            message=message,
            severity="ERROR",
        )
        return self._build_check_gate_execution(
            project_dir,
            passed=passed,
            issues=[issue],
            raw_output=issue.message,
            started=started,
            ctx=ctx,
        )

    # ------------------------------------------------------------------
    # Template hooks — subclasses override these
    # ------------------------------------------------------------------

    def _get_check_dirs(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> t.StrSequence:
        """Return directories to check. Default: discover + filter for .py files."""
        _ = ctx
        return self._dirs_with_py(project_dir, self._existing_check_dirs(project_dir))

    def _build_check_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, check_dirs: t.StrSequence
    ) -> t.StrSequence:
        """Build the tool CLI command. Default: none (check overridden directly)."""
        _ = project_dir, ctx, check_dirs
        return []

    def _parse_check_output(
        self, result: p.Cli.CommandOutput, project_dir: Path, ctx: m.Infra.GateContext
    ) -> tuple[bool, t.SequenceOf[m.Infra.Issue]]:
        """Parse tool output into (passed, issues). Default: no-op (check overridden)."""
        _ = result, project_dir, ctx
        return True, ()

    def _check_timeout(self, project_dir: Path, ctx: m.Infra.GateContext) -> int:
        """Timeout for the check command. Override for long-running tools."""
        _ = project_dir, ctx
        timeout: int = c.Infra.TIMEOUT_DEFAULT
        return timeout

    def _check_env(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> t.StrMapping | None:
        """Return a custom environment for the check command. Default: None (inherit)."""
        _ = project_dir, ctx
        return None

    # ------------------------------------------------------------------
    # Template method: fix
    # ------------------------------------------------------------------

    def fix(self, project_dir: Path, ctx: m.Infra.GateContext) -> m.Infra.GateExecution:
        """Template method: timing + targets + skip + run fix + result."""
        if ctx.check_only or not ctx.apply_fixes:
            return self._check_only_fix_result(project_dir)
        if not self.can_fix:
            return self._build_check_gate_execution(
                project_dir,
                passed=True,
                issues=(),
                raw_output=f"Gate {self.gate_id} does not support fix",
                started=time.monotonic(),
            )
        started = time.monotonic()
        targets = self._get_fix_targets(project_dir, ctx)
        if not targets:
            return self._skip_result(project_dir, started)
        cmd = self._build_fix_command(project_dir, ctx, targets)
        result = self._run(cmd, project_dir)
        return self._build_check_gate_execution(
            project_dir,
            passed=result.exit_code == 0,
            issues=(),
            raw_output=self._fix_raw_output(result),
            started=started,
        )

    def _check_only_fix_result(self, project_dir: Path) -> m.Infra.GateExecution:
        """Return a non-mutating fix preview for check-only gate contexts."""
        return self._build_check_gate_execution(
            project_dir,
            passed=True,
            issues=(),
            raw_output=f"Gate {self.gate_id} fix preview only; no files written",
            started=time.monotonic(),
        )

    # ------------------------------------------------------------------
    # Fix hooks — subclasses override these
    # ------------------------------------------------------------------

    def _get_fix_targets(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> t.StrSequence:
        """Targets for fix. Default: same as check dirs."""
        return self._get_check_dirs(project_dir, ctx)

    def _build_fix_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, targets: t.StrSequence
    ) -> t.StrSequence:
        """Build the fix CLI command. Must override if can_fix is True."""
        _ = project_dir, ctx, targets
        msg = f"Gate {self.gate_id} set can_fix=True but did not implement _build_fix_command"
        raise NotImplementedError(msg)

    def _fix_raw_output(self, result: p.Cli.CommandOutput) -> str:
        """Assemble raw output from fix result. Default: stderr only."""
        stderr: str = result.stderr
        return stderr

    @staticmethod
    def _raw_output(result: p.Cli.CommandOutput) -> str:
        """Preserve diagnostics regardless of the stream selected by a tool."""
        return "\n".join(output for output in (result.stdout, result.stderr) if output)

    def _run(
        self,
        cmd: t.StrSequence,
        cwd: Path,
        timeout: int = c.Infra.TIMEOUT_DEFAULT,
        env: t.StrMapping | None = None,
    ) -> p.Cli.CommandOutput:
        """Run."""
        runner = self._runner or u.Cli
        result = runner.run_raw(cmd, cwd=cwd, timeout=timeout, env=env)
        if result.failure:
            return m.Cli.CommandOutput(
                stdout="",
                stderr=result.error or "command execution failed",
                exit_code=1,
            )
        return result.value

    def _existing_check_dirs(self, project_dir: Path) -> t.StrSequence:
        """Return direct project-owned source directories only."""
        candidates = ("src", "tests")
        return self._dirs_with_py(project_dir, candidates)

    @staticmethod
    def _dirs_with_py(project_dir: Path, dirs: t.StrSequence) -> t.StrSequence:
        """Dirs with py."""
        out: t.MutableSequenceOf[str] = []
        for directory in dirs:
            path = project_dir / directory
            if not path.is_dir():
                continue
            if next(path.rglob(c.Infra.EXT_PYTHON_GLOB), None) or next(
                path.rglob("*.pyi"), None
            ):
                out.append(directory)
        return out

    def _skip_result(self, project_dir: Path, started: float) -> m.Infra.GateExecution:
        """Skip result."""
        return self._build_check_gate_execution(
            project_dir, passed=True, issues=(), raw_output="", started=started
        )


class FlextInfraScannerGateMixin(FlextInfraGate):
    """Mixin for gates that detect per-file issues via a rope-backed scanner.

    Subclasses provide ``scan_error_message`` and implement
    ``_detect_file_issues``.  The shared ``check`` method handles file
    discovery, rope-project lifecycle, and result assembly.
    """

    scan_error_message: ClassVar[str] = ""

    @override
    def check(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> m.Infra.GateExecution:
        """Scan all Python files in ``project_dir`` and report detected issues."""
        _ = ctx
        started = time.monotonic()
        files_result = u.Infra.iter_python_files(
            m.Infra.SourceScanRequest(project_roots=(project_dir,))
        )
        if files_result.failure:
            return self._build_single_issue_result(
                project_dir,
                Path(c.Infra.PYPROJECT_FILENAME),
                files_result.error or self.scan_error_message,
                passed=False,
                started=started,
                ctx=ctx,
            )
        rope_project = u.Infra.init_rope_project(project_dir)
        try:
            issues = [
                issue
                for file_path in files_result.value
                for issue in self._detect_file_issues(
                    file_path, project_dir, rope_project
                )
            ]
        finally:
            rope_project.close()
        return self._build_check_gate_execution(
            project_dir,
            passed=len(issues) == 0,
            issues=issues,
            raw_output="\n".join(issue.formatted for issue in issues),
            started=started,
            ctx=ctx,
        )

    def _detect_file_issues(
        self, file_path: Path, project_dir: Path, rope_project: t.Infra.RopeProject
    ) -> t.SequenceOf[m.Infra.Issue]:
        """Override in subclass to detect issues for a single file."""
        _ = file_path, project_dir, rope_project
        return ()


__all__: list[str] = ["FlextInfraGate", "FlextInfraScannerGateMixin"]
