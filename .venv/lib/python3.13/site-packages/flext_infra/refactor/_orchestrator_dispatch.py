"""Refactor orchestration dispatch mixin (CLI/file dispatch + reporting)."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING

from flext_infra import m, u
from flext_infra.refactor.violation_analyzer import FlextInfraRefactorViolationAnalyzer

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import c, p, t
    from flext_infra.refactor.loader import FlextInfraRefactorRuleLoader


class FlextInfraRefactorOrchestratorDispatchMixin:
    """Provide result builders, reporting helpers, and CLI dispatch entry points."""

    if TYPE_CHECKING:
        loader: FlextInfraRefactorRuleLoader

        def refactor_file(
            self,
            file_path: Path,
            *,
            dry_run: bool = False,
            gates: t.StrSequence | None = None,
        ) -> m.Infra.Result: ...

        def refactor_files(
            self,
            file_paths: t.SequenceOf[Path],
            *,
            dry_run: bool = False,
            gates: t.StrSequence | None = None,
        ) -> t.SequenceOf[m.Infra.Result]: ...

        def refactor_project(
            self,
            project_path: Path,
            *,
            dry_run: bool = False,
            pattern: str = c.Infra.EXT_PYTHON_GLOB,
            apply_safety: bool = True,
            gates: t.StrSequence | None = None,
        ) -> t.SequenceOf[m.Infra.Result]: ...

        def refactor_workspace(
            self,
            repository_root: Path,
            *,
            dry_run: bool = False,
            pattern: str = c.Infra.EXT_PYTHON_GLOB,
            apply_safety: bool = True,
            gates: t.StrSequence | None = None,
        ) -> t.SequenceOf[m.Infra.Result]: ...

    @staticmethod
    def _error_result(fp: Path, error: str) -> m.Infra.Result:
        """Build a failure result."""
        return m.Infra.Result(
            file_path=fp,
            success=False,
            modified=False,
            error=error,
            changes=[],
            refactored_code=None,
        )

    @staticmethod
    def _skip_result(fp: Path) -> m.Infra.Result:
        """Build a skip result for non-Python files."""
        return m.Infra.Result(
            file_path=fp,
            success=True,
            modified=False,
            changes=["Skipped non-Python file"],
            refactored_code=None,
        )

    @staticmethod
    def _refactor_debug(message: str) -> None:
        """Emit one compact refactor debug line."""
        u.Cli.info(message)

    @staticmethod
    def _refactor_header(message: str) -> None:
        """Emit one refactor section header."""
        u.Cli.header(message)

    @staticmethod
    def _print_violation_summary(analysis: m.Infra.ViolationAnalysisReport) -> None:
        """Print high-level violation analysis summary."""
        u.Cli.header("Violation Analysis")
        u.Cli.info(f"Files scanned: {analysis.files_scanned}")
        for key, value in sorted(analysis.totals.items()):
            u.Cli.info(f"- {key}: {value}")

    @staticmethod
    def _print_diff(original: str, updated: str, file_path: Path) -> None:
        """Print unified diff for one updated file."""
        diff = difflib.unified_diff(
            original.splitlines(),
            updated.splitlines(),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
        for line in diff:
            u.Cli.info(line)

    @staticmethod
    def _print_summary(results: t.SequenceOf[m.Infra.Result], *, dry_run: bool) -> None:
        """Print refactor execution summary."""
        total = len(results)
        success = sum(1 for item in results if item.success)
        failed = total - success
        modified = sum(1 for item in results if item.modified)
        mode = "[DRY-RUN] " if dry_run else ""
        u.Cli.info(f"{mode}Processed: {total} file(s)")
        u.Cli.info(f"Success: {success}  Failed: {failed}  Modified: {modified}")

    def run_analyze_violations(self, args: p.Infra.RefactorCliArgs) -> int:
        """Analyze violations across the selected file set."""
        files = self._collect_files(args)
        if files is None:
            return 1
        analysis = FlextInfraRefactorViolationAnalyzer.analyze_files(files)
        self._print_violation_summary(analysis)
        if args.analysis_output is not None:
            _ = u.Cli.json_write(args.analysis_output, analysis.model_dump(mode="json"))
            u.Cli.info(f"Analysis report written: {args.analysis_output}")
        return 0

    def _collect_files(
        self, args: p.Infra.RefactorCliArgs
    ) -> t.MutableSequenceOf[Path] | None:
        """Collect files."""
        result: t.MutableSequenceOf[Path] | None
        if args.project:
            collected = u.Infra.collect_refactor_project_files(
                self.loader.settings, args.project, pattern=args.pattern
            )
            result = None if collected is None else list(collected)
        elif args.workspace:
            result = list(
                u.Infra.collect_refactor_workspace_files(
                    self.loader.settings, args.workspace, pattern=args.pattern
                )
            )
        elif args.file:
            if not args.file.exists():
                u.Cli.error(f"File not found: {args.file}")
                result = None
            else:
                result = [args.file]
        elif args.files:
            result = [path for path in args.files if path.exists()]
        else:
            result = []
        return result

    def run_refactor(self, args: p.Infra.RefactorCliArgs) -> int:
        """Run refactor CLI dispatch for the selected scope."""
        if args.project:
            results = list(
                self.refactor_project(
                    args.project, dry_run=args.dry_run, pattern=args.pattern
                )
            )
        elif args.workspace:
            results = list(
                self.refactor_workspace(
                    args.workspace, dry_run=args.dry_run, pattern=args.pattern
                )
            )
        elif args.file:
            if not args.file.exists():
                u.Cli.error(f"File not found: {args.file}")
                return 1
            read = u.Cli.files_read_text(args.file)
            if read.failure:
                u.Cli.error(read.error or f"failed to read {args.file}")
                return 1
            original = read.value
            result = self.refactor_file(args.file, dry_run=args.dry_run)
            if args.show_diff and result.modified:
                self._print_diff(
                    original, result.refactored_code or original, args.file
                )
            results = [result]
        elif args.files:
            existing = [path for path in args.files if path.exists()]
            for path in args.files:
                if not path.exists():
                    u.Cli.error(f"File not found: {path}")
            results = list(self.refactor_files(existing, dry_run=args.dry_run))
        else:
            results = list[m.Infra.Result]()
        self._print_summary(results, dry_run=args.dry_run)
        if args.impact_map_output is not None:
            _ = u.Infra.write_impact_map(results, args.impact_map_output)
        return 0 if u.count(results, lambda item: not item.success) == 0 else 1


__all__: list[str] = ["FlextInfraRefactorOrchestratorDispatchMixin"]
