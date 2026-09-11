"""Copy-on-write safety primitives for file transformation with automatic rollback.

Provides static helpers for backing up files before transformation,
validating post-transform quality via gates, and restoring on failure.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from flext_cli import u
from flext_core import r
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.protocols import p
from flext_infra.typings import t


class FlextInfraUtilitiesSafety:
    """Static safety helpers for copy-on-write file protection."""

    @staticmethod
    def create_checkpoint(repo: Path, *, label: str = "checkpoint") -> p.Result[str]:
        """Validate that a repository is clean before file-scoped protection.

        Returns an empty string for non-repositories or clean trees.
        """
        result: p.Result[str]
        checkpoint_label = label.strip() or "checkpoint"
        repo_check = u.Cli.run_raw(
            [c.Infra.GIT, "rev-parse", "--show-toplevel"], cwd=repo
        )
        if (
            repo_check.failure
            or repo_check.value.exit_code != 0
            or Path(repo_check.value.stdout.strip()).resolve() != repo.resolve()
        ):
            result = r[str].ok("")
        else:
            status_result = u.Cli.run_raw(
                [c.Infra.GIT, "status", "--porcelain"], cwd=repo
            )
            if status_result.failure or status_result.value.exit_code != 0:
                result = r[str].fail(status_result.error or "git status failed")
            elif not status_result.value.stdout.strip():
                result = r[str].ok("")
            else:
                result = r[str].fail(
                    "dirty git worktree cannot be checkpointed automatically "
                    f"({checkpoint_label}); use file-scoped backup APIs"
                )
        return result

    @staticmethod
    def rollback_to_checkpoint(repo: Path, checkpoint: str = "") -> p.Result[bool]:
        """Reject repository-wide rollback; callers must use file-scoped backups.

        Returns success for non-repositories or empty checkpoint strings.
        """
        if not checkpoint:
            return r[bool].ok(True)
        repo_check = u.Cli.run_raw(
            [c.Infra.GIT, "rev-parse", "--show-toplevel"], cwd=repo
        )
        if (
            repo_check.failure
            or repo_check.value.exit_code != 0
            or Path(repo_check.value.stdout.strip()).resolve() != repo.resolve()
        ):
            return r[bool].ok(True)
        return r[bool].fail(
            "repository-wide checkpoint rollback is unsupported; "
            "use file-scoped backup APIs"
        )

    @staticmethod
    def backup_files(files: t.SequenceOf[Path]) -> t.SequenceOf[Path]:
        """Copy each file to .bak. Fail fast on any error."""
        bak_paths: list[Path] = []
        for file_path in files:
            if not file_path.exists():
                continue
            bak = file_path.with_suffix(
                file_path.suffix + c.Infra.SAFE_EXECUTION_BAK_SUFFIX
            )
            shutil.copy2(file_path, bak)
            bak_paths.append(bak)
        return bak_paths

    @staticmethod
    def restore_files(bak_paths: t.SequenceOf[Path]) -> None:
        """Move .bak files back to originals. Fail fast whenever possible."""
        for bak in bak_paths:
            if not bak.exists():
                continue
            original = Path(str(bak).removesuffix(c.Infra.SAFE_EXECUTION_BAK_SUFFIX))
            shutil.move(str(bak), str(original))

    @staticmethod
    def cleanup_backups(bak_paths: t.SequenceOf[Path]) -> None:
        """Remove .bak files after successful validation."""
        for bak in bak_paths:
            if bak.exists():
                bak.unlink()

    @staticmethod
    def execute_safely(
        files: t.SequenceOf[Path],
        transform: Callable[[t.SequenceOf[Path]], r[t.SequenceOf[Path]]],
        validate: Callable[[t.SequenceOf[Path]], r[bool]],
        *,
        mode: c.Infra.ExecutionMode = c.Infra.ExecutionMode.APPLY_SAFE,
    ) -> m.Infra.SafeExecutionResult:
        """Pipeline: backup -> transform -> validate -> (cleanup | rollback).

        Fail fast: any step failure = immediate rollback + raise.
        """
        file_strs = [str(f) for f in files]

        if mode == c.Infra.ExecutionMode.DRY_RUN:
            return m.Infra.SafeExecutionResult(
                mode=mode, files_backed_up=file_strs, gate_results=[], rolled_back=False
            )

        bak_paths = FlextInfraUtilitiesSafety.backup_files(files)

        transform_result = transform(files)
        if transform_result.failure:
            FlextInfraUtilitiesSafety.restore_files(bak_paths)
            return m.Infra.SafeExecutionResult(
                mode=mode,
                files_backed_up=file_strs,
                gate_results=[transform_result.error or "transform failed"],
                rolled_back=True,
            )

        if mode == c.Infra.ExecutionMode.APPLY_FORCE:
            FlextInfraUtilitiesSafety.cleanup_backups(bak_paths)
            return m.Infra.SafeExecutionResult(
                mode=mode, files_backed_up=file_strs, gate_results=[], rolled_back=False
            )

        validate_result = validate(files)
        if validate_result.failure:
            FlextInfraUtilitiesSafety.restore_files(bak_paths)
            return m.Infra.SafeExecutionResult(
                mode=mode,
                files_backed_up=file_strs,
                gate_results=[validate_result.error or "validation failed"],
                rolled_back=True,
            )

        FlextInfraUtilitiesSafety.cleanup_backups(bak_paths)
        return m.Infra.SafeExecutionResult(
            mode=mode, files_backed_up=file_strs, gate_results=[], rolled_back=False
        )


__all__: list[str] = ["FlextInfraUtilitiesSafety"]
