"""Safety management for refactor operations: checkpoints, rollback, and validation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import overload

from flext_core import r
from flext_core.utilities import FlextUtilities
from flext_infra import (
    FlextInfraCommandRunner,
    FlextInfraGitService,
    FlextInfraJsonService,
    c,
    m,
    p,
)


def _now_iso() -> str:
    """Generate ISO timestamp; delegates to ``u.generate_iso_timestamp()``."""
    return FlextUtilities.generate_iso_timestamp()


class FlextInfraRefactorSafetyManager:
    """Orchestrate pre-/post-transform safety: stash, validate, rollback."""

    def __init__(
        self,
        runner: p.Infra.SafetyRunner | None = None,
        checkpoint_path: Path | None = None,
        test_command: list[str] | None = None,
    ) -> None:
        """Initialize safety manager with runner, checkpoint path, and test command."""
        effective_runner = runner or FlextInfraCommandRunner()
        self._runner = effective_runner
        self._git = FlextInfraGitService(effective_runner)
        self._checkpoint_path = checkpoint_path or Path(
            ".sisyphus/refactor/safety-checkpoint.json"
        )
        self._test_command = test_command or [
            c.Infra.Toml.PYTHON,
            "-m",
            c.Infra.Toml.PYTEST,
            "-q",
        ]
        self._emergency_stop_reason = ""
        self._last_workspace_root: Path | None = None

    def create_checkpoint(self, project_root: Path) -> str:
        """Stash current state and return the stash reference."""
        self._last_workspace_root = project_root
        stash_result = self.create_pre_transformation_stash(project_root)
        if stash_result.is_failure:
            self.request_emergency_stop(
                stash_result.error or "checkpoint creation failed"
            )
            return ""
        return stash_result.value

    def validate_transform(self, files_changed: list[Path]) -> bool:
        """Run semantic validation after a transformation batch."""
        workspace_root = self._resolve_workspace_root(files_changed)
        if workspace_root is None:
            self.request_emergency_stop(
                "unable to resolve workspace root for validation"
            )
            return False

        validation_result = self.run_semantic_validation(workspace_root)
        if validation_result.is_failure:
            self.request_emergency_stop(
                validation_result.error or "transform validation failed"
            )
            return False
        return True

    def emergency_stop(self, reason: str) -> None:
        """Trigger an emergency stop with the given reason."""
        self.request_emergency_stop(reason)

    def request_emergency_stop(self, reason: str) -> None:
        """Record an emergency stop reason for later inspection."""
        self._emergency_stop_reason = reason.strip() or "unspecified emergency stop"

    def clear_emergency_stop(self) -> None:
        """Clear any previously recorded emergency stop."""
        self._emergency_stop_reason = ""

    def is_emergency_stop_requested(self) -> bool:
        """Return True if an emergency stop has been requested."""
        return bool(self._emergency_stop_reason)

    def ensure_can_continue(self) -> r[bool]:
        """Fail if an emergency stop is active; succeed otherwise."""
        if self._emergency_stop_reason:
            return r[bool].fail(
                f"Emergency stop requested: {self._emergency_stop_reason}",
            )
        return r[bool].ok(True)

    def is_git_repository(self, workspace_root: Path) -> bool:
        """Check whether *workspace_root* sits inside a Git work-tree."""
        return self._git.is_repo(workspace_root)

    def create_pre_transformation_stash(
        self,
        workspace_root: Path,
        *,
        label: str = "flext-refactor-pre-transform",
    ) -> r[str]:
        """Stash uncommitted changes and return the stash reference."""
        self._last_workspace_root = workspace_root
        if not self.is_git_repository(workspace_root):
            return r[str].ok("")

        status_result = self._git.has_changes(workspace_root)
        if status_result.is_failure:
            return r[str].fail(status_result.error or "git status failed")

        if not status_result.value:
            return r[str].ok("")

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        message = f"{label}:{stamp}"
        stash_result = self._git.stash_push(
            workspace_root, message, include_untracked=True
        )
        if stash_result.is_failure:
            return r[str].fail(stash_result.error or "git stash push failed")

        ref_result = self._git.stash_list(workspace_root)
        if ref_result.is_failure:
            return r[str].fail(ref_result.error or "git stash list failed")
        return r[str].ok(ref_result.value.strip())

    @overload
    def rollback(self, workspace_root: Path, stash_ref: str = "") -> r[bool]: ...

    @overload
    def rollback(self, workspace_root: str, /) -> None: ...

    def rollback(
        self,
        workspace_root: Path | str,
        stash_ref: str = "",
    ) -> r[bool] | None:
        """Restore previously stashed state, resolving workspace from context."""
        if isinstance(workspace_root, Path):
            self._last_workspace_root = workspace_root
            return self._rollback_to_stash(workspace_root, stash_ref)

        stash_reference = workspace_root
        if self._last_workspace_root is None:
            self.request_emergency_stop("rollback requested without checkpoint context")
            return None

        rollback_result = self._rollback_to_stash(
            self._last_workspace_root, stash_reference
        )
        if rollback_result.is_failure:
            self.request_emergency_stop(rollback_result.error or "rollback failed")
        return None

    def run_semantic_validation(self, workspace_root: Path) -> r[bool]:
        """Run import checks and tests against the workspace root."""
        self._last_workspace_root = workspace_root
        can_continue = self.ensure_can_continue()
        if can_continue.is_failure:
            return can_continue

        if not self.is_git_repository(workspace_root):
            return r[bool].ok(True)

        import_check = self._runner.run_checked(
            [c.Infra.Toml.PYTHON, "-m", c.Infra.Toml.PYTEST, "--collect-only", "-q"],
            cwd=workspace_root,
        )
        if import_check.is_failure:
            return r[bool].fail(import_check.error or "import validation failed")

        test_check = self._runner.run_checked(self._test_command, cwd=workspace_root)
        if test_check.is_failure:
            return r[bool].fail(test_check.error or "test validation failed")

        return r[bool].ok(True)

    def _resolve_workspace_root(self, files_changed: list[Path]) -> Path | None:
        if self._last_workspace_root is not None:
            return self._last_workspace_root
        if not files_changed:
            return None
        return files_changed[0].parent

    def _rollback_to_stash(self, workspace_root: Path, stash_ref: str) -> r[bool]:
        if not self.is_git_repository(workspace_root):
            return r[bool].ok(True)
        return self._git.stash_pop(workspace_root, stash_ref)

    def save_checkpoint(self, checkpoint: m.Infra.Refactor.Checkpoint) -> r[bool]:
        """Persist a checkpoint to disk as JSON."""
        payload = checkpoint.model_dump()
        payload["updated_at"] = _now_iso()
        return FlextInfraJsonService().write(
            self._checkpoint_path,
            payload,
            ensure_ascii=True,
        )

    def save_checkpoint_state(
        self,
        workspace_root: Path,
        *,
        status: str,
        stash_ref: str,
        processed_targets: list[str],
    ) -> r[bool]:
        """Build and persist a checkpoint from individual state components."""
        checkpoint = m.Infra.Refactor.Checkpoint(
            workspace_root=str(workspace_root),
            status=status,
            stash_ref=stash_ref,
            processed_targets=processed_targets,
        )
        return self.save_checkpoint(checkpoint)

    def load_checkpoint(self) -> r[m.Infra.Refactor.Checkpoint]:
        """Load a previously persisted checkpoint from disk."""
        if not self._checkpoint_path.exists():
            return r[m.Infra.Refactor.Checkpoint].fail("checkpoint does not exist")
        try:
            payload_text = self._checkpoint_path.read_text(
                encoding=c.Infra.Encoding.DEFAULT
            )
            checkpoint = m.Infra.Refactor.Checkpoint.model_validate_json(
                payload_text,
            )
            return r[m.Infra.Refactor.Checkpoint].ok(checkpoint)
        except (OSError, ValueError) as exc:
            return r[m.Infra.Refactor.Checkpoint].fail(f"checkpoint load failed: {exc}")

    def clear_checkpoint(self) -> r[bool]:
        """Remove the on-disk checkpoint file."""
        if not self._checkpoint_path.exists():
            return r[bool].ok(True)
        try:
            self._checkpoint_path.unlink()
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"checkpoint clear failed: {exc}")


__all__ = [
    "FlextInfraRefactorSafetyManager",
]
