"""Safety management for refactor operations: checkpoints, rollback, and validation."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias, overload

from flext_core import r
from flext_infra import c, m, p, u
from flext_infra._utilities.safety import FlextInfraUtilitiesSafety

RBool: TypeAlias = r[bool]
RStr: TypeAlias = r[str]
RCheckpoint: TypeAlias = r[m.Infra.Refactor.Checkpoint]


class FlextInfraRefactorSafetyManager:
    """Orchestrate pre-/post-transform safety: stash, validate, rollback."""

    def __init__(
        self,
        runner: p.Infra.SafetyRunner | None = None,
        checkpoint_path: Path | None = None,
        test_command: list[str] | None = None,
    ) -> None:
        """Initialize safety manager with runner, checkpoint path, and test command."""
        self._runner: p.Infra.SafetyRunner | None = runner
        self._checkpoint_path = checkpoint_path or Path(
            ".sisyphus/refactor/safety-checkpoint.json",
        )
        self._test_command = test_command or [
            c.Infra.Toml.PYTHON,
            "-m",
            c.Infra.Toml.PYTEST,
            "-q",
        ]
        self._emergency_stop_reason = ""
        self._last_workspace_root: Path | None = None

    def _run_checked(self, cmd: list[str], cwd: Path) -> RBool:
        if self._runner is not None:
            return self._runner.run_checked(cmd, cwd=cwd)
        return u.Infra.run_checked(cmd, cwd=cwd)

    def create_checkpoint(
        self,
        project_root: Path,
        *,
        label: str = "flext-safety-checkpoint",
    ) -> RStr:
        """Stash current state and return the stash reference."""
        self._last_workspace_root = project_root
        return FlextInfraUtilitiesSafety.create_checkpoint(project_root, label=label)

    def validate_transform(self, files_changed: list[Path]) -> RBool:
        """Run semantic validation after a transformation batch."""
        workspace_root = self._resolve_workspace_root(files_changed)
        if workspace_root is None:
            out: RBool = r[bool].fail("unable to resolve workspace root")
            return out
        out2: RBool = self.run_semantic_validation(workspace_root)
        return out2

    def emergency_stop(self, reason: str) -> None:
        """Trigger an emergency stop with the given reason."""
        self.request_emergency_stop(reason)

    def request_emergency_stop(self, reason: str) -> None:
        """Record an emergency stop reason for later inspection."""
        self._emergency_stop_reason = reason.strip() or "unspecified"

    def clear_emergency_stop(self) -> None:
        """Clear any previously recorded emergency stop."""
        self._emergency_stop_reason = ""

    def is_emergency_stop_requested(self) -> bool:
        """Return True if an emergency stop has been requested."""
        return bool(self._emergency_stop_reason)

    def ensure_can_continue(self) -> RBool:
        """Fail if an emergency stop is active; succeed otherwise."""
        if self._emergency_stop_reason:
            out: RBool = r[bool].fail(f"Emergency stop: {self._emergency_stop_reason}")
            return out
        out2: RBool = r[bool].ok(True)
        return out2

    def is_git_repository(self, workspace_root: Path) -> bool:
        """Check whether workspace_root sits inside a Git work-tree."""
        return u.Infra.git_is_repo(workspace_root)

    def create_pre_transformation_stash(
        self,
        workspace_root: Path,
        *,
        label: str = "flext-refactor-pre-transform",
    ) -> RStr:
        """Stash uncommitted changes and return the stash reference."""
        self._last_workspace_root = workspace_root
        return FlextInfraUtilitiesSafety.create_checkpoint(workspace_root, label=label)

    @overload
    def rollback(self, workspace_root: Path, stash_ref: str = "") -> RBool: ...

    @overload
    def rollback(self, workspace_root: str, /) -> None: ...

    def rollback(
        self,
        workspace_root: Path | str,
        stash_ref: str = "",
    ) -> RBool | None:
        """Restore previously stashed state, resolving workspace from context."""
        if isinstance(workspace_root, Path):
            self._last_workspace_root = workspace_root
            return self._rollback_to_stash(workspace_root, stash_ref)
        stash_reference = workspace_root
        if self._last_workspace_root is None:
            self.request_emergency_stop("rollback requested without checkpoint context")
            return None
        rb = self._rollback_to_stash(self._last_workspace_root, stash_reference)
        if rb.is_failure:
            self.request_emergency_stop(rb.error or "rollback failed")
        return None

    def run_semantic_validation(self, workspace_root: Path) -> RBool:
        """Run import checks and tests against the workspace root."""
        self._last_workspace_root = workspace_root
        cont = self.ensure_can_continue()
        if cont.is_failure:
            out: RBool = cont
            return out
        if not self.is_git_repository(workspace_root):
            out2: RBool = r[bool].ok(True)
            return out2
        import_cmd = [
            c.Infra.Toml.PYTHON,
            "-m",
            c.Infra.Toml.PYTEST,
            "--collect-only",
            "-q",
        ]
        ic = self._run_checked(import_cmd, workspace_root)
        if ic.is_failure:
            out3: RBool = r[bool].fail(ic.error or "import validation failed")
            return out3
        tc = self._run_checked(self._test_command, workspace_root)
        if tc.is_failure:
            out4: RBool = r[bool].fail(tc.error or "test validation failed")
            return out4
        out5: RBool = r[bool].ok(True)
        return out5

    def save_checkpoint(self, checkpoint: m.Infra.Refactor.Checkpoint) -> RBool:
        """Persist a checkpoint to disk as JSON."""
        payload = checkpoint.model_dump()
        payload["updated_at"] = u.generate_iso_timestamp()
        out: RBool = u.Infra.write_json(
            self._checkpoint_path, payload, ensure_ascii=True
        )
        return out

    def save_checkpoint_state(
        self,
        workspace_root: Path,
        *,
        status: str,
        stash_ref: str,
        processed_targets: list[str],
    ) -> RBool:
        """Build and persist a checkpoint from individual state components."""
        out: RBool = self.save_checkpoint(
            m.Infra.Refactor.Checkpoint(
                workspace_root=str(workspace_root),
                status=status,
                stash_ref=stash_ref,
                processed_targets=processed_targets,
            ),
        )
        return out

    def load_checkpoint(self) -> RCheckpoint:
        """Load a previously persisted checkpoint from disk."""
        if not self._checkpoint_path.exists():
            out: RCheckpoint = r[m.Infra.Refactor.Checkpoint].fail(
                "checkpoint does not exist"
            )
            return out
        try:
            text = self._checkpoint_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            cp: m.Infra.Refactor.Checkpoint = (
                m.Infra.Refactor.Checkpoint.model_validate_json(text)
            )
            out2: RCheckpoint = r[m.Infra.Refactor.Checkpoint].ok(cp)
            return out2
        except (OSError, ValueError) as exc:
            out3: RCheckpoint = r[m.Infra.Refactor.Checkpoint].fail(str(exc))
            return out3

    def clear_checkpoint(self) -> RBool:
        """Remove the on-disk checkpoint file."""
        if not self._checkpoint_path.exists():
            out: RBool = r[bool].ok(True)
            return out
        try:
            self._checkpoint_path.unlink()
            out2: RBool = r[bool].ok(True)
            return out2
        except OSError as exc:
            out3: RBool = r[bool].fail(str(exc))
            return out3

    def _resolve_workspace_root(self, files_changed: list[Path]) -> Path | None:
        if self._last_workspace_root is not None:
            return self._last_workspace_root
        return files_changed[0].parent if files_changed else None

    def _rollback_to_stash(self, workspace_root: Path, stash_ref: str) -> RBool:
        return FlextInfraUtilitiesSafety.rollback_to_checkpoint(
            workspace_root, stash_ref
        )


__all__ = ["FlextInfraRefactorSafetyManager"]
