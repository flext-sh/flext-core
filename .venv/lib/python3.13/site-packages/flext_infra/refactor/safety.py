"""Safety management for refactor operations: backup, validate, rollback."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraRefactorSafetyManager:
    """Orchestrate pre-/post-transform safety: .bak backup, validate, rollback."""

    def __init__(self, *, test_command: t.StrSequence | None = None) -> None:
        """Initialize safety manager with test command."""
        self._test_command = test_command or [
            c.Infra.PYTHON,
            "-m",
            c.Infra.PYTEST,
            "-q",
        ]
        self._emergency_stop_reason = ""
        self._bak_paths: t.SequenceOf[Path] = []

    def request_emergency_stop(self, reason: str) -> None:
        """Record an emergency stop reason for later inspection."""
        self._emergency_stop_reason = reason.strip() or "unspecified"

    @property
    def emergency_stop_requested(self) -> bool:
        """Whether an emergency stop has been requested."""
        return bool(self._emergency_stop_reason)

    def create_pre_transformation_checkpoint(
        self, repository_root: Path, *, label: str = "flext-refactor-pre-transform"
    ) -> p.Result[str]:
        """Back up files in repository root and return label as reference."""
        _ = label
        py_files = list(
            u.Infra.iter_matching_files(
                repository_root, includes=[c.Infra.EXT_PYTHON_GLOB]
            )
        )
        self._bak_paths = u.Infra.backup_files(py_files)
        return r[str].ok(str(repository_root))

    def rollback(
        self, repository_root: Path, checkpoint_ref: str = ""
    ) -> p.Result[bool]:
        """Restore previously backed up files."""
        _ = repository_root, checkpoint_ref
        u.Infra.restore_files(self._bak_paths)
        self._bak_paths = []
        return r[bool].ok(True)

    def save_checkpoint_state(
        self,
        repository_root: Path,
        *,
        status: str,
        checkpoint_ref: str,
        processed_targets: t.StrSequence,
    ) -> p.Result[bool]:
        """Persist checkpoint metadata for the current refactor run.

        The current safety flow relies on copy-on-write backups, so saving
        checkpoint state is intentionally a no-op hook used by integrations
        and tests to observe lifecycle sequencing.
        """
        _ = repository_root, status, checkpoint_ref, processed_targets
        return r[bool].ok(True)

    @staticmethod
    def _is_no_tests_collected_error(message: str | None) -> bool:
        """Return True when pytest exit code 5 only indicates no collected tests."""
        if not message:
            return False
        normalized = message.lower()
        return "failed (5):" in normalized and (
            "no tests collected" in normalized or "no tests ran" in normalized
        )

    def run_semantic_validation(self, repository_root: Path) -> p.Result[bool]:
        """Run import checks and tests against the repository root."""
        if self._emergency_stop_reason:
            return r[bool].fail(f"Emergency stop: {self._emergency_stop_reason}")
        ic = u.Cli.run_checked(
            [c.Infra.PYTHON, "-m", c.Infra.PYTEST, "--collect-only", "-q"],
            cwd=repository_root,
        )
        if ic.failure and not self._is_no_tests_collected_error(ic.error):
            return r[bool].fail(ic.error or "import validation failed")
        tc = u.Cli.run_checked(self._test_command, cwd=repository_root)
        if tc.failure and not self._is_no_tests_collected_error(tc.error):
            return r[bool].fail(tc.error or "test validation failed")
        return r[bool].ok(True)

    def clear_checkpoint(self, *, keep: t.SequenceOf[Path] = ()) -> p.Result[bool]:
        """Clean up transient backups while preserving requested .bak files."""
        keep_paths = {
            path.with_suffix(path.suffix + c.Infra.SAFE_EXECUTION_BAK_SUFFIX)
            for path in keep
        }
        cleanup = [bak for bak in self._bak_paths if bak not in keep_paths]
        u.Infra.cleanup_backups(cleanup)
        self._bak_paths = [bak for bak in self._bak_paths if bak in keep_paths]
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraRefactorSafetyManager"]
