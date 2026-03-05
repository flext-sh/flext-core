from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast, overload

from flext_core import r
from flext_infra.subprocess import FlextInfraCommandRunner


class FlextInfraRefactorSafetyRunnerProtocol(Protocol):
    def capture(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> r[str]: ...

    def run_checked(
        self,
        cmd: list[str],
        cwd: Path | None = None,
        timeout: int | None = None,
    ) -> r[bool]: ...


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _empty_str_list() -> list[str]:
    return []


@dataclass
class FlextInfraRefactorCheckpoint:
    workspace_root: str
    status: str = "running"
    stash_ref: str = ""
    processed_targets: list[str] = field(default_factory=_empty_str_list)
    updated_at: str = field(default_factory=_now_iso)


class FlextInfraRefactorSafetyManager:
    def __init__(
        self,
        runner: FlextInfraRefactorSafetyRunnerProtocol | None = None,
        checkpoint_path: Path | None = None,
        test_command: list[str] | None = None,
    ) -> None:
        self._runner = runner or FlextInfraCommandRunner()
        self._checkpoint_path = checkpoint_path or Path(
            ".sisyphus/refactor/safety-checkpoint.json"
        )
        self._test_command = test_command or ["python", "-m", "pytest", "-q"]
        self._emergency_stop_reason = ""
        self._last_workspace_root: Path | None = None

    def create_checkpoint(self, project_root: Path) -> str:
        self._last_workspace_root = project_root
        stash_result = self.create_pre_transformation_stash(project_root)
        if stash_result.is_failure:
            self.request_emergency_stop(
                stash_result.error or "checkpoint creation failed"
            )
            return ""
        return stash_result.value

    def validate_transform(self, files_changed: list[Path]) -> bool:
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
        self.request_emergency_stop(reason)

    def request_emergency_stop(self, reason: str) -> None:
        self._emergency_stop_reason = reason.strip() or "unspecified emergency stop"

    def clear_emergency_stop(self) -> None:
        self._emergency_stop_reason = ""

    def is_emergency_stop_requested(self) -> bool:
        return bool(self._emergency_stop_reason)

    def ensure_can_continue(self) -> r[bool]:
        if self._emergency_stop_reason:
            return r[bool].fail(
                f"Emergency stop requested: {self._emergency_stop_reason}",
            )
        return r[bool].ok(True)

    def is_git_repository(self, workspace_root: Path) -> bool:
        result = self._runner.run_checked(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=workspace_root,
        )
        return result.is_success

    def create_pre_transformation_stash(
        self,
        workspace_root: Path,
        *,
        label: str = "flext-refactor-pre-transform",
    ) -> r[str]:
        self._last_workspace_root = workspace_root
        if not self.is_git_repository(workspace_root):
            return r[str].ok("")

        status_result = self._runner.capture(
            ["git", "status", "--porcelain"],
            cwd=workspace_root,
        )
        if status_result.is_failure:
            return r[str].fail(status_result.error or "git status failed")

        if not status_result.value.strip():
            return r[str].ok("")

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        message = f"{label}:{stamp}"
        stash_result = self._runner.run_checked(
            ["git", "stash", "push", "--include-untracked", "-m", message],
            cwd=workspace_root,
        )
        if stash_result.is_failure:
            return r[str].fail(stash_result.error or "git stash push failed")

        ref_result = self._runner.capture(
            ["git", "stash", "list", "-n", "1", "--format=%gd"],
            cwd=workspace_root,
        )
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
        self._last_workspace_root = workspace_root
        can_continue = self.ensure_can_continue()
        if can_continue.is_failure:
            return can_continue

        if not self.is_git_repository(workspace_root):
            return r[bool].ok(True)

        import_check = self._runner.run_checked(
            ["python", "-m", "pytest", "--collect-only", "-q"],
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
        command = ["git", "stash", "pop"]
        if stash_ref:
            command.append(stash_ref)
        return self._runner.run_checked(command, cwd=workspace_root)

    def save_checkpoint(self, checkpoint: FlextInfraRefactorCheckpoint) -> r[bool]:
        payload = {
            "workspace_root": checkpoint.workspace_root,
            "status": checkpoint.status,
            "stash_ref": checkpoint.stash_ref,
            "processed_targets": checkpoint.processed_targets,
            "updated_at": _now_iso(),
        }
        try:
            self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._checkpoint_path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
                encoding="utf-8",
            )
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"checkpoint save failed: {exc}")

    def save_checkpoint_state(
        self,
        workspace_root: Path,
        *,
        status: str,
        stash_ref: str,
        processed_targets: list[str],
    ) -> r[bool]:
        checkpoint = FlextInfraRefactorCheckpoint(
            workspace_root=str(workspace_root),
            status=status,
            stash_ref=stash_ref,
            processed_targets=processed_targets,
        )
        return self.save_checkpoint(checkpoint)

    def load_checkpoint(self) -> r[FlextInfraRefactorCheckpoint]:
        if not self._checkpoint_path.exists():
            return r[FlextInfraRefactorCheckpoint].fail("checkpoint does not exist")
        try:
            payload_text = self._checkpoint_path.read_text(encoding="utf-8")
            payload_obj = json.loads(payload_text)
            if not isinstance(payload_obj, Mapping):
                return r[FlextInfraRefactorCheckpoint].fail(
                    "checkpoint payload must be a mapping",
                )
            typed_payload = cast("Mapping[str, object]", payload_obj)

            workspace_root = typed_payload.get("workspace_root", "")
            status = typed_payload.get("status", "running")
            stash_ref = typed_payload.get("stash_ref", "")
            processed_targets_obj = typed_payload.get("processed_targets", [])
            updated_at = typed_payload.get("updated_at", _now_iso())
            if not isinstance(workspace_root, str) or not workspace_root:
                return r[FlextInfraRefactorCheckpoint].fail(
                    "checkpoint workspace_root is invalid",
                )
            if not isinstance(status, str):
                return r[FlextInfraRefactorCheckpoint].fail(
                    "checkpoint status is invalid",
                )
            if not isinstance(stash_ref, str):
                return r[FlextInfraRefactorCheckpoint].fail(
                    "checkpoint stash_ref is invalid",
                )
            if not isinstance(updated_at, str):
                return r[FlextInfraRefactorCheckpoint].fail(
                    "checkpoint updated_at is invalid",
                )
            if not isinstance(processed_targets_obj, list):
                return r[FlextInfraRefactorCheckpoint].fail(
                    "checkpoint processed_targets is invalid",
                )
            processed_targets_raw = cast("list[object]", processed_targets_obj)
            if any(not isinstance(item, str) for item in processed_targets_raw):
                return r[FlextInfraRefactorCheckpoint].fail(
                    "checkpoint processed_targets is invalid",
                )

            processed_targets = cast("list[str]", processed_targets_raw)

            checkpoint = FlextInfraRefactorCheckpoint(
                workspace_root=workspace_root,
                status=status,
                stash_ref=stash_ref,
                processed_targets=processed_targets,
                updated_at=updated_at,
            )
            return r[FlextInfraRefactorCheckpoint].ok(checkpoint)
        except (OSError, ValueError) as exc:
            return r[FlextInfraRefactorCheckpoint].fail(
                f"checkpoint load failed: {exc}"
            )

    def clear_checkpoint(self) -> r[bool]:
        if not self._checkpoint_path.exists():
            return r[bool].ok(True)
        try:
            self._checkpoint_path.unlink()
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"checkpoint clear failed: {exc}")


__all__ = [
    "FlextInfraRefactorCheckpoint",
    "FlextInfraRefactorSafetyManager",
    "FlextInfraRefactorSafetyRunnerProtocol",
]
