"""Workspace sync service for base.mk generation and deployment.

Generates base.mk from templates and deploys to project roots with
SHA256-based idempotency and file locking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import fcntl
import hashlib
import tempfile
from pathlib import Path
from typing import override

from flext_core import r, s
from flext_infra import c, m, u
from flext_infra._utilities.output import output
from flext_infra.basemk.generator import FlextInfraBaseMkGenerator


class FlextInfraSyncService(s[m.Infra.Workspace.SyncResult]):
    """Infrastructure service for workspace base.mk synchronization.

    Generates a fresh base.mk via ``FlextInfraBaseMkGenerator``, compares its SHA256
    hash against the existing file, and writes only when content differs.
    All writes are protected by an ``fcntl`` file lock.

    """

    def __init__(
        self,
        generator: FlextInfraBaseMkGenerator | None = None,
        *,
        canonical_root: Path | None = None,
    ) -> None:
        """Initialize the sync service."""
        super().__init__(
            config_type=None,
            config_overrides=None,
            initial_context=None,
            subproject=None,
            services=None,
            factories=None,
            resources=None,
            container_overrides=None,
            wire_modules=None,
            wire_packages=None,
            wire_classes=None,
        )
        self._generator = generator or FlextInfraBaseMkGenerator()
        self._canonical_root = canonical_root

    @staticmethod
    def _atomic_write(target: Path, content: str) -> r[bool]:
        """Write content to target via atomic temp-file rename."""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(target.parent),
                delete=False,
                encoding=c.Infra.Encoding.DEFAULT,
                suffix=".tmp",
            ) as tmp:
                _ = tmp.write(content)
                tmp_path = Path(tmp.name)
            _ = tmp_path.replace(target)
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"atomic write failed: {exc}")

    @staticmethod
    def _sha256_content(content: str) -> str:
        """Compute SHA256 of string content."""
        return hashlib.sha256(content.encode(c.Infra.Encoding.DEFAULT)).hexdigest()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Compute SHA256 of file on disk."""
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @override
    def execute(self) -> r[m.Infra.Workspace.SyncResult]:
        """Not used; call sync() directly instead."""
        return r[m.Infra.Workspace.SyncResult].fail("Use sync() method directly")

    def sync(
        self,
        _source: str | None = None,
        _target: str | None = None,
        *,
        workspace_root: Path | None = None,
        config: m.Infra.Basemk.BaseMkConfig | None = None,
        canonical_root: Path | None = None,
    ) -> r[m.Infra.Workspace.SyncResult]:
        """Synchronize base.mk and .gitignore for a project.

        Copies base.mk from canonical root when available, otherwise
        generates from template. Compares SHA256 hashes and writes only if
        changed. Also ensures required .gitignore entries exist.

        Args:
            workspace_root: Workspace root directory. Required.
            config: Optional base.mk generation configuration.
            canonical_root: Workspace root with canonical base.mk.

        Returns:
            r with SyncResult on success, error message on failure.

        """
        if workspace_root is None:
            return r[m.Infra.Workspace.SyncResult].fail("workspace_root is required")
        resolved = workspace_root.resolve()
        if not resolved.is_dir():
            return r[m.Infra.Workspace.SyncResult].fail(
                f"project root does not exist: {resolved}",
            )
        lock_path = resolved / ".sync.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with lock_path.open("w", encoding=c.Infra.Encoding.DEFAULT) as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    changed = 0
                    effective_root = canonical_root or self._canonical_root
                    basemk_result = self._sync_basemk(
                        resolved,
                        config,
                        canonical_root=effective_root,
                    )
                    if basemk_result.is_failure:
                        return r[m.Infra.Workspace.SyncResult].fail(
                            basemk_result.error or "base.mk sync failed",
                        )
                    changed += 1 if basemk_result.value else 0
                    gitignore_result = self._ensure_gitignore_entries(
                        resolved,
                        c.Infra.Workspace.REQUIRED_GITIGNORE_ENTRIES,
                    )
                    if gitignore_result.is_failure:
                        return r[m.Infra.Workspace.SyncResult].fail(
                            gitignore_result.error or ".gitignore sync failed",
                        )
                    changed += 1 if gitignore_result.value else 0
                    return r[m.Infra.Workspace.SyncResult].ok(
                        m.Infra.Workspace.SyncResult(
                            files_changed=changed,
                            source=resolved,
                            target=resolved,
                        ),
                    )
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            return r[m.Infra.Workspace.SyncResult].fail(
                f"sync lock acquisition failed: {exc}",
            )

    def _ensure_gitignore_entries(
        self,
        workspace_root: Path,
        required: list[str],
    ) -> r[bool]:
        """Idempotently add missing .gitignore entries.

        Appends only entries not already present (exact line match).
        Never removes or reorders existing entries.

        Args:
            workspace_root: Root directory of the project.
            required: List of gitignore patterns that must be present.

        Returns:
            r with True if file was changed, False otherwise.

        """
        gitignore = workspace_root / c.Infra.Files.GITIGNORE
        try:
            existing_lines: list[str] = []
            if gitignore.exists():
                existing_lines = gitignore.read_text(
                    encoding=c.Infra.Encoding.DEFAULT,
                ).splitlines()
            existing_patterns = {
                line.strip() for line in existing_lines if line.strip()
            }
            missing = [p for p in required if p not in existing_patterns]
            if not missing:
                return r[bool].ok(False)
            with gitignore.open("a", encoding=c.Infra.Encoding.DEFAULT) as handle:
                _ = handle.write(
                    "\n# --- workspace-sync: required ignores (auto-managed) ---\n",
                )
                for pattern in missing:
                    _ = handle.write(f"{pattern}\n")
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f".gitignore update failed: {exc}")

    def _sync_basemk(
        self,
        workspace_root: Path,
        config: m.Infra.Basemk.BaseMkConfig | None,
        *,
        canonical_root: Path | None = None,
    ) -> r[bool]:
        """Sync base.mk from canonical root or generate from template.

        When canonical_root is provided and contains base.mk, copies it
        directly to ensure validator alignment. Falls back to generator.
        """
        canonical_basemk = (
            canonical_root / c.Infra.Files.BASE_MK
            if canonical_root is not None
            else None
        )
        if (
            canonical_basemk is not None
            and canonical_basemk.exists()
            and (
                canonical_basemk.resolve()
                != (workspace_root / c.Infra.Files.BASE_MK).resolve()
            )
        ):
            content = canonical_basemk.read_text(encoding=c.Infra.Encoding.DEFAULT)
        else:
            gen_result = self._generator.generate(config)
            if gen_result.is_failure:
                return r[bool].fail(gen_result.error or "base.mk generation failed")
            content = gen_result.value
        target_path = workspace_root / c.Infra.Files.BASE_MK
        content_hash = self._sha256_content(content)
        if target_path.exists():
            existing_hash = self._sha256_file(target_path)
            if content_hash == existing_hash:
                return r[bool].ok(False)
        return self._atomic_write(target_path, content)


def main() -> int:
    """CLI entry point for workspace sync."""
    parser = u.Infra.create_parser(
        "flext-infra workspace sync",
        "Workspace base.mk sync",
        include_apply=False,
    )
    _ = parser.add_argument(
        "--canonical-root",
        type=Path,
        default=None,
        help="Canonical workspace root",
    )
    args = parser.parse_args()
    cli = u.Infra.resolve(args)
    service = FlextInfraSyncService(
        canonical_root=getattr(args, "canonical_root", None)
    )
    result = service.sync(workspace_root=cli.workspace)
    if result.is_success:
        return 0
    output.error(result.error or "sync failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
__all__ = ["FlextInfraSyncService", "main"]
