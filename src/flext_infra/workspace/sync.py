"""Workspace sync service for base.mk generation and deployment.

Generates base.mk from templates and deploys to project roots with
SHA256-based idempotency and file locking.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import tempfile
from pathlib import Path
from typing import override

from flext_core import FlextService, r

from flext_infra import c, m
from flext_infra.basemk.generator import FlextInfraBaseMkGenerator
from flext_infra.output import output

# Patterns that MUST be in every subproject .gitignore.
_REQUIRED_GITIGNORE_ENTRIES: list[str] = [
    ".reports/",
    ".venv/",
    "__pycache__/",
]


class FlextInfraSyncService(FlextService[m.SyncResult]):
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
        super().__init__()
        self._generator = generator or FlextInfraBaseMkGenerator()
        self._canonical_root = canonical_root

    @override
    def execute(self) -> r[m.SyncResult]:
        """Not used; call sync() directly instead."""
        return r[m.SyncResult].fail("Use sync() method directly")

    def sync(
        self,
        _source: str | None = None,
        _target: str | None = None,
        *,
        project_root: Path | None = None,
        config: m.BaseMkConfig | None = None,
        canonical_root: Path | None = None,
    ) -> r[m.SyncResult]:
        """Synchronize base.mk and .gitignore for a project.

        Copies base.mk from canonical root when available, otherwise
        generates from template. Compares SHA256 hashes and writes only if
        changed. Also ensures required .gitignore entries exist.

        Args:
            project_root: Project root directory. Required.
            config: Optional base.mk generation configuration.
            canonical_root: Workspace root with canonical base.mk.

        Returns:
            FlextResult with SyncResult on success, error message on failure.

        """
        if project_root is None:
            return r[m.SyncResult].fail("project_root is required")

        resolved = project_root.resolve()
        if not resolved.is_dir():
            return r[m.SyncResult].fail(
                f"project root does not exist: {resolved}",
            )

        lock_path = resolved / ".sync.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with lock_path.open("w", encoding=c.Encoding.DEFAULT) as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    changed = 0

                    # 1. Sync base.mk (copy from canonical or generate)
                    effective_root = canonical_root or self._canonical_root
                    basemk_result = self._sync_basemk(
                        resolved, config, canonical_root=effective_root,
                    )
                    if basemk_result.is_failure:
                        return r[m.SyncResult].fail(
                            basemk_result.error or "base.mk sync failed",
                        )
                    changed += 1 if basemk_result.value else 0

                    # 2. Ensure .gitignore entries
                    gitignore_result = self._ensure_gitignore_entries(
                        resolved,
                        _REQUIRED_GITIGNORE_ENTRIES,
                    )
                    if gitignore_result.is_failure:
                        return r[m.SyncResult].fail(
                            gitignore_result.error or ".gitignore sync failed",
                        )
                    changed += 1 if gitignore_result.value else 0

                    return r[m.SyncResult].ok(
                        m.SyncResult(
                            files_changed=changed,
                            source=resolved,
                            target=resolved,
                        ),
                    )
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            return r[m.SyncResult].fail(f"sync lock acquisition failed: {exc}")

    def _sync_basemk(
        self,
        project_root: Path,
        config: m.BaseMkConfig | None,
        *,
        canonical_root: Path | None = None,
    ) -> r[bool]:
        """Sync base.mk from canonical root or generate from template.

        When canonical_root is provided and contains base.mk, copies it
        directly to ensure validator alignment. Falls back to generator.
        """
        # Prefer canonical root copy over template generation
        canonical_basemk = (
            canonical_root / "base.mk"
            if canonical_root is not None
            else None
        )
        if (
            canonical_basemk is not None
            and canonical_basemk.exists()
            and canonical_basemk.resolve() != (project_root / "base.mk").resolve()
        ):
            content = canonical_basemk.read_text(encoding=c.Encoding.DEFAULT)
        else:
            gen_result = self._generator.generate(config)
            if gen_result.is_failure:
                return r[bool].fail(gen_result.error or "base.mk generation failed")
            content = gen_result.value

        target_path = project_root / "base.mk"

        # Compare SHA256 hashes for idempotency
        content_hash = self._sha256_content(content)
        if target_path.exists():
            existing_hash = self._sha256_file(target_path)
            if content_hash == existing_hash:
                return r[bool].ok(False)  # No change needed

        # Atomic write via temp file + rename
        return self._atomic_write(target_path, content)

    def _ensure_gitignore_entries(
        self,
        project_root: Path,
        required: list[str],
    ) -> r[bool]:
        """Idempotently add missing .gitignore entries.

        Appends only entries not already present (exact line match).
        Never removes or reorders existing entries.

        Args:
            project_root: Root directory of the project.
            required: List of gitignore patterns that must be present.

        Returns:
            FlextResult with True if file was changed, False otherwise.

        """
        gitignore = project_root / ".gitignore"
        try:
            existing_lines: list[str] = []
            if gitignore.exists():
                existing_lines = gitignore.read_text(
                    encoding=c.Encoding.DEFAULT,
                ).splitlines()

            existing_patterns = {
                line.strip() for line in existing_lines if line.strip()
            }
            missing = [p for p in required if p not in existing_patterns]
            if not missing:
                return r[bool].ok(False)

            with gitignore.open("a", encoding=c.Encoding.DEFAULT) as handle:
                _ = handle.write(
                    "\n# --- workspace-sync: required ignores (auto-managed) ---\n",
                )
                for pattern in missing:
                    _ = handle.write(f"{pattern}\n")
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f".gitignore update failed: {exc}")

    @staticmethod
    def _sha256_content(content: str) -> str:
        """Compute SHA256 of string content."""
        return hashlib.sha256(content.encode(c.Encoding.DEFAULT)).hexdigest()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Compute SHA256 of file on disk."""
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _atomic_write(target: Path, content: str) -> r[bool]:
        """Write content to target via atomic temp-file rename."""
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=str(target.parent),
                delete=False,
                encoding=c.Encoding.DEFAULT,
                suffix=".tmp",
            ) as tmp:
                _ = tmp.write(content)
                tmp_path = Path(tmp.name)
            _ = tmp_path.replace(target)
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"atomic write failed: {exc}")


def main() -> int:
    """CLI entry point for workspace sync."""
    parser = argparse.ArgumentParser(description="Workspace base.mk sync")
    _ = parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help="Project root directory",
    )
    _ = parser.add_argument(
        "--canonical-root",
        type=Path,
        default=None,
        help="Canonical workspace root",
    )
    args = parser.parse_args()

    service = FlextInfraSyncService(canonical_root=args.canonical_root)
    result = service.sync(project_root=args.project_root)

    if result.is_success:
        return 0
    output.error(result.error or "sync failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FlextInfraSyncService", "main"]
