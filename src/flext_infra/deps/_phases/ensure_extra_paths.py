"""Phase: Ensure pyright/mypy extra paths are synchronized."""

from __future__ import annotations

from pathlib import Path

import tomlkit

from flext_infra import u
from flext_infra.deps import extra_paths


class EnsureExtraPathsPhase:
    """Ensure pyright/mypy extra paths are synchronized via extra_paths.sync_one."""

    def apply(
        self,
        doc: tomlkit.TOMLDocument,
        *,
        path: Path,
        is_root: bool,
        dry_run: bool = False,
    ) -> list[str]:
        result = extra_paths.sync_one(path, dry_run=dry_run, is_root=is_root)
        if result.is_failure:
            u.Infra.warning(f"extra_paths sync failed for {path}: {result.error}")
            return []
        if result.value:
            return ["synchronized extra paths (pyright/mypy)"]
        return []
