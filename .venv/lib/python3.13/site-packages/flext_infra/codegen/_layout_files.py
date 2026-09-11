"""Filesystem and Git primitives for the layout engine apply path (flext-0wuz).

Archive-not-delete law: nothing is ever destroyed; collisions and duplicates
move content into ``<archive_root>/<project>/``. Git checkouts use ``git mv``
for moves and ``git rm --cached`` plus rename for archives; plain directories
degrade to plain renames.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra import config, m, p, r, t, u
from flext_infra._utilities._sort_keys import path_depth


class FlextInfraCodegenLayoutFilesMixin:
    """Move/archive primitives shared by the layout apply orchestration."""

    def _move_entry(
        self, project_dir: Path, source: Path, target: Path, archive_rel: str
    ) -> p.Result[str]:
        """Move one entry; collisions archive the source (never delete)."""
        if not target.exists():
            moved = self._move_path(project_dir, source, target)
            if moved.failure:
                return r[str].fail(moved.error or "move failed")
            return r[str].ok(f"moved {source.name} -> {target.name}")
        if (
            source.is_file()
            and target.is_file()
            and source.read_bytes() == target.read_bytes()
        ):
            archived = self._archive_path(project_dir, source, archive_rel)
            if archived.failure:
                return r[str].fail(archived.error or "identical-source archive failed")
            return r[str].ok(f"identical target kept; source archived: {source.name}")
        archived = self._archive_path(project_dir, source, archive_rel)
        if archived.failure:
            return r[str].fail(archived.error or "collision archive failed")
        return r[str].ok(
            f"collision: target kept, source archived for review: {source.name}"
        )

    def _archive_path(
        self,
        project_dir: Path,
        source: Path,
        rel: str,
        finding: m.Infra.LayoutFinding | None = None,
    ) -> p.Result[t.Pair[t.Infra.LayoutStatus, str]]:
        """Move one entry under ``archive_root/<project>/`` (idempotent)."""
        spec = config.Infra.codegen.layout
        target = project_dir / spec.archive_root / project_dir.name / rel
        base_message = finding.message if finding is not None else source.name
        if target.exists():
            if (
                source.is_file()
                and target.is_file()
                and source.read_bytes() == target.read_bytes()
            ):
                removed = self._remove_tracked_or_unlinked(project_dir, source)
                if removed.failure:
                    return r[t.Pair[t.Infra.LayoutStatus, str]].fail(
                        removed.error or "duplicate removal failed"
                    )
                return r[t.Pair[t.Infra.LayoutStatus, str]].ok((
                    "applied",
                    f"{base_message} (already archived; duplicate removed)",
                ))
            return r[t.Pair[t.Infra.LayoutStatus, str]].ok((
                "skipped",
                f"{base_message} (archive target exists; manual review)",
            ))
        moved = self._archive_move(project_dir, source, target)
        if moved.failure:
            return r[t.Pair[t.Infra.LayoutStatus, str]].fail(
                moved.error or "archive move failed"
            )
        return r[t.Pair[t.Infra.LayoutStatus, str]].ok(("applied", base_message))

    def _move_path(
        self, project_dir: Path, source: Path, target: Path
    ) -> p.Result[bool]:
        """Move via ``git mv`` when tracked, plain rename otherwise."""
        target.parent.mkdir(parents=True, exist_ok=True)
        source_rel = source.relative_to(project_dir).as_posix()
        target_rel = target.relative_to(project_dir).as_posix()
        if self._git_tracked(project_dir, source_rel):
            moved = u.Infra.git_mv_path(
                m.Infra.GitPathPairRequest(
                    repo_root=project_dir, source=source_rel, target=target_rel
                )
            )
            if moved.failure:
                return r[bool].fail(moved.error or "git mv execution failed")
            return r[bool].ok(True)
        source.rename(target)
        return r[bool].ok(True)

    def _archive_move(
        self, project_dir: Path, source: Path, target: Path
    ) -> p.Result[bool]:
        """Untrack a git-tracked source, then move it into the archive root.

        Archives leave Git tracking (``git rm --cached``) so the preserved
        content under ``archive_root`` is ignored, never committed.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        source_rel = source.relative_to(project_dir).as_posix()
        if self._git_tracked(project_dir, source_rel):
            untracked = u.Infra.git_rm_cached(
                m.Infra.GitRelativePathRequest(
                    repo_root=project_dir, relative_path=source_rel
                )
            )
            if untracked.failure:
                return r[bool].fail(untracked.error or "git rm execution failed")
        source.rename(target)
        return r[bool].ok(True)

    def _remove_tracked_or_unlinked(
        self, project_dir: Path, source: Path
    ) -> p.Result[bool]:
        """Remove a byte-identical duplicate whose content is already archived."""
        source_rel = source.relative_to(project_dir).as_posix()
        if self._git_tracked(project_dir, source_rel):
            removed = u.Infra.git_rm_path(
                m.Infra.GitRelativePathRequest(
                    repo_root=project_dir, relative_path=source_rel
                )
            )
            if removed.failure:
                return r[bool].fail(removed.error or "git rm execution failed")
            return r[bool].ok(True)
        source.unlink()
        return r[bool].ok(True)

    @staticmethod
    def _git_tracked(project_dir: Path, rel: str) -> bool:
        """Whether a path is git-tracked; False for plain directories."""
        listed = u.Infra.git_is_tracked(
            m.Infra.GitRelativePathRequest(repo_root=project_dir, relative_path=rel)
        )
        return listed.success and listed.value.value

    @staticmethod
    def _prune_empty_dirs(root: Path) -> None:
        """Remove directories left empty by a merge, deepest first."""
        dirs = sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=path_depth,
            reverse=True,
        )
        for path in (*dirs, root):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()


__all__: list[str] = ["FlextInfraCodegenLayoutFilesMixin"]
