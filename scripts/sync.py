#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
"""Synchronize canonical workspace files into a project."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import shutil
import stat
import sys
import tempfile
from pathlib import Path


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _copy_if_changed(source: Path, target: Path) -> bool:
    if target.exists() and _sha256(source) == _sha256(target):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with (
        source.open("rb") as source_handle,
        tempfile.NamedTemporaryFile(
            mode="wb", dir=str(target.parent), delete=False
        ) as tmp,
    ):
        shutil.copyfileobj(source_handle, tmp)
        tmp_path = Path(tmp.name)
    source_mode = stat.S_IMODE(source.stat().st_mode)
    Path(tmp_path).chmod(source_mode)
    Path(tmp_path).replace(target)
    return True


def _sync_tree(source_dir: Path, target_dir: Path, *, prune: bool) -> int:
    changed = 0
    source_files = {
        p.relative_to(source_dir)
        for p in source_dir.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and not any(part.startswith(".") for part in p.parts)
    }
    for rel in sorted(source_files):
        changed += 1 if _copy_if_changed(source_dir / rel, target_dir / rel) else 0
    if prune:
        for path in sorted(target_dir.rglob("*"), reverse=True):
            if path.is_file() and path.relative_to(target_dir) not in source_files:
                path.unlink()
                changed += 1
        for path in sorted(target_dir.rglob("*"), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
                changed += 1
    return changed


def main() -> int:
    """Run file sync for base.mk and scripts tree."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--project-root", type=Path, required=True)
    _ = parser.add_argument("--canonical-root", type=Path, required=True)
    _ = parser.add_argument("--prune", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    canonical_root = args.canonical_root.resolve()
    lock_path = project_root / ".sync.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        changed = 0
        changed += (
            1
            if _copy_if_changed(canonical_root / "base.mk", project_root / "base.mk")
            else 0
        )
        changed += _sync_tree(
            canonical_root / "scripts", project_root / "scripts", prune=args.prune
        )
        # Clean up legacy top-level libs/ (now lives under scripts/libs/)
        legacy_libs = project_root / "libs"
        if legacy_libs.is_dir():
            for path in sorted(legacy_libs.rglob("*"), reverse=True):
                if path.is_file() and "__pycache__" not in path.parts:
                    path.unlink()
                    changed += 1
            for path in sorted(legacy_libs.rglob("*"), reverse=True):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
                    changed += 1
            if legacy_libs.is_dir() and not any(legacy_libs.iterdir()):
                legacy_libs.rmdir()
                changed += 1
        examples_dir = project_root / "examples"
        if not examples_dir.exists():
            examples_dir.mkdir()
            readme = examples_dir / "README.md"
            project_name = project_root.name.replace("-", " ").title()
            readme.write_text(
                f"# {project_name} Examples\n\n"
                f"Usage examples for `{project_root.name}`.\n",
                encoding="utf-8",
            )
            changed += 1
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    print(f"files_changed={changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
