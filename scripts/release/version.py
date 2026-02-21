#!/usr/bin/env python3
"""Script to update project and subproject versions in pyproject.toml files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from scripts.libs.versioning import replace_project_version
from scripts.release.shared import parse_semver, resolve_projects, workspace_root


def _needs_version_update(content: str, version: str) -> bool:
    """Return whether TOML content has a different version value."""
    match = re.search(r'^version\s*=\s*"(.+?)"', content, re.MULTILINE)
    if match is None:
        return True
    return match.group(1) != version


def _version_files(root: Path, project_names: list[str]) -> list[Path]:
    """Discover all pyproject.toml files that need version updates."""
    files: list[Path] = [root / "pyproject.toml"]
    for project in resolve_projects(root, project_names):
        pyproject = project.path / "pyproject.toml"
        if pyproject.exists():
            files.append(pyproject)
    return sorted({path.resolve() for path in files})


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the version update script."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--root", type=Path, default=Path())
    _ = parser.add_argument("--version", required=True)
    _ = parser.add_argument("--dev-suffix", type=int, default=0)
    _ = parser.add_argument("--projects", nargs="*", default=[])
    _ = parser.add_argument("--apply", action="store_true")
    _ = parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Execute the version update process.

    Returns:
        0 on success.

    """
    args = _parse_args()
    root = workspace_root(args.root)
    _ = parse_semver(args.version)
    target_version = f"{args.version}-dev" if args.dev_suffix == 1 else args.version

    changed = 0
    for file_path in _version_files(root, args.projects):
        content = file_path.read_text(encoding="utf-8")
        did_change = _needs_version_update(content, target_version)
        if did_change:
            changed += 1
            if args.apply:
                replace_project_version(file_path.parent, target_version)
            _ = print(f"update: {file_path}")

    if args.check:
        _ = print(f"checked_version={target_version}")
    _ = print(f"files_changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
