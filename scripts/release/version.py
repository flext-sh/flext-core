#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from release.shared import parse_semver, resolve_projects, workspace_root
from libs.versioning import replace_project_version


def _replace_version(content: str, version: str) -> tuple[str, bool]:
    return replace_project_version(content, version)


def _version_files(root: Path, project_names: list[str]) -> list[Path]:
    files: list[Path] = [root / "pyproject.toml"]
    for project in resolve_projects(root, project_names):
        pyproject = project.path / "pyproject.toml"
        if pyproject.exists():
            files.append(pyproject)
    dedup = sorted({path.resolve() for path in files})
    return dedup


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--root", type=Path, default=Path("."))
    _ = parser.add_argument("--version", required=True)
    _ = parser.add_argument("--dev-suffix", type=int, default=0)
    _ = parser.add_argument("--projects", nargs="*", default=[])
    _ = parser.add_argument("--apply", action="store_true")
    _ = parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = workspace_root(args.root)
    _ = parse_semver(args.version)
    target_version = f"{args.version}-dev" if args.dev_suffix == 1 else args.version

    changed = 0
    for file_path in _version_files(root, args.projects):
        content = file_path.read_text(encoding="utf-8")
        updated, did_change = _replace_version(content, target_version)
        if did_change:
            changed += 1
            if args.apply:
                _ = file_path.write_text(updated, encoding="utf-8")
            _ = print(f"update: {file_path}")

    if args.check:
        _ = print(f"checked_version={target_version}")
    _ = print(f"files_changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
