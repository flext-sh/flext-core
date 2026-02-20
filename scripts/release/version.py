#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from release.shared import discover_projects, parse_semver, workspace_root


def _replace_version(content: str, version: str) -> tuple[str, bool]:
    old = 'version = "0.10.0-dev"'
    new = f'version = "{version}"'
    if old in content:
        return content.replace(old, new), True

    marker = 'version = "'
    start = content.find(marker)
    if start < 0:
        return content, False
    value_start = start + len(marker)
    value_end = content.find('"', value_start)
    if value_end < 0:
        return content, False

    current = content[value_start:value_end]
    current_clean = current.removesuffix("-dev")
    _ = parse_semver(current_clean)
    if current == version:
        return content, False
    updated = content[:value_start] + version + content[value_end:]
    return updated, True


def _version_files(root: Path) -> list[Path]:
    files: list[Path] = [root / "pyproject.toml"]
    for project in discover_projects(root):
        pyproject = project.path / "pyproject.toml"
        if pyproject.exists():
            files.append(pyproject)
    for extra in ("algar-oud-mig", "gruponos-meltano-native"):
        pyproject = root / extra / "pyproject.toml"
        if pyproject.exists():
            files.append(pyproject)
    dedup = sorted({path.resolve() for path in files})
    return dedup


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--root", type=Path, default=Path("."))
    _ = parser.add_argument("--version", required=True)
    _ = parser.add_argument("--apply", action="store_true")
    _ = parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = workspace_root(args.root)
    _ = parse_semver(args.version)

    changed = 0
    for file_path in _version_files(root):
        content = file_path.read_text(encoding="utf-8")
        updated, did_change = _replace_version(content, args.version)
        if did_change:
            changed += 1
            if args.apply:
                _ = file_path.write_text(updated, encoding="utf-8")
            _ = print(f"update: {file_path}")

    if args.check:
        _ = print(f"checked_version={args.version}")
    _ = print(f"files_changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
