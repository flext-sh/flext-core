#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from release.shared import parse_semver, resolve_projects, workspace_root


def _replace_version(content: str, version: str) -> tuple[str, bool]:
    project_match = re.search(r"(?ms)^\[project\]\n(?P<body>.*?)(?:^\[|\Z)", content)
    if not project_match:
        return content, False

    body = project_match.group("body")
    version_match = re.search(r'(?m)^version\s*=\s*"(?P<value>[^"]+)"\s*$', body)
    if not version_match:
        return content, False

    current = version_match.group("value")
    current_clean = current.removesuffix("-dev")
    _ = parse_semver(current_clean)
    if current == version:
        return content, False

    replacement = f'version = "{version}"'
    updated_body = re.sub(
        r'(?m)^version\s*=\s*"[^"]+"\s*$',
        replacement,
        body,
        count=1,
    )
    start, end = project_match.span("body")
    updated = content[:start] + updated_body + content[end:]
    return updated, updated != content


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
    _ = parser.add_argument("--projects", nargs="*", default=[])
    _ = parser.add_argument("--apply", action="store_true")
    _ = parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = workspace_root(args.root)
    _ = parse_semver(args.version)

    changed = 0
    for file_path in _version_files(root, args.projects):
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
