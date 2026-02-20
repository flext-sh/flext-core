#!/usr/bin/env python3
# Owner-Skill: .claude/skills/workspace-maintenance/SKILL.md
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectInfo:
    path: Path
    name: str
    kind: str


def _is_git_project(path: Path) -> bool:
    return (path / ".git").exists()


def _submodule_names(workspace_root: Path) -> set[str]:
    gitmodules = workspace_root / ".gitmodules"
    if not gitmodules.exists():
        return set()
    try:
        content = gitmodules.read_text(encoding="utf-8")
    except OSError:
        return set()
    return set(re.findall(r"^\s*path\s*=\s*(.+?)\s*$", content, re.MULTILINE))


def _discover(workspace_root: Path) -> list[ProjectInfo]:
    projects: list[ProjectInfo] = []
    submodules = _submodule_names(workspace_root)
    for entry in sorted(workspace_root.iterdir(), key=lambda value: value.name):
        if not entry.is_dir() or entry.name == "cmd" or entry.name.startswith("."):
            continue
        if not _is_git_project(entry):
            continue
        if not (entry / "Makefile").exists():
            continue
        if not (entry / "pyproject.toml").exists() and not (entry / "go.mod").exists():
            continue
        kind = "submodule" if entry.name in submodules else "external"
        projects.append(ProjectInfo(path=entry, name=entry.name, kind=kind))
    return projects


def main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--kind", choices=("submodule", "external", "all"), default="all"
    )
    _ = parser.add_argument(
        "--format", choices=("human", "makefile", "json"), default="human"
    )
    _ = parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    projects = _discover(args.workspace_root.resolve())
    if args.kind != "all":
        projects = [p for p in projects if p.kind == args.kind]

    if args.format == "makefile":
        print(" ".join(project.name for project in projects))
        return 0

    if args.format == "json":
        payload = {
            "workspace_root": str(args.workspace_root.resolve()),
            "kind": args.kind,
            "count": len(projects),
            "projects": [
                {
                    "name": project.name,
                    "kind": project.kind,
                    "path": str(project.path.resolve()),
                }
                for project in projects
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    for project in projects:
        print(project.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
