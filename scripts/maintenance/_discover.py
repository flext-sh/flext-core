#!/usr/bin/env python3
# Owner-Skill: .claude/skills/workspace-maintenance/SKILL.md
"""Discover workspace projects for maintenance workflows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.libs.discovery import discover_projects


def main() -> int:
    """List discovered projects in requested output format."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument(
        "--kind", choices=("submodule", "external", "all"), default="all"
    )
    _ = parser.add_argument(
        "--format", choices=("human", "makefile", "json"), default="human"
    )
    _ = parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    projects = discover_projects(args.workspace_root.resolve())
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
