#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libs.discovery import ProjectInfo
from libs.paths import workspace_root as _workspace_root
from libs.selection import resolve_projects as _resolve_projects
from libs.subprocess import run_capture as _run_capture
from libs.subprocess import run_checked as _run_checked
from libs.versioning import bump_version as _bump_version
from libs.versioning import parse_semver as _parse_semver


Project = ProjectInfo


def workspace_root(path: str | Path = ".") -> Path:
    return _workspace_root(path)


def resolve_projects(root: Path, names: list[str]) -> list[Project]:
    try:
        return _resolve_projects(root, names)
    except RuntimeError as exc:
        raise RuntimeError(
            str(exc).replace("unknown projects", "unknown release projects")
        ) from exc


def parse_semver(version: str) -> tuple[int, int, int]:
    return _parse_semver(version)


def bump_version(current_version: str, bump: str) -> str:
    return _bump_version(current_version, bump)


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    _run_checked(command, cwd=cwd)


def run_capture(command: list[str], cwd: Path | None = None) -> str:
    return _run_capture(command, cwd=cwd)
