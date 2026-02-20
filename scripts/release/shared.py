#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
from __future__ import annotations

import re
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


SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)


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
    match = SEMVER_RE.match(version)
    if not match:
        raise ValueError(f"invalid semver version: {version}")
    return (
        int(match.group("major")),
        int(match.group("minor")),
        int(match.group("patch")),
    )


def bump_version(current_version: str, bump: str) -> str:
    major, minor, patch = parse_semver(current_version)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"unsupported bump: {bump}")


def run_checked(command: list[str], cwd: Path | None = None) -> None:
    _run_checked(command, cwd=cwd)


def run_capture(command: list[str], cwd: Path | None = None) -> str:
    return _run_capture(command, cwd=cwd)
