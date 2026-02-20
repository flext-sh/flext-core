#!/usr/bin/env python3
# Owner-Skill: .claude/skills/scripts-maintenance/SKILL.md
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$"
)


@dataclass(frozen=True)
class Project:
    name: str
    path: Path


def workspace_root(path: str | Path = ".") -> Path:
    return Path(path).resolve()


def discover_projects(root: Path) -> list[Project]:
    discover = root / "scripts" / "maintenance" / "_discover.py"
    command = [
        sys.executable,
        str(discover),
        "--workspace-root",
        str(root),
        "--kind",
        "all",
        "--format",
        "json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"project discovery failed: {msg}")
    payload = json.loads(result.stdout)
    projects: list[Project] = []
    for item in payload.get("projects", []):
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        path_value = item.get("path")
        if not isinstance(name, str) or not isinstance(path_value, str):
            continue
        projects.append(Project(name=name, path=Path(path_value).resolve()))
    return sorted(projects, key=lambda project: project.name)


def resolve_projects(root: Path, names: list[str]) -> list[Project]:
    projects = discover_projects(root)
    if not names:
        return projects

    by_name = {project.name: project for project in projects}
    missing = [name for name in names if name not in by_name]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise RuntimeError(f"unknown release projects: {missing_text}")

    resolved = [by_name[name] for name in names]
    return sorted(resolved, key=lambda project: project.name)


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
    result = subprocess.run(command, cwd=cwd, check=False)
    if result.returncode != 0:
        cmd = shlex.join(command)
        raise RuntimeError(f"command failed ({result.returncode}): {cmd}")


def run_capture(command: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        cmd = shlex.join(command)
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"command failed ({result.returncode}): {cmd}: {detail}")
    return result.stdout.strip()
