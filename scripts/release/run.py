#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tomllib

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from release.shared import (
    bump_version,
    parse_semver,
    resolve_projects,
    run_capture,
    run_checked,
    workspace_root,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--root", type=Path, default=Path("."))
    _ = parser.add_argument("--phase", default="all")
    _ = parser.add_argument("--version", default="")
    _ = parser.add_argument("--tag", default="")
    _ = parser.add_argument("--bump", default="")
    _ = parser.add_argument("--interactive", type=int, default=1)
    _ = parser.add_argument("--push", type=int, default=0)
    _ = parser.add_argument("--dry-run", type=int, default=0)
    _ = parser.add_argument("--create-branches", type=int, default=1)
    _ = parser.add_argument("--projects", nargs="*", default=[])
    return parser.parse_args()


def _current_version(root: Path) -> str:
    pyproject = root / "pyproject.toml"
    content = pyproject.read_bytes()
    data = tomllib.loads(content.decode("utf-8"))
    project = data.get("project")
    if not isinstance(project, dict):
        raise RuntimeError("unable to detect [project] section from pyproject.toml")
    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise RuntimeError("unable to detect version from pyproject.toml")
    return version.removesuffix("-dev")


def _resolve_version(args: argparse.Namespace, root: Path) -> str:
    if args.version:
        _ = parse_semver(args.version)
        return args.version

    current = _current_version(root)
    if args.bump:
        return bump_version(current, args.bump)

    if args.interactive != 1:
        return current

    print("Select version bump type: [major|minor|patch]")
    bump = input("bump> ").strip().lower()
    if bump not in {"major", "minor", "patch"}:
        raise RuntimeError("invalid bump type")
    return bump_version(current, bump)


def _resolve_tag(args: argparse.Namespace, version: str) -> str:
    if args.tag:
        if not args.tag.startswith("v"):
            raise RuntimeError("tag must start with v")
        return args.tag
    return f"v{version}"


def _create_release_branches(
    root: Path, version: str, selected_projects: list[Path]
) -> None:
    branch = f"release/{version}"
    run_checked(["git", "checkout", "-B", branch], cwd=root)
    for project_path in selected_projects:
        run_checked(["git", "checkout", "-B", branch], cwd=project_path)


def _phase_version(
    root: Path, version: str, dry_run: bool, project_names: list[str]
) -> None:
    command = [
        "python",
        "scripts/release/version.py",
        "--root",
        str(root),
        "--version",
        version,
        "--check" if dry_run else "--apply",
    ]
    if project_names:
        command.extend(["--projects", *project_names])
    run_checked(command, cwd=root)


def _phase_validate(root: Path) -> None:
    run_checked(["make", "validate", "VALIDATE_SCOPE=workspace"], cwd=root)


def _phase_build(root: Path, version: str, project_names: list[str]) -> None:
    output = root / ".reports" / "release" / f"v{version}"
    command = [
        "python",
        "scripts/release/build.py",
        "--root",
        str(root),
        "--version",
        version,
        "--output-dir",
        str(output),
    ]
    if project_names:
        command.extend(["--projects", *project_names])
    run_checked(command, cwd=root)


def _phase_publish(
    root: Path,
    version: str,
    tag: str,
    push: bool,
    dry_run: bool,
    project_names: list[str],
) -> None:
    notes = root / ".reports" / "release" / tag / "RELEASE_NOTES.md"
    notes.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "scripts/release/notes.py",
        "--root",
        str(root),
        "--tag",
        tag,
        "--version",
        version,
        "--output",
        str(notes),
    ]
    if project_names:
        command.extend(["--projects", *project_names])
    run_checked(command, cwd=root)
    if not dry_run:
        run_checked(
            [
                "python",
                "scripts/release/changelog.py",
                "--root",
                str(root),
                "--version",
                version,
                "--tag",
                tag,
                "--notes",
                str(notes),
                "--apply",
            ],
            cwd=root,
        )
        tag_exists = run_capture(["git", "tag", "-l", tag], cwd=root)
        if tag_exists.strip() != tag:
            run_checked(["git", "tag", "-a", tag, "-m", f"release: {tag}"], cwd=root)
        if push:
            run_checked(["git", "push", "origin", "HEAD"], cwd=root)
            run_checked(["git", "push", "origin", tag], cwd=root)


def main() -> int:
    args = _parse_args()
    root = workspace_root(args.root)
    selected_projects = resolve_projects(root, args.projects)
    selected_project_names = [project.name for project in selected_projects]
    selected_project_paths = [project.path for project in selected_projects]
    version = _resolve_version(args, root)
    tag = _resolve_tag(args, version)
    phases = (
        ["validate", "version", "build", "publish"]
        if args.phase == "all"
        else [part.strip() for part in args.phase.split(",") if part.strip()]
    )

    _ = print(f"release_version={version}")
    _ = print(f"release_tag={tag}")
    _ = print(f"phases={','.join(phases)}")
    _ = print(f"projects={','.join(selected_project_names)}")

    if args.create_branches == 1 and args.dry_run == 0:
        _create_release_branches(root, version, selected_project_paths)

    for phase in phases:
        if phase == "validate":
            _phase_validate(root)
            continue
        if phase == "version":
            _phase_version(root, version, args.dry_run == 1, selected_project_names)
            continue
        if phase == "build":
            _phase_build(root, version, selected_project_names)
            continue
        if phase == "publish":
            _phase_publish(
                root,
                version,
                tag,
                args.push == 1,
                args.dry_run == 1,
                selected_project_names,
            )
            continue
        raise RuntimeError(f"invalid phase: {phase}")

    _ = print("release_run=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
