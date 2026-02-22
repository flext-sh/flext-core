#!/usr/bin/env python3
"""Script to generate release notes based on Git history and project state."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.libs.config import DEFAULT_ENCODING
from scripts.libs.paths import workspace_root
from scripts.libs.selection import resolve_projects
from scripts.libs.subprocess import run_capture


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the release notes script."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--root", type=Path, default=Path())
    _ = parser.add_argument("--tag", required=True)
    _ = parser.add_argument("--output", type=Path, required=True)
    _ = parser.add_argument("--version", default="")
    _ = parser.add_argument("--projects", nargs="*", default=[])
    return parser.parse_args()


def _tag_exists(root: Path, tag: str) -> bool:
    """Check if a Git tag exists in the repository."""
    try:
        _ = run_capture(["git", "rev-parse", "--verify", f"refs/tags/{tag}"], cwd=root)
        return True
    except RuntimeError:
        return False


def _previous_tag(root: Path, tag: str) -> str:
    """Find the tag immediately preceding the given tag in history."""
    output = run_capture(["git", "tag", "--sort=-v:refname"], cwd=root)
    tags = [line.strip() for line in output.splitlines() if line.strip()]
    if tag in tags:
        idx = tags.index(tag)
        if idx + 1 < len(tags):
            return tags[idx + 1]
    for candidate in tags:
        if candidate != tag:
            return candidate
    return ""


def _collect_changes(root: Path, previous: str, tag: str) -> str:
    """Collect Git commit messages between two tags."""
    target = tag if _tag_exists(root, tag) else "HEAD"
    rev = f"{previous}..{target}" if previous else target
    return run_capture(["git", "log", "--pretty=format:- %h %s (%an)", rev], cwd=root)


def main() -> int:
    """Execute the release notes generation process.

    Returns:
        0 on success.

    """
    args = _parse_args()
    root = workspace_root(args.root)
    output_path = args.output if args.output.is_absolute() else root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    previous = _previous_tag(root, args.tag)
    changes = _collect_changes(root, previous, args.tag)
    projects = resolve_projects(root, args.projects)

    version = args.version or args.tag.removeprefix("v")
    lines: list[str] = [
        f"# Release {args.tag}",
        "",
        "## Status",
        "",
        "- Quality: Alpha",
        "- Usage: Non-production",
        "",
        "## Scope",
        "",
        f"- Workspace release version: {version}",
        f"- Projects packaged: {len(projects) + 1}",
        "",
        "## Projects impacted",
        "",
    ]
    lines.extend(
        f"- {name}"
        for name in [
            "root",
            *[project.name for project in projects],
        ]
    )
    lines.extend([
        "",
        "## Changes since last tag",
        "",
        changes or "- Initial tagged release",
        "",
        "## Verification",
        "",
        "- make release INTERACTIVE=0 CREATE_BRANCHES=0 RELEASE_PHASE=all",
        "- make validate VALIDATE_SCOPE=workspace",
        "- make build",
    ])

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding=DEFAULT_ENCODING)
    _ = print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
