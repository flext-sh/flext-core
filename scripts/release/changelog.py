#!/usr/bin/env python3
"""Script to update the CHANGELOG.md file and release notes."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

# pylint: disable=wrong-import-position
from release.shared import workspace_root  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the changelog script."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--root", type=Path, default=Path())
    _ = parser.add_argument("--version", required=True)
    _ = parser.add_argument("--tag", required=True)
    _ = parser.add_argument("--notes", type=Path, required=True)
    _ = parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def _update_changelog(existing: str, version: str, tag: str) -> str:
    """Insert a new release entry into the changelog content.

    Args:
        existing: The current content of CHANGELOG.md.
        version: The version string being released.
        tag: The Git tag associated with the release.

    Returns:
        The updated changelog content.

    """
    date = datetime.now(UTC).date().isoformat()
    heading = f"## {version} - "
    section = (
        f"{heading}{date}\n\n"
        f"- Workspace release tag: `{tag}`\n"
        "- Status: Alpha, non-production\n\n"
        f"Full notes: `docs/releases/{tag}.md`\n\n"
    )
    if heading in existing:
        return existing
    marker = "# Changelog\n\n"
    if marker in existing:
        return existing.replace(marker, marker + section, 1)
    return "# Changelog\n\n" + section + existing


def main() -> int:
    """Execute the changelog and release notes update process.

    Returns:
        0 on success.

    """
    args = _parse_args()
    root = workspace_root(args.root)
    changelog_path = root / "docs" / "CHANGELOG.md"
    latest_path = root / "docs" / "releases" / "latest.md"
    tagged_notes_path = root / "docs" / "releases" / f"{args.tag}.md"
    notes_path = args.notes if args.notes.is_absolute() else root / args.notes

    notes_text = notes_path.read_text(encoding="utf-8")
    existing = (
        changelog_path.read_text(encoding="utf-8")
        if changelog_path.exists()
        else "# Changelog\n\n"
    )
    updated = _update_changelog(existing, args.version, args.tag)

    if args.apply:
        changelog_path.parent.mkdir(parents=True, exist_ok=True)
        _ = changelog_path.write_text(updated, encoding="utf-8")
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        _ = latest_path.write_text(notes_text, encoding="utf-8")
        _ = tagged_notes_path.write_text(notes_text, encoding="utf-8")

    _ = print(f"changelog: {changelog_path}")
    _ = print(f"latest: {latest_path}")
    _ = print(f"release_notes: {tagged_notes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
