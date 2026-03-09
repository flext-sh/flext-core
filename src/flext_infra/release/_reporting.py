"""Release notes and changelog helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from flext_core import FlextLogger, r
from flext_infra import c, m, u

logger = FlextLogger.create_module_logger(__name__)


class FlextInfraReleaseReporting:
    """Helpers for release-note generation and changelog synchronization."""

    @staticmethod
    def generate_notes(
        version: str,
        tag: str,
        project_list: list[m.Infra.Workspace.ProjectInfo],
        changes: str,
        output_path: Path,
    ) -> r[bool]:
        """Generate release notes markdown from release context."""
        lines: list[str] = [
            f"# Release {tag}",
            "",
            "## Status",
            "",
            "- Quality: Alpha",
            "- Usage: Non-production",
            "",
            "## Scope",
            "",
            f"- Workspace release version: {version}",
            f"- Projects packaged: {len(project_list) + 1}",
            "",
            "## Projects impacted",
            "",
            "- root",
        ]
        lines.extend(f"- {proj.name}" for proj in project_list)
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
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            u.write_file(
                output_path,
                "\n".join(lines).rstrip() + "\n",
                encoding=c.Infra.Encoding.DEFAULT,
            )
            logger.info("release_notes_written", path=str(output_path))
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"failed to write release notes: {exc}")

    @staticmethod
    def update_changelog(
        root: Path, version: str, tag: str, notes_path: Path
    ) -> r[bool]:
        """Update docs/changelog and docs/releases entries."""
        changelog_path = root / c.Infra.Directories.DOCS / "CHANGELOG.md"
        latest_path = root / c.Infra.Directories.DOCS / "releases" / "latest.md"
        tagged_path = root / c.Infra.Directories.DOCS / "releases" / f"{tag}.md"
        try:
            notes_text = notes_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            existing = (
                changelog_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
                if changelog_path.exists()
                else "# Changelog\n\n"
            )
            date = datetime.now(UTC).date().isoformat()
            heading = f"## {version} - "
            section = f"{heading}{date}\n\n- Workspace release tag: `{tag}`\n- Status: Alpha, non-production\n\nFull notes: `docs/releases/{tag}.md`\n\n"
            if heading not in existing:
                marker = "# Changelog\n\n"
                if marker in existing:
                    updated = existing.replace(marker, marker + section, 1)
                else:
                    updated = "# Changelog\n\n" + section + existing
            else:
                updated = existing
            changelog_path.parent.mkdir(parents=True, exist_ok=True)
            u.write_file(changelog_path, updated, encoding=c.Infra.Encoding.DEFAULT)
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            u.write_file(latest_path, notes_text, encoding=c.Infra.Encoding.DEFAULT)
            u.write_file(tagged_path, notes_text, encoding=c.Infra.Encoding.DEFAULT)
            logger.info("release_changelog_written", path=str(changelog_path))
            logger.info("release_tagged_notes_written", path=str(tagged_path))
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"changelog update failed: {exc}")


__all__ = ["FlextInfraReleaseReporting"]
