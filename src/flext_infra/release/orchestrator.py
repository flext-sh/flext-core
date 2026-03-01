"""Release orchestration service.

Manages the full release lifecycle through composable phases: validate,
version, build, and publish. Each phase returns FlextResult for railway-style
error handling. Migrated from scripts/release/run.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import override

from flext_core import FlextLogger, FlextService, r, t

from flext_infra import (
    FlextInfraCommandRunner,
    FlextInfraGitService,
    FlextInfraJsonService,
    FlextInfraProjectSelector,
    FlextInfraReportingService,
    FlextInfraVersioningService,
)
from flext_infra.constants import c

logger = FlextLogger.create_module_logger(__name__)


class FlextInfraReleaseOrchestrator(FlextService[bool]):
    """Service for release lifecycle orchestration.

    Composes infrastructure services to manage the full release process
    through four distinct phases: validate, version, build, and publish.

    Example:
        service = FlextInfraReleaseOrchestrator()
        result = service.run_release(
            root=Path("."),
            version="1.0.0",
            tag="v1.0.0",
            phases=["validate", "version", "build", "publish"],
        )

    """

    @override
    def execute(self) -> r[bool]:
        """Not used directly; call run_release() or individual phase methods."""
        return r[bool].ok(True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_release(
        self,
        root: Path,
        version: str,
        tag: str,
        phases: list[str],
        *,
        project_names: list[str] | None = None,
        dry_run: bool = False,
        push: bool = False,
        dev_suffix: bool = False,
        create_branches: bool = True,
        next_dev: bool = False,
        next_bump: str = "minor",
    ) -> r[bool]:
        """Run the release process through specified phases.

        Args:
            root: Workspace root directory.
            version: Target semantic version string.
            tag: Git tag for the release (e.g. "v1.0.0").
            phases: Ordered list of phases to execute.
            project_names: Optional project filter. Empty means all.
            dry_run: If True, skip destructive operations.
            push: If True, push tag and branch to remote.
            dev_suffix: If True, append "-dev" to version.
            create_branches: If True, create release branches.
            next_dev: If True, bump to next dev version after release.
            next_bump: Bump type for next dev version.

        Returns:
            FlextResult[bool] with True on success.

        """
        names = project_names or []

        for phase in phases:
            if phase not in c.Infra.Release.VALID_PHASES:
                return r[bool].fail(f"invalid phase: {phase}")

        logger.info(
            "release_run_started",
            release_version=version,
            release_tag=tag,
            phases=phases,
            projects=names,
        )

        if create_branches and not dry_run:
            branch_result = self._create_branches(root, version, names)
            if branch_result.is_failure:
                return branch_result

        for phase in phases:
            result = self._dispatch_phase(
                phase,
                root,
                version,
                tag,
                names,
                dry_run=dry_run,
                push=push,
                dev_suffix=dev_suffix,
            )
            if result.is_failure:
                return result

        if next_dev and not dry_run:
            return self._bump_next_dev(root, version, names, next_bump)

        logger.info("release_run_completed", status="ok")
        return r[bool].ok(True)

    def phase_validate(self, root: Path, *, dry_run: bool = False) -> r[bool]:
        """Execute the validation phase.

        Runs ``make validate VALIDATE_SCOPE=workspace`` to ensure the
        workspace is in a releasable state.

        Args:
            root: Workspace root directory.
            dry_run: If True, report what would run without executing.

        Returns:
            FlextResult[bool] with True on success.

        """
        if dry_run:
            logger.info("release_phase_validate", action="dry-run", status="ok")
            return r[bool].ok(True)
        return FlextInfraCommandRunner().run_checked(
            ["make", "validate", "VALIDATE_SCOPE=workspace"],
            cwd=root,
        )

    def phase_version(
        self,
        root: Path,
        version: str,
        project_names: list[str],
        *,
        dry_run: bool = False,
        dev_suffix: bool = False,
    ) -> r[bool]:
        """Execute the versioning phase.

        Updates version fields in pyproject.toml across workspace and
        selected projects.

        Args:
            root: Workspace root directory.
            version: Target semantic version string.
            project_names: Projects to update. Empty means all.
            dry_run: If True, report changes without applying.
            dev_suffix: If True, append "-dev" to the version.

        Returns:
            FlextResult[bool] with True on success.

        """
        versioning = FlextInfraVersioningService()
        target = f"{version}-dev" if dev_suffix else version

        parse_result = versioning.parse_semver(version)
        if parse_result.is_failure:
            return r[bool].fail(parse_result.error or "invalid version")

        files = self._version_files(root, project_names)
        changed = 0
        for path in files:
            if not path.exists():
                continue
            content = path.read_text(encoding=c.Encoding.DEFAULT)
            match = c.Infra.Release.VERSION_RE.search(content)
            if match and match.group(1) == target:
                continue
            changed += 1
            if not dry_run:
                versioning.replace_project_version(path.parent, target)
            logger.info(
                "release_version_file_updated",
                path=str(path),
                target=target,
            )

        if dry_run:
            logger.info("release_phase_version_checked", checked_version=target)
        logger.info("release_phase_version_summary", files_changed=changed)
        return r[bool].ok(True)

    def phase_build(
        self,
        root: Path,
        version: str,
        project_names: list[str],
    ) -> r[bool]:
        """Execute the build phase.

        Runs ``make build`` for each target project and generates a
        JSON build report.

        Args:
            root: Workspace root directory.
            version: Release version for report naming.
            project_names: Projects to build. Empty means all.

        Returns:
            FlextResult[bool] with True if all builds succeed.

        """
        reporting = FlextInfraReportingService()
        output_dir = (
            reporting.get_report_dir(root, "project", "release") / f"v{version}"
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return r[bool].fail(f"report dir creation failed: {exc}")

        targets = self._build_targets(root, project_names)
        records: list[Mapping[str, str | int]] = []
        failures = 0

        for name, path in targets:
            make_result = self._run_make(path, "build")
            if make_result.is_failure:
                code = 1
                output = make_result.error or "make execution failed"
            else:
                code, output = make_result.value
            if code != 0:
                failures += 1
            log = output_dir / f"build-{name}.log"
            log.write_text(output + "\n", encoding=c.Encoding.DEFAULT)
            records.append({
                "project": name,
                "path": str(path),
                "exit_code": code,
                "log": str(log),
            })
            logger.info("release_phase_build_project", project=name, exit_code=code)

        report: Mapping[str, t.ConfigMapValue] = {
            "version": version,
            "total": len(records),
            "failures": failures,
            "records": records,
        }
        FlextInfraJsonService().write(
            output_dir / "build-report.json",
            report,
            sort_keys=True,
        )
        logger.info(
            "release_phase_build_report",
            report=str(output_dir / "build-report.json"),
        )

        if failures:
            return r[bool].fail(f"build failed: {failures} project(s)")
        return r[bool].ok(True)

    def phase_publish(
        self,
        root: Path,
        version: str,
        tag: str,
        project_names: list[str],
        *,
        dry_run: bool = False,
        push: bool = False,
    ) -> r[bool]:
        """Execute the publishing phase.

        Generates release notes, updates the changelog, creates a Git tag,
        and optionally pushes to the remote.

        Args:
            root: Workspace root directory.
            version: Release version string.
            tag: Git tag for the release.
            project_names: Projects included in the release.
            dry_run: If True, generate notes but skip changelog/tag/push.
            push: If True, push branch and tag to remote.

        Returns:
            FlextResult[bool] with True on success.

        """
        reporting = FlextInfraReportingService()
        notes_dir = reporting.get_report_dir(root, "project", "release") / tag
        try:
            notes_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return r[bool].fail(f"report dir creation failed: {exc}")

        notes_path = notes_dir / "RELEASE_NOTES.md"
        notes_result = self._generate_notes(
            root,
            version,
            tag,
            project_names,
            notes_path,
        )
        if notes_result.is_failure:
            return notes_result

        if not dry_run:
            changelog_result = self._update_changelog(root, version, tag, notes_path)
            if changelog_result.is_failure:
                return changelog_result

            tag_result = self._create_tag(root, tag)
            if tag_result.is_failure:
                return tag_result

            if push:
                push_result = self._push_release(root, tag)
                if push_result.is_failure:
                    return push_result

        logger.info("release_phase_publish", tag=tag, dry_run=dry_run)
        return r[bool].ok(True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dispatch_phase(
        self,
        phase: str,
        root: Path,
        version: str,
        tag: str,
        project_names: list[str],
        *,
        dry_run: bool,
        push: bool,
        dev_suffix: bool,
    ) -> r[bool]:
        """Route to the correct phase method."""
        if phase == "validate":
            return self.phase_validate(root, dry_run=dry_run)
        if phase == "version":
            return self.phase_version(
                root,
                version,
                project_names,
                dry_run=dry_run,
                dev_suffix=dev_suffix,
            )
        if phase == "build":
            return self.phase_build(root, version, project_names)
        if phase == "publish":
            return self.phase_publish(
                root,
                version,
                tag,
                project_names,
                dry_run=dry_run,
                push=push,
            )
        return r[bool].fail(f"unknown phase: {phase}")

    def _create_branches(
        self,
        root: Path,
        version: str,
        project_names: list[str],
    ) -> r[bool]:
        """Create local release branches for workspace and projects."""
        runner = FlextInfraCommandRunner()
        branch = f"release/{version}"
        result = runner.run_checked(
            ["git", "checkout", "-B", branch],
            cwd=root,
        )
        if result.is_failure:
            return result

        selector = FlextInfraProjectSelector()
        projects_result = selector.resolve_projects(root, project_names)
        if projects_result.is_success:
            for project in projects_result.value:
                proj_result = runner.run_checked(
                    ["git", "checkout", "-B", branch],
                    cwd=project.path,
                )
                if proj_result.is_failure:
                    return proj_result

        return r[bool].ok(True)

    def _version_files(
        self,
        root: Path,
        project_names: list[str],
    ) -> list[Path]:
        """Discover pyproject.toml files that need version updates."""
        files: list[Path] = [root / c.Files.PYPROJECT_FILENAME]
        selector = FlextInfraProjectSelector()
        projects_result = selector.resolve_projects(root, project_names)
        if projects_result.is_success:
            for project in projects_result.value:
                pyproject = project.path / c.Files.PYPROJECT_FILENAME
                if pyproject.exists():
                    files.append(pyproject)
        return sorted({path.resolve() for path in files if path.exists()})

    def _build_targets(
        self,
        root: Path,
        project_names: list[str],
    ) -> list[tuple[str, Path]]:
        """Resolve unique build targets from project names."""
        targets: list[tuple[str, Path]] = [("root", root)]
        selector = FlextInfraProjectSelector()
        projects_result = selector.resolve_projects(root, project_names)
        if projects_result.is_success:
            targets.extend((p.name, p.path) for p in projects_result.value)

        seen: set[str] = set()
        unique: list[tuple[str, Path]] = []
        for name, path in targets:
            if name in seen or not path.exists():
                continue
            seen.add(name)
            unique.append((name, path))
        return unique

    @staticmethod
    def _run_make(project_path: Path, verb: str) -> r[tuple[int, str]]:
        """Execute a make command for a project and return (exit_code, output)."""
        result = FlextInfraCommandRunner().run_raw([
            "make",
            "-C",
            str(project_path),
            verb,
        ])
        if result.is_failure:
            return r[tuple[int, str]].fail(result.error or "make execution failed")

        output_model = result.value
        output = (output_model.stdout + "\n" + output_model.stderr).strip()
        return r[tuple[int, str]].ok((output_model.exit_code, output))

    def _generate_notes(
        self,
        root: Path,
        version: str,
        tag: str,
        project_names: list[str],
        output_path: Path,
    ) -> r[bool]:
        """Generate release notes from Git history."""
        previous_result = self._previous_tag(root, tag)
        previous = previous_result.value if previous_result.is_success else ""
        changes_result = self._collect_changes(root, previous, tag)
        changes = changes_result.value if changes_result.is_success else ""

        selector = FlextInfraProjectSelector()
        projects_result = selector.resolve_projects(root, project_names)
        project_list = projects_result.value if projects_result.is_success else []

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
        lines.extend(f"- {p.name}" for p in project_list)
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
            output_path.write_text(
                "\n".join(lines).rstrip() + "\n",
                encoding=c.Encoding.DEFAULT,
            )
            logger.info("release_notes_written", path=str(output_path))
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"failed to write release notes: {exc}")

    def _previous_tag(self, root: Path, tag: str) -> r[str]:
        """Find the tag immediately preceding the given tag."""
        runner = FlextInfraCommandRunner()
        result = runner.capture(
            ["git", "tag", "--sort=-v:refname"],
            cwd=root,
        )
        if result.is_failure:
            return r[str].fail(result.error or "failed to list tags")
        tags = [line.strip() for line in result.value.splitlines() if line.strip()]
        if tag in tags:
            idx = tags.index(tag)
            if idx + 1 < len(tags):
                return r[str].ok(tags[idx + 1])
        for candidate in tags:
            if candidate != tag:
                return r[str].ok(candidate)
        return r[str].ok("")

    def _collect_changes(self, root: Path, previous: str, tag: str) -> r[str]:
        """Collect Git commit messages between two tags."""
        git = FlextInfraGitService()
        tag_result = git.tag_exists(root, tag)
        target = tag if (tag_result.is_success and tag_result.value) else "HEAD"
        rev = f"{previous}..{target}" if previous else target

        runner = FlextInfraCommandRunner()
        result = runner.capture(
            ["git", "log", "--pretty=format:- %h %s (%an)", rev],
            cwd=root,
        )
        if result.is_failure:
            return r[str].fail(result.error or "failed to collect git changes")
        return r[str].ok(result.value)

    def _update_changelog(
        self,
        root: Path,
        version: str,
        tag: str,
        notes_path: Path,
    ) -> r[bool]:
        """Update changelog and release notes files."""
        changelog_path = root / "docs" / "CHANGELOG.md"
        latest_path = root / "docs" / "releases" / "latest.md"
        tagged_path = root / "docs" / "releases" / f"{tag}.md"

        try:
            notes_text = notes_path.read_text(encoding=c.Encoding.DEFAULT)
            existing = (
                changelog_path.read_text(encoding=c.Encoding.DEFAULT)
                if changelog_path.exists()
                else "# Changelog\n\n"
            )

            date = datetime.now(UTC).date().isoformat()
            heading = f"## {version} - "
            section = (
                f"{heading}{date}\n\n"
                f"- Workspace release tag: `{tag}`\n"
                "- Status: Alpha, non-production\n\n"
                f"Full notes: `docs/releases/{tag}.md`\n\n"
            )

            if heading not in existing:
                marker = "# Changelog\n\n"
                if marker in existing:
                    updated = existing.replace(marker, marker + section, 1)
                else:
                    updated = "# Changelog\n\n" + section + existing
            else:
                updated = existing

            changelog_path.parent.mkdir(parents=True, exist_ok=True)
            changelog_path.write_text(updated, encoding=c.Encoding.DEFAULT)
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            latest_path.write_text(notes_text, encoding=c.Encoding.DEFAULT)
            tagged_path.write_text(notes_text, encoding=c.Encoding.DEFAULT)

            logger.info("release_changelog_written", path=str(changelog_path))
            logger.info("release_tagged_notes_written", path=str(tagged_path))
            return r[bool].ok(True)
        except OSError as exc:
            return r[bool].fail(f"changelog update failed: {exc}")

    def _create_tag(self, root: Path, tag: str) -> r[bool]:
        """Create an annotated Git tag if it doesn't exist."""
        git = FlextInfraGitService()
        exists_result = git.tag_exists(root, tag)
        if exists_result.is_success and exists_result.value:
            return r[bool].ok(True)
        runner = FlextInfraCommandRunner()
        return runner.run_checked(
            ["git", "tag", "-a", tag, "-m", f"release: {tag}"],
            cwd=root,
        )

    def _push_release(self, root: Path, tag: str) -> r[bool]:
        """Push branch and tag to remote origin."""
        runner = FlextInfraCommandRunner()
        result = runner.run_checked(["git", "push", "origin", "HEAD"], cwd=root)
        if result.is_failure:
            return result
        return runner.run_checked(["git", "push", "origin", tag], cwd=root)

    def _bump_next_dev(
        self,
        root: Path,
        version: str,
        project_names: list[str],
        bump: str,
    ) -> r[bool]:
        """Bump to the next development version."""
        versioning = FlextInfraVersioningService()
        bump_result = versioning.bump_version(version, bump)
        if bump_result.is_failure:
            return r[bool].fail(bump_result.error or "bump failed")
        next_version = bump_result.value
        result = self.phase_version(
            root,
            next_version,
            project_names,
            dev_suffix=True,
        )
        if result.is_success:
            logger.info("release_next_dev_version", version=f"{next_version}-dev")
        return result


__all__ = ["FlextInfraReleaseOrchestrator"]
