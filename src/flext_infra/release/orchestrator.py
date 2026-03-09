"""Release orchestration service.

Manages the full release lifecycle through composable phases: validate,
version, build, and publish. Each phase returns FlextResult for railway-style
error handling. Migrated from scripts/release/run.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import override

from flext_core import FlextLogger, r, s
from flext_infra import (
    FlextInfraGitService,
    FlextInfraProjectSelector,
    FlextInfraReportingService,
    FlextInfraUtilitiesSubprocess,
    FlextInfraVersioningService,
    c,
    m,
    u,
)
from flext_infra.release._reporting import FlextInfraReleaseReporting

logger = FlextLogger.create_module_logger(__name__)


class FlextInfraReleaseOrchestrator(s[bool]):
    """Service for release lifecycle orchestration."""

    @staticmethod
    def _run_make(project_path: Path, verb: str) -> r[tuple[int, str]]:
        """Execute a make command for a project and return (exit_code, output)."""
        result = FlextInfraUtilitiesSubprocess().run_raw([
            c.Infra.Cli.MAKE,
            "-C",
            str(project_path),
            verb,
        ])
        if result.is_failure:
            return r[tuple[int, str]].fail(result.error or "make execution failed")
        output_model = result.value
        output = (output_model.stdout + "\n" + output_model.stderr).strip()
        return r[tuple[int, str]].ok((output_model.exit_code, output))

    @override
    def execute(self) -> r[bool]:
        """Not used directly; call run_release() or individual phase methods."""
        return r[bool].ok(True)

    def phase_build(
        self,
        root: Path,
        version: str,
        project_names: list[str],
    ) -> r[bool]:
        """Execute the build phase and write build-report.json."""
        reporting = FlextInfraReportingService()
        output_dir = (
            reporting.get_report_dir(
                root,
                c.Infra.Toml.PROJECT,
                c.Infra.ReportKeys.RELEASE,
            )
            / f"v{version}"
        )
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return r[bool].fail(f"report dir creation failed: {exc}")
        targets = self._build_targets(root, project_names)
        records: list[m.Infra.Release.BuildRecord] = []
        failures = 0
        for name, path in targets:
            make_result = self._run_make(path, c.Infra.Directories.BUILD)
            if make_result.is_failure:
                code = 1
                output = make_result.error or "make execution failed"
            else:
                code, output = make_result.value
            if code != 0:
                failures += 1
            log = output_dir / f"build-{name}.log"
            u.write_file(log, output + "\n", encoding=c.Infra.Encoding.DEFAULT)
            records.append(
                m.Infra.Release.BuildRecord(
                    project=name,
                    path=str(path),
                    exit_code=code,
                    log=str(log),
                ),
            )
            logger.info("release_phase_build_project", project=name, exit_code=code)
        report = m.Infra.Release.BuildReport(
            version=version,
            total=len(records),
            failures=failures,
            records=records,
        )
        u.Infra.write_json(
            output_dir / "build-report.json",
            report.model_dump(mode="json"),
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
        """Execute publish phase: notes, changelog, tag, optional push."""
        reporting = FlextInfraReportingService()
        notes_dir = (
            reporting.get_report_dir(
                root,
                c.Infra.Toml.PROJECT,
                c.Infra.ReportKeys.RELEASE,
            )
            / tag
        )
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

    def phase_validate(self, root: Path, *, dry_run: bool = False) -> r[bool]:
        """Execute validation phase via make validate."""
        if dry_run:
            logger.info("release_phase_validate", action="dry-run", status="ok")
            return r[bool].ok(True)
        return FlextInfraUtilitiesSubprocess().run_checked(
            [c.Infra.Cli.MAKE, c.Infra.Verbs.VALIDATE, "VALIDATE_SCOPE=workspace"],
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
        """Execute versioning phase across workspace and selected projects."""
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
            content = path.read_text(encoding=c.Infra.Encoding.DEFAULT)
            match = c.Infra.Release.VERSION_RE.search(content)
            if match and match.group(1) == target:
                continue
            changed += 1
            if not dry_run:
                versioning.replace_project_version(path.parent, target)
            logger.info("release_version_file_updated", path=str(path), target=target)
        if dry_run:
            logger.info("release_phase_version_checked", checked_version=target)
        logger.info("release_phase_version_summary", files_changed=changed)
        return r[bool].ok(True)

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
        """Run release workflow for the provided ordered phases."""
        names = project_names or []
        spec = m.Infra.Release.ReleaseSpec(
            version=version,
            tag=tag,
            bump_type=next_bump,
        )
        for phase in phases:
            if phase not in c.Infra.Release.VALID_PHASES:
                return r[bool].fail(f"invalid phase: {phase}")
        logger.info(
            "release_run_started",
            release_version=spec.version,
            release_tag=spec.tag,
            phases=phases,
            projects=names,
        )
        if create_branches and (not dry_run):
            branch_result = self._create_branches(root, version, names)
            if branch_result.is_failure:
                return branch_result
        for phase in phases:
            result = self._dispatch_phase(
                phase,
                root,
                spec.version,
                spec.tag,
                names,
                dry_run=dry_run,
                push=push,
                dev_suffix=dev_suffix,
            )
            if result.is_failure:
                return result
        if next_dev and (not dry_run):
            return self._bump_next_dev(root, version, names, next_bump)
        logger.info("release_run_completed", status=c.Infra.ReportKeys.OK)
        return r[bool].ok(True)

    def _build_targets(
        self,
        root: Path,
        project_names: list[str],
    ) -> list[tuple[str, Path]]:
        """Resolve unique build targets from project names."""
        targets: list[tuple[str, Path]] = [(c.Infra.ReportKeys.ROOT, root)]
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
        result = self.phase_version(root, next_version, project_names, dev_suffix=True)
        if result.is_success:
            logger.info("release_next_dev_version", version=f"{next_version}-dev")
        return result

    def _collect_changes(self, root: Path, previous: str, tag: str) -> r[str]:
        """Collect Git commit messages between two tags."""
        rev = f"{previous}..{tag}" if previous else tag
        return FlextInfraGitService().log(root, rev)

    def _create_branches(
        self,
        root: Path,
        version: str,
        project_names: list[str],
    ) -> r[bool]:
        """Create local release branches for workspace and projects."""
        git = FlextInfraGitService()
        branch = f"release/{version}"
        result = git.checkout(root, branch, create=True)
        if result.is_failure:
            return result
        selector = FlextInfraProjectSelector()
        projects_result = selector.resolve_projects(root, project_names)
        if projects_result.is_success:
            for project in projects_result.value:
                proj_result = git.checkout(project.path, branch, create=True)
                if proj_result.is_failure:
                    return proj_result
        return r[bool].ok(True)

    def _create_tag(self, root: Path, tag: str) -> r[bool]:
        """Create an annotated Git tag if it does not already exist."""
        git = FlextInfraGitService()
        exists_result = git.tag_exists(root, tag)
        if exists_result.is_failure:
            return r[bool].fail(exists_result.error or "tag check failed")
        if exists_result.value:
            return r[bool].ok(True)
        return git.create_tag(root, tag, f"release: {tag}")

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
        if phase == c.Infra.Verbs.VALIDATE:
            return self.phase_validate(root, dry_run=dry_run)
        if phase == c.Infra.Toml.VERSION:
            return self.phase_version(
                root,
                version,
                project_names,
                dry_run=dry_run,
                dev_suffix=dev_suffix,
            )
        if phase == c.Infra.Directories.BUILD:
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
        project_list: list[m.Infra.Workspace.ProjectInfo] = (
            projects_result.value if projects_result.is_success else []
        )
        return FlextInfraReleaseReporting.generate_notes(
            version,
            tag,
            project_list,
            changes,
            output_path,
        )

    def _previous_tag(self, root: Path, tag: str) -> r[str]:
        """Find the tag immediately preceding the given tag."""
        return FlextInfraGitService().previous_tag(root, tag)

    def _push_release(self, root: Path, tag: str) -> r[bool]:
        """Push branch and tag to remote origin."""
        return FlextInfraGitService().push_release(root, tag)

    def _update_changelog(
        self,
        root: Path,
        version: str,
        tag: str,
        notes_path: Path,
    ) -> r[bool]:
        """Update changelog and release notes files."""
        return FlextInfraReleaseReporting.update_changelog(
            root, version, tag, notes_path
        )

    def _version_files(self, root: Path, project_names: list[str]) -> list[Path]:
        """Discover pyproject.toml files that need version updates."""
        files: list[Path] = [root / c.Infra.Files.PYPROJECT_FILENAME]
        selector = FlextInfraProjectSelector()
        projects_result = selector.resolve_projects(root, project_names)
        if projects_result.is_success:
            for project in projects_result.value:
                pyproject = project.path / c.Infra.Files.PYPROJECT_FILENAME
                if pyproject.exists():
                    files.append(pyproject)
        return sorted({path.resolve() for path in files if path.exists()})


__all__ = ["FlextInfraReleaseOrchestrator"]
