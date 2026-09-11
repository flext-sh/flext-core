"""Versioning utilities for semantic version management.

All methods are static — exposed via u.Infra.parse_semver() etc. through FLEXT.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from packaging.version import InvalidVersion, Version

from flext_cli import r, u
from flext_infra.constants import c
from flext_infra.protocols import p
from flext_infra.typings import t


class FlextInfraUtilitiesVersioning:
    """Static versioning utilities for semantic version management.

    All methods are ``@staticmethod`` — no instantiation required.
    Exposed via ``u.Infra.parse_semver()`` etc. through FLEXT.
    """

    @staticmethod
    def _extract_project_version_from_text(content: str) -> str | None:
        """Extract project version from text."""
        in_project_section = False
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                in_project_section = line == c.Infra.SEMVER_PROJECT_SECTION
                continue
            if not in_project_section or not line.startswith(c.Infra.VERSION):
                continue
            match = c.Infra.VERSION_RE.match(line)
            if match:
                version: str = match.group(1)
                return version
        return None

    @staticmethod
    def _has_project_table(content: str) -> bool:
        """Has project table."""
        return any(
            raw_line.strip() == c.Infra.SEMVER_PROJECT_SECTION
            for raw_line in content.splitlines()
        )

    @staticmethod
    def _replace_project_version_in_text(content: str, version: str) -> str | None:
        """Replace project version in text."""
        lines = content.splitlines(keepends=True)
        in_project_section = False
        updated_lines: t.MutableSequenceOf[str] = []
        replaced = False
        for raw_line in lines:
            line = raw_line.strip()
            if line.startswith("[") and line.endswith("]"):
                in_project_section = line == c.Infra.SEMVER_PROJECT_SECTION
                updated_lines.append(raw_line)
                continue
            if (
                in_project_section
                and line.startswith(c.Infra.VERSION)
                and (not replaced)
            ):
                line_ending = "\n" if raw_line.endswith("\n") else ""
                updated_lines.append(f'version = "{version}"{line_ending}')
                replaced = True
                continue
            updated_lines.append(raw_line)
        if not replaced:
            return None
        return "".join(updated_lines)

    @staticmethod
    def bump_version(
        version: str, bump_type: str | c.Infra.VersionBump
    ) -> p.Result[str]:
        """Return the next release version for one bump kind.

        A pre-release (``0.12.0rc0``) is finalized to its base release by any
        real bump: the base was already reserved when the pre-release was cut,
        so the first releasable change ships it. ``none`` returns the version
        unchanged for both shapes.
        """
        try:
            normalized_bump = c.Infra.VersionBump(bump_type)
        except ValueError:
            return r[str].fail(f"invalid bump type: {bump_type}")
        result = FlextInfraUtilitiesVersioning.parse_semver(version)
        if result.failure:
            return r[str].fail(result.error or "parse failed")
        if normalized_bump == c.Infra.VersionBump.NONE:
            return r[str].ok(version)
        major, minor, patch = result.value
        if Version(version).is_prerelease:
            return r[str].ok(f"{major}.{minor}.{patch}")
        if normalized_bump == c.Infra.VersionBump.MAJOR:
            major += 1
            minor = 0
            patch = 0
        elif normalized_bump == c.Infra.VersionBump.MINOR:
            minor += 1
            patch = 0
        else:
            patch += 1
        return r[str].ok(f"{major}.{minor}.{patch}")

    @staticmethod
    def finalize_version(version: str) -> p.Result[str]:
        """Return the base release of ``version`` (a final version is itself)."""
        result = FlextInfraUtilitiesVersioning.parse_semver(version)
        if result.failure:
            return r[str].fail(result.error or "parse failed")
        major, minor, patch = result.value
        return r[str].ok(f"{major}.{minor}.{patch}")

    @staticmethod
    def current_workspace_version(repository_root: Path) -> p.Result[str]:
        """Read the current version from the main pyproject.toml.

        Args:
            repository_root: The root directory of the workspace.

        Returns:
            r[str] with the version string.

        """
        pyproject = repository_root / c.Infra.PYPROJECT_FILENAME
        try:
            content = pyproject.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        except OSError as exc:
            return r[str].fail_op("read", exc)
        version = FlextInfraUtilitiesVersioning._extract_project_version_from_text(
            content
        )
        if version is None or not version.strip():
            return r[str].fail("version not found in pyproject.toml")
        return r[str].ok(version)

    @staticmethod
    def parse_semver(version: str) -> p.Result[t.Triple[int, int, int]]:
        """Parse one canonical FLEXT PEP 440 version into its release tuple.

        Args:
            version: The version string to parse.

        Returns:
            r with version tuple.

        """
        try:
            parsed = Version(version)
        except InvalidVersion:
            return r[t.Triple[int, int, int]].fail(f"invalid semver: {version}")
        unsupported = (
            str(parsed) != version
            or len(parsed.release) != c.Infra.VERSION_RELEASE_SEGMENTS
            or parsed.epoch != 0
            or parsed.post is not None
            or parsed.local is not None
            or (parsed.pre is not None and parsed.pre[0] != "rc")
            or (parsed.pre is not None and parsed.dev is not None)
            or (parsed.dev is not None and parsed.dev != 0)
        )
        if unsupported:
            return r[t.Triple[int, int, int]].fail(f"invalid semver: {version}")
        major, minor, patch = parsed.release
        return r[t.Triple[int, int, int]].ok((major, minor, patch))

    @staticmethod
    def version_is_newer(candidate: str, reference: str) -> p.Result[bool]:
        """Whether ``candidate`` orders after ``reference`` under PEP 440.

        Pre-release segments take part in the ordering: ``0.12.0`` is newer
        than ``0.12.0rc2`` although both share the release triple.
        """
        try:
            return r[bool].ok(Version(candidate) > Version(reference))
        except InvalidVersion as exc:
            return r[bool].fail(f"invalid version: {exc}")

    @staticmethod
    def latest_release_tag(tags: t.StrSequence) -> p.Result[str]:
        """Return the highest release tag under PEP 440, or ``""`` when none exist.

        Git's ``--sort=version:refname`` is a refname collation, not a PEP 440
        ordering: it places ``v0.12.0rc2`` above ``v0.12.0`` because the longer
        refname collates later. The release protocol then read the newest
        release as a release candidate, decided the released version still
        "awaits its tag", and never bumped again in any repository that had
        ever cut an rc (flext-1wjg1.16.34). Order by the same PEP 440 owner
        ``version_is_newer`` already uses, and fail loud on a ``v*`` tag that
        is not a version rather than silently ranking it.
        """
        prefix = c.Infra.TAG_FORMAT.format(version="")
        highest_version: Version | None = None
        highest_tag = ""
        for raw_tag in tags:
            tag = raw_tag.strip()
            if not tag:
                continue
            try:
                parsed = Version(tag.removeprefix(prefix))
            except InvalidVersion as exc:
                return r[str].fail(f"invalid release tag {tag}: {exc}")
            if highest_version is None or parsed > highest_version:
                highest_version = parsed
                highest_tag = tag
        return r[str].ok(highest_tag)

    @staticmethod
    def render_project_version(content: str, version: str) -> p.Result[str]:
        """Render one canonical project-version update without writing it."""
        version_result = FlextInfraUtilitiesVersioning.parse_semver(version)
        if version_result.failure:
            return r[str].fail(version_result.error or "invalid version")
        if not FlextInfraUtilitiesVersioning._has_project_table(content):
            return r[str].fail("missing [project] table")
        updated = FlextInfraUtilitiesVersioning._replace_project_version_in_text(
            content, version
        )
        if updated is None:
            return r[str].fail("missing [project] version")
        return r[str].ok(updated)

    @staticmethod
    def replace_project_version(project_path: Path, version: str) -> p.Result[bool]:
        """Update the version field in a project's pyproject.toml.

        Args:
            project_path: Directory containing pyproject.toml.
            version: The new version string.

        Returns:
            r[bool] with True on success.

        """
        pyproject = project_path / c.Infra.PYPROJECT_FILENAME
        try:
            content = pyproject.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        except OSError as exc:
            return r[bool].fail_op("read", exc)
        rendered = FlextInfraUtilitiesVersioning.render_project_version(
            content, version
        )
        if rendered.failure:
            return r[bool].fail(f"{rendered.error} in {pyproject}")
        written = u.Cli.atomic_write_text_file(pyproject, rendered.value)
        if written.failure:
            return written
        # Why: flext-core caches the parsed pyproject per process. This is the
        # protocol's only writer of the version, and the docs projections
        # rendered right after the stamp must see the stamped version, not
        # the document read before it (flext-cli#129 drifted that way).
        u.read_project_document_cached.cache_clear()
        return written


__all__: list[str] = ["FlextInfraUtilitiesVersioning"]
