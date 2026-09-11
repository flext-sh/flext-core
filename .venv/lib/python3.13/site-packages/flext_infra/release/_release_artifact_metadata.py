"""Release-only dependency metadata and artifact validation."""

from __future__ import annotations

import hashlib
import tarfile
import zipfile
from email.parser import Parser
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from flext_core import r
from flext_infra import c, t, u
from flext_infra.release._release_artifact_archive import (
    FlextInfraReleaseArtifactArchiveMixin,
)

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraReleaseArtifactMetadataMixin(FlextInfraReleaseArtifactArchiveMixin):
    """Render and validate registry-safe release metadata."""

    @staticmethod
    def _release_specifier(version: str) -> p.Result[str]:
        """Return the compatible-release range earned by one declared version.

        Every repository versions independently, so a published artifact must
        accept the sibling releases that semantic versioning promises are
        compatible: the same minor line before 1.0, the same major line after.
        """
        try:
            parsed = Version(version)
        except InvalidVersion as exc:
            return r[str].fail_op("parse internal dependency version", exc)
        if parsed.major == 0:
            return r[str].ok(f"~={version}")
        return r[str].ok(f">={version},<{parsed.major + 1}")

    @classmethod
    def _release_requirement(
        cls, requirement: str, versions: t.StrMapping
    ) -> p.Result[str]:
        """Return one registry-safe requirement for the release.

        An internal dependency is pinned to the compatible range of the version
        its own repository declares; a dependency this build cannot see is not
        publishable, because a guessed range would be a silent contract.
        """
        try:
            parsed = Requirement(requirement)
        except InvalidRequirement as exc:
            return r[str].fail_op("parse release requirement", exc)
        name = canonicalize_name(parsed.name)
        if not name.startswith("flext-"):
            if parsed.url is not None:
                return r[str].fail(
                    f"external direct reference is not publishable: {requirement}"
                )
            return r[str].ok(requirement.strip())
        if name not in versions:
            return r[str].fail(
                f"internal dependency version unknown to this release: {name}"
            )
        specifier = cls._release_specifier(versions[name])
        if specifier.failure:
            return r[str].fail(specifier.error or f"invalid version for {name}")
        extras = f"[{','.join(sorted(parsed.extras))}]" if parsed.extras else ""
        marker = f"; {parsed.marker}" if parsed.marker is not None else ""
        return r[str].ok(f"{parsed.name}{extras}{specifier.value}{marker}")

    @classmethod
    def _rewrite_requirement_field(
        cls,
        container: t.Cli.TomlDocument | t.Cli.TomlTable,
        key: str,
        *,
        versions: t.StrMapping,
    ) -> p.Result[bool]:
        """Rewrite one TOML requirement array in place."""
        raw_value = u.Cli.toml_value(container, key)
        if raw_value is None:
            return r[bool].ok(True)
        raw_items = u.Cli.json_as_sequence(raw_value)
        try:
            requirements = t.Infra.STR_SEQ_ADAPTER.validate_python(
                raw_items, strict=True
            )
        except c.ValidationError as exc:
            return r[bool].fail_op(f"validate release dependency group {key}", exc)
        rewritten: t.MutableSequenceOf[str] = []
        for requirement in requirements:
            result = cls._release_requirement(requirement, versions)
            if result.failure:
                return r[bool].fail(
                    result.error or f"release requirement rewrite failed: {requirement}"
                )
            rewritten.append(result.value)
        u.Cli.toml_sync_string_list(container, key, rewritten)
        return r[bool].ok(True)

    @classmethod
    def _release_pyproject(
        cls, source: str, version: str, versions: t.StrMapping
    ) -> p.Result[str]:
        """Render a pyproject suitable for a public package registry."""
        document = u.Cli.toml_parse_text(source)
        if document is None:
            return r[str].fail("release pyproject is not valid TOML")
        project = u.Cli.toml_table_child(document, c.Infra.PROJECT)
        if project is None:
            return r[str].fail("release pyproject must define [project]")
        project[c.Infra.VERSION] = version
        result = cls._rewrite_requirement_field(
            project, c.Infra.DEPENDENCIES, versions=versions
        )
        if result.failure:
            return r[str].fail(result.error or "runtime dependency rewrite failed")
        for section_name in (c.Infra.OPTIONAL_DEPENDENCIES, c.Infra.DEPENDENCY_GROUPS):
            parent = (
                project if section_name == c.Infra.OPTIONAL_DEPENDENCIES else document
            )
            section = u.Cli.toml_table_child(parent, section_name)
            if section is None:
                continue
            for group_name in tuple(section):
                group_result = cls._rewrite_requirement_field(
                    section, str(group_name), versions=versions
                )
                if group_result.failure:
                    return r[str].fail(
                        group_result.error
                        or f"dependency group rewrite failed: {group_name}"
                    )
        tool = u.Cli.toml_table_child(document, c.Infra.TOOL)
        hatch = u.Cli.toml_table_child(tool, "hatch") if tool is not None else None
        if tool is not None:
            u.Cli.toml_remove_key_if_present(tool, "uv")
        boundary_result = cls._configure_sdist_boundary(hatch)
        if boundary_result.failure:
            return r[str].fail(boundary_result.error or "release sdist boundary failed")
        if tool is not None:
            metadata = (
                u.Cli.toml_table_child(hatch, "metadata") if hatch is not None else None
            )
            if metadata is not None:
                u.Cli.toml_remove_key_if_present(metadata, "allow-direct-references")
                if not metadata and hatch is not None:
                    u.Cli.toml_remove_key_if_present(hatch, "metadata")
        rendered = u.Cli.toml_dumps(document)
        if u.Cli.toml_parse_text(rendered) is None:
            return r[str].fail("release pyproject rendering produced invalid TOML")
        return r[str].ok(rendered)

    @classmethod
    def _configure_sdist_boundary(cls, hatch: t.Cli.TomlTable | None) -> p.Result[bool]:
        """Derive the staged sdist source boundary from the wheel declaration."""
        if hatch is None:
            return r[bool].fail("release pyproject must define [tool.hatch]")
        build = u.Cli.toml_table_child(hatch, "build")
        targets = (
            u.Cli.toml_table_child(build, "targets") if build is not None else None
        )
        wheel = (
            u.Cli.toml_table_child(targets, "wheel") if targets is not None else None
        )
        if targets is None or wheel is None:
            return r[bool].fail("release pyproject must define a Hatch wheel target")
        raw_packages = u.Cli.toml_value(wheel, "packages")
        try:
            packages = t.Infra.STR_SEQ_ADAPTER.validate_python(
                u.Cli.json_as_sequence(raw_packages), strict=True
            )
        except c.ValidationError as exc:
            return r[bool].fail_op("validate Hatch wheel packages", exc)
        if not packages:
            return r[bool].fail("Hatch wheel target must declare packages")
        force_include = u.Cli.toml_table_child(wheel, "force-include")
        force_include_paths = (
            tuple(str(path) for path in force_include)
            if force_include is not None
            else ()
        )
        source_paths = tuple(dict.fromkeys((*packages, *force_include_paths)))
        for source_path in source_paths:
            path = PurePosixPath(source_path)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not cls._sdist_member_allowed(("release-root", *path.parts))
            ):
                return r[bool].fail(
                    f"Hatch source path is outside the release boundary: {source_path}"
                )
        sdist = u.Cli.toml_ensure_table(targets, "sdist")
        for key in ("exclude", "include", "packages"):
            u.Cli.toml_remove_key_if_present(sdist, key)
        u.Cli.toml_sync_string_list(sdist, "only-include", source_paths)
        return r[bool].ok(True)

    @staticmethod
    def _wheel_metadata(path: Path) -> p.Result[str]:
        """Read core metadata from a wheel."""
        try:
            with zipfile.ZipFile(path) as archive:
                names = tuple(
                    name for name in archive.namelist() if name.endswith("/METADATA")
                )
                if len(names) != 1:
                    return r[str].fail(f"wheel must contain one METADATA file: {path}")
                return r[str].ok(archive.read(names[0]).decode("utf-8"))
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError) as exc:
            return r[str].fail_op(f"read wheel metadata {path}", exc)

    @staticmethod
    def _sdist_metadata(path: Path) -> p.Result[str]:
        """Read core metadata from a source distribution."""
        try:
            with tarfile.open(path, "r:gz") as archive:
                members = tuple(
                    member
                    for member in archive.getmembers()
                    if member.name.endswith("/PKG-INFO")
                )
                if len(members) != 1:
                    return r[str].fail(f"sdist must contain one PKG-INFO file: {path}")
                extracted = archive.extractfile(members[0])
                if extracted is None:
                    return r[str].fail(f"cannot read PKG-INFO from {path}")
                return r[str].ok(extracted.read().decode("utf-8"))
        except (OSError, tarfile.TarError, UnicodeDecodeError) as exc:
            return r[str].fail_op(f"read sdist metadata {path}", exc)

    @classmethod
    def _artifact_metadata(cls, path: Path) -> p.Result[str]:
        """Read core metadata from a wheel or source distribution."""
        return (
            cls._wheel_metadata(path)
            if path.suffix == ".whl"
            else cls._sdist_metadata(path)
        )

    @staticmethod
    def _validate_artifact_identity(
        metadata: str, project: str, version: str
    ) -> p.Result[bool]:
        """Validate core metadata identity against the selected release target."""
        message = Parser().parsestr(metadata)
        artifact_name = message.get("Name")
        artifact_version = message.get("Version")
        if artifact_name is None or canonicalize_name(
            artifact_name
        ) != canonicalize_name(project):
            return r[bool].fail(
                f"artifact Name mismatch: expected {project}, found {artifact_name}"
            )
        try:
            expected_version = Version(version)
            actual_version = Version(artifact_version or "")
        except InvalidVersion as exc:
            return r[bool].fail_op("validate artifact Version", exc)
        if actual_version != expected_version:
            return r[bool].fail(
                f"artifact Version mismatch: expected {expected_version}, "
                f"found {actual_version}"
            )
        return r[bool].ok(True)

    @classmethod
    def _validate_artifact_requirements(
        cls, metadata: str, versions: t.StrMapping
    ) -> p.Result[bool]:
        """Reject direct references and internal dependencies outside their range."""
        message = Parser().parsestr(metadata)
        for requirement_text in message.get_all("Requires-Dist", []):
            try:
                requirement = Requirement(requirement_text)
            except InvalidRequirement as exc:
                return r[bool].fail_op("parse artifact requirement", exc)
            if requirement.url is not None:
                return r[bool].fail(
                    f"artifact contains direct dependency reference: {requirement_text}"
                )
            name = canonicalize_name(requirement.name)
            if not name.startswith("flext-"):
                continue
            expected = (
                cls._release_specifier(versions[name])
                if name in versions
                else r[str].fail(f"internal dependency version unknown: {name}")
            )
            if expected.failure or str(requirement.specifier) != expected.value:
                return r[bool].fail(
                    f"artifact contains unpinned FLEXT dependency: {requirement_text}"
                )
        return r[bool].ok(True)

    @classmethod
    def _validate_artifact(
        cls,
        path: Path,
        project: str,
        version: str,
        license_sha256: str,
        versions: t.StrMapping,
    ) -> p.Result[t.Pair[t.Infra.ReleaseArtifactKind, t.Infra.ReleaseArtifactSha256]]:
        """Validate one artifact and return its kind and SHA-256 digest."""
        archive_result = cls._validate_archive(path, project, license_sha256)
        if archive_result.failure:
            return r[
                t.Pair[t.Infra.ReleaseArtifactKind, t.Infra.ReleaseArtifactSha256]
            ].fail(
                archive_result.error or f"artifact archive validation failed: {path}"
            )
        metadata_result = cls._artifact_metadata(path)
        if metadata_result.failure:
            return r[
                t.Pair[t.Infra.ReleaseArtifactKind, t.Infra.ReleaseArtifactSha256]
            ].fail(metadata_result.error or f"artifact metadata unavailable: {path}")
        identity_result = cls._validate_artifact_identity(
            metadata_result.value, project, version
        )
        if identity_result.failure:
            return r[
                t.Pair[t.Infra.ReleaseArtifactKind, t.Infra.ReleaseArtifactSha256]
            ].fail(identity_result.error or "artifact identity validation failed")
        requirements_result = cls._validate_artifact_requirements(
            metadata_result.value, versions
        )
        if requirements_result.failure:
            return r[
                t.Pair[t.Infra.ReleaseArtifactKind, t.Infra.ReleaseArtifactSha256]
            ].fail(
                requirements_result.error or "artifact requirement validation failed"
            )
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            return r[
                t.Pair[t.Infra.ReleaseArtifactKind, t.Infra.ReleaseArtifactSha256]
            ].fail_op(f"hash release artifact {path}", exc)
        kind: t.Infra.ReleaseArtifactKind = (
            "wheel" if path.suffix == ".whl" else "sdist"
        )
        return r[t.Pair[t.Infra.ReleaseArtifactKind, t.Infra.ReleaseArtifactSha256]].ok((
            kind,
            digest,
        ))


__all__: list[str] = ["FlextInfraReleaseArtifactMetadataMixin"]
