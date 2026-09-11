"""Archive structure and source-distribution content policy."""

from __future__ import annotations

import hashlib
import stat
import tarfile
import zipfile
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import config

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraReleaseArtifactArchiveMixin:
    """Validate release archive boundaries before artifact persistence."""

    @staticmethod
    def _staged_member_path_error(name: str) -> str:
        """Return a staged-source sensitivity error, or an empty string."""
        path = PurePosixPath(name)
        if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
            return f"unsafe staged source path: {name}"
        # Why: a file codegen owns (`.env.example`) is a projection of the
        # fleet template, never a secret; only its name matches the pattern.
        if any(
            name == item.path.as_posix() for item in config.Infra.codegen.managed_files
        ):
            return ""
        blocked_suffixes = (".jks", ".key", ".keystore", ".p12", ".pem", ".pfx")
        for part in path.parts:
            normalized = part.casefold()
            if (
                normalized in {".env", ".secrets.baseline", "__pycache__"}
                or normalized.startswith((".env.", ".gitleaks"))
                or normalized.endswith(blocked_suffixes)
            ):
                return f"sensitive staged source path: {name}"
        return ""

    @classmethod
    def _validate_staged_source(cls, stage_path: Path) -> p.Result[bool]:
        """Reject sensitive paths and links in the committed source snapshot."""
        try:
            members = tuple(sorted(stage_path.rglob("*")))
        except OSError as exc:
            return r[bool].fail_op(f"list staged release source {stage_path}", exc)
        for member in members:
            relative = member.relative_to(stage_path).as_posix()
            error = cls._staged_member_path_error(relative)
            if error:
                return r[bool].fail(error)
            if member.is_symlink():
                return r[bool].fail(f"staged source contains symbolic link: {relative}")
        return r[bool].ok(True)

    @staticmethod
    def _member_path_error(name: str, *, root_index: int) -> str:
        """Return a policy error for one archive member, or an empty string.

        Repository-operational directories (``.git``, ``.github``, ``.reports``)
        are rejected only at the archive's content root (``root_index``): a
        package may legitimately ship such a tree as data — flext-infra packages
        the fleet's ``.github`` templates. Sensitive material is rejected at
        every depth.
        """
        path = PurePosixPath(name)
        if not name or "\\" in name or path.is_absolute() or ".." in path.parts:
            return f"unsafe archive member path: {name}"
        operational_roots = frozenset({".git", ".github", ".reports"})
        sensitive_parts = frozenset({".env", ".secrets.baseline", "__pycache__"})
        blocked_suffixes = (".jks", ".key", ".keystore", ".p12", ".pem", ".pfx")
        if (
            len(path.parts) > root_index
            and path.parts[root_index].casefold() in operational_roots
        ):
            return f"operational or sensitive archive member: {name}"
        for part in path.parts:
            normalized = part.casefold()
            if (
                normalized in sensitive_parts
                or normalized.startswith((".env.", ".gitleaks"))
                or normalized.endswith(blocked_suffixes)
            ):
                return f"operational or sensitive archive member: {name}"
        return ""

    @staticmethod
    def _license_names() -> frozenset[str]:
        """Return accepted canonical license basenames."""
        return frozenset({
            "copying",
            "copying.md",
            "copying.txt",
            "license",
            "license.md",
            "license.txt",
        })

    @classmethod
    def _source_license_digest(cls, stage_path: Path) -> p.Result[str]:
        """Return the SHA-256 of the single committed source license."""
        try:
            licenses = tuple(
                path
                for path in stage_path.iterdir()
                if path.is_file() and path.name.casefold() in cls._license_names()
            )
        except OSError as exc:
            return r[str].fail_op(f"list staged release source {stage_path}", exc)
        if len(licenses) != 1:
            return r[str].fail(
                f"release source must contain exactly one LICENSE: {stage_path}"
            )
        try:
            digest = hashlib.sha256(licenses[0].read_bytes()).hexdigest()
        except OSError as exc:
            return r[str].fail_op(f"hash source license {licenses[0]}", exc)
        return r[str].ok(digest)

    @classmethod
    def _sdist_member_allowed(cls, parts: t.StrSequence) -> bool:
        """Return whether one regular sdist member is in the public boundary."""
        relative = tuple(parts[1:])
        if not relative:
            return False
        if relative[0].casefold() in {"config", "src"}:
            return True
        if len(relative) != 1:
            return False
        basename = relative[0].casefold()
        return (
            basename in cls._license_names()
            or basename.startswith("readme")
            or basename in {".gitignore", "pkg-info", "pyproject.toml"}
        )

    @classmethod
    def _validate_wheel_archive(
        cls, path: Path, project: str, license_sha256: str
    ) -> p.Result[bool]:
        """Validate wheel roots, member paths, symlinks, and license presence."""
        try:
            with zipfile.ZipFile(path) as archive:
                return cls._validate_open_wheel(archive, path, project, license_sha256)
        except (OSError, zipfile.BadZipFile) as exc:
            return r[bool].fail_op(f"validate wheel archive {path}", exc)

    @classmethod
    def _validate_open_wheel(
        cls, archive: zipfile.ZipFile, path: Path, project: str, license_sha256: str
    ) -> p.Result[bool]:
        """Validate one already-open wheel archive."""
        package_root = project.replace("-", "_").casefold()
        members = tuple(info for info in archive.infolist() if not info.is_dir())
        licenses = tuple(
            info
            for info in members
            if PurePosixPath(info.filename).name.casefold() in cls._license_names()
        )
        for info in members:
            # A wheel's content root is the package (or its .dist-info/.data).
            error = cls._member_path_error(info.filename, root_index=0)
            if error:
                return r[bool].fail(error)
            if stat.S_ISLNK(info.external_attr >> 16):
                return r[bool].fail(f"wheel contains symbolic link: {info.filename}")
            root = PurePosixPath(info.filename).parts[0].casefold()
            metadata_root = root.startswith(f"{package_root}-") and root.endswith((
                ".data",
                ".dist-info",
            ))
            if root != package_root and not metadata_root:
                return r[bool].fail(
                    f"wheel contains unexpected top-level path: {info.filename}"
                )
        if len(licenses) != 1:
            return r[bool].fail(f"wheel must contain exactly one LICENSE: {path}")
        wheel_license_sha256 = hashlib.sha256(archive.read(licenses[0])).hexdigest()
        if wheel_license_sha256 != license_sha256:
            return r[bool].fail(f"wheel LICENSE differs from committed source: {path}")
        return r[bool].ok(True)

    @classmethod
    def _validate_sdist_archive(
        cls, path: Path, project: str, license_sha256: str
    ) -> p.Result[bool]:
        """Validate sdist root, member paths, links, and license presence."""
        try:
            with tarfile.open(path, "r:gz") as archive:
                return cls._validate_open_sdist(archive, path, project, license_sha256)
        except (OSError, tarfile.TarError) as exc:
            return r[bool].fail_op(f"validate sdist archive {path}", exc)

    @classmethod
    def _validate_open_sdist(
        cls, archive: tarfile.TarFile, path: Path, project: str, license_sha256: str
    ) -> p.Result[bool]:
        """Validate one already-open source-distribution archive."""
        expected_roots = (project.casefold(), project.replace("-", "_").casefold())
        members = tuple(archive.getmembers())
        licenses = tuple(
            member
            for member in members
            if member.isfile()
            and PurePosixPath(member.name).name.casefold() in cls._license_names()
        )
        roots: set[str] = set()
        for member in members:
            # An sdist's content root sits under its single `<name>-<version>/`.
            error = cls._member_path_error(member.name, root_index=1)
            if error:
                return r[bool].fail(error)
            parts = PurePosixPath(member.name).parts
            if parts:
                roots.add(parts[0].casefold())
            if member.issym() or member.islnk():
                return r[bool].fail(
                    f"sdist contains symbolic or hard link: {member.name}"
                )
            if member.isfile() and not cls._sdist_member_allowed(parts):
                return r[bool].fail(
                    f"sdist contains unexpected public content: {member.name}"
                )
        if len(roots) != 1:
            return r[bool].fail(
                f"sdist must contain exactly one top-level directory: {path}"
            )
        root = next(iter(roots))
        if not any(root.startswith(f"{expected}-") for expected in expected_roots):
            return r[bool].fail(f"sdist root does not match project {project}: {root}")
        if len(licenses) != 1:
            return r[bool].fail(f"sdist must contain exactly one LICENSE: {path}")
        extracted = archive.extractfile(licenses[0])
        if extracted is None:
            return r[bool].fail(f"cannot read sdist LICENSE: {path}")
        sdist_license_sha256 = hashlib.sha256(extracted.read()).hexdigest()
        if sdist_license_sha256 != license_sha256:
            return r[bool].fail(f"sdist LICENSE differs from committed source: {path}")
        return r[bool].ok(True)

    @classmethod
    def _validate_archive(
        cls, path: Path, project: str, license_sha256: str
    ) -> p.Result[bool]:
        """Validate the archive format and its publishable content boundary."""
        if path.suffix == ".whl":
            return cls._validate_wheel_archive(path, project, license_sha256)
        if path.name.endswith(".tar.gz"):
            return cls._validate_sdist_archive(path, project, license_sha256)
        return r[bool].fail(f"unsupported release artifact type: {path}")


__all__: list[str] = ["FlextInfraReleaseArtifactArchiveMixin"]
