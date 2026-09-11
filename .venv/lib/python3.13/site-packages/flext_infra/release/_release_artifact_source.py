"""Committed-source preparation and isolated build environment policy."""

from __future__ import annotations

import hashlib
import tarfile
from typing import TYPE_CHECKING

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from flext_core import r
from flext_infra import c, m, t, u
from flext_infra.release._release_artifact_metadata import (
    FlextInfraReleaseArtifactMetadataMixin,
)

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraReleaseArtifactSourceMixin(FlextInfraReleaseArtifactMetadataMixin):
    """Prepare immutable source snapshots under trusted release policies."""

    @staticmethod
    def _constraint_records(content: str) -> p.Result[t.StrSequence]:
        """Parse logical requirement records from a hashed constraint file."""
        records: t.MutableSequenceOf[str] = []
        current: t.MutableSequenceOf[str] = []
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            continued = line.endswith("\\")
            current.append(line.removesuffix("\\").strip())
            if not continued:
                records.append(" ".join(current))
                current = []
        if current:
            return r[t.StrSequence].fail(
                "release build constraints end with a continuation"
            )
        if not records:
            return r[t.StrSequence].fail("release build constraints are empty")
        return r[t.StrSequence].ok(tuple(records))

    @staticmethod
    def _constraint_name(record: str) -> p.Result[str]:
        """Validate one exact hashed registry pin and return its canonical name."""
        tokens = record.split()
        try:
            requirement = Requirement(tokens[0])
        except (IndexError, InvalidRequirement) as exc:
            return r[str].fail_op("parse release build constraint", exc)
        specifiers = tuple(requirement.specifier)
        if (
            requirement.url is not None
            or requirement.extras
            or requirement.marker is not None
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
        ):
            return r[str].fail(
                f"release build constraint is not one exact registry pin: {record}"
            )
        hashes = tuple(
            token.removeprefix("--hash=sha256:")
            for token in tokens[1:]
            if token.startswith("--hash=sha256:")
        )
        if len(hashes) != len(tokens) - 1 or not hashes:
            return r[str].fail(
                f"release build constraint lacks only SHA-256 hashes: {record}"
            )
        if any(
            len(digest) != hashlib.sha256().digest_size * 2
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in hashes
        ):
            return r[str].fail(
                f"release build constraint has invalid SHA-256: {record}"
            )
        return r[str].ok(canonicalize_name(requirement.name))

    @classmethod
    def _validate_build_constraints(cls, content: str) -> p.Result[bool]:
        """Validate the complete exact and hashed build toolchain lock."""
        records_result = cls._constraint_records(content)
        if records_result.failure:
            return r[bool].fail(
                records_result.error or "release build constraint parsing failed"
            )
        names: t.Infra.StrSet = set()
        for record in records_result.value:
            name_result = cls._constraint_name(record)
            if name_result.failure:
                return r[bool].fail(
                    name_result.error or "release build constraint validation failed"
                )
            name = name_result.value
            if name in names:
                return r[bool].fail(f"duplicate release build constraint: {name}")
            names.add(name)
        expected = c.Infra.RELEASE_BUILD_TOOLCHAIN_REQUIREMENTS
        actual = frozenset(names)
        if actual != expected:
            missing = ", ".join(sorted(expected - actual)) or "none"
            unexpected = ", ".join(sorted(actual - expected)) or "none"
            return r[bool].fail(
                f"release build toolchain mismatch: missing={missing}; "
                f"unexpected={unexpected}"
            )
        return r[bool].ok(True)

    @staticmethod
    def _scan_staged_source(
        stage_path: Path, gitleaks_config_path: Path
    ) -> p.Result[bool]:
        """Scan committed staged source with the canonical secret scanner."""
        config_result = u.Cli.files_read_text(gitleaks_config_path)
        if config_result.failure or not config_result.value.strip():
            return r[bool].fail(
                config_result.error
                or f"release Gitleaks policy is empty: {gitleaks_config_path}"
            )
        scan_result = u.Cli.run_raw(
            [
                c.Infra.GITLEAKS,
                "dir",
                "--config",
                str(gitleaks_config_path),
                "--gitleaks-ignore-path",
                str(gitleaks_config_path.parent),
                "--no-banner",
                "--no-color",
                "--redact=100",
                "--exit-code",
                str(c.Infra.GITLEAKS_LEAK_EXIT_CODE),
                "--ignore-gitleaks-allow",
                str(stage_path),
            ],
            cwd=gitleaks_config_path.parent,
            timeout=c.Infra.TIMEOUT_LONG,
            remove_env_keys=c.Infra.GITLEAKS_POLICY_ENV_KEYS,
        )
        if scan_result.failure:
            return r[bool].fail(scan_result.error or "gitleaks execution failed")
        command = scan_result.value
        if command.exit_code == c.Infra.GITLEAKS_LEAK_EXIT_CODE:
            return r[bool].fail("gitleaks detected a secret in staged release source")
        if command.exit_code != 0:
            return r[bool].fail(f"gitleaks failed with exit code {command.exit_code}")
        return r[bool].ok(True)

    @staticmethod
    def _archive_project(
        project_path: Path, stage_path: Path
    ) -> p.Result[m.Infra.SourceSnapshot]:
        """Extract one immutable commit and return its modeled source identity."""
        status_result = u.Cli.capture(
            [c.Infra.GIT, "status", "--porcelain"],
            cwd=project_path,
            timeout=c.Infra.TIMEOUT_MEDIUM,
        )
        if status_result.failure:
            return r[m.Infra.SourceSnapshot].fail(
                status_result.error or "project status failed"
            )
        if status_result.value.strip():
            return r[m.Infra.SourceSnapshot].fail(
                f"release project is dirty: {project_path}"
            )
        oid_result = u.Cli.capture(
            [c.Infra.GIT, "rev-parse", "--verify", f"{c.Infra.GIT_HEAD}^{{commit}}"],
            cwd=project_path,
            timeout=c.Infra.TIMEOUT_MEDIUM,
        )
        if oid_result.failure:
            return r[m.Infra.SourceSnapshot].fail(
                oid_result.error or "resolve release commit failed"
            )
        oid = oid_result.value.strip()
        epoch_result = u.Cli.capture(
            [c.Infra.GIT, "show", "-s", "--format=%ct", oid],
            cwd=project_path,
            timeout=c.Infra.TIMEOUT_MEDIUM,
        )
        if epoch_result.failure:
            return r[m.Infra.SourceSnapshot].fail(
                epoch_result.error or "resolve commit epoch failed"
            )
        source_date_epoch = epoch_result.value.strip()
        if not source_date_epoch.isdigit():
            return r[m.Infra.SourceSnapshot].fail(
                f"commit epoch is not an integer for {project_path}: {source_date_epoch}"
            )
        archive_path = stage_path.parent / f"{stage_path.name}.tar"
        archive_result = u.Cli.run_checked(
            [c.Infra.GIT, "archive", "--format=tar", f"--output={archive_path}", oid],
            cwd=project_path,
            timeout=c.Infra.TIMEOUT_MEDIUM,
        )
        if archive_result.failure:
            return r[m.Infra.SourceSnapshot].fail(
                archive_result.error or "git archive failed"
            )
        try:
            stage_path.mkdir(parents=True, exist_ok=False)
            with tarfile.open(archive_path, "r") as archive:
                archive.extractall(stage_path, filter="data")
        except (OSError, tarfile.TarError) as exc:
            return r[m.Infra.SourceSnapshot].fail_op(
                "extract committed release source", exc
            )
        try:
            snapshot = m.Infra.SourceSnapshot(
                commit_oid=oid, source_date_epoch=int(source_date_epoch)
            )
        except c.ValidationError as exc:
            return r[m.Infra.SourceSnapshot].fail_op(
                "validate committed release source identity", exc
            )
        return r[m.Infra.SourceSnapshot].ok(snapshot)

    @staticmethod
    def _write_release_text(path: Path, content: str) -> p.Result[bool]:
        """Write release text through the Result-based filesystem boundary."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return r[bool].fail_op(
                f"create release output directory {path.parent}", exc
            )
        return u.Cli.files_write_text(path, content)


__all__: list[str] = ["FlextInfraReleaseArtifactSourceMixin"]
