"""Domain models for the release subpackage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Self

from flext_core import m
from flext_core import u
from flext_infra import t
from flext_infra._constants.release import FlextInfraConstantsRelease as cr
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsRelease:
    """Models for release management."""

    class SourceSnapshot(m.StrictBoundaryModel):
        """Immutable committed source identity used by one project build."""

        commit_oid: Annotated[
            t.Infra.ReleaseCommitOid, m.Field(description="Source commit object ID")
        ]
        source_date_epoch: Annotated[
            t.NonNegativeInt, m.Field(description="Source commit Unix epoch")
        ]

    class BuildPolicy(m.StrictBoundaryModel):
        """Immutable policy snapshot used by every build in one report."""

        build_constraints_path: Annotated[
            t.Infra.ReleaseAbsolutePath,
            m.Field(description="Snapshotted build constraints absolute path"),
        ]
        build_constraints_sha256: Annotated[
            t.Infra.ReleaseArtifactSha256,
            m.Field(description="Pinned build constraint policy SHA-256"),
        ]
        gitleaks_policy_path: Annotated[
            t.Infra.ReleaseAbsolutePath,
            m.Field(description="Snapshotted Gitleaks policy absolute path"),
        ]
        gitleaks_policy_sha256: Annotated[
            t.Infra.ReleaseArtifactSha256,
            m.Field(description="Trusted Gitleaks policy SHA-256"),
        ]

    class BuildArtifact(m.StrictBoundaryModel):
        """Immutable artifact emitted by one release build."""

        path: Annotated[
            t.Infra.ReleaseAbsolutePath, m.Field(description="Artifact absolute path")
        ]
        kind: Annotated[
            t.Infra.ReleaseArtifactKind, m.Field(description="wheel or sdist")
        ]
        sha256: Annotated[
            t.Infra.ReleaseArtifactSha256,
            m.Field(description="Artifact SHA-256 digest"),
        ]

        @u.model_validator(mode="after")
        def validate_kind_filename(self) -> Self:
            """Require the declared artifact kind to match its immutable filename."""
            name = Path(self.path).name
            valid = (self.kind == "wheel" and name.endswith(".whl")) or (
                self.kind == "sdist" and name.endswith(".tar.gz")
            )
            if not valid:
                msg = f"artifact kind does not match filename: {self.kind}={name}"
                raise ValueError(msg)
            return self

    class BuildRecord(mm.ProjectNameMixin, m.StrictBoundaryModel):
        """Base model for build result data."""

        path: Annotated[
            t.Infra.ReleaseAbsolutePath, m.Field(description="Project absolute path")
        ]
        exit_code: Annotated[
            t.StrictInt,
            m.Field(description="Artifact build command exit code, including signals"),
        ]
        log: Annotated[
            t.Infra.ReleaseAbsolutePath, m.Field(description="Build log absolute path")
        ]
        artifacts: Annotated[
            t.VariadicTuple[FlextInfraModelsRelease.BuildArtifact],
            m.Field(default=(), description="Validated wheel and sdist artifacts"),
        ]
        commit_oid: Annotated[
            t.Infra.ReleaseCommitOid | None,
            m.Field(default=None, description="Source commit object ID"),
        ] = None
        source_date_epoch: Annotated[
            t.NonNegativeInt | None,
            m.Field(default=None, description="Source commit Unix epoch"),
        ] = None
        source_license_sha256: Annotated[
            t.Infra.ReleaseArtifactSha256 | None,
            m.Field(default=None, description="Committed source LICENSE SHA-256"),
        ] = None

        @u.model_validator(mode="after")
        def validate_provenance(self) -> Self:
            """Require source provenance to be complete whenever it is available."""
            values = (
                self.commit_oid,
                self.source_date_epoch,
                self.source_license_sha256,
            )
            if any(value is None for value in values) and any(
                value is not None for value in values
            ):
                msg = "build record source provenance must be complete"
                raise ValueError(msg)
            return self

    class ReleasePlan(m.StrictBoundaryModel):
        """The protocol's decision for one repository, derived and never typed by hand."""

        current: Annotated[
            t.NonEmptyStr, m.Field(description="Version declared by pyproject.toml")
        ]
        next: Annotated[t.NonEmptyStr, m.Field(description="Version to release")]
        bump: Annotated[
            cr.VersionBump, m.Field(description="Bump derived from merged PRs")
        ]
        previous_tag: Annotated[
            t.NonEmptyStr | None,
            m.Field(default=None, description="Latest release tag, if any"),
        ] = None
        merges: Annotated[
            tuple[str, ...],
            m.Field(default=(), description="Merged PR subjects since previous_tag"),
        ] = ()
        declared: Annotated[
            bool,
            m.Field(
                default=False,
                description=(
                    "``next`` is the declared version itself (first release, "
                    "finalized pre-release, or a version written beyond the last "
                    "tag): decided when it was written, still untagged"
                ),
            ),
        ] = False

        @m.computed_field
        @property
        def tag(self) -> str:
            """Tag that will identify ``next``."""
            return cr.TAG_FORMAT.format(version=self.next)

        @m.computed_field
        @property
        def releasable(self) -> bool:
            """Whether a release commit is due: a declared release, or a real bump."""
            return self.declared or self.next != self.current

    class BuildReport(m.StrictBoundaryModel):
        """Aggregated build report payload written to JSON."""

        version: Annotated[t.NonEmptyStr, m.Field(description="Release version")]
        total: Annotated[
            t.NonNegativeInt, m.Field(description="Total projects attempted")
        ]
        failures: Annotated[
            t.NonNegativeInt, m.Field(description="Total projects with non-zero exit")
        ]
        records: Annotated[
            t.VariadicTuple[FlextInfraModelsRelease.BuildRecord],
            m.Field(default=(), description="Per-project build records"),
        ]
        dry_run: Annotated[bool, m.Field(description="Metadata-only build report")]
        build_constraints_sha256: Annotated[
            t.Infra.ReleaseArtifactSha256,
            m.Field(description="Pinned build constraint policy SHA-256"),
        ]
        gitleaks_policy_sha256: Annotated[
            t.Infra.ReleaseArtifactSha256,
            m.Field(description="Trusted Gitleaks policy SHA-256"),
        ]

        @u.model_validator(mode="after")
        def validate_manifest(self) -> Self:
            """Require totals, project identity, outcomes, and artifacts to agree."""
            if self.total != len(self.records):
                msg = "build report total does not match record count"
                raise ValueError(msg)
            failures = sum(record.exit_code != 0 for record in self.records)
            if self.failures != failures:
                msg = "build report failures do not match record outcomes"
                raise ValueError(msg)
            projects = tuple(record.project for record in self.records)
            if len(projects) != len(set(projects)):
                msg = "build report contains duplicate projects"
                raise ValueError(msg)
            if not self.dry_run and not self.records:
                msg = "non-dry build report contains no projects"
                raise ValueError(msg)
            for record in self.records:
                kinds = {artifact.kind for artifact in record.artifacts}
                paths = tuple(artifact.path for artifact in record.artifacts)
                if record.exit_code != 0 and record.artifacts:
                    msg = f"failed build contains artifacts: {record.project}"
                    raise ValueError(msg)
                if record.exit_code == 0 and self.dry_run and record.artifacts:
                    msg = f"dry-run build contains artifacts: {record.project}"
                    raise ValueError(msg)
                if (
                    record.exit_code == 0
                    and not self.dry_run
                    and (
                        len(record.artifacts) != len(kinds)
                        or len(paths) != len(set(paths))
                        or kinds != {"wheel", "sdist"}
                    )
                ):
                    msg = f"successful build lacks wheel and sdist: {record.project}"
                    raise ValueError(msg)
                if record.exit_code == 0 and record.commit_oid is None:
                    msg = f"successful build lacks source provenance: {record.project}"
                    raise ValueError(msg)
            return self

    class ReleasePhaseDispatchConfig(
        mm.ProjectNamesListMixin,
        mm.RepositoryRootPathMixin,
        mm.VersionTagMixin,
        m.ArbitraryTypesModel,
    ):
        """Resolved input of one release phase: the repository, its declared version and its tag."""

        dry_run: Annotated[bool, m.Field(description="Dry run flag")] = False
        phase: Annotated[cr.ReleasePhase, m.Field(description="Release phase")]
        index: Annotated[
            bool, m.Field(description="Publish verified artifacts to the package index")
        ] = False
        pr_title: Annotated[
            str, m.Field(description="Pull-request title to validate, when known")
        ] = ""


__all__: list[str] = ["FlextInfraModelsRelease"]
