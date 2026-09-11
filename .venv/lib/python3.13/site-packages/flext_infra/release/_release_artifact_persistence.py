"""Atomic persistence for complete validated release artifact sets."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import t
from flext_infra.release._release_artifact_source import (
    FlextInfraReleaseArtifactSourceMixin,
)

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraReleaseArtifactPersistenceMixin(FlextInfraReleaseArtifactSourceMixin):
    """Validate and atomically persist complete project artifact sets."""

    @staticmethod
    def _build_artifact_paths(output_dir: Path) -> p.Result[t.SequenceOf[Path]]:
        """Resolve exactly one wheel and one sdist from uv build output."""
        try:
            entries = tuple(sorted(output_dir.iterdir()))
        except OSError as exc:
            return r[t.SequenceOf[Path]].fail_op(
                f"list uv build output {output_dir}", exc
            )
        wheels = tuple(path for path in entries if path.suffix == ".whl")
        sdists = tuple(path for path in entries if path.name.endswith(".tar.gz"))
        artifacts = (*wheels, *sdists)
        unexpected = tuple(path for path in entries if path not in artifacts)
        if unexpected:
            names = ", ".join(path.name for path in unexpected)
            return r[t.SequenceOf[Path]].fail(
                f"uv build emitted unexpected output: {names}"
            )
        if len(wheels) != 1 or len(sdists) != 1:
            return r[t.SequenceOf[Path]].fail(
                f"expected one wheel and one sdist, found "
                f"{len(wheels)} wheel(s) and {len(sdists)} sdist(s)"
            )
        return r[t.SequenceOf[Path]].ok(artifacts)

    @staticmethod
    def _validate_existing_artifact_set(
        artifacts: t.SequenceOf[
            t.Triple[Path, t.Infra.ReleaseArtifactKind, t.Infra.ReleaseArtifactSha256]
        ],
        destinations: t.SequenceOf[Path],
        destination_dir: Path,
    ) -> p.Result[t.SequenceOf[Path]]:
        """Validate an already persisted immutable project artifact set."""
        try:
            existing = tuple(sorted(destination_dir.iterdir()))
        except OSError as exc:
            return r[t.SequenceOf[Path]].fail_op("list persisted artifact set", exc)
        if existing != tuple(sorted(destinations)):
            return r[t.SequenceOf[Path]].fail(
                f"immutable artifact set collision at {destination_dir}"
            )
        for (source, _, _), destination in zip(artifacts, destinations, strict=True):
            try:
                matches = source.read_bytes() == destination.read_bytes()
            except OSError as exc:
                return r[t.SequenceOf[Path]].fail_op(
                    f"compare immutable artifact {destination}", exc
                )
            if not matches:
                return r[t.SequenceOf[Path]].fail(
                    f"immutable artifact collision at {destination}"
                )
        return r[t.SequenceOf[Path]].ok(destinations)

    @staticmethod
    def _commit_artifact_set(
        artifacts: t.SequenceOf[
            t.Triple[Path, t.Infra.ReleaseArtifactKind, t.Infra.ReleaseArtifactSha256]
        ],
        destinations: t.SequenceOf[Path],
        destination_dir: Path,
    ) -> p.Result[t.SequenceOf[Path]]:
        """Copy and atomically rename one complete project artifact set."""
        try:
            with TemporaryDirectory(
                prefix=f".{destination_dir.name}-", dir=destination_dir.parent
            ) as temporary:
                staging_dir = Path(temporary)
                for source, _, _ in artifacts:
                    shutil.copy2(source, staging_dir / source.name)
                staging_dir.replace(destination_dir)
        except OSError as exc:
            return r[t.SequenceOf[Path]].fail_op("persist release artifact set", exc)
        return r[t.SequenceOf[Path]].ok(destinations)

    @classmethod
    def _persist_artifact_set(
        cls,
        artifacts: t.SequenceOf[
            t.Triple[Path, t.Infra.ReleaseArtifactKind, t.Infra.ReleaseArtifactSha256]
        ],
        destination_dir: Path,
    ) -> p.Result[t.SequenceOf[Path]]:
        """Persist a project's complete validated artifact set atomically."""
        destinations = tuple(
            destination_dir / source.name for source, _, _ in artifacts
        )
        try:
            destination_dir.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return r[t.SequenceOf[Path]].fail_op("create artifact output parent", exc)
        if destination_dir.exists():
            return cls._validate_existing_artifact_set(
                artifacts, destinations, destination_dir
            )
        return cls._commit_artifact_set(artifacts, destinations, destination_dir)


__all__: list[str] = ["FlextInfraReleaseArtifactPersistenceMixin"]
