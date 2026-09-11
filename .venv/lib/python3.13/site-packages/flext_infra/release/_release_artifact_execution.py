"""Isolated command execution and strict record construction for releases."""

from __future__ import annotations

from pathlib import Path

from flext_core import r
from flext_infra import c, m, p, t, u
from flext_infra.release._release_artifact_persistence import (
    FlextInfraReleaseArtifactPersistenceMixin,
)


class FlextInfraReleaseArtifactExecutionMixin(
    FlextInfraReleaseArtifactPersistenceMixin
):
    """Execute the pinned build command and model its observable result."""

    @staticmethod
    def _build_record(
        *,
        project: str,
        project_path: Path,
        log_path: Path,
        exit_code: int,
        artifacts: t.SequenceOf[m.Infra.BuildArtifact] = (),
        snapshot: m.Infra.SourceSnapshot | None = None,
        source_license_sha256: t.Infra.ReleaseArtifactSha256 | None = None,
    ) -> m.Infra.BuildRecord:
        """Create one strict build record from absolute release paths."""
        return m.Infra.BuildRecord(
            project=project,
            path=str(project_path.resolve()),
            exit_code=exit_code,
            log=str(log_path.resolve()),
            artifacts=tuple(artifacts),
            commit_oid=snapshot.commit_oid if snapshot is not None else None,
            source_date_epoch=(
                snapshot.source_date_epoch if snapshot is not None else None
            ),
            source_license_sha256=source_license_sha256,
        )

    @staticmethod
    def _release_build_command(
        stage_path: Path, temporary_dist: Path, build_constraints_path: Path
    ) -> t.StrSequence:
        """Return the fail-closed isolated uv build command."""
        return (
            c.Infra.UV,
            "build",
            "--force-pep517",
            "--no-config",
            "--no-sources",
            "--no-python-downloads",
            "--no-create-gitignore",
            "--no-progress",
            "--color",
            "never",
            "--default-index",
            c.Infra.PYPI_SIMPLE_INDEX_URL,
            "--build-constraints",
            str(build_constraints_path),
            "--require-hashes",
            "--out-dir",
            str(temporary_dist),
            str(stage_path),
        )

    def _execute_release_build(
        self,
        *,
        stage_path: Path,
        temporary_dist: Path,
        build_constraints_path: Path,
        source_date_epoch: int,
        log_path: Path,
    ) -> p.Result[p.Cli.CommandOutput]:
        """Validate the toolchain lock, execute uv, and persist its full log."""
        constraints_result = u.Cli.files_read_text(build_constraints_path)
        if constraints_result.failure:
            return r[p.Cli.CommandOutput].fail(
                constraints_result.error
                or f"read release build constraints failed: {build_constraints_path}"
            )
        validation_result = self._validate_build_constraints(constraints_result.value)
        if validation_result.failure:
            return r[p.Cli.CommandOutput].fail(
                validation_result.error
                or f"release build constraints invalid: {build_constraints_path}"
            )
        build_result = u.Cli.run_raw(
            self._release_build_command(
                stage_path, temporary_dist, build_constraints_path
            ),
            timeout=c.Infra.TIMEOUT_LONG,
            env={
                c.Infra.SOURCE_DATE_EPOCH: str(source_date_epoch),
                c.Infra.UV_HTTP_CONNECT_TIMEOUT: (
                    c.Infra.UV_RELEASE_HTTP_CONNECT_TIMEOUT
                ),
                c.Infra.UV_HTTP_TIMEOUT: c.Infra.UV_RELEASE_HTTP_TIMEOUT,
                c.Infra.UV_HTTP_RETRIES: c.Infra.UV_RELEASE_HTTP_RETRIES,
            },
            remove_env_keys=c.Infra.UV_RELEASE_POLICY_ENV_KEYS,
        )
        if build_result.failure:
            return r[p.Cli.CommandOutput].fail(
                build_result.error or "uv release build execution failed"
            )
        command = build_result.value
        output = (command.stdout + "\n" + command.stderr).strip()
        write_result = self._write_release_text(log_path, output + "\n")
        if write_result.failure:
            return r[p.Cli.CommandOutput].fail(
                write_result.error or "write release build log failed"
            )
        return r[p.Cli.CommandOutput].ok(command)


__all__: list[str] = ["FlextInfraReleaseArtifactExecutionMixin"]
