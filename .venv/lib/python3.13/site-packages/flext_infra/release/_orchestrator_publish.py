"""Release publish phase: upload exactly what the build receipt attests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, config, m, u

if TYPE_CHECKING:
    from flext_infra import p, t

logger = u.fetch_logger(__name__)


class FlextInfraReleaseOrchestratorPublishMixin:
    """Publish-phase implementation (parent of the release-phases class).

    The receipt written by the build phase is the only input: every artifact
    it names is re-hashed before it leaves the machine, the GitHub release
    carries those files and nothing else, and the package index receives them
    in dependency order so a dependent never precedes its dependency.
    """

    @staticmethod
    def _release_output_dir(ctx: m.Infra.ReleasePhaseDispatchConfig) -> Path:
        """Return the receipt directory of the declared version."""
        return (
            u.Cli.resolve_report_dir(
                ctx.repository_root, c.Infra.PROJECT, c.Infra.RK_RELEASE
            )
            / ctx.tag
        )

    def phase_publish(self, ctx: m.Infra.ReleasePhaseDispatchConfig) -> p.Result[bool]:
        """Publish the receipt's artifacts as a GitHub release and, on request, to the index."""
        receipt = self._verified_receipt(ctx)
        if receipt.failure:
            return r[bool].fail(receipt.error or "release receipt invalid")
        report = receipt.value
        if ctx.dry_run:
            logger.info("release_phase_publish", tag=ctx.tag, dry_run=True)
            return r[bool].ok(True)
        released = self._github_release(ctx, report)
        if released.failure:
            return released
        if ctx.index:
            uploaded = self._index_upload(ctx, report)
            if uploaded.failure:
                return uploaded
        logger.info("release_phase_publish", tag=ctx.tag, index=ctx.index)
        return r[bool].ok(True)

    def _verified_receipt(
        self, ctx: m.Infra.ReleasePhaseDispatchConfig
    ) -> p.Result[m.Infra.BuildReport]:
        """Load the receipt and prove every artifact still matches its digest."""
        report_path = self._release_output_dir(ctx) / c.Infra.RELEASE_REPORT_FILENAME
        content = u.Cli.files_read_text(report_path)
        if content.failure:
            return r[m.Infra.BuildReport].fail(
                content.error or f"release receipt missing: {report_path}"
            )
        try:
            report = m.Infra.BuildReport.model_validate_json(content.value)
        except c.ValidationError as exc:
            return r[m.Infra.BuildReport].fail_op("validate release receipt", exc)
        if report.dry_run or report.failures or report.version != ctx.version:
            return r[m.Infra.BuildReport].fail(
                f"release receipt is not publishable for {ctx.version}: {report_path}"
            )
        for record in report.records:
            for artifact in record.artifacts:
                path = Path(artifact.path)
                try:
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                except OSError as exc:
                    return r[m.Infra.BuildReport].fail_op(f"read artifact {path}", exc)
                if digest != artifact.sha256:
                    return r[m.Infra.BuildReport].fail(
                        f"artifact digest differs from receipt: {path}"
                    )
        return r[m.Infra.BuildReport].ok(report)

    @staticmethod
    def _github_release(
        ctx: m.Infra.ReleasePhaseDispatchConfig, report: m.Infra.BuildReport
    ) -> p.Result[bool]:
        """Create or refresh the GitHub release with the receipt's artifacts."""
        root = ctx.repository_root
        assets = [
            artifact.path for record in report.records for artifact in record.artifacts
        ]
        notes = root / c.Infra.DIR_DOCS / "releases" / f"{ctx.tag}.md"
        exists = u.Cli.capture(
            [c.Infra.GH, "release", "view", ctx.tag, "--json", "tagName"], cwd=root
        )
        if exists.success:
            return u.Cli.run_checked(
                [c.Infra.GH, "release", "upload", ctx.tag, *assets, "--clobber"],
                cwd=root,
            )
        return u.Cli.run_checked(
            [
                c.Infra.GH,
                "release",
                "create",
                ctx.tag,
                *assets,
                "--title",
                f"Release {ctx.tag}",
                "--notes-file",
                str(notes),
            ],
            cwd=root,
        )

    @staticmethod
    def _index_upload(
        ctx: m.Infra.ReleasePhaseDispatchConfig, report: m.Infra.BuildReport
    ) -> p.Result[bool]:
        """Upload verified artifacts wave by wave through trusted publishing.

        ``--check-url`` makes the upload idempotent: files already on the index
        are skipped, never overwritten, so a re-run after a partial failure
        resumes instead of failing on the first duplicate.
        """
        artifacts: dict[str, t.StrSequence] = {
            record.project: tuple(artifact.path for artifact in record.artifacts)
            for record in report.records
        }
        waves = u.Infra.release_publish_waves(
            tuple((record.project, Path(record.path)) for record in report.records)
        )
        if waves.failure:
            return r[bool].fail(waves.error or "publish order failed")
        for wave in waves.value:
            paths = [path for project in wave for path in artifacts[project]]
            uploaded = u.Cli.run_checked(
                [
                    c.Infra.UV,
                    "publish",
                    "--no-config",
                    "--publish-url",
                    config.Infra.release.publish_url,
                    "--check-url",
                    c.Infra.PYPI_SIMPLE_INDEX_URL,
                    "--trusted-publishing",
                    "always",
                    *paths,
                ],
                cwd=ctx.repository_root,
            )
            if uploaded.failure:
                return uploaded
            logger.info("release_index_wave_published", projects=", ".join(wave))
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraReleaseOrchestratorPublishMixin"]
