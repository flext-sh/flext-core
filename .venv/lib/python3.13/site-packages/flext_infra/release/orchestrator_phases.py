"""Release build phase: registry-safe artifacts and their receipt.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, config, m, t, u
from flext_infra.release._orchestrator_publish import (
    FlextInfraReleaseOrchestratorPublishMixin,
)
from flext_infra.release._release_artifact_build import (
    FlextInfraReleaseArtifactBuildMixin,
)

if TYPE_CHECKING:
    from flext_infra import p

logger = u.fetch_logger(__name__)


class FlextInfraReleaseOrchestratorPhases(
    FlextInfraReleaseOrchestratorPublishMixin, FlextInfraReleaseArtifactBuildMixin
):
    """Build phase implementation (publish via the mixin)."""

    @staticmethod
    def _build_targets(
        repository_root: Path, project_names: t.StrSequence
    ) -> p.Result[t.SequenceOf[t.Pair[str, Path]]]:
        """Resolve release build targets from the configured eligibility policy.

        Why (aihub-ioijy.9): this used to hardcode
        ``project.name.startswith("flext-")``. Eligibility is declared data
        (``config.Infra.release.publishable_prefixes``); an empty tuple means no
        prefix filter, so a single-project repository publishes itself.
        """
        projects_result = u.Infra.resolve_projects(repository_root, project_names)
        if projects_result.failure:
            return r[t.SequenceOf[t.Pair[str, Path]]].fail(
                projects_result.error or "release project resolution failed"
            )
        prefixes = tuple(config.Infra.release.publishable_prefixes)
        seen: t.Infra.StrSet = set()
        unique: t.MutableSequenceOf[t.Pair[str, Path]] = []
        for project in projects_result.value:
            if prefixes and not project.name.startswith(prefixes):
                continue
            if project.name in seen or not project.path.exists():
                continue
            seen.add(project.name)
            unique.append((project.name, project.path))
        return r[t.SequenceOf[t.Pair[str, Path]]].ok(unique)

    @staticmethod
    def _internal_versions(repository_root: Path) -> p.Result[t.StrMapping]:
        """Map every visible internal distribution to its own declared version.

        Each repository versions independently, so release metadata pins a
        sibling to what the sibling's ``pyproject.toml`` declares. The root
        project is included so a workspace may depend on its own distribution.
        A sibling consumed through a pinned git ref (a standalone repository's
        internal dependency) is what the committed ``uv.lock`` resolved for it:
        the version that sibling declared at the pinned commit.
        """
        projects = u.Infra.resolve_projects(repository_root, ())
        if projects.failure:
            return r[t.StrMapping].fail(
                projects.error or "internal version resolution failed"
            )
        versions: dict[str, str] = dict(
            u.Infra.locked_dependency_versions(
                repository_root / c.Infra.UV_LOCK_FILENAME, sources=("git",)
            )
        )
        for project in projects.value:
            declared = u.Infra.current_workspace_version(project.path)
            if declared.failure:
                return r[t.StrMapping].fail(
                    declared.error or f"version unresolved: {project.name}"
                )
            versions[project.name] = declared.value
        return r[t.StrMapping].ok(versions)

    def _build_project_record(
        self,
        ctx: m.Infra.ReleasePhaseDispatchConfig,
        policy: m.Infra.BuildPolicy,
        name: str,
        path: Path,
        output_dir: Path,
        versions: t.StrMapping,
    ) -> p.Result[m.Infra.BuildRecord]:
        """Build one project and convert fail-loud errors into report records."""
        record_result = self._build_release_record(
            project=name,
            project_path=path,
            output_dir=output_dir,
            build_constraints_path=Path(policy.build_constraints_path),
            gitleaks_config_path=Path(policy.gitleaks_policy_path),
            version=(
                ctx.version
                if path.resolve() == ctx.repository_root.resolve()
                else versions[name]
            ),
            versions=versions,
            dry_run=ctx.dry_run,
        )
        if record_result.success:
            return record_result
        log = output_dir / f"build-{name}.log"
        write_result = self._write_release_text(
            log, (record_result.error or "release build failed") + "\n"
        )
        if write_result.failure:
            return r[m.Infra.BuildRecord].fail(
                write_result.error or f"write failed build log: {name}"
            )
        return r[m.Infra.BuildRecord].ok(
            self._build_record(
                project=name, project_path=path, log_path=log, exit_code=1
            )
        )

    def _build_records(
        self,
        ctx: m.Infra.ReleasePhaseDispatchConfig,
        policy: m.Infra.BuildPolicy,
        targets: t.SequenceOf[t.Pair[str, Path]],
        output_dir: Path,
    ) -> p.Result[t.SequenceOf[m.Infra.BuildRecord]]:
        """Build every selected project and retain its strict report record."""
        versions = self._internal_versions(ctx.repository_root)
        if versions.failure:
            return r[t.SequenceOf[m.Infra.BuildRecord]].fail(
                versions.error or "internal version resolution failed"
            )
        records: t.MutableSequenceOf[m.Infra.BuildRecord] = []
        for name, path in targets:
            record_result = self._build_project_record(
                ctx, policy, name, path, output_dir, versions.value
            )
            if record_result.failure:
                return r[t.SequenceOf[m.Infra.BuildRecord]].fail(
                    record_result.error or f"release build record failed: {name}"
                )
            record = record_result.value
            records.append(record)
            logger.info(
                "release_phase_build_project", project=name, exit_code=record.exit_code
            )
        return r[t.SequenceOf[m.Infra.BuildRecord]].ok(tuple(records))

    @staticmethod
    def _snapshot_policy_file(
        source: Path, destination: Path, *, policy_root: Path
    ) -> p.Result[str]:
        """Persist immutable policy bytes and return their SHA-256 digest."""
        destination = destination.resolve()
        if not destination.is_relative_to(policy_root.resolve()):
            return r[str].fail(
                f"release policy destination escapes policy root: {destination}"
            )
        try:
            content = source.read_bytes()
        except OSError as exc:
            return r[str].fail_op(f"read release policy {source}", exc)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return r[str].fail_op(
                f"create release policy directory {destination.parent}", exc
            )
        try:
            if destination.exists():
                if destination.read_bytes() != content:
                    return r[str].fail(
                        f"immutable release policy collision: {destination}"
                    )
            else:
                destination.write_bytes(content)
            return r[str].ok(hashlib.sha256(content).hexdigest())
        except OSError as exc:
            return r[str].fail_op(f"persist release policy {destination}", exc)

    @classmethod
    def _snapshot_build_policy(
        cls, repository_root: Path, output_dir: Path
    ) -> p.Result[m.Infra.BuildPolicy]:
        """Capture one immutable policy pair before the first project build."""
        policy_dir = output_dir / "policy"
        constraints_path = policy_dir / "build-constraints.txt"
        constraints_result = cls._snapshot_policy_file(
            repository_root / c.Infra.RELEASE_BUILD_CONSTRAINTS_PATH,
            constraints_path,
            policy_root=policy_dir,
        )
        if constraints_result.failure:
            return r[m.Infra.BuildPolicy].fail(
                constraints_result.error or "snapshot build constraints failed"
            )
        gitleaks_path = policy_dir / "gitleaks-release.toml"
        gitleaks_result = cls._snapshot_policy_file(
            repository_root / c.Infra.RELEASE_GITLEAKS_CONFIG_PATH,
            gitleaks_path,
            policy_root=policy_dir,
        )
        if gitleaks_result.failure:
            return r[m.Infra.BuildPolicy].fail(
                gitleaks_result.error or "snapshot Gitleaks policy failed"
            )
        return r[m.Infra.BuildPolicy].ok(
            m.Infra.BuildPolicy(
                build_constraints_path=str(constraints_path.resolve()),
                build_constraints_sha256=constraints_result.value,
                gitleaks_policy_path=str(gitleaks_path.resolve()),
                gitleaks_policy_sha256=gitleaks_result.value,
            )
        )

    @classmethod
    def _write_build_report(
        cls,
        ctx: m.Infra.ReleasePhaseDispatchConfig,
        policy: m.Infra.BuildPolicy,
        output_dir: Path,
        records: t.SequenceOf[m.Infra.BuildRecord],
    ) -> p.Result[int]:
        """Persist the strict build report and return its failure count."""
        failures = sum(record.exit_code != 0 for record in records)
        report = m.Infra.BuildReport(
            version=ctx.version,
            total=len(records),
            failures=failures,
            records=tuple(records),
            dry_run=ctx.dry_run,
            build_constraints_sha256=policy.build_constraints_sha256,
            gitleaks_policy_sha256=policy.gitleaks_policy_sha256,
        )
        report_path = output_dir / c.Infra.RELEASE_REPORT_FILENAME
        write_result = u.Cli.json_write(
            report_path,
            report.model_dump(mode="json"),
            m.Cli.JsonWriteOptions(sort_keys=True),
        )
        if write_result.failure:
            return r[int].fail(write_result.error or "write build report failed")
        logger.info("release_phase_build_report", report=str(report_path))
        return r[int].ok(failures)

    def phase_build(self, ctx: m.Infra.ReleasePhaseDispatchConfig) -> p.Result[bool]:
        """Build registry-safe member artifacts and write the receipt."""
        output_dir = self._release_output_dir(ctx)
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return r[bool].fail_op("report dir creation", exc)
        targets_result = self._build_targets(ctx.repository_root, ctx.project_names)
        if targets_result.failure:
            return r[bool].fail(
                targets_result.error or "release build target resolution failed"
            )
        if not targets_result.value:
            return r[bool].fail("release build selected no publishable projects")
        policy_result = self._snapshot_build_policy(ctx.repository_root, output_dir)
        if policy_result.failure:
            return r[bool].fail(
                policy_result.error or "release build policy snapshot failed"
            )
        records_result = self._build_records(
            ctx, policy_result.value, targets_result.value, output_dir
        )
        if records_result.failure:
            return r[bool].fail(records_result.error or "release build records failed")
        report_result = self._write_build_report(
            ctx, policy_result.value, output_dir, records_result.value
        )
        if report_result.failure:
            return r[bool].fail(report_result.error or "write build report failed")
        if report_result.value:
            return r[bool].fail(f"build failed: {report_result.value} project(s)")
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraReleaseOrchestratorPhases"]
