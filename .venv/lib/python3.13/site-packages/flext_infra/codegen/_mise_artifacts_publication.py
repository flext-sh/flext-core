"""Guarded live publication for a fully journaled Mise artifact set."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import m, u
from flext_infra.codegen import _mise_artifacts_files as files
from flext_infra.codegen import _mise_artifacts_verification as verify

if TYPE_CHECKING:
    from flext_infra import p


def publish(
    owner: p.Infra.MiseArtifactsOwner,
    plan: m.Infra.MiseToolchainWorkspacePlan,
    journal: m.Infra.MiseToolchainJournal,
    publications: tuple[m.Infra.MiseToolchainPublication, ...],
) -> p.Result[bool]:
    """Publish changed bytes or modes, then exercise the real consumers."""
    source_check = verify.sources(plan, journal)
    if source_check.failure:
        return source_check
    changed = 0
    for publication in publications:
        before = publication.before
        replacement = publication.replacement
        if before.content == replacement.content and before.mode == replacement.mode:
            continue
        written = files.write_publication(publication)
        if written.failure:
            return r[bool].fail(
                written.error or f"publish failed for {publication.before.path}"
            )
        changed += 1
    validated = verify.live(owner, plan, publications)
    if validated.failure:
        return r[bool].fail(
            validated.error or "published Mise consumer validation failed"
        )
    journal_sources = verify.sources(plan, journal)
    if journal_sources.failure:
        return journal_sources
    u.Cli.info(
        "mise-toolchain: published "
        f"{changed} changed artifact(s) across {len(plan.projects)} project(s)"
    )
    return r[bool].ok(True)


__all__: list[str] = ["publish"]
