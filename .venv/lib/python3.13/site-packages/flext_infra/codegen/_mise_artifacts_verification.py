"""Topology, source, and live-state verification for Mise transactions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import m
from flext_infra._utilities.project_managed_artifacts import (
    FlextInfraUtilitiesProjectManagedArtifacts,
)
from flext_infra.codegen import _mise_artifacts_files as files

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


def journal_topology(
    layout: m.Infra.MiseToolchainWorkspaceLayout, journal: m.Infra.MiseToolchainJournal
) -> p.Result[bool]:
    """Bind every untrusted journal selector to the stable workspace layout."""
    if journal.projects != tuple(project.selector for project in layout.projects):
        return r[bool].fail("Mise journal project topology differs from layout")
    source_topology = _journal_source_topology(layout, journal.sources)
    if source_topology.failure:
        return source_topology
    if journal.state == "staging":
        return r[bool].ok(True)
    if len(journal.entries) != len(layout.projects) * len(files.PUBLICATION_SPECS):
        return r[bool].fail("Mise journal entry count differs from project topology")
    expected_entries: list[tuple[str, int]] = []
    entry_projects: list[m.Infra.MiseToolchainProjectLayout] = []
    for project in layout.projects:
        for artifact, (_name, mode) in zip(
            (
                project.artifacts.config,
                project.artifacts.unix_launcher,
                project.artifacts.windows_launcher,
                project.artifacts.lock,
            ),
            files.PUBLICATION_SPECS,
            strict=True,
        ):
            selector = files.workspace_relative(layout.scope_root, artifact)
            if selector.failure:
                return r[bool].from_failure(selector)
            expected_entries.append((selector.value, mode))
            entry_projects.append(project)
    observed_entries = tuple(
        (entry.path, entry.replacement_mode) for entry in journal.entries
    )
    if observed_entries != tuple(expected_entries):
        return r[bool].fail("Mise journal artifact topology differs from workspace")
    for index, (entry, project) in enumerate(
        zip(journal.entries, entry_projects, strict=True)
    ):
        expected_backup: str | None = None
        if entry.original_exists:
            relative = files.workspace_relative(
                layout.scope_root,
                project.transaction_root / "recovery" / f"{index:04d}.original",
            )
            if relative.failure:
                return r[bool].from_failure(relative)
            expected_backup = relative.value
        if entry.original_backup != expected_backup:
            return r[bool].fail("Mise journal backup topology differs from workspace")
    return r[bool].ok(True)


def sources(
    plan: m.Infra.MiseToolchainWorkspacePlan,
    journal: m.Infra.MiseToolchainJournal | None = None,
) -> p.Result[bool]:
    """Prove source topology, bytes, and modes still equal one snapshot."""
    expected_states: list[m.Cli.AtomicFileState] = []
    for project in plan.projects:
        config_sources = (
            FlextInfraUtilitiesProjectManagedArtifacts.snapshot_config_sources(
                project.layout.root
            )
        )
        if config_sources.failure:
            return r[bool].from_failure(config_sources)
        expected = project.config.sources
        current = config_sources.value
        if current != expected:
            return r[bool].fail(f"Mise sources changed: {project.layout.selector}")
        expected_states.extend(expected)
    if journal is None:
        return r[bool].ok(True)
    topology = _journal_source_topology(plan.layout, journal.sources)
    if topology.failure:
        return topology
    if len(journal.sources) != len(expected_states):
        return r[bool].fail("Mise journal source count differs from snapshot")
    for expected, recorded in zip(expected_states, journal.sources, strict=True):
        selector = files.workspace_relative(plan.layout.scope_root, expected.path)
        if selector.failure:
            return r[bool].from_failure(selector)
        if (
            expected.content is None
            or expected.mode is None
            or recorded.path != selector.value
            or files.digest(expected.content) != recorded.sha256
            or expected.mode != recorded.mode
        ):
            return r[bool].fail(f"Mise source differs from journal: {expected.path}")
    return r[bool].ok(True)


def destinations(plan: m.Infra.MiseToolchainWorkspacePlan) -> p.Result[bool]:
    """Prove every live destination still equals the locked preflight snapshot."""
    for project in plan.projects:
        expected_states = (
            project.config.before,
            project.artifacts.unix_launcher,
            project.artifacts.windows_launcher,
            project.artifacts.lock,
        )
        for expected in expected_states:
            observed = files.read_state(expected.path, required=False)
            if observed.failure:
                return r[bool].from_failure(observed)
            if observed.value != expected:
                return r[bool].fail(
                    f"Mise destination changed after preflight: {expected.path}"
                )
    return r[bool].ok(True)


def live(
    owner: p.Infra.MiseArtifactsOwner,
    plan: m.Infra.MiseToolchainWorkspacePlan,
    publications: tuple[m.Infra.MiseToolchainPublication, ...] | None = None,
) -> p.Result[bool]:
    """Exercise each real artifact consumer and compare exact bytes plus modes."""
    source_before = sources(plan)
    if source_before.failure:
        return source_before
    replacements = {
        publication.before.path: (
            publication.replacement.content,
            publication.replacement.mode,
        )
        for publication in publications or ()
    }
    artifact_before = _artifact_snapshot(plan, replacements)
    if artifact_before.failure:
        return r[bool].from_failure(artifact_before)
    for project in plan.projects:
        validated = owner.validate_artifacts(
            project.layout.root, config_sources=project.config.sources
        )
        if validated.failure:
            return r[bool].fail(
                validated.error
                or f"published Mise validation failed for {project.layout.selector}"
            )
    artifact_after = _artifact_snapshot(plan, replacements)
    if artifact_after.failure:
        return r[bool].from_failure(artifact_after)
    if artifact_after.value != artifact_before.value:
        return r[bool].fail("published Mise artifacts changed during validation")
    source_after = sources(plan)
    if source_after.failure:
        return source_after
    artifact_final = _artifact_snapshot(plan, replacements)
    if artifact_final.failure:
        return r[bool].from_failure(artifact_final)
    if artifact_final.value != artifact_after.value:
        return r[bool].fail("published Mise artifacts changed after validation")
    return r[bool].ok(True)


def _artifact_snapshot(
    plan: m.Infra.MiseToolchainWorkspacePlan,
    replacements: dict[Path, tuple[bytes | None, int | None]],
) -> p.Result[tuple[m.Cli.AtomicFileState, ...]]:
    """Capture and validate one complete ordered artifact-state barrier."""
    root_launchers: tuple[bytes, bytes] | None = None
    states: list[m.Cli.AtomicFileState] = []
    for project in plan.projects:
        artifacts = (
            project.config.before,
            project.artifacts.unix_launcher,
            project.artifacts.windows_launcher,
            project.artifacts.lock,
        )
        observed: list[bytes] = []
        for expected, (_name, required_mode) in zip(
            artifacts, files.PUBLICATION_SPECS, strict=True
        ):
            current = files.read_state(expected.path, required=True)
            if current.failure or current.value.content is None:
                return r[tuple[m.Cli.AtomicFileState, ...]].fail(
                    current.error
                    or f"published Mise artifact is absent: {expected.path}"
                )
            expected_state = replacements.get(
                expected.path, (expected.content, expected.mode)
            )
            if (current.value.content, current.value.mode) != expected_state:
                return r[tuple[m.Cli.AtomicFileState, ...]].fail(
                    f"published Mise artifact differs from plan: {expected.path}"
                )
            if current.value.mode != required_mode:
                return r[tuple[m.Cli.AtomicFileState, ...]].fail(
                    f"published Mise artifact mode is noncanonical: {expected.path}"
                )
            observed.append(current.value.content)
            states.append(current.value)
        launchers = (observed[1], observed[2])
        if root_launchers is None:
            root_launchers = launchers
        elif launchers != root_launchers:
            return r[tuple[m.Cli.AtomicFileState, ...]].fail(
                f"published Mise launchers differ in {project.layout.selector}"
            )
    return r[tuple[m.Cli.AtomicFileState, ...]].ok(tuple(states))


def _journal_source_topology(
    layout: m.Infra.MiseToolchainWorkspaceLayout,
    sources_to_validate: tuple[m.Infra.MiseToolchainJournalSource, ...],
) -> p.Result[bool]:
    grouped: dict[Path, list[str]] = {project.root: [] for project in layout.projects}
    seen: set[str] = set()
    for source in sources_to_validate:
        if source.path in seen:
            return r[bool].fail(f"duplicate Mise journal source: {source.path}")
        seen.add(source.path)
        resolved = files.resolve_relative(
            layout.scope_root, source.path, purpose="Mise journal source"
        )
        if resolved.failure:
            return r[bool].from_failure(resolved)
        owners = tuple(
            project
            for project in layout.projects
            if resolved.value.parent == project.root / "config"
            and resolved.value.suffix == ".yaml"
        )
        if len(owners) != 1:
            return r[bool].fail(
                f"Mise journal source is outside topology: {source.path}"
            )
        grouped[owners[0].root].append(source.path)
    expected_order: list[str] = []
    for project in layout.projects:
        project_sources = grouped[project.root]
        expected_group = sorted(project_sources)
        if project_sources != expected_group:
            return r[bool].fail(
                f"Mise journal source topology differs for {project.selector}"
            )
        expected_order.extend(expected_group)
    if tuple(source.path for source in sources_to_validate) != tuple(expected_order):
        return r[bool].fail("Mise journal source order differs from workspace")
    return r[bool].ok(True)


__all__: list[str] = ["destinations", "journal_topology", "live", "sources"]
