"""Validated receipt and publication candidates for Mise transactions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import m, u
from flext_infra.codegen import _mise_artifacts_files as files

if TYPE_CHECKING:
    from flext_infra import p


def normalize_lock_mode(path: Path) -> p.Result[bool]:
    """Normalize an external lock output through guarded byte-mode publication."""
    state = files.read_state(path, required=True)
    if state.failure:
        return r[bool].from_failure(state)
    if state.value.content is None or state.value.mode is None:
        return r[bool].fail(f"generated Mise lock is absent: {path}")
    return u.Cli.atomic_write_binary_file_guarded(
        path,
        state.value.content,
        expected_bytes=state.value.content,
        expected_mode=state.value.mode,
        permission_mode=files.ARTIFACT_SPECS[2][1],
    )


def receipt_states(receipt: Path) -> p.Result[tuple[m.Cli.AtomicFileState, ...]]:
    """Capture the two exact launcher states from one validated receipt."""
    states: list[m.Cli.AtomicFileState] = []
    for name, expected_mode in files.ARTIFACT_SPECS[:2]:
        state = files.read_state(receipt / name, required=True)
        if state.failure:
            return r[tuple[m.Cli.AtomicFileState, ...]].from_failure(state)
        if state.value.mode != expected_mode:
            return r[tuple[m.Cli.AtomicFileState, ...]].fail(
                f"Mise receipt mode differs after validation: {name}"
            )
        states.append(state.value)
    return r[tuple[m.Cli.AtomicFileState, ...]].ok(tuple(states))


def publication_plan(
    projects: tuple[m.Infra.MiseToolchainProjectState, ...], stages: tuple[Path, ...]
) -> p.Result[tuple[m.Infra.MiseToolchainPublication, ...]]:
    """Bind each staged artifact to its exact pre-lock destination state."""
    publications: list[m.Infra.MiseToolchainPublication] = []
    for project, stage in zip(projects, stages, strict=True):
        before_states = (
            project.config.before,
            project.artifacts.unix_launcher,
            project.artifacts.windows_launcher,
            project.artifacts.lock,
        )
        for before, (name, mode) in zip(
            before_states, files.PUBLICATION_SPECS, strict=True
        ):
            replacement = files.read_state(stage / name, required=True)
            if replacement.failure or replacement.value.content is None:
                return r[tuple[m.Infra.MiseToolchainPublication, ...]].fail(
                    replacement.error or f"missing staged Mise artifact: {name}"
                )
            if replacement.value.mode != mode:
                return r[tuple[m.Infra.MiseToolchainPublication, ...]].fail(
                    f"staged Mise artifact mode differs: {stage / name}"
                )
            publications.append(
                m.Infra.MiseToolchainPublication(
                    before=before, replacement=replacement.value
                )
            )
    return r[tuple[m.Infra.MiseToolchainPublication, ...]].ok(tuple(publications))


__all__: list[str] = ["normalize_lock_mode", "publication_plan", "receipt_states"]
