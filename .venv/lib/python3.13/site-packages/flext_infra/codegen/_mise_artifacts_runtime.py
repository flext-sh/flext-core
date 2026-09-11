"""Isolated newest-Mise runtime and credential-source owner."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import m, u
from flext_infra.codegen import _mise_artifacts_files as files
from flext_infra.codegen import _mise_artifacts_process as process

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraMiseRuntime:
    """Generate and authenticate the newest launcher in isolated state."""

    def __init__(self, owner: p.Infra.MiseArtifactsOwner) -> None:
        self._owner = owner

    def latest_receipt(
        self, root: m.Infra.MiseToolchainProjectState, *, credential_command: str
    ) -> p.Result[Path]:
        """Resolve latest with an isolated seed, then return an exact receipt."""
        scratch = root.layout.transaction_root / "runtime"
        prepared = process.prepare_isolation(scratch)
        if prepared.failure:
            return r[Path].from_failure(prepared)
        seed_state = (
            root.artifacts.windows_launcher
            if os.name == "nt"
            else root.artifacts.unix_launcher
        )
        if seed_state.content is None or seed_state.mode is None:
            return r[Path].fail(f"native Mise seed is absent: {seed_state.path}")
        seed = scratch / "seed" / "bin" / seed_state.path.name
        created = process.write_new(seed, seed_state.content, seed_state.mode)
        if created.failure:
            return r[Path].from_failure(created)
        seed_validation = self._owner.validate_seed(seed)
        if seed_validation.failure:
            return r[Path].fail(
                seed_validation.error or "captured Mise seed is invalid"
            )
        environment = process.credential_environment(scratch, credential_command)
        updated = process.run(
            (str(seed), "self-update", "--yes", "--no-plugins"),
            cwd=scratch,
            env=environment,
            operation="Mise latest release resolution",
        )
        if updated.failure:
            return r[Path].from_failure(updated)
        resolved_runtime = process.run(
            (str(seed), "--version"),
            cwd=scratch,
            env=environment,
            operation="Mise resolved release identity",
        )
        if resolved_runtime.failure:
            return r[Path].from_failure(resolved_runtime)
        resolved_release = (
            resolved_runtime.value.split(maxsplit=1)[0]
            if resolved_runtime.value
            else ""
        )
        if not self._owner.is_mise_release(resolved_release):
            return r[Path].fail("Mise latest resolution returned an invalid release")
        receipt = scratch / "receipt"
        generated = process.run(
            (
                str(seed),
                "-C",
                str(scratch),
                "generate",
                "install-script",
                "--write",
                str(receipt / "bin" / "mise"),
                "--windows",
                "--version",
                resolved_release,
            ),
            cwd=scratch,
            env=environment,
            operation="Mise exact latest launcher generation",
        )
        if generated.failure:
            return r[Path].from_failure(generated)
        exact_modes = self._apply_receipt_modes(receipt)
        if exact_modes.failure:
            return r[Path].from_failure(exact_modes)
        validation = self._owner.validate_launchers(receipt)
        if validation.failure:
            return r[Path].fail(validation.error or "generated Mise receipt is invalid")
        launcher = receipt / "bin" / ("mise.cmd" if os.name == "nt" else "mise")
        runtime = process.run(
            (str(launcher), "--version"),
            cwd=scratch,
            env=environment,
            operation="Mise newest runtime identity",
        )
        if runtime.failure:
            return r[Path].from_failure(runtime)
        release = runtime.value.split(maxsplit=1)[0] if runtime.value else ""
        embedded = self._owner.launcher_release(receipt)
        if (
            not self._owner.is_mise_release(release)
            or embedded.failure
            or release != embedded.value
            or release != resolved_release
        ):
            return r[Path].fail("Mise newest runtime differs from its receipt")
        u.Cli.info(f"mise-toolchain: resolved latest runtime={release}")
        return r[Path].ok(receipt)

    @staticmethod
    def _apply_receipt_modes(receipt: Path) -> p.Result[bool]:
        for name, mode in files.ARTIFACT_SPECS[:2]:
            state = files.read_state(receipt / name, required=True)
            if state.failure:
                return r[bool].from_failure(state)
            if state.value.content is None or state.value.mode is None:
                return r[bool].fail(f"generated Mise receipt is absent: {name}")
            normalized = u.Cli.atomic_write_binary_file_guarded(
                state.value.path,
                state.value.content,
                expected_bytes=state.value.content,
                expected_mode=state.value.mode,
                permission_mode=mode,
            )
            if normalized.failure:
                return r[bool].fail(
                    normalized.error or f"cannot normalize Mise receipt mode: {name}"
                )
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraMiseRuntime"]
