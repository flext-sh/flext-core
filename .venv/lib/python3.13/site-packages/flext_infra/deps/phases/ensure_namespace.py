"""Phase: Ensure namespace discovery is reflected across project tooling tables."""

from __future__ import annotations

from pathlib import Path

from flext_infra import c, config, m, t, u
from flext_infra.deps.toml_phase import FlextInfraTomlPhaseService


class FlextInfraEnsureNamespaceToolingPhase:
    """Ensure namespace discovery is reflected across project tooling tables."""

    def _phase(self, detected: t.StrSequence) -> m.Infra.Deps.Toml.PhaseConfig:
        """Build the deptry namespace phase for one detected namespace set."""
        return (
            m.Infra.Deps.Toml.PhaseConfig
            .Builder("namespace-tooling")
            .table(c.Infra.DEPTRY)
            .list(c.Infra.KNOWN_FIRST_PARTY_UNDERSCORE, detected)
            .build()
        )

    def apply(self, doc: t.Cli.TomlDocument, *, path: Path) -> t.StrSequence:
        """Apply detected first-party namespaces to dependency tooling tables."""
        detected = sorted({
            *config.Infra.tooling.tools.deptry.known_first_party,
            *u.Infra.discover_first_party_namespaces(path.parent),
            # flext-j47u (codex): declared FLEXT dependencies are first-party.
            *u.Infra.flext_dependency_namespaces(doc),
        })
        if not detected:
            return ()
        return FlextInfraTomlPhaseService.apply_phases(doc, self._phase(detected))

    def apply_payload(
        self, payload: t.MutableJsonMapping, *, path: Path
    ) -> t.StrSequence:
        """Apply detected first-party namespaces to one normalized payload."""
        detected = sorted({
            *config.Infra.tooling.tools.deptry.known_first_party,
            *u.Infra.discover_first_party_namespaces(path.parent),
            *u.Infra.flext_dependency_namespaces_from_payload(payload),
            u.Infra.project_name_from_payload(path, payload).replace("-", "_"),
        })
        if not detected:
            return ()
        return FlextInfraTomlPhaseService.apply_payload_phases(
            payload, self._phase(detected)
        )


__all__: list[str] = ["FlextInfraEnsureNamespaceToolingPhase"]
