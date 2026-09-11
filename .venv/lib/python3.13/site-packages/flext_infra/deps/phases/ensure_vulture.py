"""Phase: ensure Vulture production-reachability configuration."""

from __future__ import annotations

from flext_infra import m, t
from flext_infra.deps.toml_phase import FlextInfraTomlPhaseService


class FlextInfraEnsureVultureConfigPhase:
    """Propagate the validated Vulture policy to every project."""

    def __init__(self, tool_config: m.Infra.ToolConfigDocument) -> None:
        """Store the canonical tooling document."""
        self._tool_config = tool_config

    def _phase(self) -> m.Infra.Deps.Toml.PhaseConfig:
        """Build the config-owned Vulture TOML phase."""
        # flext-j47u: reachability policy is data; projects receive no local branch.
        vulture = self._tool_config.tools.vulture
        return (
            m.Infra.Deps.Toml.PhaseConfig
            .Builder("vulture")
            .table("vulture")
            .deprecated("min-confidence")
            .list("exclude", vulture.exclude)
            # flext-j47u (codex): Vulture TOML uses parser keys, not CLI flags.
            .value("min_confidence", vulture.min_confidence)
            .list("paths", vulture.paths)
            .value("verbose", vulture.verbose)
            .build()
        )

    def apply(self, doc: t.Cli.TomlDocument) -> t.StrSequence:
        """Apply Vulture policy to one TOML document."""
        return FlextInfraTomlPhaseService.apply_phases(doc, self._phase())

    def apply_payload(self, payload: t.MutableJsonMapping) -> t.StrSequence:
        """Apply Vulture policy to one normalized payload."""
        return FlextInfraTomlPhaseService.apply_payload_phases(payload, self._phase())


__all__: list[str] = ["FlextInfraEnsureVultureConfigPhase"]
