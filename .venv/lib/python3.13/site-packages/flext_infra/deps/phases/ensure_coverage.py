"""Phase: Ensure coverage configuration in pyproject.toml."""

from __future__ import annotations

from flext_infra import m, t
from flext_infra.deps.toml_phase import FlextInfraTomlPhaseService


class FlextInfraEnsureCoverageConfigPhase:
    """Ensure coverage report configuration with per-project-type thresholds."""

    def __init__(self, tool_config: m.Infra.ToolConfigDocument) -> None:
        """Capture tool configuration used to build canonical coverage settings."""
        self._tool_config = tool_config

    def _phases(
        self, *, project_kind: str = "core"
    ) -> tuple[m.Infra.Deps.Toml.PhaseConfig, m.Infra.Deps.Toml.PhaseConfig]:
        """Build the canonical coverage phases for the selected project kind."""
        cov_config = self._tool_config.tools.coverage
        fail_under_map: t.IntMapping = {
            "core": cov_config.fail_under.core,
            "domain": cov_config.fail_under.domain,
            "platform": cov_config.fail_under.platform,
            "integration": cov_config.fail_under.integration,
            "app": cov_config.fail_under.app,
        }
        fail_under = fail_under_map.get(project_kind, cov_config.fail_under.core)
        report_phase = (
            m.Infra.Deps.Toml.PhaseConfig
            .Builder("coverage-report")
            .table("coverage", "report")
            .value("fail_under", fail_under)
            # flext-j47u (codex): policy values come only from config.Infra.tooling.
            .value("show_missing", cov_config.show_missing)
            .value("skip_covered", cov_config.skip_covered)
            .value("precision", cov_config.precision)
            .list("exclude_also", sorted(set(cov_config.exclude_also)))
            .build()
        )
        run_phase = (
            m.Infra.Deps.Toml.PhaseConfig
            .Builder("coverage-run")
            .table("coverage", "run")
            .list("source", list(cov_config.source))
            .list("omit", sorted(set(cov_config.omit)))
            .build()
        )
        return (report_phase, run_phase)

    def apply(
        self, doc: t.Cli.TomlDocument, *, project_kind: str = "core"
    ) -> t.StrSequence:
        """Apply canonical coverage report/run tables for the selected project kind."""
        return FlextInfraTomlPhaseService.apply_phases(
            doc, *self._phases(project_kind=project_kind)
        )

    def apply_payload(
        self, payload: t.MutableJsonMapping, *, project_kind: str = "core"
    ) -> t.StrSequence:
        """Apply canonical coverage settings to one normalized payload."""
        return FlextInfraTomlPhaseService.apply_payload_phases(
            payload, *self._phases(project_kind=project_kind)
        )


__all__: list[str] = ["FlextInfraEnsureCoverageConfigPhase"]
