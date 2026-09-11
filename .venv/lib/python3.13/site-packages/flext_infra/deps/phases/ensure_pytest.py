"""Phase: Ensure canonical fail-closed pytest configuration."""

from __future__ import annotations

from flext_infra import c, m, t
from flext_infra.deps.toml_phase import FlextInfraTomlPhaseService


class FlextInfraEnsurePytestConfigPhase:
    """Ensure canonical pytest policy while preserving extension declarations."""

    def __init__(self, tool_config: m.Infra.ToolConfigDocument) -> None:
        """Store tool configuration used to compose canonical pytest defaults."""
        self._tool_config = tool_config

    def _phase(self) -> m.Infra.Deps.Toml.PhaseConfig:
        """Build the canonical pytest phase definition."""
        pytest = self._tool_config.tools.pytest
        addopts = (*pytest.standard_addopts, f"--timeout={pytest.case_timeout_seconds}")
        return (
            m.Infra.Deps.Toml.PhaseConfig
            .Builder("pytest")
            .table(c.Infra.PYTEST, c.Infra.INI_OPTIONS)
            # flext-j47u (codex): no pytest policy literal survives outside config.
            .value(c.Infra.MINVERSION, pytest.min_version)
            .value(c.Infra.FLEXT_SLOW_TIMEOUT_SECONDS, str(pytest.slow_timeout_seconds))
            .list(
                c.Infra.PYTHON_CLASSES,
                pytest.python_classes,
                strategy=c.Infra.TomlMergeMode.MERGE,
            )
            .list(
                c.Infra.PYTHON_FILES,
                pytest.python_files,
                strategy=c.Infra.TomlMergeMode.MERGE,
            )
            # flext-wkii.17 (codex): replace stale collection roots and warning bypasses.
            .list(
                "testpaths", pytest.test_paths, strategy=c.Infra.TomlMergeMode.REPLACE
            )
            .list(
                c.Infra.ADDOPTS,
                addopts,
                # flext-pulj (codex): replace stale coverage, collection, and bypass flags.
                strategy=c.Infra.TomlMergeMode.REPLACE,
            )
            .list(
                c.Infra.MARKERS,
                pytest.standard_markers,
                strategy=c.Infra.TomlMergeMode.MERGE,
            )
            .list(
                "filterwarnings",
                pytest.filter_warnings,
                strategy=c.Infra.TomlMergeMode.REPLACE,
            )
            .build()
        )

    def apply(self, doc: t.Cli.TomlDocument) -> t.StrSequence:
        """Apply canonical pytest policy while preserving extension declarations."""
        return FlextInfraTomlPhaseService.apply_phases(doc, self._phase())

    def apply_payload(self, payload: t.MutableJsonMapping) -> t.StrSequence:
        """Apply pytest defaults directly to one normalized payload."""
        return FlextInfraTomlPhaseService.apply_payload_phases(payload, self._phase())


__all__: list[str] = ["FlextInfraEnsurePytestConfigPhase"]
