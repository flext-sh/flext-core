"""Phase: Ensure standard mypy configuration with pydantic plugin across all projects."""

from __future__ import annotations

from flext_infra import c, config, m, t, u
from flext_infra.deps.toml_phase import FlextInfraTomlPhaseService


class FlextInfraEnsureMypyConfigPhase:
    """Ensure standard mypy configuration with pydantic plugin across all projects."""

    def __init__(self, tool_config: m.Infra.ToolConfigDocument) -> None:
        """Store tool configuration used to generate the canonical mypy section."""
        self._tool_config = tool_config

    def _phase(self) -> m.Infra.Deps.Toml.PhaseConfig:
        """Build the canonical mypy phase definition."""
        configured = self._tool_config.tools.mypy.overrides
        expected_overrides: t.SequenceOf[t.JsonDict] = [
            {
                "module": u.normalize_to_json_value(list(entry.modules)),
                "disable_error_code": u.normalize_to_json_value(
                    list(entry.disable_error_codes)
                ),
            }
            for entry in configured
        ]
        phase_builder = (
            m.Infra.Deps.Toml.PhaseConfig
            .Builder("mypy")
            .table(c.Infra.MYPY)
            .deprecated("strict_concatenate")
            # flext-j47u (codex): tool Python derives from the codegen toolchain SSOT.
            .value(
                c.Infra.PYTHON_VERSION_UNDERSCORE,
                config.Infra.codegen.toolchain.python_version,
            )
            .list(
                c.Infra.PLUGINS,
                self._tool_config.tools.mypy.plugins,
                strategy=c.Infra.TomlMergeMode.REPLACE,
            )
            .list(
                c.Infra.DISABLE_ERROR_CODE,
                tuple(sorted(self._tool_config.tools.mypy.disabled_error_codes)),
                strategy=c.Infra.TomlMergeMode.REPLACE,
            )
        )
        if self._tool_config.tools.mypy.exclude:
            phase_builder = phase_builder.value(
                c.Infra.EXCLUDE, self._tool_config.tools.mypy.exclude
            )
        else:
            phase_builder = phase_builder.deprecated(c.Infra.EXCLUDE)
        overrides_payload: t.JsonValueList = list(expected_overrides)
        phase_builder = phase_builder.value("overrides", overrides_payload)
        for key, value in self._tool_config.tools.mypy.boolean_settings.items():
            phase_builder = phase_builder.value(key, value)
        for key, value in self._tool_config.tools.mypy.string_settings.items():
            phase_builder = phase_builder.value(key, value)
        return phase_builder.build()

    def apply(self, doc: t.Cli.TomlDocument) -> t.StrSequence:
        """Apply mypy defaults, overrides, and toggles from tool configuration."""
        return FlextInfraTomlPhaseService.apply_phases(doc, self._phase())

    def apply_payload(self, payload: t.MutableJsonMapping) -> t.StrSequence:
        """Apply canonical mypy settings directly to one normalized payload."""
        return FlextInfraTomlPhaseService.apply_payload_phases(payload, self._phase())


__all__: list[str] = ["FlextInfraEnsureMypyConfigPhase"]
