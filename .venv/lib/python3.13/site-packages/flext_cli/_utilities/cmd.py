"""CLI command-service helpers shared through ``u.Cli``."""

from __future__ import annotations

from flext_cli import c, m, p, r, t
from flext_cli._utilities.settings import FlextCliUtilitiesSettings as us
from flext_core import u


class FlextCliUtilitiesCmd:
    """Utility helpers for FlextCliCmd service orchestration."""

    @staticmethod
    def cmd_status() -> m.Cli.RuntimeStatus:
        """Return the canonical public CLI runtime status model."""
        return m.Cli.RuntimeStatus(
            status=c.Cli.ServiceStatus.OPERATIONAL,
            service=c.Cli.FLEXT_CLI,
            timestamp=u.generate("timestamp"),
            version=c.Cli.CLI_VERSION,
            components=m.Cli.RuntimeComponents(
                settings="available",
                formatters="available",
                prompts="available",
                rules="available",
            ),
        )

    @staticmethod
    def cmd_settings_snapshot() -> p.Result[m.Cli.SettingsSnapshot]:
        """Return settings snapshot with canonical error mapping."""
        try:
            return r[m.Cli.SettingsSnapshot].ok(us.settings_snapshot())
        except c.Cli.CLI_SAFE_EXCEPTIONS as exc:
            return r[m.Cli.SettingsSnapshot].fail(
                c.Cli.ERR_SETTINGS_INFO_FAILED.format(error=exc)
            )

    @staticmethod
    def cmd_show_settings(logger: p.Logger) -> p.Result[bool]:
        """Resolve and log current settings snapshot."""
        info_result = FlextCliUtilitiesCmd.cmd_settings_snapshot()
        if info_result.failure:
            return r[bool].fail(
                c.Cli.ERR_SHOW_SETTINGS_FAILED.format(error=info_result.error)
            )
        logger.info(
            c.Cli.LOG_MSG_SETTINGS_DISPLAYED,
            settings=info_result.value.model_dump_json(),
        )
        return r[bool].ok(True)

    @staticmethod
    def cmd_validate_settings(logger: p.Logger) -> p.Result[bool]:
        """Validate canonical settings structure and log normalized results."""
        try:
            results = us.validate_settings_structure()
            if results:
                logger.info(
                    c.Cli.LOG_MSG_SETTINGS_VALIDATION_RESULTS.format(results=results)
                )
            return r[bool].ok(True)
        except c.Cli.CLI_SAFE_EXCEPTIONS as exc:
            return r[bool].fail(c.Cli.ERR_SETTINGS_VALIDATION_FAILED.format(error=exc))


__all__: t.MutableSequenceOf[str] = ["FlextCliUtilitiesCmd"]
