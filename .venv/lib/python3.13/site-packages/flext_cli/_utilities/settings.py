"""CLI settings helpers shared through ``u.Cli``."""

from __future__ import annotations

import os
from pathlib import Path
from typing import overload

from flext_cli import c, m, p, t
from flext_core import u


class FlextCliUtilitiesSettings:
    """Settings and selector methods exposed directly on ``u.Cli``."""

    @staticmethod
    def cli_test_env(cli_settings: p.Cli.CliSettings) -> bool:
        """Detect test/CI runtime from the flat CLI settings scalars.

        NOTE (multi-agent): replaces the removed ``settings.Cli.test_env``
        computed field — behavior lives in the utilities layer, settings stay
        pure flat data (§2.6).
        """
        normalized_shell = (cli_settings.cli_shell_command or "").strip().lower()
        return (
            cli_settings.cli_pytest_current_test is not None
            or "pytest" in normalized_shell
            or cli_settings.cli_ci
        )

    @staticmethod
    def project_names_from_values(
        *values: t.Cli.ProjectNamesValue | None,
    ) -> t.MutableSequenceOf[str] | None:
        """Normalize repeated or comma-separated CLI selector values."""
        names: t.MutableSequenceOf[str] = []
        for value in values:
            if value is None:
                continue
            raw_values = [value] if isinstance(value, str) else list(value)
            for raw_value in raw_values:
                for comma_group in raw_value.split(","):
                    names.extend(
                        item.strip() for item in comma_group.split() if item.strip()
                    )
        return names or None

    @overload
    @staticmethod
    def project_numbers_from_values(
        *values: t.Cli.ProjectNamesValue | None, default: t.SequenceOf[int]
    ) -> t.MutableSequenceOf[int]: ...

    @overload
    @staticmethod
    def project_numbers_from_values(
        *values: t.Cli.ProjectNamesValue | None, default: None = None
    ) -> t.MutableSequenceOf[int] | None: ...

    @staticmethod
    def project_numbers_from_values(
        *values: t.Cli.ProjectNamesValue | None,
        default: t.SequenceOf[int] | None = None,
    ) -> t.MutableSequenceOf[int] | None:
        """Normalize selector values into integers with optional default fallback."""
        names = FlextCliUtilitiesSettings.project_names_from_values(*values)
        if names is None:
            return list(default) if default is not None else None
        return [int(name) for name in names]

    @staticmethod
    def settings_snapshot() -> m.Cli.SettingsSnapshot:
        """Return the canonical CLI settings snapshot."""
        path = Path.home() / c.Cli.PATH_FLEXT_DIR_NAME
        exists = path.exists()
        return m.Cli.SettingsSnapshot(
            settings_dir=str(path),
            settings_exists=exists,
            settings_readable=exists and os.access(path, os.R_OK),
            settings_writable=exists and os.access(path, os.W_OK),
            timestamp=u.now().isoformat(),
        )

    @staticmethod
    def validate_settings_structure() -> t.StrSequence:
        """Validate the canonical CLI settings directory structure."""
        base = Path.home() / c.Cli.PATH_FLEXT_DIR_NAME
        ok = c.Cli.SYMBOL_SUCCESS_MARK
        fail = c.Cli.SYMBOL_FAILURE_MARK
        lines = [
            f"{ok} Settings directory exists"
            if base.exists()
            else f"{fail} Settings directory missing"
        ]
        for subdir in c.Cli.STANDARD_SUBDIRS:
            path = base / subdir
            lines.append(
                c.Cli.MSG_SUBDIR_EXISTS.format(symbol=ok, subdir=subdir)
                if path.exists()
                else c.Cli.MSG_SUBDIR_MISSING.format(symbol=fail, subdir=subdir)
            )
        return lines


__all__: t.MutableSequenceOf[str] = ["FlextCliUtilitiesSettings"]
