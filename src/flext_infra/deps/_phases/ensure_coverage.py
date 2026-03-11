"""Phase: Ensure coverage configuration in pyproject.toml."""

from __future__ import annotations

import tomlkit
from tomlkit.items import Table

from flext_infra import c, u
from flext_infra.deps.tool_config import FlextInfraToolConfigDocument


class EnsureCoverageConfigPhase:
    """Ensure coverage report configuration with per-project-type thresholds."""

    def __init__(self, tool_config: FlextInfraToolConfigDocument) -> None:
        self._tool_config = tool_config

    def apply(self, doc: tomlkit.TOMLDocument) -> list[str]:
        changes: list[str] = []
        tool: object | None = None
        if c.Infra.Toml.TOOL in doc:
            tool = doc[c.Infra.Toml.TOOL]
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool

        coverage_tbl = u.Infra.ensure_table(tool, c.Infra.Toml.COVERAGE)
        report_tbl = u.Infra.ensure_table(coverage_tbl, "report")

        # Get coverage config from tool_config
        cov_config = self._tool_config.tools.coverage

        # Set fail_under (using core threshold as default for now)
        current_fail_under = u.Infra.unwrap_item(u.Infra.get(report_tbl, "fail_under"))
        if current_fail_under != cov_config.fail_under_core:
            report_tbl["fail_under"] = cov_config.fail_under_core
            changes.append(
                f"tool.coverage.report.fail_under set to {cov_config.fail_under_core}"
            )

        # Set show_missing
        current_show_missing = u.Infra.unwrap_item(
            u.Infra.get(report_tbl, "show_missing")
        )
        if current_show_missing is not True:
            report_tbl["show_missing"] = True
            changes.append("tool.coverage.report.show_missing set to true")

        # Set skip_covered
        current_skip_covered = u.Infra.unwrap_item(
            u.Infra.get(report_tbl, "skip_covered")
        )
        if current_skip_covered is not False:
            report_tbl["skip_covered"] = False
            changes.append("tool.coverage.report.skip_covered set to false")

        # Set precision
        current_precision = u.Infra.unwrap_item(u.Infra.get(report_tbl, "precision"))
        if current_precision != cov_config.precision:
            report_tbl["precision"] = cov_config.precision
            changes.append(
                f"tool.coverage.report.precision set to {cov_config.precision}"
            )

        return changes
