"""Phase: Ensure coverage configuration in pyproject.toml."""

from __future__ import annotations

import tomlkit
from tomlkit.items import Item, Table

from flext_infra import c, u
from flext_infra.deps._models import FlextInfraDepsModels


class EnsureCoverageConfigPhase:
    """Ensure coverage report configuration with per-project-type thresholds."""

    def __init__(self, tool_config: FlextInfraDepsModels.ToolConfigDocument) -> None:
        self._tool_config = tool_config

    def apply(
        self,
        doc: tomlkit.TOMLDocument,
        *,
        project_kind: str = "core",
    ) -> list[str]:
        changes: list[str] = []
        tool: Item | None = None
        if c.Infra.Toml.TOOL in doc:
            raw_tool = doc[c.Infra.Toml.TOOL]
            if isinstance(raw_tool, Item):
                tool = raw_tool
        if not isinstance(tool, Table):
            tool = tomlkit.table()
            doc[c.Infra.Toml.TOOL] = tool

        coverage_tbl = u.Infra.ensure_table(tool, "coverage")
        report_tbl = u.Infra.ensure_table(coverage_tbl, "report")

        cov_config = self._tool_config.tools.coverage
        fail_under_map: dict[str, int] = {
            "core": cov_config.fail_under.core,
            "domain": cov_config.fail_under.domain,
            "platform": cov_config.fail_under.platform,
            "integration": cov_config.fail_under.integration,
            "app": cov_config.fail_under.app,
        }
        fail_under = fail_under_map.get(project_kind, cov_config.fail_under.core)

        current_fail_under = u.Infra.unwrap_item(u.Infra.get(report_tbl, "fail_under"))
        if current_fail_under != fail_under:
            report_tbl["fail_under"] = fail_under
            changes.append(f"tool.coverage.report.fail_under set to {fail_under}")

        current_show_missing = u.Infra.unwrap_item(
            u.Infra.get(report_tbl, "show_missing")
        )
        if current_show_missing is not True:
            report_tbl["show_missing"] = True
            changes.append("tool.coverage.report.show_missing set to true")

        current_skip_covered = u.Infra.unwrap_item(
            u.Infra.get(report_tbl, "skip_covered")
        )
        if current_skip_covered is not False:
            report_tbl["skip_covered"] = False
            changes.append("tool.coverage.report.skip_covered set to false")

        current_precision = u.Infra.unwrap_item(u.Infra.get(report_tbl, "precision"))
        if current_precision != cov_config.precision:
            report_tbl["precision"] = cov_config.precision
            changes.append(
                f"tool.coverage.report.precision set to {cov_config.precision}"
            )

        return changes
