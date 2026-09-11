"""Gate contract reporting."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, m, t, u

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraGateContractReportMixin:
    """Render gate contract validation output."""

    @staticmethod
    def _eprint(message: str) -> None:
        _ = sys.stderr.write(f"{message}\n")

    @staticmethod
    def _report_path_for(root: Path) -> Path:
        return root / ".claude" / "skills" / "scripts-infra" / "report.json"

    @staticmethod
    def _severity_count(script: m.Infra.GateContractScriptInfo, severity: str) -> int:
        return sum(
            1 for violation in script.violations if violation.severity == severity
        )

    @staticmethod
    def _visible_scripts(
        scripts: t.SequenceOf[m.Infra.GateContractScriptInfo],
    ) -> t.SequenceOf[m.Infra.GateContractScriptInfo]:
        return tuple(
            script
            for script in scripts
            if script.role != "other" or bool(script.violations)
        )

    @staticmethod
    def _gate_scripts(
        scripts: t.SequenceOf[m.Infra.GateContractScriptInfo],
    ) -> t.SequenceOf[m.Infra.GateContractScriptInfo]:
        return tuple(
            script for script in scripts if script.role in {"validator", "fixer"}
        )

    @staticmethod
    def _status_for(errors: int, warnings: int) -> str:
        if errors > 0:
            return f"{c.Infra.RED}FAIL{c.Infra.RESET}"
        if warnings > 0:
            return f"{c.Infra.YELLOW}WARN{c.Infra.RESET}"
        return f"{c.Infra.GREEN}OK{c.Infra.RESET}"

    @staticmethod
    def _detail_for(errors: int, warnings: int) -> str:
        details = [
            *([f"{errors} error(s)"] if errors > 0 else []),
            *([f"{warnings} warning(s)"] if warnings > 0 else []),
        ]
        return ", ".join(details) if details else "contract compliant"

    def _print_script_result(self, script: m.Infra.GateContractScriptInfo) -> None:
        errors = self._severity_count(script, c.Infra.GateSeverity.ERROR.value)
        warnings = self._severity_count(script, c.Infra.GateSeverity.WARNING.value)
        u.Cli.formatters_print(
            f"{script.path:<60} {script.role:<10} "
            f"{self._status_for(errors, warnings):<22} "
            f"{self._detail_for(errors, warnings)}"
        )
        for violation in script.violations:
            color = (
                c.Infra.RED
                if violation.severity == c.Infra.GateSeverity.ERROR.value
                else c.Infra.YELLOW
            )
            u.Cli.formatters_print(
                f"  {color}[{violation.check}]{c.Infra.RESET} {violation.message}"
            )

    def _print_results(
        self, scripts: t.SequenceOf[m.Infra.GateContractScriptInfo]
    ) -> None:
        u.Cli.formatters_print(f"{c.Infra.CYAN}Gate Contract Validation{c.Infra.RESET}")
        u.Cli.formatters_print(
            f"{c.Infra.CYAN}{'SCRIPT':<60} {'ROLE':<10} {'STATUS':<10} DETAILS{c.Infra.RESET}"
        )
        for script in self._visible_scripts(scripts):
            self._print_script_result(script)

    @staticmethod
    def _violation_rows(
        scripts: t.SequenceOf[m.Infra.GateContractScriptInfo],
    ) -> t.SequenceOf[t.JsonDict]:
        def violation_key(row: t.JsonDict) -> tuple[str, str]:
            return str(row.get("script", "")), str(row.get("check", ""))

        rows = [
            t.json_dict_adapter().validate_python(violation.model_dump())
            for script in scripts
            for violation in script.violations
        ]
        return tuple(sorted(rows, key=violation_key))

    def _summary_for(
        self, scripts: t.SequenceOf[m.Infra.GateContractScriptInfo]
    ) -> m.Infra.GateContractSummary:
        gate_scripts = self._gate_scripts(scripts)
        errors = sum(
            self._severity_count(script, c.Infra.GateSeverity.ERROR.value)
            for script in scripts
        )
        warnings = sum(
            self._severity_count(script, c.Infra.GateSeverity.WARNING.value)
            for script in scripts
        )
        ok = sum(
            1
            for script in gate_scripts
            if self._severity_count(script, c.Infra.GateSeverity.ERROR.value) == 0
        )
        return m.Infra.GateContractSummary(
            errors=errors, gate_scripts=len(gate_scripts), ok=ok, warnings=warnings
        )

    def _write_report(
        self,
        root: Path,
        scripts: t.SequenceOf[m.Infra.GateContractScriptInfo],
        mode: str,
    ) -> p.Result[Path]:
        report_path = self._report_path_for(root)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        summary = self._summary_for(scripts)
        violations: t.JsonValue = t.json_value_adapter().validate_python(
            list(self._violation_rows(scripts))
        )
        payload: t.JsonMapping = {
            "checked": len(self._visible_scripts(scripts)),
            "errors": summary.errors,
            "mode": mode,
            "violations": violations,
            "warnings": summary.warnings,
        }
        write = u.Cli.json_write(
            report_path,
            payload,
            options=m.Cli.JsonWriteOptions(indent=2, sort_keys=True),
        )
        if write.failure:
            return r[Path].fail(write.error or f"cannot write {report_path}")
        return r[Path].ok(report_path)

    def _print_summary(
        self, summary: m.Infra.GateContractSummary, report_path: Path
    ) -> None:
        u.Cli.formatters_print(
            f"\n{c.Infra.CYAN}Summary:{c.Infra.RESET} "
            f"gate_scripts={summary.gate_scripts} "
            f"{c.Infra.GREEN}ok={summary.ok}{c.Infra.RESET} "
            f"{c.Infra.RED}errors={summary.errors}{c.Infra.RESET} "
            f"{c.Infra.YELLOW}warnings={summary.warnings}{c.Infra.RESET}"
        )
        u.Cli.formatters_print(f"Report: {report_path}")


__all__: list[str] = ["FlextInfraGateContractReportMixin"]
