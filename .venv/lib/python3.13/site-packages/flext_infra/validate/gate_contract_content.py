"""Gate contract content checks."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, m

if TYPE_CHECKING:
    from flext_infra import t


class FlextInfraGateContractContentMixin:
    """Validate script body content."""

    @staticmethod
    def _check_interactive(
        script: str, content: str, extension: str
    ) -> t.SequenceOf[m.Infra.GateContractViolation]:
        if c.Infra.SKILL_INTERACTIVE_GATE_RE.search(content):
            return ()

        pattern = (
            c.Infra.SKILL_INTERACTIVE_PY_RE
            if extension == ".py"
            else c.Infra.SKILL_INTERACTIVE_SH_RE
        )
        violations: list[m.Infra.GateContractViolation] = []
        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if extension == ".py" and stripped.startswith(('"""', "'''", '"', "'")):
                continue
            if pattern.search(line):
                violations.append(
                    m.Infra.GateContractViolation(
                        check="interactive",
                        message=(
                            f"line {i}: interactive prompt without --interactive gate"
                        ),
                        script=script,
                        severity=c.Infra.GateSeverity.WARNING.value,
                    )
                )
        return tuple(violations)

    @staticmethod
    def _check_artifact_naming(
        script: str, content: str
    ) -> t.SequenceOf[m.Infra.GateContractViolation]:
        violations: list[m.Infra.GateContractViolation] = []
        for i, line in enumerate(content.splitlines(), 1):
            for match in c.Infra.SKILL_REPORTS_PATH_RE.finditer(line):
                filename = Path(match.group(1)).name
                if "$" in filename or "*" in filename or "{" in filename:
                    continue
                if filename == ".gitkeep":
                    continue
                if c.Infra.SKILL_REPORT_ARTIFACT_NAME_RE.fullmatch(filename):
                    continue
                violations.append(
                    m.Infra.GateContractViolation(
                        check="artifact_naming",
                        message=(
                            f"line {i}: artifact '{filename}' does not match "
                            "<skill>--<kind>--<slug>.<ext>"
                        ),
                        script=script,
                        severity=c.Infra.GateSeverity.WARNING.value,
                    )
                )
        return tuple(violations)


__all__: list[str] = ["FlextInfraGateContractContentMixin"]
