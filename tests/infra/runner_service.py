from __future__ import annotations

from pydantic import Field

from flext_core import r, t
from flext_infra._utilities.subprocess import FlextInfraUtilitiesSubprocess
from flext_infra.models import FlextInfraModels as m
from flext_tests.base import FlextTestsUtilityBase


class RealSubprocessRunner(FlextTestsUtilityBase[str]):
    subprocess_utility: type[FlextInfraUtilitiesSubprocess] = Field(
        default=FlextInfraUtilitiesSubprocess,
        description="Injected subprocess utility implementation.",
    )
    allowed_commands: frozenset[str] = Field(
        default_factory=lambda: frozenset({"echo", "pwd", "ls", "git"}),
        description="Allowlisted root commands for safe execution.",
    )

    def _validate_safe_command(self, cmd: list[str]) -> r[bool]:
        if not cmd:
            return r[bool].fail("command must not be empty")
        if cmd[0] not in self.allowed_commands:
            return r[bool].fail(f"command '{cmd[0]}' is not in the safe allowlist")
        return r[bool].ok(True)

    def _failure_message(self, result: r[m.Infra.Core.CommandOutput]) -> str:
        return result.error or "subprocess execution failed"

    def run_safe(self, cmd: list[str]) -> r[str]:
        validation = self._validate_safe_command(cmd)
        if validation.is_failure:
            return r[str].fail(validation.error or "unsafe command")

        result = self.subprocess_utility.run(cmd)
        if result.is_failure:
            return r[str].fail(self._failure_message(result))
        return r[str].ok(result.value.stdout.strip())

    def capture_output(self, cmd: list[str]) -> r[t.LazyExportType]:
        validation = self._validate_safe_command(cmd)
        if validation.is_failure:
            return r[t.LazyExportType].fail(validation.error or "unsafe command")

        result = self.subprocess_utility.run(cmd)
        if result.is_failure:
            return r[t.LazyExportType].fail(self._failure_message(result))
        output = result.value
        return r[t.LazyExportType].ok((output.stdout.strip(), output.stderr.strip()))


__all__ = ["RealSubprocessRunner"]
