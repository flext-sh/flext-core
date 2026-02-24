"""Protocol definitions for flext-infra services and adapters.

Defines structural contracts (runtime-checkable Protocols) for orchestration,
command execution, validation, and reporting services used across the
infrastructure layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from flext_core.result import FlextResult


class InfraProtocols:
    """Structural contracts for flext-infra services and adapters."""

    @runtime_checkable
    class ProjectInfo(Protocol):
        """Minimal project descriptor used by orchestration services."""

        @property
        def name(self) -> str:
            """Return the project name."""
            ...

        @property
        def root(self) -> object:
            """Return the project root path."""
            ...

    @runtime_checkable
    class CommandOutput(Protocol):
        """Minimal command execution output contract."""

        @property
        def stdout(self) -> str:
            """Return the command standard output."""
            ...

        @property
        def stderr(self) -> str:
            """Return the command standard error."""
            ...

        @property
        def returncode(self) -> int:
            """Return the command exit code."""
            ...

    @runtime_checkable
    class CheckerProtocol(Protocol):
        """Contract for project quality gate runners."""

        def run(
            self,
            project: str,
            gates: Sequence[str],
        ) -> FlextResult[object]:
            """Execute quality gates for a project."""
            ...

    @runtime_checkable
    class SyncerProtocol(Protocol):
        """Contract for workspace synchronization services."""

        def sync(
            self,
            source: object,
            target: object,
        ) -> FlextResult[object]:
            """Synchronize source and target objects."""
            ...

    @runtime_checkable
    class GeneratorProtocol(Protocol):
        """Contract for text/artifact generators."""

        def generate(
            self,
            config: Mapping[str, object],
        ) -> FlextResult[str]:
            """Generate text or artifacts from configuration."""
            ...

    @runtime_checkable
    class ReporterProtocol(Protocol):
        """Contract for report writers that persist validation outputs."""

        def report(
            self,
            results: Sequence[FlextResult[object]],
        ) -> FlextResult[Path]:
            """Write validation results to a report file."""
            ...

    @runtime_checkable
    class ValidatorProtocol(Protocol):
        """Contract for validation services."""

        def validate(self, target: object) -> FlextResult[bool]:
            """Validate a target object."""
            ...

    @runtime_checkable
    class OrchestratorProtocol(Protocol):
        """Contract for project orchestration services."""

        def orchestrate(
            self,
            projects: Sequence[InfraProtocols.ProjectInfo],
            verb: str,
        ) -> FlextResult[object]:
            """Orchestrate operations across multiple projects."""
            ...

    @runtime_checkable
    class DiscoveryProtocol(Protocol):
        """Contract for project discovery services."""

        def discover(
            self,
            root: object,
        ) -> FlextResult[list[InfraProtocols.ProjectInfo]]:
            """Discover projects in a workspace root."""
            ...

    @runtime_checkable
    class CommandRunnerProtocol(Protocol):
        """Contract for command execution services."""

        def run(
            self,
            cmd: Sequence[str],
            cwd: Path | None = None,
        ) -> FlextResult[InfraProtocols.CommandOutput]:
            """Execute a command and return output."""
            ...


ip = InfraProtocols

FlextInfraProtocols = InfraProtocols
p = InfraProtocols

__all__ = [
    "FlextInfraProtocols",
    "InfraProtocols",
    "ip",
    "p",
]
