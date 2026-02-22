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
        def name(self) -> str: ...

        @property
        def root(self) -> object: ...

    @runtime_checkable
    class CommandOutput(Protocol):
        """Minimal command execution output contract."""

        @property
        def stdout(self) -> str: ...

        @property
        def stderr(self) -> str: ...

        @property
        def returncode(self) -> int: ...

    @runtime_checkable
    class CheckerProtocol(Protocol):
        """Contract for project quality gate runners."""

        def run(
            self,
            project: str,
            gates: Sequence[str],
        ) -> FlextResult[object]: ...

    @runtime_checkable
    class SyncerProtocol(Protocol):
        """Contract for workspace synchronization services."""

        def sync(
            self,
            source: object,
            target: object,
        ) -> FlextResult[object]: ...

    @runtime_checkable
    class GeneratorProtocol(Protocol):
        """Contract for text/artifact generators."""

        def generate(
            self,
            config: Mapping[str, object],
        ) -> FlextResult[str]: ...

    @runtime_checkable
    class ReporterProtocol(Protocol):
        """Contract for report writers that persist validation outputs."""

        def report(
            self,
            results: Sequence[FlextResult[object]],
        ) -> FlextResult[Path]: ...

    @runtime_checkable
    class ValidatorProtocol(Protocol):
        """Contract for validation services."""

        def validate(self, target: object) -> FlextResult[bool]: ...

    @runtime_checkable
    class OrchestratorProtocol(Protocol):
        """Contract for project orchestration services."""

        def orchestrate(
            self,
            projects: Sequence[InfraProtocols.ProjectInfo],
            verb: str,
        ) -> FlextResult[object]: ...

    @runtime_checkable
    class DiscoveryProtocol(Protocol):
        """Contract for project discovery services."""

        def discover(
            self,
            root: object,
        ) -> FlextResult[list[InfraProtocols.ProjectInfo]]: ...

    @runtime_checkable
    class CommandRunnerProtocol(Protocol):
        """Contract for command execution services."""

        def run(
            self,
            cmd: Sequence[str],
            cwd: Path | None = None,
        ) -> FlextResult[InfraProtocols.CommandOutput]: ...


ip = InfraProtocols

FlextInfraProtocols = InfraProtocols
p = InfraProtocols

__all__ = [
    "FlextInfraProtocols",
    "InfraProtocols",
    "ip",
    "p",
]
