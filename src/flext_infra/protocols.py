from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from flext_core import FlextResult


class InfraProtocols:
    @runtime_checkable
    class ProjectInfo(Protocol):
        @property
        def name(self) -> str: ...

        @property
        def root(self) -> object: ...

    @runtime_checkable
    class CommandOutput(Protocol):
        @property
        def stdout(self) -> str: ...

        @property
        def stderr(self) -> str: ...

        @property
        def returncode(self) -> int: ...

    @runtime_checkable
    class CheckerProtocol(Protocol):
        def run(
            self,
            project: str,
            gates: Sequence[str],
        ) -> FlextResult[object]: ...

    @runtime_checkable
    class SyncerProtocol(Protocol):
        def sync(
            self,
            source: object,
            target: object,
        ) -> FlextResult[object]: ...

    @runtime_checkable
    class GeneratorProtocol(Protocol):
        def generate(
            self,
            config: Mapping[str, object],
        ) -> FlextResult[str]: ...

    @runtime_checkable
    class ReporterProtocol(Protocol):
        def report(
            self,
            results: Sequence[FlextResult[object]],
        ) -> FlextResult[Path]: ...

    @runtime_checkable
    class ValidatorProtocol(Protocol):
        def validate(self, target: object) -> FlextResult[bool]: ...

    @runtime_checkable
    class OrchestratorProtocol(Protocol):
        def orchestrate(
            self,
            projects: Sequence[InfraProtocols.ProjectInfo],
            verb: str,
        ) -> FlextResult[object]: ...

    @runtime_checkable
    class DiscoveryProtocol(Protocol):
        def discover(
            self,
            root: object,
        ) -> FlextResult[list[InfraProtocols.ProjectInfo]]: ...

    @runtime_checkable
    class CommandRunnerProtocol(Protocol):
        def run(
            self,
            cmd: Sequence[str],
            cwd: Path | None = None,
        ) -> FlextResult[InfraProtocols.CommandOutput]: ...


ip = InfraProtocols

FlextInfraProtocols = InfraProtocols
p = InfraProtocols

__all__ = [
    "InfraProtocols",
    "FlextInfraProtocols",
    "ip",
    "p",
]
