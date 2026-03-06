"""Protocol definitions for flext-infra services and adapters.

Defines structural contracts (runtime-checkable Protocols) for orchestration,
command execution, validation, and reporting services used across the
infrastructure layer.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from flext_core.protocols import FlextProtocols
from flext_core.result import FlextResult

from flext_infra.typings import FlextInfraTypings as t


class FlextInfraProtocols(FlextProtocols):
    """Structural contracts for flext-infra services and adapters.

    All parent protocols (Result, Config, DI, Service, etc.) are inherited
    transparently from ``FlextProtocols`` via MRO.  Infra-specific protocols
    live as nested classes below.
    """

    class Infra:
        @runtime_checkable
        class ProjectInfoProtocol(Protocol):
            """Minimal project descriptor used by orchestration services."""

            @property
            def name(self) -> str:
                """Return the project name."""
                ...

            @property
            def root(self) -> Path:
                """Return the project root path."""
                ...

        @runtime_checkable
        class CommandOutputProtocol(Protocol):
            """Minimal command execution output contract."""

            @property
            def returncode(self) -> int:
                """Return the command exit code."""
                ...

            @property
            def stderr(self) -> str:
                """Return the command standard error."""
                ...

            @property
            def stdout(self) -> str:
                """Return the command standard output."""
                ...

        @runtime_checkable
        class CheckerProtocol(Protocol):
            """Contract for project quality gate runners."""

            def run(
                self,
                project: str,
                gates: Sequence[str],
            ) -> FlextResult[t.Infra.PayloadMap]:
                """Execute quality gates for a project."""
                ...

        @runtime_checkable
        class SyncerProtocol(Protocol):
            """Contract for workspace synchronization services."""

            def sync(
                self,
                source: Path,
                target: Path,
            ) -> FlextResult[t.Infra.PayloadMap]:
                """Synchronize source and target paths."""
                ...

        @runtime_checkable
        class GeneratorProtocol(Protocol):
            """Contract for text/artifact generators."""

            def generate(
                self,
                config: t.Infra.PayloadMap,
            ) -> FlextResult[str]:
                """Generate text or artifacts from configuration."""
                ...

        @runtime_checkable
        class ReporterProtocol(Protocol):
            """Contract for report writers that persist validation outputs."""

            def report(
                self,
                results: Sequence[FlextResult[t.Infra.PayloadMap]],
            ) -> FlextResult[Path]:
                """Write validation results to a report file."""
                ...

        @runtime_checkable
        class ValidatorProtocol(Protocol):
            """Contract for validation services."""

            def validate(self, target: Path) -> FlextResult[bool]:
                """Validate a target path."""
                ...

        @runtime_checkable
        class OrchestratorProtocol(Protocol):
            """Contract for project orchestration services."""

            def orchestrate(
                self,
                projects: Sequence[FlextInfraProtocols.ProjectInfoProtocol],
                verb: str,
            ) -> FlextResult[t.Infra.PayloadMap]:
                """Orchestrate operations across multiple projects."""
                ...

        @runtime_checkable
        class DiscoveryProtocol(Protocol):
            """Contract for project discovery services."""

            def discover(
                self,
                root: Path,
            ) -> FlextResult[list[FlextInfraProtocols.ProjectInfoProtocol]]:
                """Discover projects in a workspace root."""
                ...

        @runtime_checkable
        class CommandRunnerProtocol(Protocol):
            """Contract for command execution services."""

            def run(
                self,
                cmd: Sequence[str],
                cwd: Path | None = None,
            ) -> FlextResult[FlextInfraProtocols.CommandOutputProtocol]:
                """Execute a command and return output."""
                ...


p = FlextInfraProtocols

__all__ = [
    "FlextInfraProtocols",
    "p",
]
