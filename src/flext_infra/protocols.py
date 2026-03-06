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

from flext_core import FlextProtocols, r
from flext_infra.typings import t


class FlextInfraProtocols(FlextProtocols):
    """Structural contracts for flext-infra services and adapters.

    All parent protocols (Result, Config, DI, Service, etc.) are inherited
    transparently from ``FlextProtocols`` via MRO.  Infra-specific protocols
    live as nested classes below.
    """

    class Infra:
        """Infrastructure-domain protocols."""

        @runtime_checkable
        class ProjectInfo(Protocol):
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
        class CommandOutput(Protocol):
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
        class Checker(Protocol):
            """Contract for project quality gate runners."""

            def run(
                self,
                project: str,
                gates: Sequence[str],
            ) -> r[t.Infra.PayloadMap]:
                """Execute quality gates for a project."""
                ...

        @runtime_checkable
        class Syncer(Protocol):
            """Contract for workspace synchronization services."""

            def sync(
                self,
                source: Path,
                target: Path,
            ) -> r[t.Infra.PayloadMap]:
                """Synchronize source and target paths."""
                ...

        @runtime_checkable
        class Generator(Protocol):
            """Contract for text/artifact generators."""

            def generate(
                self,
                config: t.Infra.PayloadMap,
            ) -> r[str]:
                """Generate text or artifacts from configuration."""
                ...

        @runtime_checkable
        class Reporter(Protocol):
            """Contract for report writers that persist validation outputs."""

            def report(
                self,
                results: Sequence[r[t.Infra.PayloadMap]],
            ) -> r[Path]:
                """Write validation results to a report file."""
                ...

        @runtime_checkable
        class Validator(Protocol):
            """Contract for validation services."""

            def validate(self, target: Path) -> r[bool]:
                """Validate a target path."""
                ...

        @runtime_checkable
        class Orchestrator(Protocol):
            """Contract for project orchestration services."""

            def orchestrate(
                self,
                projects: Sequence[FlextInfraProtocols.Infra.ProjectInfo],
                verb: str,
            ) -> r[t.Infra.PayloadMap]:
                """Orchestrate operations across multiple projects."""
                ...

        @runtime_checkable
        class Discovery(Protocol):
            """Contract for project discovery services."""

            def discover(
                self,
                root: Path,
            ) -> r[list[FlextInfraProtocols.Infra.ProjectInfo]]:
                """Discover projects in a workspace root."""
                ...

        @runtime_checkable
        class CommandRunner(Protocol):
            """Contract for command execution services."""

            def run(
                self,
                cmd: Sequence[str],
                cwd: Path | None = None,
            ) -> r[FlextInfraProtocols.Infra.CommandOutput]:
                """Execute a command and return output."""
                ...

        @runtime_checkable
        class SafetyRunner(Protocol):
            """Protocol for command execution backends used by the safety manager."""

            def capture(
                self,
                cmd: list[str],
                cwd: Path | None = None,
                timeout: int | None = None,
            ) -> r[str]:
                """Run a command and capture its stdout."""
                ...

            def run_checked(
                self,
                cmd: list[str],
                cwd: Path | None = None,
                timeout: int | None = None,
            ) -> r[bool]:
                """Run a command and return success/failure."""
                ...


p = FlextInfraProtocols

__all__ = [
    "FlextInfraProtocols",
    "p",
]
