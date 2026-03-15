"""Protocol definitions for flext-infra utilities and services.

Defines structural contracts (runtime-checkable Protocols) for orchestration,
command execution, validation, and reporting services used across the
infrastructure layer.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from flext_core import FlextProtocols, r
from flext_infra.models import m
from flext_infra.typings import t


class FlextInfraProtocols(FlextProtocols):
    """Structural contracts for flext-infra utilities and services.

    All parent protocols (Result, Config, DI, Service, etc.) are inherited
    transparently from ``FlextProtocols`` via MRO. Infra-specific utility
    protocols live as nested classes below.
    """

    class Infra:
        """Infra-specific structural protocol definitions."""

        @runtime_checkable
        class ProjectInfo(Protocol):
            """Minimal project descriptor used by orchestration services."""

            @property
            def name(self) -> str:
                """Return the project name."""
                ...

            @property
            def path(self) -> Path:
                """Return the project path."""
                ...

        @runtime_checkable
        class CommandOutput(Protocol):
            """Minimal command execution output contract."""

            @property
            def exit_code(self) -> int:
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
            ) -> r[list[m.Infra.Check.ProjectResult]]:
                """Execute quality gates for a project."""
                ...

        @runtime_checkable
        class Syncer(Protocol):
            """Contract for workspace synchronization services."""

            def sync(
                self,
                workspace_root: Path,
                *,
                config: m.Infra.Basemk.BaseMkConfig | None = None,
                canonical_root: Path | None = None,
            ) -> r[m.Infra.Workspace.SyncResult]:
                """Synchronize source and target paths."""
                ...

        @runtime_checkable
        class Generator(Protocol):
            """Contract for text/artifact generators."""

            def generate(
                self,
                config: m.Infra.Basemk.BaseMkConfig | object | None = None,
            ) -> r[str]:
                """Generate text or artifacts from configuration."""
                ...

        @runtime_checkable
        class Reporter(Protocol):
            """Contract for report writers that persist validation outputs."""

            def report(self, results: Sequence[m.Infra.Check.ProjectResult]) -> r[Path]:
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
                projects: Sequence[str],
                verb: str,
            ) -> r[list[m.Infra.Core.CommandOutput]]:
                """Orchestrate operations across multiple projects."""
                ...

        @runtime_checkable
        class Discovery(Protocol):
            """Contract for project discovery services."""

            def discover_projects(
                self,
                workspace_root: Path,
            ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
                """Discover projects in a workspace root."""
                ...

        @runtime_checkable
        class TomlReader(Protocol):
            """Contract for TOML file readers used by dependency services."""

            def read_plain(self, path: Path) -> r[t.Infra.TomlConfig]:
                """Read and parse a TOML file as a plain dict with r error handling."""
                ...

        @runtime_checkable
        class CommandRunner(Protocol):
            """Contract for command execution services."""

            def run(
                self,
                cmd: Sequence[str],
                cwd: Path | None = None,
                timeout: int | None = None,
                env: Mapping[str, str] | None = None,
            ) -> r[m.Infra.Core.CommandOutput]:
                """Execute a command and return output."""
                ...

            def capture(
                self,
                cmd: Sequence[str],
                cwd: Path | None = None,
                timeout: int | None = None,
                env: Mapping[str, str] | None = None,
            ) -> r[str]:
                """Execute a command and return captured stdout."""
                ...

            def run_raw(
                self,
                cmd: Sequence[str],
                cwd: Path | None = None,
                timeout: int | None = None,
                env: Mapping[str, str] | None = None,
            ) -> r[m.Infra.Core.CommandOutput]:
                """Execute a command and return raw output, including non-zero exit codes."""
                ...

            def run_checked(
                self,
                cmd: Sequence[str],
                cwd: Path | None = None,
                timeout: int | None = None,
                env: Mapping[str, str] | None = None,
            ) -> r[bool]:
                """Execute a command and return success/failure status."""
                ...

            def run_to_file(
                self,
                cmd: Sequence[str],
                output_file: Path,
                cwd: Path | None = None,
                timeout: int | None = None,
                env: Mapping[str, str] | None = None,
            ) -> r[int]:
                """Execute a command and write combined output to file."""
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

        @runtime_checkable
        class Scanner(Protocol):
            """Protocol for file scanners that detect violations."""

            def scan_file(self, *, file_path: Path) -> m.Infra.Utilities.ScanResult:
                """Scan a single file and return scan result."""
                ...


p = FlextInfraProtocols
__all__ = ["FlextInfraProtocols", "p"]
