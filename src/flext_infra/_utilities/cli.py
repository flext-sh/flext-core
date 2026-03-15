"""CLI argument parsing utilities for flext-infra commands.

Centralizes argument parser creation and argument resolution for all
flext_infra CLI commands, providing a consistent interface for workspace,
apply/dry-run, format, check, and project selection flags.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import BaseModel

from flext_core import FlextRuntime, r
from flext_core.models import FlextModels
from flext_infra import t
from flext_infra._utilities.discovery import FlextInfraUtilitiesDiscovery
from flext_infra._utilities.output import output
from flext_infra.models import FlextInfraModels as m


class FlextInfraUtilitiesCli:
    """Static facade for CLI argument parsing and resolution.

    Provides standardized argument parser creation and resolution for
    flext_infra commands, supporting optional flags for apply/dry-run,
    output format, check mode, and project selection.
    """

    class CliArgs(FlextModels.FrozenStrictModel):
        """Parsed CLI arguments with strict validation.

        Immutable model representing resolved command-line arguments,
        with optional fields for apply, format, check, and project selection.
        """

        workspace: Path = Path.cwd()
        apply: bool = False
        output_format: str = "text"
        check: bool = False
        project: str | None = None
        projects: str | None = None

        @property
        def dry_run(self) -> bool:
            return not self.apply

        @property
        def mode_label(self) -> str:
            return "apply" if self.apply else "dry-run"

        def project_names(self) -> list[str] | None:
            names: list[str] = []
            if self.project:
                names.append(self.project)
            if self.projects:
                names.extend(p.strip() for p in self.projects.split(",") if p.strip())
            return names or None

        def project_dirs(self) -> list[Path] | None:
            names = self.project_names()
            if names is None:
                return None
            return [self.workspace / name for name in names]

    @staticmethod
    def _shared_flags_parser(
        *,
        include_apply: bool = True,
        include_format: bool = False,
        include_check: bool = False,
        include_project: bool = False,
    ) -> ArgumentParser:
        base = ArgumentParser(add_help=False)
        _ = base.add_argument(
            "--workspace",
            type=Path,
            default=Path.cwd(),
            help="Workspace root directory (default: cwd)",
        )
        if include_apply:
            mode = base.add_mutually_exclusive_group(required=False)
            _ = mode.add_argument(
                "--dry-run", action="store_true", help="Plan/Scan only"
            )
            _ = mode.add_argument("--apply", action="store_true", help="Apply changes")
        if include_format:
            _ = base.add_argument(
                "--format",
                dest="output_format",
                choices=["json", "text"],
                default="text",
                help="Output format (default: text)",
            )
        if include_check:
            _ = base.add_argument(
                "--check", action="store_true", help="Run in check mode"
            )
        if include_project:
            _ = base.add_argument(
                "--project",
                type=str,
                default=None,
                help="Single project to process",
            )
            _ = base.add_argument(
                "--projects",
                type=str,
                default=None,
                help="Multiple projects (comma-separated or glob pattern)",
            )
        return base

    @staticmethod
    def create_subcommand_parser(
        prog: str,
        description: str,
        *,
        subcommands: Mapping[str, str],
        include_apply: bool = True,
        include_format: bool = False,
        include_check: bool = False,
        include_project: bool = False,
    ) -> tuple[ArgumentParser, dict[str, ArgumentParser]]:
        shared = FlextInfraUtilitiesCli._shared_flags_parser(
            include_apply=include_apply,
            include_format=include_format,
            include_check=include_check,
            include_project=include_project,
        )
        parser = ArgumentParser(prog=prog, description=description, parents=[shared])
        subparsers = parser.add_subparsers(dest="command")
        command_parsers: dict[str, ArgumentParser] = {}
        for command, command_help in subcommands.items():
            command_parsers[command] = subparsers.add_parser(
                command,
                help=command_help,
                parents=[shared],
            )
        return parser, command_parsers

    @staticmethod
    def create_parser(
        prog: str,
        description: str,
        *,
        include_apply: bool = True,
        include_format: bool = False,
        include_check: bool = False,
        include_project: bool = False,
    ) -> ArgumentParser:
        """Create a standard ArgumentParser with common CLI flags.

        Args:
            prog: Program name for the parser.
            description: Description of the command.
            include_apply: If True, add --dry-run/--apply mutually exclusive group.
            include_format: If True, add --format flag for output format selection.
            include_check: If True, add --check flag for validation mode.
            include_project: If True, add --project and --projects flags.

        Returns:
            Configured ArgumentParser instance.

        """
        parser = ArgumentParser(prog=prog, description=description)

        # Always add workspace argument
        _ = parser.add_argument(
            "--workspace",
            type=Path,
            default=Path.cwd(),
            help="Workspace root directory (default: cwd)",
        )

        # Add apply/dry-run group if requested
        if include_apply:
            mode = parser.add_mutually_exclusive_group(required=False)
            _ = mode.add_argument(
                "--dry-run", action="store_true", help="Plan/Scan only"
            )
            _ = mode.add_argument("--apply", action="store_true", help="Apply changes")

        # Add format flag if requested
        if include_format:
            _ = parser.add_argument(
                "--format",
                dest="output_format",
                choices=["json", "text"],
                default="text",
                help="Output format (default: text)",
            )

        # Add check flag if requested
        if include_check:
            _ = parser.add_argument(
                "--check", action="store_true", help="Run in check mode"
            )

        # Add project selection flags if requested
        if include_project:
            _ = parser.add_argument(
                "--project",
                type=str,
                default=None,
                help="Single project to process",
            )
            _ = parser.add_argument(
                "--projects",
                type=str,
                default=None,
                help="Multiple projects (comma-separated or glob pattern)",
            )

        return parser

    @staticmethod
    def resolve(args: Namespace) -> FlextInfraUtilitiesCli.CliArgs:
        """Resolve parsed arguments into a typed CliArgs model.

        Args:
            args: Parsed arguments from ArgumentParser.

        Returns:
            CliArgs instance with resolved values.

        Raises:
            ValidationError: If argument values fail Pydantic validation.

        """
        # Determine apply flag: True if --apply was set, False otherwise
        apply_flag = bool(getattr(args, "apply", False))

        # Get output format, defaulting to "text"
        output_format = getattr(args, "output_format", "text")

        # Get check flag, defaulting to False
        check_flag = bool(getattr(args, "check", False))

        raw_project = getattr(args, "project", None)
        project = raw_project if isinstance(raw_project, str) else None

        raw_projects = getattr(args, "projects", None)
        projects = raw_projects if isinstance(raw_projects, str) else None

        # Resolve workspace path
        workspace_path: Path = args.workspace.resolve()

        return FlextInfraUtilitiesCli.CliArgs(
            workspace=workspace_path,
            apply=apply_flag,
            output_format=output_format,
            check=check_flag,
            project=project,
            projects=projects,
        )

    @staticmethod
    def run_cli(
        main_fn: Callable[[list[str] | None], int],
        argv: list[str] | None = None,
    ) -> int:
        try:
            FlextRuntime.ensure_structlog_configured()
            return main_fn(argv)
        except SystemExit as exc:
            exit_value = exc.code
            if isinstance(exit_value, int):
                return exit_value
            if exit_value is None:
                return 0
            output.error(str(exit_value))
            return 1
        except Exception as exc:
            output.error(str(exc))
            return 1

    @staticmethod
    def exit_code[T](
        result: r[T],
        *,
        failure_msg: str = "operation failed",
    ) -> int:
        if result.is_success:
            return 0
        output.error(result.error or failure_msg)
        return 1

    @staticmethod
    def emit(
        data: BaseModel | Mapping[str, t.Scalar],
        *,
        text_fn: Callable[..., str] | None = None,
        cli: FlextInfraUtilitiesCli.CliArgs,
    ) -> None:
        """Emit formatted data to stdout based on output format.

        Outputs machine-readable payload to stdout. Diagnostic/status messages
        go to stderr via the output module.

        Args:
            data: BaseModel instance or mapping of scalar values to emit.
            text_fn: Optional formatter function for text output. If provided
                and output_format is "text", used to format data. Otherwise,
                str(data) is used.
            cli: CliArgs instance specifying output format and other options.

        """
        if cli.output_format == "json":
            # JSON output: use model_dump_json() for BaseModel, json.dumps() for Mapping
            if isinstance(data, BaseModel):
                output.write(data.model_dump_json())
            else:
                output.write(json.dumps(data))
        elif cli.output_format == "text":
            # Text output: use text_fn if provided, otherwise str(data)
            if text_fn is not None:
                output.write(text_fn(data))
            else:
                output.write(str(data))

    @staticmethod
    def iter_projects(
        cli: FlextInfraUtilitiesCli.CliArgs,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        """Discover and filter projects based on CLI arguments.

        Calls discover_projects() and filters by project_names() if specified.
        Returns sorted list of ProjectInfo wrapped in r[T].

        Args:
            cli: CliArgs instance with workspace and project selection.

        Returns:
            Result containing sorted list of ProjectInfo, or failure if discovery fails.

        """
        # Discover all projects in workspace
        discovery_result = FlextInfraUtilitiesCli._discover_projects_impl(cli.workspace)
        if discovery_result.is_failure:
            return discovery_result

        projects = discovery_result.value
        project_names = cli.project_names()

        # Filter by project names if specified
        if project_names is not None:
            filtered = [p for p in projects if p.name in project_names]
            return r[list[m.Infra.Workspace.ProjectInfo]].ok(
                sorted(filtered, key=lambda p: p.name)
            )

        # Return all projects sorted
        return r[list[m.Infra.Workspace.ProjectInfo]].ok(
            sorted(projects, key=lambda p: p.name)
        )

    @staticmethod
    def _discover_projects_impl(
        workspace: Path,
    ) -> r[list[m.Infra.Workspace.ProjectInfo]]:
        """Internal implementation that delegates to discovery module.

        This method exists to allow iter_projects() to call the discovery
        method without circular imports or tight coupling.

        Args:
            workspace: Workspace root path.

        Returns:
            Result containing list of ProjectInfo or failure message.

        """
        # Import here to avoid circular imports at module level

        return FlextInfraUtilitiesDiscovery.discover_projects(workspace)


__all__ = ["FlextInfraUtilitiesCli"]
