"""CLI argument parsing utilities for flext-infra commands.

Centralizes argument parser creation and argument resolution for all
flext_infra CLI commands, providing a consistent interface for workspace,
apply/dry-run, format, check, and project selection flags.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from collections.abc import Mapping
from pathlib import Path

from flext_core.models import FlextModels


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


__all__ = ["FlextInfraUtilitiesCli"]
