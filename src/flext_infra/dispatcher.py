"""CQRS command classes and dispatcher for flext_infra CLI operations.

Provides command definitions for each CLI group (check, basemk, workspace, release,
docs, github, core, deps) and a unified dispatcher for command routing and execution.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Literal, override

from flext_core import FlextDispatcher, FlextService, m, p, r, t
from pydantic import Field


class FlextInfraDispatcher(FlextService[bool]):
    """Unified dispatcher for flext_infra CLI operations with integrated command classes."""

    @override
    def execute(self) -> r[bool]:
        """Execute dispatcher operation - returns success indicator."""
        return r[bool].ok(True)

    class CheckCommand(m.Command):
        """Command to run workspace validation and linting checks."""

        command_type: Literal["flext_infra.check"] = "flext_infra.check"
        action: str = Field(
            default="validate",
            description="Check action (validate, fix, report)",
        )
        args: list[str] = Field(
            default_factory=list,
            description="Additional arguments for check operation",
        )

    class BaseMkCommand(m.Command):
        """Command to generate or update base.mk templates."""

        command_type: Literal["flext_infra.basemk"] = "flext_infra.basemk"
        action: str = Field(
            default="generate",
            description="BaseMk action (generate, validate, sync)",
        )
        args: list[str] = Field(
            default_factory=list,
            description="Additional arguments for basemk operation",
        )

    class WorkspaceCommand(m.Command):
        """Command to manage workspace detection, sync, and orchestration."""

        command_type: Literal["flext_infra.workspace"] = "flext_infra.workspace"
        action: str = Field(
            default="detect",
            description="Workspace action (detect, sync, migrate, orchestrate)",
        )
        args: list[str] = Field(
            default_factory=list,
            description="Additional arguments for workspace operation",
        )

    class ReleaseCommand(m.Command):
        """Command to orchestrate release operations."""

        command_type: Literal["flext_infra.release"] = "flext_infra.release"
        action: str = Field(
            default="plan",
            description="Release action (plan, execute, validate)",
        )
        args: list[str] = Field(
            default_factory=list,
            description="Additional arguments for release operation",
        )

    class DocsCommand(m.Command):
        """Command to audit, generate, and validate documentation."""

        command_type: Literal["flext_infra.docs"] = "flext_infra.docs"
        action: str = Field(
            default="audit",
            description="Docs action (audit, generate, validate, build)",
        )
        args: list[str] = Field(
            default_factory=list,
            description="Additional arguments for docs operation",
        )

    class GithubCommand(m.Command):
        """Command to manage GitHub workflows and PR automation."""

        command_type: Literal["flext_infra.github"] = "flext_infra.github"
        action: str = Field(
            default="lint",
            description="GitHub action (lint, sync, pr-manage)",
        )
        args: list[str] = Field(
            default_factory=list,
            description="Additional arguments for github operation",
        )

    class CoreCommand(m.Command):
        """Command to run infrastructure validators and diagnostics."""

        command_type: Literal["flext_infra.core"] = "flext_infra.core"
        action: str = Field(
            default="validate",
            description="Core action (validate, diagnose, inventory)",
        )
        args: list[str] = Field(
            default_factory=list,
            description="Additional arguments for core operation",
        )

    class DepsCommand(m.Command):
        """Command to detect, sync, and modernize dependencies."""

        command_type: Literal["flext_infra.deps"] = "flext_infra.deps"
        action: str = Field(
            default="detect",
            description="Deps action (detect, sync, modernize)",
        )
        args: list[str] = Field(
            default_factory=list,
            description="Additional arguments for deps operation",
        )

    @classmethod
    def build_dispatcher(cls) -> p.CommandBus:
        """Create a dispatcher instance wired to flext_infra commands.

        Returns:
            p.CommandBus: Configured dispatcher with all command handlers registered.

        """
        dispatcher = FlextDispatcher()

        # Generic handler callable with message_type for dispatcher routing
        class _CommandHandler:
            """Callable handler with message_type attribute for dispatcher routing."""

            def __init__(
                self, message_type_value: str, command_cls: type[m.Command]
            ) -> None:
                self.message_type = message_type_value
                self._command_cls = command_cls

            def __call__(self, command: t.GeneralValueType) -> t.GeneralValueType:
                if isinstance(command, self._command_cls):
                    return {
                        "command_type": command.command_type,
                        "action": command.action,
                        "args": command.args,
                    }
                return {}

        check_handler = _CommandHandler("flext_infra.check", cls.CheckCommand)
        basemk_handler = _CommandHandler("flext_infra.basemk", cls.BaseMkCommand)
        workspace_handler = _CommandHandler(
            "flext_infra.workspace", cls.WorkspaceCommand
        )
        release_handler = _CommandHandler("flext_infra.release", cls.ReleaseCommand)
        docs_handler = _CommandHandler("flext_infra.docs", cls.DocsCommand)
        github_handler = _CommandHandler("flext_infra.github", cls.GithubCommand)
        core_handler = _CommandHandler("flext_infra.core", cls.CoreCommand)
        deps_handler = _CommandHandler("flext_infra.deps", cls.DepsCommand)

        # Register all handlers with the dispatcher
        dispatcher.register_handler(check_handler)
        dispatcher.register_handler(basemk_handler)
        dispatcher.register_handler(workspace_handler)
        dispatcher.register_handler(release_handler)
        dispatcher.register_handler(docs_handler)
        dispatcher.register_handler(github_handler)
        dispatcher.register_handler(core_handler)
        dispatcher.register_handler(deps_handler)

        return dispatcher


__all__ = [
    "FlextInfraDispatcher",
]
