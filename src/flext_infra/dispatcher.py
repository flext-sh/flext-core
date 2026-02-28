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

        # Create handler functions for each command type
        # Each handler must expose message_type for dispatcher routing

        def check_handler(command: t.GeneralValueType) -> t.GeneralValueType:
            if isinstance(command, cls.CheckCommand):
                return {
                    "command_type": command.command_type,
                    "action": command.action,
                    "args": command.args,
                }
            return {}

        check_handler.message_type = "flext_infra.check"  # type: ignore[attr-defined]

        def basemk_handler(command: t.GeneralValueType) -> t.GeneralValueType:
            if isinstance(command, cls.BaseMkCommand):
                return {
                    "command_type": command.command_type,
                    "action": command.action,
                    "args": command.args,
                }
            return {}

        basemk_handler.message_type = "flext_infra.basemk"  # type: ignore[attr-defined]

        def workspace_handler(command: t.GeneralValueType) -> t.GeneralValueType:
            if isinstance(command, cls.WorkspaceCommand):
                return {
                    "command_type": command.command_type,
                    "action": command.action,
                    "args": command.args,
                }
            return {}

        workspace_handler.message_type = "flext_infra.workspace"  # type: ignore[attr-defined]

        def release_handler(command: t.GeneralValueType) -> t.GeneralValueType:
            if isinstance(command, cls.ReleaseCommand):
                return {
                    "command_type": command.command_type,
                    "action": command.action,
                    "args": command.args,
                }
            return {}

        release_handler.message_type = "flext_infra.release"  # type: ignore[attr-defined]

        def docs_handler(command: t.GeneralValueType) -> t.GeneralValueType:
            if isinstance(command, cls.DocsCommand):
                return {
                    "command_type": command.command_type,
                    "action": command.action,
                    "args": command.args,
                }
            return {}

        docs_handler.message_type = "flext_infra.docs"  # type: ignore[attr-defined]

        def github_handler(command: t.GeneralValueType) -> t.GeneralValueType:
            if isinstance(command, cls.GithubCommand):
                return {
                    "command_type": command.command_type,
                    "action": command.action,
                    "args": command.args,
                }
            return {}

        github_handler.message_type = "flext_infra.github"  # type: ignore[attr-defined]

        def core_handler(command: t.GeneralValueType) -> t.GeneralValueType:
            if isinstance(command, cls.CoreCommand):
                return {
                    "command_type": command.command_type,
                    "action": command.action,
                    "args": command.args,
                }
            return {}

        core_handler.message_type = "flext_infra.core"  # type: ignore[attr-defined]

        def deps_handler(command: t.GeneralValueType) -> t.GeneralValueType:
            if isinstance(command, cls.DepsCommand):
                return {
                    "command_type": command.command_type,
                    "action": command.action,
                    "args": command.args,
                }
            return {}

        deps_handler.message_type = "flext_infra.deps"  # type: ignore[attr-defined]

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
