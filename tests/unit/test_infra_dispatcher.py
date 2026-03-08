"""Tests for flext_infra dispatcher CQRS command classes and handlers.

Tests command creation, validation, handler registration, and dispatch operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra.dispatcher import FlextInfraDispatcher


class TestCheckCommand:
    """Tests for CheckCommand."""

    def test_check_command_creation_with_defaults(self) -> None:
        """Test CheckCommand creation with default values."""
        cmd = FlextInfraDispatcher.CheckCommand()
        assert cmd.command_type == "flext_infra.check"
        assert cmd.action == "validate"
        assert cmd.args == []
        assert cmd.message_type == "command"

    def test_check_command_creation_with_custom_values(self) -> None:
        """Test CheckCommand creation with custom values."""
        cmd = FlextInfraDispatcher.CheckCommand(
            action="fix",
            args=["--project", "flext-core"],
        )
        assert cmd.command_type == "flext_infra.check"
        assert cmd.action == "fix"
        assert cmd.args == ["--project", "flext-core"]

    def test_check_command_has_command_id(self) -> None:
        """Test CheckCommand has auto-generated command_id."""
        cmd = FlextInfraDispatcher.CheckCommand()
        assert cmd.command_id.startswith("cmd_")
        assert len(cmd.command_id) > 4


class TestBaseMkCommand:
    """Tests for BaseMkCommand."""

    def test_basemk_command_creation_with_defaults(self) -> None:
        """Test BaseMkCommand creation with default values."""
        cmd = FlextInfraDispatcher.BaseMkCommand()
        assert cmd.command_type == "flext_infra.basemk"
        assert cmd.action == "generate"
        assert cmd.args == []

    def test_basemk_command_creation_with_custom_values(self) -> None:
        """Test BaseMkCommand creation with custom values."""
        cmd = FlextInfraDispatcher.BaseMkCommand(
            action="sync",
            args=["--all"],
        )
        assert cmd.command_type == "flext_infra.basemk"
        assert cmd.action == "sync"
        assert cmd.args == ["--all"]


class TestWorkspaceCommand:
    """Tests for WorkspaceCommand."""

    def test_workspace_command_creation_with_defaults(self) -> None:
        """Test WorkspaceCommand creation with default values."""
        cmd = FlextInfraDispatcher.WorkspaceCommand()
        assert cmd.command_type == "flext_infra.workspace"
        assert cmd.action == "detect"
        assert cmd.args == []

    def test_workspace_command_creation_with_custom_values(self) -> None:
        """Test WorkspaceCommand creation with custom values."""
        cmd = FlextInfraDispatcher.WorkspaceCommand(
            action="migrate",
            args=["--from", "v1", "--to", "v2"],
        )
        assert cmd.command_type == "flext_infra.workspace"
        assert cmd.action == "migrate"
        assert cmd.args == ["--from", "v1", "--to", "v2"]


class TestReleaseCommand:
    """Tests for ReleaseCommand."""

    def test_release_command_creation_with_defaults(self) -> None:
        """Test ReleaseCommand creation with default values."""
        cmd = FlextInfraDispatcher.ReleaseCommand()
        assert cmd.command_type == "flext_infra.release"
        assert cmd.action == "plan"
        assert cmd.args == []

    def test_release_command_creation_with_custom_values(self) -> None:
        """Test ReleaseCommand creation with custom values."""
        cmd = FlextInfraDispatcher.ReleaseCommand(
            action="execute",
            args=["--version", "1.0.0"],
        )
        assert cmd.command_type == "flext_infra.release"
        assert cmd.action == "execute"
        assert cmd.args == ["--version", "1.0.0"]


class TestDocsCommand:
    """Tests for DocsCommand."""

    def test_docs_command_creation_with_defaults(self) -> None:
        """Test DocsCommand creation with default values."""
        cmd = FlextInfraDispatcher.DocsCommand()
        assert cmd.command_type == "flext_infra.docs"
        assert cmd.action == "audit"
        assert cmd.args == []

    def test_docs_command_creation_with_custom_values(self) -> None:
        """Test DocsCommand creation with custom values."""
        cmd = FlextInfraDispatcher.DocsCommand(
            action="generate",
            args=["--format", "markdown"],
        )
        assert cmd.command_type == "flext_infra.docs"
        assert cmd.action == "generate"
        assert cmd.args == ["--format", "markdown"]


class TestGithubCommand:
    """Tests for GithubCommand."""

    def test_github_command_creation_with_defaults(self) -> None:
        """Test GithubCommand creation with default values."""
        cmd = FlextInfraDispatcher.GithubCommand()
        assert cmd.command_type == "flext_infra.github"
        assert cmd.action == "lint"
        assert cmd.args == []

    def test_github_command_creation_with_custom_values(self) -> None:
        """Test GithubCommand creation with custom values."""
        cmd = FlextInfraDispatcher.GithubCommand(
            action="sync",
            args=["--repo", "flext"],
        )
        assert cmd.command_type == "flext_infra.github"
        assert cmd.action == "sync"
        assert cmd.args == ["--repo", "flext"]


class TestCoreCommand:
    """Tests for CoreCommand."""

    def test_core_command_creation_with_defaults(self) -> None:
        """Test CoreCommand creation with default values."""
        cmd = FlextInfraDispatcher.CoreCommand()
        assert cmd.command_type == "flext_infra.core"
        assert cmd.action == "validate"
        assert cmd.args == []

    def test_core_command_creation_with_custom_values(self) -> None:
        """Test CoreCommand creation with custom values."""
        cmd = FlextInfraDispatcher.CoreCommand(
            action="diagnose",
            args=["--verbose"],
        )
        assert cmd.command_type == "flext_infra.core"
        assert cmd.action == "diagnose"
        assert cmd.args == ["--verbose"]


class TestDepsCommand:
    """Tests for DepsCommand."""

    def test_deps_command_creation_with_defaults(self) -> None:
        """Test DepsCommand creation with default values."""
        cmd = FlextInfraDispatcher.DepsCommand()
        assert cmd.command_type == "flext_infra.deps"
        assert cmd.action == "detect"
        assert cmd.args == []

    def test_deps_command_creation_with_custom_values(self) -> None:
        """Test DepsCommand creation with custom values."""
        cmd = FlextInfraDispatcher.DepsCommand(
            action="modernize",
            args=["--python", "3.13"],
        )
        assert cmd.command_type == "flext_infra.deps"
        assert cmd.action == "modernize"
        assert cmd.args == ["--python", "3.13"]


class TestFlextInfraDispatcher:
    """Tests for FlextInfraDispatcher service."""

    def test_dispatcher_execute_returns_bool(self) -> None:
        """Test dispatcher execute returns bool result."""
        dispatcher = FlextInfraDispatcher()
        result = dispatcher.execute()
        assert result.is_success
        assert result.value is True

    def test_build_dispatcher_creates_command_bus(self) -> None:
        """Test build_dispatcher creates a valid command bus."""
        bus = FlextInfraDispatcher.build_dispatcher()
        assert bus is not None

    def test_build_dispatcher_registers_handlers(self) -> None:
        """Test build_dispatcher registers all command handlers."""
        bus = FlextInfraDispatcher.build_dispatcher()
        # Verify dispatcher is functional by checking it's not None
        assert bus is not None
        # Dispatcher should have handlers registered
        assert hasattr(bus, "register_handler")
        assert hasattr(bus, "dispatch")

    def test_command_dispatch_check_command(self) -> None:
        """Test dispatching a CheckCommand."""
        bus = FlextInfraDispatcher.build_dispatcher()
        cmd = FlextInfraDispatcher.CheckCommand(action="validate")
        result = bus.dispatch(cmd)
        assert result.is_success

    def test_command_dispatch_basemk_command(self) -> None:
        """Test dispatching a BaseMkCommand."""
        bus = FlextInfraDispatcher.build_dispatcher()
        cmd = FlextInfraDispatcher.BaseMkCommand(action="generate")
        result = bus.dispatch(cmd)
        assert result.is_success

    def test_command_dispatch_workspace_command(self) -> None:
        """Test dispatching a WorkspaceCommand."""
        bus = FlextInfraDispatcher.build_dispatcher()
        cmd = FlextInfraDispatcher.WorkspaceCommand(action="detect")
        result = bus.dispatch(cmd)
        assert result.is_success

    def test_command_dispatch_release_command(self) -> None:
        """Test dispatching a ReleaseCommand."""
        bus = FlextInfraDispatcher.build_dispatcher()
        cmd = FlextInfraDispatcher.ReleaseCommand(action="plan")
        result = bus.dispatch(cmd)
        assert result.is_success

    def test_command_dispatch_docs_command(self) -> None:
        """Test dispatching a DocsCommand."""
        bus = FlextInfraDispatcher.build_dispatcher()
        cmd = FlextInfraDispatcher.DocsCommand(action="audit")
        result = bus.dispatch(cmd)
        assert result.is_success

    def test_command_dispatch_github_command(self) -> None:
        """Test dispatching a GithubCommand."""
        bus = FlextInfraDispatcher.build_dispatcher()
        cmd = FlextInfraDispatcher.GithubCommand(action="lint")
        result = bus.dispatch(cmd)
        assert result.is_success

    def test_command_dispatch_core_command(self) -> None:
        """Test dispatching a CoreCommand."""
        bus = FlextInfraDispatcher.build_dispatcher()
        cmd = FlextInfraDispatcher.CoreCommand(action="validate")
        result = bus.dispatch(cmd)
        assert result.is_success

    def test_command_dispatch_deps_command(self) -> None:
        """Test dispatching a DepsCommand."""
        bus = FlextInfraDispatcher.build_dispatcher()
        cmd = FlextInfraDispatcher.DepsCommand(action="detect")
        result = bus.dispatch(cmd)
        assert result.is_success


class TestCommandValidation:
    """Tests for command validation."""

    def test_command_type_is_literal(self) -> None:
        """Test command_type is properly typed as Literal."""
        cmd = FlextInfraDispatcher.CheckCommand()
        # Verify command_type is the expected literal value
        assert cmd.command_type == "flext_infra.check"

    def test_all_commands_have_message_type(self) -> None:
        """Test all commands have message_type set to 'command'."""
        commands = [
            FlextInfraDispatcher.CheckCommand(),
            FlextInfraDispatcher.BaseMkCommand(),
            FlextInfraDispatcher.WorkspaceCommand(),
            FlextInfraDispatcher.ReleaseCommand(),
            FlextInfraDispatcher.DocsCommand(),
            FlextInfraDispatcher.GithubCommand(),
            FlextInfraDispatcher.CoreCommand(),
            FlextInfraDispatcher.DepsCommand(),
        ]
        for cmd in commands:
            assert cmd.message_type == "command"

    def test_all_commands_have_command_id(self) -> None:
        """Test all commands have auto-generated command_id."""
        commands = [
            FlextInfraDispatcher.CheckCommand(),
            FlextInfraDispatcher.BaseMkCommand(),
            FlextInfraDispatcher.WorkspaceCommand(),
            FlextInfraDispatcher.ReleaseCommand(),
            FlextInfraDispatcher.DocsCommand(),
            FlextInfraDispatcher.GithubCommand(),
            FlextInfraDispatcher.CoreCommand(),
            FlextInfraDispatcher.DepsCommand(),
        ]
        for cmd in commands:
            assert cmd.command_id.startswith("cmd_")
            assert len(cmd.command_id) > 4

    def test_command_serialization(self) -> None:
        """Test command can be serialized to dict."""
        cmd = FlextInfraDispatcher.CheckCommand(
            action="fix",
            args=["--project", "test"],
        )
        data = cmd.model_dump()
        assert data["command_type"] == "flext_infra.check"
        assert data["action"] == "fix"
        assert data["args"] == ["--project", "test"]

    def test_command_deserialization(self) -> None:
        """Test command can be deserialized from dict."""
        data = {
            "command_type": "flext_infra.check",
            "action": "validate",
            "args": ["--all"],
        }
        cmd = FlextInfraDispatcher.CheckCommand.model_validate(data)
        assert cmd.command_type == "flext_infra.check"
        assert cmd.action == "validate"
        assert cmd.args == ["--all"]
