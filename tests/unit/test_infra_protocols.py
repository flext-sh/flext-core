"""Tests for FlextInfraProtocols facade.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_infra import FlextInfraProtocols, p


class TestFlextInfraProtocolsImport:
    """Test FlextInfraProtocols class import and structure."""

    def test_flext_infra_protocols_is_importable(self) -> None:
        """Test that FlextInfraProtocols can be imported."""
        assert FlextInfraProtocols is not None

    def test_flext_infra_protocols_has_project_info_protocol(self) -> None:
        """Test that ProjectInfo is defined."""
        assert hasattr(FlextInfraProtocols.Infra, "ProjectInfo")

    def test_flext_infra_protocols_has_command_output_protocol(self) -> None:
        """Test that CommandOutput is defined."""
        assert hasattr(FlextInfraProtocols.Infra, "CommandOutput")

    def test_flext_infra_protocols_has_checker_protocol(self) -> None:
        """Test that Checker is defined."""
        assert hasattr(FlextInfraProtocols.Infra, "Checker")

    def test_flext_infra_protocols_has_syncer_protocol(self) -> None:
        """Test that Syncer is defined."""
        assert hasattr(FlextInfraProtocols.Infra, "Syncer")

    def test_flext_infra_protocols_has_generator_protocol(self) -> None:
        """Test that Generator is defined."""
        assert hasattr(FlextInfraProtocols.Infra, "Generator")

    def test_flext_infra_protocols_has_reporter_protocol(self) -> None:
        """Test that Reporter is defined."""
        assert hasattr(FlextInfraProtocols.Infra, "Reporter")

    def test_flext_infra_protocols_has_validator_protocol(self) -> None:
        """Test that Validator is defined."""
        assert hasattr(FlextInfraProtocols.Infra, "Validator")

    def test_flext_infra_protocols_has_orchestrator_protocol(self) -> None:
        """Test that Orchestrator is defined."""
        assert hasattr(FlextInfraProtocols.Infra, "Orchestrator")

    def test_flext_infra_protocols_has_discovery_protocol(self) -> None:
        """Test that Discovery is defined."""
        assert hasattr(FlextInfraProtocols.Infra, "Discovery")

    def test_flext_infra_protocols_has_command_runner_protocol(self) -> None:
        """Test that CommandRunner is defined."""
        assert hasattr(FlextInfraProtocols.Infra, "CommandRunner")

    def test_runtime_alias_p_is_flext_infra_protocols(self) -> None:
        """Test that p is an alias for FlextInfraProtocols."""
        assert p is FlextInfraProtocols

    def test_project_info_protocol_has_name_property(self) -> None:
        """Test that ProjectInfo defines name property."""
        proto = FlextInfraProtocols.Infra.ProjectInfo
        assert hasattr(proto, "name")

    def test_project_info_protocol_has_path_property(self) -> None:
        """Test that ProjectInfo defines path property."""
        proto = FlextInfraProtocols.Infra.ProjectInfo
        assert hasattr(proto, "path")

    def test_command_output_protocol_has_stdout_property(self) -> None:
        """Test that CommandOutput defines stdout property."""
        proto = FlextInfraProtocols.Infra.CommandOutput
        assert hasattr(proto, "stdout")

    def test_command_output_protocol_has_stderr_property(self) -> None:
        """Test that CommandOutput defines stderr property."""
        proto = FlextInfraProtocols.Infra.CommandOutput
        assert hasattr(proto, "stderr")

    def test_command_output_protocol_has_exit_code_property(self) -> None:
        """Test that CommandOutput defines exit_code property."""
        proto = FlextInfraProtocols.Infra.CommandOutput
        assert hasattr(proto, "exit_code")

    def test_checker_protocol_has_run_method(self) -> None:
        """Test that Checker defines run method."""
        proto = FlextInfraProtocols.Infra.Checker
        assert hasattr(proto, "run")

    def test_syncer_protocol_has_sync_method(self) -> None:
        """Test that Syncer defines sync method."""
        proto = FlextInfraProtocols.Infra.Syncer
        assert hasattr(proto, "sync")

    def test_generator_protocol_has_generate_method(self) -> None:
        """Test that Generator defines generate method."""
        proto = FlextInfraProtocols.Infra.Generator
        assert hasattr(proto, "generate")

    def test_reporter_protocol_has_report_method(self) -> None:
        """Test that Reporter defines report method."""
        proto = FlextInfraProtocols.Infra.Reporter
        assert hasattr(proto, "report")

    def test_validator_protocol_has_validate_method(self) -> None:
        """Test that Validator defines validate method."""
        proto = FlextInfraProtocols.Infra.Validator
        assert hasattr(proto, "validate")

    def test_orchestrator_protocol_has_orchestrate_method(self) -> None:
        """Test that Orchestrator defines orchestrate method."""
        proto = FlextInfraProtocols.Infra.Orchestrator
        assert hasattr(proto, "orchestrate")

    def test_discovery_protocol_has_discover_projects_method(self) -> None:
        """Test that Discovery defines discover_projects method."""
        proto = FlextInfraProtocols.Infra.Discovery
        assert hasattr(proto, "discover_projects")

    def test_command_runner_protocol_has_run_method(self) -> None:
        """Test that CommandRunner defines run method."""
        proto = FlextInfraProtocols.Infra.CommandRunner
        assert hasattr(proto, "run")
