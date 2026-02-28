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
        """Test that ProjectInfoProtocol is defined."""
        assert hasattr(FlextInfraProtocols, "ProjectInfoProtocol")

    def test_flext_infra_protocols_has_command_output_protocol(self) -> None:
        """Test that CommandOutputProtocol is defined."""
        assert hasattr(FlextInfraProtocols, "CommandOutputProtocol")

    def test_flext_infra_protocols_has_checker_protocol(self) -> None:
        """Test that CheckerProtocol is defined."""
        assert hasattr(FlextInfraProtocols, "CheckerProtocol")

    def test_flext_infra_protocols_has_syncer_protocol(self) -> None:
        """Test that SyncerProtocol is defined."""
        assert hasattr(FlextInfraProtocols, "SyncerProtocol")

    def test_flext_infra_protocols_has_generator_protocol(self) -> None:
        """Test that GeneratorProtocol is defined."""
        assert hasattr(FlextInfraProtocols, "GeneratorProtocol")

    def test_flext_infra_protocols_has_reporter_protocol(self) -> None:
        """Test that ReporterProtocol is defined."""
        assert hasattr(FlextInfraProtocols, "ReporterProtocol")

    def test_flext_infra_protocols_has_validator_protocol(self) -> None:
        """Test that ValidatorProtocol is defined."""
        assert hasattr(FlextInfraProtocols, "ValidatorProtocol")

    def test_flext_infra_protocols_has_orchestrator_protocol(self) -> None:
        """Test that OrchestratorProtocol is defined."""
        assert hasattr(FlextInfraProtocols, "OrchestratorProtocol")

    def test_flext_infra_protocols_has_discovery_protocol(self) -> None:
        """Test that DiscoveryProtocol is defined."""
        assert hasattr(FlextInfraProtocols, "DiscoveryProtocol")

    def test_flext_infra_protocols_has_command_runner_protocol(self) -> None:
        """Test that CommandRunnerProtocol is defined."""
        assert hasattr(FlextInfraProtocols, "CommandRunnerProtocol")

    def test_runtime_alias_p_is_flext_infra_protocols(self) -> None:
        """Test that p is an alias for FlextInfraProtocols."""
        assert p is FlextInfraProtocols

    def test_project_info_protocol_has_name_property(self) -> None:
        """Test that ProjectInfoProtocol defines name property."""
        proto = FlextInfraProtocols.ProjectInfoProtocol
        assert hasattr(proto, "name")

    def test_project_info_protocol_has_root_property(self) -> None:
        """Test that ProjectInfoProtocol defines root property."""
        proto = FlextInfraProtocols.ProjectInfoProtocol
        assert hasattr(proto, "root")

    def test_command_output_protocol_has_stdout_property(self) -> None:
        """Test that CommandOutputProtocol defines stdout property."""
        proto = FlextInfraProtocols.CommandOutputProtocol
        assert hasattr(proto, "stdout")

    def test_command_output_protocol_has_stderr_property(self) -> None:
        """Test that CommandOutputProtocol defines stderr property."""
        proto = FlextInfraProtocols.CommandOutputProtocol
        assert hasattr(proto, "stderr")

    def test_command_output_protocol_has_returncode_property(self) -> None:
        """Test that CommandOutputProtocol defines returncode property."""
        proto = FlextInfraProtocols.CommandOutputProtocol
        assert hasattr(proto, "returncode")

    def test_checker_protocol_has_run_method(self) -> None:
        """Test that CheckerProtocol defines run method."""
        proto = FlextInfraProtocols.CheckerProtocol
        assert hasattr(proto, "run")

    def test_syncer_protocol_has_sync_method(self) -> None:
        """Test that SyncerProtocol defines sync method."""
        proto = FlextInfraProtocols.SyncerProtocol
        assert hasattr(proto, "sync")

    def test_generator_protocol_has_generate_method(self) -> None:
        """Test that GeneratorProtocol defines generate method."""
        proto = FlextInfraProtocols.GeneratorProtocol
        assert hasattr(proto, "generate")

    def test_reporter_protocol_has_report_method(self) -> None:
        """Test that ReporterProtocol defines report method."""
        proto = FlextInfraProtocols.ReporterProtocol
        assert hasattr(proto, "report")

    def test_validator_protocol_has_validate_method(self) -> None:
        """Test that ValidatorProtocol defines validate method."""
        proto = FlextInfraProtocols.ValidatorProtocol
        assert hasattr(proto, "validate")

    def test_orchestrator_protocol_has_orchestrate_method(self) -> None:
        """Test that OrchestratorProtocol defines orchestrate method."""
        proto = FlextInfraProtocols.OrchestratorProtocol
        assert hasattr(proto, "orchestrate")

    def test_discovery_protocol_has_discover_method(self) -> None:
        """Test that DiscoveryProtocol defines discover method."""
        proto = FlextInfraProtocols.DiscoveryProtocol
        assert hasattr(proto, "discover")

    def test_command_runner_protocol_has_run_method(self) -> None:
        """Test that CommandRunnerProtocol defines run method."""
        proto = FlextInfraProtocols.CommandRunnerProtocol
        assert hasattr(proto, "run")
