"""Tests for FlextInfraInventoryService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

from flext_infra.core.inventory import FlextInfraInventoryService


class TestFlextInfraInventoryService:
    """Test suite for FlextInfraInventoryService."""

    def test_init_creates_service_instance(self) -> None:
        """Test that InventoryService initializes correctly."""
        service = FlextInfraInventoryService()
        assert service is not None
        assert hasattr(service, "_json")

    def test_generate_with_empty_workspace_returns_success(
        self, tmp_path: Path
    ) -> None:
        """Test that generate returns success for empty workspace."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path

        result = service.generate(workspace_root)
        assert result.is_success
        assert isinstance(result.value, dict)

    def test_generate_with_output_dir_creates_reports(self, tmp_path: Path) -> None:
        """Test that generate creates reports in output directory."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        result = service.generate(workspace_root, output_dir=output_dir)
        assert result.is_success

    def test_generate_returns_flextresult(self, tmp_path: Path) -> None:
        """Test that generate returns FlextResult type."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path

        result = service.generate(workspace_root)
        assert hasattr(result, "is_success")
        assert hasattr(result, "is_failure")

    def test_generate_with_python_scripts_scans_correctly(self, tmp_path: Path) -> None:
        """Test that generate scans Python scripts."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path

        script_dir = workspace_root / "scripts"
        script_dir.mkdir()
        script_file = script_dir / "test.py"
        script_file.write_text("#!/usr/bin/env python3\nprint('hello')")

        result = service.generate(workspace_root)
        assert result.is_success

    def test_generate_with_bash_scripts_scans_correctly(self, tmp_path: Path) -> None:
        """Test that generate scans Bash scripts."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path

        script_dir = workspace_root / "scripts"
        script_dir.mkdir()
        script_file = script_dir / "test.sh"
        script_file.write_text("#!/bin/bash\necho 'hello'")

        result = service.generate(workspace_root)
        assert result.is_success

    def test_generate_with_multiple_scripts_counts_all(self, tmp_path: Path) -> None:
        """Test that generate counts all scripts."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path

        script_dir = workspace_root / "scripts"
        script_dir.mkdir()
        (script_dir / "script1.py").write_text("")
        (script_dir / "script2.sh").write_text("")
        (script_dir / "script3.py").write_text("")

        result = service.generate(workspace_root)
        assert result.is_success
        assert result.value["total_scripts"] == 3

    def test_generate_with_nested_scripts_finds_all(self, tmp_path: Path) -> None:
        """Test that generate finds scripts in nested directories."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path

        script_dir = workspace_root / "scripts"
        subdir = script_dir / "subdir"
        subdir.mkdir(parents=True)
        (script_dir / "script1.py").write_text("")
        (subdir / "script2.sh").write_text("")

        result = service.generate(workspace_root)
        assert result.is_success
        assert result.value["total_scripts"] == 2

    def test_generate_ignores_non_script_files(self, tmp_path: Path) -> None:
        """Test that generate ignores non-script files."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path

        script_dir = workspace_root / "scripts"
        script_dir.mkdir()
        (script_dir / "script.py").write_text("")
        (script_dir / "readme.txt").write_text("")
        (script_dir / "config.json").write_text("")

        result = service.generate(workspace_root)
        assert result.is_success
        assert result.value["total_scripts"] == 1

    def test_generate_with_missing_scripts_dir_returns_success(
        self, tmp_path: Path
    ) -> None:
        """Test that generate handles missing scripts directory."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path

        result = service.generate(workspace_root)
        assert result.is_success
        assert result.value["total_scripts"] == 0

    def test_generate_returns_reports_written_list(self, tmp_path: Path) -> None:
        """Test that generate returns list of written reports."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        result = service.generate(workspace_root, output_dir=output_dir)
        assert result.is_success
        assert "reports_written" in result.value
        assert isinstance(result.value["reports_written"], list)

    def test_generate_with_json_write_failure_returns_failure(
        self, tmp_path: Path
    ) -> None:
        """Test that generate handles JSON write failures."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        with patch.object(
            type(service._json),
            "write",
            return_value=Mock(is_failure=True, error="write failed"),
        ):
            result = service.generate(workspace_root, output_dir=output_dir)
            assert result.is_failure

    def test_generate_with_exception_returns_failure(self, tmp_path: Path) -> None:
        """Test that generate handles exceptions."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path / "nonexistent"

        result = service.generate(workspace_root)
        assert result.is_success

    def test_generate_creates_inventory_report(self, tmp_path: Path) -> None:
        """Test that generate creates inventory report."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        script_dir = workspace_root / "scripts"
        script_dir.mkdir()
        (script_dir / "test.py").write_text("")

        result = service.generate(workspace_root, output_dir=output_dir)
        assert result.is_success

    def test_generate_creates_wiring_report(self, tmp_path: Path) -> None:
        """Test that generate creates wiring report."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        result = service.generate(workspace_root, output_dir=output_dir)
        assert result.is_success

    def test_generate_creates_external_candidates_report(self, tmp_path: Path) -> None:
        """Test that generate creates external candidates report."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        result = service.generate(workspace_root, output_dir=output_dir)
        assert result.is_success

    def test_generate_uses_default_reports_dir(self, tmp_path: Path) -> None:
        """Test that generate uses default .reports directory."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path

        with patch.object(
            type(service._json), "write", return_value=Mock(is_failure=False)
        ):
            result = service.generate(workspace_root)
            assert result.is_success

    def test_generate_includes_generated_at_timestamp(self, tmp_path: Path) -> None:
        """Test that generate includes generated_at timestamp."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        with patch.object(
            type(service._json), "write", return_value=Mock(is_failure=False)
        ):
            result = service.generate(workspace_root, output_dir=output_dir)
            assert result.is_success

    def test_generate_includes_repo_root_path(self, tmp_path: Path) -> None:
        """Test that generate includes repo_root path."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path
        output_dir = tmp_path / "reports"
        output_dir.mkdir()

        with patch.object(
            type(service._json), "write", return_value=Mock(is_failure=False)
        ):
            result = service.generate(workspace_root, output_dir=output_dir)
            assert result.is_success

    def test_generate_sorts_scripts_alphabetically(self, tmp_path: Path) -> None:
        """Test that generate sorts scripts alphabetically."""
        service = FlextInfraInventoryService()
        workspace_root = tmp_path

        script_dir = workspace_root / "scripts"
        script_dir.mkdir()
        (script_dir / "z_script.py").write_text("")
        (script_dir / "a_script.py").write_text("")
        (script_dir / "m_script.py").write_text("")

        result = service.generate(workspace_root)
        assert result.is_success
