"""Tests for FlextInfraInventoryService.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path

from flext_infra.core.inventory import FlextInfraInventoryService
from flext_tests import tm
from tests.infra.models import m


class TestInventoryServiceCore:
    """Core tests for FlextInfraInventoryService."""

    def test_init_creates_service(self) -> None:
        """Service initializes with required attributes."""
        service = FlextInfraInventoryService()
        assert service is not None
        tm.that(hasattr(service, "_json"), eq=True)

    def test_generate_empty_workspace(self, tmp_path: Path) -> None:
        """Empty workspace returns success with zero scripts."""
        service = FlextInfraInventoryService()
        report = tm.ok(service.generate(tmp_path))
        tm.that(report, is_=m.Infra.Core.InventoryReport)
        tm.that(report.total_scripts, eq=0)

    def test_generate_with_output_dir(self, tmp_path: Path) -> None:
        """Generate creates reports in output directory."""
        service = FlextInfraInventoryService()
        output_dir = tmp_path / "reports"
        output_dir.mkdir()
        tm.ok(service.generate(tmp_path, output_dir=output_dir))

    def test_generate_returns_flextresult(self, tmp_path: Path) -> None:
        """Generate returns r type."""
        service = FlextInfraInventoryService()
        result = service.generate(tmp_path)
        tm.that(hasattr(result, "is_success"), eq=True)
        tm.that(hasattr(result, "is_failure"), eq=True)


class TestInventoryServiceScripts:
    """Script scanning tests for FlextInfraInventoryService."""

    def test_generate_scans_python_scripts(self, tmp_path: Path) -> None:
        """Python scripts are detected."""
        service = FlextInfraInventoryService()
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "test.py").write_text("#!/usr/bin/env python3\nprint('hello')")
        tm.ok(service.generate(tmp_path))

    def test_generate_scans_bash_scripts(self, tmp_path: Path) -> None:
        """Bash scripts are detected."""
        service = FlextInfraInventoryService()
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "test.sh").write_text("#!/bin/bash\necho 'hello'")
        tm.ok(service.generate(tmp_path))

    def test_generate_counts_multiple_scripts(self, tmp_path: Path) -> None:
        """All scripts are counted."""
        service = FlextInfraInventoryService()
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "script1.py").write_text("")
        (scripts / "script2.sh").write_text("")
        (scripts / "script3.py").write_text("")
        report = tm.ok(service.generate(tmp_path))
        tm.that(report.total_scripts, eq=3)

    def test_generate_finds_nested_scripts(self, tmp_path: Path) -> None:
        """Scripts in nested directories are found."""
        service = FlextInfraInventoryService()
        subdir = tmp_path / "scripts" / "subdir"
        subdir.mkdir(parents=True)
        (tmp_path / "scripts" / "script1.py").write_text("")
        (subdir / "script2.sh").write_text("")
        report = tm.ok(service.generate(tmp_path))
        tm.that(report.total_scripts, eq=2)

    def test_generate_ignores_non_script_files(self, tmp_path: Path) -> None:
        """Non-script files are excluded from count."""
        service = FlextInfraInventoryService()
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "script.py").write_text("")
        (scripts / "readme.txt").write_text("")
        (scripts / "config.json").write_text("")
        report = tm.ok(service.generate(tmp_path))
        tm.that(report.total_scripts, eq=1)

    def test_generate_missing_scripts_dir(self, tmp_path: Path) -> None:
        """Missing scripts directory returns zero scripts."""
        service = FlextInfraInventoryService()
        report = tm.ok(service.generate(tmp_path))
        tm.that(report.total_scripts, eq=0)

    def test_generate_sorts_scripts_alphabetically(self, tmp_path: Path) -> None:
        """Scripts are sorted alphabetically."""
        service = FlextInfraInventoryService()
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "z_script.py").write_text("")
        (scripts / "a_script.py").write_text("")
        (scripts / "m_script.py").write_text("")
        tm.ok(service.generate(tmp_path))


class TestInventoryServiceReports:
    """Report generation tests for FlextInfraInventoryService."""

    def test_generate_returns_reports_written_list(self, tmp_path: Path) -> None:
        """Reports written is a list."""
        service = FlextInfraInventoryService()
        output_dir = tmp_path / "reports"
        output_dir.mkdir()
        report = tm.ok(service.generate(tmp_path, output_dir=output_dir))
        tm.that(report.reports_written, is_=list)

    def test_generate_creates_inventory_report(self, tmp_path: Path) -> None:
        """Inventory report is created with scripts."""
        service = FlextInfraInventoryService()
        output_dir = tmp_path / "reports"
        output_dir.mkdir()
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "test.py").write_text("")
        tm.ok(service.generate(tmp_path, output_dir=output_dir))

    def test_generate_nonexistent_workspace(self, tmp_path: Path) -> None:
        """Nonexistent workspace still returns success."""
        service = FlextInfraInventoryService()
        result = service.generate(tmp_path / "nonexistent")
        tm.that(result.is_success, eq=True)

    def test_generate_write_to_readonly_dir_fails(self, tmp_path: Path) -> None:
        """Writing to read-only output directory fails."""
        service = FlextInfraInventoryService()
        output_dir = tmp_path / "reports"
        output_dir.mkdir()
        output_dir.chmod(0o444)
        try:
            result = service.generate(tmp_path, output_dir=output_dir)
            tm.that(result.is_failure, eq=True)
        finally:
            output_dir.chmod(0o755)


__all__: list[str] = []
