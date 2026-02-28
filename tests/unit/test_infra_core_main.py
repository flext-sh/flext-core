"""Tests for flext_infra.core.__main__ CLI entry point.

Tests argument parsing, subcommand routing, and main flow with mocked
services and sys.argv manipulation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from flext_core import r
from flext_infra.core.__main__ import (
    _run_basemk_validate,
    _run_inventory,
    _run_pytest_diag,
    _run_scan,
    _run_skill_validate,
    _run_stub_validate,
    main,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestCoreMainBaseMkValidate:
    """Test basemk-validate subcommand."""

    def test_run_basemk_validate_success(self, tmp_path: Path) -> None:
        """Test _run_basemk_validate succeeds."""
        args = MagicMock()
        args.root = str(tmp_path)
        with patch(
            "flext_infra.core.__main__.FlextInfraBaseMkValidator"
        ) as mock_validator:
            mock_validator_inst = mock_validator.return_value
            mock_report = MagicMock()
            mock_report.summary = "OK"
            mock_report.violations = []
            mock_report.passed = True
            mock_validator_inst.validate.return_value = r[MagicMock].ok(mock_report)
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_basemk_validate(args)
                assert result == 0
                mock_output.info.assert_called_once()

    def test_run_basemk_validate_with_violations(self, tmp_path: Path) -> None:
        """Test _run_basemk_validate reports violations."""
        args = MagicMock()
        args.root = str(tmp_path)
        with patch(
            "flext_infra.core.__main__.FlextInfraBaseMkValidator"
        ) as mock_validator:
            mock_validator_inst = mock_validator.return_value
            mock_report = MagicMock()
            mock_report.summary = "FAILED"
            mock_report.violations = ["violation 1", "violation 2"]
            mock_report.passed = False
            mock_validator_inst.validate.return_value = r[MagicMock].ok(mock_report)
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_basemk_validate(args)
                assert result == 1
                assert mock_output.warning.call_count == 2

    def test_run_basemk_validate_failure(self, tmp_path: Path) -> None:
        """Test _run_basemk_validate handles validation failure."""
        args = MagicMock()
        args.root = str(tmp_path)
        with patch(
            "flext_infra.core.__main__.FlextInfraBaseMkValidator"
        ) as mock_validator:
            mock_validator_inst = mock_validator.return_value
            mock_validator_inst.validate.return_value = r[MagicMock].fail(
                "validation error"
            )
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_basemk_validate(args)
                assert result == 1
                mock_output.error.assert_called_once()


class TestCoreMainInventory:
    """Test inventory subcommand."""

    def test_run_inventory_success(self, tmp_path: Path) -> None:
        """Test _run_inventory succeeds."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.output_dir = None
        with patch(
            "flext_infra.core.__main__.FlextInfraInventoryService"
        ) as mock_service:
            mock_service_inst = mock_service.return_value
            mock_service_inst.generate.return_value = r[dict].ok({
                "reports_written": ["/path/to/report.json"]
            })
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_inventory(args)
                assert result == 0
                mock_output.info.assert_called_once()

    def test_run_inventory_with_output_dir(self, tmp_path: Path) -> None:
        """Test _run_inventory with output directory."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.output_dir = str(tmp_path / "output")
        with patch(
            "flext_infra.core.__main__.FlextInfraInventoryService"
        ) as mock_service:
            mock_service_inst = mock_service.return_value
            mock_service_inst.generate.return_value = r[dict].ok({
                "reports_written": []
            })
            with patch("flext_infra.core.__main__.output"):
                result = _run_inventory(args)
                assert result == 0

    def test_run_inventory_failure(self, tmp_path: Path) -> None:
        """Test _run_inventory handles failure."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.output_dir = None
        with patch(
            "flext_infra.core.__main__.FlextInfraInventoryService"
        ) as mock_service:
            mock_service_inst = mock_service.return_value
            mock_service_inst.generate.return_value = r[dict].fail("generation error")
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_inventory(args)
                assert result == 1
                mock_output.error.assert_called_once()


class TestCoreMainPytestDiag:
    """Test pytest-diag subcommand."""

    def test_run_pytest_diag_success(self, tmp_path: Path) -> None:
        """Test _run_pytest_diag succeeds."""
        junit_path = tmp_path / "junit.xml"
        log_path = tmp_path / "pytest.log"
        junit_path.touch()
        log_path.touch()
        args = MagicMock()
        args.junit = str(junit_path)
        args.log = str(log_path)
        args.failed = None
        args.errors = None
        args.warnings = None
        args.slowest = None
        args.skips = None
        with patch(
            "flext_infra.core.__main__.FlextInfraPytestDiagExtractor"
        ) as mock_extractor:
            mock_extractor_inst = mock_extractor.return_value
            mock_extractor_inst.extract.return_value = r[dict].ok({
                "failed_cases": [],
                "error_traces": [],
                "warning_lines": [],
                "slow_entries": [],
                "skip_cases": [],
            })
            result = _run_pytest_diag(args)
            assert result == 0

    def test_run_pytest_diag_writes_failed_cases(self, tmp_path: Path) -> None:
        """Test _run_pytest_diag writes failed cases."""
        junit_path = tmp_path / "junit.xml"
        log_path = tmp_path / "pytest.log"
        failed_path = tmp_path / "failed.txt"
        junit_path.touch()
        log_path.touch()
        args = MagicMock()
        args.junit = str(junit_path)
        args.log = str(log_path)
        args.failed = str(failed_path)
        args.errors = None
        args.warnings = None
        args.slowest = None
        args.skips = None
        with patch(
            "flext_infra.core.__main__.FlextInfraPytestDiagExtractor"
        ) as mock_extractor:
            mock_extractor_inst = mock_extractor.return_value
            mock_extractor_inst.extract.return_value = r[dict].ok({
                "failed_cases": ["test_a", "test_b"],
                "error_traces": [],
                "warning_lines": [],
                "slow_entries": [],
                "skip_cases": [],
            })
            result = _run_pytest_diag(args)
            assert result == 0
            assert failed_path.exists()

    def test_run_pytest_diag_writes_errors(self, tmp_path: Path) -> None:
        """Test _run_pytest_diag writes error traces."""
        junit_path = tmp_path / "junit.xml"
        log_path = tmp_path / "pytest.log"
        errors_path = tmp_path / "errors.txt"
        junit_path.touch()
        log_path.touch()
        args = MagicMock()
        args.junit = str(junit_path)
        args.log = str(log_path)
        args.failed = None
        args.errors = str(errors_path)
        args.warnings = None
        args.slowest = None
        args.skips = None
        with patch(
            "flext_infra.core.__main__.FlextInfraPytestDiagExtractor"
        ) as mock_extractor:
            mock_extractor_inst = mock_extractor.return_value
            mock_extractor_inst.extract.return_value = r[dict].ok({
                "failed_cases": [],
                "error_traces": ["error 1", "error 2"],
                "warning_lines": [],
                "slow_entries": [],
                "skip_cases": [],
            })
            result = _run_pytest_diag(args)
            assert result == 0
            assert errors_path.exists()

    def test_run_pytest_diag_writes_warnings(self, tmp_path: Path) -> None:
        """Test _run_pytest_diag writes warnings."""
        junit_path = tmp_path / "junit.xml"
        log_path = tmp_path / "pytest.log"
        warnings_path = tmp_path / "warnings.txt"
        junit_path.touch()
        log_path.touch()
        args = MagicMock()
        args.junit = str(junit_path)
        args.log = str(log_path)
        args.failed = None
        args.errors = None
        args.warnings = str(warnings_path)
        args.slowest = None
        args.skips = None
        with patch(
            "flext_infra.core.__main__.FlextInfraPytestDiagExtractor"
        ) as mock_extractor:
            mock_extractor_inst = mock_extractor.return_value
            mock_extractor_inst.extract.return_value = r[dict].ok({
                "failed_cases": [],
                "error_traces": [],
                "warning_lines": ["warning 1"],
                "slow_entries": [],
                "skip_cases": [],
            })
            result = _run_pytest_diag(args)
            assert result == 0
            assert warnings_path.exists()

    def test_run_pytest_diag_writes_slowest(self, tmp_path: Path) -> None:
        """Test _run_pytest_diag writes slowest entries."""
        junit_path = tmp_path / "junit.xml"
        log_path = tmp_path / "pytest.log"
        slowest_path = tmp_path / "slowest.txt"
        junit_path.touch()
        log_path.touch()
        args = MagicMock()
        args.junit = str(junit_path)
        args.log = str(log_path)
        args.failed = None
        args.errors = None
        args.warnings = None
        args.slowest = str(slowest_path)
        args.skips = None
        with patch(
            "flext_infra.core.__main__.FlextInfraPytestDiagExtractor"
        ) as mock_extractor:
            mock_extractor_inst = mock_extractor.return_value
            mock_extractor_inst.extract.return_value = r[dict].ok({
                "failed_cases": [],
                "error_traces": [],
                "warning_lines": [],
                "slow_entries": ["slow_test"],
                "skip_cases": [],
            })
            result = _run_pytest_diag(args)
            assert result == 0
            assert slowest_path.exists()

    def test_run_pytest_diag_writes_skips(self, tmp_path: Path) -> None:
        """Test _run_pytest_diag writes skipped cases."""
        junit_path = tmp_path / "junit.xml"
        log_path = tmp_path / "pytest.log"
        skips_path = tmp_path / "skips.txt"
        junit_path.touch()
        log_path.touch()
        args = MagicMock()
        args.junit = str(junit_path)
        args.log = str(log_path)
        args.failed = None
        args.errors = None
        args.warnings = None
        args.slowest = None
        args.skips = str(skips_path)
        with patch(
            "flext_infra.core.__main__.FlextInfraPytestDiagExtractor"
        ) as mock_extractor:
            mock_extractor_inst = mock_extractor.return_value
            mock_extractor_inst.extract.return_value = r[dict].ok({
                "failed_cases": [],
                "error_traces": [],
                "warning_lines": [],
                "slow_entries": [],
                "skip_cases": ["skip_test"],
            })
            result = _run_pytest_diag(args)
            assert result == 0
            assert skips_path.exists()

    def test_run_pytest_diag_failure(self, tmp_path: Path) -> None:
        """Test _run_pytest_diag handles failure."""
        junit_path = tmp_path / "junit.xml"
        log_path = tmp_path / "pytest.log"
        junit_path.touch()
        log_path.touch()
        args = MagicMock()
        args.junit = str(junit_path)
        args.log = str(log_path)
        args.failed = None
        args.errors = None
        args.warnings = None
        args.slowest = None
        args.skips = None
        with patch(
            "flext_infra.core.__main__.FlextInfraPytestDiagExtractor"
        ) as mock_extractor:
            mock_extractor_inst = mock_extractor.return_value
            mock_extractor_inst.extract.return_value = r[dict].fail("extraction error")
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_pytest_diag(args)
                assert result == 1
                mock_output.error.assert_called_once()


class TestCoreMainScan:
    """Test scan subcommand."""

    def test_run_scan_success_no_violations(self, tmp_path: Path) -> None:
        """Test _run_scan succeeds with no violations."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.pattern = "TODO"
        args.include = ["*.py"]
        args.exclude = []
        args.match = "present"
        with patch(
            "flext_infra.core.__main__.FlextInfraTextPatternScanner"
        ) as mock_scanner:
            mock_scanner_inst = mock_scanner.return_value
            mock_scanner_inst.scan.return_value = r[dict].ok({"violation_count": 0})
            result = _run_scan(args)
            assert result == 0

    def test_run_scan_with_violations(self, tmp_path: Path) -> None:
        """Test _run_scan returns 1 when violations found."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.pattern = "TODO"
        args.include = ["*.py"]
        args.exclude = []
        args.match = "present"
        with patch(
            "flext_infra.core.__main__.FlextInfraTextPatternScanner"
        ) as mock_scanner:
            mock_scanner_inst = mock_scanner.return_value
            mock_scanner_inst.scan.return_value = r[dict].ok({"violation_count": 5})
            result = _run_scan(args)
            assert result == 1

    def test_run_scan_with_excludes(self, tmp_path: Path) -> None:
        """Test _run_scan with exclude patterns."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.pattern = "TODO"
        args.include = ["*.py"]
        args.exclude = ["test_*.py"]
        args.match = "present"
        with patch(
            "flext_infra.core.__main__.FlextInfraTextPatternScanner"
        ) as mock_scanner:
            mock_scanner_inst = mock_scanner.return_value
            mock_scanner_inst.scan.return_value = r[dict].ok({"violation_count": 0})
            result = _run_scan(args)
            assert result == 0

    def test_run_scan_failure(self, tmp_path: Path) -> None:
        """Test _run_scan handles scan failure."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.pattern = "TODO"
        args.include = ["*.py"]
        args.exclude = []
        args.match = "present"
        with patch(
            "flext_infra.core.__main__.FlextInfraTextPatternScanner"
        ) as mock_scanner:
            mock_scanner_inst = mock_scanner.return_value
            mock_scanner_inst.scan.return_value = r[dict].fail("scan error")
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_scan(args)
                assert result == 1
                mock_output.error.assert_called_once()


class TestCoreMainSkillValidate:
    """Test skill-validate subcommand."""

    def test_run_skill_validate_success(self, tmp_path: Path) -> None:
        """Test _run_skill_validate succeeds."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.skill = "test-skill"
        args.mode = "baseline"
        with patch(
            "flext_infra.core.__main__.FlextInfraSkillValidator"
        ) as mock_validator:
            mock_validator_inst = mock_validator.return_value
            mock_report = MagicMock()
            mock_report.summary = "OK"
            mock_report.violations = []
            mock_report.passed = True
            mock_validator_inst.validate.return_value = r[MagicMock].ok(mock_report)
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_skill_validate(args)
                assert result == 0
                mock_output.info.assert_called_once()

    def test_run_skill_validate_with_violations(self, tmp_path: Path) -> None:
        """Test _run_skill_validate reports violations."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.skill = "test-skill"
        args.mode = "strict"
        with patch(
            "flext_infra.core.__main__.FlextInfraSkillValidator"
        ) as mock_validator:
            mock_validator_inst = mock_validator.return_value
            mock_report = MagicMock()
            mock_report.summary = "FAILED"
            mock_report.violations = ["violation 1"]
            mock_report.passed = False
            mock_validator_inst.validate.return_value = r[MagicMock].ok(mock_report)
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_skill_validate(args)
                assert result == 1
                mock_output.warning.assert_called_once()

    def test_run_skill_validate_failure(self, tmp_path: Path) -> None:
        """Test _run_skill_validate handles validation failure."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.skill = "test-skill"
        args.mode = "baseline"
        with patch(
            "flext_infra.core.__main__.FlextInfraSkillValidator"
        ) as mock_validator:
            mock_validator_inst = mock_validator.return_value
            mock_validator_inst.validate.return_value = r[MagicMock].fail(
                "validation error"
            )
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_skill_validate(args)
                assert result == 1
                mock_output.error.assert_called_once()


class TestCoreMainStubValidate:
    """Test stub-validate subcommand."""

    def test_run_stub_validate_success(self, tmp_path: Path) -> None:
        """Test _run_stub_validate succeeds."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.project = None
        with patch("flext_infra.core.__main__.FlextInfraStubSupplyChain") as mock_chain:
            mock_chain_inst = mock_chain.return_value
            mock_report = MagicMock()
            mock_report.summary = "OK"
            mock_report.violations = []
            mock_report.passed = True
            mock_chain_inst.validate.return_value = r[MagicMock].ok(mock_report)
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_stub_validate(args)
                assert result == 0
                mock_output.info.assert_called_once()

    def test_run_stub_validate_with_projects(self, tmp_path: Path) -> None:
        """Test _run_stub_validate with specific projects."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.project = ["proj1", "proj2"]
        with patch("flext_infra.core.__main__.FlextInfraStubSupplyChain") as mock_chain:
            mock_chain_inst = mock_chain.return_value
            mock_report = MagicMock()
            mock_report.summary = "OK"
            mock_report.violations = []
            mock_report.passed = True
            mock_chain_inst.validate.return_value = r[MagicMock].ok(mock_report)
            with patch("flext_infra.core.__main__.output"):
                result = _run_stub_validate(args)
                assert result == 0

    def test_run_stub_validate_with_violations(self, tmp_path: Path) -> None:
        """Test _run_stub_validate reports violations."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.project = None
        with patch("flext_infra.core.__main__.FlextInfraStubSupplyChain") as mock_chain:
            mock_chain_inst = mock_chain.return_value
            mock_report = MagicMock()
            mock_report.summary = "FAILED"
            mock_report.violations = ["violation 1", "violation 2"]
            mock_report.passed = False
            mock_chain_inst.validate.return_value = r[MagicMock].ok(mock_report)
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_stub_validate(args)
                assert result == 1
                assert mock_output.warning.call_count == 2

    def test_run_stub_validate_failure(self, tmp_path: Path) -> None:
        """Test _run_stub_validate handles validation failure."""
        args = MagicMock()
        args.root = str(tmp_path)
        args.project = None
        with patch("flext_infra.core.__main__.FlextInfraStubSupplyChain") as mock_chain:
            mock_chain_inst = mock_chain.return_value
            mock_chain_inst.validate.return_value = r[MagicMock].fail(
                "validation error"
            )
            with patch("flext_infra.core.__main__.output") as mock_output:
                result = _run_stub_validate(args)
                assert result == 1
                mock_output.error.assert_called_once()


class TestCoreMainFlow:
    """Test main() orchestration."""

    def test_main_basemk_validate(self) -> None:
        """Test main() routes to basemk-validate."""
        with patch.object(sys, "argv", ["prog", "basemk-validate", "--root", "."]):
            with patch("flext_infra.core.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.core.__main__._run_basemk_validate",
                    return_value=0,
                ) as mock_handler:
                    result = main()
                    assert result == 0
                    mock_handler.assert_called_once()

    def test_main_inventory(self) -> None:
        """Test main() routes to inventory."""
        with patch.object(sys, "argv", ["prog", "inventory", "--root", "."]):
            with patch("flext_infra.core.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.core.__main__._run_inventory", return_value=0
                ) as mock_handler:
                    result = main()
                    assert result == 0
                    mock_handler.assert_called_once()

    def test_main_pytest_diag(self) -> None:
        """Test main() routes to pytest-diag."""
        with patch.object(
            sys,
            "argv",
            ["prog", "pytest-diag", "--junit", "junit.xml", "--log", "pytest.log"],
        ):
            with patch("flext_infra.core.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.core.__main__._run_pytest_diag", return_value=0
                ) as mock_handler:
                    result = main()
                    assert result == 0
                    mock_handler.assert_called_once()

    def test_main_scan(self) -> None:
        """Test main() routes to scan."""
        with patch.object(
            sys,
            "argv",
            [
                "prog",
                "scan",
                "--root",
                ".",
                "--pattern",
                "TODO",
                "--include",
                "*.py",
            ],
        ):
            with patch("flext_infra.core.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.core.__main__._run_scan", return_value=0
                ) as mock_handler:
                    result = main()
                    assert result == 0
                    mock_handler.assert_called_once()

    def test_main_skill_validate(self) -> None:
        """Test main() routes to skill-validate."""
        with patch.object(
            sys,
            "argv",
            ["prog", "skill-validate", "--skill", "test-skill", "--root", "."],
        ):
            with patch("flext_infra.core.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.core.__main__._run_skill_validate", return_value=0
                ) as mock_handler:
                    result = main()
                    assert result == 0
                    mock_handler.assert_called_once()

    def test_main_stub_validate(self) -> None:
        """Test main() routes to stub-validate."""
        with patch.object(sys, "argv", ["prog", "stub-validate", "--root", "."]):
            with patch("flext_infra.core.__main__.FlextRuntime"):
                with patch(
                    "flext_infra.core.__main__._run_stub_validate", return_value=0
                ) as mock_handler:
                    result = main()
                    assert result == 0
                    mock_handler.assert_called_once()

    def test_main_no_command(self) -> None:
        """Test main() prints help when no command given."""
        with patch.object(sys, "argv", ["prog"]):
            with patch("flext_infra.core.__main__.FlextRuntime"):
                result = main()
                assert result == 1

    def test_main_unknown_command(self) -> None:
        """Test main() handles unknown command."""
        with patch.object(sys, "argv", ["prog", "unknown"]):
            with patch("flext_infra.core.__main__.FlextRuntime"):
                with pytest.raises(SystemExit):
                    main()

    def test_main_entry_point_via_sys_exit(self) -> None:
        """Test __main__ entry point via sys.exit (line 242)."""
        result = subprocess.run(
            ["python", "-m", "flext_infra.core", "--help"],  # noqa: S607
            capture_output=True,
            text=True,
            cwd="/home/marlonsc/flext/flext-core",
            check=False,
        )
        # Should succeed with help output
        assert result.returncode == 0
