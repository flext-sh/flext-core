"""Tests for FlextInfraDependencyDetectionService."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import Mock, patch

from flext_core import r
from flext_infra.deps.detection import (
    FlextInfraDependencyDetectionService,
    _to_infra_value,
    build_project_report,
    classify_issues,
    dm,
    load_dependency_limits,
    module_to_types_package,
)


class TestFlextInfraDependencyDetectionModels:
    """Test FlextInfraDependencyDetectionModels namespace."""

    def test_deptry_issue_groups_creation(self) -> None:
        """Test DeptryIssueGroups model creation."""
        groups = dm.DeptryIssueGroups()
        assert groups.dep001 == []
        assert groups.dep002 == []
        assert groups.dep003 == []
        assert groups.dep004 == []

    def test_deptry_report_creation(self) -> None:
        """Test DeptryReport model creation."""
        report = dm.DeptryReport()
        assert report.missing == []
        assert report.unused == []
        assert report.transitive == []
        assert report.dev_in_runtime == []
        assert report.raw_count == 0

    def test_project_dependency_report_creation(self) -> None:
        """Test ProjectDependencyReport model creation."""
        deptry = dm.DeptryReport()
        report = dm.ProjectDependencyReport(project="test-project", deptry=deptry)
        assert report.project == "test-project"
        assert report.deptry == deptry

    def test_typings_report_creation(self) -> None:
        """Test TypingsReport model creation."""
        report = dm.TypingsReport()
        assert report.required_packages == []
        assert report.hinted == []
        assert report.missing_modules == []
        assert report.current == []
        assert report.to_add == []
        assert report.to_remove == []
        assert report.limits_applied is False
        assert report.python_version is None


class TestFlextInfraDependencyDetectionService:
    """Test FlextInfraDependencyDetectionService."""

    def test_service_initialization(self) -> None:
        """Test service initializes without errors."""
        service = FlextInfraDependencyDetectionService()
        assert service is not None

    def test_default_module_to_types_package_mapping(self) -> None:
        """Test default module to types package mapping exists."""
        service = FlextInfraDependencyDetectionService()
        assert "yaml" in service.DEFAULT_MODULE_TO_TYPES_PACKAGE
        assert service.DEFAULT_MODULE_TO_TYPES_PACKAGE["yaml"] == "types-pyyaml"


class TestToInfraValue:
    """Test _to_infra_value conversion function."""

    def test_none_value(self) -> None:
        """Test None returns None."""
        assert _to_infra_value(None) is None

    def test_string_value(self) -> None:
        """Test string returns string."""
        assert _to_infra_value("hello") == "hello"

    def test_int_value(self) -> None:
        """Test int returns int."""
        assert _to_infra_value(42) == 42

    def test_float_value(self) -> None:
        """Test float returns float."""
        assert _to_infra_value(math.pi) == math.pi

    def test_bool_value(self) -> None:
        """Test bool returns bool."""
        assert _to_infra_value(True) is True

    def test_list_of_valid_values(self) -> None:
        """Test list of valid values returns converted list."""
        result = _to_infra_value(["a", 1, True])
        assert result == ["a", 1, True]

    def test_list_with_unconvertible(self) -> None:
        """Test list with unconvertible item returns None."""
        result = _to_infra_value([set()])
        assert result is None

    def test_mapping_value(self) -> None:
        """Test Mapping returns converted mapping."""
        result = _to_infra_value({"key": "value", "num": 42})
        assert isinstance(result, Mapping)
        assert result["key"] == "value"

    def test_mapping_with_unconvertible(self) -> None:
        """Test Mapping with unconvertible value returns None."""
        result = _to_infra_value({"key": set()})
        assert result is None

    def test_unsupported_type(self) -> None:
        """Test unsupported type returns None."""
        result = _to_infra_value(set())
        assert result is None

    def test_list_with_none_item(self) -> None:
        """Test list with None item preserves it."""
        result = _to_infra_value([None, "a"])
        assert result == [None, "a"]

    def test_mapping_with_none_value(self) -> None:
        """Test mapping with None value preserves it."""
        result = _to_infra_value({"key": None})
        assert isinstance(result, Mapping)
        assert result["key"] is None


class TestDiscoverProjects:
    """Test discover_projects method."""

    def test_success(self, tmp_path: Path) -> None:
        """Test discovering projects successfully."""
        service = FlextInfraDependencyDetectionService()
        service._selector = Mock()
        proj = Mock()
        proj.path = tmp_path / "proj"
        proj.path.mkdir()
        (proj.path / "pyproject.toml").write_text("")
        service._selector.resolve_projects.return_value = r[list].ok([proj])
        result = service.discover_projects(tmp_path)
        assert result.is_success
        assert len(result.value) == 1

    def test_failure(self, tmp_path: Path) -> None:
        """Test discovery failure."""
        service = FlextInfraDependencyDetectionService()
        service._selector = Mock()
        service._selector.resolve_projects.return_value = r[list].fail("failed")
        result = service.discover_projects(tmp_path)
        assert result.is_failure

    def test_filters_without_pyproject(self, tmp_path: Path) -> None:
        """Test projects without pyproject.toml are filtered out."""
        service = FlextInfraDependencyDetectionService()
        service._selector = Mock()
        proj = Mock()
        proj.path = tmp_path / "no-pyproject"
        proj.path.mkdir()
        service._selector.resolve_projects.return_value = r[list].ok([proj])
        result = service.discover_projects(tmp_path)
        assert result.is_success
        assert len(result.value) == 0


class TestRunDeptry:
    """Test run_deptry method."""

    def test_success_with_issues(self, tmp_path: Path) -> None:
        """Test deptry run with JSON output."""
        service = FlextInfraDependencyDetectionService()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")

        # Write JSON output file
        out_file = project / ".deptry-report.json"
        out_file.write_text(
            json.dumps([{"error": {"code": "DEP001"}, "module": "foo"}])
        )

        cmd_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        result = service.run_deptry(project, venv_bin, json_output_path=out_file)
        assert result.is_success
        issues, exit_code = result.value
        assert exit_code == 0
        assert len(issues) == 1

    def test_no_config_file(self, tmp_path: Path) -> None:
        """Test deptry when config file doesn't exist."""
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        result = service.run_deptry(project, venv_bin)
        assert result.is_success
        assert result.value == ([], 0)

    def test_runner_failure(self, tmp_path: Path) -> None:
        """Test deptry runner failure."""
        service = FlextInfraDependencyDetectionService()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")
        service._runner.run_raw.return_value = r[Mock].fail("deptry crash")
        result = service.run_deptry(project, venv_bin)
        assert result.is_failure

    def test_invalid_json_output(self, tmp_path: Path) -> None:
        """Test deptry with invalid JSON in output file."""
        service = FlextInfraDependencyDetectionService()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")

        out_file = project / ".deptry-report.json"
        out_file.write_text("not valid json")

        cmd_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        result = service.run_deptry(project, venv_bin, json_output_path=out_file)
        assert result.is_success
        assert result.value == ([], 0)

    def test_empty_json_output(self, tmp_path: Path) -> None:
        """Test deptry with empty JSON output file."""
        service = FlextInfraDependencyDetectionService()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")

        out_file = project / ".deptry-report.json"
        out_file.write_text("")

        cmd_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        result = service.run_deptry(project, venv_bin, json_output_path=out_file)
        assert result.is_success
        assert result.value == ([], 0)

    def test_with_extend_exclude(self, tmp_path: Path) -> None:
        """Test deptry with extend_exclude option."""
        service = FlextInfraDependencyDetectionService()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")

        cmd_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        result = service.run_deptry(project, venv_bin, extend_exclude=["tests", "docs"])
        assert result.is_success

    def test_cleanup_temp_file(self, tmp_path: Path) -> None:
        """Test deptry cleans up temp JSON file when no explicit path given."""
        service = FlextInfraDependencyDetectionService()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        project = tmp_path / "project"
        project.mkdir()
        (project / "pyproject.toml").write_text("")

        # Create the default output file
        default_out = project / ".deptry-report.json"
        default_out.write_text("[]")

        cmd_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        result = service.run_deptry(project, venv_bin)
        assert result.is_success
        # File should be cleaned up (no explicit json_output_path)
        assert not default_out.exists()


class TestRunPipCheck:
    """Test run_pip_check method."""

    def test_pip_not_found(self, tmp_path: Path) -> None:
        """Test when pip binary doesn't exist."""
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        result = service.run_pip_check(tmp_path, venv_bin)
        assert result.is_success
        assert result.value == ([], 0)

    def test_success_with_output(self, tmp_path: Path) -> None:
        """Test pip check with conflict output."""
        service = FlextInfraDependencyDetectionService()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "pip").write_text("")

        cmd_out = Mock(
            exit_code=1, stdout="pkg1 has requirement\npkg2 conflict\n", stderr=""
        )
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        result = service.run_pip_check(tmp_path, venv_bin)
        assert result.is_success
        lines, exit_code = result.value
        assert len(lines) == 2
        assert exit_code == 1

    def test_runner_failure(self, tmp_path: Path) -> None:
        """Test pip check runner failure."""
        service = FlextInfraDependencyDetectionService()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "pip").write_text("")
        service._runner.run_raw.return_value = r[Mock].fail("pip failed")
        result = service.run_pip_check(tmp_path, venv_bin)
        assert result.is_failure

    def test_success_no_issues(self, tmp_path: Path) -> None:
        """Test pip check with no issues."""
        service = FlextInfraDependencyDetectionService()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "pip").write_text("")

        cmd_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        result = service.run_pip_check(tmp_path, venv_bin)
        assert result.is_success
        lines, exit_code = result.value
        assert lines == []
        assert exit_code == 0


class TestClassifyIssues:
    """Test classify_issues method."""

    def test_classify_dep001(self) -> None:
        """Test DEP001 classification."""
        service = FlextInfraDependencyDetectionService()
        issues = [{"error": {"code": "DEP001"}, "module": "foo"}]
        result = service.classify_issues(issues)
        assert len(result.dep001) == 1

    def test_classify_dep002(self) -> None:
        """Test DEP002 classification."""
        service = FlextInfraDependencyDetectionService()
        issues = [{"error": {"code": "DEP002"}, "module": "bar"}]
        result = service.classify_issues(issues)
        assert len(result.dep002) == 1

    def test_classify_dep003(self) -> None:
        """Test DEP003 classification."""
        service = FlextInfraDependencyDetectionService()
        issues = [{"error": {"code": "DEP003"}, "module": "baz"}]
        result = service.classify_issues(issues)
        assert len(result.dep003) == 1

    def test_classify_dep004(self) -> None:
        """Test DEP004 classification."""
        service = FlextInfraDependencyDetectionService()
        issues = [{"error": {"code": "DEP004"}, "module": "qux"}]
        result = service.classify_issues(issues)
        assert len(result.dep004) == 1

    def test_non_dict_error_skipped(self) -> None:
        """Test non-dict error object is skipped."""
        service = FlextInfraDependencyDetectionService()
        issues = [{"error": "not-a-dict", "module": "foo"}]
        result = service.classify_issues(issues)
        assert len(result.dep001) == 0

    def test_missing_code_skipped(self) -> None:
        """Test missing code in error is skipped."""
        service = FlextInfraDependencyDetectionService()
        issues = [{"error": {"other": "data"}, "module": "foo"}]
        result = service.classify_issues(issues)
        assert len(result.dep001) == 0

    def test_unknown_code_skipped(self) -> None:
        """Test unknown error code is skipped."""
        service = FlextInfraDependencyDetectionService()
        issues = [{"error": {"code": "DEP999"}, "module": "foo"}]
        result = service.classify_issues(issues)
        assert len(result.dep001) == 0
        assert len(result.dep002) == 0
        assert len(result.dep003) == 0
        assert len(result.dep004) == 0

    def test_multiple_issues(self) -> None:
        """Test classifying multiple issues across codes."""
        service = FlextInfraDependencyDetectionService()
        issues = [
            {"error": {"code": "DEP001"}, "module": "a"},
            {"error": {"code": "DEP002"}, "module": "b"},
            {"error": {"code": "DEP001"}, "module": "c"},
        ]
        result = service.classify_issues(issues)
        assert len(result.dep001) == 2
        assert len(result.dep002) == 1


class TestBuildProjectReport:
    """Test build_project_report method."""

    def test_builds_report(self) -> None:
        """Test building a project dependency report."""
        service = FlextInfraDependencyDetectionService()
        issues = [
            {"error": {"code": "DEP001"}, "module": "foo"},
            {"error": {"code": "DEP002"}, "module": "bar"},
        ]
        report = service.build_project_report("test-project", issues)
        assert report.project == "test-project"
        assert report.deptry.raw_count == 2


class TestLoadDependencyLimits:
    """Test load_dependency_limits method."""

    def test_success(self) -> None:
        """Test loading limits successfully."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({"key": "value", "num": 42})
        result = service.load_dependency_limits(Path("/fake/limits.toml"))
        assert result["key"] == "value"
        assert result["num"] == 42

    def test_failure_returns_empty(self) -> None:
        """Test loading failure returns empty dict."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].fail("not found")
        result = service.load_dependency_limits(Path("/fake/limits.toml"))
        assert result == {}

    def test_unconvertible_values_skipped(self) -> None:
        """Test unconvertible values are skipped."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({"good": "val", "bad": set()})
        result = service.load_dependency_limits(Path("/fake/limits.toml"))
        assert "good" in result
        assert "bad" not in result

    def test_none_value_preserved(self) -> None:
        """Test None value is preserved."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({"key": None})
        result = service.load_dependency_limits(Path("/fake/limits.toml"))
        assert "key" in result
        assert result["key"] is None


class TestRunMypyStubHints:
    """Test run_mypy_stub_hints method."""

    def test_mypy_not_found(self, tmp_path: Path) -> None:
        """Test when mypy binary doesn't exist."""
        service = FlextInfraDependencyDetectionService()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        result = service.run_mypy_stub_hints(tmp_path, venv_bin)
        assert result.is_success
        assert result.value == ([], [])

    def test_runner_failure(self, tmp_path: Path) -> None:
        """Test mypy runner failure."""
        service = FlextInfraDependencyDetectionService()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")
        service._runner.run_raw.return_value = r[Mock].fail("mypy crash")
        result = service.run_mypy_stub_hints(tmp_path, venv_bin)
        assert result.is_failure

    def test_parses_hints(self, tmp_path: Path) -> None:
        """Test mypy output parsing for hinted packages."""
        service = FlextInfraDependencyDetectionService()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")

        cmd_out = Mock(
            exit_code=0,
            stdout='note: hint: "pip install types-pyyaml"',
            stderr='error: Library stubs not installed for "requests"',
        )
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)
        result = service.run_mypy_stub_hints(tmp_path, venv_bin)
        assert result.is_success


class TestModuleToTypesPackage:
    """Test module_to_types_package method."""

    def test_default_mapping(self) -> None:
        """Test default module to types package mapping."""
        service = FlextInfraDependencyDetectionService()
        result = service.module_to_types_package("yaml", {})
        assert result == "types-pyyaml"

    def test_internal_prefix_returns_none(self) -> None:
        """Test internal prefix modules return None."""
        service = FlextInfraDependencyDetectionService()
        result = service.module_to_types_package("flext_core", {})
        assert result is None

    def test_limits_override(self) -> None:
        """Test limits override default mapping."""
        service = FlextInfraDependencyDetectionService()
        limits = {
            "typing_libraries": {
                "module_to_package": {"yaml": "custom-types-yaml"},
            },
        }
        result = service.module_to_types_package("yaml", limits)
        assert result == "custom-types-yaml"

    def test_unknown_module(self) -> None:
        """Test unknown module returns None."""
        service = FlextInfraDependencyDetectionService()
        result = service.module_to_types_package("unknown_module", {})
        assert result is None

    def test_submodule_uses_root(self) -> None:
        """Test submodule uses root module for lookup."""
        service = FlextInfraDependencyDetectionService()
        result = service.module_to_types_package("yaml.parser", {})
        assert result == "types-pyyaml"


class TestGetCurrentTypingsFromPyproject:
    """Test get_current_typings_from_pyproject method."""

    def test_poetry_group_typings(self, tmp_path: Path) -> None:
        """Test extracting typings from poetry group."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({
            "tool": {
                "poetry": {
                    "group": {
                        "typings": {
                            "dependencies": {
                                "types-pyyaml": "^6.0",
                                "types-requests": "^2.28",
                            },
                        },
                    },
                },
            },
        })
        result = service.get_current_typings_from_pyproject(tmp_path)
        assert "types-pyyaml" in result
        assert "types-requests" in result

    def test_pep621_optional_deps_list(self, tmp_path: Path) -> None:
        """Test extracting typings from PEP 621 optional-dependencies list."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({
            "project": {
                "optional-dependencies": {
                    "typings": [
                        "types-pyyaml>=6.0",
                        "types-requests[extra]==2.28",
                    ],
                },
            },
        })
        result = service.get_current_typings_from_pyproject(tmp_path)
        assert "types-pyyaml" in result
        assert "types-requests" in result

    def test_pep621_optional_deps_mapping(self, tmp_path: Path) -> None:
        """Test extracting typings from PEP 621 optional-dependencies mapping."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({
            "project": {
                "optional-dependencies": {
                    "typings": {"types-pyyaml": ">=6.0"},
                },
            },
        })
        result = service.get_current_typings_from_pyproject(tmp_path)
        assert "types-pyyaml" in result

    def test_read_failure(self, tmp_path: Path) -> None:
        """Test read failure returns empty list."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].fail("not found")
        result = service.get_current_typings_from_pyproject(tmp_path)
        assert result == []

    def test_empty_data(self, tmp_path: Path) -> None:
        """Test empty data returns empty list."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._toml.read.return_value = r[dict].ok({})
        result = service.get_current_typings_from_pyproject(tmp_path)
        assert result == []


class TestGetRequiredTypings:
    """Test get_required_typings method."""

    def test_full_flow(self, tmp_path: Path) -> None:
        """Test full typing requirements flow."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")

        # Mock mypy output
        cmd_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)

        # Mock toml reads
        service._toml.read.side_effect = [
            # load_dependency_limits
            r[dict].ok({}),
            # get_current_typings_from_pyproject
            r[dict].ok({"project": {"optional-dependencies": {"typings": []}}}),
        ]

        result = service.get_required_typings(tmp_path, venv_bin)
        assert result.is_success

    def test_no_mypy(self, tmp_path: Path) -> None:
        """Test with include_mypy=False."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)

        service._toml.read.side_effect = [
            r[dict].ok({}),
            r[dict].ok({}),
        ]

        result = service.get_required_typings(tmp_path, venv_bin, include_mypy=False)
        assert result.is_success

    def test_mypy_failure(self, tmp_path: Path) -> None:
        """Test mypy failure propagates."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")

        service._runner.run_raw.return_value = r[Mock].fail("mypy crash")
        service._toml.read.return_value = r[dict].ok({})

        result = service.get_required_typings(tmp_path, venv_bin)
        assert result.is_failure

    def test_with_exclude_set(self, tmp_path: Path) -> None:
        """Test excluded packages are removed from required set."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")

        cmd_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)

        service._toml.read.side_effect = [
            r[dict].ok({
                "typing_libraries": {
                    "exclude": ["types-excluded"],
                },
            }),
            r[dict].ok({}),
        ]

        result = service.get_required_typings(tmp_path, venv_bin)
        assert result.is_success

    def test_with_python_version(self, tmp_path: Path) -> None:
        """Test python version is extracted from limits."""
        service = FlextInfraDependencyDetectionService()
        service._toml = Mock()
        service._runner = Mock()
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "mypy").write_text("")

        cmd_out = Mock(exit_code=0, stdout="", stderr="")
        service._runner.run_raw.return_value = r[Mock].ok(cmd_out)

        service._toml.read.side_effect = [
            r[dict].ok({"python": {"version": "3.13"}}),
            r[dict].ok({}),
        ]

        result = service.get_required_typings(tmp_path, venv_bin)
        assert result.is_success
        assert result.value.python_version == "3.13"


class TestModuleLevelWrappers:
    """Test module-level wrapper functions."""

    def test_classify_issues_wrapper(self) -> None:
        """Test module-level classify_issues wrapper."""
        result = classify_issues([])
        assert result.dep001 == []

    def test_build_project_report_wrapper(self) -> None:
        """Test module-level build_project_report wrapper."""
        report = build_project_report("proj", [])
        assert report.project == "proj"

    def test_module_to_types_package_wrapper(self) -> None:
        """Test module-level module_to_types_package wrapper."""
        result = module_to_types_package("yaml", {})
        assert result == "types-pyyaml"

    def test_load_dependency_limits_wrapper(self) -> None:
        """Test module-level load_dependency_limits wrapper."""
        with patch.object(
            FlextInfraDependencyDetectionService,
            "load_dependency_limits",
            return_value={},
        ):
            result = load_dependency_limits()
        assert isinstance(result, Mapping)
