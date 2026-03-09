from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

from flext_infra.deps.detection import (
    FlextInfraDependencyDetectionService,
    _to_infra_value,
    dm,
)
from flext_tests import tm
from tests.infra import h


class TestFlextInfraDependencyDetectionModels:
    def test_deptry_issue_groups_creation(self) -> None:
        groups = dm.DeptryIssueGroups()
        tm.that(groups.dep001, eq=[])
        tm.that(groups.dep002, eq=[])
        tm.that(groups.dep003, eq=[])
        tm.that(groups.dep004, eq=[])

    def test_deptry_report_creation(self) -> None:
        report = dm.DeptryReport()
        tm.that(report.missing, eq=[])
        tm.that(report.unused, eq=[])
        tm.that(report.transitive, eq=[])
        tm.that(report.dev_in_runtime, eq=[])
        tm.that(report.raw_count, eq=0)

    def test_project_dependency_report_creation(self) -> None:
        deptry = dm.DeptryReport()
        report = dm.ProjectDependencyReport(project="test-project", deptry=deptry)
        tm.that(report.project, eq="test-project")
        tm.that(report.deptry, eq=deptry)

    def test_typings_report_creation(self) -> None:
        report = dm.TypingsReport()
        tm.that(report.required_packages, eq=[])
        tm.that(report.hinted, eq=[])
        tm.that(report.missing_modules, eq=[])
        tm.that(report.current, eq=[])
        tm.that(report.to_add, eq=[])
        tm.that(report.to_remove, eq=[])
        tm.that(report.limits_applied, eq=False)
        tm.that(report.python_version, eq=None)


class TestFlextInfraDependencyDetectionService:
    def test_service_initialization(self) -> None:
        service = FlextInfraDependencyDetectionService()
        tm.that(hasattr(service, "runner"), eq=True)

    def test_default_module_to_types_package_mapping(self) -> None:
        service = FlextInfraDependencyDetectionService()
        tm.that("yaml" in service.DEFAULT_MODULE_TO_TYPES_PACKAGE, eq=True)
        tm.that(service.DEFAULT_MODULE_TO_TYPES_PACKAGE["yaml"], eq="types-pyyaml")


class TestToInfraValue:
    def test_none_value(self) -> None:
        tm.that(_to_infra_value(None), eq=None)

    def test_string_value(self) -> None:
        tm.that(_to_infra_value("hello"), eq="hello")

    def test_int_value(self) -> None:
        tm.that(_to_infra_value(42), eq=42)

    def test_float_value(self) -> None:
        tm.that(_to_infra_value(math.pi), eq=math.pi)

    def test_bool_value(self) -> None:
        tm.that(_to_infra_value(True), eq=True)

    def test_list_of_valid_values(self) -> None:
        tm.that(_to_infra_value(["a", 1, True]), eq=["a", 1, True])

    def test_list_with_unconvertible(self) -> None:
        tm.that(_to_infra_value([Path("/tmp")]), eq=None)

    def test_mapping_value(self) -> None:
        result = _to_infra_value({"key": "value", "num": 42})
        tm.that(isinstance(result, Mapping), eq=True)
        tm.that(result, eq={"key": "value", "num": 42})

    def test_mapping_with_unconvertible(self) -> None:
        tm.that(_to_infra_value({"key": Path("/tmp")}), eq=None)

    def test_unsupported_type(self) -> None:
        tm.that(_to_infra_value(Path("/tmp")), eq=None)

    def test_list_with_none_item(self) -> None:
        tm.that(_to_infra_value([None, "a"]), eq=[None, "a"])

    def test_mapping_with_none_value(self) -> None:
        result = _to_infra_value({"key": None})
        tm.that(isinstance(result, Mapping), eq=True)
        tm.that(result, eq={"key": None})


def test_helpers_alias_available() -> None:
    tm.that(h.__name__, eq="FlextInfraTestHelpers")
