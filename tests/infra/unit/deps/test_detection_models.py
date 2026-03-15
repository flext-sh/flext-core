from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from flext_infra import (
    FlextInfraDependencyDetectionService,
    dm,
    t,
)
from flext_tests import tm


class TestFlextInfraDependencyDetectionModels:
    def test_deptry_issue_groups_creation(self) -> None:
        groups = dm.DeptryIssueGroups()
        assert groups.dep001 == []
        assert groups.dep002 == []
        assert groups.dep003 == []
        assert groups.dep004 == []

    def test_deptry_report_creation(self) -> None:
        report = dm.DeptryReport(
            missing=[], unused=[], transitive=[], dev_in_runtime=[], raw_count=0
        )
        tm.that(report.missing, eq=[])
        tm.that(report.unused, eq=[])
        tm.that(report.transitive, eq=[])
        tm.that(report.dev_in_runtime, eq=[])
        tm.that(report.raw_count, eq=0)

    def test_project_dependency_report_creation(self) -> None:
        deptry = dm.DeptryReport(
            missing=[], unused=[], transitive=[], dev_in_runtime=[], raw_count=0
        )
        report = dm.ProjectDependencyReport(project="test-project", deptry=deptry)
        tm.that(report.project, eq="test-project")
        tm.that(report.deptry, eq=deptry)

    def test_typings_report_creation(self) -> None:
        report = dm.TypingsReport(
            required_packages=[],
            hinted=[],
            missing_modules=[],
            current=[],
            to_add=[],
            to_remove=[],
        )
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
        assert FlextInfraDependencyDetectionService.to_infra_value(None) is None

    def test_string_value(self) -> None:
        assert FlextInfraDependencyDetectionService.to_infra_value("hello") == "hello"

    def test_int_value(self) -> None:
        assert FlextInfraDependencyDetectionService.to_infra_value(42) == 42

    def test_float_value(self) -> None:
        assert FlextInfraDependencyDetectionService.to_infra_value(math.pi) == math.pi

    def test_bool_value(self) -> None:
        assert FlextInfraDependencyDetectionService.to_infra_value(True) is True

    def test_list_of_valid_values(self) -> None:
        assert FlextInfraDependencyDetectionService.to_infra_value(["a", 1, True]) == [
            "a",
            1,
            True,
        ]

    def test_list_with_unconvertible(self) -> None:
        assert (
            FlextInfraDependencyDetectionService.to_infra_value(
                cast("t.Infra.InfraValue", [Path("/tmp")])
            )
            is None
        )

    def test_mapping_value(self) -> None:
        result = FlextInfraDependencyDetectionService.to_infra_value({
            "key": "value",
            "num": 42,
        })
        assert isinstance(result, Mapping)
        assert result == {"key": "value", "num": 42}

    def test_mapping_with_unconvertible(self) -> None:
        assert (
            FlextInfraDependencyDetectionService.to_infra_value(
                cast("t.Infra.InfraValue", {"key": Path("/tmp")})
            )
            is None
        )

    def test_unsupported_type(self) -> None:
        assert (
            FlextInfraDependencyDetectionService.to_infra_value(
                cast("t.Infra.InfraValue", Path("/tmp"))
            )
            is None
        )

    def test_list_with_none_item(self) -> None:
        assert FlextInfraDependencyDetectionService.to_infra_value([None, "a"]) == [
            None,
            "a",
        ]

    def test_mapping_with_none_value(self) -> None:
        result = FlextInfraDependencyDetectionService.to_infra_value({"key": None})
        assert isinstance(result, Mapping)
        assert result == {"key": None}
