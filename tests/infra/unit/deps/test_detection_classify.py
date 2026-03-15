from __future__ import annotations

from flext_infra import FlextInfraDependencyDetectionService, t
from flext_tests import tm


class TestClassifyIssues:
    def test_classify_dep001(self) -> None:
        service = FlextInfraDependencyDetectionService()
        issues: list[t.Infra.IssueMap] = [
            {"error": {"code": "DEP001"}, "module": "foo"}
        ]
        tm.that(len(service.classify_issues(issues).dep001), eq=1)

    def test_classify_dep002(self) -> None:
        service = FlextInfraDependencyDetectionService()
        issues: list[t.Infra.IssueMap] = [
            {"error": {"code": "DEP002"}, "module": "bar"}
        ]
        tm.that(len(service.classify_issues(issues).dep002), eq=1)

    def test_classify_dep003(self) -> None:
        service = FlextInfraDependencyDetectionService()
        issues: list[t.Infra.IssueMap] = [
            {"error": {"code": "DEP003"}, "module": "baz"}
        ]
        tm.that(len(service.classify_issues(issues).dep003), eq=1)

    def test_classify_dep004(self) -> None:
        service = FlextInfraDependencyDetectionService()
        issues: list[t.Infra.IssueMap] = [
            {"error": {"code": "DEP004"}, "module": "qux"}
        ]
        tm.that(len(service.classify_issues(issues).dep004), eq=1)

    def test_non_dict_error_skipped(self) -> None:
        service = FlextInfraDependencyDetectionService()
        issues: list[t.Infra.IssueMap] = [{"error": "not-a-dict", "module": "foo"}]
        tm.that(len(service.classify_issues(issues).dep001), eq=0)

    def test_missing_code_skipped(self) -> None:
        service = FlextInfraDependencyDetectionService()
        issues: list[t.Infra.IssueMap] = [{"error": {"other": "data"}, "module": "foo"}]
        tm.that(len(service.classify_issues(issues).dep001), eq=0)

    def test_unknown_code_skipped(self) -> None:
        service = FlextInfraDependencyDetectionService()
        issues: list[t.Infra.IssueMap] = [
            {"error": {"code": "DEP999"}, "module": "foo"}
        ]
        groups = service.classify_issues(issues)
        assert groups.dep001 == []
        assert groups.dep002 == []
        assert groups.dep003 == []
        assert groups.dep004 == []

    def test_multiple_issues(self) -> None:
        service = FlextInfraDependencyDetectionService()
        issues: list[t.Infra.IssueMap] = [
            {"error": {"code": "DEP001"}, "module": "a"},
            {"error": {"code": "DEP002"}, "module": "b"},
            {"error": {"code": "DEP001"}, "module": "c"},
        ]
        groups = service.classify_issues(issues)
        tm.that(len(groups.dep001), eq=2)
        tm.that(len(groups.dep002), eq=1)

    def test_classify_issues_with_missing_error_field(self) -> None:
        service = FlextInfraDependencyDetectionService()
        issues: list[t.Infra.IssueMap] = [{"module": "foo"}]
        tm.that(len(service.classify_issues(issues).dep001), eq=0)


class TestBuildProjectReport:
    def test_builds_report(self) -> None:
        service = FlextInfraDependencyDetectionService()
        issues: list[t.Infra.IssueMap] = [
            {"error": {"code": "DEP001"}, "module": "foo"},
            {"error": {"code": "DEP002"}, "module": "bar"},
        ]
        report = service.build_project_report("test-project", issues)
        tm.that(report.project, eq="test-project")
        tm.that(report.deptry.raw_count, eq=2)


class TestDetectionUncoveredLines:
    def test_module_to_types_package_with_custom_limits(self) -> None:
        service = FlextInfraDependencyDetectionService()
        inner = FlextInfraDependencyDetectionService.to_infra_value(
            {"custom_module": "types-custom"},
        )
        assert inner is not None
        limits: dict[str, t.Infra.TomlValue] = {
            "typing_libraries": {"module_to_package": inner},
        }
        tm.that(
            service.module_to_types_package("custom_module", limits), eq="types-custom"
        )
