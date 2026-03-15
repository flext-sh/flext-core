from __future__ import annotations

from flext_infra import dm
from flext_tests import tm


class TestFlextInfraDependencyDetectorModels:
    def test_dependency_limits_info_creation(self) -> None:
        info = dm.DependencyLimitsInfo(python_version=None, limits_path="")
        tm.that(info.python_version, eq=None)
        tm.that(info.limits_path, eq="")

    def test_pip_check_report_creation(self) -> None:
        report = dm.PipCheckReport(ok=True, lines=[])
        tm.that(report.ok, eq=True)
        tm.that(report.lines, eq=[])

    def test_workspace_dependency_report_creation(self) -> None:
        report = dm.WorkspaceDependencyReport(workspace="test-workspace", projects={})
        tm.that(report.workspace, eq="test-workspace")
        tm.that(report.projects, eq={})
        tm.that(report.pip_check, eq=None)
        tm.that(report.dependency_limits, eq=None)
