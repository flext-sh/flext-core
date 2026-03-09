from __future__ import annotations

from flext_infra.deps.detector import ddm
from flext_tests import tm


class TestFlextInfraDependencyDetectorModels:
    def test_dependency_limits_info_creation(self) -> None:
        info = ddm.DependencyLimitsInfo()
        tm.that(info.python_version, eq=None)
        tm.that(info.limits_path, eq="")

    def test_pip_check_report_creation(self) -> None:
        report = ddm.PipCheckReport()
        tm.that(report.ok, eq=True)
        tm.that(report.lines, eq=[])

    def test_workspace_dependency_report_creation(self) -> None:
        report = ddm.WorkspaceDependencyReport(workspace="test-workspace")
        tm.that(report.workspace, eq="test-workspace")
        tm.that(report.projects, eq={})
        tm.that(report.pip_check, eq=None)
        tm.that(report.dependency_limits, eq=None)
