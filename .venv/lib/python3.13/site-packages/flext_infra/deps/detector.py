"""Runtime vs dev dependency detector CLI with deptry, pip-check, and typing analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, override

from flext_infra import m, p, t, u
from flext_infra.base_selection import FlextInfraProjectSelectionServiceBase
from flext_infra.deps.detection import FlextInfraDependencyDetectionService
from flext_infra.deps.detector_runtime import FlextInfraDependencyDetectorRuntime


class FlextInfraRuntimeDevDependencyDetector(
    FlextInfraProjectSelectionServiceBase[bool]
):
    """CLI tool for detecting runtime vs dev dependencies across workspace."""

    output_format: Annotated[
        str, m.Field(alias="format", description="Output format for dependency report")
    ] = "text"
    output: Annotated[str | None, m.Field(None, description="Optional output path")] = (
        None
    )
    quiet: Annotated[bool, m.Field(False, description="Reduce command output")] = False
    no_fail: Annotated[
        bool, m.Field(alias="no-fail", description="Exit successfully even with issues")
    ] = False
    typings: Annotated[
        bool, m.Field(False, description="Detect required typing packages")
    ] = False
    apply_typings: Annotated[
        bool,
        m.Field(alias="apply-typings", description="Install missing typing packages"),
    ] = False
    no_pip_check: Annotated[
        bool, m.Field(alias="no-pip-check", description="Skip workspace pip check")
    ] = False
    limits: Annotated[
        str | None, m.Field(None, description="Dependency limits TOML")
    ] = None
    deps: Annotated[
        p.Infra.DepsService,
        m.Field(exclude=True, description="Dependency analysis service"),
    ] = m.Field(default_factory=FlextInfraDependencyDetectionService)
    runner: Annotated[
        p.Infra.RunnerService,
        m.Field(exclude=True, description="Command runner for follow-up operations"),
    ] = m.Field(default_factory=lambda: u.Cli)

    @property
    def output_path(self) -> Path | None:
        """Resolved explicit output path when provided."""
        if self.output is None:
            return None
        return Path(self.output).expanduser().resolve()

    @property
    def limits_path(self) -> Path | None:
        """Resolved dependency limits path when provided."""
        if self.limits is None:
            return None
        return Path(self.limits).expanduser().resolve()

    @override
    def execute(self) -> p.Result[bool]:
        """Execute dependency detection and generate workspace report."""
        payload: dict[str, t.Infra.InfraValue] = {
            "workspace": str(self.root),
            "apply": self.apply_changes,
            "format": self.output_format,
            "output": self.output,
            "quiet": self.quiet,
            "no-fail": self.no_fail,
            "typings": self.typings,
            "apply-typings": self.apply_typings,
            "no-pip-check": self.no_pip_check,
            "limits": self.limits,
        }
        if self.selected_projects is not None:
            projects_list: t.JsonValueList = list(self.selected_projects)
            payload["projects"] = projects_list
        params = m.Infra.DetectCommand.model_validate(payload)
        runtime = FlextInfraDependencyDetectorRuntime(
            detector=self,
            workspace_report_factory=m.Infra.WorkspaceDependencyReport,
            dependency_limits_factory=m.Infra.DependencyLimitsInfo,
            pip_check_factory=m.Infra.PipCheckReport,
        )
        return runtime.run(params)


__all__: list[str] = ["FlextInfraRuntimeDevDependencyDetector"]
