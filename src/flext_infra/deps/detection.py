from __future__ import annotations

from pathlib import Path

from scripts.dependencies import dependency_detection as legacy


class DependencyDetector:
    def discover_projects(
        self,
        workspace_root: Path,
        projects_filter: list[str] | None = None,
    ) -> list[Path]:
        return legacy.discover_projects(workspace_root, projects_filter=projects_filter)

    def run_deptry(
        self,
        project_path: Path,
        venv_bin: Path,
        *,
        config_path: Path | None = None,
        json_output_path: Path | None = None,
        extend_exclude: list[str] | None = None,
    ) -> tuple[list[dict[str, object]], int]:
        return legacy.run_deptry(
            project_path,
            venv_bin,
            config_path=config_path,
            json_output_path=json_output_path,
            extend_exclude=extend_exclude,
        )


discover_projects = legacy.discover_projects
run_deptry = legacy.run_deptry
run_pip_check = legacy.run_pip_check
classify_issues = legacy.classify_issues
build_project_report = legacy.build_project_report
load_dependency_limits = legacy.load_dependency_limits
run_mypy_stub_hints = legacy.run_mypy_stub_hints
module_to_types_package = legacy.module_to_types_package
get_current_typings_from_pyproject = legacy.get_current_typings_from_pyproject
get_required_typings = legacy.get_required_typings
