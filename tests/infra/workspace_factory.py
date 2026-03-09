from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class WorkspaceFactory(BaseModel):
    model_config = ConfigDict(frozen=True)

    default_python: str = Field(default="^3.13")
    default_version: str = Field(default="0.1.0")
    encoding: str = Field(default="utf-8")

    def create_minimal(self, tmp_path: Path, name: str = "test-proj") -> Path:
        return self._create_project(tmp_path=tmp_path, name=name, deps=[])

    def create_full(self, tmp_path: Path, name: str) -> Path:
        project_root = self.create_minimal(tmp_path=tmp_path, name=name)
        docs_dir = project_root / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        (project_root / "AGENTS.md").write_text(
            "# AGENTS\n\nProject rules.\n", encoding=self.encoding
        )
        (project_root / "README.md").write_text(
            f"# {name}\n\nGenerated full project fixture.\n",
            encoding=self.encoding,
        )
        (docs_dir / "README.md").write_text(
            "# Docs\n\nDocumentation placeholder.\n",
            encoding=self.encoding,
        )
        return project_root

    def create_with_deps(self, tmp_path: Path, name: str, deps: list[str]) -> Path:
        return self._create_project(tmp_path=tmp_path, name=name, deps=deps)

    def create_workspace(self, tmp_path: Path, projects: int = 3) -> Path:
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir(parents=True, exist_ok=True)
        project_names = [f"test-proj-{idx + 1}" for idx in range(projects)]
        for project_name in project_names:
            self.create_minimal(tmp_path=workspace_root, name=project_name)
        members = ", ".join(f'"{name}"' for name in project_names)
        workspace_pyproject = (
            "[tool.poetry]\n"
            'name = "workspace"\n'
            f'version = "{self.default_version}"\n'
            'description = "Generated workspace fixture"\n'
            'authors = ["FLEXT Tests <tests@flext.dev>"]\n\n'
            "[tool.flext.workspace]\n"
            f"members = [{members}]\n"
        )
        (workspace_root / "pyproject.toml").write_text(
            workspace_pyproject,
            encoding=self.encoding,
        )
        (workspace_root / "Makefile").write_text(
            "check:\n\t@echo workspace-check\n",
            encoding=self.encoding,
        )
        return workspace_root

    def _create_project(self, tmp_path: Path, name: str, deps: list[str]) -> Path:
        project_root = tmp_path / name
        package_dir = project_root / "src" / name.replace("-", "_")
        tests_dir = project_root / "tests"
        package_dir.mkdir(parents=True, exist_ok=True)
        tests_dir.mkdir(parents=True, exist_ok=True)
        (project_root / "pyproject.toml").write_text(
            self._project_pyproject(name=name, deps=deps),
            encoding=self.encoding,
        )
        (project_root / "Makefile").write_text(
            "check:\n\t@echo project-check\n",
            encoding=self.encoding,
        )
        (package_dir / "__init__.py").write_text(
            f'"""{name} package."""\n',
            encoding=self.encoding,
        )
        (tests_dir / "__init__.py").write_text("", encoding=self.encoding)
        return project_root

    def _project_pyproject(self, name: str, deps: list[str]) -> str:
        dependency_lines = [f'python = "{self.default_python}"']
        dependency_lines.extend(f'{dep} = "*"' for dep in deps)
        dependencies = "\n".join(dependency_lines)
        return (
            "[tool.poetry]\n"
            f'name = "{name}"\n'
            f'version = "{self.default_version}"\n'
            'description = "Generated test project"\n'
            'authors = ["FLEXT Tests <tests@flext.dev>"]\n'
            'packages = [{ include = "'
            + name.replace("-", "_")
            + '", from = "src" }]\n\n'
            "[tool.poetry.dependencies]\n"
            f"{dependencies}\n"
        )


__all__ = ["WorkspaceFactory"]
