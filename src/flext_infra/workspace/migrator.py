"""Migrate projects to unified FLEXT infrastructure."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import override

import tomlkit
from pydantic import BaseModel, JsonValue, TypeAdapter
from tomlkit.items import Item, Table
from tomlkit.toml_document import TOMLDocument

from flext_core import r, s, t
from flext_infra import c, m, p, u
from flext_infra.basemk.generator import FlextInfraBaseMkGenerator

_OBJECT_LIST_ADAPTER: TypeAdapter[list[JsonValue]] = TypeAdapter(list[JsonValue])


class FlextInfraProjectMigrator(s):
    """Migrate projects to standardized base.mk, Makefile, and pyproject structure."""

    def __init__(
        self,
        *,
        discovery: p.Infra.Discovery | None = None,
        generator: FlextInfraBaseMkGenerator | None = None,
    ) -> None:
        """Initialize migrator with optional custom discovery and generator services."""
        super().__init__(
            config_type=None,
            config_overrides=None,
            initial_context=None,
            subproject=None,
            services=None,
            factories=None,
            resources=None,
            container_overrides=None,
            wire_modules=None,
            wire_packages=None,
            wire_classes=None,
        )
        self._discovery = discovery
        self._generator = generator or FlextInfraBaseMkGenerator()

    @staticmethod
    def _action_text(action: str, *, dry_run: bool) -> str:
        return f"[DRY-RUN] {action}" if dry_run else action

    @staticmethod
    def _append_result(result: r[str], changes: list[str], errors: list[str]) -> None:
        if result.is_failure:
            errors.append(result.error or "migration action failed")
            return
        val: str = result.value
        if val:
            changes.append(val)

    @staticmethod
    def _ensure_table(document: tomlkit.TOMLDocument, key: str) -> Table:
        current = FlextInfraProjectMigrator._toml_get(document, key)
        if isinstance(current, Table):
            return current
        created = tomlkit.table()
        document[key] = created
        return created

    @staticmethod
    def _toml_get(
        container: TOMLDocument | Table,
        key: str,
    ) -> Item | Table | object | None:
        if key not in container:
            return None
        return container[key]

    @staticmethod
    def _has_flext_core_dependency(document: tomlkit.TOMLDocument) -> bool:
        project = FlextInfraProjectMigrator._toml_get(document, c.Infra.Toml.PROJECT)
        if isinstance(project, Table):
            deps = FlextInfraProjectMigrator._toml_get(
                project,
                c.Infra.Toml.DEPENDENCIES,
            )
            if isinstance(deps, list):
                deps_list: list[JsonValue] = _OBJECT_LIST_ADAPTER.validate_python(deps)
                for dep_raw in deps_list:
                    dep: str = str(dep_raw)
                    if str(dep).strip().startswith(c.Infra.Packages.CORE):
                        return True
        tool = FlextInfraProjectMigrator._toml_get(document, c.Infra.Toml.TOOL)
        if not isinstance(tool, Table):
            return False
        poetry = FlextInfraProjectMigrator._toml_get(tool, c.Infra.Toml.POETRY)
        if not isinstance(poetry, Table):
            return False
        poetry_deps = FlextInfraProjectMigrator._toml_get(
            poetry,
            c.Infra.Toml.DEPENDENCIES,
        )
        if not isinstance(poetry_deps, Table):
            return False
        return c.Infra.Packages.CORE in poetry_deps

    @staticmethod
    def _sha256_text(value: str) -> str:
        return hashlib.sha256(value.encode(c.Infra.Encoding.DEFAULT)).hexdigest()

    @staticmethod
    def _workspace_root_project(
        workspace_root: Path,
    ) -> m.Infra.Workspace.ProjectInfo | None:
        """Detect workspace root as a project if it has Makefile, pyproject.toml, and .git."""
        has_makefile = (workspace_root / c.Infra.Files.MAKEFILE_FILENAME).is_file()
        has_pyproject = (workspace_root / c.Infra.Files.PYPROJECT_FILENAME).is_file()
        has_git = (workspace_root / c.Infra.Git.DIR).exists()
        if not (has_makefile and has_pyproject and has_git):
            return None
        return m.Infra.Workspace.ProjectInfo(
            name=workspace_root.name,
            path=workspace_root,
            stack="python/workspace",
            has_tests=(workspace_root / c.Infra.Directories.TESTS).is_dir(),
            has_src=(workspace_root / c.Infra.Paths.DEFAULT_SRC_DIR).is_dir(),
        )

    @override
    def execute(
        self,
    ) -> r[t.NormalizedValue | BaseModel | list[t.NormalizedValue | BaseModel]]:
        return r[
            t.NormalizedValue | BaseModel | list[t.NormalizedValue | BaseModel]
        ].fail(
            "Use migrate() method directly",
        )

    def migrate(
        self,
        *,
        workspace_root: Path,
        dry_run: bool = False,
    ) -> r[list[m.Infra.Workspace.MigrationResult]]:
        """Migrate all projects in workspace."""
        root = workspace_root.resolve()
        if not root.is_dir():
            return r[list[m.Infra.Workspace.MigrationResult]].fail(
                f"workspace root does not exist: {root}",
            )
        if self._discovery is not None:
            discovered = self._discovery.discover_projects(root)
        else:
            discovered = u.Infra.discover_projects(root)
        if discovered.is_failure:
            return r[list[m.Infra.Workspace.MigrationResult]].fail(
                discovered.error or "project discovery failed",
            )
        discovered_projects: list[m.Infra.Workspace.ProjectInfo] = discovered.value
        projects = list(discovered_projects)
        workspace_project = self._workspace_root_project(root)
        if workspace_project is not None and all(
            existing.path != workspace_project.path for existing in projects
        ):
            projects.append(workspace_project)
        results: list[m.Infra.Workspace.MigrationResult] = [
            self._migrate_project(project=project, dry_run=dry_run)
            for project in projects
        ]
        return r[list[m.Infra.Workspace.MigrationResult]].ok(results)

    def _migrate_basemk(self, project_root: Path, *, dry_run: bool) -> r[str]:
        generated = self._generator.generate()
        if generated.is_failure:
            return r[str].fail(generated.error or "base.mk generation failed")
        generated_text: str = generated.value
        target = project_root / c.Infra.Files.BASE_MK
        current = (
            target.read_text(encoding=c.Infra.Encoding.DEFAULT)
            if target.exists()
            else ""
        )
        if self._sha256_text(current) == self._sha256_text(generated_text):
            if dry_run:
                return r[str].ok(
                    self._action_text("base.mk already up-to-date", dry_run=True),
                )
            return r[str].ok("")
        if not dry_run:
            try:
                u.write_file(target, generated_text, encoding=c.Infra.Encoding.DEFAULT)
            except OSError as exc:
                return r[str].fail(f"base.mk update failed: {exc}")
        return r[str].ok(
            self._action_text(
                "base.mk regenerated via BaseMkGenerator",
                dry_run=dry_run,
            ),
        )

    def _migrate_gitignore(self, project_root: Path, *, dry_run: bool) -> r[str]:
        gitignore_path = project_root / c.Infra.Files.GITIGNORE
        try:
            existing_lines = (
                gitignore_path.read_text(encoding=c.Infra.Encoding.DEFAULT).splitlines()
                if gitignore_path.exists()
                else []
            )
        except OSError as exc:
            return r[str].fail(f".gitignore read failed: {exc}")
        filtered = [
            line
            for line in existing_lines
            if line.strip() not in c.Infra.Workspace.GITIGNORE_REMOVE_EXACT
        ]
        existing_patterns = {line.strip() for line in filtered if line.strip()}
        missing = [
            pattern
            for pattern in c.Infra.Workspace.GITIGNORE_REQUIRED_PATTERNS
            if pattern not in existing_patterns
        ]
        if not missing and len(filtered) == len(existing_lines):
            if dry_run:
                return r[str].ok(
                    self._action_text(".gitignore already normalized", dry_run=True),
                )
            return r[str].ok("")
        next_lines = list(filtered)
        if missing:
            if next_lines and next_lines[-1].strip():
                next_lines.append("")
            next_lines.append(
                "# --- workspace-migrate: required ignores (auto-managed) ---",
            )
            next_lines.extend(missing)
        if not dry_run:
            body = "\n".join(next_lines).rstrip("\n") + "\n"
            try:
                u.write_file(gitignore_path, body, encoding=c.Infra.Encoding.DEFAULT)
            except OSError as exc:
                return r[str].fail(f".gitignore update failed: {exc}")
        return r[str].ok(
            self._action_text(
                ".gitignore cleaned from scripts/ and normalized",
                dry_run=dry_run,
            ),
        )

    def _migrate_makefile(self, project_root: Path, *, dry_run: bool) -> r[str]:
        makefile_path = project_root / c.Infra.Files.MAKEFILE_FILENAME
        if not makefile_path.exists():
            if dry_run:
                return r[str].ok(self._action_text("Makefile not found", dry_run=True))
            return r[str].ok("")
        try:
            original = makefile_path.read_text(encoding=c.Infra.Encoding.DEFAULT)
        except OSError as exc:
            return r[str].fail(f"Makefile read failed: {exc}")
        updated = original
        for before, after in c.Infra.Workspace.MAKEFILE_REPLACEMENTS:
            updated = updated.replace(before, after)
        if updated == original:
            if dry_run:
                return r[str].ok(
                    self._action_text("Makefile already migrated", dry_run=True),
                )
            return r[str].ok("")
        if not dry_run:
            try:
                u.write_file(makefile_path, updated, encoding=c.Infra.Encoding.DEFAULT)
            except OSError as exc:
                return r[str].fail(f"Makefile update failed: {exc}")
        return r[str].ok(
            self._action_text("Makefile scripts/ references migrated", dry_run=dry_run),
        )

    def _migrate_project(
        self,
        *,
        project: p.Infra.ProjectInfo,
        dry_run: bool,
    ) -> m.Infra.Workspace.MigrationResult:
        changes: list[str] = []
        errors: list[str] = []
        self._append_result(
            self._migrate_basemk(project.path, dry_run=dry_run),
            changes,
            errors,
        )
        self._append_result(
            self._migrate_makefile(project.path, dry_run=dry_run),
            changes,
            errors,
        )
        self._append_result(
            self._migrate_pyproject(
                project.path,
                project_name=project.name,
                dry_run=dry_run,
            ),
            changes,
            errors,
        )
        self._append_result(
            self._migrate_gitignore(project.path, dry_run=dry_run),
            changes,
            errors,
        )
        if not changes and (not errors):
            changes.append("no changes needed")
        return m.Infra.Workspace.MigrationResult(
            project=project.name,
            changes=changes,
            errors=errors,
        )

    def _migrate_pyproject(
        self,
        project_root: Path,
        *,
        project_name: str,
        dry_run: bool,
    ) -> r[str]:
        pyproject_path = project_root / c.Infra.Files.PYPROJECT_FILENAME
        if not pyproject_path.exists():
            if dry_run:
                return r[str].ok(
                    self._action_text("pyproject.toml not found", dry_run=True),
                )
            return r[str].ok("")
        if project_name == c.Infra.Packages.CORE:
            if dry_run:
                return r[str].ok(
                    self._action_text(
                        "pyproject.toml dependency unchanged for flext-core",
                        dry_run=True,
                    ),
                )
            return r[str].ok("")
        document_result = u.Infra.read_document(pyproject_path)
        if document_result.is_failure:
            return r[str].fail(
                document_result.error or "pyproject parse failed",
            )
        document: tomlkit.TOMLDocument = document_result.value
        if self._has_flext_core_dependency(document):
            if dry_run:
                return r[str].ok(
                    self._action_text(
                        "pyproject.toml already includes flext-core dependency",
                        dry_run=True,
                    ),
                )
            return r[str].ok("")
        project_table = self._ensure_table(document, c.Infra.Toml.PROJECT)
        dependencies_raw = self._toml_get(project_table, c.Infra.Toml.DEPENDENCIES)
        dependencies: list[str] = []
        if isinstance(dependencies_raw, list):
            dependency_items: list[JsonValue] = _OBJECT_LIST_ADAPTER.validate_python(
                dependencies_raw
            )
            dependencies = [str(dep_raw) for dep_raw in dependency_items]
        dependency_spec = "flext-core @ ../flext-core"
        if dependency_spec not in dependencies:
            dependencies.append(dependency_spec)
        project_table[c.Infra.Toml.DEPENDENCIES] = dependencies
        if not dry_run:
            write_result = u.Infra.write_document(pyproject_path, document)
            if write_result.is_failure:
                return r[str].fail(
                    write_result.error or "pyproject update failed",
                )
        return r[str].ok(
            self._action_text(
                "pyproject.toml adds flext-core dependency",
                dry_run=dry_run,
            ),
        )


__all__ = ["FlextInfraProjectMigrator"]
