"""Migrate projects to unified FLEXT infrastructure."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import override

import tomlkit
from flext_core.result import FlextResult as r
from flext_core.service import FlextService
from tomlkit.exceptions import ParseError
from tomlkit.items import Table

from flext_infra.basemk.generator import BaseMkGenerator
from flext_infra.constants import ic
from flext_infra.discovery import DiscoveryService
from flext_infra.models import im

_MAKEFILE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    # scripts/ path → unified CLI: detection
    (
        'python3 "$(BASE_MK_DIR)/scripts/mode.py"',
        "python -m flext_infra workspace detect",
    ),
    # scripts/ path → unified CLI: sync
    (
        'python3 "$(WORKSPACE_ROOT)/scripts/sync.py"',
        "python -m flext_infra workspace sync",
    ),
    # scripts/ path → unified CLI: deps
    (
        'python3 "$(WORKSPACE_ROOT)/scripts/dependencies/sync_internal_deps.py"',
        "python -m flext_infra deps internal-sync",
    ),
    # scripts/ path → unified CLI: check
    (
        'python "$(WORKSPACE_ROOT)/scripts/check/fix_pyrefly_config.py"',
        "python -m flext_infra check fix-pyrefly-config",
    ),
    (
        'python "$(WORKSPACE_ROOT)/scripts/check/workspace_check.py"',
        "python -m flext_infra check run",
    ),
    # scripts/ path → unified CLI: pytest-diag
    (
        '$(VENV_PYTHON) "$(BASE_MK_DIR)/scripts/core/pytest_diag_extract.py"',
        "$(VENV_PYTHON) -m flext_infra core pytest-diag",
    ),
    # scripts/ path → unified CLI: pr
    (
        'python3 "$(WORKSPACE_ROOT)/scripts/github/pr_manager.py"',
        "python3 -m flext_infra github pr",
    ),
)

_GITIGNORE_REMOVE_EXACT: frozenset[str] = frozenset(
    {
        "!scripts/",
        "!scripts/**",
        "scripts/",
        "/scripts/",
    },
)

_GITIGNORE_REQUIRED_PATTERNS: tuple[str, ...] = (
    ".reports/",
    ".venv/",
    "__pycache__/",
)

_PYPROJECT_FILE = ic.Files.PYPROJECT_FILENAME


class ProjectMigrator(FlextService[list[im.MigrationResult]]):
    """Migrate projects to standardized base.mk, Makefile, and pyproject structure."""

    def __init__(
        self,
        *,
        discovery: DiscoveryService | None = None,
        generator: BaseMkGenerator | None = None,
    ) -> None:
        """Initialize migrator with optional custom discovery and generator services."""
        super().__init__()
        self._discovery = discovery or DiscoveryService()
        self._generator = generator or BaseMkGenerator()

    @override
    def execute(self) -> r[list[im.MigrationResult]]:
        return r[list[im.MigrationResult]].fail("Use migrate() method directly")

    def migrate(
        self,
        *,
        workspace_root: Path,
        dry_run: bool = False,
    ) -> r[list[im.MigrationResult]]:
        """Migrate all projects in workspace."""
        root = workspace_root.resolve()
        if not root.is_dir():
            return r[list[im.MigrationResult]].fail(
                f"workspace root does not exist: {root}",
            )

        discovered = self._discovery.discover_projects(root)
        if discovered.is_failure:
            return r[list[im.MigrationResult]].fail(
                discovered.error or "project discovery failed",
            )

        projects = list(discovered.value)
        workspace_project = self._workspace_root_project(root)
        if workspace_project is not None and all(
            existing.path != workspace_project.path for existing in projects
        ):
            projects.append(workspace_project)

        results: list[im.MigrationResult] = [
            self._migrate_project(project=project, dry_run=dry_run)
            for project in projects
        ]

        return r[list[im.MigrationResult]].ok(results)

    @staticmethod
    def _workspace_root_project(workspace_root: Path) -> im.ProjectInfo | None:
        """Detect workspace root as a project if it has Makefile, pyproject.toml, and .git."""
        has_makefile = (workspace_root / ic.Files.MAKEFILE_FILENAME).is_file()
        has_pyproject = (workspace_root / _PYPROJECT_FILE).is_file()
        has_git = (workspace_root / ".git").exists()
        if not (has_makefile and has_pyproject and has_git):
            return None

        return im.ProjectInfo(
            name=workspace_root.name,
            path=workspace_root,
            stack="python/workspace",
            has_tests=(workspace_root / "tests").is_dir(),
            has_src=(workspace_root / ic.Paths.DEFAULT_SRC_DIR).is_dir(),
        )

    def _migrate_project(
        self,
        *,
        project: im.ProjectInfo,
        dry_run: bool,
    ) -> im.MigrationResult:
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

        if not changes and not errors:
            changes.append("no changes needed")

        return im.MigrationResult(project=project.name, changes=changes, errors=errors)

    @staticmethod
    def _append_result(
        result: r[str],
        changes: list[str],
        errors: list[str],
    ) -> None:
        if result.is_failure:
            errors.append(result.error or "migration action failed")
            return
        if result.value:
            changes.append(result.value)

    def _migrate_basemk(self, project_root: Path, *, dry_run: bool) -> r[str]:
        generated = self._generator.generate()
        if generated.is_failure:
            return r[str].fail(
                generated.error or "base.mk generation failed",
            )

        target = project_root / "base.mk"
        current = (
            target.read_text(encoding=ic.Encoding.DEFAULT) if target.exists() else ""
        )
        if self._sha256_text(current) == self._sha256_text(generated.value):
            if dry_run:
                return r[str].ok(
                    self._action_text("base.mk already up-to-date", dry_run=True)
                )
            return r[str].ok("")

        if not dry_run:
            try:
                _ = target.write_text(generated.value, encoding=ic.Encoding.DEFAULT)
            except OSError as exc:
                return r[str].fail(f"base.mk update failed: {exc}")

        return r[str].ok(
            self._action_text(
                "base.mk regenerated via BaseMkGenerator",
                dry_run=dry_run,
            )
        )

    def _migrate_makefile(self, project_root: Path, *, dry_run: bool) -> r[str]:
        makefile_path = project_root / ic.Files.MAKEFILE_FILENAME
        if not makefile_path.exists():
            if dry_run:
                return r[str].ok(self._action_text("Makefile not found", dry_run=True))
            return r[str].ok("")

        try:
            original = makefile_path.read_text(encoding=ic.Encoding.DEFAULT)
        except OSError as exc:
            return r[str].fail(f"Makefile read failed: {exc}")

        updated = original
        for before, after in _MAKEFILE_REPLACEMENTS:
            updated = updated.replace(before, after)

        if updated == original:
            if dry_run:
                return r[str].ok(
                    self._action_text("Makefile already migrated", dry_run=True)
                )
            return r[str].ok("")

        if not dry_run:
            try:
                _ = makefile_path.write_text(updated, encoding=ic.Encoding.DEFAULT)
            except OSError as exc:
                return r[str].fail(f"Makefile update failed: {exc}")

        return r[str].ok(
            self._action_text("Makefile scripts/ references migrated", dry_run=dry_run)
        )

    def _migrate_pyproject(
        self,
        project_root: Path,
        *,
        project_name: str,
        dry_run: bool,
    ) -> r[str]:
        pyproject_path = project_root / _PYPROJECT_FILE
        if not pyproject_path.exists():
            if dry_run:
                return r[str].ok(
                    self._action_text("pyproject.toml not found", dry_run=True)
                )
            return r[str].ok("")
        if project_name == "flext-core":
            if dry_run:
                return r[str].ok(
                    self._action_text(
                        "pyproject.toml dependency unchanged for flext-core",
                        dry_run=True,
                    )
                )
            return r[str].ok("")

        try:
            document = tomlkit.parse(
                pyproject_path.read_text(encoding=ic.Encoding.DEFAULT)
            )
        except (ParseError, OSError) as exc:
            return r[str].fail(f"pyproject parse failed: {exc}")

        if self._has_flext_core_dependency(document):
            if dry_run:
                return r[str].ok(
                    self._action_text(
                        "pyproject.toml already includes flext-core dependency",
                        dry_run=True,
                    )
                )
            return r[str].ok("")

        project_table = self._ensure_table(document, "project")
        dependencies = project_table.get("dependencies")
        if not isinstance(dependencies, list):
            dependencies = tomlkit.array()
            _ = dependencies.multiline(True)
            project_table["dependencies"] = dependencies

        dependencies.append("flext-core @ ../flext-core")

        if not dry_run:
            try:
                _ = pyproject_path.write_text(
                    tomlkit.dumps(document),
                    encoding=ic.Encoding.DEFAULT,
                )
            except OSError as exc:
                return r[str].fail(f"pyproject update failed: {exc}")

        return r[str].ok(
            self._action_text(
                "pyproject.toml adds flext-core dependency",
                dry_run=dry_run,
            )
        )

    def _migrate_gitignore(self, project_root: Path, *, dry_run: bool) -> r[str]:
        gitignore_path = project_root / ".gitignore"
        try:
            existing_lines = (
                gitignore_path.read_text(encoding=ic.Encoding.DEFAULT).splitlines()
                if gitignore_path.exists()
                else []
            )
        except OSError as exc:
            return r[str].fail(f".gitignore read failed: {exc}")

        filtered = [
            line
            for line in existing_lines
            if line.strip() not in _GITIGNORE_REMOVE_EXACT
        ]

        existing_patterns = {line.strip() for line in filtered if line.strip()}
        missing = [
            pattern
            for pattern in _GITIGNORE_REQUIRED_PATTERNS
            if pattern not in existing_patterns
        ]

        if not missing and len(filtered) == len(existing_lines):
            if dry_run:
                return r[str].ok(
                    self._action_text(".gitignore already normalized", dry_run=True)
                )
            return r[str].ok("")

        next_lines = list(filtered)
        if missing:
            if next_lines and next_lines[-1].strip():
                next_lines.append("")
            next_lines.append(
                "# --- workspace-migrate: required ignores (auto-managed) ---"
            )
            next_lines.extend(missing)

        if not dry_run:
            body = "\n".join(next_lines).rstrip("\n") + "\n"
            try:
                _ = gitignore_path.write_text(body, encoding=ic.Encoding.DEFAULT)
            except OSError as exc:
                return r[str].fail(f".gitignore update failed: {exc}")

        return r[str].ok(
            self._action_text(
                ".gitignore cleaned from scripts/ and normalized",
                dry_run=dry_run,
            )
        )

    @staticmethod
    def _action_text(action: str, *, dry_run: bool) -> str:
        return f"[DRY-RUN] {action}" if dry_run else action

    @staticmethod
    def _sha256_text(value: str) -> str:
        return hashlib.sha256(value.encode(ic.Encoding.DEFAULT)).hexdigest()

    @staticmethod
    def _has_flext_core_dependency(document: tomlkit.TOMLDocument) -> bool:
        project = document.get("project")
        if isinstance(project, Table):
            deps = project.get("dependencies")
            if isinstance(deps, list):
                for dep in deps:
                    if str(dep).strip().startswith("flext-core"):
                        return True

        tool = document.get("tool")
        if not isinstance(tool, Table):
            return False
        poetry = tool.get("poetry")
        if not isinstance(poetry, Table):
            return False
        poetry_deps = poetry.get("dependencies")
        if not isinstance(poetry_deps, Table):
            return False
        return "flext-core" in poetry_deps

    @staticmethod
    def _ensure_table(document: tomlkit.TOMLDocument, key: str) -> Table:
        current = document.get(key)
        if isinstance(current, Table):
            return current
        created = tomlkit.table()
        document[key] = created
        return created


__all__ = ["ProjectMigrator"]
