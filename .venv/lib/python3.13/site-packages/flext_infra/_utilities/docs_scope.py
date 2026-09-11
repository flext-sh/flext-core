"""Docs scope helpers for FLEXT-only discovery and project classification."""

from __future__ import annotations

import operator
from fnmatch import fnmatch
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

from flext_cli import FlextCliUtilities as u
from flext_core.result import FlextResult as r
from flext_infra._models.workspace import FlextInfraModelsWorkspace as mw
from flext_infra._utilities.dependencies import FlextInfraUtilitiesDependencies
from flext_infra._utilities.project_discovery import FlextInfraUtilitiesProjectDiscovery
from flext_infra._utilities.pyproject import FlextInfraUtilitiesPyproject
from flext_infra.constants import FlextInfraConstants as c
from flext_infra.typings import FlextInfraTypes as t

if TYPE_CHECKING:
    from flext_infra import FlextInfraProtocols as p


class FlextInfraUtilitiesDocsScope:
    """Utility helpers for docs scope policy and project classification."""

    @staticmethod
    @cache
    def _project_state(project_root: str) -> mw.ProjectPyprojectState:
        """Return cached parsed pyproject state for one project root.

        When the pyproject is absent or empty, the returned state carries
        empty ``project_name``/``package_name`` (legitimate "not a project"
        signal). When the pyproject is present but missing ``[project]`` or
        ``[project].name``, :meth:`project_name_from_payload` raises — no
        silent fallback to directory-name.
        """
        root = Path(project_root)
        pyproject_path = root / c.Infra.PYPROJECT_FILENAME
        payload = FlextInfraUtilitiesPyproject.pyproject_payload(pyproject_path)
        docs_meta = FlextInfraUtilitiesDocsScope.docs_meta_from_payload(payload)
        dependency_names = tuple(
            FlextInfraUtilitiesDependencies.declared_dependency_names_from_payload(
                payload
            )
        )
        if not payload:
            empty_state: mw.ProjectPyprojectState = (
                mw.ProjectPyprojectState.model_construct(
                    project_root=root,
                    pyproject_path=pyproject_path,
                    payload=payload,
                    docs_meta=docs_meta,
                    project_name="",
                    package_name="",
                    dependency_names=dependency_names,
                )
            )
            return empty_state
        state: mw.ProjectPyprojectState = mw.ProjectPyprojectState.model_construct(
            project_root=root,
            pyproject_path=pyproject_path,
            payload=payload,
            docs_meta=docs_meta,
            project_name=FlextInfraUtilitiesDocsScope.project_name_from_payload(
                root, payload
            ),
            package_name=FlextInfraUtilitiesDocsScope.package_name_from_payload(
                root, payload, docs_meta
            ),
            dependency_names=dependency_names,
        )
        return state

    @staticmethod
    def project_state(project_root: Path) -> mw.ProjectPyprojectState:
        """Return the centralized parsed state for one project root."""
        return FlextInfraUtilitiesDocsScope._project_state(str(project_root.resolve()))

    @staticmethod
    def resolve_projects(
        repository_root: Path, names: t.StrSequence
    ) -> p.Result[t.SequenceOf[mw.ProjectInfo]]:
        """Resolve project names through repository-local topology only."""
        discover_result = FlextInfraUtilitiesDocsScope.discover_projects(
            repository_root
        )
        if discover_result.failure:
            return r[t.SequenceOf[mw.ProjectInfo]].fail(
                discover_result.error or "discovery failed"
            )
        projects = list(discover_result.value)
        resolved_repository_root = repository_root.resolve()
        if all(
            project.path.resolve() != resolved_repository_root for project in projects
        ):
            root_project = FlextInfraUtilitiesDocsScope._project_info_for_entry(
                resolved_repository_root,
                declared_repositories=FlextInfraUtilitiesDocsScope._declared_repository_path_set(
                    resolved_repository_root
                ),
            )
            if root_project is not None:
                projects.append(root_project)
        if not names:
            return r[t.SequenceOf[mw.ProjectInfo]].ok(
                sorted(projects, key=operator.attrgetter("name"))
            )
        by_name: dict[str, mw.ProjectInfo] = {}
        for project in projects:
            by_name.setdefault(project.name, project)
            by_name.setdefault(project.path.name, project)
            project_path = project.path.resolve()
            if project_path == resolved_repository_root:
                by_name.setdefault(".", project)
            elif project_path.is_relative_to(resolved_repository_root):
                by_name.setdefault(
                    project_path.relative_to(resolved_repository_root).as_posix(),
                    project,
                )
        missing = [name for name in names if name not in by_name]
        if missing:
            missing_text = ", ".join(sorted(missing))
            return r[t.SequenceOf[mw.ProjectInfo]].fail(
                f"unknown projects: {missing_text}"
            )
        return r[t.SequenceOf[mw.ProjectInfo]].ok(
            sorted((by_name[name] for name in names), key=operator.attrgetter("name"))
        )

    @staticmethod
    def project_name_from_payload(entry: Path, payload: t.JsonMapping) -> str:
        """Return the declared project name from ``[project].name``."""
        return FlextInfraUtilitiesPyproject.project_name_from_payload(entry, payload)

    @staticmethod
    def _declared_repository_path_set(repository_root: Path) -> frozenset[Path]:
        """Return resolved declared_repositories declared by this root's ``.gitmodules``."""
        resolved_root = repository_root.resolve()
        return frozenset(
            (resolved_root / path).resolve()
            for path in FlextInfraUtilitiesPyproject.workspace_project_paths(
                resolved_root
            )
        )

    @staticmethod
    def _project_info_for_entry(
        entry: Path, *, declared_repositories: frozenset[Path]
    ) -> mw.ProjectInfo | None:
        """Build one canonical project descriptor for one discovered project root."""
        entry = entry.resolve()
        pyproject = entry / c.Infra.PYPROJECT_FILENAME
        if not pyproject.is_file():
            return None
        # Pre-validate [project].name BEFORE triggering the strict cached state builder.
        payload_preview = FlextInfraUtilitiesPyproject.pyproject_payload(pyproject)
        project_section = payload_preview.get("project")
        project_name = (
            project_section.get("name") if isinstance(project_section, dict) else None
        )
        if (
            not isinstance(project_section, dict)
            or not isinstance(project_name, str)
            or not project_name.strip()
        ):
            return None
        project_state = FlextInfraUtilitiesDocsScope.project_state(entry)
        is_declared = entry in declared_repositories
        enabled = project_state.docs_meta.get("enabled", True)
        if isinstance(enabled, bool) and not enabled:
            return None
        has_src = (entry / c.Infra.DEFAULT_SRC_DIR).is_dir()
        has_tests = (entry / c.Infra.DIR_TESTS).is_dir()
        has_deps = bool(project_section.get("dependencies"))
        if not is_declared and not has_src and not has_tests and not has_deps:
            return None
        # Topology is proven by the checkout itself. Declared-repository is the
        # aggregate's relationship to this path, not the repository's own
        # workspace/standalone classification.
        make_profile = (
            c.Infra.MakeProfile.WORKSPACE
            if (entry / c.Infra.GITMODULES).is_file()
            else c.Infra.MakeProfile.STANDALONE
        )
        project_info: mw.ProjectInfo = mw.ProjectInfo.model_construct(
            path=entry,
            name=project_state.project_name,
            stack="python/flext",
            has_tests=has_tests,
            has_src=has_src,
            project_class=(
                FlextInfraUtilitiesDocsScope.classify_project_from_meta(
                    project_state.project_name, project_state.docs_meta
                )
            ),
            package_name=project_state.package_name,
            make_profile=make_profile,
            declared_subproject=is_declared,
        )
        return project_info

    @staticmethod
    def config_path(repository_root: Path) -> Path:
        """Return the minimal docs policy settings path."""
        dir_docs: str = c.Infra.DIR_DOCS
        docs_config: str = c.Infra.DOCS_CONFIG_FILENAME
        return repository_root / dir_docs / docs_config

    @staticmethod
    def project_payload(project_root: Path) -> t.JsonMapping:
        """Return a project's ``pyproject.toml`` payload as a plain mapping."""
        return FlextInfraUtilitiesDocsScope.project_state(project_root).payload

    @staticmethod
    def load_config(repository_root: Path) -> t.JsonMapping:
        """Load the minimal docs policy settings if present."""
        path = FlextInfraUtilitiesDocsScope.config_path(repository_root)
        empty: t.JsonMapping = {}
        if not path.exists():
            return empty
        result = u.Cli.json_read(path)
        if result.success:
            value = result.value
            if isinstance(value, dict):
                return dict(value)
        return empty

    @staticmethod
    def excluded_roots(repository_root: Path) -> t.Infra.StrSet:
        """Return explicitly excluded root directories from docs scope."""
        payload = FlextInfraUtilitiesDocsScope.load_config(repository_root)
        scope = payload.get("scope")
        if not isinstance(scope, dict):
            return set()
        excluded = scope.get("exclude_roots")
        if not isinstance(excluded, list):
            return set()
        return {str(item).strip() for item in excluded if str(item).strip()}

    @staticmethod
    def project_docs_meta(project_root: Path) -> t.JsonMapping:
        """Return optional ``tool.flext.docs`` metadata from a project pyproject."""
        return FlextInfraUtilitiesDocsScope.project_state(project_root).docs_meta

    @staticmethod
    def docs_meta_list(project_root: Path, key: str) -> t.StrSequence:
        """Return one normalized string-list value from ``tool.flext.docs``."""
        docs_meta = FlextInfraUtilitiesDocsScope.project_docs_meta(project_root)
        raw = docs_meta.get(key)
        if not isinstance(raw, list):
            return ()
        return [str(item).strip() for item in raw if str(item).strip()]

    @staticmethod
    def is_excluded_doc_path(project_root: Path, relative_path: Path) -> bool:
        """Return whether a relative docs path is excluded by ``tool.flext.docs``."""
        candidate = relative_path.as_posix()
        for pattern in FlextInfraUtilitiesDocsScope.docs_meta_list(
            project_root, "exclude_docs"
        ):
            if fnmatch(candidate, pattern):
                return True
        return False

    @staticmethod
    def is_governed_project(project_name: str, repository_root: Path) -> bool:
        """Return whether a project belongs to the governed FLEXT docs scope."""
        project_root = repository_root / project_name
        docs_meta = FlextInfraUtilitiesDocsScope.project_docs_meta(project_root)
        enabled = docs_meta.get("enabled", True)
        is_enabled = enabled if isinstance(enabled, bool) else True
        return (
            project_name.startswith(c.Infra.PKG_PREFIX_HYPHEN)
            and project_name
            not in FlextInfraUtilitiesDocsScope.excluded_roots(repository_root)
            and is_enabled
        )

    @staticmethod
    def docs_meta_from_payload(payload: t.JsonMapping) -> t.JsonMapping:
        """Extract ``tool.flext.docs`` metadata from an already-parsed payload."""
        return FlextInfraUtilitiesPyproject.docs_meta_from_payload(payload)

    @staticmethod
    def classify_project_from_meta(project_name: str, docs_meta: t.JsonMapping) -> str:
        """Classify a project using pre-loaded docs metadata (avoids re-parsing).

        Project-prefix heuristics derive from ``c.Infra.INTEGRATION_CLASS_PREFIXES``
        (SSOT for integration project family) so adding a new family member
        requires editing only the canonical class-prefix tuple.
        """
        configured = docs_meta.get("project_class")
        if isinstance(configured, str) and configured.strip():
            return configured.strip()
        integration_prefixes = tuple(
            f"{c.Infra.PKG_PREFIX_HYPHEN}{prefix.removeprefix('Flext').lower()}-"
            for prefix in c.Infra.INTEGRATION_CLASS_PREFIXES
        )
        if project_name.startswith(integration_prefixes):
            return "integration"
        if project_name == f"{c.Infra.PKG_PREFIX_HYPHEN}infra":
            return "infra"
        if project_name == f"{c.Infra.PKG_PREFIX_HYPHEN}tests":
            return "test"
        return "domain"

    @staticmethod
    def package_name_from_payload(
        project_root: Path, payload: t.JsonMapping, docs_meta: t.JsonMapping
    ) -> str:
        """Return the primary package name using pre-loaded payload.

        Resolution order (no silent fallbacks for flext projects):
          1. Explicit ``[tool.flext.docs].package_name`` override.
          2. ``[tool.hatch.build.targets.wheel.packages]`` first entry.
          3. First ``src/<pkg>/__init__.py`` directory.
          4. Empty string for non-flext projects (roots).

        Raises ``ValueError`` only for flext- projects unable to resolve.
        """
        return FlextInfraUtilitiesPyproject.package_name_from_payload(
            project_root, payload, docs_meta
        )

    @staticmethod
    def project_package_name(project_root: Path) -> str:
        """Return the primary Python package name for a project."""
        return FlextInfraUtilitiesPyproject.project_package_name(project_root)

    @staticmethod
    def discover_projects(
        repository_root: Path,
    ) -> p.Result[t.SequenceOf[mw.ProjectInfo]]:
        """Discover the root or projects declared by its own ``.gitmodules``."""
        if not repository_root.exists() or not repository_root.is_dir():
            return r[t.SequenceOf[mw.ProjectInfo]].fail(
                f"discovery failed: invalid repository root {repository_root}"
            )
        excluded = FlextInfraUtilitiesDocsScope.excluded_roots(repository_root)
        declared_repositories = (
            FlextInfraUtilitiesDocsScope._declared_repository_path_set(repository_root)
        )
        project_roots = FlextInfraUtilitiesProjectDiscovery.discover_project_candidates(
            repository_root
        )
        root_project: mw.ProjectInfo | None = None
        projects: list[mw.ProjectInfo] = []
        for project_root in project_roots:
            if project_root.name == "cmd" or project_root.name in excluded:
                continue
            if (
                project_root == repository_root.resolve()
                and not (project_root / c.Infra.DEFAULT_SRC_DIR).is_dir()
            ):
                continue
            project_info = FlextInfraUtilitiesDocsScope._project_info_for_entry(
                project_root, declared_repositories=declared_repositories
            )
            if project_info is None:
                continue
            if project_root == repository_root.resolve():
                root_project = project_info
                continue
            projects.append(project_info)
        if not projects and root_project is not None:
            return r[t.SequenceOf[mw.ProjectInfo]].ok([root_project])
        return r[t.SequenceOf[mw.ProjectInfo]].ok(projects)

    @staticmethod
    def required_project_files() -> t.StrSequence:
        """Return the required standard docs contract for FLEXT projects."""
        return [
            "README.md",
            "docs/index.md",
            "docs/guides/README.md",
            "docs/api-reference/README.md",
            "docs/api-reference/generated/overview.md",
            "docs/api-reference/generated/public-api.md",
            "docs/api-reference/generated/modules/index.md",
            "mkdocs.yml",
        ]


__all__: list[str] = ["FlextInfraUtilitiesDocsScope"]
