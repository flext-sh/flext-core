"""Stable topology and coherent state snapshots for Mise publication."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core import r
from flext_infra import c, m, u
from flext_infra._utilities.project_managed_artifacts import (
    FlextInfraUtilitiesProjectManagedArtifacts,
)
from flext_infra.codegen import _mise_artifacts_files as files
from flext_infra.workspace.detector import FlextInfraWorkspaceDetector

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraMiseWorkspacePlanner:
    """Resolve layout once, then snapshot mutable inputs only under the lock."""

    def __init__(self, owner: p.Infra.MiseArtifactsOwner) -> None:
        self._owner = owner

    def scope_root(self) -> p.Result[Path]:
        """Resolve the stable lock scope from physical Git identity only."""
        requested = self._owner.repository_root.expanduser().absolute()
        physical = self._physical_directory(requested)
        if physical.failure:
            return r[Path].from_failure(physical)
        identity = u.Infra.git_identity(m.Infra.GitRepoRequest(repo_root=requested))
        if identity.failure:
            return r[Path].fail(
                identity.error or "cannot resolve Mise workspace Git identity"
            )
        if not identity.value.is_submodule:
            return r[Path].ok(requested)
        superproject_root = identity.value.superproject_root
        if superproject_root is None:
            return r[Path].fail(
                f"Git submodule has no Mise coordination root: {requested}"
            )
        scope_root = superproject_root.expanduser().absolute()
        physical_scope = self._physical_directory(scope_root)
        if physical_scope.failure:
            return r[Path].from_failure(physical_scope)
        return r[Path].ok(scope_root)

    def layout(
        self, scope_root: Path | None = None
    ) -> p.Result[m.Infra.MiseToolchainWorkspaceLayout]:
        """Resolve governed topology after the stable workspace lock is held."""
        requested = self._owner.repository_root.expanduser().absolute()
        resolved_scope = (
            self.scope_root() if scope_root is None else r[Path].ok(scope_root)
        )
        if resolved_scope.failure:
            return r[m.Infra.MiseToolchainWorkspaceLayout].from_failure(resolved_scope)
        scope_root = resolved_scope.value
        workspace = FlextInfraWorkspaceDetector.load_workspace_spec(scope_root)
        if workspace.failure:
            return r[m.Infra.MiseToolchainWorkspaceLayout].fail(
                workspace.error or "cannot load governed Mise workspace"
            )
        if requested != scope_root and not any(
            (scope_root / project.path).absolute() == requested
            for project in workspace.value.declared_repositories
        ):
            return r[m.Infra.MiseToolchainWorkspaceLayout].fail(
                f"Git submodule is absent from governed workspace: {requested}"
            )
        selectors = (
            ".",
            *(
                project.path.as_posix()
                for project in workspace.value.declared_repositories
            ),
        )
        return self.layout_from_selectors(scope_root, selectors)

    def layout_from_selectors(
        self, scope_root: Path, selectors: tuple[str, ...]
    ) -> p.Result[m.Infra.MiseToolchainWorkspaceLayout]:
        """Rebuild exact topology from already authenticated journal selectors."""
        if not selectors or len(set(selectors)) != len(selectors):
            return r[m.Infra.MiseToolchainWorkspaceLayout].fail(
                "Mise project selectors must be nonempty and unique"
            )
        projects: list[m.Infra.MiseToolchainProjectLayout] = []
        for selector in selectors:
            project = self._project_layout(scope_root, selector)
            if project.failure:
                return r[m.Infra.MiseToolchainWorkspaceLayout].from_failure(project)
            projects.append(project.value)
        state_root = self._state_root(scope_root)
        if state_root.failure:
            return r[m.Infra.MiseToolchainWorkspaceLayout].from_failure(state_root)
        return r[m.Infra.MiseToolchainWorkspaceLayout].ok(
            m.Infra.MiseToolchainWorkspaceLayout(
                scope_root=scope_root,
                state_root=state_root.value,
                projects=tuple(projects),
            )
        )

    def select_layout(
        self,
        layout: m.Infra.MiseToolchainWorkspaceLayout,
        config_plans: tuple[m.Infra.CodegenFilePlan, ...] = (),
    ) -> p.Result[m.Infra.MiseToolchainWorkspaceLayout]:
        """Select only projects owned by this conform request or direct caller."""
        if config_plans:
            planned_paths = tuple(item.path for item in config_plans)
            if len(set(planned_paths)) != len(planned_paths):
                return r[m.Infra.MiseToolchainWorkspaceLayout].fail(
                    "duplicate Mise configuration plans"
                )
            known = {project.artifacts.config for project in layout.projects}
            unknown = tuple(path for path in planned_paths if path not in known)
            if unknown:
                return r[m.Infra.MiseToolchainWorkspaceLayout].fail(
                    f"Mise configuration plan is outside workspace: {unknown[0]}"
                )
            selected = tuple(
                project
                for project in layout.projects
                if project.artifacts.config in planned_paths
            )
        else:
            requested = self._owner.repository_root.expanduser().absolute()
            selected = (
                layout.projects
                if requested == layout.scope_root
                else tuple(
                    project for project in layout.projects if project.root == requested
                )
            )
        if not selected:
            return r[m.Infra.MiseToolchainWorkspaceLayout].fail(
                "Mise transaction selected no governed project"
            )
        return r[m.Infra.MiseToolchainWorkspaceLayout].ok(
            m.Infra.MiseToolchainWorkspaceLayout(
                scope_root=layout.scope_root,
                state_root=layout.state_root,
                projects=selected,
            )
        )

    def layout_for_config_plans(
        self, scope_root: Path, config_plans: tuple[m.Infra.CodegenFilePlan, ...]
    ) -> p.Result[m.Infra.MiseToolchainWorkspaceLayout]:
        """Derive exact selected project topology from the locked conform plan."""
        if not config_plans:
            return r[m.Infra.MiseToolchainWorkspaceLayout].fail(
                "Mise transaction requires configuration plans"
            )
        selectors: list[str] = []
        expected_paths: list[Path] = []
        for plan in config_plans:
            if plan.path.name != files.CONFIG_SPEC[0] or ".." in plan.path.parts:
                return r[m.Infra.MiseToolchainWorkspaceLayout].fail(
                    f"invalid Mise configuration plan path: {plan.path}"
                )
            try:
                selector = (
                    plan.path.parent
                    .absolute()
                    .relative_to(scope_root.absolute())
                    .as_posix()
                )
            except ValueError:
                return r[m.Infra.MiseToolchainWorkspaceLayout].fail(
                    f"Mise configuration plan escapes scope: {plan.path}"
                )
            selectors.append(selector)
            expected_paths.append(plan.path)
        layout = self.layout_from_selectors(scope_root, tuple(selectors))
        if layout.failure:
            return layout
        if tuple(
            project.artifacts.config for project in layout.value.projects
        ) != tuple(expected_paths):
            return r[m.Infra.MiseToolchainWorkspaceLayout].fail(
                "Mise configuration plan paths differ from derived topology"
            )
        return layout

    def snapshot(
        self,
        layout: m.Infra.MiseToolchainWorkspaceLayout,
        config_plans: tuple[m.Infra.CodegenFilePlan, ...] = (),
    ) -> p.Result[m.Infra.MiseToolchainWorkspacePlan]:
        """Capture one complete byte-and-mode snapshot for a stable layout."""
        planned_configs = {item.path: item for item in config_plans}
        expected_paths = {project.artifacts.config for project in layout.projects}
        if config_plans and (
            len(planned_configs) != len(config_plans)
            or set(planned_configs) != expected_paths
        ):
            return r[m.Infra.MiseToolchainWorkspacePlan].fail(
                "Mise configuration plans differ from workspace topology"
            )
        projects: list[m.Infra.MiseToolchainProjectState] = []
        for project_layout in layout.projects:
            project = self._project_state(
                project_layout, planned_configs.get(project_layout.artifacts.config)
            )
            if project.failure:
                return r[m.Infra.MiseToolchainWorkspacePlan].from_failure(project)
            projects.append(project.value)
        return r[m.Infra.MiseToolchainWorkspacePlan].ok(
            m.Infra.MiseToolchainWorkspacePlan(layout=layout, projects=tuple(projects))
        )

    @staticmethod
    def _project_state(
        layout: m.Infra.MiseToolchainProjectLayout,
        config_plan: m.Infra.CodegenFilePlan | None,
    ) -> p.Result[m.Infra.MiseToolchainProjectState]:
        config_state = files.read_state(
            layout.artifacts.config, required=config_plan is None
        )
        if config_state.failure:
            return r[m.Infra.MiseToolchainProjectState].from_failure(config_state)
        if config_plan is None:
            current_sources = (
                FlextInfraUtilitiesProjectManagedArtifacts.snapshot_config_sources(
                    layout.root
                )
            )
            if current_sources.failure:
                return r[m.Infra.MiseToolchainProjectState].from_failure(
                    current_sources
                )
            if config_state.value.content is None:
                return r[m.Infra.MiseToolchainProjectState].fail(
                    f"committed Mise configuration is absent: {layout.artifacts.config}"
                )
            replacement_content = config_state.value.content
            config_sources = current_sources.value
        else:
            if config_plan.absent or config_plan.blocked or not config_plan.rendered:
                return r[m.Infra.MiseToolchainProjectState].fail(
                    f"invalid Mise configuration plan: {config_plan.path}"
                )
            expected_sha256 = u.Cli.sha256_content(config_plan.rendered)
            if config_plan.expected_sha256 != expected_sha256:
                return r[m.Infra.MiseToolchainProjectState].fail(
                    f"Mise configuration plan digest differs: {config_plan.path}"
                )
            replacement_content = config_plan.rendered.encode(c.Cli.ENCODING_DEFAULT)
            config_sources = config_plan.source_states
        artifacts: list[m.Cli.AtomicFileState] = []
        for path in (
            layout.artifacts.unix_launcher,
            layout.artifacts.windows_launcher,
            layout.artifacts.lock,
        ):
            state = files.read_state(path, required=False)
            if state.failure:
                return r[m.Infra.MiseToolchainProjectState].from_failure(state)
            artifacts.append(state.value)
        artifact_set = m.Infra.MiseToolchainArtifactSet(
            unix_launcher=artifacts[0], windows_launcher=artifacts[1], lock=artifacts[2]
        )
        native_seed = (
            artifact_set.windows_launcher
            if os.name == "nt"
            else artifact_set.unix_launcher
        )
        if layout.selector == "." and native_seed.content is None:
            return r[m.Infra.MiseToolchainProjectState].fail(
                f"native committed Mise seed is missing: {native_seed.path}"
            )
        return r[m.Infra.MiseToolchainProjectState].ok(
            m.Infra.MiseToolchainProjectState(
                layout=layout,
                config=m.Infra.MiseToolchainConfigState(
                    before=config_state.value,
                    replacement_content=replacement_content,
                    replacement_mode=files.CONFIG_SPEC[1],
                    sources=config_sources,
                ),
                artifacts=artifact_set,
            )
        )

    def _project_layout(
        self, scope_root: Path, selector: str
    ) -> p.Result[m.Infra.MiseToolchainProjectLayout]:
        root = self._project_root(scope_root, selector)
        if root.failure:
            return r[m.Infra.MiseToolchainProjectLayout].from_failure(root)
        state_root = self._state_root(root.value)
        if state_root.failure:
            return r[m.Infra.MiseToolchainProjectLayout].from_failure(state_root)
        return r[m.Infra.MiseToolchainProjectLayout].ok(
            m.Infra.MiseToolchainProjectLayout(
                selector=selector,
                root=root.value,
                transaction_root=state_root.value / files.TRANSACTION_DIR_NAME,
                artifacts=m.Infra.MiseToolchainArtifactPaths(
                    config=root.value / files.CONFIG_SPEC[0],
                    unix_launcher=root.value / files.ARTIFACT_NAMES[0],
                    windows_launcher=root.value / files.ARTIFACT_NAMES[1],
                    lock=root.value / files.ARTIFACT_NAMES[2],
                ),
            )
        )

    def _project_root(self, scope_root: Path, selector: str) -> p.Result[Path]:
        relative = Path(selector)
        if selector != "." and (
            relative.is_absolute()
            or relative.as_posix() != selector
            or ".." in relative.parts
        ):
            return r[Path].fail(f"unsafe Mise project selector: {selector}")
        cursor = scope_root.absolute()
        for part in relative.parts:
            cursor /= part
            physical = self._physical_directory(cursor)
            if physical.failure:
                return r[Path].from_failure(physical)
        if not cursor.is_relative_to(scope_root.absolute()):
            return r[Path].fail(f"Mise project escapes workspace: {selector}")
        return r[Path].ok(cursor)

    def _state_root(self, root: Path) -> p.Result[Path]:
        cursor = root.absolute()
        for part in files.STATE_DIRECTORY.parts:
            cursor /= part
            if not cursor.exists() and not cursor.is_symlink():
                continue
            physical = self._physical_directory(cursor)
            if physical.failure:
                return r[Path].from_failure(physical)
        return r[Path].ok(cursor)

    @staticmethod
    def _physical_directory(path: Path) -> p.Result[bool]:
        try:
            state = path.lstat()
        except OSError as exc:
            return r[bool].fail_op(f"inspect Mise directory {path}", exc)
        reparse = getattr(state, "st_file_attributes", 0) & getattr(
            stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0
        )
        if not stat.S_ISDIR(state.st_mode) or reparse:
            return r[bool].fail(f"Mise directory is not physical: {path}")
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraMiseWorkspacePlanner"]
