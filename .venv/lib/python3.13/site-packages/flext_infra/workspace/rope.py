"""Public Rope workspace DSL and facade mixin."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Annotated, ClassVar, Self, override

from flext_core import r
from flext_infra import c, m, p, t, u
from flext_infra.base import s


class FlextInfraRopeWorkspace(s[m.Infra.RopeWorkspaceSession]):
    """Open one shared Rope workspace with cached public DSL methods."""

    _IDENTIFIER_PATTERN: ClassVar[t.Infra.RegexPattern] = c.Infra.IDENTIFIER_PATTERN

    rope_repository_root_override: Annotated[
        Path | None,
        m.Field(description="Optional Rope project root; defaults to repository_root"),
    ] = None

    _rope_repository_root: Path
    _rope_project: t.Infra.RopeProject | None = u.PrivateAttr(
        default_factory=lambda: None
    )
    _workspace_index: m.Infra.RopeWorkspaceIndex | None = u.PrivateAttr(
        default_factory=lambda: None
    )
    _codegen_projects: tuple[p.Infra.ProjectInfo, ...] | None = u.PrivateAttr(
        default_factory=lambda: None
    )
    _project_layout_cache: dict[str, m.Infra.RopeProjectLayout | None] = u.PrivateAttr(
        default_factory=dict
    )
    _package_context_cache: dict[str, m.Infra.LazyInitPackageContext] = u.PrivateAttr(
        default_factory=dict
    )
    _module_policy_cache: dict[tuple[str, str, str], m.Infra.NamespaceModulePolicy] = (
        u.PrivateAttr(default_factory=dict)
    )
    _module_convention_cache: dict[str, m.Infra.RopeModuleConvention] = u.PrivateAttr(
        default_factory=dict
    )
    _module_object_cache: dict[
        tuple[str, bool, bool], tuple[m.Infra.Census.Object, ...]
    ] = u.PrivateAttr(default_factory=dict)
    _resource_cache: dict[str, t.Infra.RopeResource | None] = u.PrivateAttr(
        default_factory=dict
    )
    _name_index: dict[str, tuple[tuple[Path, str, tuple[int, ...]], ...]] | None = (
        u.PrivateAttr(default_factory=lambda: None)
    )
    _import_dependents_index: dict[str, tuple[Path, ...]] | None = u.PrivateAttr(
        default_factory=lambda: None
    )

    @override
    def model_post_init(self, __context: t.ScalarMapping | None, /) -> None:
        """Resolve the canonical Rope root once for the full session."""
        super().model_post_init(__context)
        self._rope_repository_root = (
            self.rope_repository_root_override
            or u.Infra.rope_repository_root(self.repository_root)
        )

    @classmethod
    def open_workspace(
        cls, repository_root: Path, *, rope_repository_root: Path | None = None
    ) -> Self:
        """Create one ready-to-use Rope workspace session."""
        # NOTE (multi-agent, flext-wkii.17.24): scan policy is owned only by the
        # validated config singleton, never copied into a session.
        resolved_rope_root = rope_repository_root or u.Infra.rope_repository_root(
            repository_root
        )
        workspace = cls(
            repository_root=repository_root,
            rope_repository_root_override=resolved_rope_root,
        )
        _ = workspace.rope_project
        return workspace

    @property
    def rope_repository_root(self) -> Path:
        """Canonical root used for the shared Rope project."""
        return self._rope_repository_root

    @property
    def rope_project(self) -> t.Infra.RopeProject:
        """Shared Rope project, opening it lazily once."""
        rope_project = self._rope_project
        if rope_project is None:
            started_at = perf_counter()
            u.Cli.info(f"rope: opening workspace at {self._rope_repository_root}")
            rope_project = u.Infra.init_rope_project(self._rope_repository_root)
            self._rope_project = rope_project
            u.Cli.info(f"rope: workspace ready in {perf_counter() - started_at:.2f}s")
        return rope_project

    @property
    def workspace_index(self) -> m.Infra.RopeWorkspaceIndex:
        """Cached workspace index for the shared Rope project."""
        workspace_index = self._workspace_index
        if workspace_index is None:
            started_at = perf_counter()
            u.Cli.info(
                f"rope: indexing python workspace at {self._rope_repository_root}"
            )
            workspace_index = u.Infra.index_rope_workspace(
                self.rope_project, self._rope_repository_root
            )
            self._workspace_index = workspace_index
            u.Cli.info(
                "rope: indexed "
                f"{len(workspace_index.package_dirs)} package dirs and "
                f"{len(workspace_index.modules_by_path)} modules in "
                f"{perf_counter() - started_at:.2f}s"
            )
        return workspace_index

    @override
    def execute(self) -> p.Result[m.Infra.RopeWorkspaceSession]:
        """Materialize the public Rope session snapshot."""
        snapshot: m.Infra.RopeWorkspaceSession = self.session_snapshot()
        return r[m.Infra.RopeWorkspaceSession].ok(snapshot)

    def session_snapshot(self) -> m.Infra.RopeWorkspaceSession:
        """Return the current public Rope session state."""
        return m.Infra.RopeWorkspaceSession(
            repository_root=self.repository_root,
            rope_repository_root=self._rope_repository_root,
            workspace_index=self.workspace_index,
        )

    def refresh(
        self, *, preserve_indexes: bool = False, validate_project: bool = True
    ) -> m.Infra.RopeWorkspaceSession:
        """Invalidate Rope caches without reopening the Rope project.

        ``preserve_indexes=True`` is reserved for flows that temporarily wrote
        files and already restored the original on-disk content before the
        refresh runs. ``validate_project=False`` is reserved for those reverted
        preview flows so cleanup does not rescan an already-restored project.
        """
        if validate_project and self._rope_project is not None:
            self._rope_project.validate()
        self._package_context_cache.clear()
        self._module_policy_cache.clear()
        self._module_convention_cache.clear()
        self._module_object_cache.clear()
        self._resource_cache.clear()
        if not preserve_indexes:
            self._name_index = None
            self._import_dependents_index = None
        return self.session_snapshot()

    def reload(self) -> m.Infra.RopeWorkspaceSession:
        """Reopen the shared Rope project and drop all transient caches."""
        self.close()
        _ = self.rope_project
        return self.session_snapshot()

    def resource(self, file_path: Path) -> t.Infra.RopeResource | None:
        """Return one cached Rope resource for the requested file path."""
        cache_key = str(file_path.resolve())
        cached = self._resource_cache.get(cache_key)
        if cache_key in self._resource_cache:
            return cached
        resource = u.Infra.get_resource_from_path(self.rope_project, file_path)
        self._resource_cache[cache_key] = resource
        return resource

    def module(self, file_path: Path) -> m.Infra.RopeModuleIndexEntry | None:
        """Return one indexed module entry for the requested file path."""
        raw = self.workspace_index.modules_by_path.get(str(file_path.resolve()))
        if raw is None:
            return None
        validated: m.Infra.RopeModuleIndexEntry = (
            m.Infra.RopeModuleIndexEntry.model_validate(raw)
        )
        return validated

    def package(self, package_dir: Path) -> m.Infra.RopePackageIndexEntry | None:
        """Return one indexed package entry for the requested directory."""
        raw = self.workspace_index.packages_by_dir.get(str(package_dir.resolve()))
        if raw is None:
            return None
        validated: m.Infra.RopePackageIndexEntry = (
            m.Infra.RopePackageIndexEntry.model_validate(raw)
        )
        return validated

    def modules(
        self, *, project_names: t.StrSequence | None = None
    ) -> t.SequenceOf[m.Infra.RopeModuleIndexEntry]:
        """Return sorted module entries, optionally filtered by project names."""

        def module_path(entry: m.Infra.RopeModuleIndexEntry) -> str:
            return entry.file_path.as_posix()

        modules = tuple(
            sorted(self.workspace_index.modules_by_path.values(), key=module_path)
        )
        if not project_names:
            return modules
        project_filter = frozenset(project_names)
        return tuple(
            entry
            for entry in modules
            if entry.project_root is not None
            and entry.project_root.name in project_filter
        )

    def source(self, file_path: Path) -> str:
        """Return one module source snapshot from the active Rope workspace."""
        text: str = self._resource_for(file_path).read()
        return text

    def import_dependents(self, import_target: str) -> tuple[Path, ...]:
        """Return cached module paths that semantically import ``import_target``."""
        if not import_target:
            return ()
        index = self._import_dependents_index
        if index is None:
            dependents: dict[str, set[Path]] = defaultdict(set)
            for module in self.modules():
                file_path = module.file_path.resolve()
                for target in self.semantic(file_path).semantic_imports.values():
                    if not target:
                        continue
                    dependents[target].add(file_path)
            index = {
                target: tuple(sorted(paths)) for target, paths in dependents.items()
            }
            self._import_dependents_index = index
        return index.get(import_target, ())

    def name_index(
        self,
    ) -> t.MappingKV[str, tuple[tuple[Path, str, tuple[int, ...]], ...]]:
        """Return a cached ``{name: ((path, surface, lines), ...)}`` workspace index.

        Built once per workspace session via a single regex scan of every
        indexed ``.py`` module. Short-circuits rope's ``find_occurrences``
        when a symbol's surface distribution alone answers the
        unused classification question.
        """
        if self._name_index is not None:
            return self._name_index
        index: dict[str, list[tuple[Path, str, list[int]]]] = {}
        for entry in self.workspace_index.modules_by_path.values():
            py_file = entry.file_path
            read = u.Cli.files_read_text(py_file)
            if read.failure:
                msg = f"rope name index failed to read {py_file}: {read.error}"
                raise RuntimeError(msg)
            source_text = read.value
            surface = self._reference_surface_for(py_file)
            lines_by_name: dict[str, list[int]] = {}
            for lineno, source_line in enumerate(source_text.splitlines(), start=1):
                for match in self._IDENTIFIER_PATTERN.finditer(source_line):
                    name = match.group(0)
                    lines_by_name.setdefault(name, []).append(lineno)
            for name, line_numbers in lines_by_name.items():
                index.setdefault(name, []).append((py_file, surface, line_numbers))
        self._name_index = {
            name: tuple((path, surface, tuple(lines)) for path, surface, lines in refs)
            for name, refs in index.items()
        }
        return self._name_index

    _SURFACE_DIRS: ClassVar[t.StrSequence] = (
        c.Infra.DIR_TESTS,
        c.Infra.DIR_EXAMPLES,
        c.Infra.DIR_SCRIPTS,
    )

    @classmethod
    def _reference_surface_for(cls, file_path: Path) -> str:
        """Return the reference surface for a file path."""
        for part in file_path.parts:
            if part in cls._SURFACE_DIRS:
                surface: str = part
                return surface
        default_src: str = c.Infra.DEFAULT_SRC_DIR
        return default_src

    def objects(
        self,
        file_path: Path,
        *,
        include_local_scopes: bool = True,
        include_references: bool = True,
    ) -> t.SequenceOf[m.Infra.Census.Object]:
        """Return Rope-only discovered objects for one module path."""
        resolved_file = file_path.resolve()
        cache_key = (str(resolved_file), include_local_scopes, include_references)
        cached = self._module_object_cache.get(cache_key)
        if cached is not None:
            return cached
        objects = u.Infra.objects(
            self.rope_project,
            self._resource_for(resolved_file),
            module_entry=self.module(resolved_file),
            convention=self.convention(resolved_file),
            include_local_scopes=include_local_scopes,
            include_references=include_references,
            rope_workspace=self,
        )
        self._module_object_cache[cache_key] = objects
        return objects

    def projects(self) -> t.SequenceOf[p.Infra.ProjectInfo]:
        """Return the canonical codegen project selection for this workspace."""
        if self._codegen_projects is None:
            projects_result = u.Infra.projects(self.repository_root)
            if projects_result.failure:
                self._codegen_projects = ()
            else:
                discovered_projects: t.SequenceOf[p.Infra.ProjectInfo] = (
                    projects_result.unwrap()
                )
                self._codegen_projects = tuple(discovered_projects)
        return self._codegen_projects

    def layout(self, project_root: Path) -> m.Infra.RopeProjectLayout | None:
        """Return one centralized project layout contract for codegen pipelines."""
        resolved_root = project_root.resolve()
        cache_key = str(resolved_root)
        if cache_key in self._project_layout_cache:
            return self._project_layout_cache[cache_key]
        project_info = next(
            (
                project
                for project in self.projects()
                if project.path.resolve() == resolved_root
            ),
            None,
        )
        layout = u.Infra.layout(resolved_root, project=project_info)
        self._project_layout_cache[cache_key] = layout
        return layout

    def package_context(self, package_dir: Path) -> m.Infra.LazyInitPackageContext:
        """Return one centralized lazy-init package context for a package dir."""
        resolved_dir = package_dir.resolve()
        cache_key = str(resolved_dir)
        cached = self._package_context_cache.get(cache_key)
        if cached is not None:
            return cached
        package_entry = self.package(resolved_dir)
        init_path = resolved_dir / c.Infra.INIT_PY
        current_pkg = package_entry.package_name if package_entry is not None else ""
        generated_init = init_path.is_file() and (
            u.Cli
            .files_read_text(init_path)
            .unwrap()
            .startswith(c.Infra.AUTOGEN_HEADERS)
        )
        context = m.Infra.LazyInitPackageContext(
            pkg_dir=resolved_dir,
            init_path=init_path,
            current_pkg=current_pkg,
            surface=current_pkg.split(".", maxsplit=1)[0] if current_pkg else "",
            generated_init=generated_init,
            importable=bool(current_pkg),
        )
        self._package_context_cache[cache_key] = context
        return context

    def policy(
        self, file_path: Path, *, rel_path: Path | None = None, current_pkg: str = ""
    ) -> m.Infra.NamespaceModulePolicy:
        """Return the centralized naming policy for one module path."""
        resolved_file = file_path.resolve()
        resolved_rel_path = (
            rel_path if rel_path is not None else Path(resolved_file.name)
        )
        cache_key = (str(resolved_file), str(resolved_rel_path), current_pkg)
        cached = self._module_policy_cache.get(cache_key)
        if cached is not None:
            return cached
        policy: m.Infra.NamespaceModulePolicy = u.Infra.policy(
            resolved_file, rel_path=resolved_rel_path, current_pkg=current_pkg
        )
        self._module_policy_cache[cache_key] = policy
        return policy

    def convention(
        self, file_path: Path, *, rel_path: Path | None = None
    ) -> m.Infra.RopeModuleConvention:
        """Return one unified project/package/module convention contract."""
        resolved_file = file_path.resolve()
        cache_key = f"{resolved_file}::{rel_path or ''}"
        cached = self._module_convention_cache.get(cache_key)
        if cached is not None:
            return cached
        module_entry = self.module(resolved_file)
        package_dir = (
            module_entry.package_dir.resolve()
            if module_entry is not None
            else resolved_file.parent.resolve()
        )
        resolved_rel_path = (
            rel_path
            if rel_path is not None
            else resolved_file.relative_to(package_dir)
            if resolved_file.is_relative_to(package_dir)
            else Path(resolved_file.name)
        )
        package_context = self.package_context(package_dir)
        policy = self.policy(
            resolved_file,
            rel_path=resolved_rel_path,
            current_pkg=package_context.current_pkg,
        )
        project_root = (
            module_entry.project_root
            if module_entry is not None and module_entry.project_root is not None
            else u.Infra.project_root(resolved_file)
        )
        project_layout = self.layout(project_root) if project_root is not None else None
        convention = m.Infra.RopeModuleConvention(
            file_path=resolved_file,
            relative_path=resolved_rel_path,
            module_name=module_entry.module_name
            if module_entry is not None
            else resolved_file.stem,
            package_name=package_context.current_pkg,
            package_dir=package_dir,
            package_context=package_context,
            module_policy=policy,
            project_layout=project_layout,
        )
        self._module_convention_cache[cache_key] = convention
        return convention

    def semantic(self, file_path: Path) -> m.Infra.ModuleSemanticState:
        """Return one cached semantic snapshot for a module path."""
        state: m.Infra.ModuleSemanticState = u.Infra.get_module_semantic_state(
            self.rope_project, self._resource_for(file_path)
        )
        return state

    def exports(
        self, file_path: Path, *, export_options: m.Infra.ExportOptions | None = None
    ) -> t.StrSequence:
        """Return public export names for one module path."""
        resolved_export_options = export_options or m.Infra.ExportOptions()
        return u.Infra.get_module_export_names(
            self.rope_project,
            self._resource_for(file_path),
            export_options=resolved_export_options,
        )

    def close(self) -> None:
        """Close the shared Rope project and clear transient caches."""
        if self._rope_project is not None:
            self._rope_project.close()
            self._rope_project = None
        self._workspace_index = None
        self._codegen_projects = None
        self._project_layout_cache.clear()
        self._package_context_cache.clear()
        self._module_policy_cache.clear()
        self._module_convention_cache.clear()
        self._module_object_cache.clear()
        self._resource_cache.clear()
        self._name_index = None
        self._import_dependents_index = None

    def __enter__(self) -> Self:
        """Open the Rope project on context-manager entry."""
        _ = self.rope_project
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: TracebackType | None,
    ) -> None:
        """Close the Rope project on context-manager exit."""
        self.close()

    def _resource_for(self, file_path: Path) -> t.Infra.RopeResource:
        """Require a resource inside the active Rope workspace."""
        resource = self.resource(file_path)
        if resource is not None:
            return resource
        msg = f"path is outside the active rope workspace: {file_path}"
        raise ValueError(msg)


__all__: t.StrSequence = ("FlextInfraRopeWorkspace",)
