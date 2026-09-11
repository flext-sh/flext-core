"""Rope-only object inventory helpers for workspace-wide census."""

from __future__ import annotations

from operator import itemgetter
from pathlib import Path

from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore
from flext_infra._utilities.rope_imports import FlextInfraUtilitiesRopeImports
from flext_infra._utilities.rope_runtime import FlextInfraUtilitiesRopeRuntime
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.protocols import p
from flext_infra.typings import t


class FlextInfraUtilitiesRopeInventory:
    """Generic Rope-only inventory helpers for Python objects."""

    get_pymodule = staticmethod(FlextInfraUtilitiesRopeCore.get_pymodule)

    @classmethod
    def objects(
        cls,
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        *,
        module_entry: m.Infra.RopeModuleIndexEntry | None,
        convention: m.Infra.RopeModuleConvention,
        include_local_scopes: bool,
        include_references: bool = True,
        rope_workspace: p.Infra.RopeWorkspaceDsl | None = None,
    ) -> tuple[m.Infra.Census.Object, ...]:
        """Return all same-file defined objects for one Rope module."""
        try:
            pymodule = cls.get_pymodule(rope_project, resource)
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors() as exc:
            msg = (
                "rope inventory failed to load "
                f"{resource.path}: {type(exc).__name__}: {exc!s}"
            )
            raise RuntimeError(msg) from exc
        except (
            RecursionError,
            SyntaxError,
            ValueError,
            *FlextInfraUtilitiesRopeRuntime.rope_error_types(),
        ) as exc:
            msg = (
                "rope inventory failed to load "
                f"{resource.path}: {type(exc).__name__}: {exc!s}"
            )
            raise RuntimeError(msg) from exc
        source = resource.read()
        items: t.MutableSequenceOf[m.Infra.Census.Object] = []
        module_scope = pymodule.get_scope()
        if module_scope is None:
            msg = f"rope inventory scope unavailable for {resource.path}"
            raise TypeError(msg)
        child_scopes = tuple(module_scope.get_scopes())
        for name, pyname in cls._sorted_module_names(pymodule, resource):
            record_options = m.Infra.RopeInventoryRecordInput(
                rope_project=rope_project,
                resource=resource,
                source=source,
                name=name,
                pyname=pyname,
                module_name=module_entry.module_name
                if module_entry is not None
                else "",
                project_name=convention.project_layout.project_name
                if convention.project_layout is not None
                else convention.file_path.parent.name,
                convention=convention,
                scope_chain=(),
                class_chain=(),
                child_scope=cls._child_scope_for(child_scopes, pyname),
                rope_workspace=rope_workspace,
            )
            record = cls._record(record_options, include_references=include_references)
            if record is None:
                continue
            items.append(record)
            items.extend(
                cls._child_scope_objects(
                    record=record,
                    child_scope=record_options.child_scope,
                    record_options=record_options,
                    include_local_scopes=include_local_scopes,
                    include_references=include_references,
                )
            )
        return tuple(items)

    @classmethod
    def _scope_objects(
        cls,
        scope: p.Infra.RopeScopeDsl,
        *,
        parent_options: m.Infra.RopeInventoryRecordInput,
        include_references: bool = True,
    ) -> tuple[m.Infra.Census.Object, ...]:
        """Scope objects."""
        items: t.MutableSequenceOf[m.Infra.Census.Object] = []
        child_scopes = tuple(scope.get_scopes())
        for name, pyname in cls._sorted_scope_names(scope, parent_options.resource):
            child_scope = cls._child_scope_for(child_scopes, pyname)
            record_options = parent_options.model_copy(
                update={"name": name, "pyname": pyname, "child_scope": child_scope}
            )
            record = cls._record(record_options, include_references=include_references)
            if record is None:
                continue
            items.append(record)
            items.extend(
                cls._child_scope_objects(
                    record=record,
                    child_scope=child_scope,
                    record_options=record_options,
                    include_references=include_references,
                )
            )
        return tuple(items)

    @classmethod
    def _child_scope_objects(
        cls,
        *,
        record: m.Infra.Census.Object,
        child_scope: p.Infra.RopeScopeDsl | None,
        record_options: m.Infra.RopeInventoryRecordInput,
        include_local_scopes: bool = True,
        include_references: bool = True,
    ) -> tuple[m.Infra.Census.Object, ...]:
        """Child scope objects."""
        if (
            not include_local_scopes
            or record.kind not in {"class", "function", "method"}
            or child_scope is None
        ):
            return ()
        return cls._scope_objects(
            scope=child_scope,
            parent_options=cls._descend_options(record_options, record),
            include_references=include_references,
        )

    @staticmethod
    def _descend_options(
        parent_options: m.Infra.RopeInventoryRecordInput, record: m.Infra.Census.Object
    ) -> m.Infra.RopeInventoryRecordInput:
        """Descend options."""
        result: m.Infra.RopeInventoryRecordInput = parent_options.model_copy(
            update={
                "scope_chain": tuple(
                    part for part in record.scope_path.split(".") if part
                ),
                "class_chain": tuple(
                    part for part in record.class_path.split(".") if part
                ),
            }
        )
        return result

    @staticmethod
    def _sorted_module_names(
        pymodule: t.Infra.RopePyModule, resource: t.Infra.RopeResource
    ) -> tuple[tuple[str, t.Infra.RopePyName], ...]:
        """Sorted module names."""
        return FlextInfraUtilitiesRopeInventory._sorted_names(
            pymodule.get_attributes(), resource
        )

    @staticmethod
    def _sorted_scope_names(
        scope: p.Infra.RopeScopeDsl, resource: t.Infra.RopeResource
    ) -> tuple[tuple[str, t.Infra.RopePyName], ...]:
        """Sorted scope names."""
        return FlextInfraUtilitiesRopeInventory._sorted_names(
            scope.get_names(), resource
        )

    @staticmethod
    def _sorted_names(
        names: t.MappingKV[str, t.Infra.RopePyName], resource: t.Infra.RopeResource
    ) -> tuple[tuple[str, t.Infra.RopePyName], ...]:
        """Sorted names."""
        candidates: list[tuple[int, str, t.Infra.RopePyName]] = []
        for name, pyname in names.items():
            if FlextInfraUtilitiesRopeRuntime.is_imported_name(pyname):
                continue
            line = FlextInfraUtilitiesRopeInventory._definition_line(pyname, resource)
            if line is None:
                continue
            candidates.append((line, name, pyname))
        return tuple(
            (name, pyname) for _, name, pyname in sorted(candidates, key=itemgetter(0))
        )

    @classmethod
    def _record(
        cls, options: m.Infra.RopeInventoryRecordInput, *, include_references: bool
    ) -> m.Infra.Census.Object | None:
        """Record."""
        line = cls._definition_line(options.pyname, options.resource)
        if line is None:
            return None
        expected_alias = options.convention.module_policy.expected_alias or ""
        kind = (
            "assignment"
            if not options.scope_chain and options.name == expected_alias
            else cls._kind_for(
                options.pyname,
                class_chain=options.class_chain,
                scope_chain=options.scope_chain,
                name=options.name,
            )
        )
        if kind == "parameter" and options.name in {"self", "cls"}:
            return None
        is_facade_member = cls._is_facade_member(
            options.convention, name=options.name, scope_chain=options.scope_chain
        )
        if (
            include_references
            and not is_facade_member
            and not options.name.startswith("_")
        ):
            runtime_reference_sites, script_reference_sites = cls._reference_sites(
                options.rope_project,
                options.resource,
                source=options.source,
                module_name=options.module_name,
                name=options.name,
                line=line,
                rope_workspace=options.rope_workspace,
            )
        else:
            runtime_reference_sites = ()
            script_reference_sites = ()
        references_count = len(runtime_reference_sites) + len(script_reference_sites)
        scope_path = ".".join((*options.scope_chain, options.name))
        class_path = (
            ".".join((*options.class_chain, options.name))
            if kind == "class"
            else ".".join(options.class_chain)
        )
        return m.Infra.Census.Object(
            name=options.name,
            kind=kind,
            file_path=str(options.convention.file_path),
            line=line,
            project=options.project_name,
            class_path=class_path,
            module_name=options.module_name,
            scope_path=scope_path,
            actual_tier=cls._actual_tier(options.convention),
            expected_tier=cls._expected_tier(options.convention, kind=kind),
            is_facade_member=is_facade_member,
            references_count=references_count,
            runtime_references_count=len(runtime_reference_sites),
            script_references_count=len(script_reference_sites),
            runtime_reference_sites=runtime_reference_sites,
            script_reference_sites=script_reference_sites,
            fingerprint=cls._fingerprint(
                options.source,
                name=options.name,
                line=line,
                child_scope=options.child_scope,
            ),
        )

    @staticmethod
    def _definition_line(
        pyname: t.Infra.RopePyName, resource: t.Infra.RopeResource
    ) -> int | None:
        """Definition line."""
        location = pyname.get_definition_location()
        module, line = location
        origin = module.get_resource() if module is not None else None
        if line is None or origin is None or origin.path != resource.path:
            return None
        result: int = line
        return result

    @staticmethod
    def _child_scope_for(
        scopes: t.SequenceOf[p.Infra.RopeScopeDsl], pyname: t.Infra.RopePyName
    ) -> p.Infra.RopeScopeDsl | None:
        """Child scope for."""
        location = pyname.get_definition_location()
        _, line = location
        if line is None:
            return None
        scope = next((scope for scope in scopes if scope.get_start() == line), None)
        if scope is not None:
            validated_existing_scope: p.Infra.RopeScopeDsl = scope
            return validated_existing_scope
        if FlextInfraUtilitiesRopeRuntime.is_assigned_name(
            pyname
        ) or FlextInfraUtilitiesRopeRuntime.is_parameter_name(pyname):
            return None
        getter = getattr(pyname.get_object(), "get_scope", None)
        candidate = getter() if callable(getter) else None
        if not isinstance(candidate, p.Infra.RopeScopeDsl):
            return None
        validated_scope: p.Infra.RopeScopeDsl = candidate
        return validated_scope

    @staticmethod
    def _kind_for(
        pyname: t.Infra.RopePyName,
        *,
        class_chain: t.StrSequence,
        scope_chain: t.StrSequence,
        name: str,
    ) -> str:
        """Kind for."""
        result: str
        if FlextInfraUtilitiesRopeRuntime.is_parameter_name(pyname):
            result = "parameter"
        elif FlextInfraUtilitiesRopeRuntime.is_assigned_name(pyname):
            if class_chain and len(scope_chain) == len(class_chain):
                result = "attribute"
            elif scope_chain:
                result = "local" if not name.isupper() else "constant"
            elif name.isupper():
                result = "constant"
            else:
                result = "assignment"
        else:
            obj = pyname.get_object()
            if FlextInfraUtilitiesRopeRuntime.is_abstract_class(obj):
                result = "class"
            elif FlextInfraUtilitiesRopeRuntime.is_py_function(obj):
                result = (
                    "method"
                    if class_chain and len(scope_chain) == len(class_chain)
                    else "function"
                )
            elif class_chain and len(scope_chain) == len(class_chain):
                result = "attribute"
            elif scope_chain:
                result = "local" if not name.isupper() else "constant"
            elif (
                FlextInfraUtilitiesRopeRuntime.is_defined_name(pyname)
                and name.isupper()
            ):
                result = "constant"
            else:
                result = "assignment"
        return result

    @staticmethod
    def _reference_sites(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        *,
        source: str,
        name: str,
        line: int,
        rope_workspace: p.AttributeProbe | None = None,
        module_name: str,
    ) -> tuple[
        tuple[m.Infra.Census.ReferenceSite, ...],
        tuple[m.Infra.Census.ReferenceSite, ...],
    ]:
        """Collect the reference sites for a symbol."""
        lines = source.splitlines(keepends=True)
        offset = FlextInfraUtilitiesRopeCore.find_identifier_offset_in_lines(
            lines, line=line, symbol=name
        )
        if offset is None:
            return ((), ())
        definition_path = FlextInfraUtilitiesRopeCore.resource_file_path(
            rope_project, resource
        )
        search_resources: tuple[t.Infra.RopeResource, ...] | None = None
        if rope_workspace is not None and definition_path is not None:
            fast_path = (
                FlextInfraUtilitiesRopeInventory._fast_reference_sites_from_index(
                    rope_workspace,
                    name=name,
                    definition_path=definition_path,
                    line=line,
                )
            )
            if fast_path is not None:
                return fast_path
            dependent_import_targets = (
                (module_name, f"{module_name}.{name}")
                if module_name
                and FlextInfraUtilitiesRopeInventory._reference_surface(definition_path)
                != c.Infra.DEFAULT_SRC_DIR
                else ()
            )
            search_resources = FlextInfraUtilitiesRopeImports.indexed_search_resources(
                rope_workspace,
                resource=resource,
                name=name,
                definition_path=definition_path,
                dependent_import_targets=dependent_import_targets,
            )
        hits = FlextInfraUtilitiesRopeImports.find_occurrences(
            rope_project, resource, offset, resources=search_resources
        )
        runtime_reference_sites: list[m.Infra.Census.ReferenceSite] = []
        script_reference_sites: list[m.Infra.Census.ReferenceSite] = []
        seen_sites: set[tuple[str, int, str]] = set()
        skipped_definition = False
        for hit in hits:
            if (
                not skipped_definition
                and FlextInfraUtilitiesRopeInventory._is_definition_occurrence(
                    hit, definition_path=definition_path, line=line, offset=offset
                )
            ):
                skipped_definition = True
                continue
            reference_site = FlextInfraUtilitiesRopeInventory._reference_site(hit)
            if reference_site is None:
                continue
            if Path(reference_site.file_path).name == c.Infra.INIT_PY:
                continue
            site_key = (
                reference_site.file_path,
                reference_site.line,
                reference_site.surface,
            )
            if site_key in seen_sites:
                continue
            seen_sites.add(site_key)
            # Tests and examples are deliberately outside production reachability.
            if reference_site.surface in {c.Infra.DIR_TESTS, c.Infra.DIR_EXAMPLES}:
                continue
            if reference_site.surface == c.Infra.DIR_SCRIPTS:
                script_reference_sites.append(reference_site)
                continue
            runtime_reference_sites.append(reference_site)
        if not skipped_definition and definition_path is not None:
            surface = FlextInfraUtilitiesRopeInventory._reference_surface(
                definition_path
            )
            if surface == c.Infra.DIR_SCRIPTS:
                FlextInfraUtilitiesRopeInventory._discard_definition_site(
                    script_reference_sites, definition_path=definition_path, line=line
                )
            elif surface not in {c.Infra.DIR_TESTS, c.Infra.DIR_EXAMPLES}:
                FlextInfraUtilitiesRopeInventory._discard_definition_site(
                    runtime_reference_sites, definition_path=definition_path, line=line
                )
        has_reference_sites = bool(runtime_reference_sites or script_reference_sites)
        if (
            not has_reference_sites
            and rope_workspace is not None
            and definition_path is not None
            and module_name
        ):
            (fallback_runtime_reference_sites, fallback_script_reference_sites) = (
                FlextInfraUtilitiesRopeInventory._fallback_reference_sites_from_index(
                    rope_workspace,
                    definition_path=definition_path,
                    module_name=module_name,
                    name=name,
                )
            )
            runtime_reference_sites.extend(fallback_runtime_reference_sites)
            script_reference_sites.extend(fallback_script_reference_sites)
        return (tuple(runtime_reference_sites), tuple(script_reference_sites))

    @staticmethod
    def _fallback_reference_sites_from_index(
        rope_workspace: p.AttributeProbe,
        *,
        definition_path: Path,
        module_name: str,
        name: str,
    ) -> tuple[
        tuple[m.Infra.Census.ReferenceSite, ...],
        tuple[m.Infra.Census.ReferenceSite, ...],
    ]:
        """Fallback reference sites from indexed dependent modules.

        Rope can miss external class-name uses for some non-src surfaces even when
        semantic import analysis already proves the dependent module imports the
        exact symbol. In that narrow case, synthesize reference sites from the
        name index restricted to the semantic dependents only.
        """
        name_index_getter = getattr(rope_workspace, "name_index", None)
        import_dependents_getter = getattr(rope_workspace, "import_dependents", None)
        if name_index_getter is None or not callable(import_dependents_getter):
            return ((), ())
        dependent_paths: set[str] = set()
        for import_target in (module_name, f"{module_name}.{name}"):
            dependent_paths_raw = import_dependents_getter(import_target)
            if not isinstance(dependent_paths_raw, tuple):
                msg = (
                    "rope import_dependents returned non-tuple for "
                    f"{import_target}: {type(dependent_paths_raw).__name__}"
                )
                raise TypeError(msg)
            for path in dependent_paths_raw:
                if not isinstance(path, Path):
                    msg = (
                        "rope import_dependents returned invalid path for "
                        f"{import_target}: {type(path).__name__}"
                    )
                    raise TypeError(msg)
                dependent_paths.add(
                    FlextInfraUtilitiesRopeInventory._normalize_file_path(
                        path.resolve()
                    )
                )
        if not dependent_paths:
            return ((), ())
        normalized_definition = FlextInfraUtilitiesRopeInventory._normalize_file_path(
            definition_path.resolve()
        )
        runtime_reference_sites: list[m.Infra.Census.ReferenceSite] = []
        script_reference_sites: list[m.Infra.Census.ReferenceSite] = []
        seen_sites: set[tuple[str, int, str]] = set()
        for path, surface, lines in name_index_getter().get(name, ()):
            normalized_path = FlextInfraUtilitiesRopeInventory._normalize_file_path(
                path.resolve()
            )
            if path.name == c.Infra.INIT_PY or normalized_path == normalized_definition:
                continue
            if normalized_path not in dependent_paths:
                continue
            for line in lines:
                site_key = (normalized_path, line, surface)
                if site_key in seen_sites:
                    continue
                seen_sites.add(site_key)
                reference_site = m.Infra.Census.ReferenceSite(
                    file_path=normalized_path, line=line, surface=surface
                )
                if surface in {c.Infra.DIR_TESTS, c.Infra.DIR_EXAMPLES}:
                    continue
                if surface == c.Infra.DIR_SCRIPTS:
                    script_reference_sites.append(reference_site)
                    continue
                runtime_reference_sites.append(reference_site)
        return (tuple(runtime_reference_sites), tuple(script_reference_sites))

    @staticmethod
    def _fast_reference_sites_from_index(
        rope_workspace: p.AttributeProbe, *, name: str, definition_path: Path, line: int
    ) -> (
        tuple[
            tuple[m.Infra.Census.ReferenceSite, ...],
            tuple[m.Infra.Census.ReferenceSite, ...],
        ]
        | None
    ):
        """Fast-path reference classification from pre-scanned workspace text index.

        Only short-circuits rope when the index shows the symbol has ZERO
        external-file occurrences (truly unused candidate). For any symbol
        with external references the caller falls back to rope's semantic
        ``find_occurrences`` to correctly handle intra-module ``__all__``
        literals, decorator references, and same-name collisions.
        """
        name_index_getter = getattr(rope_workspace, "name_index", None)
        if name_index_getter is None:
            return None
        occurrences = name_index_getter().get(name, ())
        if not occurrences:
            return ((), ())
        resolved_definition = definition_path.resolve()
        has_same_file_non_definition = any(
            path.resolve() == resolved_definition
            and any(occurrence_line != line for occurrence_line in lines)
            for path, _surface, lines in occurrences
        )
        if has_same_file_non_definition:
            return None
        has_external = any(
            path.resolve() != resolved_definition and path.name != c.Infra.INIT_PY
            for path, _surface, _lines in occurrences
        )
        if has_external:
            return None
        return ((), ())

    @staticmethod
    def _location_file_path(location: t.Infra.RopeLocation) -> Path | None:
        """Location file path."""
        return FlextInfraUtilitiesRopeImports.location_file_path(location)

    @staticmethod
    def _location_line(location: t.Infra.RopeLocation) -> int:
        """Location line."""
        line = getattr(location, "lineno", None)
        return line if isinstance(line, int) and line >= 0 else 0

    @staticmethod
    def _reference_site(
        location: t.Infra.RopeLocation,
    ) -> m.Infra.Census.ReferenceSite | None:
        """Build a reference site from a rope location."""
        file_path = FlextInfraUtilitiesRopeInventory._location_file_path(location)
        if file_path is None:
            return None
        return m.Infra.Census.ReferenceSite(
            file_path=FlextInfraUtilitiesRopeInventory._normalize_file_path(file_path),
            line=FlextInfraUtilitiesRopeInventory._location_line(location),
            surface=FlextInfraUtilitiesRopeInventory._reference_surface(file_path),
        )

    @staticmethod
    def _discard_definition_site(
        sites: list[m.Infra.Census.ReferenceSite], *, definition_path: Path, line: int
    ) -> None:
        """Discard definition site."""
        definition_key = (
            FlextInfraUtilitiesRopeInventory._normalize_file_path(definition_path),
            line,
        )
        match_index = next(
            (
                index
                for index, site in enumerate(sites)
                if (site.file_path, site.line) == definition_key
            ),
            None,
        )
        if match_index is not None:
            sites.pop(match_index)

    @staticmethod
    def _normalize_file_path(file_path: Path) -> str:
        """Normalize file path."""
        return str(file_path.resolve() if file_path.is_absolute() else file_path)

    @staticmethod
    def _is_definition_occurrence(
        location: t.Infra.RopeLocation,
        *,
        definition_path: Path | None,
        line: int,
        offset: int,
    ) -> bool:
        """Is definition occurrence."""
        location_path = FlextInfraUtilitiesRopeInventory._location_file_path(location)
        if (
            definition_path is not None
            and location_path is not None
            and location_path != definition_path
        ):
            return False
        location_line = getattr(location, "lineno", None)
        if isinstance(location_line, int) and location_line != line:
            return False
        location_offset = getattr(location, "offset", None)
        if isinstance(location_offset, int):
            return location_offset == offset
        return bool(
            definition_path is not None
            and location_path == definition_path
            and location_line == line
        )

    @staticmethod
    def _reference_surface(file_path: Path | None) -> str:
        """Return the reference surface for a file path."""
        default_src: str = c.Infra.DEFAULT_SRC_DIR
        if file_path is None:
            return default_src
        parts = set(file_path.parts)
        tests_dir: str = c.Infra.DIR_TESTS
        if tests_dir in parts:
            return tests_dir
        examples_dir: str = c.Infra.DIR_EXAMPLES
        if examples_dir in parts:
            return examples_dir
        scripts_dir: str = c.Infra.DIR_SCRIPTS
        if scripts_dir in parts:
            return scripts_dir
        return default_src

    @staticmethod
    def _fingerprint(
        source: str, *, name: str, line: int, child_scope: p.Infra.RopeScopeDsl | None
    ) -> str:
        """Fingerprint."""
        start = max(1, line)
        end = child_scope.get_end() if child_scope is not None else start
        end = max(end, start)
        lines = source.splitlines()
        snippet = "\n".join(lines[start - 1 : end])
        return " ".join(snippet.replace(name, "<name>").split())

    @staticmethod
    def _actual_tier(convention: m.Infra.RopeModuleConvention) -> str:
        """Actual tier."""
        policy = convention.module_policy
        expected: str = policy.expected_family or ""
        if expected:
            return expected
        if "services" in convention.relative_path.parts:
            return "Services"
        stem: str = convention.file_path.stem
        return stem

    @staticmethod
    def _expected_tier(convention: m.Infra.RopeModuleConvention, *, kind: str) -> str:
        """Return the expected tier for a module convention."""
        expected: str = convention.module_policy.expected_family or ""
        if expected:
            return expected
        if kind == "constant":
            constants: str = c.Infra.FAMILY_SUFFIXES.get("c", "Constants")
            return constants
        return FlextInfraUtilitiesRopeInventory._actual_tier(convention)

    @staticmethod
    def _is_facade_member(
        convention: m.Infra.RopeModuleConvention,
        *,
        name: str,
        scope_chain: t.StrSequence,
    ) -> bool:
        """Is facade member."""
        if scope_chain:
            return False
        layout = convention.project_layout
        family = convention.module_policy.expected_family or ""
        alias = convention.module_policy.expected_alias or ""
        if name == alias:
            return True
        return bool(
            layout is not None and family and name == f"{layout.class_stem}{family}"
        )


__all__: list[str] = ["FlextInfraUtilitiesRopeInventory"]
