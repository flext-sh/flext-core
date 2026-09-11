"""Semantic Rope analysis helpers."""

from __future__ import annotations

import ast
import importlib.util as _importlib_util
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from flext_infra._settings import settings
from flext_infra._utilities.rope_core import FlextInfraUtilitiesRopeCore
from flext_infra._utilities.rope_runtime import FlextInfraUtilitiesRopeRuntime
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t

if TYPE_CHECKING:
    from flext_infra.protocols import p


class FlextInfraUtilitiesRopeAnalysis:
    """Rope-backed semantic analysis helpers."""

    _INSTALL_LAZY_IMPORTS_ARG_INDEX: ClassVar[int] = 2
    _STRING_LITERAL_MIN_LENGTH: ClassVar[int] = 2
    _TRIPLE_QUOTE_LENGTH: ClassVar[int] = 3
    _IMPORT_ALIAS_AS_PARTS: ClassVar[int] = 3

    _parse_project: ClassVar[t.Infra.RopeProject | None] = None
    _SEMANTIC_STATE_CACHE: ClassVar[
        dict[tuple[str, str, int], m.Infra.ModuleSemanticState]
    ] = {}
    _EXPORT_NAMES_CACHE: ClassVar[
        dict[tuple[str, str, int, bool, bool, bool, bool, bool], t.StrSequence]
    ] = {}

    @staticmethod
    def _resource_cache_key(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> tuple[str, str, int]:
        """Resource cache key."""
        file_path = FlextInfraUtilitiesRopeCore.resource_file_path(
            rope_project, resource
        )
        mtime_ns = (
            file_path.stat().st_mtime_ns
            if file_path is not None and file_path.exists()
            else 0
        )
        project_root = getattr(getattr(rope_project, "root", None), "real_path", "")
        return (str(project_root), resource.path, mtime_ns)

    @staticmethod
    def package_name_for_module(
        module_name: str, resource: t.Infra.RopeResource
    ) -> str:
        """Return the dotted package prefix for one Rope module."""
        return FlextInfraUtilitiesRopeAnalysis._package_name_for_module(
            module_name, resource
        )

    @staticmethod
    def _package_name_for_module(
        module_name: str, resource: t.Infra.RopeResource
    ) -> str:
        """Package name for module."""
        if (
            resource.path.endswith(f"/{c.Infra.INIT_PY}")
            or resource.path == c.Infra.INIT_PY
        ):
            return module_name
        return module_name.rsplit(".", maxsplit=1)[0] if "." in module_name else ""

    @staticmethod
    def resolve_import_module(
        *, current_package: str, module_name: str, level: int
    ) -> str:
        """Resolve one declared import to an absolute module name."""
        return FlextInfraUtilitiesRopeAnalysis._resolve_import_module(
            current_package=current_package, module_name=module_name, level=level
        )

    @staticmethod
    def _resolve_import_module(
        *, current_package: str, module_name: str, level: int
    ) -> str:
        """Resolve import module."""
        if level <= 0:
            return module_name
        try:
            return _importlib_util.resolve_name(
                f"{'.' * level}{module_name}", current_package
            )
        except (ImportError, ValueError):
            return module_name

    @staticmethod
    def get_module_semantic_state(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> m.Infra.ModuleSemanticState:
        """Return local classes plus declared and semantic imports in one pass.

        Uses rope's ``PyModule.get_attributes()`` for class discovery and
        ``rope.refactor.importutils.get_module_imports`` for import-table
        construction — no ``ast`` walking is performed.
        """
        cache_key = FlextInfraUtilitiesRopeAnalysis._resource_cache_key(
            rope_project, resource
        )
        cached = FlextInfraUtilitiesRopeAnalysis._SEMANTIC_STATE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(rope_project, resource)
            state = (
                FlextInfraUtilitiesRopeAnalysis._module_semantic_state_from_pymodule(
                    rope_project=rope_project, resource=resource, pymodule=pymodule
                )
            )
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
            state = FlextInfraUtilitiesRopeAnalysis._empty_module_semantic_state()
        FlextInfraUtilitiesRopeAnalysis._SEMANTIC_STATE_CACHE[cache_key] = state
        return state

    @staticmethod
    def _empty_module_semantic_state() -> m.Infra.ModuleSemanticState:
        """Return an empty semantic state."""
        return m.Infra.ModuleSemanticState(
            class_infos=(), declared_imports={}, semantic_imports={}
        )

    @staticmethod
    def _module_semantic_state_from_pymodule(
        *,
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        pymodule: t.Infra.RopePyModule,
    ) -> m.Infra.ModuleSemanticState:
        """Build semantic state from one resolved Rope module."""
        current_package = FlextInfraUtilitiesRopeAnalysis._package_name_for_module(
            pymodule.get_name(), resource
        )
        declared_imports, semantic_imports = (
            FlextInfraUtilitiesRopeAnalysis._module_import_maps(
                rope_project=rope_project,
                resource=resource,
                current_package=current_package,
            )
        )
        return m.Infra.ModuleSemanticState(
            class_infos=tuple(
                FlextInfraUtilitiesRopeAnalysis._module_class_infos(
                    pymodule=pymodule, resource=resource
                )
            ),
            declared_imports=declared_imports,
            semantic_imports=semantic_imports,
        )

    @staticmethod
    def _module_class_infos(
        *, pymodule: t.Infra.RopePyModule, resource: t.Infra.RopeResource
    ) -> t.SequenceOf[m.Infra.ClassInfo]:
        """Return local class infos for one resolved Rope module."""
        class_infos: t.MutableSequenceOf[m.Infra.ClassInfo] = []
        ast_bases_by_class = {
            class_info.name: class_info.bases
            for class_info in FlextInfraUtilitiesRopeAnalysis.class_info_from_source(
                resource.read()
            )
        }
        for name, pyname in pymodule.get_attributes().items():
            if not FlextInfraUtilitiesRopeAnalysis._is_local_name(pyname, resource):
                continue
            obj = pyname.get_object()
            if not FlextInfraUtilitiesRopeRuntime.is_abstract_class(obj):
                continue
            location = pyname.get_definition_location()
            line = location[1] if location and location[1] else 1
            bases = tuple(
                base_name
                for superclass in obj.get_superclasses()
                if (
                    base_name := FlextInfraUtilitiesRopeAnalysis._superclass_name(
                        superclass
                    )
                )
            )
            if not bases:
                bases = ast_bases_by_class.get(name, ())
            class_infos.append(m.Infra.ClassInfo(name=name, line=line, bases=bases))
        return tuple(class_infos)

    @staticmethod
    def superclass_name(superclass: p.AttributeProbe) -> str:
        """Return a superclass name from Rope objects with uneven public APIs."""
        return FlextInfraUtilitiesRopeAnalysis._superclass_name(superclass)

    @staticmethod
    def _superclass_name(
        superclass: p.AttributeProbe, *, visited: frozenset[int] | None = None
    ) -> str:
        """Return a superclass name from Rope objects with uneven public APIs."""
        visited_ids = visited or frozenset()
        superclass_id = id(superclass)
        if superclass_id in visited_ids:
            return ""
        next_visited = visited_ids | {superclass_id}
        get_name = getattr(superclass, "get_name", None)
        if callable(get_name):
            name = get_name()
            if isinstance(name, str) and name:
                return name
        get_type = getattr(superclass, "get_type", None)
        if callable(get_type):
            superclass_type = get_type()
            if superclass_type is not None:
                type_name = FlextInfraUtilitiesRopeAnalysis._superclass_name(
                    superclass_type, visited=next_visited
                )
                if type_name:
                    return type_name
        for attr_name in ("name", "_name"):
            name = getattr(superclass, attr_name, "")
            if isinstance(name, str) and name:
                return name
        return ""

    @staticmethod
    def _module_import_maps(
        *,
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        current_package: str,
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Return declared and semantic import maps for one module."""
        semantic_imports: dict[str, str] = {}
        declared_imports: dict[str, str] = {}
        module_imports = FlextInfraUtilitiesRopeCore.get_module_imports(
            rope_project, resource
        )
        raw_imports = (
            getattr(module_imports, "imports", ()) if module_imports is not None else ()
        )
        import_stmts: tuple[t.Infra.RopeImportStatement, ...] = tuple(raw_imports)
        for import_stmt in import_stmts:
            FlextInfraUtilitiesRopeAnalysis._merge_import_statement(
                current_package=current_package,
                declared_imports=declared_imports,
                import_stmt=import_stmt,
                semantic_imports=semantic_imports,
            )
        return declared_imports, semantic_imports

    @staticmethod
    def _merge_import_statement(
        *,
        current_package: str,
        declared_imports: dict[str, str],
        import_stmt: t.Infra.RopeImportStatement,
        semantic_imports: dict[str, str],
    ) -> None:
        """Merge one Rope import statement into the import maps."""
        info = import_stmt.import_info
        module_name = getattr(info, "module_name", "") or ""
        resolved_module = FlextInfraUtilitiesRopeAnalysis._resolved_import_module(
            current_package=current_package,
            module_name=module_name,
            level=getattr(info, "level", 0) or 0,
        )
        for alias_name, alias_as in info.names_and_aliases or ():
            FlextInfraUtilitiesRopeAnalysis._merge_import_alias(
                alias_name=alias_name,
                alias_as=alias_as,
                declared_imports=declared_imports,
                module_name=module_name,
                resolved_module=resolved_module,
                semantic_imports=semantic_imports,
            )

    @staticmethod
    def _resolved_import_module(
        *, current_package: str, module_name: str, level: int
    ) -> str:
        """Resolve the module path represented by one Rope import info."""
        return (
            FlextInfraUtilitiesRopeAnalysis._resolve_import_module(
                current_package=current_package, module_name=module_name, level=level
            )
            if module_name
            else ""
        )

    @staticmethod
    def _merge_import_alias(
        *,
        alias_name: str,
        alias_as: str | None,
        declared_imports: dict[str, str],
        module_name: str,
        resolved_module: str,
        semantic_imports: dict[str, str],
    ) -> None:
        """Merge one import alias into declared and semantic maps."""
        if alias_name == "*":
            return
        if module_name:
            local_name = alias_as or alias_name
            target = (
                f"{resolved_module}.{alias_name}" if resolved_module else alias_name
            )
        else:
            local_name = alias_as or alias_name.partition(".")[0]
            target = alias_name
        if not local_name:
            return
        declared_imports[local_name] = target
        semantic_imports[local_name] = target

    @staticmethod
    def find_definition_offset(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource, symbol: str
    ) -> int | None:
        """Return offset of symbol's definition via semantic analysis."""
        result: int | None = None
        source = resource.read()
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(rope_project, resource)
            result = FlextInfraUtilitiesRopeAnalysis._definition_offset_from_pymodule(
                pymodule=pymodule, source=source, symbol=symbol
            )
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
            pass
        return result

    @staticmethod
    def _definition_offset_from_pymodule(
        *, pymodule: t.Infra.RopePyModule, source: str, symbol: str
    ) -> int | None:
        """Return identifier offset for one symbol from a resolved Rope module."""
        attributes = pymodule.get_attributes()
        if symbol not in attributes:
            return None
        pyname = attributes[symbol]
        definition_module, definition_line = pyname.get_definition_location()
        if definition_module is not None:
            definition_resource = definition_module.get_resource()
            if definition_resource is not None:
                source = definition_resource.read()
        if definition_line is None:
            return None
        lines = source.splitlines(keepends=True)
        return FlextInfraUtilitiesRopeCore.find_identifier_offset_in_lines(
            lines, line=definition_line, symbol=symbol
        )

    @staticmethod
    def get_semantic_module_imports(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.StrMapping:
        """Return {local_name: fully_qualified_name} for all imports in a module."""
        imports: t.StrMapping = (
            FlextInfraUtilitiesRopeAnalysis.get_module_semantic_state(
                rope_project, resource
            ).semantic_imports
        )
        return imports

    @staticmethod
    def get_declared_module_imports(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.StrMapping:
        """Return {local_name: declared import path} without resolving re-exports."""
        imports: t.StrMapping = (
            FlextInfraUtilitiesRopeAnalysis.get_module_semantic_state(
                rope_project, resource
            ).declared_imports
        )
        return imports

    @staticmethod
    def get_module_classes(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.StrSequence:
        """Return names of all classes defined in a module."""
        return tuple(
            class_info.name
            for class_info in FlextInfraUtilitiesRopeAnalysis.get_module_semantic_state(
                rope_project, resource
            ).class_infos
        )

    @staticmethod
    def scope_definitions(
        pymodule: t.Infra.RopePyModule,
    ) -> t.SequenceOf[m.Infra.ScopeDefinition]:
        """Return every def/class scope in a module via rope's semantic tree.

        Uses ``PyModule.get_scope()`` and recursive ``PyScope.get_scopes()`` —
        never ``get_ast``/``ast.walk``. Each entry carries the scope kind
        (Function/Class/Comprehension), its 1-based start line, and whether it
        is a direct child of the module (global) scope.
        """
        root_scope = pymodule.get_scope()
        if root_scope is None:
            return ()
        definitions: t.MutableSequenceOf[m.Infra.ScopeDefinition] = []
        FlextInfraUtilitiesRopeAnalysis._collect_scope_definitions(
            scope=root_scope, is_module_level=True, definitions=definitions
        )
        return tuple(definitions)

    @staticmethod
    def _collect_scope_definitions(
        *,
        scope: t.Infra.RopeScope,
        is_module_level: bool,
        definitions: t.MutableSequenceOf[m.Infra.ScopeDefinition],
    ) -> None:
        """Recurse the rope scope tree, appending one entry per child scope."""
        for child in scope.get_scopes():
            start = child.get_start()
            # NOTE (multi-agent, flext-f8vk / kimi): RopeScope.get_start() is
            # declared int in p.Infra; the old isinstance guard was dead code.
            definitions.append(
                m.Infra.ScopeDefinition(
                    name=FlextInfraUtilitiesRopeAnalysis._scope_name(child),
                    kind=FlextInfraUtilitiesRopeAnalysis._scope_kind(child),
                    line=start if start > 0 else 1,
                    is_module_level=is_module_level,
                )
            )
            FlextInfraUtilitiesRopeAnalysis._collect_scope_definitions(
                scope=child, is_module_level=False, definitions=definitions
            )

    @staticmethod
    def _scope_kind(scope: t.Infra.RopeScope) -> c.Infra.RopeScopeKind:
        """Map rope's ``get_kind()`` to the typed enum (None -> UNKNOWN)."""
        raw_kind = scope.get_kind()
        try:
            return c.Infra.RopeScopeKind(raw_kind)
        except ValueError:
            return c.Infra.RopeScopeKind.UNKNOWN

    @staticmethod
    def _scope_name(scope: t.Infra.RopeScope) -> str:
        """Return the def/class name backing one rope scope."""
        name: str = scope.pyobject.get_name()
        return name

    @staticmethod
    def get_module_export_names(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        *,
        export_options: m.Infra.ExportOptions | None = None,
    ) -> t.StrSequence:
        """Return module-local export names from Rope metadata."""
        resolved_export_options = export_options or m.Infra.ExportOptions()
        cache_key = (
            *FlextInfraUtilitiesRopeAnalysis._resource_cache_key(
                rope_project, resource
            ),
            resolved_export_options.include_dunder,
            resolved_export_options.allow_main,
            resolved_export_options.allow_assignments,
            resolved_export_options.allow_functions,
            resolved_export_options.require_explicit_all,
        )
        cached = FlextInfraUtilitiesRopeAnalysis._EXPORT_NAMES_CACHE.get(cache_key)
        export_names: t.StrSequence
        if cached is not None:
            export_names = cached
        else:
            try:
                pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(
                    rope_project, resource
                )
                export_names = FlextInfraUtilitiesRopeAnalysis._module_export_names(
                    export_options=resolved_export_options,
                    pymodule=pymodule,
                    resource=resource,
                )
                FlextInfraUtilitiesRopeAnalysis._EXPORT_NAMES_CACHE[cache_key] = (
                    export_names
                )
            except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
                FlextInfraUtilitiesRopeAnalysis._EXPORT_NAMES_CACHE[cache_key] = ()
                export_names = ()
        return export_names

    @staticmethod
    def _module_export_names(
        *,
        export_options: m.Infra.ExportOptions,
        pymodule: t.Infra.RopePyModule,
        resource: t.Infra.RopeResource,
    ) -> t.StrSequence:
        """Return export names for one resolved Rope module."""
        attributes = pymodule.get_attributes()
        if export_options.include_dunder:
            return FlextInfraUtilitiesRopeAnalysis._dunder_export_names(
                attributes=attributes, resource=resource
            )
        explicit_all = FlextInfraUtilitiesRopeAnalysis._explicit_export_names(
            attributes=attributes, pymodule=pymodule, resource=resource
        )
        if explicit_all is not None:
            return tuple(dict.fromkeys(explicit_all))
        if export_options.require_explicit_all:
            return ()
        return FlextInfraUtilitiesRopeAnalysis._implicit_export_names(
            attributes=attributes,
            export_options=export_options,
            resource=resource,
            pymodule=pymodule,
        )

    @staticmethod
    def _dunder_export_names(
        *,
        attributes: t.MappingKV[str, t.Infra.RopePyName],
        resource: t.Infra.RopeResource,
    ) -> t.StrSequence:
        """Return locally assigned dunder export names."""
        return tuple(
            dict.fromkeys(
                name
                for name, pyname in attributes.items()
                if name != c.Infra.DUNDER_ALL
                and name.startswith("__")
                and name.endswith("__")
                and FlextInfraUtilitiesRopeRuntime.is_assigned_name(pyname)
                and FlextInfraUtilitiesRopeAnalysis._is_local_name(pyname, resource)
            )
        )

    @staticmethod
    def _explicit_export_names(
        *,
        attributes: t.MappingKV[str, t.Infra.RopePyName],
        pymodule: t.Infra.RopePyModule,
        resource: t.Infra.RopeResource,
    ) -> t.StrSequence | None:
        """Return explicit ``__all__`` export names when declared locally."""
        explicit_all_name = attributes.get(c.Infra.DUNDER_ALL)
        if (
            explicit_all_name is None
            or not FlextInfraUtilitiesRopeRuntime.is_assigned_name(explicit_all_name)
        ):
            return None
        assigned_all: t.Infra.RopeAssignedName = explicit_all_name
        if not FlextInfraUtilitiesRopeAnalysis._is_local_name(assigned_all, resource):
            return None
        return FlextInfraUtilitiesRopeAnalysis._explicit_all_names(
            assigned_all, pymodule
        )

    @staticmethod
    def _implicit_export_names(
        *,
        attributes: t.MappingKV[str, t.Infra.RopePyName],
        export_options: m.Infra.ExportOptions,
        resource: t.Infra.RopeResource,
        pymodule: t.Infra.RopePyModule,
    ) -> t.StrSequence:
        """Return implicit export names accepted by the export options."""
        guard_spans = FlextInfraUtilitiesRopeAnalysis._script_guard_spans(pymodule)
        names: t.MutableSequenceOf[str] = []
        for name, pyname in attributes.items():
            if name == c.Infra.DUNDER_ALL:
                continue
            if not FlextInfraUtilitiesRopeAnalysis._is_local_name(pyname, resource):
                continue
            if FlextInfraUtilitiesRopeAnalysis._is_export_name(
                export_options=export_options,
                name=name,
                pyname=pyname,
                guard_spans=guard_spans,
            ):
                names.append(name)
        return tuple(dict.fromkeys(names))

    @staticmethod
    def _script_guard_spans(
        pymodule: t.Infra.RopePyModule,
    ) -> t.SequenceOf[t.Pair[int, int]]:
        """Return the line spans of top-level ``if __name__ == "__main__":`` blocks.

        Names bound there exist only when the module runs as a script; a
        package facade that re-exported them (flext-core's examples exported
        ``result`` and ``msg``) failed every importer.
        """
        module_ast = pymodule.get_ast()
        body = getattr(module_ast, "body", ())
        spans: t.MutableSequenceOf[t.Pair[int, int]] = []
        for node in body:
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if not (
                isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
            ):
                continue
            sides = (test.left, test.comparators[0])
            names = {side.id for side in sides if isinstance(side, ast.Name)}
            values = {side.value for side in sides if isinstance(side, ast.Constant)}
            if names == {"__name__"} and values == {"__main__"}:
                spans.append((node.lineno, node.end_lineno or node.lineno))
        return tuple(spans)

    @staticmethod
    def _is_export_name(
        *,
        export_options: m.Infra.ExportOptions,
        name: str,
        pyname: t.Infra.RopePyName,
        guard_spans: t.SequenceOf[t.Pair[int, int]] = (),
    ) -> bool:
        """Return whether one Rope name is exportable under the options."""
        if FlextInfraUtilitiesRopeRuntime.is_imported_name(pyname):
            return False
        if FlextInfraUtilitiesRopeRuntime.is_assigned_name(pyname):
            allow_assignments: bool = export_options.allow_assignments
            if not allow_assignments:
                return False
            lines = tuple(
                line
                for assignment in pyname.assignments
                if (line := getattr(assignment.ast_node, "lineno", None)) is not None
            )
            return not lines or not all(
                any(start <= line <= end for start, end in guard_spans)
                for line in lines
            )
        if not FlextInfraUtilitiesRopeRuntime.is_defined_name(pyname):
            return False
        obj = pyname.get_object()
        if FlextInfraUtilitiesRopeRuntime.is_abstract_class(obj):
            return True
        if not FlextInfraUtilitiesRopeRuntime.is_py_function(obj):
            return False
        allow_main: bool = export_options.allow_main
        allow_functions: bool = export_options.allow_functions
        return (allow_main and name == "main") or allow_functions

    @staticmethod
    def _explicit_all_names(
        pyname: t.Infra.RopeAssignedName, pymodule: t.Infra.RopePyModule
    ) -> t.StrSequence | None:
        """Return literal ``__all__`` names from a Rope-cached source slice.

        Reads the assignment's source range via the module's ``source_code``
        and extracts string literals with ``c.Infra.STRING_LITERAL_RE`` —
        avoiding ``ast`` walking entirely.
        """
        for assignment in pyname.assignments:
            node = assignment.ast_node
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            if start is None:
                continue
            lines = pymodule.source_code.splitlines()
            slice_text = "\n".join(lines[start - 1 : end])
            return tuple(dict.fromkeys(c.Infra.STRING_LITERAL_RE.findall(slice_text)))
        return None

    @staticmethod
    def public_export_names_source(source: str) -> t.StrSequence:
        """Return the explicit public ABI declared by one module source."""
        return FlextInfraUtilitiesRopeAnalysis.module_assignment_strings_source(
            source, c.Infra.DUNDER_ALL
        )

    @staticmethod
    def _is_local_name(
        pyname: t.Infra.RopePyName, resource: t.Infra.RopeResource
    ) -> bool:
        """Return whether one Rope name is defined in ``resource``."""
        # NOTE (multi-agent, flext-f8vk / kimi): p.Infra declares
        # get_definition_location() as tuple-always (every other caller
        # unpacks directly); the old None guard was dead code.
        module, line = pyname.get_definition_location()
        origin = module.get_resource() if module is not None else None
        return line is not None and origin is not None and origin.path == resource.path

    @staticmethod
    def is_pyclass(obj: p.AttributeProbe) -> bool:
        """Return whether a rope object is a ``PyClass`` (abstract class type)."""
        return FlextInfraUtilitiesRopeRuntime.is_abstract_class(obj)

    @staticmethod
    def is_pyfunction(obj: p.AttributeProbe) -> bool:
        """Return whether a rope object is a ``PyFunction``."""
        return FlextInfraUtilitiesRopeRuntime.is_py_function(obj)

    @staticmethod
    def module_has_docstring_source(source: str) -> bool:
        """Return whether ``source`` starts with a module docstring (rope-parsed)."""
        pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if pymodule is None:
            return False
        try:
            return bool(pymodule.get_doc())
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
            return False

    @staticmethod
    def module_docstring_summary_source(source: str) -> str:
        """Return the PEP 257 summary line of the module docstring (rope-parsed)."""
        pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if pymodule is None:
            return ""
        try:
            doc = pymodule.get_doc() or ""
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
            return ""
        summary = next((line for line in doc.splitlines() if line.strip()), "")
        # The summary is rendered verbatim into generated markdown, where runs
        # of spaces are a lint failure nobody can hand-fix in a generated file.
        # Collapsing them here keeps the summary faithful and renderable.
        return " ".join(summary.split())

    @staticmethod
    def symbol_has_docstring_source(source: str, symbol_name: str) -> bool:
        """Return whether ``symbol_name`` in ``source`` carries a docstring (rope-parsed)."""
        pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if pymodule is None:
            return False
        try:
            pyname = pymodule.get_attributes().get(symbol_name)
            if pyname is None:
                return False
            return bool(pyname.get_object().get_doc())
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
            return False

    @staticmethod
    def assignment_docstrings_source(source: str) -> t.StrSequence:
        """Return assignment names followed by a string-literal expression (rope-parsed).

        Iterates the parsed module's body via ``_fields`` access and pairs each
        ``Assign``/``AnnAssign`` target with the next sibling ``Expr(Constant(str))``.
        """
        pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if pymodule is None:
            return ()
        module_ast = pymodule.get_ast()
        body = getattr(module_ast, "body", []) or []
        names: list[str] = []
        previous_targets: list[str] = []
        for statement in body:
            kind = FlextInfraUtilitiesRopeAnalysis.node_kind(statement)
            if kind in {"Assign", "AnnAssign"}:
                previous_targets = (
                    FlextInfraUtilitiesRopeAnalysis._statement_target_names(statement)
                )
                continue
            if kind == "Expr" and previous_targets:
                value = getattr(statement, "value", None)
                if (
                    value is not None
                    and FlextInfraUtilitiesRopeAnalysis.node_kind(value) == "Constant"
                    and isinstance(getattr(value, "value", None), str)
                ):
                    names.extend(previous_targets)
            previous_targets = []
        return tuple(dict.fromkeys(names))

    @staticmethod
    def module_body_nodes_source(source: str) -> t.SequenceOf[p.AttributeProbe]:
        """Return top-level parsed statements for one source module."""
        _ = source
        return ()

    @staticmethod
    def module_reachable_nodes_source(source: str) -> t.SequenceOf[p.AttributeProbe]:
        """Return parsed nodes reachable from one source module."""
        _ = source
        return ()

    @staticmethod
    def literal_string_sequence(node: p.AttributeProbe | None) -> t.StrSequence:
        """Return string entries from a parsed literal sequence node."""
        if node is None:
            return ()
        if FlextInfraUtilitiesRopeAnalysis.node_kind(node) not in {
            "List",
            "Tuple",
            "Set",
        }:
            return ()
        values: list[str] = []
        for element in getattr(node, "elts", ()) or ():
            element_kind = FlextInfraUtilitiesRopeAnalysis.node_kind(element)
            if element_kind == "Constant":
                value = getattr(element, "value", None)
            elif element_kind == "Str":
                value = getattr(element, "s", None)
            else:
                return ()
            if not isinstance(value, str):
                return ()
            values.append(value)
        return tuple(values)

    @staticmethod
    def _sequence_constructor_ref_source(source: str) -> str:
        """Return the name wrapped by ``tuple``/``list``/``set`` or a bare name."""
        text = source.strip()
        if not text:
            return ""
        if text.isidentifier():
            return text
        for constructor in ("tuple", "list", "frozenset", "set"):
            prefix = f"{constructor}("
            if not text.startswith(prefix) or not text.endswith(")"):
                continue
            inner = text[len(prefix) : -1].strip()
            if inner.isidentifier():
                return inner
        return ""

    @staticmethod
    def module_assignment_strings_source(source: str, name: str) -> t.StrSequence:
        """Collect strings from a literal module-level assignment."""
        value_source = FlextInfraUtilitiesRopeAnalysis._assignment_value_source(
            source, name
        )
        values = FlextInfraUtilitiesRopeAnalysis._literal_string_sequence_source(
            value_source
        )
        if values:
            return values
        # Generated roots use ``__all__ = tuple(_PUBLIC_EXPORTS)``; follow the
        # bound name so docs validate matches the live lazy-init ABI.
        nested_name = FlextInfraUtilitiesRopeAnalysis._sequence_constructor_ref_source(
            value_source
        )
        if not nested_name or nested_name == name:
            return ()
        return FlextInfraUtilitiesRopeAnalysis.module_assignment_strings_source(
            source, nested_name
        )

    @staticmethod
    def module_mapping_assignment_source(
        source: str, name: str
    ) -> tuple[tuple[tuple[str, t.StrSequence], ...], t.StrSequence]:
        """Collect mapping entries and referenced names from an assignment."""
        value_source = FlextInfraUtilitiesRopeAnalysis._assignment_value_source(
            source, name
        )
        return FlextInfraUtilitiesRopeAnalysis._mapping_entries_refs_source(
            value_source
        )

    @staticmethod
    def mapping_entries_refs(
        node: p.AttributeProbe | None,
    ) -> tuple[tuple[tuple[str, t.StrSequence], ...], t.StrSequence]:
        """Return literal mapping entries plus variable references."""
        if node is None:
            return ((), ())
        kind = FlextInfraUtilitiesRopeAnalysis.node_kind(node)
        if kind == "Name":
            node_name = FlextInfraUtilitiesRopeAnalysis.name_of(node)
            return ((), (node_name,) if node_name else ())
        if kind == "Dict":
            return FlextInfraUtilitiesRopeAnalysis._dict_entries_refs(node)
        if kind != "Call":
            return ((), ())
        function_name = FlextInfraUtilitiesRopeAnalysis.name_of(
            getattr(node, "func", None)
        )
        args = getattr(node, "args", ()) or ()
        if function_name in {"MappingProxyType", "build_lazy_import_map"} and args:
            return FlextInfraUtilitiesRopeAnalysis.mapping_entries_refs(args[0])
        if function_name != "merge_lazy_imports":
            return ((), ())
        entries: list[tuple[str, t.StrSequence]] = []
        refs: list[str] = []
        for argument in args:
            next_entries, next_refs = (
                FlextInfraUtilitiesRopeAnalysis.mapping_entries_refs(argument)
            )
            entries.extend(next_entries)
            refs.extend(next_refs)
        return (tuple(entries), tuple(dict.fromkeys(refs)))

    @staticmethod
    def _dict_entries_refs(
        node: p.AttributeProbe,
    ) -> tuple[tuple[tuple[str, t.StrSequence], ...], t.StrSequence]:
        """Return string-sequence dict entries and unpack references."""
        keys = getattr(node, "keys", ()) or ()
        values = getattr(node, "values", ()) or ()
        entries: list[tuple[str, t.StrSequence]] = []
        refs: list[str] = []
        for key_node, value_node in zip(keys, values, strict=False):
            if key_node is None:
                ref_name = FlextInfraUtilitiesRopeAnalysis.name_of(value_node)
                if ref_name:
                    refs.append(ref_name)
                continue
            key_value = getattr(key_node, "value", None)
            if not isinstance(key_value, str):
                continue
            value_strings = FlextInfraUtilitiesRopeAnalysis.literal_string_sequence(
                value_node
            )
            if value_strings:
                entries.append((key_value, value_strings))
        return (tuple(entries), tuple(dict.fromkeys(refs)))

    @staticmethod
    def relative_import_module_name(
        *, current_module: str, imported_module: str, level: int, package_module: bool
    ) -> str:
        """Resolve a parsed ``from`` import module into an absolute module name."""
        if level == 0:
            return imported_module
        current_parts = current_module.split(".")
        base_count = len(current_parts) - level + (1 if package_module else 0)
        base = ".".join(current_parts[: max(base_count, 0)])
        return ".".join(part for part in (base, imported_module) if part)

    @staticmethod
    def imported_symbol_binding_source(
        source: str, *, current_module: str, symbol_name: str, package_module: bool
    ) -> tuple[str, str]:
        """Return ``(module, original_name)`` for one imported symbol binding."""
        for (
            module_source,
            level,
            original_name,
            bound_name,
        ) in FlextInfraUtilitiesRopeAnalysis._from_import_bindings_source(source):
            if bound_name != symbol_name:
                continue
            module_name = FlextInfraUtilitiesRopeAnalysis.relative_import_module_name(
                current_module=current_module,
                imported_module=module_source,
                level=level,
                package_module=package_module,
            )
            return (module_name, original_name or symbol_name)
        return ("", "")

    @staticmethod
    def class_bases_source(source: str, class_name: str) -> t.StrSequence:
        """Return declared base names for one class in source."""
        class_source = FlextInfraUtilitiesRopeAnalysis._class_header_source(
            source, class_name
        )
        if not class_source:
            return ()
        open_index = class_source.find("(")
        close_index = class_source.rfind(")")
        if open_index < 0 or close_index <= open_index:
            return ()
        bases_source = class_source[open_index + 1 : close_index]
        return tuple(
            base_name
            for base_name in (
                FlextInfraUtilitiesRopeAnalysis._symbol_name_source(item)
                for item in FlextInfraUtilitiesRopeAnalysis._split_top_level_commas(
                    bases_source
                )
            )
            if base_name
        )

    @staticmethod
    def class_declared_source(source: str, class_name: str) -> bool:
        """Return whether one class is declared in source."""
        return bool(
            FlextInfraUtilitiesRopeAnalysis._class_header_source(source, class_name)
        )

    @staticmethod
    def lazy_public_exports_source(source: str) -> tuple[t.StrSequence, str]:
        """Return lazy-loader public exports or the local symbol holding them."""
        call_args = FlextInfraUtilitiesRopeAnalysis._call_args_source(
            source, "install_lazy_exports"
        )
        public_exports = FlextInfraUtilitiesRopeAnalysis._keyword_value_source(
            call_args, "public_exports"
        )
        if public_exports:
            values = FlextInfraUtilitiesRopeAnalysis._literal_string_sequence_source(
                public_exports
            )
            if values:
                return (values, "")
            return (
                (),
                FlextInfraUtilitiesRopeAnalysis._symbol_name_source(public_exports),
            )
        return ((), "")

    @staticmethod
    def lazy_imports_name_source(source: str) -> str:
        """Return the local symbol passed as the lazy import map."""
        call_args = FlextInfraUtilitiesRopeAnalysis._call_args_source(
            source, "install_lazy_exports"
        )
        if (
            len(call_args)
            > FlextInfraUtilitiesRopeAnalysis._INSTALL_LAZY_IMPORTS_ARG_INDEX
        ):
            return FlextInfraUtilitiesRopeAnalysis._symbol_name_source(
                call_args[
                    FlextInfraUtilitiesRopeAnalysis._INSTALL_LAZY_IMPORTS_ARG_INDEX
                ]
            )
        keyword_value = FlextInfraUtilitiesRopeAnalysis._keyword_value_source(
            call_args, "lazy_imports"
        )
        if keyword_value:
            return FlextInfraUtilitiesRopeAnalysis._symbol_name_source(keyword_value)
        return ""

    @staticmethod
    def _assignment_value_source(source: str, name: str) -> str:
        """Return the source value assigned to one top-level symbol."""
        lines = source.splitlines()
        for index, line in enumerate(lines):
            if line[: len(line) - len(line.lstrip())]:
                continue
            stripped = line.strip()
            if not stripped.startswith(name):
                continue
            tail = stripped[len(name) :].lstrip()
            if not tail or tail[0] not in {":", "="}:
                continue
            statement = FlextInfraUtilitiesRopeAnalysis._collect_statement(lines, index)
            _head, separator, value = statement.partition("=")
            if not separator:
                return ""
            return value.strip()
        return ""

    @staticmethod
    def _collect_statement(lines: t.StrSequence, start_index: int) -> str:
        """Collect a balanced Python statement starting at ``start_index``."""
        collected: list[str] = []
        depth = 0
        for line in lines[start_index:]:
            collected.append(line)
            depth += FlextInfraUtilitiesRopeAnalysis._bracket_depth_delta(line)
            if depth <= 0:
                break
        return "\n".join(collected)

    @staticmethod
    def _bracket_depth_delta(source: str) -> int:
        """Return bracket nesting delta for one source line."""
        depth = 0
        quote = ""
        escaped = False
        for char in source:
            if quote:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == quote:
                    quote = ""
                continue
            if char == "#":
                break
            if char in {"'", '"'}:
                quote = char
                continue
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
        return depth

    @staticmethod
    def _split_top_level_commas(source: str) -> t.StrSequence:
        """Split one source fragment on commas outside nested delimiters."""
        parts: list[str] = []
        start = 0
        depth = 0
        quote = ""
        escaped = False
        for index, char in enumerate(source):
            if quote:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == quote:
                    quote = ""
                continue
            if char in {"'", '"'}:
                quote = char
                continue
            if char in "([{":
                depth += 1
                continue
            if char in ")]}":
                depth -= 1
                continue
            if char == "," and depth == 0:
                item = source[start:index].strip()
                if item:
                    parts.append(item)
                start = index + 1
        tail = source[start:].strip()
        if tail:
            parts.append(tail)
        return tuple(parts)

    @staticmethod
    def _top_level_partition(source: str, separator: str) -> tuple[str, str, str]:
        """Partition one source fragment at a top-level separator."""
        depth = 0
        quote = ""
        escaped = False
        for index, char in enumerate(source):
            if quote:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == quote:
                    quote = ""
                continue
            if char in {"'", '"'}:
                quote = char
                continue
            if char in "([{":
                depth += 1
                continue
            if char in ")]}":
                depth -= 1
                continue
            if char == separator and depth == 0:
                return (source[:index], separator, source[index + 1 :])
        return (source, "", "")

    @staticmethod
    def _literal_string_source(source: str) -> str:
        """Return the value of a simple string literal source fragment."""
        text = source.strip()
        while text and text[0].isalpha() and len(text) > 1 and text[1] in {"'", '"'}:
            text = text[1:]
        if len(
            text
        ) < FlextInfraUtilitiesRopeAnalysis._STRING_LITERAL_MIN_LENGTH or text[
            0
        ] not in {"'", '"'}:
            return ""
        quote = text[0]
        triple_quote = quote * FlextInfraUtilitiesRopeAnalysis._TRIPLE_QUOTE_LENGTH
        if text.startswith(triple_quote):
            end = text.find(
                triple_quote, FlextInfraUtilitiesRopeAnalysis._TRIPLE_QUOTE_LENGTH
            )
            return (
                text[FlextInfraUtilitiesRopeAnalysis._TRIPLE_QUOTE_LENGTH : end]
                if end >= FlextInfraUtilitiesRopeAnalysis._TRIPLE_QUOTE_LENGTH
                else ""
            )
        end = 1
        escaped = False
        while end < len(text):
            char = text[end]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                break
            end += 1
        return text[1:end]

    @staticmethod
    def _literal_string_sequence_source(source: str) -> t.StrSequence:
        """Return string entries from a literal sequence source fragment."""
        text = source.strip()
        if not text or text[0] not in "[{(":
            return ()
        close = {"[": "]", "{": "}", "(": ")"}[text[0]]
        if not text.endswith(close):
            return ()
        inner = text[1:-1]
        values: list[str] = []
        for item in FlextInfraUtilitiesRopeAnalysis._split_top_level_commas(inner):
            value = FlextInfraUtilitiesRopeAnalysis._literal_string_source(item)
            if not value:
                return ()
            values.append(value)
        return tuple(values)

    @staticmethod
    def _call_args_source(source: str, function_name: str) -> t.StrSequence:
        """Return top-level argument sources for the first matching call."""
        search_from = 0
        while True:
            name_index = source.find(function_name, search_from)
            if name_index < 0:
                return ()
            open_index = source.find("(", name_index + len(function_name))
            if open_index < 0:
                return ()
            between = source[name_index + len(function_name) : open_index].strip()
            if between:
                search_from = name_index + len(function_name)
                continue
            close_index = FlextInfraUtilitiesRopeAnalysis._matching_close_index(
                source, open_index
            )
            if close_index < 0:
                return ()
            return FlextInfraUtilitiesRopeAnalysis._split_top_level_commas(
                source[open_index + 1 : close_index]
            )

    @staticmethod
    def _matching_close_index(source: str, open_index: int) -> int:
        """Return the index of the closing delimiter matching ``open_index``."""
        open_char = source[open_index]
        close_char = {"(": ")", "[": "]", "{": "}"}[open_char]
        depth = 0
        quote = ""
        escaped = False
        for index in range(open_index, len(source)):
            char = source[index]
            if quote:
                if escaped:
                    escaped = False
                    continue
                if char == "\\":
                    escaped = True
                    continue
                if char == quote:
                    quote = ""
                continue
            if char in {"'", '"'}:
                quote = char
                continue
            if char == open_char:
                depth += 1
                continue
            if char == close_char:
                depth -= 1
                if depth == 0:
                    return index
        return -1

    @staticmethod
    def _keyword_value_source(args: t.StrSequence, keyword: str) -> str:
        """Return a keyword argument value source from split call args."""
        prefix = f"{keyword}="
        # NOTE (multi-agent, flext-f8vk / kimi): args is t.StrSequence
        # (SequenceOf[str]); the old isinstance guard was dead code.
        for arg in args:
            text = arg.strip()
            if text.startswith(prefix):
                value: str = t.Infra.STR_ADAPTER.validate_python(
                    text[len(prefix) :].strip()
                )
                return value
        return ""

    @staticmethod
    def _mapping_entries_refs_source(
        source: str,
    ) -> tuple[tuple[tuple[str, t.StrSequence], ...], t.StrSequence]:
        """Return lazy-map entries and referenced mapping symbols from source."""
        text = source.strip()
        if not text:
            return ((), ())
        if FlextInfraUtilitiesRopeAnalysis._is_symbol_source(text):
            return ((), (text,))
        call_name = FlextInfraUtilitiesRopeAnalysis._call_name_source(text)
        if call_name in {"MappingProxyType", "build_lazy_import_map"}:
            args = FlextInfraUtilitiesRopeAnalysis._call_args_source(text, call_name)
            return (
                FlextInfraUtilitiesRopeAnalysis._mapping_entries_refs_source(args[0])
                if args
                else ((), ())
            )
        if call_name == "merge_lazy_imports":
            entries: list[tuple[str, t.StrSequence]] = []
            refs: list[str] = []
            for arg in FlextInfraUtilitiesRopeAnalysis._call_args_source(
                text, call_name
            ):
                next_entries, next_refs = (
                    FlextInfraUtilitiesRopeAnalysis._mapping_entries_refs_source(arg)
                )
                entries.extend(next_entries)
                refs.extend(next_refs)
            return (tuple(entries), tuple(dict.fromkeys(refs)))
        if not text.startswith("{") or not text.endswith("}"):
            return ((), ())
        entries = []
        refs = []
        for item in FlextInfraUtilitiesRopeAnalysis._split_top_level_commas(text[1:-1]):
            stripped = item.strip()
            if stripped.startswith("**"):
                ref_name = FlextInfraUtilitiesRopeAnalysis._symbol_name_source(
                    stripped[2:]
                )
                if ref_name:
                    refs.append(ref_name)
                continue
            key_source, separator, value_source = (
                FlextInfraUtilitiesRopeAnalysis._top_level_partition(stripped, ":")
            )
            if not separator:
                continue
            key = FlextInfraUtilitiesRopeAnalysis._literal_string_source(key_source)
            values = FlextInfraUtilitiesRopeAnalysis._literal_string_sequence_source(
                value_source
            )
            if key and values:
                entries.append((key, values))
        return (tuple(entries), tuple(dict.fromkeys(refs)))

    @staticmethod
    def _call_name_source(source: str) -> str:
        """Return the final symbol name for a call source fragment."""
        text = source.strip()
        open_index = text.find("(")
        if open_index < 0:
            return ""
        return FlextInfraUtilitiesRopeAnalysis._symbol_name_source(text[:open_index])

    @staticmethod
    def _is_symbol_source(source: str) -> bool:
        """Return whether source is a simple dotted or underscored symbol."""
        text = source.strip()
        return bool(text) and all(char.isalnum() or char in {"_", "."} for char in text)

    @staticmethod
    def _symbol_name_source(source: str) -> str:
        """Return the final identifier from one symbol source fragment."""
        text = source.strip()
        if not FlextInfraUtilitiesRopeAnalysis._is_symbol_source(text):
            return ""
        return text.rsplit(".", maxsplit=1)[-1]

    @staticmethod
    def _from_import_bindings_source(
        source: str,
    ) -> tuple[tuple[str, int, str, str], ...]:
        """Return ``(module, level, original, bound)`` for ``from`` imports."""
        lines = source.splitlines()
        bindings: list[tuple[str, int, str, str]] = []
        for index, line in enumerate(lines):
            stripped = line.strip()
            if not stripped.startswith("from ") or " import " not in stripped:
                continue
            statement = FlextInfraUtilitiesRopeAnalysis._collect_statement(
                lines, index
            ).strip()
            import_index = statement.find(" import ")
            module_source = statement[5:import_index].strip()
            level = len(module_source) - len(module_source.lstrip("."))
            module_name = module_source[level:]
            aliases_source = statement[import_index + len(" import ") :].strip()
            if aliases_source.startswith("(") and aliases_source.endswith(")"):
                aliases_source = aliases_source[1:-1]
            for alias_source in FlextInfraUtilitiesRopeAnalysis._split_top_level_commas(
                aliases_source
            ):
                original, bound = FlextInfraUtilitiesRopeAnalysis._import_alias_names(
                    alias_source
                )
                if original and bound:
                    bindings.append((module_name, level, original, bound))
        return tuple(bindings)

    @staticmethod
    def _import_alias_names(source: str) -> tuple[str, str]:
        """Return ``(original, bound)`` names for one import alias source."""
        parts = source.strip().split()
        if (
            len(parts) == FlextInfraUtilitiesRopeAnalysis._IMPORT_ALIAS_AS_PARTS
            and parts[1] == "as"
        ):
            return (parts[0], parts[2])
        if len(parts) == 1:
            return (parts[0], parts[0])
        return ("", "")

    @staticmethod
    def _class_header_source(source: str, class_name: str) -> str:
        """Return the collected class header source for one top-level class."""
        lines = source.splitlines()
        prefix = f"class {class_name}"
        for index, line in enumerate(lines):
            if line[: len(line) - len(line.lstrip())]:
                continue
            stripped = line.strip()
            if not stripped.startswith(prefix):
                continue
            tail = stripped[len(prefix) :]
            if tail and tail[0] not in {"(", ":"}:
                continue
            statement = FlextInfraUtilitiesRopeAnalysis._collect_statement(lines, index)
            return statement.rsplit(":", maxsplit=1)[0]
        return ""

    @staticmethod
    def _statement_target_names(statement: object) -> list[str]:
        """Extract target names from an Assign/AnnAssign statement."""
        kind = FlextInfraUtilitiesRopeAnalysis.node_kind(statement)
        if kind == "AnnAssign":
            target = getattr(statement, "target", None)
            name = getattr(target, "id", "") if target is not None else ""
            return [name] if name else []
        targets = getattr(statement, "targets", []) or []
        names: list[str] = []
        for target in targets:
            name = getattr(target, "id", "")
            if isinstance(name, str) and name:
                names.append(name)
        return names

    @staticmethod
    def export_target_modules_source(
        source: str, package_name: str, exports: t.StrSequence
    ) -> dict[str, str]:
        """Map exports → defining module via rope's parsed-source import table."""
        export_names = {name for name in exports if name}
        target_map: dict[str, str] = dict.fromkeys(export_names, package_name)
        pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if pymodule is None:
            return target_map
        module_ast = pymodule.get_ast()
        for node in FlextInfraUtilitiesRopeAnalysis.walk_ast_nodes(module_ast):
            kind = FlextInfraUtilitiesRopeAnalysis.node_kind(node)
            if kind == "ImportFrom":
                module_name = getattr(node, "module", "") or ""
                names = getattr(node, "names", []) or []
                for alias in names:
                    local = getattr(alias, "asname", None) or getattr(alias, "name", "")
                    if isinstance(local, str) and local in export_names and module_name:
                        target_map[local] = module_name
            elif kind == "Import":
                names = getattr(node, "names", []) or []
                for alias in names:
                    raw_name = getattr(alias, "name", "") or ""
                    local = getattr(alias, "asname", None) or raw_name.partition(".")[0]
                    if isinstance(local, str) and local in export_names and raw_name:
                        module_name = (
                            raw_name.rsplit(".", maxsplit=1)[0]
                            if "." in raw_name
                            else raw_name
                        )
                        target_map[local] = module_name
        return target_map

    @staticmethod
    def parse_string_module(source: str) -> t.Infra.RopePyModule | None:
        """Parse ``source`` to a rope ``PyModule`` via a shared parsing project.

        Uses rope's ``libutils.get_string_module`` so callers don't need to
        manage temporary files. Returns ``None`` on parse failure.
        """
        rope_project = FlextInfraUtilitiesRopeAnalysis._shared_parse_project()
        try:
            pymodule = FlextInfraUtilitiesRopeRuntime.get_string_module(
                rope_project, source
            )
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
            return None
        result: t.Infra.RopePyModule | None = pymodule
        return result

    @staticmethod
    def _shared_parse_project() -> t.Infra.RopeProject:
        """Return a process-wide rope project usable for string parsing."""
        cached = FlextInfraUtilitiesRopeAnalysis._parse_project
        if cached is None:
            # flext-o6h5 (agent: kimi) — root-cause fix: the anchor was a hardcoded
            # operator path that crashed CI (FileNotFoundError) and silently bound
            # the parse project to the wrong tree locally. Anchor on the validated
            # settings SSOT, with cwd as last resort — both exist where CLI runs.
            # Path() coercion keeps this correct while settings migrates the
            # field from str to Path (both accepted).
            repository_root = settings.Infra.repository_root
            anchor = Path(repository_root) if repository_root else Path.cwd()
            cached = FlextInfraUtilitiesRopeCore.init_rope_project(anchor)
            FlextInfraUtilitiesRopeAnalysis._parse_project = cached
        return cached

    @staticmethod
    def decorator_names(pyfunction: object) -> t.StrSequence:
        """Extract decorator names from a rope ``PyFunction`` (no ast import)."""
        decorators = getattr(pyfunction, "decorators", None) or ()
        names: list[str] = []
        for decorator in decorators:
            name = getattr(decorator, "id", None) or getattr(decorator, "attr", None)
            if isinstance(name, str) and name:
                names.append(name)
                continue
            func = getattr(decorator, "func", None)
            inner = getattr(func, "id", None) or getattr(func, "attr", None)
            if isinstance(inner, str) and inner:
                names.append(inner)
        return names

    @staticmethod
    def first_decorator_line(pyfunction: object, *, default_line: int) -> int:
        """Return the lowest line number among ``pyfunction``'s decorators."""
        decorators = getattr(pyfunction, "decorators", None) or ()
        candidate_lines = [
            decorator.lineno
            for decorator in decorators
            if isinstance(getattr(decorator, "lineno", None), int)
        ]
        return min(candidate_lines) if candidate_lines else default_line

    @staticmethod
    def node_kind(node: object) -> str:
        """Return an AST node's class name (e.g. ``"AnnAssign"``) without importing ast."""
        return type(node).__name__

    @staticmethod
    def walk_ast_nodes(root: object) -> t.SequenceOf[object]:
        """Recursively yield every AST node reachable from ``root`` via ``_fields``.

        Equivalent to ``ast.walk`` but uses only public attribute access on
        rope-provided AST objects, so no ``import ast`` is needed at the
        consumer layer.
        """
        collected: list[object] = []
        stack: list[object] = [root]
        while stack:
            node = stack.pop()
            collected.append(node)
            for field_name in getattr(node, "_fields", ()):
                value = getattr(node, field_name, None)
                if isinstance(value, list):
                    stack.extend(item for item in value if hasattr(item, "_fields"))
                elif hasattr(value, "_fields"):
                    stack.append(value)
        return collected

    @staticmethod
    def ast_parent_map(root: object) -> dict[int, object]:
        """Return a child-id -> parent map for the full AST reachable from ``root``.

        Uses only public ``_fields`` access (no ``import ast``); the shared SSOT
        for parent lookups across every rope detector.
        """
        parent_map: dict[int, object] = {}
        stack: list[object] = [root]
        while stack:
            parent = stack.pop()
            for field_name in getattr(parent, "_fields", ()):
                value = getattr(parent, field_name, None)
                if isinstance(value, list):
                    for child in value:
                        if hasattr(child, "_fields"):
                            parent_map[id(child)] = parent
                            stack.append(child)
                elif hasattr(value, "_fields"):
                    parent_map[id(value)] = parent
                    stack.append(value)
        return parent_map

    @classmethod
    def is_module_level_node(cls, node: object, parent_map: dict[int, object]) -> bool:
        """Return True when ``node`` is a direct child of the module body.

        Walks the parent chain; a node nested inside any ClassDef/FunctionDef is
        NOT module-level. Shared SSOT for placement detectors.
        """
        current = node
        while True:
            parent = parent_map.get(id(current))
            if parent is None:
                return False
            parent_kind = cls.node_kind(parent)
            if parent_kind in {"ClassDef", "FunctionDef", "AsyncFunctionDef"}:
                return False
            if parent_kind == "Module":
                return True
            current = parent

    @staticmethod
    def name_of(node: object) -> str:
        """Return ``node.id`` (Name) or ``node.attr`` (Attribute) or ``""``."""
        identifier = getattr(node, "id", None)
        if isinstance(identifier, str) and identifier:
            return identifier
        attr = getattr(node, "attr", None)
        if isinstance(attr, str) and attr:
            return attr
        return ""

    @staticmethod
    def line_col_range(node: object) -> tuple[int, int, int, int] | None:
        """Return ``(lineno, col_offset, end_lineno, end_col_offset)`` for an AST node."""
        lineno = getattr(node, "lineno", None)
        col_offset = getattr(node, "col_offset", None)
        end_lineno = getattr(node, "end_lineno", None) or lineno
        end_col_offset = getattr(node, "end_col_offset", None) or col_offset
        if not (
            isinstance(lineno, int)
            and isinstance(col_offset, int)
            and isinstance(end_lineno, int)
            and isinstance(end_col_offset, int)
        ):
            return None
        return (lineno, col_offset, end_lineno, end_col_offset)

    @staticmethod
    def _body_nodes(node: p.AttributeProbe) -> t.SequenceOf[p.AttributeProbe]:
        """Return direct AST body children for a Rope AST node."""
        body = getattr(node, "body", ())
        if not isinstance(body, (list, tuple)):
            return ()
        nodes: list[p.AttributeProbe] = [
            child for child in body if hasattr(child, "_fields")
        ]
        return tuple(nodes)

    @staticmethod
    def _class_body_nodes(
        tree: p.AttributeProbe, *, class_name: str
    ) -> t.SequenceOf[p.AttributeProbe]:
        """Return direct body nodes for a top-level class name."""
        for node in FlextInfraUtilitiesRopeAnalysis._body_nodes(tree):
            if FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "ClassDef":
                continue
            if getattr(node, "name", "") == class_name:
                return FlextInfraUtilitiesRopeAnalysis._body_nodes(node)
        return ()

    @staticmethod
    def assignment_target_names(node: p.AttributeProbe) -> t.StrSequence:
        """Return direct assignment target names represented by one AST node."""
        return FlextInfraUtilitiesRopeAnalysis._assignment_target_names(node)

    @staticmethod
    def _assignment_target_names(node: p.AttributeProbe) -> t.StrSequence:
        """Return direct assignment target names represented by one AST node."""
        node_kind = FlextInfraUtilitiesRopeAnalysis.node_kind(node)
        if node_kind == "AnnAssign":
            target_name = FlextInfraUtilitiesRopeAnalysis.name_of(
                getattr(node, "target", None)
            )
            return (target_name,) if target_name else ()
        if node_kind != "Assign":
            return ()
        targets = getattr(node, "targets", ())
        if not isinstance(targets, (list, tuple)):
            return ()
        names: list[str] = []
        for target in targets:
            target_name = FlextInfraUtilitiesRopeAnalysis.name_of(target)
            if target_name:
                names.append(target_name)
        return tuple(names)

    @staticmethod
    def _class_symbol_names(
        class_body: t.SequenceOf[p.AttributeProbe],
    ) -> t.StrSequence:
        """Return direct method, nested-class and attribute symbols for a class body."""
        names: set[str] = set()
        for node in class_body:
            node_kind = FlextInfraUtilitiesRopeAnalysis.node_kind(node)
            if node_kind in {"AsyncFunctionDef", "ClassDef", "FunctionDef"}:
                node_name = getattr(node, "name", "")
                if isinstance(node_name, str) and node_name:
                    names.add(node_name)
                continue
            names.update(FlextInfraUtilitiesRopeAnalysis._assignment_target_names(node))
        return tuple(sorted(names))

    @staticmethod
    def get_class_info(
        rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
    ) -> t.SequenceOf[m.Infra.ClassInfo]:
        """Return ClassInfo (name, line, bases) for all classes in a module."""
        class_infos: t.SequenceOf[m.Infra.ClassInfo] = (
            FlextInfraUtilitiesRopeAnalysis.get_module_semantic_state(
                rope_project, resource
            ).class_infos
        )
        return class_infos

    @staticmethod
    def class_info_from_source(source: str) -> t.SequenceOf[m.Infra.ClassInfo]:
        """Return class info from the current source text without Rope resource cache."""
        pymodule = FlextInfraUtilitiesRopeAnalysis.parse_string_module(source)
        if pymodule is None:
            return ()
        body = getattr(pymodule.get_ast(), "body", ())
        if not isinstance(body, (list, tuple)):
            return ()
        return tuple(
            class_info
            for node in body
            if (
                class_info := FlextInfraUtilitiesRopeAnalysis._class_info_from_ast(node)
            )
            is not None
        )

    @staticmethod
    def _class_info_from_ast(node: object) -> m.Infra.ClassInfo | None:
        """Return ClassInfo for one top-level ClassDef AST node."""
        if FlextInfraUtilitiesRopeAnalysis.node_kind(node) != "ClassDef":
            return None
        name = getattr(node, "name", "")
        if not isinstance(name, str) or not name:
            return None
        line = getattr(node, "lineno", 1)
        raw_bases = getattr(node, "bases", ())
        if not isinstance(raw_bases, (list, tuple)):
            raw_bases = ()
        return m.Infra.ClassInfo(
            name=name,
            line=line if isinstance(line, int) and line > 0 else 1,
            bases=tuple(
                base_name
                for base in raw_bases
                if (base_name := FlextInfraUtilitiesRopeAnalysis._class_base_name(base))
            ),
        )

    @staticmethod
    def class_base_name(node: object) -> str:
        """Return terminal base name from an AST base expression."""
        return FlextInfraUtilitiesRopeAnalysis._class_base_name(node)

    @staticmethod
    def _class_base_name(node: object) -> str:
        """Return terminal base name from an AST base expression."""
        for attr_name in ("id", "attr", "name"):
            value = getattr(node, attr_name, "")
            if isinstance(value, str) and value:
                return value
        subscript_value = getattr(node, "value", None)
        if subscript_value is not None:
            return FlextInfraUtilitiesRopeAnalysis._class_base_name(subscript_value)
        return ""

    @staticmethod
    def get_class_symbol_count(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        class_name: str,
    ) -> int:
        """Return direct symbol count for a top-level class without semantic imports."""
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(rope_project, resource)
            tree: p.AttributeProbe = pymodule.get_ast()
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
            return 0
        class_body = FlextInfraUtilitiesRopeAnalysis._class_body_nodes(
            tree, class_name=class_name
        )
        return len(FlextInfraUtilitiesRopeAnalysis._class_symbol_names(class_body))

    @staticmethod
    def get_class_bases(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        class_name: str,
    ) -> t.StrSequence:
        """Return base class names for a given class in a module."""
        for info in FlextInfraUtilitiesRopeAnalysis.get_class_info(
            rope_project, resource
        ):
            if info.name == class_name:
                return list(info.bases)
        return ()

    @staticmethod
    def get_class_methods(
        rope_project: t.Infra.RopeProject,
        resource: t.Infra.RopeResource,
        class_name: str,
        *,
        include_private: bool = False,
    ) -> t.StrMapping:
        """Return {method_name: kind} for methods of a class."""
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(rope_project, resource)
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
            return {}
        return FlextInfraUtilitiesRopeAnalysis._class_methods_from_pymodule(
            class_name=class_name, include_private=include_private, pymodule=pymodule
        )

    @staticmethod
    def _class_methods_from_pymodule(
        *, class_name: str, include_private: bool, pymodule: t.Infra.RopePyModule
    ) -> t.StrMapping:
        """Return method symbols for a class from one resolved Rope module."""
        result: t.MutableStrMapping = {}
        attributes = pymodule.get_attributes()
        if class_name not in attributes:
            return result
        obj = attributes[class_name].get_object()
        if not FlextInfraUtilitiesRopeRuntime.is_abstract_class(obj):
            return result
        for name, pyname in obj.get_attributes().items():
            if not include_private and name.startswith("_"):
                continue
            child = pyname.get_object()
            if not FlextInfraUtilitiesRopeRuntime.is_py_function(child):
                continue
            result[name] = child.get_kind()
        return result

    # ------------------------------------------------------------------
    # Path-based rope queries for consumers that previously called
    # ``ast.parse`` on file contents. Each method opens (and closes) a
    # rope project, fetches the cached ``PyModule``, and returns a
    # consumer-shaped result via rope's ``PyObject`` API — no ``ast``
    # parsing, walking, or pattern matching is performed.
    # ------------------------------------------------------------------

    @staticmethod
    def _open_pymodule(
        project_root: Path, file_path: Path
    ) -> tuple[t.Infra.RopePyModule, t.Infra.RopeProject] | None:
        """Open a rope project and resolve ``file_path`` to a ``PyModule``."""
        rope_project = FlextInfraUtilitiesRopeCore.init_rope_project(project_root)
        resource = FlextInfraUtilitiesRopeCore.fetch_python_resource(
            rope_project, file_path
        )
        if resource is None:
            rope_project.close()
            return None
        try:
            pymodule = FlextInfraUtilitiesRopeCore.get_pymodule(rope_project, resource)
        except FlextInfraUtilitiesRopeRuntime.rope_runtime_errors():
            rope_project.close()
            return None
        return pymodule, rope_project

    @classmethod
    def module_has_docstring(cls, project_root: Path, file_path: Path) -> bool:
        """Return whether ``file_path``'s module carries a docstring (rope)."""
        opened = cls._open_pymodule(project_root, file_path)
        if opened is None:
            return False
        pymodule, rope_project = opened
        try:
            return bool(pymodule.get_doc())
        finally:
            rope_project.close()

    @classmethod
    def symbol_has_docstring(
        cls, project_root: Path, file_path: Path, symbol_name: str
    ) -> bool:
        """Return whether ``symbol_name`` carries a docstring via rope."""
        opened = cls._open_pymodule(project_root, file_path)
        if opened is None:
            return False
        pymodule, rope_project = opened
        try:
            pyname = pymodule.get_attributes().get(symbol_name)
            if pyname is None:
                return False
            return bool(pyname.get_object().get_doc())
        finally:
            rope_project.close()

    @classmethod
    def export_target_modules(
        cls,
        project_root: Path,
        file_path: Path,
        package_name: str,
        exports: t.StrSequence,
    ) -> dict[str, str]:
        """Map exports → defining module via rope's import table."""
        export_names = {name for name in exports if name}
        target_map: dict[str, str] = dict.fromkeys(export_names, package_name)
        opened = cls._open_pymodule(project_root, file_path)
        if opened is None:
            return target_map
        pymodule, rope_project = opened
        try:
            resource = pymodule.get_resource()
            if resource is None:
                return target_map
            declared = cls.get_declared_module_imports(rope_project, resource)
            for local_name, declared_path in declared.items():
                if local_name not in export_names:
                    continue
                module_name = (
                    declared_path.rsplit(".", maxsplit=1)[0]
                    if "." in declared_path
                    else declared_path
                )
                if module_name:
                    target_map[local_name] = module_name
        finally:
            rope_project.close()
        return target_map

    @classmethod
    def parent_constants_targets(
        cls,
        constants_file: Path,
        project_root: Path,
        *,
        return_module: bool,
        current_root: str,
    ) -> t.StrSequence:
        """Resolve parent ``Constants`` import targets via rope semantic state.

        Uses ``get_module_semantic_state`` (PyObject-backed class info plus
        ``get_module_imports`` declared-imports table) — no ``ast`` walks.
        """
        opened = cls._open_pymodule(project_root, constants_file)
        if opened is None:
            return ()
        pymodule, rope_project = opened
        try:
            resource = pymodule.get_resource()
            if resource is None:
                return ()
            source_class_bases = {
                class_info.name: class_info.bases
                for class_info in cls.class_info_from_source(resource.read())
            }
            state = cls.get_module_semantic_state(rope_project, resource)
        finally:
            rope_project.close()
        seen: set[str] = set()
        resolved: list[str] = []
        for class_info in state.class_infos:
            if "Constants" not in class_info.name:
                continue
            for base_name in (
                *class_info.bases,
                *source_class_bases.get(class_info.name, ()),
            ):
                full_path = state.declared_imports.get(
                    base_name, ""
                ) or state.declared_imports.get(base_name.split(".", maxsplit=1)[0], "")
                if not full_path:
                    continue
                package_root = full_path.split(".", maxsplit=1)[0]
                if package_root == current_root:
                    continue
                target = (
                    package_root
                    if return_module
                    else full_path.rsplit(".", maxsplit=1)[-1]
                )
                if target and target not in seen:
                    seen.add(target)
                    resolved.append(target)
        return tuple(resolved)


__all__: list[str] = ["FlextInfraUtilitiesRopeAnalysis"]
