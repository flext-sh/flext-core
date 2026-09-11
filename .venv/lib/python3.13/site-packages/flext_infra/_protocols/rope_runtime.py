"""Structural protocols for the external Rope runtime boundary.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    # flext-j47u (codex): retained only until the remaining get_ast consumers are
    # converted atomically; this import never enters the runtime dependency graph.
    import ast

    from flext_infra import p, t


@runtime_checkable
class FlextInfraProtocolsRopeRuntime(Protocol):
    """Rope object contracts used instead of importing untyped Rope classes."""

    @runtime_checkable
    class RopeRoot(Protocol):
        """Rope project root shape."""

        real_path: str

    @runtime_checkable
    class RopeResource(Protocol):
        """Rope file resource shape."""

        path: str
        real_path: str

        def read(self) -> str: ...

        def write(self, contents: str) -> None: ...

    @runtime_checkable
    class RopePyObject(Protocol):
        """Rope semantic object shape."""

        def get_attribute(
            self, name: str
        ) -> FlextInfraProtocolsRopeRuntime.RopePyName: ...

        def get_attributes(
            self,
        ) -> t.MappingKV[str, FlextInfraProtocolsRopeRuntime.RopePyName]: ...

        def get_doc(self) -> str | None: ...

        def get_name(self) -> str: ...

        def get_kind(self) -> str: ...

        def get_scope(self) -> FlextInfraProtocolsRopeRuntime.RopeScope | None: ...

        def get_superclasses(
            self,
        ) -> t.SequenceOf[FlextInfraProtocolsRopeRuntime.RopePyObject]: ...

    @runtime_checkable
    class RopeAssignment(Protocol):
        """Rope assignment shape."""

        ast_node: ast.AST

    @runtime_checkable
    class RopePyName(Protocol):
        """Rope semantic name shape."""

        def get_object(self) -> FlextInfraProtocolsRopeRuntime.RopePyObject: ...

        def get_definition_location(
            self,
        ) -> tuple[FlextInfraProtocolsRopeRuntime.RopePyModule | None, int | None]: ...

    @runtime_checkable
    class RopeAssignedName(RopePyName, Protocol):
        """Rope assigned-name marker shape."""

        assignments: t.SequenceOf[FlextInfraProtocolsRopeRuntime.RopeAssignment]

    @runtime_checkable
    class RopeScope(Protocol):
        """Rope semantic scope shape."""

        def get_scopes(
            self,
        ) -> t.SequenceOf[FlextInfraProtocolsRopeRuntime.RopeScope]: ...

        def get_names(
            self,
        ) -> t.MappingKV[str, FlextInfraProtocolsRopeRuntime.RopePyName]: ...

        def get_defined_names(
            self,
        ) -> t.MappingKV[str, FlextInfraProtocolsRopeRuntime.RopePyName]: ...

        def get_kind(self) -> str: ...

        def get_start(self) -> int: ...

        def get_end(self) -> int: ...

        @property
        def pyobject(self) -> FlextInfraProtocolsRopeRuntime.RopePyObject: ...

    @runtime_checkable
    class RopePyModule(Protocol):
        """Rope parsed module shape."""

        source_code: str

        def get_attribute(
            self, name: str
        ) -> FlextInfraProtocolsRopeRuntime.RopePyName: ...

        def get_name(self) -> str: ...

        def get_doc(self) -> str | None: ...

        # NOTE (multi-agent, flext-f8vk / kimi): rope returns None for
        # string-parsed modules (pycore.get_string_module(resource=None));
        # the widened contract keeps every consumer None guard live.
        def get_resource(
            self,
        ) -> FlextInfraProtocolsRopeRuntime.RopeResource | None: ...

        def get_attributes(
            self,
        ) -> t.MappingKV[str, FlextInfraProtocolsRopeRuntime.RopePyName]: ...

        def get_ast(self) -> ast.AST: ...

        def get_scope(self) -> FlextInfraProtocolsRopeRuntime.RopeScope | None: ...

    @runtime_checkable
    class RopeProject(Protocol):
        """Rope project shape used by flext-infra."""

        root: FlextInfraProtocolsRopeRuntime.RopeRoot

        def get_resource(
            self, path: str
        ) -> FlextInfraProtocolsRopeRuntime.RopeResource: ...

        def get_pymodule(
            self, resource: FlextInfraProtocolsRopeRuntime.RopeResource
        ) -> FlextInfraProtocolsRopeRuntime.RopePyModule: ...

        def find_module(
            self, module_name: str
        ) -> FlextInfraProtocolsRopeRuntime.RopeResource | None: ...

        def get_module(
            self,
            module_name: str,
            folder: FlextInfraProtocolsRopeRuntime.RopeRoot | None = None,
        ) -> FlextInfraProtocolsRopeRuntime.RopePyModule: ...

        def get_python_files(
            self,
        ) -> t.SequenceOf[FlextInfraProtocolsRopeRuntime.RopeResource]: ...

        def do(self, changes: FlextInfraProtocolsRopeRuntime.RopeChangeSet) -> None: ...

        def validate(
            self, root: FlextInfraProtocolsRopeRuntime.RopeRoot | None = None
        ) -> None: ...

        def close(self) -> None: ...

    @runtime_checkable
    class RopeLocation(Protocol):
        """Rope occurrence location shape."""

        # flext-j47u (codex): detectors consume Rope's semantic occurrence
        # coordinates directly instead of rediscovering references textually.
        resource: FlextInfraProtocolsRopeRuntime.RopeResource | None
        offset: int
        lineno: int

    @runtime_checkable
    class RopeImportInfo(Protocol):
        """Common Rope import-info shape."""

        names_and_aliases: t.SequenceOf[tuple[str, str | None]]

    @runtime_checkable
    class RopeFromImport(RopeImportInfo, Protocol):
        """Rope from-import marker shape."""

        module_name: str
        level: int

    @runtime_checkable
    class RopeNormalImport(RopeImportInfo, Protocol):
        """Rope normal-import marker shape."""

    @runtime_checkable
    class RopeImportStatement(Protocol):
        """Rope import statement shape."""

        import_info: FlextInfraProtocolsRopeRuntime.RopeImportInfo
        start_line: int
        end_line: int

    @runtime_checkable
    class RopeWorder(Protocol):
        """Rope word/call classifier consumed by the static fact engine."""

        def get_primary_at(self, offset: int) -> str: ...

        def is_a_function_being_called(self, offset: int) -> bool: ...

        def get_word_parens_range(self, offset: int) -> tuple[int, int]: ...

        def is_function_keyword_parameter(self, offset: int) -> bool: ...

    @runtime_checkable
    class RopeChangeSet(Protocol):
        """Rope change set shape."""

        changes: list[p.AttributeProbe]

    @runtime_checkable
    class RopeModuleImports(Protocol):
        """Rope mutable module-import collection shape."""

        imports: list[FlextInfraProtocolsRopeRuntime.RopeImportStatement]

        def add_import(
            self, import_info: FlextInfraProtocolsRopeRuntime.RopeImportInfo
        ) -> None: ...

        def remove_duplicates(self) -> None: ...

        def sort_imports(self) -> None: ...

        def get_changed_source(self) -> str: ...

    @runtime_checkable
    class RopeImportOrganizer(Protocol):
        """Rope import organizer shape."""

        def organize_imports(
            self, resource: FlextInfraProtocolsRopeRuntime.RopeResource
        ) -> FlextInfraProtocolsRopeRuntime.RopeChangeSet | None: ...

    @runtime_checkable
    class RopeMoveGlobal(Protocol):
        """Rope MoveGlobal refactoring shape."""

        def get_changes(
            self, target: FlextInfraProtocolsRopeRuntime.RopeResource
        ) -> FlextInfraProtocolsRopeRuntime.RopeChangeSet: ...

    @runtime_checkable
    class RopeOccurrence(Protocol):
        """Rope occurrence shape."""

        offset: int

        def get_word_range(self) -> tuple[int, int]: ...

    @runtime_checkable
    class RopeOccurrenceFinder(Protocol):
        """Rope occurrence finder shape."""

        def find_occurrences(
            self, *, resource: FlextInfraProtocolsRopeRuntime.RopeResource
        ) -> t.SequenceOf[FlextInfraProtocolsRopeRuntime.RopeOccurrence]: ...


__all__: list[str] = ["FlextInfraProtocolsRopeRuntime"]
