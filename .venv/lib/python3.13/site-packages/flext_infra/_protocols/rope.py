"""Application-level protocols for Rope-backed infra utilities.

Concrete Rope objects are typed in ``t.Infra.Rope*``. This module keeps only
FLEXT callback contracts that do not have a concrete Rope class.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from types import TracebackType

    from flext_infra import m, p, t


@runtime_checkable
class FlextInfraProtocolsRope(Protocol):
    """Application contracts layered around the concrete Rope boundary."""

    # NOTE (multi-agent, flext-wkii.17.24 / agent: codex): source discovery is a
    # Rope boundary and consumes the exact field-only runtime request model.
    @runtime_checkable
    class SourceScanRequest(Protocol):
        """Fields required by production-source discovery."""

        @property
        def project_roots(self) -> tuple[Path, ...]:
            """Non-empty ordered project roots selected at the boundary."""
            ...

    @runtime_checkable
    class ChangeTracker(Protocol):
        """Transformer contract for in-memory source rewrites with change tracking."""

        changes: t.MutableSequenceOf[str]

        def apply_to_source(self, source: str) -> t.Infra.TransformResult: ...

    @runtime_checkable
    class RopeScopeDsl(Protocol):
        """Public scope contract for Rope semantic traversal."""

        def get_scopes(self) -> t.SequenceOf[FlextInfraProtocolsRope.RopeScopeDsl]: ...

        def get_names(self) -> t.MappingKV[str, t.Infra.RopePyName]: ...

        def get_defined_names(self) -> t.MappingKV[str, t.Infra.RopePyName]: ...

        def get_kind(self) -> str: ...

        def get_start(self) -> int: ...

        def get_end(self) -> int: ...

        @property
        def pyobject(self) -> t.Infra.RopePyObject: ...

    @runtime_checkable
    class RopeWorkspaceDsl(Protocol):
        """Public DSL contract for one shared Rope workspace session."""

        repository_root: Path

        @property
        def rope_repository_root(self) -> Path: ...

        @property
        def rope_project(self) -> t.Infra.RopeProject: ...

        @property
        def workspace_index(self) -> m.Infra.RopeWorkspaceIndex: ...

        def refresh(
            self, *, preserve_indexes: bool = False, validate_project: bool = True
        ) -> m.Infra.RopeWorkspaceSession: ...

        def reload(self) -> m.Infra.RopeWorkspaceSession: ...

        def __enter__(self) -> Self: ...

        def __exit__(
            self,
            _exc_type: type[BaseException] | None,
            _exc: BaseException | None,
            _tb: TracebackType | None,
        ) -> None: ...

        def close(self) -> None: ...

        def resource(self, file_path: Path) -> t.Infra.RopeResource | None: ...

        def module(self, file_path: Path) -> m.Infra.RopeModuleIndexEntry | None: ...

        def package(
            self, package_dir: Path
        ) -> m.Infra.RopePackageIndexEntry | None: ...

        def modules(
            self, *, project_names: t.StrSequence | None = None
        ) -> t.SequenceOf[m.Infra.RopeModuleIndexEntry]: ...

        def source(self, file_path: Path) -> str: ...

        def name_index(
            self,
        ) -> t.MappingKV[str, tuple[tuple[Path, str, tuple[int, ...]], ...]]: ...

        def objects(
            self,
            file_path: Path,
            *,
            include_local_scopes: bool = True,
            include_references: bool = True,
        ) -> t.SequenceOf[m.Infra.Census.Object]: ...

        def projects(self) -> t.SequenceOf[p.Infra.ProjectInfo]: ...

        def layout(self, project_root: Path) -> m.Infra.RopeProjectLayout | None: ...

        def package_context(
            self, package_dir: Path
        ) -> m.Infra.LazyInitPackageContext: ...

        def policy(
            self,
            file_path: Path,
            *,
            rel_path: Path | None = None,
            current_pkg: str = "",
        ) -> m.Infra.NamespaceModulePolicy: ...

        def convention(
            self, file_path: Path, *, rel_path: Path | None = None
        ) -> m.Infra.RopeModuleConvention: ...

        def semantic(self, file_path: Path) -> m.Infra.ModuleSemanticState: ...

        def exports(
            self,
            file_path: Path,
            *,
            export_options: m.Infra.ExportOptions | None = None,
        ) -> t.StrSequence: ...

    @runtime_checkable
    class PatchingASTWalker(Protocol):
        """Structural contract for rope's internal ``_PatchingASTWalker``.

        Used by ``FlextInfraUtilitiesRopePep695Patch`` to install PEP 695
        type-parameter handlers without depending on rope's private class.
        """

        # The two handler slots the PEP 695 patch replaces. Rope exposes them
        # under private names because it has no public registration API; the
        # protocol states that shape so the patch stays statically typed
        # instead of reaching into an untyped probe.
        _handle_function_def_node: Callable[..., None]
        _ClassDef: Callable[..., None]

        # flext-j47u (codex): model Rope node capabilities structurally; the
        # FLEXT static path never imports or traverses Python's AST directly.
        @runtime_checkable
        class PositionedNode(Protocol):
            """Source position exposed by a Rope parser node."""

            lineno: int
            col_offset: int

        @runtime_checkable
        class TypeParameterOwner(Protocol):
            """Rope node carrying PEP 695 type parameters."""

            type_params: t.SequenceOf[p.AttributeProbe]

        @runtime_checkable
        class FunctionDefinitionNode(TypeParameterOwner, Protocol):
            """Function-definition capabilities consumed by the Rope patch."""

            decorator_list: t.SequenceOf[p.AttributeProbe]
            name: str
            args: p.AttributeProbe
            body: t.SequenceOf[p.AttributeProbe]

        @runtime_checkable
        class ClassDefinitionNode(TypeParameterOwner, Protocol):
            """Class-definition capabilities consumed by the Rope patch."""

            decorator_list: t.SequenceOf[p.AttributeProbe]
            name: str
            bases: t.SequenceOf[p.AttributeProbe]
            body: t.SequenceOf[p.AttributeProbe]

        @runtime_checkable
        class TypeAliasNode(TypeParameterOwner, Protocol):
            """Type-alias capabilities consumed by the Rope patch."""

            name: p.AttributeProbe
            value: p.AttributeProbe

        @runtime_checkable
        class TypeVariableNode(Protocol):
            """Bound type-variable capabilities consumed by the Rope patch."""

            name: str
            bound: p.AttributeProbe | None

        @runtime_checkable
        class NamedNode(Protocol):
            """Rope node exposing a name."""

            name: str

        @runtime_checkable
        class MatchSequenceNode(PositionedNode, Protocol):
            """Sequence-pattern capabilities consumed by the Rope patch."""

            patterns: t.SequenceOf[p.AttributeProbe]

        @runtime_checkable
        class MatchSingletonNode(Protocol):
            """Singleton-pattern capabilities consumed by the Rope patch."""

            value: p.AttributeProbe

        @runtime_checkable
        class MatchStarNode(Protocol):
            """Star-pattern capabilities consumed by the Rope patch."""

            name: str | None

        @runtime_checkable
        class MatchOrNode(Protocol):
            """Alternative-pattern capabilities consumed by the Rope patch."""

            patterns: t.SequenceOf[p.AttributeProbe]

        @runtime_checkable
        class SourceLines(Protocol):
            """Minimal line adapter contract exposed by rope patched AST walkers."""

            def get_line_start(self, lineno: int) -> int: ...

        @runtime_checkable
        class SourceBuffer(Protocol):
            """Minimal source buffer contract exposed by rope patched AST walkers."""

            source: str

        lines: FlextInfraProtocolsRope.PatchingASTWalker.SourceLines
        source: FlextInfraProtocolsRope.PatchingASTWalker.SourceBuffer
        empty_tuple: p.AttributeProbe

        def _handle(
            self,
            node: p.AttributeProbe,
            children: list[p.AttributeProbe],
            *,
            eat_parens: bool = False,
            eat_spaces: bool = False,
        ) -> None: ...

        def _child_nodes(
            self, nodes: t.SequenceOf[p.AttributeProbe], separator: str
        ) -> list[p.AttributeProbe]: ...

    @runtime_checkable
    class RopeAnalysisMethods(Protocol):
        """Class contract shared by the Rope analysis mixins."""

        @staticmethod
        def get_module_classes(
            rope_project: t.Infra.RopeProject, resource: t.Infra.RopeResource
        ) -> t.StrSequence: ...

        @staticmethod
        def get_class_methods(
            rope_project: t.Infra.RopeProject,
            resource: t.Infra.RopeResource,
            class_name: str,
            *,
            include_private: bool = False,
        ) -> t.StrMapping: ...

        @staticmethod
        def project_root(file_path: Path) -> Path | None: ...

        @staticmethod
        def init_rope_project(repository_root: Path) -> t.Infra.RopeProject: ...

        @staticmethod
        def get_resource_from_path(
            rope_project: t.Infra.RopeProject, file_path: Path
        ) -> t.Infra.RopeResource | None: ...


__all__: list[str] = ["FlextInfraProtocolsRope"]
