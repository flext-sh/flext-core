"""Domain models for rope refactoring operations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from flext_core import m
from flext_infra import c, p, t
from flext_infra._models._defaults import ImmutableEmptyMapping
from flext_infra._models.codegen import FlextInfraModelsCodegen
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsRope:
    """Rope operation result models — accessed via m.Infra.Rope.*."""

    # NOTE (multi-agent, flext-wkii.17.24 / agent: codex): callers resolve project
    # selection once; source iteration has one exact, branch-free request shape.
    class SourceScanRequest(m.ContractModel):
        """Exact project roots selected for one production-source scan."""

        project_roots: Annotated[
            tuple[Path, ...],
            m.Field(
                min_length=1, description="Non-empty ordered project roots to scan"
            ),
        ]

    class ExportOptions(m.ContractModel):
        """Canonical options for Rope module export discovery."""

        include_dunder: Annotated[
            bool, m.Field(description="Whether dunder exports should be returned.")
        ] = False
        allow_main: Annotated[
            bool, m.Field(description="Whether a module-level main() may be exported.")
        ] = False
        allow_assignments: Annotated[
            bool,
            m.Field(description="Whether assignment-backed names may be exported."),
        ] = False
        allow_functions: Annotated[
            bool, m.Field(description="Whether module functions may be exported.")
        ] = False
        require_explicit_all: Annotated[
            bool,
            m.Field(
                description="Whether __all__ must exist for exports to be returned."
            ),
        ] = False

    class ClassInfo(mm.PositiveLineMixin, m.ContractModel):
        """Semantic class info from rope — name, line, bases in one shot."""

        name: Annotated[str, m.Field(description="Class name")]
        bases: Annotated[t.StrSequence, m.Field(description="Base class names")] = ()

    class ScopeDefinition(mm.PositiveLineMixin, m.ContractModel):
        """One semantic scope (def/class) discovered via rope's scope tree.

        Built from ``PyScope.get_kind()``/``get_scopes()`` and the scope's
        ``pyobject.get_name()`` — no ``ast`` walking. ``is_module_level`` is True
        when the scope is a direct child of the module (global) scope.
        """

        name: Annotated[str, m.Field(description="Definition name")]
        kind: Annotated[
            c.Infra.RopeScopeKind,
            m.Field(description="Rope scope kind (Module/Function/Class/Unknown)"),
        ]
        is_module_level: Annotated[
            bool,
            m.Field(description="Whether the scope is a direct child of the module"),
        ]

    class LogicalStatement(mm.PositiveLineMixin, m.ContractModel):
        """One logical statement from the rope structure boundary (no ``ast``).

        Built from a rope ``LogicalLineFinder`` region plus an indent stack over
        the rope-owned source; carries the lexical category, indentation, the
        enclosing def/class scope, and the rope source slice for lexical probes.
        """

        indent: Annotated[
            int, m.Field(ge=0, description="Leading-whitespace column of the statement")
        ]
        # flext-j47u (codex): Rope owns both boundaries so multiline consumers
        # never reconstruct statement ranges from source text.
        end_line: Annotated[
            int, m.Field(ge=1, description="Final line in the Rope logical region")
        ]
        category: Annotated[
            c.Infra.StatementCategory,
            m.Field(description="Lexical category of the leading token"),
        ]
        enclosing_kind: Annotated[
            c.Infra.RopeScopeKind,
            m.Field(description="Kind of the nearest enclosing def/class scope"),
        ] = c.Infra.RopeScopeKind.MODULE
        enclosing_name: Annotated[
            str,
            m.Field(description="Name of the nearest enclosing def/class, or empty"),
        ] = ""
        # flext-j47u (codex): consumers share this Rope-derived guard fact instead
        # of rebuilding TYPE_CHECKING control flow with stdlib AST visitors.
        type_checking_guarded: Annotated[
            bool, m.Field(description="Whether the statement is inside TYPE_CHECKING")
        ] = False
        text: Annotated[
            str, m.Field(description="Rope-owned source slice for the statement")
        ] = ""

    # flext-j47u (codex): normalize Rope payloads before enforcement consumes them.
    class ImportFact(mm.PositiveLineMixin, m.ContractModel):
        """One normalized binding emitted by Rope import-info semantics."""

        module: t.NonEmptyStr = m.Field(description="Imported module path")
        member: str = m.Field(default="", description="Imported member")
        local_name: t.NonEmptyStr = m.Field(description="Bound local name")
        is_from_import: bool = m.Field(description="From-import marker")

    class IgnoredRegion(mm.PositiveLineMixin, m.ContractModel):
        """One Rope-classified string or comment region in source text."""

        start_offset: int = m.Field(ge=0, description="Inclusive offset")
        end_offset: int = m.Field(ge=1, description="Exclusive offset")
        text: t.NonEmptyStr = m.Field(description="Exact source region")
        is_comment: bool = m.Field(description="Comment marker")

    class ConstantInfo(
        mm.NonNegativeLineMixin, mm.NestedClassPathMixin, m.ContractModel
    ):
        """Final-annotated constant definition from rope semantic analysis."""

        name: Annotated[str, m.Field(description="Constant name")]
        annotation: Annotated[str, m.Field(description="Type annotation text")] = ""
        value: Annotated[str, m.Field(description="Value representation")] = ""

    class SymbolInfo(mm.NonNegativeLineMixin, m.ContractModel):
        """Top-level symbol metadata from rope semantic analysis."""

        name: Annotated[str, m.Field(description="Symbol name")]
        kind: Annotated[
            str, m.Field(description="Symbol kind: class, function, assignment")
        ]

    class ModuleSemanticState(m.ContractModel):
        """Unified semantic snapshot for one Rope module analysis pass."""

        class_infos: Annotated[
            tuple[FlextInfraModelsRope.ClassInfo, ...],
            m.Field(description="Local classes discovered in the module"),
        ] = ()
        declared_imports: Annotated[
            t.StrMapping,
            m.Field(
                default_factory=ImmutableEmptyMapping,
                description="Declared import targets by name",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        semantic_imports: Annotated[
            t.StrMapping,
            m.Field(
                default_factory=ImmutableEmptyMapping,
                description="Resolved import targets by name",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)

    class RopeModuleIndexEntry(m.ContractModel):
        """Generic Rope-backed index entry for one Python module resource."""

        file_path: Annotated[Path, m.Field(description="Absolute filesystem path")]
        resource_path: Annotated[
            str, m.Field(description="Rope resource path relative to the project root")
        ]
        module_name: Annotated[
            str, m.Field(description="Fully-qualified Rope module name for this file")
        ]
        package_name: Annotated[
            str,
            m.Field(
                description="Importable package resolved for the containing directory"
            ),
        ]
        package_dir: Annotated[
            Path, m.Field(description="Absolute package directory for this module")
        ]
        project_root: Annotated[
            Path | None,
            m.Field(
                description="Owning project root resolved from the Rope source folder"
            ),
        ] = None
        is_package_init: Annotated[
            bool,
            m.Field(description="Whether this resource is the package __init__.py"),
        ] = False

    class RopePackageIndexEntry(m.ContractModel):
        """Generic Rope-backed package aggregation entry."""

        package_dir: Annotated[
            Path,
            m.Field(description="Absolute package directory represented by this entry"),
        ]
        init_path: Annotated[
            Path,
            m.Field(description="Expected __init__.py path for this package directory"),
        ]
        package_name: Annotated[
            str,
            m.Field(
                description="Importable package name, or empty when non-importable"
            ),
        ]
        project_root: Annotated[
            Path | None,
            m.Field(
                description="Owning project root resolved for this package directory"
            ),
        ] = None
        modules: Annotated[
            tuple[FlextInfraModelsRope.RopeModuleIndexEntry, ...],
            m.Field(
                description="Direct Python module resources that belong to this package"
            ),
        ] = ()
        direct_child_dirs: Annotated[
            tuple[Path, ...],
            m.Field(
                description="Direct child package directories discovered from Rope"
            ),
        ] = ()
        descendant_child_dirs: Annotated[
            tuple[Path, ...],
            m.Field(
                description="All descendant package directories discovered from Rope"
            ),
        ] = ()

    class RopeWorkspaceIndex(m.ContractModel):
        """Generic Rope-backed workspace index for package planning."""

        repository_root: Annotated[
            Path,
            m.Field(
                description="Absolute repository root used to open the Rope project"
            ),
        ]
        package_dirs: Annotated[
            tuple[Path, ...],
            m.Field(
                description="All package directories discovered from Rope resources"
            ),
        ] = ()
        packages_by_dir: Annotated[
            t.MappingKV[str, FlextInfraModelsRope.RopePackageIndexEntry],
            m.Field(
                default_factory=ImmutableEmptyMapping,
                description="Package entries keyed by absolute directory path",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        modules_by_path: Annotated[
            t.MappingKV[str, FlextInfraModelsRope.RopeModuleIndexEntry],
            m.Field(
                default_factory=ImmutableEmptyMapping,
                description="Module entries keyed by absolute file path",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        package_dir_by_name: Annotated[
            t.MappingKV[str, Path],
            m.Field(
                default_factory=ImmutableEmptyMapping,
                description="Importable package directory keyed by package name",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        project_package_by_root: Annotated[
            t.StrMapping,
            m.Field(
                default_factory=ImmutableEmptyMapping,
                description="Canonical source package name keyed by project root path",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)

    class RopeProjectLayout(m.ContractModel):
        """Canonical project layout derived once for Rope-backed codegen flows."""

        project_root: Annotated[Path, m.Field(description="Resolved project root path")]
        project_name: Annotated[str, m.Field(description="Canonical project name")]
        package_name: Annotated[str, m.Field(description="Primary Python package name")]
        package_alias: Annotated[
            str, m.Field(description="Canonical root alias derived from the package")
        ]
        class_stem: Annotated[
            str,
            m.Field(description="Canonical facade class stem derived from the project"),
        ]
        src_dir: Annotated[
            Path, m.Field(description="Resolved source directory for the project")
        ]
        package_dir: Annotated[
            Path, m.Field(description="Resolved package directory for the project")
        ]
        init_path: Annotated[
            Path, m.Field(description="Resolved package __init__.py path")
        ]
        runtime_aliases: Annotated[
            t.StrSequence,
            m.Field(
                description="Canonical runtime aliases published by the package root"
            ),
        ] = ()

    class RopeModuleConvention(m.ContractModel):
        """Unified module naming and namespace convention for one file."""

        file_path: Annotated[Path, m.Field(description="Resolved Python module path")]
        relative_path: Annotated[
            Path, m.Field(description="Module path relative to its package directory")
        ]
        module_name: Annotated[str, m.Field(description="Fully-qualified module name")]
        package_name: Annotated[
            str, m.Field(description="Importable package name for the module")
        ]
        package_dir: Annotated[
            Path,
            m.Field(description="Resolved package directory containing the module"),
        ]
        package_context: Annotated[
            FlextInfraModelsCodegen.LazyInitPackageContext,
            m.Field(description="Resolved lazy-init package context for the module"),
        ]
        module_policy: Annotated[
            FlextInfraModelsCodegen.NamespaceModulePolicy,
            m.Field(description="Canonical module policy derived for the module"),
        ]
        project_layout: Annotated[
            FlextInfraModelsRope.RopeProjectLayout | None,
            m.Field(
                description="Resolved project layout, when the module belongs to one"
            ),
        ] = None

    class RopeInventoryRecordInput(m.ArbitraryTypesModel):
        """Validated payload for building one rope inventory census object."""

        rope_project: Annotated[
            t.Infra.RopeProject,
            m.Field(description="Rope project used for reference discovery"),
        ]
        resource: Annotated[
            t.Infra.RopeResource,
            m.Field(description="Rope resource containing the symbol definition"),
        ]
        source: Annotated[
            str, m.Field(description="Full source text used to compute fingerprints")
        ]
        name: Annotated[str, m.Field(description="Resolved symbol name being recorded")]
        pyname: Annotated[
            t.Infra.RopePyName, m.Field(description="Rope pyname node for the symbol")
        ]
        module_name: Annotated[
            str, m.Field(description="Resolved module name for the symbol")
        ]
        project_name: Annotated[
            str, m.Field(description="Resolved project name for census attribution")
        ]
        convention: Annotated[
            FlextInfraModelsRope.RopeModuleConvention,
            m.Field(description="Module convention metadata for tier/facade checks"),
        ]
        scope_chain: Annotated[
            t.StrSequence,
            m.Field(description="Nested scope path leading to the symbol"),
        ] = ()
        class_chain: Annotated[
            t.StrSequence,
            m.Field(description="Nested class path leading to the symbol"),
        ] = ()
        child_scope: Annotated[
            p.Infra.RopeScopeDsl | None,
            m.Field(description="Optional child scope object from rope"),
        ] = None
        rope_workspace: Annotated[
            p.Infra.RopeWorkspaceDsl | None,
            m.Field(description="Optional shared rope workspace/session handle"),
        ] = None

    class RopeWorkspaceSession(m.ContractModel):
        """Public Rope workspace snapshot used by the service DSL."""

        repository_root: Annotated[
            Path,
            m.Field(description="Resolved repository root requested by the caller"),
        ]
        rope_repository_root: Annotated[
            Path,
            m.Field(description="Canonical root used to open the shared Rope project"),
        ]
        # NOTE (multi-agent, flext-wkii.17.24): policy stays in config.Infra;
        # this field-only model retains only materialized session state.
        workspace_index: Annotated[
            FlextInfraModelsRope.RopeWorkspaceIndex,
            m.Field(description="Materialized workspace index for the open session"),
        ]


__all__: list[str] = ["FlextInfraModelsRope"]
