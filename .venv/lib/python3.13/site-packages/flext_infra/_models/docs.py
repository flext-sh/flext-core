"""Domain models for the docs subpackage."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar

from flext_core import m
from flext_infra import c, t
from flext_infra._models.config import FlextInfraConfigModels


class _FlextInfraDocsContracts:
    """Field-only source and rendering contracts for documentation."""

    class DocsExportBinding(m.ContractModel):
        """One public export bound to its defining module."""

        export_name: Annotated[str, m.Field(description="Public export name")]
        module_name: Annotated[str, m.Field(description="Defining module name")]

    class DocsCatalogEntry(m.ContractModel):
        """One project row in the generated workspace catalog."""

        name: Annotated[str, m.Field(description="Project name")]
        project_class: Annotated[str, m.Field(description="Project class")]
        package_name: Annotated[str, m.Field(description="Import package name")]
        description: Annotated[str, m.Field(description="Project description")]
        api_page: Annotated[str, m.Field(description="Generated API page")]

    class DocsProjectIndexEntry(m.ContractModel):
        """One project row in the workspace module index."""

        name: Annotated[str, m.Field(description="Project name")]
        module_count: Annotated[
            t.NonNegativeInt, m.Field(description="Generated module page count")
        ]

    class DocsClassCount(m.ContractModel):
        """Count of discovered projects in one documentation class."""

        project_class: Annotated[str, m.Field(description="Project class")]
        count: Annotated[t.NonNegativeInt, m.Field(description="Project count")]


# NOTE (multi-agent, flext-wkii.17.23 / agent: uv_overlay_owner): docs transport
# retains the exact metadata/config models and declares only analysis deltas.
class FlextInfraModelsDocs(_FlextInfraDocsContracts):
    """Models for documentation services."""

    class DocsGenerateRequest(m.ContractModel):
        """Canonical docs generation request payload.

        Consolidates generate-call parameters into one typed contract so callers
        reuse the same validation rules and avoid ad-hoc multi-parameter calls.
        """

        repository_root: Annotated[
            Path, m.Field(description="Repository root for docs generation")
        ]
        projects: Annotated[
            t.StrSequence | None, m.Field(description="Optional selected project names")
        ] = None
        output_dir: Annotated[
            Path | str | None,
            m.Field(description="Optional docs output directory override"),
        ] = Path(c.Infra.DEFAULT_DOCS_OUTPUT_DIR)
        apply: Annotated[bool, m.Field(description="Apply writes to disk")] = False

    class DocsPhaseItemModel(m.Value):
        """Unified item payload for docs phase reports."""

        phase: Annotated[
            str,
            m.Field(description="Docs phase: audit, fix, build, generate, validate"),
        ]
        file: Annotated[str, m.Field(description="Relative file path")] = ""
        issue_type: Annotated[str, m.Field(description="Audit issue type")] = ""
        severity: Annotated[str, m.Field(description="Audit issue severity")] = ""
        message: Annotated[str, m.Field(description="Item detail message")] = ""
        links: Annotated[
            t.NonNegativeInt, m.Field(description="Applied link fixes")
        ] = 0
        toc: Annotated[t.NonNegativeInt, m.Field(description="Applied TOC updates")] = 0
        codeblocks: Annotated[
            t.NonNegativeInt, m.Field(description="Applied python codeblock fixes")
        ] = 0
        path: Annotated[str, m.Field(description="Generated file path")] = ""
        written: Annotated[bool, m.Field(description="Generated file write flag")] = (
            False
        )

    class DocScope(m.ArbitraryTypesModel):
        """Documentation scope targeting a project or repository root."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Scope name")]
        path: Annotated[Path, m.Field(description="Absolute path to scope root")]
        report_dir: Annotated[
            Path, m.Field(description="Report output directory for scope")
        ]
        project_class: Annotated[
            str, m.Field(description="Docs scope classification")
        ] = "root"
        package_name: Annotated[
            str, m.Field(description="Primary package name for scope")
        ] = ""

    class DocstringCoverage(m.ContractModel):
        """Docstring coverage metric for one docs scope (declaration only).

        ``checked`` counts every public docstring target (package module,
        public modules, exported symbols); ``documented`` counts targets
        that carry a docstring. ``percent`` is a stored value computed by the
        ``u.Infra.docstring_coverage`` factory — the model holds no behavior.
        """

        checked: Annotated[
            t.NonNegativeInt, m.Field(description="Public docstring targets evaluated")
        ]
        documented: Annotated[
            t.NonNegativeInt, m.Field(description="Targets carrying a docstring")
        ]
        percent: Annotated[
            float,
            m.Field(description="Coverage percentage (100.0 when nothing checked)"),
        ]

    class AuditIssue(m.ContractModel):
        """Single documentation audit finding."""

        file: Annotated[str, m.Field(description="File path relative to scope")]
        issue_type: Annotated[str, m.Field(description="Issue category")]
        severity: Annotated[str, m.Field(description="Issue severity")]
        message: Annotated[str, m.Field(description="Issue description")]

    class DocsPublicContract(m.ArbitraryTypesModel):
        """Exact project/config objects plus derived public API analysis."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            arbitrary_types_allowed=True, extra="forbid", frozen=True
        )

        metadata: Annotated[
            m.ProjectMetadata,
            m.Field(description="Exact canonical project metadata model"),
        ]
        repository: Annotated[
            FlextInfraConfigModels.RepositoryRef | None,
            m.Field(description="Exact repository catalog model"),
        ] = None
        provider: Annotated[
            FlextInfraConfigModels.ProviderSpec | None,
            m.Field(description="Exact Git provider model"),
        ] = None
        package_name: Annotated[str, m.Field(description="Documented package name")]
        doc_summary: Annotated[str, m.Field(description="Package docstring summary")]
        site_title: Annotated[str, m.Field(description="Resolved site title")]
        site_url: Annotated[str, m.Field(description="Resolved site URL")]
        repo_url: Annotated[str, m.Field(description="Resolved repository URL")]
        exports: Annotated[t.StrTuple, m.Field(default=(), description="Exports")] = ()
        aliases: Annotated[
            t.StrTuple, m.Field(default=(), description="One-letter alias exports")
        ] = ()
        facades: Annotated[
            t.StrTuple, m.Field(default=(), description="Public facade exports")
        ] = ()
        module_exports: Annotated[
            t.StrTuple, m.Field(default=(), description="Exported module shortcuts")
        ] = ()
        public_symbols: Annotated[
            t.StrTuple, m.Field(default=(), description="Rope-resolved public symbols")
        ] = ()
        export_bindings: Annotated[
            tuple[_FlextInfraDocsContracts.DocsExportBinding, ...],
            m.Field(default=(), description="Export-to-module bindings"),
        ] = ()
        modules: Annotated[
            t.StrTuple, m.Field(default=(), description="Generated public module pages")
        ] = ()
        source_paths: Annotated[
            t.StrTuple, m.Field(default=(), description="Mkdocstrings source roots")
        ] = ()

    class GeneratedFile(m.ContractModel):
        """Record of a generated file operation."""

        path: Annotated[str, m.Field(description="File path")]
        changed: Annotated[
            bool, m.Field(description="Whether generated content differs on disk")
        ]
        written: Annotated[bool, m.Field(description="Whether file was written")] = (
            False
        )

    class AuditScopeParams(m.ContractModel):
        """Bundled parameters for a single audit scope run."""

        check: Annotated[str, m.Field(description="Comma-separated checks")] = "all"
        strict: Annotated[bool, m.Field(description="Strict mode")] = True
        docstring_min: Annotated[
            float | None,
            m.Field(
                description="Minimum docstring coverage percent; breach fails the scope"
            ),
        ] = None
        budgets: Annotated[
            tuple[int | None, t.IntMapping] | None,
            m.Field(description="Budget tuple (default, by_scope)"),
        ] = None

    class DocsPhaseReport(m.ContractModel):
        """Unified report payload for docs phases (declaration only)."""

        phase: Annotated[
            str,
            m.Field(description="Docs phase: audit, fix, build, generate, validate"),
        ]
        scope: Annotated[str, m.Field(description="Scope name")]
        result: Annotated[str, m.Field(description="Result status")] = ""
        reason: Annotated[str, m.Field(description="Result reason")] = ""
        message: Annotated[
            str, m.Field(description="Human-readable summary message")
        ] = ""
        site_dir: Annotated[str, m.Field(description="Built site directory path")] = ""
        checks: Annotated[t.StrSequence, m.Field(description="Executed checks")] = (
            m.Field(default_factory=tuple)
        )
        strict: Annotated[bool, m.Field(description="Strict-mode flag")] = False
        passed: Annotated[bool, m.Field(description="Whether phase passed")] = False
        changed_files: Annotated[
            t.NonNegativeInt, m.Field(description="Changed files count")
        ] = 0
        applied: Annotated[bool, m.Field(description="Apply mode flag")] = False
        generated: Annotated[
            t.NonNegativeInt, m.Field(description="Generated files count")
        ] = 0
        source: Annotated[
            str, m.Field(description="Source marker for generated content")
        ] = ""
        missing_adr_skills: Annotated[
            t.StrSequence, m.Field(description="Missing ADR skill references")
        ] = m.Field(default_factory=tuple)
        todo_written: Annotated[
            bool, m.Field(description="Whether TODOS.md was written")
        ] = False
        items: Annotated[
            t.SequenceOf[FlextInfraModelsDocs.DocsPhaseItemModel],
            m.Field(description="Phase-specific item payloads"),
        ] = m.Field(default_factory=tuple)


__all__: list[str] = ["FlextInfraModelsDocs"]
