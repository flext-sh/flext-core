"""Domain models for the refactor subpackage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import MutableMapping, MutableSequence, MutableSet
from pathlib import Path
from typing import Annotated, ClassVar

from flext_core import m
from flext_core import u
from flext_infra import t
from flext_infra._models.mixins import FlextInfraModelsMixins as mm
from flext_infra._models.refactor_ast_grep import FlextInfraModelsRefactorGrep
from flext_infra._models.refactor_census import FlextInfraModelsRefactorCensus
from flext_infra._models.refactor_namespace_enforcer import (
    FlextInfraModelsNamespaceEnforcer,
)
from flext_infra._models.refactor_renames import FlextInfraModelsRefactorRenames
from flext_infra._models.refactor_violations import FlextInfraModelsRefactorViolations


class FlextInfraModelsRefactor(
    FlextInfraModelsRefactorGrep,
    FlextInfraModelsNamespaceEnforcer,
    FlextInfraModelsRefactorCensus,
    FlextInfraModelsRefactorRenames,
    FlextInfraModelsRefactorViolations,
):
    """Models for refactor workflows and related tools.

    Canonical base policy:
    - ``ContractModel`` for configuration/policy contracts.
    - ``ArbitraryTypesModel`` for mutable report/result payloads.
    """

    class RefactorNamespaceEnforceInput(mm.WriteMixin, m.ContractModel):
        """CLI/service request for namespace enforcement."""

    class ModernizeInput(mm.WriteMixin, m.ContractModel):
        """CLI/service request for generic modernize transformers."""

    class AccessorMigrationInput(mm.WriteMixin, m.ContractModel):
        """CLI/service request for accessor migration dry-runs and applies."""

        preview_limit: Annotated[
            t.PositiveInt,
            m.Field(
                alias="preview-limit",
                description="Maximum number of file previews to include in the report",
            ),
        ] = 10

    class Result(m.ArbitraryTypesModel):
        """Result of applying refactor rules to a single file."""

        file_path: Annotated[Path, m.Field(description="Target file path")]
        success: Annotated[bool, m.Field(description="Whether the operation succeeded")]
        modified: Annotated[
            bool, m.Field(description="Whether the file was actually modified")
        ]
        error: Annotated[
            str | None, m.Field(description="Error message on failure")
        ] = None
        changes: Annotated[
            t.StrSequence, m.Field(description="Human-readable change descriptions")
        ] = m.Field(default_factory=tuple)
        refactored_code: Annotated[
            str | None,
            m.Field(description="Resulting source code after transformation"),
        ] = None

    class RefactorProjectInfo(m.ArbitraryTypesModel):
        """Project metadata with mutable package-root set.

        Enforcement exemption: ``package_roots`` is a ``set[str]``
        accumulator populated during scanning.
        """

        name: Annotated[t.NonEmptyStr, m.Field(description="Project directory name")]
        path: Annotated[Path, m.Field(description="Absolute project path")]
        src_path: Annotated[Path, m.Field(description="Absolute src/ path")]
        package_roots: Annotated[
            MutableSet[str],
            m.Field(description="Top-level Python package roots in src/"),
        ] = m.Field(default_factory=set)

    class FileImportData(m.ArbitraryTypesModel):
        """File-level import data with mutable set accumulators.

        Enforcement exemption: ``imported_modules``/``imported_symbols``
        accumulate during a scan; keep ``set[str]`` as the declared type.
        """

        imported_modules: Annotated[
            MutableSet[str], m.Field(description="Imported module roots")
        ] = m.Field(default_factory=set)
        imported_symbols: Annotated[
            MutableSet[str], m.Field(description="Imported symbol names")
        ] = m.Field(default_factory=set)

    class MethodInfo(m.ArbitraryTypesModel):
        """Metadata about a method used for ordering inside classes."""

        name: Annotated[t.NonEmptyStr, m.Field(description="Method name")]
        category: Annotated[str, m.Field(description="Method category classification")]
        node: Annotated[
            t.Infra.RopePyObject | None,
            m.Field(
                description="Node representation from Rope or PyObject", exclude=True
            ),
        ]
        decorators: Annotated[
            t.StrSequence, m.Field(description="Decorator names applied to this method")
        ] = m.Field(default_factory=tuple)

    class Checkpoint(mm.CheckpointRefMixin, m.ArbitraryTypesModel):
        """Serialisable checkpoint state for refactor safety recovery."""

        repository_root: Annotated[
            t.NonEmptyStr, m.Field(description="Repository root path")
        ]
        status: Annotated[str, m.Field(description="Checkpoint status")] = "running"
        processed_targets: Annotated[
            t.StrSequence, m.Field(description="Already-processed file targets")
        ] = m.Field(default_factory=tuple)
        updated_at: Annotated[
            str, m.Field(description="ISO 8601 timestamp of last update")
        ] = m.Field(default_factory=lambda: u.now().isoformat())

    class ClassOccurrence(m.ArbitraryTypesModel):
        """A single class definition occurrence within a source file."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        name: Annotated[t.NonEmptyStr, m.Field(description="Class name")]
        line: Annotated[
            t.NonNegativeInt, m.Field(description="Line number (0 = unknown)")
        ]
        is_top_level: Annotated[
            bool, m.Field(description="Whether class is at module top level")
        ]

    class LooseClassViolation(m.ArbitraryTypesModel):
        """A detected loose-class naming violation with confidence."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        file: Annotated[t.NonEmptyStr, m.Field(description="Source file path")]
        line: Annotated[t.PositiveInt, m.Field(description="Line number")]
        class_name: Annotated[
            t.NonEmptyStr, m.Field(description="Violating class name")
        ]
        expected_prefix: Annotated[
            str, m.Field(description="Expected namespace prefix")
        ]
        rule: Annotated[t.NonEmptyStr, m.Field(description="Violated rule id")]
        reason: Annotated[str, m.Field(description="Human-readable reason")]
        confidence: Annotated[str, m.Field(description="Confidence level")]
        score: Annotated[t.DecimalFraction, m.Field(description="Confidence score")]

    class ProjectClassification(m.ArbitraryTypesModel):
        """Result of classifying a project by kind and family chains."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        project_kind: Annotated[
            t.NonEmptyStr,
            m.Field(
                description="Project kind (core, domain, platform, integration, app)"
            ),
        ]
        family_chains: Annotated[
            t.MappingKV[str, t.StrSequence],
            m.Field(description="Family letter to FLEXT chain mapping"),
        ]

    # NOTE (multi-agent): the two models below replace the
    # dataclass payloads that lived in refactor/_wrapper_rewrite.py and
    # refactor/classvar_constant_autofix.py (deep-FLEXT: models only in m).
    class WrapperRewriteAccumulator(m.ArbitraryTypesModel):
        """Aggregates per-file rewrite stats across the wrapper-root verb run.

        Mutable accumulator (never frozen): every field is appended to or
        incremented while the run scans candidate files.
        """

        updates: Annotated[
            MutableMapping[Path, str],
            m.Field(description="Pending file content updates keyed by path"),
        ] = m.Field(default_factory=dict)
        wrapper_candidates: Annotated[
            MutableSequence[Path],
            m.Field(description="Files that carry wrapper import candidates"),
        ] = m.Field(default_factory=list)
        changed_files: Annotated[
            MutableSequence[str],
            m.Field(description="String paths of files changed by the run"),
        ] = m.Field(default_factory=list)
        total_replacements: Annotated[
            int, m.Field(description="Total replacements applied across the run")
        ] = 0
        total_core_replacements: Annotated[
            int, m.Field(description="Total Core.Tests chain rewrites applied")
        ] = 0
        import_rewrite_candidates: Annotated[
            int, m.Field(description="Count of wrapper import rewrite candidates")
        ] = 0
        per_project_changes: Annotated[
            defaultdict[str, int],
            m.Field(description="Changed file count keyed by project name"),
        ] = m.Field(default_factory=lambda: defaultdict(int))
        per_project_replacements: Annotated[
            defaultdict[str, int],
            m.Field(description="Replacement count keyed by project name"),
        ] = m.Field(default_factory=lambda: defaultdict(int))

    class ClassvarConstantAutofixPlan(m.ArbitraryTypesModel):
        """Planned edits for one ENFORCE-079 ClassVar-constant autofix."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        class_module: Annotated[
            str, m.Field(description="Module that declares the owning class")
        ]
        class_name: Annotated[
            str, m.Field(description="Class that currently declares the constant")
        ]
        constant_name: Annotated[
            str, m.Field(description="Name of the ClassVar constant being moved")
        ]
        constants_module: Annotated[
            str,
            m.Field(description="Canonical _constants module receiving the constant"),
        ]
        source_resource: Annotated[
            t.Infra.RopeResource,
            m.Field(description="Rope resource for the class module"),
        ]
        target_resource: Annotated[
            t.Infra.RopeResource,
            m.Field(description="Rope resource for the constants module"),
        ]
        declaration_line: Annotated[
            str, m.Field(description="Exact source line that declares the constant")
        ]
        class_lineno: Annotated[
            int, m.Field(description="1-based line where the class starts")
        ]

    class ClassvarConstantAutofixResult(m.ArbitraryTypesModel):
        """Outcome of one ENFORCE-079 ClassVar-constant autofix.

        Dry-run populates ``source_text``/``target_text``/``rewrites``; apply
        populates ``constant_module``. ``touched_files`` is always present.
        """

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        touched_files: Annotated[
            t.StrSequence, m.Field(description="Files the autofix created or rewrote")
        ]
        source_text: Annotated[
            str | None,
            m.Field(description="Rewritten source-module preview (dry-run only)"),
        ] = None
        target_text: Annotated[
            str | None,
            m.Field(description="Rewritten constants-module preview (dry-run only)"),
        ] = None
        constant_module: Annotated[
            str | None, m.Field(description="Canonical _constants module (apply only)")
        ] = None
        rewrites: Annotated[
            t.MappingKV[str, t.SequenceOf[tuple[int, int, str]]] | None,
            m.Field(description="Per-file textual edits planned (dry-run only)"),
        ] = None

    # -- Namespace Enforcer Models ---------------------------------------------

    class ParsedPythonModule(m.ArbitraryTypesModel):
        """Result of parsing a Python source file into AST."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        source: Annotated[str, m.Field(description="Raw source text")]
        tree: Annotated[
            t.Infra.RopePyModule,
            m.Field(description="Parsed PyObject module representation"),
        ]


__all__: list[str] = ["FlextInfraModelsRefactor"]
