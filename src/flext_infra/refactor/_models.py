"""Domain models for the refactor subpackage."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import libcst as cst
from pydantic import ConfigDict, Field

from flext_core import FlextModels
from flext_infra.refactor._models_ast_grep import FlextInfraRefactorAstGrepModels
from flext_infra.refactor._models_namespace_enforcer import (
    FlextInfraNamespaceEnforcerModels,
)


class FlextInfraRefactorModels(
    FlextInfraRefactorAstGrepModels,
    FlextInfraNamespaceEnforcerModels,
):
    """Models for the refactor engine and related tools.

    Canonical base policy:
    - ``FrozenStrictModel`` for configuration/policy contracts.
    - ``ArbitraryTypesModel`` for mutable engine/report/result payloads.
    """

    class Result(FlextModels.ArbitraryTypesModel):
        """Result of applying refactor rules to a single file."""

        file_path: Annotated[Path, Field(description="Target file path")]
        success: Annotated[bool, Field(description="Whether the operation succeeded")]
        modified: Annotated[
            bool, Field(description="Whether the file was actually modified")
        ]
        error: Annotated[
            str | None, Field(default=None, description="Error message on failure")
        ] = None
        changes: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Human-readable change descriptions",
            ),
        ]
        refactored_code: Annotated[
            str | None,
            Field(
                default=None,
                description="Resulting source code after transformation",
            ),
        ]

    class ProjectInfo(FlextModels.ArbitraryTypesModel):
        name: Annotated[str, Field(min_length=1, description="Project directory name")]
        path: Annotated[Path, Field(description="Absolute project path")]
        src_path: Annotated[Path, Field(description="Absolute src/ path")]
        package_roots: Annotated[
            set[str],
            Field(
                default_factory=set,
                description="Top-level Python package roots in src/",
            ),
        ]

    class FileImportData(FlextModels.ArbitraryTypesModel):
        imported_modules: Annotated[
            set[str],
            Field(
                default_factory=set,
                description="Imported module roots",
            ),
        ]
        imported_symbols: Annotated[
            set[str],
            Field(
                default_factory=set,
                description="Imported symbol names",
            ),
        ]

    class MethodInfo(FlextModels.ArbitraryTypesModel):
        """Metadata about a method used for ordering inside classes."""

        name: Annotated[str, Field(min_length=1, description="Method name")]
        category: Annotated[str, Field(description="Method category classification")]
        node: Annotated[
            cst.FunctionDef,
            Field(
                description="LibCST FunctionDef node",
                exclude=True,
            ),
        ]
        decorators: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Decorator names applied to this method",
            ),
        ]

    class Checkpoint(FlextModels.ArbitraryTypesModel):
        """Serialisable checkpoint state for refactor safety recovery."""

        workspace_root: Annotated[
            str, Field(min_length=1, description="Workspace root path")
        ]
        status: Annotated[
            str, Field(default="running", description="Checkpoint status")
        ] = "running"
        stash_ref: Annotated[
            str, Field(default="", description="Git stash reference")
        ] = ""
        processed_targets: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Already-processed file targets",
            ),
        ]
        updated_at: Annotated[
            str,
            Field(
                default_factory=lambda: datetime.now(UTC).isoformat(),
                description="ISO 8601 timestamp of last update",
            ),
        ] = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    class ClassOccurrence(FlextModels.ArbitraryTypesModel):
        """A single class definition occurrence within a source file."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(min_length=1, description="Class name")]
        line: Annotated[int, Field(ge=0, description="Line number (0 = unknown)")]
        is_top_level: Annotated[
            bool, Field(description="Whether class is at module top level")
        ]

    class LooseClassViolation(FlextModels.ArbitraryTypesModel):
        """A detected loose-class naming violation with confidence."""

        model_config = ConfigDict(frozen=True)

        file: Annotated[str, Field(min_length=1, description="Source file path")]
        line: Annotated[int, Field(ge=1, description="Line number")]
        class_name: Annotated[
            str, Field(min_length=1, description="Violating class name")
        ]
        expected_prefix: Annotated[str, Field(description="Expected namespace prefix")]
        rule: Annotated[str, Field(min_length=1, description="Violated rule id")]
        reason: Annotated[str, Field(description="Human-readable reason")]
        confidence: Annotated[str, Field(description="Confidence level")]
        score: Annotated[float, Field(ge=0.0, le=1.0, description="Confidence score")]

    class FamilyMROResolution(FlextModels.ArbitraryTypesModel):
        """Resolution payload for one facade family MRO."""

        model_config = ConfigDict(frozen=True)

        family: Annotated[str, Field(min_length=1, description="Facade family letter")]
        expected_bases: Annotated[
            tuple[str, ...],
            Field(
                description="Expected base class names in order",
            ),
        ]
        resolved_mro: Annotated[
            tuple[str, ...], Field(description="Resolved MRO class names")
        ]
        accessible_namespaces: Annotated[
            tuple[str, ...],
            Field(
                description="Namespaces accessible through the MRO",
            ),
        ]

    class ProjectClassification(FlextModels.ArbitraryTypesModel):
        """Result of classifying a project by kind and family chains."""

        model_config = ConfigDict(frozen=True)

        project_kind: Annotated[
            str,
            Field(
                min_length=1,
                description="Project kind (core, domain, platform, integration, app)",
            ),
        ]
        family_chains: Annotated[
            dict[str, list[str]],
            Field(
                description="Family letter to MRO chain mapping",
            ),
        ]

    class ClassNestingMapping(FlextModels.ArbitraryTypesModel):
        """Unified mapping contract for class-nesting rewrite planning."""

        model_config = ConfigDict(extra="ignore", frozen=True)

        loose_name: Annotated[
            str, Field(default="", description="Original loose class name")
        ] = ""
        current_file: Annotated[
            str, Field(default="", description="File containing class")
        ] = ""
        target_namespace: Annotated[
            str,
            Field(
                min_length=1,
                description="Target namespace class name",
            ),
        ]
        target_name: Annotated[
            str, Field(default="", description="Target class name")
        ] = ""
        confidence: Annotated[str, Field(min_length=1, description="Confidence level")]
        reason: Annotated[
            str, Field(default="", description="Optional mapping rationale")
        ] = ""
        rewrite_scope: Annotated[
            str | None,
            Field(
                default=None,
                description="Rewrite scope (file/project/workspace)",
            ),
        ] = None

    class ClassNestingViolation(FlextModels.ArbitraryTypesModel):
        """Normalized class-nesting violation with rewrite metadata."""

        model_config = ConfigDict(frozen=True)

        file: Annotated[str, Field(min_length=1, description="Source module path")]
        line: Annotated[int, Field(ge=1, description="Line number")]
        class_name: Annotated[str, Field(min_length=1, description="Class name")]
        target_namespace: Annotated[
            str,
            Field(
                default="",
                description="Expected namespace class",
            ),
        ] = ""
        confidence: Annotated[
            str, Field(default="low", description="Confidence level")
        ] = "low"
        rewrite_scope: Annotated[
            str,
            Field(
                default="file",
                description="Rewrite scope",
            ),
        ] = "file"

    class ClassNestingPolicy(FlextModels.FrozenStrictModel):
        """Validated policy contract used by class-nesting transformers."""

        model_config = ConfigDict(extra="ignore", frozen=True)

        family_name: Annotated[
            str, Field(min_length=1, description="Module family name")
        ]
        allowed_operations: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Enabled operation identifiers for this family",
            ),
        ]
        forbidden_operations: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Disabled operation identifiers for this family",
            ),
        ]
        forbidden_targets: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Target namespaces forbidden for this family",
            ),
        ]
        enable_class_nesting: Annotated[
            bool,
            Field(
                default=True,
                description="Allow moving top-level classes under a namespace",
            ),
        ]
        allow_namespace_creation: Annotated[
            bool,
            Field(
                default=True,
                description="Allow creating a target namespace when absent",
            ),
        ]
        allow_existing_namespace_merge: Annotated[
            bool,
            Field(
                default=True,
                description="Allow merging nested classes into existing namespace",
            ),
        ]
        enable_helper_consolidation: Annotated[
            bool,
            Field(
                default=True,
                description="Allow consolidating helper functions into namespaces",
            ),
        ]
        allow_helper_call_rewrite: Annotated[
            bool,
            Field(
                default=True,
                description="Allow rewriting helper call sites to namespaced calls",
            ),
        ]
        require_signature_validation: Annotated[
            bool,
            Field(
                default=False,
                description="Require signature checks before helper migration",
            ),
        ]
        required_parameters: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Function parameters that must exist in helper signatures",
            ),
        ]
        forbidden_parameters: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Function parameters that must not exist in helper signatures",
            ),
        ]
        allow_vararg: Annotated[
            bool,
            Field(
                default=True,
                description="Allow variadic positional parameter usage",
            ),
        ]
        allow_kwarg: Annotated[
            bool,
            Field(
                default=True,
                description="Allow variadic keyword parameter usage",
            ),
        ]
        allow_positional_only_params: Annotated[
            bool,
            Field(
                default=True,
                description="Allow positional-only parameters",
            ),
        ]
        allow_keyword_only_params: Annotated[
            bool,
            Field(
                default=True,
                description="Allow keyword-only parameters",
            ),
        ]
        propagate_imports: Annotated[
            bool,
            Field(
                default=True,
                description="Allow propagating import rewrite rules",
            ),
        ]
        propagate_name_references: Annotated[
            bool,
            Field(
                default=True,
                description="Allow propagating direct name reference rewrites",
            ),
        ]
        propagate_attribute_references: Annotated[
            bool,
            Field(
                default=True,
                description="Allow propagating attribute reference rewrites",
            ),
        ]
        blocked_reference_prefixes: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Name prefixes blocked from rewrite propagation",
            ),
        ]
        allowed_targets: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Explicitly allowed target namespaces",
            ),
        ]

    class ClassNestingReport(FlextModels.ArbitraryTypesModel):
        """Aggregated class-nesting analysis report."""

        violations_count: Annotated[int, Field(ge=0, description="Total violations")]
        confidence_counts: Annotated[
            dict[str, int],
            Field(
                default_factory=dict,
                description="Confidence histogram",
            ),
        ]
        violations: Annotated[
            list[FlextInfraRefactorModels.ClassNestingViolation],
            Field(
                default_factory=lambda: list[
                    FlextInfraRefactorModels.ClassNestingViolation
                ](),
                description="Violation details",
            ),
        ]
        per_file_counts: Annotated[
            dict[str, int],
            Field(
                default_factory=dict,
                description="Violation counts per file",
            ),
        ]

    class HelperClassification(FlextModels.ArbitraryTypesModel):
        """Classification result for a helper function."""

        file: Annotated[str, Field(min_length=1, description="Source file")]
        function: Annotated[str, Field(min_length=1, description="Function name")]
        category: Annotated[str, Field(min_length=1, description="Assigned category")]
        target_namespace: Annotated[
            str,
            Field(
                min_length=1,
                description="Target namespace path",
            ),
        ]
        dependencies: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Imported dependencies used by function",
            ),
        ]
        manual_review: Annotated[
            bool,
            Field(
                default=False,
                description="Whether manual review is required",
            ),
        ]
        review_reason: Annotated[
            str,
            Field(
                default="",
                description="Manual review rationale",
            ),
        ]

    class HelperClassificationReport(FlextModels.ArbitraryTypesModel):
        """Aggregated helper-function classification payload."""

        totals: Annotated[
            dict[str, int],
            Field(
                default_factory=dict,
                description="Category totals",
            ),
        ]
        suggestions: Annotated[
            list[FlextInfraRefactorModels.HelperClassification],
            Field(
                default_factory=lambda: list[
                    FlextInfraRefactorModels.HelperClassification
                ](),
                description="Classification suggestions",
            ),
        ]
        manual_review: Annotated[
            list[FlextInfraRefactorModels.HelperClassification],
            Field(
                default_factory=lambda: list[
                    FlextInfraRefactorModels.HelperClassification
                ](),
                description="Manual-review candidates",
            ),
        ]

    class HelperFileAnalysis(FlextModels.ArbitraryTypesModel):
        suggestions: Annotated[
            list[FlextInfraRefactorModels.HelperClassification],
            Field(
                default_factory=lambda: list[
                    FlextInfraRefactorModels.HelperClassification
                ](),
                description="Helper classifications from one file",
            ),
        ]
        totals: Annotated[
            dict[str, int],
            Field(
                default_factory=dict,
                description="Category totals for file helpers",
            ),
        ]
        manual_review: Annotated[
            list[FlextInfraRefactorModels.HelperClassification],
            Field(
                default_factory=lambda: list[
                    FlextInfraRefactorModels.HelperClassification
                ](),
                description="Helpers requiring manual review",
            ),
        ]

    class ViolationAnalysisReport(FlextModels.ArbitraryTypesModel):
        """Full violation analysis report for refactor diagnostics."""

        class TopFileSection(FlextModels.ArbitraryTypesModel):
            """One ranked hotspot entry in violation analysis output."""

            file: Annotated[str, Field(min_length=1, description="File path")]
            total: Annotated[int, Field(ge=0, description="Total violations in file")]
            counts: Annotated[
                dict[str, int],
                Field(
                    default_factory=dict,
                    description="Per-pattern counts",
                ),
            ]

        totals: Annotated[
            dict[str, int],
            Field(
                default_factory=dict,
                description="Aggregate counts by pattern",
            ),
        ]
        files: Annotated[
            dict[str, dict[str, int]],
            Field(
                default_factory=dict,
                description="Per-file per-pattern counts",
            ),
        ]
        top_files: list[
            FlextInfraRefactorModels.ViolationAnalysisReport.TopFileSection
        ] = Field(
            default_factory=lambda: list[
                FlextInfraRefactorModels.ViolationAnalysisReport.TopFileSection
            ](),
            description="Top hotspot files",
        )
        files_scanned: Annotated[int, Field(ge=0, description="Files scanned")]
        helper_classification: Annotated[
            FlextInfraRefactorModels.HelperClassificationReport,
            Field(description="Helper classification summary"),
        ]
        class_nesting: Annotated[
            FlextInfraRefactorModels.ClassNestingReport,
            Field(
                description="Class nesting analysis summary",
            ),
        ]

    # -- MRO Target Specification -----------------------------------------------

    class MROTargetSpec(FlextModels.FrozenStrictModel):
        """Specification for an MRO target family."""

        family_alias: Annotated[
            str, Field(min_length=1, description="Family alias letter")
        ]
        file_names: Annotated[frozenset[str], Field(description="File name patterns")]
        package_directory: Annotated[
            str,
            Field(
                min_length=1,
                description="Package directory name",
            ),
        ]
        class_suffix: Annotated[str, Field(min_length=1, description="Class suffix")]

    # -- Pydantic Centralizer Models -------------------------------------------

    class ClassMove(FlextModels.FrozenStrictModel):
        """Tracks a class definition being moved during centralization."""

        name: Annotated[str, Field(min_length=1, description="Class name")]
        start: Annotated[int, Field(ge=0, description="Start line number")]
        end: Annotated[int, Field(ge=0, description="End line number")]
        source: Annotated[str, Field(description="Source code text")]
        kind: Annotated[
            str, Field(min_length=1, description="Model kind classification")
        ]

    class AliasMove(FlextModels.FrozenStrictModel):
        """Tracks a type alias being moved during centralization."""

        name: Annotated[str, Field(min_length=1, description="Alias name")]
        start: Annotated[int, Field(ge=0, description="Start line number")]
        end: Annotated[int, Field(ge=0, description="End line number")]
        alias_expr: Annotated[str, Field(description="Alias expression text")]

    class CentralizerFailureStats(FlextModels.ArbitraryTypesModel):
        """Mutable statistics for centralizer parse failures."""

        parse_syntax_errors: Annotated[
            int,
            Field(
                default=0,
                ge=0,
                description="Syntax error count",
            ),
        ] = 0
        parse_encoding_errors: Annotated[
            int,
            Field(
                default=0,
                ge=0,
                description="Encoding error count",
            ),
        ] = 0
        parse_io_errors: Annotated[
            int, Field(default=0, ge=0, description="I/O error count")
        ] = 0

    # -- Namespace Enforcer Models ---------------------------------------------

    class ParsedPythonModule(FlextModels.ArbitraryTypesModel):
        """Result of parsing a Python source file into AST."""

        model_config = ConfigDict(frozen=True)

        source: Annotated[str, Field(description="Raw source text")]
        tree: Annotated[ast.Module, Field(description="Parsed AST module node")]

    # -- MRO Generic Models ----------------------------------------------------

    class MROFamilyTarget(FlextModels.ArbitraryTypesModel):
        """Parametrized target for an MRO family scan or operations.

        Defines which MRO family to scan (e.g. utilities, constants, models).
        """

        model_config = ConfigDict(frozen=True)

        family: Annotated[
            str, Field(min_length=1, description="Family alias letter (c/t/p/m/u)")
        ]
        class_suffix: Annotated[
            str, Field(description="Class name suffix (e.g. 'Utilities')")
        ]
        package_dir: Annotated[
            str,
            Field(
                description="Relative path to _xxx package dir (e.g. 'flext_core/_utilities')",
            ),
        ]
        facade_module: Annotated[
            str,
            Field(
                description="Relative path to facade (e.g. 'flext_core/utilities.py')",
            ),
        ]
        facade_class_prefix: Annotated[
            str,
            Field(
                default="Flext",
                description="Class name prefix for facade (e.g. 'Flext')",
            ),
        ] = "Flext"
        core_project: Annotated[
            str,
            Field(
                default="flext-core",
                description="Core project directory name",
            ),
        ] = "flext-core"

    # -- Census Models ---------------------------------------------------------

    class CensusMethodInfo(FlextModels.ArbitraryTypesModel):
        """A public method extracted from a _utilities class."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(min_length=1, description="Method name")]
        method_type: Annotated[
            str, Field(description="Method kind: static, class, instance")
        ]
        source_file: Annotated[str, Field(description="Source filename")]

    class CensusUsageRecord(FlextModels.ArbitraryTypesModel):
        """A single method usage found via CST analysis."""

        model_config = ConfigDict(frozen=True)

        class_name: Annotated[
            str, Field(min_length=1, description="Utilities class name")
        ]
        method_name: Annotated[str, Field(min_length=1, description="Method name")]
        access_mode: Annotated[
            str, Field(description="Access mode: alias_flat, alias_namespaced, direct")
        ]
        file_path: Annotated[str, Field(description="Source file path")]
        project: Annotated[str, Field(description="Project name")]

    class CensusMethodSummary(FlextModels.ArbitraryTypesModel):
        """Aggregated usage counts for a single method."""

        model_config = ConfigDict(frozen=True)

        name: Annotated[str, Field(min_length=1, description="Method name")]
        method_type: Annotated[str, Field(description="Method kind")]
        alias_flat: Annotated[int, Field(ge=0, description="u.method count")]
        alias_namespaced: Annotated[
            int, Field(ge=0, description="u.Class.method count")
        ]
        direct: Annotated[int, Field(ge=0, description="Direct class.method count")]
        total: Annotated[int, Field(ge=0, description="Total usages")]

    class CensusClassSummary(FlextModels.ArbitraryTypesModel):
        """Aggregated census for one _utilities class."""

        model_config = ConfigDict(frozen=True)

        class_name: Annotated[
            str, Field(min_length=1, description="Utilities class name")
        ]
        source_file: Annotated[str, Field(description="Source filename")]
        methods: Annotated[
            list[FlextInfraRefactorModels.CensusMethodSummary],
            Field(
                default_factory=lambda: list[
                    FlextInfraRefactorModels.CensusMethodSummary
                ](),
                description="Method summaries",
            ),
        ]

    class CensusProjectMethodUsage(FlextModels.ArbitraryTypesModel):
        """Usage of a method within a specific project."""

        model_config = ConfigDict(frozen=True)

        class_name: Annotated[
            str, Field(min_length=1, description="Utilities class name")
        ]
        method_name: Annotated[str, Field(min_length=1, description="Method name")]
        access_mode: Annotated[str, Field(description="Access mode")]
        count: Annotated[int, Field(ge=0, description="Usage count")]

    class CensusProjectSummary(FlextModels.ArbitraryTypesModel):
        """Usage breakdown for one project."""

        model_config = ConfigDict(frozen=True)

        project_name: Annotated[
            str, Field(min_length=1, description="Project directory name")
        ]
        usages: Annotated[
            list[FlextInfraRefactorModels.CensusProjectMethodUsage],
            Field(
                default_factory=lambda: list[
                    FlextInfraRefactorModels.CensusProjectMethodUsage
                ](),
                description="Per-method usages",
            ),
        ]
        total: Annotated[int, Field(ge=0, description="Total usages in project")]

    class CensusReport(FlextModels.ArbitraryTypesModel):
        """Full census report for _utilities method usage."""

        classes: Annotated[
            list[FlextInfraRefactorModels.CensusClassSummary],
            Field(
                default_factory=lambda: list[
                    FlextInfraRefactorModels.CensusClassSummary
                ](),
                description="Per-class summaries",
            ),
        ]
        projects: Annotated[
            list[FlextInfraRefactorModels.CensusProjectSummary],
            Field(
                default_factory=lambda: list[
                    FlextInfraRefactorModels.CensusProjectSummary
                ](),
                description="Per-project breakdowns",
            ),
        ]
        total_classes: Annotated[
            int, Field(ge=0, description="Number of utility classes")
        ]
        total_methods: Annotated[
            int, Field(ge=0, description="Number of public methods")
        ]
        total_usages: Annotated[int, Field(ge=0, description="Total usage records")]
        total_unused: Annotated[
            int, Field(ge=0, description="Methods with zero usages")
        ]
        files_scanned: Annotated[int, Field(ge=0, description="Files scanned")]
        parse_errors: Annotated[
            int, Field(ge=0, description="Files that failed to parse")
        ]


__all__ = ["FlextInfraRefactorModels"]
