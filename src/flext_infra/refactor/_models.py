"""Domain models for the refactor subpackage."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import libcst as cst
from pydantic import ConfigDict, Field

from flext_core import FlextModels
from flext_infra.refactor._models_ast_grep import FlextInfraRefactorAstGrepModels
from flext_infra.refactor._models_namespace_enforcer import (
    FlextInfraNamespaceEnforcerModels,
)


@dataclass(frozen=True)
class MROTargetSpec:
    family_alias: str
    file_names: frozenset[str]
    package_directory: str
    class_suffix: str


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

        file_path: Path = Field(description="Target file path")
        success: bool = Field(description="Whether the operation succeeded")
        modified: bool = Field(description="Whether the file was actually modified")
        error: str | None = Field(default=None, description="Error message on failure")
        changes: list[str] = Field(
            default_factory=list,
            description="Human-readable change descriptions",
        )
        refactored_code: str | None = Field(
            default=None,
            description="Resulting source code after transformation",
        )

    class ProjectInfo(FlextModels.ArbitraryTypesModel):
        name: str = Field(min_length=1, description="Project directory name")
        path: Path = Field(description="Absolute project path")
        src_path: Path = Field(description="Absolute src/ path")
        package_roots: set[str] = Field(
            default_factory=set,
            description="Top-level Python package roots in src/",
        )

    class FileImportData(FlextModels.ArbitraryTypesModel):
        imported_modules: set[str] = Field(
            default_factory=set,
            description="Imported module roots",
        )
        imported_symbols: set[str] = Field(
            default_factory=set,
            description="Imported symbol names",
        )

    class MethodInfo(FlextModels.ArbitraryTypesModel):
        """Metadata about a method used for ordering inside classes."""

        name: str = Field(min_length=1, description="Method name")
        category: str = Field(description="Method category classification")
        node: cst.FunctionDef = Field(
            description="LibCST FunctionDef node",
            exclude=True,
        )
        decorators: list[str] = Field(
            default_factory=list,
            description="Decorator names applied to this method",
        )

    class Checkpoint(FlextModels.ArbitraryTypesModel):
        """Serialisable checkpoint state for refactor safety recovery."""

        workspace_root: str = Field(min_length=1, description="Workspace root path")
        status: str = Field(default="running", description="Checkpoint status")
        stash_ref: str = Field(default="", description="Git stash reference")
        processed_targets: list[str] = Field(
            default_factory=list,
            description="Already-processed file targets",
        )
        updated_at: str = Field(
            default_factory=lambda: datetime.now(UTC).isoformat(),
            description="ISO 8601 timestamp of last update",
        )

    class ClassOccurrence(FlextModels.ArbitraryTypesModel):
        """A single class definition occurrence within a source file."""

        model_config = ConfigDict(frozen=True)

        name: str = Field(min_length=1, description="Class name")
        line: int = Field(ge=0, description="Line number (0 = unknown)")
        is_top_level: bool = Field(description="Whether class is at module top level")

    class LooseClassViolation(FlextModels.ArbitraryTypesModel):
        """A detected loose-class naming violation with confidence."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")
        line: int = Field(ge=1, description="Line number")
        class_name: str = Field(min_length=1, description="Violating class name")
        expected_prefix: str = Field(description="Expected namespace prefix")
        rule: str = Field(min_length=1, description="Violated rule id")
        reason: str = Field(description="Human-readable reason")
        confidence: str = Field(description="Confidence level")
        score: float = Field(ge=0.0, le=1.0, description="Confidence score")

    class FamilyMROResolution(FlextModels.ArbitraryTypesModel):
        """Resolution payload for one facade family MRO."""

        model_config = ConfigDict(frozen=True)

        family: str = Field(min_length=1, description="Facade family letter")
        expected_bases: tuple[str, ...] = Field(
            description="Expected base class names in order"
        )
        resolved_mro: tuple[str, ...] = Field(description="Resolved MRO class names")
        accessible_namespaces: tuple[str, ...] = Field(
            description="Namespaces accessible through the MRO"
        )

    class ProjectClassification(FlextModels.ArbitraryTypesModel):
        """Result of classifying a project by kind and family chains."""

        model_config = ConfigDict(frozen=True)

        project_kind: str = Field(
            min_length=1,
            description="Project kind (core, domain, platform, integration, app)",
        )
        family_chains: dict[str, list[str]] = Field(
            description="Family letter to MRO chain mapping"
        )

    class ClassNestingMapping(FlextModels.ArbitraryTypesModel):
        """Unified mapping contract for class-nesting rewrite planning."""

        model_config = ConfigDict(extra="ignore", frozen=True)

        loose_name: str = Field(default="", description="Original loose class name")
        current_file: str = Field(default="", description="File containing class")
        target_namespace: str = Field(
            min_length=1,
            description="Target namespace class name",
        )
        target_name: str = Field(default="", description="Target class name")
        confidence: str = Field(min_length=1, description="Confidence level")
        reason: str = Field(default="", description="Optional mapping rationale")
        rewrite_scope: str | None = Field(
            default=None,
            description="Rewrite scope (file/project/workspace)",
        )

    class ClassNestingViolation(FlextModels.ArbitraryTypesModel):
        """Normalized class-nesting violation with rewrite metadata."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source module path")
        line: int = Field(ge=1, description="Line number")
        class_name: str = Field(min_length=1, description="Class name")
        target_namespace: str = Field(
            default="",
            description="Expected namespace class",
        )
        confidence: str = Field(default="low", description="Confidence level")
        rewrite_scope: str = Field(
            default="file",
            description="Rewrite scope",
        )

    class ClassNestingPolicy(FlextModels.FrozenStrictModel):
        """Validated policy contract used by class-nesting transformers."""

        model_config = ConfigDict(extra="ignore", frozen=True)

        family_name: str = Field(min_length=1, description="Module family name")
        allowed_operations: list[str] = Field(
            default_factory=list,
            description="Enabled operation identifiers for this family",
        )
        forbidden_operations: list[str] = Field(
            default_factory=list,
            description="Disabled operation identifiers for this family",
        )
        forbidden_targets: list[str] = Field(
            default_factory=list,
            description="Target namespaces forbidden for this family",
        )
        enable_class_nesting: bool = Field(
            default=True,
            description="Allow moving top-level classes under a namespace",
        )
        allow_namespace_creation: bool = Field(
            default=True,
            description="Allow creating a target namespace when absent",
        )
        allow_existing_namespace_merge: bool = Field(
            default=True,
            description="Allow merging nested classes into existing namespace",
        )
        enable_helper_consolidation: bool = Field(
            default=True,
            description="Allow consolidating helper functions into namespaces",
        )
        allow_helper_call_rewrite: bool = Field(
            default=True,
            description="Allow rewriting helper call sites to namespaced calls",
        )
        require_signature_validation: bool = Field(
            default=False,
            description="Require signature checks before helper migration",
        )
        required_parameters: list[str] = Field(
            default_factory=list,
            description="Function parameters that must exist in helper signatures",
        )
        forbidden_parameters: list[str] = Field(
            default_factory=list,
            description="Function parameters that must not exist in helper signatures",
        )
        allow_vararg: bool = Field(
            default=True,
            description="Allow variadic positional parameter usage",
        )
        allow_kwarg: bool = Field(
            default=True,
            description="Allow variadic keyword parameter usage",
        )
        allow_positional_only_params: bool = Field(
            default=True,
            description="Allow positional-only parameters",
        )
        allow_keyword_only_params: bool = Field(
            default=True,
            description="Allow keyword-only parameters",
        )
        propagate_imports: bool = Field(
            default=True,
            description="Allow propagating import rewrite rules",
        )
        propagate_name_references: bool = Field(
            default=True,
            description="Allow propagating direct name reference rewrites",
        )
        propagate_attribute_references: bool = Field(
            default=True,
            description="Allow propagating attribute reference rewrites",
        )
        blocked_reference_prefixes: list[str] = Field(
            default_factory=list,
            description="Name prefixes blocked from rewrite propagation",
        )
        allowed_targets: list[str] = Field(
            default_factory=list,
            description="Explicitly allowed target namespaces",
        )

    class ClassNestingReport(FlextModels.ArbitraryTypesModel):
        """Aggregated class-nesting analysis report."""

        violations_count: int = Field(ge=0, description="Total violations")
        confidence_counts: dict[str, int] = Field(
            default_factory=dict,
            description="Confidence histogram",
        )
        violations: list[FlextInfraRefactorModels.ClassNestingViolation] = Field(
            default_factory=lambda: list[
                FlextInfraRefactorModels.ClassNestingViolation
            ](),
            description="Violation details",
        )
        per_file_counts: dict[str, int] = Field(
            default_factory=dict,
            description="Violation counts per file",
        )

    class HelperClassification(FlextModels.ArbitraryTypesModel):
        """Classification result for a helper function."""

        file: str = Field(min_length=1, description="Source file")
        function: str = Field(min_length=1, description="Function name")
        category: str = Field(min_length=1, description="Assigned category")
        target_namespace: str = Field(
            min_length=1,
            description="Target namespace path",
        )
        dependencies: list[str] = Field(
            default_factory=list,
            description="Imported dependencies used by function",
        )
        manual_review: bool = Field(
            default=False,
            description="Whether manual review is required",
        )
        review_reason: str = Field(
            default="",
            description="Manual review rationale",
        )

    class HelperClassificationReport(FlextModels.ArbitraryTypesModel):
        """Aggregated helper-function classification payload."""

        totals: dict[str, int] = Field(
            default_factory=dict,
            description="Category totals",
        )
        suggestions: list[FlextInfraRefactorModels.HelperClassification] = Field(
            default_factory=lambda: list[
                FlextInfraRefactorModels.HelperClassification
            ](),
            description="Classification suggestions",
        )
        manual_review: list[FlextInfraRefactorModels.HelperClassification] = Field(
            default_factory=lambda: list[
                FlextInfraRefactorModels.HelperClassification
            ](),
            description="Manual-review candidates",
        )

    class HelperFileAnalysis(FlextModels.ArbitraryTypesModel):
        suggestions: list[FlextInfraRefactorModels.HelperClassification] = Field(
            default_factory=lambda: list[
                FlextInfraRefactorModels.HelperClassification
            ](),
            description="Helper classifications from one file",
        )
        totals: dict[str, int] = Field(
            default_factory=dict,
            description="Category totals for file helpers",
        )
        manual_review: list[FlextInfraRefactorModels.HelperClassification] = Field(
            default_factory=lambda: list[
                FlextInfraRefactorModels.HelperClassification
            ](),
            description="Helpers requiring manual review",
        )

    class ViolationAnalysisReport(FlextModels.ArbitraryTypesModel):
        """Full violation analysis report for refactor diagnostics."""

        class TopFileSection(FlextModels.ArbitraryTypesModel):
            """One ranked hotspot entry in violation analysis output."""

            file: str = Field(min_length=1, description="File path")
            total: int = Field(ge=0, description="Total violations in file")
            counts: dict[str, int] = Field(
                default_factory=dict,
                description="Per-pattern counts",
            )

        totals: dict[str, int] = Field(
            default_factory=dict,
            description="Aggregate counts by pattern",
        )
        files: dict[str, dict[str, int]] = Field(
            default_factory=dict,
            description="Per-file per-pattern counts",
        )
        top_files: list[
            FlextInfraRefactorModels.ViolationAnalysisReport.TopFileSection
        ] = Field(
            default_factory=lambda: list[
                FlextInfraRefactorModels.ViolationAnalysisReport.TopFileSection
            ](),
            description="Top hotspot files",
        )
        files_scanned: int = Field(ge=0, description="Files scanned")
        helper_classification: FlextInfraRefactorModels.HelperClassificationReport = (
            Field(description="Helper classification summary")
        )
        class_nesting: FlextInfraRefactorModels.ClassNestingReport = Field(
            description="Class nesting analysis summary"
        )


__all__ = ["FlextInfraRefactorModels"]
