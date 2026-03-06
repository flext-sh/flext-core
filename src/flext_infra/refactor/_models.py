"""Domain models for the refactor subpackage."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import libcst as cst
from pydantic import ConfigDict, Field

from flext_core import FlextModels


class FlextInfraRefactorModels:
    """Models for the refactor engine and related tools."""

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

    class Suggestion(FlextModels.ArbitraryTypesModel):
        """A single refactoring suggestion from analysis."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")
        name: str = Field(min_length=1, description="Symbol name")
        kind: str = Field(description="Symbol kind (function, class, etc.)")
        target: str = Field(description="Suggested target location")
        reason: str = Field(default="", description="Reason for suggestion")

    class ManualReviewItem(FlextModels.ArbitraryTypesModel):
        """An item flagged for manual review during analysis."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Source file path")
        name: str = Field(min_length=1, description="Symbol name")
        kind: str = Field(description="Symbol kind (function, class, etc.)")
        reason: str = Field(description="Reason for manual review")

    class ClassNestingMappingEntry(FlextModels.ArbitraryTypesModel):
        """Mapping metadata for class-nesting rewrite planning."""

        model_config = ConfigDict(frozen=True)

        target_namespace: str = Field(
            min_length=1,
            description="Target namespace class name",
        )
        confidence: str = Field(min_length=1, description="Confidence level")
        rewrite_scope: str = Field(
            min_length=1,
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

    class ClassNestingReport(FlextModels.ArbitraryTypesModel):
        """Aggregated class-nesting analysis report."""

        violations_count: int = Field(ge=0, description="Total violations")
        confidence_counts: dict[str, int] = Field(
            default_factory=dict,
            description="Confidence histogram",
        )
        violations: Sequence[object] = Field(
            default_factory=list,
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
        suggestions: Sequence[object] = Field(
            default_factory=list, description="Classification suggestions"
        )
        manual_review: Sequence[object] = Field(
            default_factory=list,
            description="Manual-review candidates",
        )

    class ViolationTopFile(FlextModels.ArbitraryTypesModel):
        """One ranked file entry in violation analysis output."""

        file: str = Field(min_length=1, description="File path")
        total: int = Field(ge=0, description="Total violations in file")
        counts: dict[str, int] = Field(
            default_factory=dict,
            description="Per-pattern counts",
        )

    class ViolationAnalysisReport(FlextModels.ArbitraryTypesModel):
        """Full violation analysis report for refactor diagnostics."""

        totals: dict[str, int] = Field(
            default_factory=dict,
            description="Aggregate counts by pattern",
        )
        files: dict[str, dict[str, int]] = Field(
            default_factory=dict,
            description="Per-file per-pattern counts",
        )
        top_files: Sequence[object] = Field(
            default_factory=list, description="Top hotspot files"
        )
        files_scanned: int = Field(ge=0, description="Files scanned")
        helper_classification: object = Field(
            description="Helper classification summary"
        )
        class_nesting: object = Field(description="Class nesting analysis summary")


__all__ = ["FlextInfraRefactorModels"]
