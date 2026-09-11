"""Violation and helper classification models for the refactor subpackage."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m
from flext_infra import t
from flext_infra._models._defaults import immutable_empty_mapping
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsRefactorViolations:
    """Class-nesting violation, helper classification, and analysis report models."""

    class ClassNestingViolationRequest(m.ContractModel):
        """Validated input for class-nesting policy violation checks."""

        symbol: Annotated[t.NonEmptyStr, m.Field(description="Source symbol name")]
        family: Annotated[t.NonEmptyStr, m.Field(description="Module family key")]
        target_namespace: Annotated[
            t.NonEmptyStr, m.Field(description="Destination namespace for the symbol")
        ]
        operation: Annotated[
            t.NonEmptyStr, m.Field(description="Policy operation being validated")
        ]

    class ClassNestingMapping(m.ArbitraryTypesModel):
        """Unified mapping contract for class-nesting rewrite planning."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        loose_name: Annotated[str, m.Field(description="Original loose class name")] = (
            ""
        )
        current_file: Annotated[str, m.Field(description="File containing class")] = ""
        target_namespace: Annotated[
            t.NonEmptyStr, m.Field(description="Target namespace class name")
        ]
        target_name: Annotated[str, m.Field(description="Target class name")] = ""
        confidence: Annotated[t.NonEmptyStr, m.Field(description="Confidence level")]
        reason: Annotated[str, m.Field(description="Optional mapping rationale")] = ""
        rewrite_scope: Annotated[
            str | None, m.Field(description="Rewrite scope (file/project/workspace)")
        ] = None

    class ClassNestingViolation(
        mm.ConfidenceLevelMixin,
        mm.RewriteScopeMixin,
        mm.FileLineViolationMixin,
        m.ArbitraryTypesModel,
    ):
        """Normalized class-nesting violation with rewrite metadata."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        class_name: Annotated[t.NonEmptyStr, m.Field(description="Class name")]
        target_namespace: Annotated[
            str, m.Field(description="Expected namespace class")
        ] = ""

    class ClassNestingPolicy(m.ContractModel):
        """Validated policy contract used by class-nesting transformers."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        family_name: Annotated[t.NonEmptyStr, m.Field(description="Module family name")]
        module_patterns: t.StrSequence = m.Field(
            default_factory=tuple, description="Glob patterns matching module paths."
        )
        facade_family: Annotated[str, m.Field(description="Facade family alias")] = ""
        allowed_operations: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Rewrite operations explicitly allowed by the policy.",
        )
        forbidden_operations: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Rewrite operations blocked by the policy.",
        )
        forbidden_targets: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Namespace targets blocked by the policy.",
        )
        validation_requirements: t.MappingKV[str, t.StrSequence] = m.Field(
            default_factory=immutable_empty_mapping,
            description="Validation requirements by stage.",
        )
        enable_class_nesting: Annotated[
            bool,
            m.Field(description="Allow moving top-level classes under a namespace"),
        ] = True
        allow_namespace_creation: Annotated[
            bool, m.Field(description="Allow creating a target namespace when absent")
        ] = True
        allow_existing_namespace_merge: Annotated[
            bool,
            m.Field(description="Allow merging nested classes into existing namespace"),
        ] = True
        enable_helper_consolidation: Annotated[
            bool,
            m.Field(description="Allow consolidating helper functions into namespaces"),
        ] = True
        allow_helper_call_rewrite: Annotated[
            bool,
            m.Field(
                description="Allow rewriting helper call sites to namespaced calls"
            ),
        ] = True
        require_signature_validation: Annotated[
            bool,
            m.Field(description="Require signature checks before helper migration"),
        ] = False
        required_parameters: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Parameters that must be present before rewriting.",
        )
        forbidden_parameters: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Parameters that block rewriting when present.",
        )
        allow_vararg: Annotated[
            bool, m.Field(description="Allow variadic positional parameter usage")
        ] = True
        allow_kwarg: Annotated[
            bool, m.Field(description="Allow variadic keyword parameter usage")
        ] = True
        allow_positional_only_params: Annotated[
            bool, m.Field(description="Allow positional-only parameters")
        ] = True
        allow_keyword_only_params: Annotated[
            bool, m.Field(description="Allow keyword-only parameters")
        ] = True
        propagate_imports: Annotated[
            bool, m.Field(description="Allow propagating import rewrite rules")
        ] = True
        propagate_name_references: Annotated[
            bool,
            m.Field(description="Allow propagating direct name reference rewrites"),
        ] = True
        propagate_attribute_references: Annotated[
            bool, m.Field(description="Allow propagating attribute reference rewrites")
        ] = True
        blocked_reference_prefixes: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Reference prefixes that must never be rewritten.",
        )
        allowed_targets: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Namespace targets explicitly allowed for rewrites.",
        )

    class ClassNestingReport(m.ArbitraryTypesModel):
        """Aggregated class-nesting analysis report."""

        violations_count: Annotated[
            t.NonNegativeInt, m.Field(description="Total violations")
        ]
        confidence_counts: t.IntMapping = m.Field(
            default_factory=immutable_empty_mapping, description="Confidence histogram"
        )
        violations: tuple[
            FlextInfraModelsRefactorViolations.ClassNestingViolation, ...
        ] = m.Field(default_factory=tuple, description="Violation details")
        per_file_counts: t.IntMapping = m.Field(
            default_factory=immutable_empty_mapping,
            description="Violation counts per file",
        )

    class HelperClassification(m.ArbitraryTypesModel):
        """Classification result for a helper function."""

        file: Annotated[t.NonEmptyStr, m.Field(description="Source file")]
        function: Annotated[t.NonEmptyStr, m.Field(description="Function name")]
        category: Annotated[t.NonEmptyStr, m.Field(description="Assigned category")]
        target_namespace: Annotated[
            t.NonEmptyStr, m.Field(description="Target namespace path")
        ]
        dependencies: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Dependency symbols referenced by the helper.",
        )
        manual_review: Annotated[
            bool, m.Field(description="Whether manual review is required")
        ] = False
        review_reason: Annotated[
            str, m.Field(description="Manual review rationale")
        ] = ""

    class HelperClassificationReport(m.ArbitraryTypesModel):
        """Aggregated helper-function classification payload."""

        totals: t.IntMapping = m.Field(
            default_factory=immutable_empty_mapping, description="Category totals"
        )
        suggestions: tuple[
            FlextInfraModelsRefactorViolations.HelperClassification, ...
        ] = m.Field(default_factory=tuple, description="Classification suggestions")
        manual_review: tuple[
            FlextInfraModelsRefactorViolations.HelperClassification, ...
        ] = m.Field(default_factory=tuple, description="Manual-review candidates")

    class HelperFileAnalysis(m.ArbitraryTypesModel):
        """Helper file analysis."""

        suggestions: tuple[
            FlextInfraModelsRefactorViolations.HelperClassification, ...
        ] = m.Field(
            default_factory=tuple, description="Helper classifications from one file"
        )
        totals: t.IntMapping = m.Field(
            default_factory=immutable_empty_mapping,
            description="Category totals for file helpers",
        )
        manual_review: tuple[
            FlextInfraModelsRefactorViolations.HelperClassification, ...
        ] = m.Field(
            default_factory=tuple, description="Helpers requiring manual review"
        )

    class ViolationTopFileSection(m.ArbitraryTypesModel):
        """One ranked hotspot entry in violation analysis output."""

        file: Annotated[t.NonEmptyStr, m.Field(description="File path")]
        total: Annotated[
            t.NonNegativeInt, m.Field(description="Total violations in file")
        ]
        counts: t.IntMapping = m.Field(
            default_factory=immutable_empty_mapping, description="Per-pattern counts"
        )

    class ViolationAnalysisReport(m.ArbitraryTypesModel):
        """Full violation analysis report for refactor diagnostics."""

        totals: t.IntMapping = m.Field(
            default_factory=immutable_empty_mapping,
            description="Aggregate counts by pattern",
        )
        files: t.MappingKV[str, t.IntMapping] = m.Field(
            default_factory=immutable_empty_mapping,
            description="Per-file per-pattern counts",
        )
        top_files: tuple[
            FlextInfraModelsRefactorViolations.ViolationTopFileSection, ...
        ] = m.Field(default_factory=tuple, description="Top hotspot files")
        files_scanned: Annotated[t.NonNegativeInt, m.Field(description="Files scanned")]
        helper_classification: FlextInfraModelsRefactorViolations.HelperClassificationReport = m.Field(
            description="Helper classification summary"
        )
        class_nesting: FlextInfraModelsRefactorViolations.ClassNestingReport = m.Field(
            description="Class nesting analysis summary"
        )


__all__: list[str] = ["FlextInfraModelsRefactorViolations"]
