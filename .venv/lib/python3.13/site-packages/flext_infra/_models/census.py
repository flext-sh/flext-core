"""Unified census pipeline models — accessed via m.Infra.Census.*."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m
from flext_infra import c, t
from flext_infra._models._defaults import ImmutableEmptyMapping
from flext_infra._models.mixins import FlextInfraModelsMixins as mm


class FlextInfraModelsCensus:
    """Unified census pipeline models scoped under m.Infra.Census.*."""

    class Census:
        """Namespace for unified census pipeline data contracts."""

        class ReferenceSite(
            mm.AbsoluteFilePathTextMixin,
            mm.RequiredNonNegativeLineMixin,
            m.ArbitraryTypesModel,
        ):
            """Single reference site supporting a census classification."""

            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

            surface: Annotated[
                str,
                m.Field(description="Reference surface (src/tests/examples/scripts)"),
            ] = c.Infra.DEFAULT_SRC_DIR

        class Object(
            mm.AbsoluteFilePathTextMixin,
            mm.RequiredNonNegativeLineMixin,
            mm.ProjectNameMixin,
            mm.NestedClassPathMixin,
            m.ArbitraryTypesModel,
        ):
            """Single discovered Python object with tier and classification metadata."""

            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

            name: Annotated[t.NonEmptyStr, m.Field(description="Object identifier")]
            kind: Annotated[
                str,
                m.Field(
                    description="Object kind (class/function/method/constant/local/...)"
                ),
            ]
            module_name: Annotated[
                str, m.Field(description="Fully-qualified module name for the object")
            ] = ""
            scope_path: Annotated[
                str, m.Field(description="Canonical owner/scope path for the object")
            ] = ""
            actual_tier: Annotated[
                str, m.Field(description="Tier derived from file location")
            ] = ""
            expected_tier: Annotated[
                str, m.Field(description="Tier determined by classifier")
            ] = ""
            is_facade_member: Annotated[
                bool, m.Field(description="Whether object is exposed via facade FLEXT")
            ] = False
            references_count: Annotated[
                t.NonNegativeInt,
                m.Field(
                    description="Number of references excluding the definition site"
                ),
            ] = 0
            runtime_references_count: Annotated[
                t.NonNegativeInt,
                m.Field(description="Number of references from runtime/source modules"),
            ] = 0
            script_references_count: Annotated[
                t.NonNegativeInt,
                m.Field(description="Number of references from script modules"),
            ] = 0
            runtime_reference_sites: tuple[
                FlextInfraModelsCensus.Census.ReferenceSite, ...
            ] = m.Field(
                default_factory=tuple, description="Runtime/source reference sites"
            )
            script_reference_sites: tuple[
                FlextInfraModelsCensus.Census.ReferenceSite, ...
            ] = m.Field(default_factory=tuple, description="Script reference sites")
            fingerprint: Annotated[
                str, m.Field(description="Normalized Rope-derived semantic fingerprint")
            ] = ""

        class RemovalCandidate(
            mm.AbsoluteFilePathTextMixin,
            mm.RequiredNonNegativeLineMixin,
            mm.ProjectNameMixin,
            m.ArbitraryTypesModel,
        ):
            """Explicit aggressive-removal candidate derived from census results."""

            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

            object_name: Annotated[
                t.NonEmptyStr, m.Field(description="Candidate object name")
            ]
            object_kind: Annotated[str, m.Field(description="Candidate object kind")]
            scope_path: Annotated[
                str, m.Field(description="Canonical owner/scope path for the candidate")
            ] = ""
            reason: Annotated[str, m.Field(description="Candidate reason (unused)")]
            suggested_action: Annotated[
                str, m.Field(description="Suggested removal action for this candidate")
            ]
            runtime_reference_sites: tuple[
                FlextInfraModelsCensus.Census.ReferenceSite, ...
            ] = m.Field(
                default_factory=tuple,
                description="Runtime/source references blocking full deletion",
            )
            script_reference_sites: tuple[
                FlextInfraModelsCensus.Census.ReferenceSite, ...
            ] = m.Field(
                default_factory=tuple,
                description="Script references supporting this candidate",
            )

        class Violation(mm.ProjectNameMixin, m.ArbitraryTypesModel):
            """Detected census violation with fix metadata."""

            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

            object_name: Annotated[
                t.NonEmptyStr, m.Field(description="Name of the violating object")
            ]
            object_kind: Annotated[
                str,
                m.Field(
                    description="Object kind (constant/type/protocol/model/utility)"
                ),
            ]
            kind: Annotated[
                str,
                m.Field(
                    description="Violation kind (misplaced/duplicate/unused/missing_flext_base/flat_alias/wrong_tier)"
                ),
            ]
            severity: Annotated[str, m.Field(description="Severity level")] = (
                c.Infra.GateSeverity.WARNING.value
            )
            file_path: Annotated[str, m.Field(description="File containing violation")]
            line: Annotated[t.NonNegativeInt, m.Field(description="Line number")] = 0
            fixable: Annotated[
                bool, m.Field(description="Whether auto-fix is available")
            ] = False
            fix_action: Annotated[
                str, m.Field(description="Recommended fix action identifier")
            ] = ""
            description: Annotated[
                str, m.Field(description="Human-readable violation description")
            ] = ""

        class Fix(m.ArbitraryTypesModel):
            """Applied or proposed auto-fix operation."""

            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

            object_name: Annotated[
                t.NonEmptyStr, m.Field(description="Name of the fixed object")
            ]
            action: Annotated[
                str,
                m.Field(
                    description="Fix action applied (move_to_tier/deduplicate/...)"
                ),
            ]
            source_file: Annotated[str, m.Field(description="Original file path")]
            target_file: Annotated[
                str, m.Field(description="Destination file path (for moves)")
            ] = ""
            files_changed: Annotated[
                t.NonNegativeInt, m.Field(description="Number of files modified")
            ] = 0
            applied: Annotated[
                bool, m.Field(description="Whether fix was actually applied")
            ] = False
            dry_run_diff: Annotated[
                str, m.Field(description="Unified diff preview (dry-run mode)")
            ] = ""

        class ScanConfig(m.ArbitraryTypesModel):
            """Resolved per-collect scan configuration shared across modules."""

            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

            kind_names: Annotated[
                t.StrSequence | None, m.Field(description="Symbol-kind filters")
            ]
            rule_names: Annotated[
                t.StrSequence | None, m.Field(description="Violation-rule filters")
            ]
            selected_families: Annotated[
                frozenset[str], m.Field(description="Resolved namespace families")
            ]
            selected_kinds: Annotated[
                frozenset[str] | None, m.Field(description="Precomputed kind set")
            ]
            selected_rules: Annotated[
                frozenset[str] | None, m.Field(description="Precomputed rule set")
            ]
            collect_object_inventory: Annotated[
                bool,
                m.Field(description="Whether to collect the full object inventory"),
            ]
            include_object_references: Annotated[
                bool, m.Field(description="Whether to resolve object references")
            ]
            include_local_scopes: Annotated[
                bool, m.Field(description="Whether to include local/nested scopes")
            ]
            applied: Annotated[
                frozenset[str], m.Field(description="Fix keys already applied")
            ]

        class DuplicateGroup(m.ArbitraryTypesModel):
            """Cross-project duplicate object cluster."""

            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

            name: Annotated[
                t.NonEmptyStr, m.Field(description="Shared object name across projects")
            ]
            kind: Annotated[str, m.Field(description="Object kind")]
            definitions: list[FlextInfraModelsCensus.Census.Object] = m.Field(
                description="All definitions of this object"
            )
            canonical: Annotated[
                str, m.Field(description="Most-upstream project (canonical source)")
            ] = ""
            value_identical: Annotated[
                bool,
                m.Field(description="Whether all definitions have identical values"),
            ] = False

        class ProjectReport(mm.ProjectNameMixin, m.ArbitraryTypesModel):
            """Per-project census summary."""

            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

            objects: tuple[FlextInfraModelsCensus.Census.Object, ...] = m.Field(
                default_factory=tuple, description="Objects discovered for this project"
            )
            objects_total: Annotated[
                t.NonNegativeInt, m.Field(description="Total objects discovered")
            ] = 0
            objects_by_kind: Annotated[
                t.IntMapping, m.Field(description="Object count per kind")
            ] = m.Field(default_factory=ImmutableEmptyMapping)
            violations: tuple[FlextInfraModelsCensus.Census.Violation, ...] = m.Field(
                default_factory=tuple, description="Detected violations"
            )
            fixes: tuple[FlextInfraModelsCensus.Census.Fix, ...] = m.Field(
                default_factory=tuple, description="Proposed or applied fixes"
            )
            violations_total: Annotated[
                t.NonNegativeInt, m.Field(description="Total violation count")
            ] = 0
            fixes_applied: Annotated[
                t.NonNegativeInt, m.Field(description="Fixes applied count")
            ] = 0
            unused_count: Annotated[
                t.NonNegativeInt,
                m.Field(description="Objects with no non-definition references"),
            ] = 0
            removal_candidate_count: Annotated[
                t.NonNegativeInt,
                m.Field(description="Objects eligible for aggressive removal review"),
            ] = 0
            removal_candidates: tuple[
                FlextInfraModelsCensus.Census.RemovalCandidate, ...
            ] = m.Field(
                default_factory=tuple,
                description="Explicit aggressive-removal candidates for this project",
            )

        class WorkspaceReport(m.ArbitraryTypesModel):
            """Workspace-wide census summary."""

            projects: tuple[FlextInfraModelsCensus.Census.ProjectReport, ...] = m.Field(
                default_factory=tuple, description="Per-project reports"
            )
            total_objects: Annotated[
                t.NonNegativeInt, m.Field(description="Total objects across workspace")
            ] = 0
            total_violations: Annotated[
                t.NonNegativeInt,
                m.Field(description="Total violations across workspace"),
            ] = 0
            total_fixable: Annotated[
                t.NonNegativeInt, m.Field(description="Total fixable violations")
            ] = 0
            fixes_total: Annotated[
                t.NonNegativeInt, m.Field(description="Total proposed or applied fixes")
            ] = 0
            duplicates: tuple[FlextInfraModelsCensus.Census.DuplicateGroup, ...] = (
                m.Field(
                    default_factory=tuple, description="Cross-project duplicate groups"
                )
            )
            unused_count: Annotated[
                t.NonNegativeInt, m.Field(description="Total unused objects")
            ] = 0
            removal_candidate_count: Annotated[
                t.NonNegativeInt,
                m.Field(
                    description="Total objects eligible for aggressive removal review"
                ),
            ] = 0
            removal_candidates: tuple[
                FlextInfraModelsCensus.Census.RemovalCandidate, ...
            ] = m.Field(
                default_factory=tuple,
                description="Explicit aggressive-removal candidates across workspace",
            )
            scan_duration_seconds: Annotated[
                float, m.Field(description="Wall-clock scan duration")
            ] = 0.0
            parse_errors: Annotated[
                t.NonNegativeInt, m.Field(description="Files that failed to parse")
            ] = 0


__all__: list[str] = ["FlextInfraModelsCensus"]
