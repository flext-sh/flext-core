"""Refactor migration model mixins for rope-oriented orchestration."""

from __future__ import annotations

from typing import Annotated, ClassVar

from flext_core import m
from flext_infra import c, t
from flext_infra._models._defaults import ImmutableEmptyMapping


class FlextInfraModelsRefactorGrep:
    """Mixin containing migration/reporting contracts for refactor orchestration."""

    class RefactorConfig(m.ContractModel):
        """Refactor file-selection config."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        project_scan_dirs: t.StrSequence = m.Field(
            default_factory=lambda: [
                c.Infra.DEFAULT_SRC_DIR,
                c.Infra.DIR_TESTS,
                c.Infra.DIR_SCRIPTS,
                c.Infra.DIR_EXAMPLES,
            ],
            description="Relative directories scanned for candidate files",
        )
        ignore_patterns: t.StrSequence = m.Field(
            default_factory=tuple, description="Glob/file patterns ignored during scan"
        )
        file_extensions: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Allowed file extensions (empty = all by pattern)",
        )

    class MethodOrderRule(m.ContractModel):
        """A declarative method ordering rule for class reconstruction.

        Enforcement exemption: internal tooling model with intentional
        mutable state.
        """

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict()

        category: Annotated[str | None, m.Field(description="Method category")] = None
        visibility: Annotated[str | None, m.Field(description="Visibility filter")] = (
            None
        )
        exclude_decorators: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Decorators excluded from the ordering rule",
        )
        decorators: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Decorators required by the ordering rule",
        )
        patterns: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Method name patterns included by the ordering rule",
        )
        order: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Expected method-category order for matching methods",
        )

    class SignatureMigration(m.ContractModel):
        """Declarative signature migration rule for callsite propagation.

        Enforcement exemption: internal tooling model with intentional
        mutable state.
        """

        id: Annotated[str, m.Field(description="Migration ID")] = "signature-migration"
        enabled: Annotated[bool, m.Field(description="Whether migration is active")] = (
            True
        )
        target_qualified_names: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Qualified symbol names targeted by the migration",
        )
        target_simple_names: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Simple symbol names targeted by the migration",
        )
        keyword_renames: t.StrMapping = m.Field(
            default_factory=ImmutableEmptyMapping, description="Keyword rename mapping"
        )
        remove_keywords: t.StrSequence = m.Field(
            default_factory=tuple,
            description="Keywords removed from matching callsites",
        )
        add_keywords: t.StrMapping = m.Field(
            default_factory=ImmutableEmptyMapping, description="Keywords to add"
        )

    class ImportModernizerRuleConfig(m.ContractModel):
        """Configuration for a single import modernizer rule.

        Enforcement exemption: internal tooling model with intentional
        mutable state.
        """

        module: Annotated[str, m.Field(description="Module path to modernize")] = ""
        symbol_mapping: t.StrMapping = m.Field(
            default_factory=ImmutableEmptyMapping, description="Symbol-to-alias mapping"
        )

    class AccessorMigrationRule(m.ContractModel):
        """Declarative symbol-rename rule for accessor migration."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        source_name: Annotated[
            t.NonEmptyStr, m.Field(description="Canonical symbol name to replace")
        ]
        replacement_name: Annotated[
            t.NonEmptyStr,
            m.Field(description="Canonical symbol name used as replacement"),
        ]
        reason: Annotated[
            t.NonEmptyStr,
            m.Field(description="Human-readable explanation for the rename"),
        ]
        origin: Annotated[
            str, m.Field(description="Canonical API origin this rewrite is tied to")
        ] = ""

    class AccessorMigrationChange(m.ArbitraryTypesModel):
        """Single automated rename or manual warning emitted by accessor migration."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        file: Annotated[t.NonEmptyStr, m.Field(description="Absolute file path")]
        line: Annotated[
            t.NonNegativeInt,
            m.Field(description="1-based source line number when available"),
        ]
        original_name: Annotated[
            t.NonEmptyStr, m.Field(description="Original accessor or helper name")
        ]
        replacement_name: Annotated[
            str, m.Field(description="Suggested or applied replacement name")
        ] = ""
        automated: Annotated[
            bool,
            m.Field(description="Whether the migration was performed automatically"),
        ]
        reason: Annotated[
            str, m.Field(description="Human-readable migration rationale")
        ]

    class AccessorMigrationFile(m.ArbitraryTypesModel):
        """Per-file preview for accessor migration dry-runs and applies.

        Enforcement exemption: internal tooling model with intentional
        mutable state.
        """

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        file: Annotated[t.NonEmptyStr, m.Field(description="Absolute file path")]
        lint_tools: t.VariadicTuple[str] = m.Field(
            default_factory=tuple,
            description="Selected lint tools used for preview rendering",
        )
        automated_changes: t.VariadicTuple[
            FlextInfraModelsRefactorGrep.AccessorMigrationChange
        ] = m.Field(
            default_factory=tuple,
            description="Automated rewrites captured for this file",
        )
        warnings: t.VariadicTuple[
            FlextInfraModelsRefactorGrep.AccessorMigrationChange
        ] = m.Field(
            default_factory=tuple, description="Manual follow-up warnings for this file"
        )
        diff: Annotated[
            str, m.Field(description="Unified diff preview for the file")
        ] = ""
        lint_before: Annotated[
            t.MappingKV[str, t.StrSequence],
            m.Field(description="Lint output before the proposed rewrite"),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        lint_after: Annotated[
            t.MappingKV[str, t.StrSequence],
            m.Field(description="Lint output after the proposed rewrite"),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        new_lint_errors: Annotated[
            t.MappingKV[str, t.StrSequence],
            m.Field(description="Lint errors introduced by the proposed rewrite"),
        ] = m.Field(default_factory=ImmutableEmptyMapping)

    class AccessorMigrationReport(m.ArbitraryTypesModel):
        """Workspace-scale report for accessor migration orchestration.

        Enforcement exemption: internal tooling model with intentional
        mutable state.
        """

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)

        workspace: Annotated[t.NonEmptyStr, m.Field(description="Repository root path")]
        dry_run: Annotated[bool, m.Field(description="Dry-run indicator")]
        files_scanned: Annotated[
            t.NonNegativeInt, m.Field(description="Total Python files scanned")
        ]
        files_with_changes: Annotated[
            t.NonNegativeInt, m.Field(description="Files with automated rewrites")
        ]
        automated_change_count: Annotated[
            t.NonNegativeInt, m.Field(description="Total automated rewrites detected")
        ]
        warning_count: Annotated[
            t.NonNegativeInt,
            m.Field(description="Total manual follow-up warnings detected"),
        ]
        lint_tools: t.VariadicTuple[str] = m.Field(
            default_factory=tuple,
            description="Canonical lint tool list used by this run",
        )
        lint_before_totals: Annotated[
            t.IntMapping,
            m.Field(description="Per-tool count of lint lines before rewrites"),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        lint_after_totals: Annotated[
            t.IntMapping,
            m.Field(description="Per-tool count of lint lines after rewrites"),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        new_lint_error_totals: Annotated[
            t.IntMapping,
            m.Field(description="Per-tool count of newly introduced lint lines"),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        files: t.VariadicTuple[FlextInfraModelsRefactorGrep.AccessorMigrationFile] = (
            m.Field(
                default_factory=tuple,
                description="Preview entries included in this report",
            )
        )


__all__: list[str] = ["FlextInfraModelsRefactorGrep"]
