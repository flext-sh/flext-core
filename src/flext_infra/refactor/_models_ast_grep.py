"""Ast-grep and MRO migration model mixins for refactor."""

from __future__ import annotations

from pydantic import AliasPath, ConfigDict, Field

from flext_core import FlextModels


class FlextInfraRefactorAstGrepModels:
    """Mixin containing ast-grep and migration model contracts."""

    class AstGrepMatchEnvelope(FlextModels.ArbitraryTypesModel):
        """Compact ast-grep envelope carrying file, symbol and location."""

        model_config = ConfigDict(extra="ignore", populate_by_name=True)

        file: str = Field(min_length=1, description="Matched file path")
        symbol_name: str | None = Field(
            default=None,
            validation_alias=AliasPath("metaVariables", "single", "NAME", "text"),
            description="Captured symbol name from ast-grep metadata",
        )
        start_line: int | None = Field(
            default=None,
            validation_alias=AliasPath("range", "start", "line"),
            description="Start line from ast-grep range",
        )

    class MROSymbolCandidate(FlextModels.ArbitraryTypesModel):
        """Unified symbol candidate used by MRO scan and rewrites."""

        model_config = ConfigDict(frozen=True)

        symbol: str = Field(min_length=1, description="Symbol name")
        line: int = Field(ge=1, description="Source line number")
        kind: str = Field(default="constant", description="constant|typevar|typealias")
        class_name: str = Field(default="", description="Target class name")
        facade_name: str = Field(default="", description="Facade alias/import name")

    class MROImportRewrite(FlextModels.ArbitraryTypesModel):
        """Unified import rewrite payload for MRO reference updates."""

        model_config = ConfigDict(frozen=True)

        module: str = Field(min_length=1, description="Import module path")
        import_name: str = Field(min_length=1, description="Imported symbol name")
        as_name: str | None = Field(default=None, description="Optional alias")
        symbol: str = Field(default="", description="Resolved symbol in facade")
        facade_name: str = Field(default="", description="Facade alias/import name")

    class MROScanReport(FlextModels.ArbitraryTypesModel):
        """Scan result for one constants module candidate file."""

        model_config = ConfigDict(frozen=True)

        file: str = Field(min_length=1, description="Absolute file path")
        module: str = Field(min_length=1, description="Import module path")
        constants_class: str = Field(
            default="", description="First constants class name",
        )
        facade_alias: str = Field(default="c", description="Facade alias letter")
        candidates: tuple[FlextInfraRefactorAstGrepModels.MROSymbolCandidate, ...] = (
            Field(
                default_factory=tuple,
                description="Module-level symbol candidates",
            )
        )

    class MROFileMigration(FlextModels.ArbitraryTypesModel):
        """Migration summary for one transformed file."""

        file: str = Field(min_length=1, description="Absolute file path")
        module: str = Field(min_length=1, description="Import module path")
        moved_symbols: tuple[str, ...] = Field(
            default_factory=tuple,
            description="Symbols moved to facade class",
        )
        created_classes: tuple[str, ...] = Field(
            default_factory=tuple,
            description="Facade classes created during migration",
        )

    class MRORewriteResult(FlextModels.ArbitraryTypesModel):
        """Reference rewrite summary for one file."""

        file: str = Field(min_length=1, description="Absolute file path")
        replacements: int = Field(ge=0, description="Reference replacements applied")

    class MROMigrationReport(FlextModels.ArbitraryTypesModel):
        """End-to-end report for migrate-to-mro command execution."""

        workspace: str = Field(min_length=1, description="Workspace root path")
        target: str = Field(min_length=1, description="constants|typings|all")
        dry_run: bool = Field(description="Dry-run indicator")
        files_scanned: int = Field(ge=0, description="Total scanned Python files")
        files_with_candidates: int = Field(
            ge=0,
            description="Files containing movable declarations",
        )
        migrations: tuple[FlextInfraRefactorAstGrepModels.MROFileMigration, ...] = (
            Field(
                default_factory=tuple,
                description="File migration summaries",
            )
        )
        rewrites: tuple[FlextInfraRefactorAstGrepModels.MRORewriteResult, ...] = Field(
            default_factory=tuple,
            description="Reference rewrite summaries",
        )
        remaining_violations: int = Field(
            ge=0,
            description="Loose declarations remaining after run",
        )
        mro_failures: int = Field(ge=0, description="MRO validation failures")
        stash_ref: str = Field(default="", description="Git stash rollback ref")
        warnings: tuple[str, ...] = Field(default_factory=tuple, description="Warnings")
        errors: tuple[str, ...] = Field(default_factory=tuple, description="Errors")

    class EngineConfig(FlextModels.FrozenStrictModel):
        model_config = ConfigDict(extra="ignore", frozen=True)

        project_scan_dirs: list[str] = Field(
            default_factory=lambda: ["src", "tests", "scripts", "examples"],
            description="Relative directories scanned for candidate files",
        )
        ignore_patterns: list[str] = Field(
            default_factory=list,
            description="Glob/file patterns ignored during scan",
        )
        file_extensions: list[str] = Field(
            default_factory=list,
            description="Allowed file extensions (empty = all by pattern)",
        )

    class RuleConfigs:
        """Configuration schemas parsed by refactor rules at runtime."""

        class MethodOrderRule(FlextModels.FrozenStrictModel):
            """A declarative method ordering rule for class reconstruction."""

            model_config = ConfigDict(extra="ignore")

            class PatternRule(FlextModels.FrozenStrictModel):
                """Structured matcher entry for method pattern rules."""

                regex: str = Field(default="", description="Regex matcher")
                decorators: list[str] = Field(
                    default_factory=list,
                    description="Required decorators for this pattern",
                )

            category: str | None = Field(default=None, description="Method category")
            visibility: str | None = Field(
                default=None, description="Visibility filter",
            )
            exclude_decorators: list[str] = Field(
                default_factory=list,
                description="Decorators to exclude",
            )
            decorators: list[str] = Field(
                default_factory=list,
                description="Decorators to match",
            )
            patterns: list[str | PatternRule] = Field(
                default_factory=list,
                description="Pattern rules",
            )
            order: list[str] = Field(
                default_factory=list,
                description="Explicit method order",
            )

        class SignatureMigration(FlextModels.FrozenStrictModel):
            """Declarative signature migration rule for callsite propagation."""

            id: str = Field(default="signature-migration", description="Migration ID")
            enabled: bool = Field(
                default=True, description="Whether migration is active",
            )
            target_qualified_names: list[str] = Field(
                default_factory=list, description="Qualified names to match",
            )
            target_simple_names: list[str] = Field(
                default_factory=list, description="Simple names to match",
            )
            keyword_renames: dict[str, str] = Field(
                default_factory=dict, description="Keyword rename mapping",
            )
            remove_keywords: list[str] = Field(
                default_factory=list, description="Keywords to remove",
            )
            add_keywords: dict[str, str] = Field(
                default_factory=dict, description="Keywords to add",
            )

        class ImportModernizerRuleConfig(FlextModels.FrozenStrictModel):
            """Configuration for a single import modernizer rule."""

            module: str = Field(default="", description="Module path to modernize")
            symbol_mapping: dict[str, str] = Field(
                default_factory=dict, description="Symbol-to-alias mapping",
            )


__all__ = ["FlextInfraRefactorAstGrepModels"]
