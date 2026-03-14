"""Ast-grep and MRO migration model mixins for refactor."""

from __future__ import annotations

from typing import Annotated

from pydantic import AliasPath, ConfigDict, Field

from flext_core import FlextModels


class FlextInfraRefactorAstGrepModels:
    """Mixin containing ast-grep and migration model contracts."""

    class AstGrepMatchEnvelope(FlextModels.ArbitraryTypesModel):
        """Compact ast-grep envelope carrying file, symbol and location."""

        model_config = ConfigDict(extra="ignore", populate_by_name=True)

        file: Annotated[str, Field(min_length=1, description="Matched file path")]
        symbol_name: Annotated[
            str | None,
            Field(
                default=None,
                validation_alias=AliasPath("metaVariables", "single", "NAME", "text"),
                description="Captured symbol name from ast-grep metadata",
            ),
        ]
        start_line: Annotated[
            int | None,
            Field(
                default=None,
                validation_alias=AliasPath("range", "start", "line"),
                description="Start line from ast-grep range",
            ),
        ]

    class MROSymbolCandidate(FlextModels.ArbitraryTypesModel):
        """Unified symbol candidate used by MRO scan and rewrites."""

        model_config = ConfigDict(frozen=True)

        symbol: Annotated[str, Field(min_length=1, description="Symbol name")]
        line: Annotated[int, Field(ge=1, description="Source line number")]
        kind: Annotated[
            str, Field(default="constant", description="constant|typevar|typealias")
        ] = "constant"
        class_name: Annotated[
            str, Field(default="", description="Target class name")
        ] = ""
        facade_name: Annotated[
            str, Field(default="", description="Facade alias/import name")
        ] = ""

    class MROImportRewrite(FlextModels.ArbitraryTypesModel):
        """Unified import rewrite payload for MRO reference updates."""

        model_config = ConfigDict(frozen=True)

        module: Annotated[str, Field(min_length=1, description="Import module path")]
        import_name: Annotated[
            str, Field(min_length=1, description="Imported symbol name")
        ]
        as_name: Annotated[
            str | None, Field(default=None, description="Optional alias")
        ] = None
        symbol: Annotated[
            str, Field(default="", description="Resolved symbol in facade")
        ] = ""
        facade_name: Annotated[
            str, Field(default="", description="Facade alias/import name")
        ] = ""

    class MROScanReport(FlextModels.ArbitraryTypesModel):
        """Scan result for one constants module candidate file."""

        model_config = ConfigDict(frozen=True)

        file: Annotated[str, Field(min_length=1, description="Absolute file path")]
        module: Annotated[str, Field(min_length=1, description="Import module path")]
        constants_class: Annotated[
            str,
            Field(
                default="",
                description="First constants class name",
            ),
        ] = ""
        facade_alias: Annotated[
            str, Field(default="c", description="Facade alias letter")
        ] = "c"
        candidates: Annotated[
            tuple[FlextInfraRefactorAstGrepModels.MROSymbolCandidate, ...],
            Field(
                default_factory=tuple,
                description="Module-level symbol candidates",
            ),
        ] = Field(default_factory=tuple)

    class MROFileMigration(FlextModels.ArbitraryTypesModel):
        """Migration summary for one transformed file."""

        file: Annotated[str, Field(min_length=1, description="Absolute file path")]
        module: Annotated[str, Field(min_length=1, description="Import module path")]
        moved_symbols: Annotated[
            tuple[str, ...],
            Field(
                default_factory=tuple,
                description="Symbols moved to facade class",
            ),
        ]
        created_classes: Annotated[
            tuple[str, ...],
            Field(
                default_factory=tuple,
                description="Facade classes created during migration",
            ),
        ]

    class MRORewriteResult(FlextModels.ArbitraryTypesModel):
        """Reference rewrite summary for one file."""

        file: Annotated[str, Field(min_length=1, description="Absolute file path")]
        replacements: Annotated[
            int, Field(ge=0, description="Reference replacements applied")
        ]

    class MROMigrationReport(FlextModels.ArbitraryTypesModel):
        """End-to-end report for migrate-to-mro command execution."""

        workspace: Annotated[
            str, Field(min_length=1, description="Workspace root path")
        ]
        target: Annotated[str, Field(min_length=1, description="constants|typings|all")]
        dry_run: Annotated[bool, Field(description="Dry-run indicator")]
        files_scanned: Annotated[
            int, Field(ge=0, description="Total scanned Python files")
        ]
        files_with_candidates: Annotated[
            int,
            Field(
                ge=0,
                description="Files containing movable declarations",
            ),
        ]
        migrations: Annotated[
            tuple[FlextInfraRefactorAstGrepModels.MROFileMigration, ...],
            Field(
                default_factory=tuple,
                description="File migration summaries",
            ),
        ]
        rewrites: Annotated[
            tuple[FlextInfraRefactorAstGrepModels.MRORewriteResult, ...],
            Field(
                default_factory=tuple,
                description="Reference rewrite summaries",
            ),
        ]
        remaining_violations: Annotated[
            int,
            Field(
                ge=0,
                description="Loose declarations remaining after run",
            ),
        ]
        mro_failures: Annotated[int, Field(ge=0, description="MRO validation failures")]
        stash_ref: Annotated[
            str, Field(default="", description="Git stash rollback ref")
        ]
        warnings: Annotated[
            tuple[str, ...], Field(default_factory=tuple, description="Warnings")
        ]
        errors: Annotated[
            tuple[str, ...], Field(default_factory=tuple, description="Errors")
        ]

    class EngineConfig(FlextModels.FrozenStrictModel):
        model_config = ConfigDict(extra="ignore", frozen=True)

        project_scan_dirs: Annotated[
            list[str],
            Field(
                default_factory=lambda: ["src", "tests", "scripts", "examples"],
                description="Relative directories scanned for candidate files",
            ),
        ]
        ignore_patterns: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Glob/file patterns ignored during scan",
            ),
        ]
        file_extensions: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Allowed file extensions (empty = all by pattern)",
            ),
        ]

    class RuleConfigs:
        """Configuration schemas parsed by refactor rules at runtime."""

        class MethodOrderRule(FlextModels.FrozenStrictModel):
            """A declarative method ordering rule for class reconstruction."""

            model_config = ConfigDict(extra="ignore")

            class PatternRule(FlextModels.FrozenStrictModel):
                """Structured matcher entry for method pattern rules."""

                regex: Annotated[str, Field(default="", description="Regex matcher")]
                decorators: Annotated[
                    list[str],
                    Field(
                        default_factory=list,
                        description="Required decorators for this pattern",
                    ),
                ]

            category: Annotated[
                str | None, Field(default=None, description="Method category")
            ]
            visibility: Annotated[
                str | None,
                Field(
                    default=None,
                    description="Visibility filter",
                ),
            ]
            exclude_decorators: Annotated[
                list[str],
                Field(
                    default_factory=list,
                    description="Decorators to exclude",
                ),
            ]
            decorators: Annotated[
                list[str],
                Field(
                    default_factory=list,
                    description="Decorators to match",
                ),
            ]
            patterns: Annotated[
                list[str],
                Field(
                    default_factory=list,
                    description="Pattern rules",
                ),
            ]
            order: Annotated[
                list[str],
                Field(
                    default_factory=list,
                    description="Explicit method order",
                ),
            ]

        class SignatureMigration(FlextModels.FrozenStrictModel):
            """Declarative signature migration rule for callsite propagation."""

            id: Annotated[
                str, Field(default="signature-migration", description="Migration ID")
            ]
            enabled: Annotated[
                bool,
                Field(
                    default=True,
                    description="Whether migration is active",
                ),
            ]
            target_qualified_names: Annotated[
                list[str],
                Field(
                    default_factory=list,
                    description="Qualified names to match",
                ),
            ]
            target_simple_names: Annotated[
                list[str],
                Field(
                    default_factory=list,
                    description="Simple names to match",
                ),
            ]
            keyword_renames: Annotated[
                dict[str, str],
                Field(
                    default_factory=dict,
                    description="Keyword rename mapping",
                ),
            ]
            remove_keywords: Annotated[
                list[str],
                Field(
                    default_factory=list,
                    description="Keywords to remove",
                ),
            ]
            add_keywords: Annotated[
                dict[str, str],
                Field(
                    default_factory=dict,
                    description="Keywords to add",
                ),
            ]

        class ImportModernizerRuleConfig(FlextModels.FrozenStrictModel):
            """Configuration for a single import modernizer rule."""

            module: Annotated[
                str, Field(default="", description="Module path to modernize")
            ]
            symbol_mapping: Annotated[
                dict[str, str],
                Field(
                    default_factory=dict,
                    description="Symbol-to-alias mapping",
                ),
            ]


__all__ = ["FlextInfraRefactorAstGrepModels"]
