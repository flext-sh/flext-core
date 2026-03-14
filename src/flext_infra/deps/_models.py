"""Domain models for the deps subpackage."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import ConfigDict, Field

from flext_core import FlextModels

type _DeptryIssueMap = Mapping[str, str | int | float | bool | None]


def _empty_deptry_issues() -> list[_DeptryIssueMap]:
    return []


class FlextInfraDepsModels:
    """Models for dependency detection and modernization reporting."""

    class RuffFormatConfig(FlextModels.ArbitraryTypesModel):
        """Ruff format settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        docstring_code_format: Annotated[
            bool,
            Field(
                alias="docstring-code-format",
                description="Enable ruff docstring code block formatting.",
            ),
        ]
        indent_style: Annotated[
            str,
            Field(
                alias="indent-style",
                description="Indent style for ruff formatter output.",
            ),
        ]
        line_ending: Annotated[
            str,
            Field(
                alias="line-ending",
                description="Line ending style for ruff formatter output.",
            ),
        ]
        quote_style: Annotated[
            str,
            Field(
                alias="quote-style",
                description="Quote style for ruff formatter output.",
            ),
        ]

    class RuffIsortConfig(FlextModels.ArbitraryTypesModel):
        """Ruff isort settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        combine_as_imports: Annotated[
            bool,
            Field(
                alias="combine-as-imports",
                description="Combine `as` imports in grouped isort blocks.",
            ),
        ]
        force_single_line: Annotated[
            bool,
            Field(
                alias="force-single-line",
                description="Force single-line imports in isort output.",
            ),
        ]
        split_on_trailing_comma: Annotated[
            bool,
            Field(
                alias="split-on-trailing-comma",
                description="Split imports when a trailing comma exists.",
            ),
        ]

    class RuffLintConfig(FlextModels.ArbitraryTypesModel):
        """Ruff lint settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        select: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Ruff lint rule selectors.",
            ),
        ]
        ignore: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Ruff lint rule ignore list.",
            ),
        ]
        isort: FlextInfraDepsModels.RuffIsortConfig
        per_file_ignores: Annotated[
            dict[str, list[str]],
            Field(
                alias="per-file-ignores",
                description="Per-file ignore mapping from glob pattern to ruff rule IDs.",
            ),
        ]

    class RuffConfig(FlextModels.ArbitraryTypesModel):
        """Ruff top-level settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        exclude: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Directory/file globs excluded from ruff checks.",
            ),
        ]
        fix: Annotated[bool, Field(description="Enable automatic ruff fixes")]
        line_length: Annotated[
            int, Field(alias="line-length", description="Maximum line length.")
        ]
        preview: Annotated[bool, Field(description="Enable preview ruff behavior.")]
        respect_gitignore: Annotated[
            bool,
            Field(
                alias="respect-gitignore",
                description="Respect .gitignore exclusions.",
            ),
        ]
        show_fixes: Annotated[
            bool,
            Field(
                alias="show-fixes",
                description="Display fixed violations in ruff output.",
            ),
        ]
        src: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Source roots used by ruff import analysis.",
            ),
        ]
        target_version: Annotated[
            str,
            Field(
                alias="target-version",
                description="Python target version for ruff.",
            ),
        ]
        format: FlextInfraDepsModels.RuffFormatConfig
        lint: FlextInfraDepsModels.RuffLintConfig

    class MypyConfig(FlextModels.ArbitraryTypesModel):
        """Mypy baseline settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        plugins: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Mypy plugins list.",
            ),
        ]
        disabled_error_codes: Annotated[
            list[str],
            Field(
                alias="disabled-error-codes",
                description="Mypy error codes disabled by default.",
            ),
        ]
        boolean_settings: Annotated[
            dict[str, bool],
            Field(
                alias="boolean-settings",
                description="Mypy boolean settings keyed by option name.",
            ),
        ]

    class PydanticMypyConfig(FlextModels.ArbitraryTypesModel):
        """Pydantic mypy plugin settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        init_forbid_extra: Annotated[
            bool,
            Field(
                description="Enable forbid-extra init behavior in pydantic mypy plugin.",
            ),
        ]
        init_typed: Annotated[
            bool,
            Field(
                description="Enable typed __init__ signatures in pydantic mypy plugin.",
            ),
        ]
        warn_required_dynamic_aliases: Annotated[
            bool,
            Field(
                description="Warn on required dynamic aliases in pydantic mypy plugin.",
            ),
        ]

    class PyrightConfig(FlextModels.ArbitraryTypesModel):
        """Pyright strict settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        strict_settings: Annotated[
            dict[str, str],
            Field(
                alias="strict-settings",
                description="Pyright strict baseline options.",
            ),
        ]
        extended_settings: Annotated[
            dict[str, str],
            Field(
                default_factory=dict,
                alias="extended-settings",
                description="Pyright extended settings options.",
            ),
        ]

    class PyreflyConfig(FlextModels.ArbitraryTypesModel):
        """Pyrefly strict settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        strict_errors: Annotated[
            list[str],
            Field(
                alias="strict-errors",
                description="Pyrefly errors enabled as strict defaults.",
            ),
        ]
        disabled_errors: Annotated[
            list[str],
            Field(
                alias="disabled-errors",
                description="Pyrefly errors disabled by default.",
            ),
        ]

    class PytestConfig(FlextModels.ArbitraryTypesModel):
        """Pytest baseline settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        standard_markers: Annotated[
            list[str],
            Field(
                alias="standard-markers",
                description="Standard pytest markers enforced by modernizer.",
            ),
        ]
        standard_addopts: Annotated[
            list[str],
            Field(
                alias="standard-addopts",
                description="Standard pytest addopts enforced by modernizer.",
            ),
        ]

    class TomlsortConfig(FlextModels.ArbitraryTypesModel):
        """tomlsort baseline settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        all: Annotated[bool, Field(description="Sort all TOML tables and entries.")]
        in_place: Annotated[bool, Field(description="Apply TOML sorting in place.")]
        sort_first: Annotated[
            list[str],
            Field(
                description="Top-level TOML sections ordered first.",
            ),
        ]

    class YamlfixConfig(FlextModels.ArbitraryTypesModel):
        """yamlfix baseline settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        line_length: Annotated[int, Field(description="Maximum YAML line length.")]
        preserve_quotes: Annotated[
            bool, Field(description="Preserve quote style in YAML output.")
        ]
        whitelines: Annotated[
            int, Field(description="Blank line count between YAML entries.")
        ]
        section_whitelines: Annotated[
            int,
            Field(
                description="Blank line count between YAML sections.",
            ),
        ]
        explicit_start: Annotated[
            bool, Field(description="Emit explicit YAML start marker.")
        ]

    class CoverageFailUnderConfig(FlextModels.ArbitraryTypesModel):
        """Coverage fail-under thresholds by layer."""

        model_config = ConfigDict(extra="forbid")
        core: Annotated[
            int,
            Field(description="Minimum coverage percentage required for core layer."),
        ]
        domain: Annotated[
            int,
            Field(description="Minimum coverage percentage required for domain layer."),
        ]
        platform: Annotated[
            int,
            Field(
                description="Minimum coverage percentage required for platform layer."
            ),
        ]
        integration: Annotated[
            int,
            Field(
                description="Minimum coverage percentage required for integration layer."
            ),
        ]
        app: Annotated[
            int,
            Field(description="Minimum coverage percentage required for app layer."),
        ]

    class CoverageConfig(FlextModels.ArbitraryTypesModel):
        """Coverage baseline settings loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        fail_under: Annotated[
            FlextInfraDepsModels.CoverageFailUnderConfig,
            Field(
                alias="fail-under",
                description="Coverage fail-under thresholds by layer.",
            ),
        ]
        show_missing: Annotated[
            bool,
            Field(
                default=True,
                alias="show-missing",
                description="Display missing lines in coverage report.",
            ),
        ]
        skip_covered: Annotated[
            bool,
            Field(
                default=False,
                alias="skip-covered",
                description="Skip covered files in coverage report.",
            ),
        ]
        precision: Annotated[
            int,
            Field(
                default=2,
                description="Decimal precision for coverage percentages.",
            ),
        ]

    class ToolConfigTools(FlextModels.ArbitraryTypesModel):
        """Tool map loaded from YAML."""

        model_config = ConfigDict(extra="forbid")
        ruff: FlextInfraDepsModels.RuffConfig
        mypy: FlextInfraDepsModels.MypyConfig
        pydantic_mypy: Annotated[
            FlextInfraDepsModels.PydanticMypyConfig,
            Field(
                alias="pydantic-mypy",
                description="Pydantic mypy plugin configuration.",
            ),
        ]
        pyright: FlextInfraDepsModels.PyrightConfig
        pyrefly: FlextInfraDepsModels.PyreflyConfig
        pytest: FlextInfraDepsModels.PytestConfig
        tomlsort: FlextInfraDepsModels.TomlsortConfig
        yamlfix: FlextInfraDepsModels.YamlfixConfig
        coverage: Annotated[
            FlextInfraDepsModels.CoverageConfig,
            Field(
                description="Coverage configuration with per-project-type thresholds.",
            ),
        ]

    class ProjectTypeOverrideConfig(FlextModels.ArbitraryTypesModel):
        """Per-project-type override settings."""

        model_config = ConfigDict(extra="forbid")
        pyright: dict[str, str] = Field(
            default_factory=dict,
            description="Pyright override settings for this project type.",
        )

    class ProjectTypeOverridesConfig(FlextModels.ArbitraryTypesModel):
        """Project-type-specific override matrix from tool_config.yml."""

        model_config = ConfigDict(extra="forbid")
        core: Annotated[
            FlextInfraDepsModels.ProjectTypeOverrideConfig,
            Field(),
        ] = Field(
            default_factory=lambda: FlextInfraDepsModels.ProjectTypeOverrideConfig()
        )
        domain: Annotated[
            FlextInfraDepsModels.ProjectTypeOverrideConfig,
            Field(),
        ] = Field(
            default_factory=lambda: FlextInfraDepsModels.ProjectTypeOverrideConfig()
        )
        platform: Annotated[
            FlextInfraDepsModels.ProjectTypeOverrideConfig,
            Field(),
        ] = Field(
            default_factory=lambda: FlextInfraDepsModels.ProjectTypeOverrideConfig()
        )
        integration: Annotated[
            FlextInfraDepsModels.ProjectTypeOverrideConfig,
            Field(),
        ] = Field(
            default_factory=lambda: FlextInfraDepsModels.ProjectTypeOverrideConfig()
        )
        app: Annotated[
            FlextInfraDepsModels.ProjectTypeOverrideConfig,
            Field(),
        ] = Field(
            default_factory=lambda: FlextInfraDepsModels.ProjectTypeOverrideConfig()
        )

    class DependencyLimitsInfo(FlextModels.ArbitraryTypesModel):
        """Dependency limits configuration metadata."""

        python_version: str | None = None
        limits_path: Annotated[str, Field(default="")]

    class PipCheckReport(FlextModels.ArbitraryTypesModel):
        """Pip check execution report with status and output lines."""

        ok: bool = True
        lines: Annotated[list[str], Field(default_factory=list)]

    class ToolConfigDocument(FlextModels.ArbitraryTypesModel):
        """Root schema for tool_config.yml."""

        model_config = ConfigDict(extra="forbid")
        tools: FlextInfraDepsModels.ToolConfigTools
        project_type_overrides: Annotated[
            FlextInfraDepsModels.ProjectTypeOverridesConfig,
            Field(
                alias="project-type-overrides",
                description="Per-project-type configuration overrides.",
            ),
        ] = Field(
            default_factory=lambda: FlextInfraDepsModels.ProjectTypeOverridesConfig(),
            alias="project-type-overrides",
        )

    class DependencyReport(FlextModels.ArbitraryTypesModel):
        """Report of dependency detection for a single project."""

        project: Annotated[str, Field(min_length=1, description="Project name")]
        missing: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Missing dependencies",
            ),
        ]
        unused: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Unused dependencies",
            ),
        ]
        outdated: Annotated[
            list[str],
            Field(
                default_factory=list,
                description="Outdated dependencies",
            ),
        ]

    class ModernizerFileChanges(FlextModels.ArbitraryTypesModel):
        """Modernizer changes for one pyproject file."""

        file: Annotated[str, Field(min_length=1, description="Relative pyproject path")]
        changes: Annotated[
            list[str], Field(default_factory=list, description="Applied changes")
        ]

    class DeptryIssueGroups(FlextModels.ArbitraryTypesModel):
        """Deptry issue grouping model by error code (DEP001-DEP004)."""

        dep001: list[_DeptryIssueMap] = Field(
            default_factory=_empty_deptry_issues,
            description="DEP001 issues",
        )
        dep002: list[_DeptryIssueMap] = Field(
            default_factory=_empty_deptry_issues,
            description="DEP002 issues",
        )
        dep003: list[_DeptryIssueMap] = Field(
            default_factory=_empty_deptry_issues,
            description="DEP003 issues",
        )
        dep004: list[_DeptryIssueMap] = Field(
            default_factory=_empty_deptry_issues,
            description="DEP004 issues",
        )

    class DeptryReport(FlextModels.ArbitraryTypesModel):
        """Deptry analysis report with categorized issue modules."""

        missing: Annotated[list[str], Field(default_factory=list)]
        unused: Annotated[list[str], Field(default_factory=list)]
        transitive: Annotated[list[str], Field(default_factory=list)]
        dev_in_runtime: Annotated[list[str], Field(default_factory=list)]
        raw_count: Annotated[int, Field(default=0, ge=0)]

    class ProjectDependencyReport(FlextModels.ArbitraryTypesModel):
        """Project-level dependency report combining deptry results."""

        project: Annotated[str, Field(min_length=1)]
        deptry: FlextInfraDepsModels.DeptryReport

    class TypingsReport(FlextModels.ArbitraryTypesModel):
        """Typing stubs analysis report with required/current/delta packages."""

        required_packages: Annotated[list[str], Field(default_factory=list)]
        hinted: Annotated[list[str], Field(default_factory=list)]
        missing_modules: Annotated[list[str], Field(default_factory=list)]
        current: Annotated[list[str], Field(default_factory=list)]
        to_add: Annotated[list[str], Field(default_factory=list)]
        to_remove: Annotated[list[str], Field(default_factory=list)]
        limits_applied: bool = False
        python_version: str | None = None

    class ProjectRuntimeReport(FlextModels.ArbitraryTypesModel):
        """Project runtime dependency and typings report."""

        deptry: FlextInfraDepsModels.DeptryReport
        typings: FlextInfraDepsModels.TypingsReport | None = None

    class WorkspaceDependencyReport(FlextModels.ArbitraryTypesModel):
        """Workspace-level dependency analysis report aggregating all projects."""

        workspace: str
        projects: Annotated[
            dict[str, FlextInfraDepsModels.ProjectRuntimeReport],
            Field(default_factory=dict),
        ]
        pip_check: FlextInfraDepsModels.PipCheckReport | None = None
        dependency_limits: FlextInfraDepsModels.DependencyLimitsInfo | None = None


__all__ = ["FlextInfraDepsModels"]
