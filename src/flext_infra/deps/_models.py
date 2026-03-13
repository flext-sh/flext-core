"""Domain models for the deps subpackage."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from flext_core import FlextModels
from flext_infra.typings import t


class RuffFormatConfig(FlextModels.ArbitraryTypesModel):
    """Ruff format settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    docstring_code_format: bool = Field(
        alias="docstring-code-format",
        description="Enable ruff docstring code block formatting.",
    )
    indent_style: str = Field(
        alias="indent-style",
        description="Indent style for ruff formatter output.",
    )
    line_ending: str = Field(
        alias="line-ending",
        description="Line ending style for ruff formatter output.",
    )
    quote_style: str = Field(
        alias="quote-style",
        description="Quote style for ruff formatter output.",
    )


class RuffIsortConfig(FlextModels.ArbitraryTypesModel):
    """Ruff isort settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    combine_as_imports: bool = Field(
        alias="combine-as-imports",
        description="Combine `as` imports in grouped isort blocks.",
    )
    force_single_line: bool = Field(
        alias="force-single-line",
        description="Force single-line imports in isort output.",
    )
    split_on_trailing_comma: bool = Field(
        alias="split-on-trailing-comma",
        description="Split imports when a trailing comma exists.",
    )


class RuffLintConfig(FlextModels.ArbitraryTypesModel):
    """Ruff lint settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    select: list[str] = Field(
        default_factory=list,
        description="Ruff lint rule selectors.",
    )
    ignore: list[str] = Field(
        default_factory=list,
        description="Ruff lint rule ignore list.",
    )
    isort: RuffIsortConfig
    per_file_ignores: dict[str, list[str]] = Field(
        alias="per-file-ignores",
        description="Per-file ignore mapping from glob pattern to ruff rule IDs.",
    )


class RuffConfig(FlextModels.ArbitraryTypesModel):
    """Ruff top-level settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    exclude: list[str] = Field(
        default_factory=list,
        description="Directory/file globs excluded from ruff checks.",
    )
    fix: bool = Field(description="Enable automatic ruff fixes.")
    line_length: int = Field(alias="line-length", description="Maximum line length.")
    preview: bool = Field(description="Enable preview ruff behavior.")
    respect_gitignore: bool = Field(
        alias="respect-gitignore",
        description="Respect .gitignore exclusions.",
    )
    show_fixes: bool = Field(
        alias="show-fixes",
        description="Display fixed violations in ruff output.",
    )
    src: list[str] = Field(
        default_factory=list,
        description="Source roots used by ruff import analysis.",
    )
    target_version: str = Field(
        alias="target-version",
        description="Python target version for ruff.",
    )
    format: RuffFormatConfig
    lint: RuffLintConfig


class MypyConfig(FlextModels.ArbitraryTypesModel):
    """Mypy baseline settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    plugins: list[str] = Field(
        default_factory=list,
        description="Mypy plugins list.",
    )
    disabled_error_codes: list[str] = Field(
        alias="disabled-error-codes",
        description="Mypy error codes disabled by default.",
    )
    boolean_settings: dict[str, bool] = Field(
        alias="boolean-settings",
        description="Mypy boolean settings keyed by option name.",
    )


class PydanticMypyConfig(FlextModels.ArbitraryTypesModel):
    """Pydantic mypy plugin settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    init_forbid_extra: bool = Field(
        description="Enable forbid-extra init behavior in pydantic mypy plugin.",
    )
    init_typed: bool = Field(
        description="Enable typed __init__ signatures in pydantic mypy plugin.",
    )
    warn_required_dynamic_aliases: bool = Field(
        description="Warn on required dynamic aliases in pydantic mypy plugin.",
    )


class PyrightConfig(FlextModels.ArbitraryTypesModel):
    """Pyright strict settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    strict_settings: dict[str, str] = Field(
        alias="strict-settings",
        description="Pyright strict baseline options.",
    )
    extended_settings: dict[str, str] = Field(
        default_factory=dict,
        alias="extended-settings",
        description="Pyright extended settings options.",
    )


class PyreflyConfig(FlextModels.ArbitraryTypesModel):
    """Pyrefly strict settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    strict_errors: list[str] = Field(
        alias="strict-errors",
        description="Pyrefly errors enabled as strict defaults.",
    )
    disabled_errors: list[str] = Field(
        alias="disabled-errors",
        description="Pyrefly errors disabled by default.",
    )


class PytestConfig(FlextModels.ArbitraryTypesModel):
    """Pytest baseline settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    standard_markers: list[str] = Field(
        alias="standard-markers",
        description="Standard pytest markers enforced by modernizer.",
    )
    standard_addopts: list[str] = Field(
        alias="standard-addopts",
        description="Standard pytest addopts enforced by modernizer.",
    )


class TomlsortConfig(FlextModels.ArbitraryTypesModel):
    """tomlsort baseline settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    all: bool = Field(description="Sort all TOML tables and entries.")
    in_place: bool = Field(description="Apply TOML sorting in place.")
    sort_first: list[str] = Field(
        description="Top-level TOML sections ordered first.",
    )


class YamlfixConfig(FlextModels.ArbitraryTypesModel):
    """yamlfix baseline settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    line_length: int = Field(description="Maximum YAML line length.")
    preserve_quotes: bool = Field(description="Preserve quote style in YAML output.")
    whitelines: int = Field(description="Blank line count between YAML entries.")
    section_whitelines: int = Field(
        description="Blank line count between YAML sections.",
    )
    explicit_start: bool = Field(description="Emit explicit YAML start marker.")


class CoverageFailUnderConfig(FlextModels.ArbitraryTypesModel):
    """Coverage fail-under thresholds by layer."""

    model_config = ConfigDict(extra="forbid")
    core: int = Field(
        description="Minimum coverage percentage required for core layer."
    )
    domain: int = Field(
        description="Minimum coverage percentage required for domain layer."
    )
    platform: int = Field(
        description="Minimum coverage percentage required for platform layer."
    )
    integration: int = Field(
        description="Minimum coverage percentage required for integration layer."
    )
    app: int = Field(description="Minimum coverage percentage required for app layer.")


class CoverageConfig(FlextModels.ArbitraryTypesModel):
    """Coverage baseline settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    fail_under: CoverageFailUnderConfig = Field(
        alias="fail-under",
        description="Coverage fail-under thresholds by layer.",
    )
    show_missing: bool = Field(
        default=True,
        alias="show-missing",
        description="Display missing lines in coverage report.",
    )
    skip_covered: bool = Field(
        default=False,
        alias="skip-covered",
        description="Skip covered files in coverage report.",
    )
    precision: int = Field(
        default=2,
        description="Decimal precision for coverage percentages.",
    )


class ToolConfigTools(FlextModels.ArbitraryTypesModel):
    """Tool map loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    ruff: RuffConfig
    mypy: MypyConfig
    pydantic_mypy: PydanticMypyConfig = Field(
        alias="pydantic-mypy",
        description="Pydantic mypy plugin configuration.",
    )
    pyright: PyrightConfig
    pyrefly: PyreflyConfig
    pytest: PytestConfig
    tomlsort: TomlsortConfig
    yamlfix: YamlfixConfig
    coverage: CoverageConfig = Field(
        description="Coverage configuration with per-project-type thresholds.",
    )


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
    core: ProjectTypeOverrideConfig = Field(
        default_factory=ProjectTypeOverrideConfig,
    )
    domain: ProjectTypeOverrideConfig = Field(
        default_factory=ProjectTypeOverrideConfig,
    )
    platform: ProjectTypeOverrideConfig = Field(
        default_factory=ProjectTypeOverrideConfig,
    )
    integration: ProjectTypeOverrideConfig = Field(
        default_factory=ProjectTypeOverrideConfig,
    )
    app: ProjectTypeOverrideConfig = Field(
        default_factory=ProjectTypeOverrideConfig,
    )


class DependencyLimitsInfo(FlextModels.ArbitraryTypesModel):
    """Dependency limits configuration metadata."""

    python_version: str | None = None
    limits_path: str = Field(default="")


class PipCheckReport(FlextModels.ArbitraryTypesModel):
    """Pip check execution report with status and output lines."""

    ok: bool = True
    lines: list[str] = Field(default_factory=list)


class WorkspaceDependencyReport(FlextModels.ArbitraryTypesModel):
    """Workspace-level dependency analysis report aggregating all projects."""

    workspace: str
    projects: dict[str, ProjectRuntimeReport] = Field(default_factory=dict)
    pip_check: PipCheckReport | None = None
    dependency_limits: DependencyLimitsInfo | None = None


class ToolConfigDocument(FlextModels.ArbitraryTypesModel):
    """Root schema for tool_config.yml."""

    model_config = ConfigDict(extra="forbid")
    tools: ToolConfigTools
    project_type_overrides: ProjectTypeOverridesConfig = Field(
        alias="project-type-overrides",
        default_factory=ProjectTypeOverridesConfig,
        description="Per-project-type configuration overrides.",
    )


class DependencyReport(FlextModels.ArbitraryTypesModel):
    """Report of dependency detection for a single project."""

    project: str = Field(min_length=1, description="Project name")
    missing: list[str] = Field(
        default_factory=list,
        description="Missing dependencies",
    )
    unused: list[str] = Field(
        default_factory=list,
        description="Unused dependencies",
    )
    outdated: list[str] = Field(
        default_factory=list,
        description="Outdated dependencies",
    )


class ModernizerFileChanges(FlextModels.ArbitraryTypesModel):
    """Modernizer changes for one pyproject file."""

    file: str = Field(min_length=1, description="Relative pyproject path")
    changes: list[str] = Field(default_factory=list, description="Applied changes")


def _empty_issue_list() -> list[t.Infra.IssueMap]:
    """Return empty issue list for reports."""
    return []


class DeptryIssueGroups(FlextModels.ArbitraryTypesModel):
    """Deptry issue grouping model by error code (DEP001-DEP004)."""

    dep001: list[t.Infra.IssueMap] = Field(default_factory=_empty_issue_list)
    dep002: list[t.Infra.IssueMap] = Field(default_factory=_empty_issue_list)
    dep003: list[t.Infra.IssueMap] = Field(default_factory=_empty_issue_list)
    dep004: list[t.Infra.IssueMap] = Field(default_factory=_empty_issue_list)


class DeptryReport(FlextModels.ArbitraryTypesModel):
    """Deptry analysis report with categorized issue modules."""

    missing: list[str] = Field(default_factory=list)
    unused: list[str] = Field(default_factory=list)
    transitive: list[str] = Field(default_factory=list)
    dev_in_runtime: list[str] = Field(default_factory=list)
    raw_count: int = Field(default=0, ge=0)


class ProjectDependencyReport(FlextModels.ArbitraryTypesModel):
    """Project-level dependency report combining deptry results."""

    project: str = Field(min_length=1)
    deptry: DeptryReport


class TypingsReport(FlextModels.ArbitraryTypesModel):
    """Typing stubs analysis report with required/current/delta packages."""

    required_packages: list[str] = Field(default_factory=list)
    hinted: list[str] = Field(default_factory=list)
    missing_modules: list[str] = Field(default_factory=list)
    current: list[str] = Field(default_factory=list)
    to_add: list[str] = Field(default_factory=list)
    to_remove: list[str] = Field(default_factory=list)
    limits_applied: bool = False
    python_version: str | None = None


class ProjectRuntimeReport(FlextModels.ArbitraryTypesModel):
    deptry: DeptryReport
    typings: TypingsReport | None = None


class FlextInfraDepsModels:
    """Models for dependency detection and modernization reporting."""

    RuffFormatConfig = RuffFormatConfig
    RuffIsortConfig = RuffIsortConfig
    RuffLintConfig = RuffLintConfig
    RuffConfig = RuffConfig
    MypyConfig = MypyConfig
    PydanticMypyConfig = PydanticMypyConfig
    PyrightConfig = PyrightConfig
    PyreflyConfig = PyreflyConfig
    PytestConfig = PytestConfig
    TomlsortConfig = TomlsortConfig
    YamlfixConfig = YamlfixConfig
    CoverageFailUnderConfig = CoverageFailUnderConfig
    CoverageConfig = CoverageConfig
    ToolConfigTools = ToolConfigTools
    ProjectTypeOverrideConfig = ProjectTypeOverrideConfig
    ProjectTypeOverridesConfig = ProjectTypeOverridesConfig
    ToolConfigDocument = ToolConfigDocument
    DependencyLimitsInfo = DependencyLimitsInfo
    PipCheckReport = PipCheckReport
    ProjectRuntimeReport = ProjectRuntimeReport
    WorkspaceDependencyReport = WorkspaceDependencyReport
    DependencyReport = DependencyReport
    ModernizerFileChanges = ModernizerFileChanges
    DeptryIssueGroups = DeptryIssueGroups
    DeptryReport = DeptryReport
    ProjectDependencyReport = ProjectDependencyReport
    TypingsReport = TypingsReport


class FlextInfraDependencyDetectionModels(FlextInfraDepsModels):
    """Alias for dependency detection models to maintain compatibility during migration."""


dm = FlextInfraDepsModels


__all__ = [
    "DependencyLimitsInfo",
    "DependencyReport",
    "DeptryIssueGroups",
    "DeptryReport",
    "FlextInfraDependencyDetectionModels",
    "FlextInfraDepsModels",
    "ModernizerFileChanges",
    "MypyConfig",
    "PipCheckReport",
    "ProjectDependencyReport",
    "ProjectRuntimeReport",
    "ProjectTypeOverrideConfig",
    "ProjectTypeOverridesConfig",
    "PydanticMypyConfig",
    "PyreflyConfig",
    "PyrightConfig",
    "PytestConfig",
    "RuffConfig",
    "RuffFormatConfig",
    "RuffIsortConfig",
    "RuffLintConfig",
    "TomlsortConfig",
    "ToolConfigDocument",
    "ToolConfigTools",
    "TypingsReport",
    "WorkspaceDependencyReport",
    "YamlfixConfig",
    "dm",
]
