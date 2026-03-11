"""Typed loader for deps tool_config.yml."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from yaml import YAMLError, safe_load

from flext_core import r
from flext_infra import c, t


class FlextInfraRuffFormatConfig(BaseModel):
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


class FlextInfraRuffIsortConfig(BaseModel):
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


class FlextInfraRuffLintConfig(BaseModel):
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
    isort: FlextInfraRuffIsortConfig
    per_file_ignores: dict[str, list[str]] = Field(
        alias="per-file-ignores",
        description="Per-file ignore mapping from glob pattern to ruff rule IDs.",
    )


class FlextInfraRuffConfig(BaseModel):
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
    format: FlextInfraRuffFormatConfig
    lint: FlextInfraRuffLintConfig


class FlextInfraMypyConfig(BaseModel):
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


class FlextInfraPydanticMypyConfig(BaseModel):
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


class FlextInfraPyrightConfig(BaseModel):
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


class FlextInfraPyreflyConfig(BaseModel):
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


class FlextInfraPytestConfig(BaseModel):
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


class FlextInfraTomlsortConfig(BaseModel):
    """tomlsort baseline settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    all: bool = Field(description="Sort all TOML tables and entries.")
    in_place: bool = Field(description="Apply TOML sorting in place.")
    sort_first: list[str] = Field(
        description="Top-level TOML sections ordered first.",
    )


class FlextInfraYamlfixConfig(BaseModel):
    """yamlfix baseline settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    line_length: int = Field(description="Maximum YAML line length.")
    preserve_quotes: bool = Field(description="Preserve quote style in YAML output.")
    whitelines: int = Field(description="Blank line count between YAML entries.")
    section_whitelines: int = Field(
        description="Blank line count between YAML sections.",
    )
    explicit_start: bool = Field(description="Emit explicit YAML start marker.")


class FlextInfraCoverageFailUnderConfig(BaseModel):
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


class FlextInfraCoverageConfig(BaseModel):
    """Coverage baseline settings loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    fail_under: FlextInfraCoverageFailUnderConfig = Field(
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


class FlextInfraToolConfigTools(BaseModel):
    """Tool map loaded from YAML."""

    model_config = ConfigDict(extra="forbid")
    ruff: FlextInfraRuffConfig
    mypy: FlextInfraMypyConfig
    pydantic_mypy: FlextInfraPydanticMypyConfig = Field(
        alias="pydantic-mypy",
        description="Pydantic mypy plugin configuration.",
    )
    pyright: FlextInfraPyrightConfig
    pyrefly: FlextInfraPyreflyConfig
    pytest: FlextInfraPytestConfig
    tomlsort: FlextInfraTomlsortConfig
    yamlfix: FlextInfraYamlfixConfig
    coverage: FlextInfraCoverageConfig = Field(
        description="Coverage configuration with per-project-type thresholds.",
    )


class FlextInfraToolConfigDocument(BaseModel):
    """Root schema for tool_config.yml."""

    model_config = ConfigDict(extra="forbid")
    tools: FlextInfraToolConfigTools


@lru_cache(maxsize=1)
def _load_tool_config_cached() -> r[FlextInfraToolConfigDocument]:
    """Load, validate, and cache tool_config.yml."""
    try:
        raw_text = (
            files("flext_infra.deps")
            .joinpath("tool_config.yml")
            .read_text(
                encoding=c.Infra.Encoding.DEFAULT,
            )
        )
        parsed_raw: t.ContainerValue | None = safe_load(raw_text)
        if not isinstance(parsed_raw, Mapping):
            return r[FlextInfraToolConfigDocument].fail(
                "tool_config.yml must contain a top-level mapping",
            )
        payload: dict[str, t.ContainerValue] = dict(parsed_raw.items())
        validated = FlextInfraToolConfigDocument.model_validate(payload)
        return r[FlextInfraToolConfigDocument].ok(validated)
    except (FileNotFoundError, OSError, YAMLError, ValidationError, TypeError) as exc:
        return r[FlextInfraToolConfigDocument].fail(
            f"failed to load tool_config.yml: {exc}",
        )


def load_tool_config() -> r[FlextInfraToolConfigDocument]:
    """Public cached accessor for tool_config.yml."""
    return _load_tool_config_cached()


__all__ = [
    "FlextInfraToolConfigDocument",
    "load_tool_config",
]
