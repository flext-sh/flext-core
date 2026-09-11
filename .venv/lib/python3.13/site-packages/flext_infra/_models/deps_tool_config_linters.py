"""Ruff and Mypy tool configuration models for the deps subpackage."""

from __future__ import annotations

from typing import Annotated

from flext_core import m
from flext_infra import t
from flext_infra._models._defaults import ImmutableEmptyMapping
from flext_infra._models.deps_tool_config_project import (
    FlextInfraModelsDepsToolConfigProject,
)


class FlextInfraModelsDepsToolConfigLinters(FlextInfraModelsDepsToolConfigProject):
    """Linters tool configuration models."""

    class CodespellConfig(m.ArbitraryTypesModel):
        """Codespell settings loaded from YAML."""

        check_filenames: Annotated[
            bool,
            m.Field(
                alias="check-filenames",
                description="Check filenames in addition to file contents.",
            ),
        ]
        ignore_words_list: Annotated[
            str,
            m.Field(
                alias="ignore-words-list",
                description="Comma-separated allowlist for known project terms.",
            ),
        ] = ""

    class RuffFormatConfig(m.ArbitraryTypesModel):
        """Ruff format settings loaded from YAML."""

        docstring_code_format: Annotated[
            bool,
            m.Field(
                alias="docstring-code-format",
                description="Enable ruff docstring code block formatting.",
            ),
        ]
        indent_style: Annotated[
            str,
            m.Field(
                alias="indent-style",
                description="Indent style for ruff formatter output.",
            ),
        ]
        line_ending: Annotated[
            str,
            m.Field(
                alias="line-ending",
                description="Line ending style for ruff formatter output.",
            ),
        ]
        quote_style: Annotated[
            str,
            m.Field(
                alias="quote-style",
                description="Quote style for ruff formatter output.",
            ),
        ]
        skip_magic_trailing_comma: Annotated[
            bool,
            m.Field(
                alias="skip-magic-trailing-comma",
                description="Collapse short comma-terminated constructs onto one line.",
            ),
        ]

    class RuffIsortConfig(m.ArbitraryTypesModel):
        """Ruff isort settings loaded from YAML."""

        combine_as_imports: Annotated[
            bool,
            m.Field(
                alias="combine-as-imports",
                description="Combine `as` imports in grouped isort blocks.",
            ),
        ]
        force_single_line: Annotated[
            bool,
            m.Field(
                alias="force-single-line",
                description="Force single-line imports in isort output.",
            ),
        ]
        split_on_trailing_comma: Annotated[
            bool,
            m.Field(
                alias="split-on-trailing-comma",
                description="Split imports when a trailing comma exists.",
            ),
        ]

    class RuffLintConfig(m.ArbitraryTypesModel):
        """Ruff lint settings loaded from YAML."""

        select: Annotated[
            t.StrSequence, m.Field(description="Ruff lint rule selectors.")
        ] = m.Field(default_factory=tuple)
        ignore: Annotated[
            t.StrSequence, m.Field(description="Ruff lint rule ignore list.")
        ] = m.Field(default_factory=tuple)
        ignored_rule_rationales: Annotated[
            t.StrMapping,
            m.Field(
                alias="ignored-rule-rationales",
                description="Global Ruff exclusions mapped to verified architecture rationales.",
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        banned_api: Annotated[
            t.StrMapping,
            m.Field(
                alias="banned-api",
                description="Forbidden direct APIs and their canonical alternatives.",
            ),
        ]
        isort: FlextInfraModelsDepsToolConfigLinters.RuffIsortConfig = m.Field(
            description="Ruff isort configuration"
        )
        per_file_ignores: Annotated[
            t.Infra.PerFileIgnores,
            m.Field(
                alias="per-file-ignores",
                description="Per-file ignore mapping from glob pattern to ruff rule IDs.",
            ),
        ]

    class RuffConfig(m.ArbitraryTypesModel):
        """Ruff top-level settings loaded from YAML."""

        exclude: Annotated[
            t.StrSequence,
            m.Field(description="Directory/file globs excluded from ruff checks."),
        ] = m.Field(default_factory=tuple)
        namespace_packages: Annotated[
            t.StrSequence,
            m.Field(
                alias="namespace-packages",
                description="Intentional PEP 420 namespace roots checked by Ruff.",
            ),
        ] = m.Field(default_factory=tuple)
        fix: Annotated[bool, m.Field(description="Enable automatic ruff fixes")]
        line_length: Annotated[
            int, m.Field(alias="line-length", description="Maximum line length.")
        ]
        preview: Annotated[bool, m.Field(description="Enable preview ruff behavior.")]
        respect_gitignore: Annotated[
            bool,
            m.Field(
                alias="respect-gitignore", description="Respect .gitignore exclusions."
            ),
        ]
        show_fixes: Annotated[
            bool,
            m.Field(
                alias="show-fixes",
                description="Display fixed violations in ruff output.",
            ),
        ]
        src: Annotated[
            t.StrSequence,
            m.Field(description="Source roots used by ruff import analysis."),
        ] = m.Field(default_factory=tuple)
        target_version: Annotated[
            str,
            m.Field(
                alias="target-version", description="Python target version for ruff."
            ),
        ]
        format: FlextInfraModelsDepsToolConfigLinters.RuffFormatConfig = m.Field(
            description="Ruff format configuration"
        )
        lint: FlextInfraModelsDepsToolConfigLinters.RuffLintConfig = m.Field(
            description="Ruff lint configuration"
        )

    class MypyOverrideConfig(m.ArbitraryTypesModel):
        """Single [[tool.mypy.overrides]] entry."""

        modules: Annotated[
            t.StrSequence, m.Field(description="Module patterns for this override.")
        ]
        disable_error_codes: Annotated[
            t.StrSequence,
            m.Field(
                alias="disable-error-codes",
                description="Error codes disabled for these modules.",
            ),
        ]
        justification: Annotated[
            str,
            m.Field(
                description=(
                    "Required citation (GitHub issue / PEP / mypy docs) justifying "
                    "this override. AGENTS.md:319 forbids suppressions without "
                    "evidence; leave empty only for strictly transitional overrides "
                    "with a TODO in the module comment."
                )
            ),
        ] = ""

    class MypyConfig(m.ArbitraryTypesModel):
        """Mypy baseline settings loaded from YAML."""

        plugins: Annotated[t.StrSequence, m.Field(description="Mypy plugins list.")] = (
            m.Field(default_factory=tuple)
        )
        exclude: Annotated[
            str,
            m.Field(
                description="Regex used to exclude generated or fixture-like paths from mypy."
            ),
        ] = ""
        disabled_error_codes: Annotated[
            t.StrMapping,
            m.Field(
                alias="disabled-error-codes",
                description=(
                    "Mypy error codes mapped to their tested facade-FLEXT rationale."
                ),
            ),
        ]
        boolean_settings: Annotated[
            t.BoolMapping,
            m.Field(
                alias="boolean-settings",
                description="Mypy boolean settings keyed by option name.",
            ),
        ]
        string_settings: Annotated[
            t.StrMapping,
            m.Field(
                alias="string-settings",
                description=(
                    "Mypy string-valued settings keyed by option name "
                    "(e.g. follow_imports='normal')."
                ),
            ),
        ] = m.Field(default_factory=ImmutableEmptyMapping)
        overrides: Annotated[
            tuple[FlextInfraModelsDepsToolConfigLinters.MypyOverrideConfig, ...],
            m.Field(
                description="Per-module mypy overrides for auto-generated files and PEP 695 generics."
            ),
        ] = m.Field(
            default_factory=tuple,
            description="Per-module mypy overrides for auto-generated files and PEP 695 generics.",
        )

    class PydanticMypyConfig(m.ArbitraryTypesModel):
        """Pydantic mypy plugin settings loaded from YAML."""

        init_forbid_extra: Annotated[
            bool,
            m.Field(
                description="Enable forbid-extra init behavior in pydantic mypy plugin."
            ),
        ]
        init_typed: Annotated[
            bool,
            m.Field(
                description="Enable typed __init__ signatures in pydantic mypy plugin."
            ),
        ]
        warn_required_dynamic_aliases: Annotated[
            bool,
            m.Field(
                description="Warn on required dynamic aliases in pydantic mypy plugin."
            ),
        ]
        warn_untyped_fields: Annotated[
            bool,
            m.Field(
                alias="warn-untyped-fields",
                description=(
                    "Warn when Pydantic model fields are inferred as Any instead of "
                    "explicitly typed. Aligns with AGENTS.md:279 'no Any allowed'."
                ),
            ),
        ] = False


__all__: list[str] = ["FlextInfraModelsDepsToolConfigLinters"]
