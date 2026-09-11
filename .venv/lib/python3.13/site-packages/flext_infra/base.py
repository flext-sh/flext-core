"""Shared service foundation for flext-infra command services."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Self, override

from flext_core import s
from flext_infra import c, m, p, settings, t, u
from flext_infra._base_payload import FlextInfraCommandPayloadMixin
from flext_infra._utilities.base import FlextInfraUtilitiesBase as ub

type _InfraResultValue = t.Cli.ResultValue


class FlextInfraServiceBase[TDomainResult: _InfraResultValue](
    s[TDomainResult], FlextInfraCommandPayloadMixin
):
    """Domain command context shared by all flext-infra CLI services.

    Provides settings/bootstrap and normalized fields for workspace location,
    apply/dry-run toggles, output formatting, and project filtering.
    """

    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
        validate_by_name=True, validate_by_alias=True
    )

    @classmethod
    def _runtime_bootstrap_options(cls) -> p.RuntimeBootstrapOptions:
        """Bootstrap service runtime using the shared CLI settings namespace."""
        # flext-j47u: configure the inherited runtime once; no settings proxy/property.
        return m.RuntimeBootstrapOptions(settings_type=type(settings))

    repository_root: Annotated[
        Path,
        m.BeforeValidator(
            lambda v: ub.resolve_repository_root_or_cwd(
                v if isinstance(v, Path) else Path(v)
            )
        ),
    ] = m.Field(
        default_factory=ub.resolve_repository_root_or_cwd,
        alias="workspace",
        description="Repository root",
    )
    apply_changes: bool = m.Field(
        default=False,
        alias="apply",
        description="Apply changes",
        json_schema_extra={"typer_param_decls": list(c.Infra.CLI_APPLY_OPTION_DECLS)},
    )
    check_only: bool = m.Field(default=False, alias="check", description="Check mode")
    dry_run: Annotated[bool, m.Field(description="Dry-run mode")] = False
    output_format: Annotated[
        str,
        m.Field(description="Output format (json|text)"),
        m.BeforeValidator(lambda v: v.strip().lower()),
    ] = "text"
    project_filter: Annotated[
        str | None,
        m.Field(description="Project filter (comma-separated)", exclude=True),
    ] = None
    target_module: str | None = m.Field(
        default=None,
        alias="module",
        description=(
            "Dotted module path to scope the verb to a single module "
            "(e.g. flext_core.result). Composes with --workspace/--projects."
        ),
    )
    target_namespace: str | None = m.Field(
        default=None,
        alias="namespace",
        description=(
            "Alias namespace (c|m|p|t|u|r|e|h|s|x[.<Domain>]) to scope the "
            "verb to a single facade slot."
        ),
    )
    report_path: Annotated[
        Path | None,
        m.Field(description="Report output path", exclude=True),
        m.BeforeValidator(ub.normalize_optional_path),
    ] = None
    output_dir: Annotated[
        Path | None, m.Field(description="Output directory", exclude=True)
    ] = None

    @m.field_validator("project_filter", mode="before")
    @classmethod
    def _normalize_project_filter(cls, value: str | t.StrSequence | None) -> str | None:
        """Normalize project filters into a compact comma-separated string."""
        if value is None:
            return None
        normalized_values = (
            ub.normalize_cli_values(value)
            if isinstance(value, str)
            else ub.normalize_cli_values(*value)
        )
        return ",".join(normalized_values) or None

    @m.field_validator("output_dir", mode="before")
    @classmethod
    def _normalize_output_dir(cls, value: str | Path | None) -> Path | None:
        """Preserve relative output dirs so callers can scope them under repository roots."""
        if value is None:
            return None
        path: Path = u.Cli.resolve_optional_path(value, default=Path())
        return path.resolve() if path.is_absolute() else path

    @m.computed_field
    @property
    def root(self) -> Path:
        """Canonical normalized repository root."""
        return self.repository_root

    @property
    def fail_fast(self) -> bool:
        """Stop at the first failure as an invariant, never a CLI choice."""
        return True

    @m.computed_field
    @property
    def effective_dry_run(self) -> bool:
        """Normalized write-mode decision for CLI services."""
        return self.dry_run or self.check_only or (not self.apply_changes)

    @override
    def execute(self) -> p.Result[TDomainResult]:
        """Execute the service contract and return a typed result."""
        raise NotImplementedError

    @property
    def log(self) -> p.Logger:
        """Service logger through the canonical FlextService kernel."""
        return self.logger

    @classmethod
    def execute_command(cls, params: Self) -> p.Result[TDomainResult]:
        """Execute the validated CLI service instance directly."""
        return params.execute()


s = FlextInfraServiceBase

__all__: list[str] = ["FlextInfraServiceBase", "s"]
