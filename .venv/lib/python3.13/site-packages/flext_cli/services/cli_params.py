"""FLEXT CLI Common Parameters - Auto-generated from FlextSettings properties."""

from __future__ import annotations

from flext_cli import c, m, p, r, s, t, u


class FlextCliCommonParams(s):
    """Common CLI parameters auto-generated from FlextSettings field metadata.

    Business Rules:
    ───────────────
    1. Parameters MUST be auto-generated from FlextCliSettings Pydantic fields
    2. Parameters MUST be enabled by default (cannot be disabled without error)
    3. Parameter types MUST match FlextCliSettings field types
    4. Choices MUST be validated against allowed values
    5. Boolean parameters automatically become Typer flags
    """

    @classmethod
    def apply_to_config(
        cls,
        settings: p.Cli.Settings,
        params: p.Cli.CliParamsConfig | None = None,
        **kwargs: t.Cli.CliParamValue,
    ) -> p.Result[p.Cli.Settings]:
        """Apply CLI parameter values to FlextSettings using Pydantic validation.

        Business Rule: Applies CLI parameter values with Pydantic validation.
        Trace mode requires debug mode to be enabled.
        """
        try:
            params_to_use = u.Cli.params_resolve(params, kwargs)
            return u.Cli.params_apply(settings, params_to_use)
        except c.Cli.CLI_SAFE_EXCEPTIONS as exc:
            return r[p.Cli.Settings].fail(
                c.Cli.CLI_PARAM_ERR_APPLY_FAILED_FMT.format(error=exc)
            )

    @classmethod
    def create_option(cls, field_name: str) -> m.Cli.OptionSpec:
        """Create one validated framework-neutral option model."""
        if field_name not in c.Cli.CLI_PARAM_REGISTRY:
            msg = c.Cli.CLI_PARAM_ERR_FIELD_NOT_FOUND_FMT.format(field_name=field_name)
            raise ValueError(msg)
        option = u.Cli.build_option(field_name, c.Cli.CLI_PARAM_REGISTRY)
        if not isinstance(option, m.Cli.OptionSpec):
            msg = "CLI option builder returned an unexpected contract"
            raise TypeError(msg)
        return option


__all__: t.MutableSequenceOf[str] = ["FlextCliCommonParams"]
