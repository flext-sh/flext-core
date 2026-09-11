"""CLI option helpers shared through ``u.Cli``."""

from __future__ import annotations

from flext_cli import c, t
from flext_cli.models import m


class FlextCliUtilitiesOptionBuilder:
    """Implementation part for FlextCliUtilitiesOptionBuilder."""

    def __init__(self, field_name: str, registry: t.Cli.OptionRegistry) -> None:
        """Initialize the option builder."""
        super().__init__()
        self.field_name = field_name
        self.registry = registry

    def build(self) -> m.Cli.OptionSpec:
        """Build one CLI option spec from field metadata."""
        field_meta_raw = self.registry.get(self.field_name, {})
        if not field_meta_raw:
            msg = "Option registry metadata must support key lookup"
            raise TypeError(msg)
        field_meta = m.Cli.OptionMetadata.model_validate(field_meta_raw)
        help_text = field_meta.help
        short_flag = field_meta.short
        has_default = c.Cli.CLI_PARAM_KEY_DEFAULT in field_meta_raw

        cli_param_name: str = (
            field_meta.field_name_override
            if field_meta.field_name_override is not None
            else self.field_name
        )

        option_args: t.MutableSequenceOf[str] = [
            f"--{cli_param_name.replace('_', '-')}"
        ]
        if cli_param_name == "project":
            option_args.append("--projects")
        if short_flag:
            option_args.append(f"-{short_flag}")

        return m.Cli.OptionSpec(
            declarations=tuple(option_args),
            help_text=help_text,
            default=field_meta.default if has_default else None,
            required=not has_default,
        )


__all__: list[str] = ["FlextCliUtilitiesOptionBuilder"]
