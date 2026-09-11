"""CLI option helpers shared through ``u.Cli``."""

from __future__ import annotations

from collections.abc import Mapping

from flext_cli import c, t
from flext_cli.models import m
from flext_cli._utilities._options_parts.flextcliutilitiesoptionbuilder_part_01 import (
    FlextCliUtilitiesOptionBuilder,
)
from flext_cli._utilities._options_parts.flextcliutilitiesoptions_part_01 import (
    FlextCliUtilitiesOptions as FlextCliUtilitiesOptionsPart01,
)


class FlextCliUtilitiesOptions(FlextCliUtilitiesOptionsPart01):
    """Implementation part for FlextCliUtilitiesOptions."""

    @classmethod
    def field_default(
        cls, field_name: str, field_info: m.FieldInfo, settings: t.Cli.ModelLike | None
    ) -> t.Cli.CliValue | None:
        """Resolve CLI default from settings first, then from model field metadata."""
        default_factory = getattr(field_info, "default_factory", None)
        source_value = (
            getattr(settings, field_name)
            if settings is not None and hasattr(settings, field_name)
            else default_factory()
            if callable(default_factory)
            else getattr(field_info, "default", None)
        )
        try:
            normalized_source = t.Cli.CLI_DEFAULT_SOURCE_ADAPTER.validate_python(
                source_value
            )
        except c.EXC_VALIDATION_TYPE_VALUE:
            normalized_source = None
        if normalized_source is None:
            return None
        match normalized_source:
            case _ if (
                normalized_atom := cls.normalize_cli_atom(normalized_source)
            ) is not None:
                normalized_default: t.Cli.CliValue | None = normalized_atom
            case Mapping() as normalized_source_mapping:
                normalized_mapping: t.Cli.MutableDefaultMapping = {}
                for key, item_value in normalized_source_mapping.items():
                    normalized_item = cls.normalize_cli_atom(item_value)
                    if normalized_item is not None:
                        normalized_mapping[key] = normalized_item
                normalized_default = normalized_mapping or None
            case _ if cls.is_string_sequence(normalized_source):
                normalized_default = t.Cli.STR_SEQUENCE_ADAPTER.validate_python(
                    normalized_source
                )
            case _:
                normalized_default = None
        return normalized_default

    @staticmethod
    def build_option(
        field_name: str, registry: t.Cli.OptionRegistry
    ) -> m.Cli.OptionSpec:
        """Build one CLI option spec from the canonical registry."""
        return FlextCliUtilitiesOptionBuilder(field_name, registry).build()

    @staticmethod
    def reorder_prefixed_options(
        args: t.StrSequence,
        *,
        bool_options: t.StrSequence,
        value_options: t.StrSequence,
    ) -> list[str]:
        """Move shared options before subcommand to right after the subcommand."""
        if not args:
            return []
        bool_set = set(bool_options)
        value_set = set(value_options)
        prefix_tokens: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            normalized = token.split("=", 1)[0]
            if normalized in bool_set:
                prefix_tokens.append(token)
                index += 1
                continue
            if normalized in value_set:
                prefix_tokens.append(token)
                if "=" not in token and index + 1 < len(args):
                    prefix_tokens.append(args[index + 1])
                    index += 2
                else:
                    index += 1
                continue
            if token.startswith("-"):
                break
            subcommand = token
            suffix_tokens = list(args[index + 1 :])
            return [subcommand, *prefix_tokens, *suffix_tokens]
        return list(args)


__all__: list[str] = ["FlextCliUtilitiesOptions"]
