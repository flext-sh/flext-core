"""CLI option helpers shared through ``u.Cli``."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import GenericAlias, NoneType, UnionType
from typing import Annotated, TypeAliasType, get_args, get_origin

from flext_cli import c, t


class FlextCliUtilitiesOptions:
    """Implementation part for FlextCliUtilitiesOptions."""

    @staticmethod
    def resolve_typer_annotation(
        annotation: t.Cli.RuntimeAnnotation,
    ) -> type | GenericAlias:
        """Resolve runtime annotations to concrete types accepted by Typer."""
        annotated_origin = get_origin(Annotated[str, "meta"])
        sequence_origins: frozenset[object] = frozenset(
            filter(
                None,
                [
                    get_origin(Sequence[str]),
                    get_origin(list[str]),
                    # mro-j2yt (codex): Typer repeats canonical tuple model fields.
                    get_origin(tuple[str, ...]),
                    get_origin(t.StrSequence),
                    t.SequenceOf,
                    t.MutableSequenceOf,
                ],
            )
        )
        set_origins: dict[object, type] = {
            o: t_
            for o, t_ in [
                (get_origin(dict[str, t.Scalar]), dict),
                (get_origin(frozenset[str]), frozenset),
                (get_origin(set[str]), set),
            ]
            if o is not None
        }
        resolved_annotation_input = annotation
        origin = get_origin(resolved_annotation_input)
        while (
            isinstance(resolved_annotation_input, TypeAliasType)
            or origin == annotated_origin
        ):
            resolved_annotation_input = (
                resolved_annotation_input.__value__
                if isinstance(resolved_annotation_input, TypeAliasType)
                else get_args(resolved_annotation_input)[0]
            )
            origin = get_origin(resolved_annotation_input)

        if isinstance(resolved_annotation_input, UnionType):
            resolved_args = tuple(
                FlextCliUtilitiesOptions.resolve_typer_annotation(arg)
                for arg in get_args(resolved_annotation_input)
            )
            non_none_args = tuple(arg for arg in resolved_args if arg is not NoneType)
            if (
                len(resolved_args) == c.Cli.OPTIONAL_UNION_ARG_COUNT
                and len(non_none_args) == 1
            ):
                return non_none_args[0]
            return str

        if origin in sequence_origins:
            inner_annotation = next(iter(get_args(resolved_annotation_input)), str)
            resolved_inner = FlextCliUtilitiesOptions.resolve_typer_annotation(
                inner_annotation
            )
            sequence_item = resolved_inner if isinstance(resolved_inner, type) else str
            return GenericAlias(list, (sequence_item,))

        set_annotation = set_origins.get(origin)
        if set_annotation is not None:
            return set_annotation

        return (
            resolved_annotation_input
            if isinstance(resolved_annotation_input, GenericAlias | type)
            else str
        )

    @staticmethod
    def is_string_sequence(value: t.Cli.CliDefaultSource) -> bool:
        """Return True for concrete string sequences accepted by repeated CLI options."""
        if isinstance(value, Path) or not isinstance(value, Sequence):
            return False
        if isinstance(value, str | bytes):
            return False
        return all(isinstance(item, str) for item in value)

    @classmethod
    def normalize_cli_atom(
        cls, value: t.Cli.CliDefaultSource
    ) -> t.Cli.DefaultAtom | None:
        """Normalize one runtime value into an allowed Typer scalar or string sequence."""
        if isinstance(value, c.Cli.CLI_SCALAR_TYPES_TUPLE):
            return value
        if isinstance(value, Path):
            return str(value)
        if cls.is_string_sequence(value):
            normalized_sequence = t.Cli.STR_SEQUENCE_ADAPTER.validate_python(value)
            return tuple(normalized_sequence)
        return None


__all__: list[str] = ["FlextCliUtilitiesOptions"]
