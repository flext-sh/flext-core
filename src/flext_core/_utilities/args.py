"""Utilities module - FlextUtilitiesArgs.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, MutableSequence
from enum import Enum, StrEnum
from functools import wraps
from types import UnionType
from typing import (
    Annotated,
    ClassVar,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import ConfigDict, TypeAdapter, ValidationError, validate_call
from pydantic.errors import PydanticSchemaGenerationError

from flext_core import P, R, m, r, t


class FlextUtilitiesArgs:
    """Utilities for automatic args/kwargs parsing."""

    _V: ClassVar[type[m.Validators]] = m.Validators

    @staticmethod
    def _validate_enum_type(
        candidate: t.MessageTypeSpecifier,
    ) -> r[type[StrEnum]]:
        """Validate that candidate is a StrEnum subclass."""
        try:
            return r[type[StrEnum]].ok(
                FlextUtilitiesArgs._V.enum_type_adapter().validate_python(candidate),
            )
        except ValidationError:
            return r[type[StrEnum]].fail("Candidate is not a valid StrEnum type")

    @staticmethod
    def get_enum_params(func: Callable[..., R]) -> Mapping[str, type[StrEnum]]:
        """Extract parameters that are StrEnum from function signature."""
        hints: Mapping[str, t.TypeHintSpecifier]
        try:
            resolved_hints = get_type_hints(func, include_extras=True)
            hints = {str(name): hint for name, hint in resolved_hints.items()}
        except (NameError, TypeError, AttributeError):
            fallback_annotations = getattr(func, "__annotations__", None)
            if isinstance(fallback_annotations, Mapping):
                try:
                    hints = TypeAdapter(
                        Mapping[str, t.TypeHintSpecifier],
                    ).validate_python(
                        fallback_annotations,
                    )
                except (ValidationError, PydanticSchemaGenerationError):
                    return {}
            else:
                return {}
        enum_params: MutableMapping[str, type[StrEnum]] = {}
        for name, hint in hints.items():
            if name == "return":
                continue
            current_hint = hint
            origin = get_origin(current_hint)
            while origin is Annotated:
                current_hint = get_args(current_hint)[0]
                origin = get_origin(current_hint)
            if isinstance(current_hint, str) or (
                isinstance(current_hint, type) and issubclass(current_hint, Enum)
            ):
                validated_hint = FlextUtilitiesArgs._validate_enum_type(current_hint)
            else:
                validated_hint = r[type[StrEnum]].fail(
                    "Candidate is not a valid StrEnum type",
                )
            if validated_hint.is_success:
                enum_params[name] = validated_hint.value
            elif origin is UnionType:
                for arg in get_args(current_hint):
                    current_arg = arg
                    arg_origin = get_origin(current_arg)
                    while arg_origin is Annotated:
                        current_arg = get_args(current_arg)[0]
                        arg_origin = get_origin(current_arg)
                    if isinstance(current_arg, str) or (
                        isinstance(current_arg, type) and issubclass(current_arg, Enum)
                    ):
                        validated_arg = FlextUtilitiesArgs._validate_enum_type(
                            current_arg,
                        )
                    else:
                        validated_arg = r[type[StrEnum]].fail(
                            "Candidate is not a valid StrEnum type",
                        )
                    if validated_arg.is_success:
                        enum_params[name] = validated_arg.value
                        break
        return enum_params

    @staticmethod
    def parse_kwargs[E: StrEnum](
        kwargs: Mapping[str, t.ValueOrModel],
        enum_fields: Mapping[str, type[E]],
    ) -> r[Mapping[str, t.ValueOrModel]]:
        """Parse kwargs converting specific fields to StrEnums.

        Example:
             result = FlextUtilitiesArgs.parse_kwargs(
                 kwargs={"status": "active", "name": "John"},
                 enum_fields={"status": Status},
             )
             if result.is_success:
                 # result.value = {"status": Status.ACTIVE, "name": "John"}

        """
        parsed = dict(kwargs)
        errors: MutableSequence[str] = []
        for field, enum_cls in enum_fields.items():
            if field in parsed:
                value = parsed[field]
                adapter = TypeAdapter(enum_cls)
                try:
                    parsed[field] = adapter.validate_python(value)
                except ValidationError:
                    members_dict = getattr(enum_cls, "__members__", {})
                    enum_members = list(members_dict.values())
                    valid = ", ".join(m.value for m in enum_members)
                    errors.append(f"{field}: '{value}' not in [{valid}]")
        if errors:
            return r[Mapping[str, t.ValueOrModel]].fail(
                f"Invalid values: {'; '.join(errors)}",
            )
        return r[Mapping[str, t.ValueOrModel]].ok(parsed)

    @staticmethod
    def validated(func: Callable[P, R]) -> Callable[P, R]:
        """Decorator that uses @validate_call from Pydantic internally."""
        return validate_call(
            config=ConfigDict(arbitrary_types_allowed=True, use_enum_values=False),
            validate_return=False,
        )(func)

    @staticmethod
    def validated_with_result[V, **P](
        func: Callable[P, r[V]],
    ) -> Callable[P, r[V]]:
        """Decorator that converts ValidationError to ``r.fail()``."""
        validated_func: Callable[P, r[V]] = validate_call(
            config=ConfigDict(arbitrary_types_allowed=True, use_enum_values=False),
            validate_return=False,
        )(func)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> r[V]:
            try:
                return validated_func(*args, **kwargs)
            except (ValidationError, TypeError, ValueError) as e:
                return r[V].fail(str(e))

        return wrapper


__all__ = ["FlextUtilitiesArgs"]
