"""Utilities module - FlextUtilitiesArgs.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableSequence
from enum import StrEnum
from functools import wraps
from inspect import BoundArguments, Parameter, signature
from typing import ClassVar, get_args, get_origin, get_type_hints

from pydantic import TypeAdapter, ValidationError

from flext_core import m, r, t


class FlextUtilitiesArgs:
    """Utilities for automatic args/kwargs parsing."""

    _V: ClassVar[type[m.Validators]] = m.Validators

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
    def _resolve_enum_annotation(annotation: object) -> type[StrEnum] | None:
        if isinstance(annotation, type) and issubclass(annotation, StrEnum):
            return annotation
        origin = get_origin(annotation)
        if origin is None:
            return None
        for arg in get_args(annotation):
            resolved = FlextUtilitiesArgs._resolve_enum_annotation(arg)
            if resolved is not None:
                return resolved
        return None

    @classmethod
    def get_enum_params(
        cls,
        func: Callable[..., object],
    ) -> Mapping[str, type[StrEnum]]:
        """Extract StrEnum-typed parameters from a callable signature."""
        hints_map: Mapping[str, object]
        try:
            hints_map = get_type_hints(func)
        except (AttributeError, NameError, TypeError):
            hints_map = dict[str, object]()
        enum_fields: dict[str, type[StrEnum]] = {}
        for name, param in signature(func).parameters.items():
            if param.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
                continue
            enum_cls = cls._resolve_enum_annotation(
                hints_map.get(name, param.annotation),
            )
            if enum_cls is not None:
                enum_fields[name] = enum_cls
        return enum_fields

    @staticmethod
    def _coerce_bound_arguments(
        bound: BoundArguments,
        enum_fields: Mapping[str, type[StrEnum]],
    ) -> None:
        for field, enum_cls in enum_fields.items():
            if field not in bound.arguments:
                continue
            adapter = TypeAdapter(enum_cls)
            bound.arguments[field] = adapter.validate_python(bound.arguments[field])

    @classmethod
    def validated[**P, R](cls, func: Callable[P, R]) -> Callable[P, R]:
        """Decorator that coerces StrEnum parameters before function execution."""
        enum_fields = cls.get_enum_params(func)
        func_signature = signature(func)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not enum_fields:
                return func(*args, **kwargs)
            bound = func_signature.bind(*args, **kwargs)
            cls._coerce_bound_arguments(bound, enum_fields)
            return func(*bound.args, **bound.kwargs)

        return wrapper

    @classmethod
    def validated_with_result[**P, R](
        cls,
        func: Callable[P, R | r[R]],
    ) -> Callable[P, r[R]]:
        """Decorator that coerces StrEnums and converts failures into `r.fail()`."""
        enum_fields = cls.get_enum_params(func)
        func_signature = signature(func)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> r[R]:
            try:
                if enum_fields:
                    bound = func_signature.bind(*args, **kwargs)
                    cls._coerce_bound_arguments(bound, enum_fields)
                    result = func(*bound.args, **bound.kwargs)
                else:
                    result = func(*args, **kwargs)
            except ValidationError as exc:
                return r[R].fail(f"Validation failed: {exc}")
            except (
                ValueError,
                TypeError,
                KeyError,
                AttributeError,
                RuntimeError,
            ) as exc:
                return r[R].fail(str(exc))
            if isinstance(result, r):
                return result
            return r[R].ok(result)

        return wrapper


__all__ = ["FlextUtilitiesArgs"]
