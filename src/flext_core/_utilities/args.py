"""Utilities module - FlextUtilitiesArgs.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import Enum, StrEnum
from functools import wraps
from types import UnionType
from typing import (
    Annotated,
    ParamSpec,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import ConfigDict, TypeAdapter, ValidationError, validate_call

from flext_core import p, r, t

_ValidatedParams = ParamSpec("_ValidatedParams")
_ValidatedReturn = TypeVar("_ValidatedReturn")


class FlextUtilitiesArgs:
    """Utilities for automatic args/kwargs parsing.

    PHILOSOPHY:
    ──────────
    - Parse once, use everywhere
    - Decorators that eliminate manual validation
    - Integration with inspect.signature for introspection
    - ParamSpec (PEP 612) for correct decorator typing

    References:
    ────────────
    - PEP 612: https://peps.python.org/pep-0612/
    - inspect.signature: https://docs.python.org/3/library/inspect.html
    - validate_call: https://docs.pydantic.dev/latest/concepts/validation_decorator/

    """

    _enum_type_adapter: TypeAdapter[type[StrEnum]] = TypeAdapter(type[StrEnum])

    @staticmethod
    def _validate_enum_type(candidate: type[Enum] | str) -> type[StrEnum] | None:
        """Validate that candidate is a StrEnum subclass."""
        try:
            return FlextUtilitiesArgs._enum_type_adapter.validate_python(candidate)
        except ValidationError:
            return None

    # ─────────────────────────────────────────────────────────────
    # METHOD 1: @validate_call from Pydantic (recommended)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def validated(
        func: Callable[_ValidatedParams, _ValidatedReturn],
    ) -> Callable[_ValidatedParams, _ValidatedReturn]:
        """Decorator that uses @validate_call from Pydantic internally.

        ADVANTAGE:
        - Zero validation code in method
        - Pydantic handles ALL conversion and validation
        - Automatic friendly errors
        - Works with StrEnum, Pydantic models, etc.

        BEFORE:
             def process(self, status: str) -> bool:
                 if status not in Status._value2member_map_:
                     raise ValueError(...)
                 status = Status(status)
                 ...

        AFTER:
             @FlextUtilitiesArgs.validated
             def process(self, status: Status) -> bool:
                 # status is already Status, validated automatically!
                 ...

        HOW IT WORKS:
        - Annotate parameters with StrEnum → accepts string OR enum
        - Pydantic converts automatically
        - Validation error → ValidationError (can be caught)
        """
        return validate_call(
            config=ConfigDict(
                arbitrary_types_allowed=True,
                use_enum_values=False,  # Keep enum, don't convert to string
            ),
            validate_return=False,
        )(func)

    @staticmethod
    def validated_with_result[V, **P](
        func: Callable[P, p.Result[V]],
    ) -> Callable[P, p.Result[V]]:
        """Decorator that converts ValidationError to r.fail().

        USE WHEN:
        - Method returns r
        - Want validation errors to become r.fail()
        - Don't want exceptions leaking

        Example:
             @FlextUtilitiesArgs.validated_with_result
             def process(self, status: Status) -> "r[bool]":
                 # If status invalid → returns r.fail()
                 # If status valid → executes normally
                 return r.ok(True)

        """
        validated_func = validate_call(
            config=ConfigDict(
                arbitrary_types_allowed=True,
                use_enum_values=False,
            ),
            validate_return=False,
        )(func)

        @wraps(func)
        def wrapper(
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> p.Result[V]:
            try:
                # Type safe call via Pydantic validated_func
                return validated_func(*args, **kwargs)  # type: ignore[no-any-return]
            except (ValidationError, TypeError, ValueError) as e:
                # Return failed result with error message
                return r[V].fail(str(e))

        # wrapper has correct type via @wraps preserving signature
        return wrapper

    # ─────────────────────────────────────────────────────────────
    # METHOD 2: Parse kwargs to typed dict
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def parse_kwargs[E: StrEnum](
        kwargs: Mapping[str, t.GuardInputValue],
        enum_fields: Mapping[str, type[E]],
    ) -> p.Result[Mapping[str, t.GuardInputValue]]:
        """Parse kwargs converting specific fields to StrEnums.

        Example:
             result = FlextUtilitiesArgs.parse_kwargs(
                 kwargs={"status": "active", "name": "John"},
                 enum_fields={"status": Status},
             )
             if result.is_success:
                 # result.value = {"status": Status.ACTIVE, "name": "John"}

        """
        # Convert Mapping to dict for mutability
        parsed = dict(kwargs)
        errors: list[str] = []

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
            return r[Mapping[str, t.GuardInputValue]].fail(
                f"Invalid values: {'; '.join(errors)}",
            )
        return r[Mapping[str, t.GuardInputValue]].ok(parsed)

    # ─────────────────────────────────────────────────────────────
    # METHOD 3: Signature introspection for auto-parsing
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def get_enum_params(
        func: p.CallableWithHints,
    ) -> Mapping[str, type[StrEnum]]:
        """Extract parameters that are StrEnum from function signature.

        Example:
             def process(self, status: Status, name: str) -> bool: ...

             params = FlextUtilitiesArgs.get_enum_params(process)
             # params = {"status": Status}

        """
        try:
            hints = get_type_hints(func)
        except (NameError, TypeError, AttributeError):
            return {}

        enum_params: dict[str, type[StrEnum]] = {}

        for name, hint in hints.items():
            if name == "return":
                continue

            # Unwrap Annotated
            current_hint = hint
            origin = get_origin(hint)
            if origin is Annotated:
                current_hint = get_args(hint)[0]
                origin = get_origin(current_hint)

            # Check if it's a StrEnum
            validated_hint = FlextUtilitiesArgs._validate_enum_type(current_hint)
            if validated_hint is not None:
                enum_params[name] = validated_hint

            # Check Union types (str | Status) - Python 3.10+ uses UnionType
            elif origin is UnionType:
                for arg in get_args(current_hint):
                    validated_arg = FlextUtilitiesArgs._validate_enum_type(arg)
                    if validated_arg is not None:
                        enum_params[name] = validated_arg
                        break

        return enum_params


__all__ = [
    "FlextUtilitiesArgs",
]
