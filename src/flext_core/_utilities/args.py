"""Utilities module - FlextUtilitiesArgs.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from functools import wraps
from types import UnionType
from typing import (
    Annotated,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import ConfigDict, validate_call

from flext_core.protocols import p
from flext_core.result import r
from flext_core.runtime import FlextRuntime
from flext_core.typings import P, R, t

# TypeVar for validated_with_result: constrained to r or RuntimeResult
_ValidatedResultT = TypeVar(
    "_ValidatedResultT",
    r[t.GeneralValueType],
    FlextRuntime.RuntimeResult[t.GeneralValueType],
)


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

    # ─────────────────────────────────────────────────────────────
    # METHOD 1: @validate_call from Pydantic (recommended)
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def validated(func: Callable[P, R]) -> Callable[P, R]:
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
    def validated_with_result(
        func: Callable[P, _ValidatedResultT],
    ) -> Callable[P, _ValidatedResultT]:
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
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> _ValidatedResultT:
            try:
                # Type narrowing: validated_func returns _ValidatedResultT
                result: _ValidatedResultT = validated_func(*args, **kwargs)
                return result
            except Exception as e:
                # Return fail result - type annotation ensures correct type
                fail_result: r[t.GeneralValueType] = r[t.GeneralValueType].fail(str(e))
                return fail_result

        # wrapper has correct type via @wraps preserving signature
        return wrapper

    # ─────────────────────────────────────────────────────────────
    # METHOD 2: Parse kwargs to typed dict
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def parse_kwargs[E: StrEnum](
        kwargs: Mapping[str, t.FlexibleValue],
        enum_fields: Mapping[str, type[E]],
    ) -> r[dict[str, t.FlexibleValue]]:
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
        parsed: dict[str, t.FlexibleValue] = dict(kwargs)
        errors: list[str] = []

        for field, enum_cls in enum_fields.items():
            if field in parsed:
                value = parsed[field]
                if isinstance(value, str):
                    try:
                        parsed[field] = enum_cls(value)
                    except ValueError:
                        members_dict = getattr(enum_cls, "__members__", {})
                        enum_members = list(members_dict.values())
                        valid = ", ".join(m.value for m in enum_members)
                        errors.append(f"{field}: '{value}' not in [{valid}]")

        if errors:
            return r[dict[str, t.FlexibleValue]].fail(
                f"Invalid values: {'; '.join(errors)}",
            )
        return r[dict[str, t.FlexibleValue]].ok(parsed)

    # ─────────────────────────────────────────────────────────────
    # METHOD 3: Signature introspection for auto-parsing
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def get_enum_params(
        func: p.CallableWithHints,
    ) -> t.StringStrEnumTypeDict:
        """Extract parameters that are StrEnum from function signature.

        Example:
             def process(self, status: Status, name: str) -> bool: ...

             params = FlextUtilitiesArgs.get_enum_params(process)
             # params = {"status": Status}

        """
        try:
            hints = get_type_hints(func)
        except Exception:
            return {}

        enum_params: t.StringStrEnumTypeDict = {}

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
            if isinstance(current_hint, type) and issubclass(current_hint, StrEnum):
                enum_params[name] = current_hint

            # Check Union types (str | Status) - Python 3.10+ uses UnionType
            elif origin is UnionType:
                for arg in get_args(current_hint):
                    if isinstance(arg, type) and issubclass(arg, StrEnum):
                        enum_params[name] = arg
                        break

        return enum_params


__all__ = [
    "FlextUtilitiesArgs",
]
