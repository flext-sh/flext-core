"""Utilities module - FlextUtilitiesArgs.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from functools import wraps
from typing import Annotated, Protocol, get_args, get_origin, get_type_hints

from pydantic import ConfigDict, validate_call

from flext_core.result import FlextResult
from flext_core.typings import FlextTypes, P, R


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
        func: Callable[P, FlextResult[R]],
    ) -> Callable[P, FlextResult[R]]:
        """Decorator that converts ValidationError to FlextResult.fail().

        USE WHEN:
        - Method returns FlextResult
        - Want validation errors to become FlextResult.fail()
        - Don't want exceptions leaking

        Example:
             @FlextUtilitiesArgs.validated_with_result
             def process(self, status: Status) -> FlextResult[bool]:
                 # If status invalid → returns FlextResult.fail()
                 # If status valid → executes normally
                 return FlextResult.ok(True)

        """
        from flext_core.result import FlextResult

        validated_func = validate_call(
            config=ConfigDict(
                arbitrary_types_allowed=True,
                use_enum_values=False,
            ),
            validate_return=False,
        )(func)

        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> FlextResult[R]:
            try:
                return validated_func(*args, **kwargs)
            except Exception as e:
                return FlextResult.fail(str(e))

        return wrapper

    # ─────────────────────────────────────────────────────────────
    # METHOD 2: Parse kwargs to typed dict
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def parse_kwargs[E: StrEnum](
        kwargs: Mapping[str, FlextTypes.FlexibleValue],
        enum_fields: Mapping[str, type[E]],
    ) -> FlextResult[dict[str, FlextTypes.FlexibleValue]]:
        """Parse kwargs converting specific fields to StrEnums.

        Example:
             result = FlextUtilitiesArgs.parse_kwargs(
                 kwargs={"status": "active", "name": "John"},
                 enum_fields={"status": Status},
             )
             if result.is_success:
                 # result.value = {"status": Status.ACTIVE, "name": "John"}

        """
        from flext_core.result import FlextResult

        parsed = dict(kwargs)
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
            return FlextResult.fail(f"Invalid values: {'; '.join(errors)}")
        return FlextResult.ok(parsed)

    # ─────────────────────────────────────────────────────────────
    # METHOD 3: Signature introspection for auto-parsing
    # ─────────────────────────────────────────────────────────────

    class _CallableWithHints(Protocol):
        """Protocol for callables that support type hints introspection."""

        __annotations__: dict[str, object]

    @staticmethod
    def get_enum_params(
        func: _CallableWithHints,
    ) -> dict[str, type[StrEnum]]:
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
            if isinstance(current_hint, type) and issubclass(current_hint, StrEnum):
                enum_params[name] = current_hint

            # Check Union types (str | Status)
            elif origin is type(str | int):  # UnionType
                for arg in get_args(current_hint):
                    if isinstance(arg, type) and issubclass(arg, StrEnum):
                        enum_params[name] = arg
                        break

        return enum_params
