"""Utilities module - FlextUtilitiesArgs.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from enum import StrEnum

from flext_core import e, r, t
from flext_core._constants.pydantic import FlextConstantsPydantic
from flext_core._models.pydantic import FlextModelsPydantic


class FlextUtilitiesArgs:
    """Utilities for automatic args/kwargs parsing."""

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
             if result.success:
                 # result.value = {"status": Status.ACTIVE, "name": "John"}

        """
        parsed = dict(kwargs)
        errors: MutableSequence[str] = []
        for field, enum_cls in enum_fields.items():
            if field in parsed:
                value = parsed[field]
                adapter = FlextModelsPydantic.TypeAdapter(enum_cls)
                try:
                    parsed[field] = adapter.validate_python(value)
                except FlextConstantsPydantic.ValidationError:
                    members_dict = getattr(enum_cls, "__members__", {})
                    enum_members = list(members_dict.values())
                    valid = ", ".join(m.value for m in enum_members)
                    errors.append(f"{field}: '{value}' not in [{valid}]")
        if errors:
            return e.fail_validation(
                "kwargs",
                error=f"Invalid values: {'; '.join(errors)}",
            )
        return r[Mapping[str, t.ValueOrModel]].ok(parsed)

    @staticmethod
    def parse_model[M: FlextModelsPydantic.BaseModel](
        kwargs: Mapping[str, t.ValueOrModel],
        model_cls: type[M],
        *,
        allow_empty: bool = True,
    ) -> r[M]:
        """Parse kwargs directly into a Pydantic model with detailed error collection.

        Args:
            kwargs: Dictionary of arguments.
            model_cls: BaseModel subclass to populate.
            allow_empty: If true, empty kwargs will validate empty model instances successfully.

        Returns:
            Result containing hydrated model, or detailed string of failed validation fields.

        """
        if not kwargs and allow_empty:
            kwargs = {}
        try:
            return r[M].ok(model_cls.model_validate(kwargs))
        except (
            FlextConstantsPydantic.ValidationError,
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
        ) as exc:
            return e.fail_validation(error=exc)

    @staticmethod
    def resolve_options[M: FlextModelsPydantic.BaseModel](
        options: M | None,
        kwargs: Mapping[str, t.ValueOrModel],
        model_cls: type[M],
        *,
        allow_empty: bool = True,
    ) -> r[M]:
        """Resolve options from a pre-instantiated model or kwargs concisely.

        Reduces boilerplate by returning an r[M] which callers can unwrap_or()
        or gracefully fail.
        """
        if options is not None:
            return r[M].ok(options)
        return FlextUtilitiesArgs.parse_model(
            kwargs,
            model_cls,
            allow_empty=allow_empty,
        )


__all__: list[str] = ["FlextUtilitiesArgs"]
