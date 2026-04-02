"""Utilities module - FlextUtilitiesArgs.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence
from enum import StrEnum
from typing import ClassVar

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


__all__ = ["FlextUtilitiesArgs"]
