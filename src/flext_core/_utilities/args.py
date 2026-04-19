"""Utilities module - FlextUtilitiesArgs.

Extracted from flext_core for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from flext_core import c, e, p, r, t


class FlextUtilitiesArgs:
    """Utilities for model-based option parsing."""

    @staticmethod
    def parse_model[M: p.Model](
        kwargs: Mapping[str, t.ValueOrModel],
        model_cls: p.ModelType[M],
        *,
        allow_empty: bool = True,
    ) -> p.Result[M]:
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
            c.ValidationError,
            ValueError,
            TypeError,
            AttributeError,
            RuntimeError,
        ) as exc:
            return e.fail_validation(error=exc)

    @staticmethod
    def resolve_options[M: p.Model](
        options: M | None,
        kwargs: Mapping[str, t.ValueOrModel],
        model_cls: p.ModelType[M],
        *,
        allow_empty: bool = True,
    ) -> p.Result[M]:
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
