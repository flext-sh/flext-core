"""Utilities module - FlextUtilitiesModel.

Extracted from flext_core.utilities for better modularity.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from flext_core.result import FlextResult
from flext_core.typings import FlextTypes


class FlextUtilitiesModel:
    """Utilities for Pydantic model initialization.

    PHILOSOPHY:
    ──────────
    - model_validate() to create from dicts
    - Automatic StrEnum coercion
    - Merge defaults with overrides
    - No initialization code bloat

    References:
    ────────────
    - model_validate: https://docs.pydantic.dev/latest/api/base_model/
    - ConfigDict: https://docs.pydantic.dev/latest/api/config/

    """

    @staticmethod
    def from_dict[M: BaseModel](
        model_cls: type[M],
        data: Mapping[str, FlextTypes.FlexibleValue],
        *,
        strict: bool = False,
    ) -> FlextResult[M]:
        """Create Pydantic model from dict with FlextResult.

        Example:
             result = FlextUtilitiesModel.from_dict(
                 UserModel,
                 {"status": "active", "name": "John"},
             )
             if result.is_success:
                 user: UserModel = result.value

        """
        try:
            instance = model_cls.model_validate(data, strict=strict)
            return FlextResult.ok(instance)
        except Exception as e:
            return FlextResult.fail(f"Model validation failed: {e}")

    @staticmethod
    def from_kwargs[M: BaseModel](
        model_cls: type[M],
        **kwargs: FlextTypes.FlexibleValue,
    ) -> FlextResult[M]:
        """Create Pydantic model from kwargs with FlextResult.

        Example:
             result = FlextUtilitiesModel.from_kwargs(
                 UserModel,
                 status="active",
                 name="John",
             )

        """
        return FlextUtilitiesModel.from_dict(model_cls, kwargs)

    @staticmethod
    def merge_defaults[M: BaseModel](
        model_cls: type[M],
        defaults: Mapping[str, FlextTypes.FlexibleValue],
        overrides: Mapping[str, FlextTypes.FlexibleValue],
    ) -> FlextResult[M]:
        """Merge defaults with overrides and create model.

        Example:
             DEFAULTS = {"status": Status.PENDING, "retries": 3}

             result = FlextUtilitiesModel.merge_defaults(
                 ConfigModel,
                 defaults=DEFAULTS,
                 overrides={"status": "active"},  # Overrides
             )
             # result.value.status = Status.ACTIVE
             # result.value.retries = 3

        """
        merged = {**defaults, **overrides}
        return FlextUtilitiesModel.from_dict(model_cls, merged)

    @staticmethod
    def update[M: BaseModel](
        instance: M,
        **updates: FlextTypes.FlexibleValue,
    ) -> FlextResult[M]:
        """Update existing model with new values.

        Example:
             user = UserModel(status=Status.ACTIVE, name="John")
             result = FlextUtilitiesModel.update(user, status="inactive")
             # result.value = UserModel with status=Status.INACTIVE

        """
        try:
            # Use model_copy with update - modern Pydantic approach
            # This preserves the type M without needing casts or recreating
            updated_instance = instance.model_copy(update=updates)
            return FlextResult.ok(updated_instance)
        except Exception as e:
            return FlextResult.fail(f"Model update failed: {e}")

    @staticmethod
    def to_dict(
        instance: BaseModel,
        *,
        by_alias: bool = False,
        exclude_none: bool = False,
    ) -> dict[str, FlextTypes.FlexibleValue]:
        """Convert model to dict (simple wrapper).

        Example:
             user = UserModel(status=Status.ACTIVE, name="John")
             data = FlextUtilitiesModel.to_dict(user)
             # data = {"status": "active", "name": "John"}

        """
        return instance.model_dump(
            by_alias=by_alias,
            exclude_none=exclude_none,
        )
