"""Base Pydantic models - Foundation for FLEXT ecosystem.

TIER 0: Uses only stdlib, pydantic, and Tier 0 modules (constants, typings).

This module provides the fundamental base classes for all Pydantic models
in the FLEXT ecosystem. All classes are nested inside FlextModelsBase
following the namespace pattern.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

# mro-wkii.17.26 (codex): internal static init must not re-enter the root lazy ABI.
from flext_core.constants import FlextConstants as c
from flext_core._models.pydantic import FlextModelsPydantic as mp
from flext_core._utilities.enforcement import FlextUtilitiesEnforcement as ue

if TYPE_CHECKING:
    from flext_core._typings.base import FlextTypingBase as t


class FlextModelsBase:
    """Container for base model classes - Tier 0, 100% standalone."""

    class EnforcedModel(mp.BaseModel):
        """Base model that enforces architectural rules on subclasses."""

        @classmethod
        @override
        def __pydantic_init_subclass__(cls, **kwargs: t.JsonValue) -> None:
            super().__pydantic_init_subclass__(**kwargs)
            ue.run(cls)

    class ManagedModel(EnforcedModel):
        """Shared preset for assignment validation with forbidden extra fields."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            validate_assignment=True, extra=c.EXTRA_CONFIG_FORBID
        )

    class EnumManagedModel(ManagedModel):
        """Shared preset for managed models that serialize enum values."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(use_enum_values=True)

    class NormalizedModel(EnumManagedModel):
        """Shared preset for managed models with whitespace normalization."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(str_strip_whitespace=True)

    class StrictManagedModel(NormalizedModel):
        """Shared preset for strict managed validation boundaries."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            strict=True, validate_default=True
        )

    class StrictModel(StrictManagedModel):
        """Reusable strict model preset for validated domain boundaries."""

    class FrozenModel(StrictModel):
        """Immutable strict domain model preset."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(frozen=True)

    class ArbitraryTypesModel(ManagedModel):
        """Base model with arbitrary types support."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(
            arbitrary_types_allowed=True
        )

    class StrictBoundaryModel(FrozenModel):
        """Strict boundary model for API/external boundaries."""

    class FlexibleInternalModel(NormalizedModel):
        """Flexible internal model for domain logic."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(extra="ignore")

    class ImmutableValueModel(ManagedModel):
        """Immutable value model for value objects."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(frozen=True)

    class TaggedModel(EnforcedModel):
        """Base pattern for tagged discriminated unions."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(extra="forbid")
        tag: ClassVar[str]

    class FlexibleModel(ArbitraryTypesModel):
        """Model for dynamic configuration - allows extra fields."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(extra="ignore")

    class DynamicModel(FlexibleModel):
        """Dynamic domain model preset with string whitespace normalization."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(str_strip_whitespace=True)

    class FrozenDynamicModel(DynamicModel):
        """Immutable dynamic domain model preset."""

        model_config: ClassVar[mp.ConfigDict] = mp.ConfigDict(frozen=True)


__all__: list[str] = ["FlextModelsBase"]
