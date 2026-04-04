"""Runtime enforcement utilities for Pydantic v2 governance.

Static methods called from __pydantic_init_subclass__ hooks on FLEXT
base model classes. Constants come from c.ENFORCEMENT_* via MRO.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, MutableSequence, Sequence
from typing import get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from flext_core import c


class FlextUtilitiesEnforcement:
    """Pydantic v2 enforcement check utilities.

    All methods are static — called from __pydantic_init_subclass__
    on FLEXT base models. Uses c.ENFORCEMENT_* constants exclusively.
    """

    @staticmethod
    def is_exempt(model_type: type) -> bool:
        """Check if a class is exempt from enforcement."""
        if getattr(model_type, "_flext_enforcement_exempt", False):
            return True
        if model_type.__name__ in c.ENFORCEMENT_INFRASTRUCTURE_BASES:
            return True
        module = getattr(model_type, "__module__", "") or ""
        return any(frag in module for frag in c.ENFORCEMENT_EXEMPT_MODULE_FRAGMENTS)

    @staticmethod
    def own_fields(model_type: type[BaseModel]) -> Mapping[str, FieldInfo]:
        """Get fields defined directly on this class (not inherited)."""
        own_annotations = set(vars(model_type).get("__annotations__", {}))
        return {
            name: info
            for name, info in model_type.model_fields.items()
            if name in own_annotations
        }

    @staticmethod
    def check_no_bare_collections(
        own: Mapping[str, FieldInfo],
    ) -> Sequence[str]:
        """Reject dict/list/set as field annotation origins."""
        replacements: Mapping[str, str] = dict(
            c.ENFORCEMENT_COLLECTION_REPLACEMENTS,
        )
        errors: MutableSequence[str] = []
        for name, info in own.items():
            origin = get_origin(info.annotation)
            if (
                origin is not None
                and origin.__name__ in c.ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS
            ):
                fix = replacements.get(origin.__name__, str(origin))
                errors.append(
                    f'Field "{name}": bare {origin.__name__}[...] FORBIDDEN. '
                    f"Use {fix}.",
                )
        return errors

    @staticmethod
    def check_no_v1_patterns(model_type: type[BaseModel]) -> Sequence[str]:
        """Reject class Config (v1 style)."""
        config_attr = model_type.__dict__.get("Config")
        if config_attr is not None and isinstance(config_attr, type):
            return [
                (
                    "class Config is Pydantic v1. "
                    "Use model_config: ClassVar[ConfigDict] = ConfigDict(...)."
                ),
            ]
        return []

    @staticmethod
    def check_field_descriptions(
        own: Mapping[str, FieldInfo],
    ) -> Sequence[str]:
        """Require Field(description=...) on all public fields."""
        errors: MutableSequence[str] = []
        for name, info in own.items():
            if name.startswith("_"):
                continue
            if not info.description:
                errors.append(
                    f'Field "{name}": Field() must include description="...".',
                )
        return errors

    @staticmethod
    def check_extra_policy(model_type: type[BaseModel]) -> Sequence[str]:
        """Require extra='forbid' unless inheriting from relaxed base."""
        for base in model_type.__mro__:
            if base.__name__ in c.ENFORCEMENT_RELAXED_EXTRA_BASES:
                return []
        extra = model_type.model_config.get("extra")
        if extra is None:
            return [
                (
                    'model_config must set extra="forbid" or inherit from '
                    "a configured FLEXT base (ArbitraryTypesModel, etc.)."
                ),
            ]
        if extra != "forbid" and "extra" in model_type.__dict__.get("model_config", {}):
            return [
                (
                    f'model_config extra="{extra}" not allowed. '
                    f"Use FlexibleModel or FlexibleInternalModel."
                ),
            ]
        return []

    @staticmethod
    def run(model_type: type[BaseModel]) -> None:
        """Orchestrate all enforcement checks on a model class.

        Called from __pydantic_init_subclass__ on FLEXT base models.
        """
        mode = c.ENFORCEMENT_MODE
        if mode == "off":
            return
        if FlextUtilitiesEnforcement.is_exempt(model_type):
            return

        fields = FlextUtilitiesEnforcement.own_fields(model_type)

        # HARD checks — always TypeError
        hard_errors: MutableSequence[str] = []
        hard_errors.extend(
            FlextUtilitiesEnforcement.check_no_bare_collections(fields),
        )
        hard_errors.extend(
            FlextUtilitiesEnforcement.check_no_v1_patterns(model_type),
        )

        if hard_errors:
            msg = (
                f"\n{model_type.__qualname__} violates Pydantic v2 HARD rules:\n"
                + "\n".join(f"  - {e}" for e in hard_errors)
                + "\n\nFix: See AGENTS.md § Code Style > Module Design."
                + "\nExempt: _flext_enforcement_exempt: ClassVar[bool] = True"
            )
            if mode == "strict":
                raise TypeError(msg)
            warnings.warn(msg, UserWarning, stacklevel=3)

        # CONFIGURABLE checks
        config_errors: MutableSequence[str] = []
        config_errors.extend(
            FlextUtilitiesEnforcement.check_field_descriptions(fields),
        )
        config_errors.extend(
            FlextUtilitiesEnforcement.check_extra_policy(model_type),
        )

        if config_errors:
            detail = (
                f"\n{model_type.__qualname__} violates Pydantic v2 best practices:\n"
                + "\n".join(f"  - {e}" for e in config_errors)
                + "\n\nFix: See AGENTS.md § Code Style > Module Design."
            )
            if mode == "strict":
                raise TypeError(detail)
            warnings.warn(detail, UserWarning, stacklevel=3)
