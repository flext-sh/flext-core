"""Runtime enforcement utilities for Pydantic v2 governance.

Static methods called from __pydantic_init_subclass__ hooks on FLEXT
base model classes. Constants come from c.ENFORCEMENT_* via MRO.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
import warnings
from collections.abc import Mapping, MutableSequence, Sequence
from typing import get_args, get_origin

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
    def check_no_any(
        own: Mapping[str, FieldInfo],
    ) -> Sequence[str]:
        """Reject Any in field annotations — use t.* contracts."""
        errors: MutableSequence[str] = []
        for name, info in own.items():
            annotation = info.annotation
            if annotation is typing.Any:
                errors.append(
                    f'Field "{name}": Any is FORBIDDEN. Use a t.* type contract.',
                )
                continue
            for arg in get_args(annotation):
                if arg is typing.Any:
                    errors.append(
                        f'Field "{name}": Any in type args FORBIDDEN. Use t.* contracts.',
                    )
                    break
        return errors

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
    def check_no_object(
        own: Mapping[str, FieldInfo],
    ) -> Sequence[str]:
        """Reject object as field annotation — use specific t.* contract."""
        errors: MutableSequence[str] = []
        for name, info in own.items():
            if info.annotation is object:
                errors.append(
                    f'Field "{name}": object is FORBIDDEN. Use a t.* type contract.',
                )
        return errors

    @staticmethod
    def check_no_str_none_with_empty_default(
        own: Mapping[str, FieldInfo],
    ) -> Sequence[str]:
        """Reject str | None with default='' — use str with default='' instead."""
        errors: MutableSequence[str] = []
        for name, info in own.items():
            origin = get_origin(info.annotation)
            if origin is not None:
                continue
            args = get_args(info.annotation)
            if not args:
                continue
            has_str = str in args
            has_none = type(None) in args
            if (
                has_str
                and has_none
                and isinstance(info.default, str)
                and not info.default
            ):
                errors.append(
                    f'Field "{name}": str | None with default="" is wrong. '
                    f'Use str with default="" (None has no business meaning here).',
                )
        return errors

    @staticmethod
    def check_frozen_value_objects(model_type: type[BaseModel]) -> Sequence[str]:
        """Value objects (ImmutableValueModel/FrozenValueModel) must be frozen."""
        value_bases = {"ImmutableValueModel", "FrozenValueModel", "FrozenStrictModel"}
        is_value = False
        for base in model_type.__mro__:
            if base.__name__ in value_bases:
                is_value = True
                break
        if not is_value:
            return []
        if not model_type.model_config.get("frozen", False):
            return [
                (
                    "Value objects must be frozen=True. "
                    "Inherit from ImmutableValueModel or FrozenValueModel."
                ),
            ]
        return []

    @staticmethod
    def check_no_mutable_field_defaults(
        own: Mapping[str, FieldInfo],
    ) -> Sequence[str]:
        """Reject mutable defaults ([], {}, set()) — use Field(default_factory=...)."""
        errors: MutableSequence[str] = []
        for name, info in own.items():
            default = info.default
            if isinstance(default, (list, dict, set)) and len(default) == 0:
                type_name = type(default).__name__
                errors.append(
                    f'Field "{name}": mutable default {type_name}() is FORBIDDEN. '
                    f"Use Field(default_factory={type_name}).",
                )
        return errors

    @staticmethod
    def check_no_inline_union_types(
        own: Mapping[str, FieldInfo],
    ) -> Sequence[str]:
        """Flag complex inline union types that should be t.* aliases."""
        errors: MutableSequence[str] = []
        for name, info in own.items():
            args = get_args(info.annotation)
            # Count non-None union members
            real_args = [a for a in args if a is not type(None)]
            max_inline_union = 2
            if len(real_args) > max_inline_union:
                errors.append(
                    f'Field "{name}": complex inline union with {len(real_args)}+ types. '
                    f"Centralize as a t.* type alias in typings.py.",
                )
        return errors

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

        # HARD checks — always TypeError in strict
        hard_errors: MutableSequence[str] = []
        hard_errors.extend(
            FlextUtilitiesEnforcement.check_no_any(fields),
        )
        hard_errors.extend(
            FlextUtilitiesEnforcement.check_no_object(fields),
        )
        hard_errors.extend(
            FlextUtilitiesEnforcement.check_no_bare_collections(fields),
        )
        hard_errors.extend(
            FlextUtilitiesEnforcement.check_no_v1_patterns(model_type),
        )
        hard_errors.extend(
            FlextUtilitiesEnforcement.check_no_mutable_field_defaults(fields),
        )

        if hard_errors:
            msg = (
                f"\n{model_type.__qualname__} violates Pydantic v2 HARD rules:\n"
                + "\n".join(f"  - {e}" for e in hard_errors)
                + "\n\nFix: See AGENTS.md § Code Style > Module Design."
                + "\nExempt: _flext_enforcement_exempt: ClassVar[bool] = True"
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            if mode == "strict":
                raise TypeError(msg)

        # CONFIGURABLE checks
        config_errors: MutableSequence[str] = []
        config_errors.extend(
            FlextUtilitiesEnforcement.check_field_descriptions(fields),
        )
        config_errors.extend(
            FlextUtilitiesEnforcement.check_extra_policy(model_type),
        )
        config_errors.extend(
            FlextUtilitiesEnforcement.check_frozen_value_objects(model_type),
        )
        config_errors.extend(
            FlextUtilitiesEnforcement.check_no_str_none_with_empty_default(fields),
        )
        config_errors.extend(
            FlextUtilitiesEnforcement.check_no_inline_union_types(fields),
        )

        if config_errors:
            detail = (
                f"\n{model_type.__qualname__} violates Pydantic v2 best practices:\n"
                + "\n".join(f"  - {e}" for e in config_errors)
                + "\n\nFix: See AGENTS.md § Code Style > Module Design."
            )
            warnings.warn(detail, UserWarning, stacklevel=3)
            if mode == "strict":
                raise TypeError(detail)
