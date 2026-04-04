"""Runtime enforcement utilities for Pydantic v2 governance.

Static methods called from __pydantic_init_subclass__ hooks on FLEXT
base model classes and __init_subclass__ on facade classes.
Constants come from _ec.ENFORCEMENT_* via MRO.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import typing
import warnings
from collections.abc import Mapping, MutableSequence, Sequence
from enum import EnumType
from typing import Protocol, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from flext_core._constants.enforcement import FlextConstantsEnforcement as _ec


class FlextUtilitiesEnforcement:
    """Pydantic v2 enforcement check utilities.

    All methods are static — called from __pydantic_init_subclass__
    on FLEXT base models. Uses _ec.ENFORCEMENT_* constants exclusively.
    """

    @staticmethod
    def is_exempt(model_type: type) -> bool:
        """Check if a class is exempt from enforcement."""
        if getattr(model_type, "_flext_enforcement_exempt", False):
            return True
        if model_type.__name__ in _ec.ENFORCEMENT_INFRASTRUCTURE_BASES:
            return True
        module = getattr(model_type, "__module__", "") or ""
        return any(frag in module for frag in _ec.ENFORCEMENT_EXEMPT_MODULE_FRAGMENTS)

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
            _ec.ENFORCEMENT_COLLECTION_REPLACEMENTS,
        )
        errors: MutableSequence[str] = []
        for name, info in own.items():
            origin = get_origin(info.annotation)
            if (
                origin is not None
                and origin.__name__ in _ec.ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS
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
            if base.__name__ in _ec.ENFORCEMENT_RELAXED_EXTRA_BASES:
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
        mode = _ec.ENFORCEMENT_MODE
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

    # ------------------------------------------------------------------ #
    # Shared helpers for all-layer enforcement                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_layer_exempt(target: type) -> bool:
        """Check if a non-model class is exempt from layer enforcement."""
        if getattr(target, "_flext_enforcement_exempt", False):
            return True
        module = getattr(target, "__module__", "") or ""
        return any(frag in module for frag in _ec.ENFORCEMENT_EXEMPT_MODULE_FRAGMENTS)

    @staticmethod
    def _emit(
        qualname: str,
        layer: str,
        errors: Sequence[str],
        severity: str,
    ) -> None:
        """Emit enforcement violation as warning and optionally TypeError."""
        mode = _ec.ENFORCEMENT_MODE
        msg = (
            f"\n{qualname} violates FLEXT {layer} {severity} rules:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + f"\n\nFix: See AGENTS.md § {layer} governance."
            + "\nExempt: _flext_enforcement_exempt: ClassVar[bool] = True"
        )
        warnings.warn(msg, UserWarning, stacklevel=4)
        if mode == "strict":
            raise TypeError(msg)

    # ------------------------------------------------------------------ #
    # Constants layer enforcement                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_constant_attr(name: str, value: object) -> bool:
        """Return True if name/value looks like a constant attribute."""
        if name.startswith("_"):
            return False
        if name in _ec.ENFORCEMENT_CONSTANTS_SKIP_ATTRS:
            return False
        if isinstance(value, (type, classmethod, staticmethod, property)):
            return False
        return not callable(value)

    @staticmethod
    def check_constants_no_mutable_values(target: type) -> Sequence[str]:
        """Constants must not have mutable default values (list/dict/set)."""
        errors: MutableSequence[str] = []
        own_dict = vars(target)
        for name, value in own_dict.items():
            if not FlextUtilitiesEnforcement._is_constant_attr(name, value):
                continue
            if isinstance(value, (list, dict, set)):
                type_name = type(value).__name__
                errors.append(
                    f"{target.__qualname__}.{name}: mutable {type_name} FORBIDDEN. "
                    f"Use frozenset, tuple, or MappingProxyType.",
                )
        # Recursively check inner namespace classes (not StrEnum/IntEnum)
        for inner in own_dict.values():
            if not isinstance(inner, type):
                continue
            if isinstance(inner, EnumType):
                continue
            errors.extend(
                FlextUtilitiesEnforcement.check_constants_no_mutable_values(inner),
            )
        return errors

    @staticmethod
    def check_constants_final_hints(target: type) -> Sequence[str]:
        """Public constant attributes must have Final[...] type hints."""
        errors: MutableSequence[str] = []
        own_annotations = vars(target).get("__annotations__", {})
        own_dict = vars(target)
        for name, value in own_dict.items():
            if not FlextUtilitiesEnforcement._is_constant_attr(name, value):
                continue
            if name not in own_annotations:
                errors.append(
                    f"{target.__qualname__}.{name}: missing type annotation. "
                    f"Use Final[type] or ClassVar[type].",
                )
                continue
            annotation_str = str(own_annotations[name])
            if "Final" not in annotation_str and "ClassVar" not in annotation_str:
                errors.append(
                    f"{target.__qualname__}.{name}: must be Final[...] or ClassVar[...]. "
                    f"Constants must be immutable.",
                )
        return errors

    @staticmethod
    def check_constants_upper_case(target: type) -> Sequence[str]:
        """Public constant attribute names must be UPPER_CASE."""
        errors: MutableSequence[str] = []
        own_dict = vars(target)
        for name, value in own_dict.items():
            if not FlextUtilitiesEnforcement._is_constant_attr(name, value):
                continue
            if name != name.upper():
                errors.append(
                    f"{target.__qualname__}.{name}: constant names must be UPPER_CASE.",
                )
        return errors

    @staticmethod
    def run_constants(target: type) -> None:
        """Enforce governance rules on FlextConstants subclasses.

        Called from FlextConstants.__init_subclass__ on facade.
        """
        mode = _ec.ENFORCEMENT_MODE
        if mode == "off":
            return
        if FlextUtilitiesEnforcement._is_layer_exempt(target):
            return

        hard_errors: MutableSequence[str] = []
        hard_errors.extend(
            FlextUtilitiesEnforcement.check_constants_no_mutable_values(target),
        )

        if hard_errors:
            FlextUtilitiesEnforcement._emit(
                target.__qualname__,
                "Constants",
                hard_errors,
                "HARD",
            )

        config_errors: MutableSequence[str] = []
        config_errors.extend(
            FlextUtilitiesEnforcement.check_constants_final_hints(target),
        )
        config_errors.extend(
            FlextUtilitiesEnforcement.check_constants_upper_case(target),
        )

        if config_errors:
            FlextUtilitiesEnforcement._emit(
                target.__qualname__,
                "Constants",
                config_errors,
                "best practices",
            )

    # ------------------------------------------------------------------ #
    # Protocols layer enforcement                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_protocols_inner_classes(target: type) -> Sequence[str]:
        """Inner classes in protocol facades should be Protocol subclasses."""
        errors: MutableSequence[str] = []
        own_dict = vars(target)
        for name, value in own_dict.items():
            if not isinstance(value, type):
                continue
            if name.startswith("_"):
                continue
            if issubclass(value, Protocol):
                continue
            if isinstance(value, EnumType):
                continue
            errors.append(
                f"{target.__qualname__}.{name}: inner class must be a Protocol subclass. "
                f"Use class {name}(Protocol): ... with @runtime_checkable.",
            )
        return errors

    @staticmethod
    def check_protocols_runtime_checkable(target: type) -> Sequence[str]:
        """Protocol inner classes must be @runtime_checkable."""
        errors: MutableSequence[str] = []
        own_dict = vars(target)
        for name, value in own_dict.items():
            if not isinstance(value, type):
                continue
            if name.startswith("_"):
                continue
            if not issubclass(value, Protocol):
                continue
            if not getattr(value, "_is_runtime_protocol", False):
                errors.append(
                    f"{target.__qualname__}.{name}: Protocol must be @runtime_checkable.",
                )
        return errors

    @staticmethod
    def run_protocols(target: type) -> None:
        """Enforce governance rules on FlextProtocols subclasses.

        Called from FlextProtocols.__init_subclass__ on facade.
        """
        mode = _ec.ENFORCEMENT_MODE
        if mode == "off":
            return
        if FlextUtilitiesEnforcement._is_layer_exempt(target):
            return

        config_errors: MutableSequence[str] = []
        config_errors.extend(
            FlextUtilitiesEnforcement.check_protocols_inner_classes(target),
        )
        config_errors.extend(
            FlextUtilitiesEnforcement.check_protocols_runtime_checkable(target),
        )

        if config_errors:
            FlextUtilitiesEnforcement._emit(
                target.__qualname__,
                "Protocols",
                config_errors,
                "best practices",
            )

    # ------------------------------------------------------------------ #
    # Types layer enforcement                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_types_no_any_in_aliases(target: type) -> Sequence[str]:
        """PEP 695 type aliases must not reference Any."""
        errors: MutableSequence[str] = []
        own_dict = vars(target)
        for name, value in own_dict.items():
            if name.startswith("_"):
                continue
            if not hasattr(value, "__value__"):
                continue
            alias_str = str(value)
            if "Any" in alias_str:
                errors.append(
                    f"{target.__qualname__}.{name}: Any in type alias FORBIDDEN. "
                    f"Use t.* contracts.",
                )
        return errors

    @staticmethod
    def check_types_typeadapter_placement(target: type) -> Sequence[str]:
        """TypeAdapter instances must be in FlextTypes* hierarchy only."""
        errors: MutableSequence[str] = []
        own_dict = vars(target)
        for name, value in own_dict.items():
            if name.startswith("_"):
                continue
            type_name = type(value).__name__
            if (
                type_name == "TypeAdapter"
                and not name.startswith("ADAPTER_")
                and name.upper() != name
            ):
                errors.append(
                    f"{target.__qualname__}.{name}: TypeAdapter should use "
                    f"UPPER_CASE naming (e.g., ADAPTER_{name.upper()}).",
                )
        return errors

    @staticmethod
    def run_types(target: type) -> None:
        """Enforce governance rules on FlextTypes subclasses.

        Called from FlextTypes.__init_subclass__ on facade.
        """
        mode = _ec.ENFORCEMENT_MODE
        if mode == "off":
            return
        if FlextUtilitiesEnforcement._is_layer_exempt(target):
            return

        hard_errors: MutableSequence[str] = []
        hard_errors.extend(
            FlextUtilitiesEnforcement.check_types_no_any_in_aliases(target),
        )

        if hard_errors:
            FlextUtilitiesEnforcement._emit(
                target.__qualname__,
                "Types",
                hard_errors,
                "HARD",
            )

        config_errors: MutableSequence[str] = []
        config_errors.extend(
            FlextUtilitiesEnforcement.check_types_typeadapter_placement(target),
        )

        if config_errors:
            FlextUtilitiesEnforcement._emit(
                target.__qualname__,
                "Types",
                config_errors,
                "best practices",
            )

    # ------------------------------------------------------------------ #
    # Utilities layer enforcement                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_utilities_method_types(target: type) -> Sequence[str]:
        """Public methods on utility classes must be static or classmethod."""
        errors: MutableSequence[str] = []
        own_dict = vars(target)
        exempt = _ec.ENFORCEMENT_UTILITIES_EXEMPT_METHODS
        for name, value in own_dict.items():
            if name.startswith("_") and name not in exempt:
                continue
            if name in exempt:
                continue
            if isinstance(value, (staticmethod, classmethod)):
                continue
            if not inspect.isfunction(value):
                continue
            errors.append(
                f"{target.__qualname__}.{name}(): must be @staticmethod or @classmethod. "
                f"Utilities must be stateless.",
            )
        return errors

    @staticmethod
    def run_utilities(target: type) -> None:
        """Enforce governance rules on FlextUtilities subclasses.

        Called from FlextUtilities.__init_subclass__ on facade.
        """
        mode = _ec.ENFORCEMENT_MODE
        if mode == "off":
            return
        if FlextUtilitiesEnforcement._is_layer_exempt(target):
            return

        config_errors: MutableSequence[str] = []
        config_errors.extend(
            FlextUtilitiesEnforcement.check_utilities_method_types(target),
        )

        if config_errors:
            FlextUtilitiesEnforcement._emit(
                target.__qualname__,
                "Utilities",
                config_errors,
                "best practices",
            )
