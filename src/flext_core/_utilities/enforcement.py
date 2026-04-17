"""Runtime enforcement utilities for Pydantic v2 governance.

Static methods called from __pydantic_init_subclass__ hooks on FLEXT
base model classes and __init_subclass__ on facade classes.
Constants come from c.ENFORCEMENT_* via MRO.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Mapping, MutableSequence, MutableSet, Sequence
from enum import EnumType
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, get_args, get_origin

from pydantic.fields import FieldInfo

from flext_core import (
    FlextConstantsEnforcement as c,
    FlextModelsPydantic as mp,
    FlextUtilitiesBeartypeEngine,
)

if TYPE_CHECKING:
    from flext_core import t


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
    def own_fields(
        model_type: type[mp.BaseModel],
    ) -> Mapping[str, FieldInfo]:
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
    ) -> t.StrSequence:
        """Reject Any in field annotations — recursive via beartype.door."""
        errors: MutableSequence[str] = []
        for name, info in own.items():
            if FlextUtilitiesBeartypeEngine.contains_any(info.annotation):
                errors.append(
                    f'Field "{name}": Any is FORBIDDEN (detected recursively). '
                    f"Use a t.* type contract.",
                )
        return errors

    @staticmethod
    def check_no_bare_collections(
        own: Mapping[str, FieldInfo],
    ) -> t.StrSequence:
        """Reject dict/list/set as field annotation origins via beartype engine."""
        replacements: t.StrMapping = dict(
            c.ENFORCEMENT_COLLECTION_REPLACEMENTS,
        )
        errors: MutableSequence[str] = []
        for name, info in own.items():
            is_forbidden, origin_name = (
                FlextUtilitiesBeartypeEngine.has_forbidden_collection_origin(
                    info.annotation,
                    c.ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS,
                )
            )
            if is_forbidden:
                fix = replacements.get(origin_name, origin_name)
                errors.append(
                    f'Field "{name}": bare {origin_name}[...] FORBIDDEN. Use {fix}.',
                )
        return errors

    @staticmethod
    def check_no_v1_patterns(
        model_type: type[mp.BaseModel],
    ) -> t.StrSequence:
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
        model_type: type[mp.BaseModel],
        own: Mapping[str, FieldInfo],
    ) -> t.StrSequence:
        """Require m.Field(description=...) on all public fields."""
        errors: MutableSequence[str] = []
        raw_annotation_map = vars(model_type).get("__annotations__", {})
        resolved_annotations = inspect.get_annotations(model_type, eval_str=False)
        for name, info in own.items():
            if name.startswith("_"):
                continue
            description = info.description
            if not description:
                raw_annotation = raw_annotation_map.get(name)
                if isinstance(raw_annotation, str) and "description=" in raw_annotation:
                    description = raw_annotation
            if not description:
                annotation = resolved_annotations.get(name)
                if get_origin(annotation) is Annotated:
                    for metadata in get_args(annotation)[1:]:
                        if isinstance(metadata, FieldInfo) and metadata.description:
                            description = metadata.description
                            break
            if not description:
                errors.append(
                    f'Field "{name}": m.Field() must include description="...".',
                )
        return errors

    @staticmethod
    def check_extra_policy(
        model_type: type[mp.BaseModel],
    ) -> t.StrSequence:
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
    ) -> t.StrSequence:
        """Reject object as field annotation — use specific t.* contract."""
        errors: MutableSequence[str] = []
        for name, info in own.items():
            if info.annotation is object:
                errors.append(
                    f'Field "{name}" is FORBIDDEN. Use a t.* type contract.',
                )
        return errors

    @staticmethod
    def check_no_str_none_with_empty_default(
        own: Mapping[str, FieldInfo],
    ) -> t.StrSequence:
        """Reject str | None with default='' via beartype engine."""
        errors: MutableSequence[str] = []
        for name, info in own.items():
            if not FlextUtilitiesBeartypeEngine.is_str_none_union(info.annotation):
                continue
            if isinstance(info.default, str) and not info.default:
                errors.append(
                    f'Field "{name}": str | None with default="" is wrong. '
                    f'Use str with default="" (None has no business meaning here).',
                )
        return errors

    @staticmethod
    def check_frozen_value_objects(
        model_type: type[mp.BaseModel],
    ) -> t.StrSequence:
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
    ) -> t.StrSequence:
        """Reject mutable defaults ([], {}, set()) — use m.Field(default_factory=...)."""
        errors: MutableSequence[str] = []
        for name, info in own.items():
            default = info.default
            if isinstance(default, (list, dict, set)) and not default:
                if isinstance(default, list):
                    type_name = "list"
                elif isinstance(default, dict):
                    type_name = "dict"
                else:
                    type_name = "set"
                errors.append(
                    f'Field "{name}": mutable default {type_name}() is FORBIDDEN. '
                    f"Use m.Field(default_factory={type_name}).",
                )
        return errors

    @staticmethod
    def check_no_inline_union_types(
        own: Mapping[str, FieldInfo],
    ) -> t.StrSequence:
        """Flag complex inline union types via beartype engine."""
        errors: MutableSequence[str] = []
        max_inline_union = 2
        for name, info in own.items():
            count = FlextUtilitiesBeartypeEngine.count_union_members(info.annotation)
            if count > max_inline_union:
                errors.append(
                    f'Field "{name}": complex inline union with {count}+ types. '
                    f"Centralize as a t.* type alias in typings.py.",
                )
        return errors

    @staticmethod
    def run(model_type: type[mp.BaseModel]) -> None:
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
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
            if mode == "strict":
                raise TypeError(msg)

        # CONFIGURABLE checks
        config_errors: MutableSequence[str] = []
        config_errors.extend(
            FlextUtilitiesEnforcement.check_field_descriptions(model_type, fields),
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
        return any(frag in module for frag in c.ENFORCEMENT_EXEMPT_MODULE_FRAGMENTS)

    @staticmethod
    def _emit(
        qualname: str,
        layer: str,
        errors: t.StrSequence,
        severity: str,
    ) -> None:
        """Emit enforcement violation as warning and optionally TypeError."""
        mode = c.ENFORCEMENT_MODE
        msg = (
            f"\n{qualname} violates FLEXT {layer} {severity} rules:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + f"\n\nFix: See AGENTS.md § {layer} governance."
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
        if name in c.ENFORCEMENT_CONSTANTS_SKIP_ATTRS:
            return False
        if isinstance(value, (type, classmethod, staticmethod, property)):
            return False
        return not callable(value)

    @staticmethod
    def check_constants_no_mutable_values(target: type) -> t.StrSequence:
        """Constants must not have mutable default values (list/dict/set)."""
        errors: MutableSequence[str] = []
        own_dict = vars(target)
        for name, value in own_dict.items():
            if not FlextUtilitiesEnforcement._is_constant_attr(name, value):
                continue
            if isinstance(value, (list, dict, set)):
                if isinstance(value, list):
                    type_name = "list"
                elif isinstance(value, dict):
                    type_name = "dict"
                else:
                    type_name = "set"
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
    def check_constants_final_hints(target: type) -> t.StrSequence:
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
    def check_constants_upper_case(target: type) -> t.StrSequence:
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
        mode = c.ENFORCEMENT_MODE
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

        FlextUtilitiesEnforcement.run_namespace_checks(target, "constants")

    # ------------------------------------------------------------------ #
    # Protocols layer enforcement                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_protocol_class(target: type) -> bool:
        """Return True when target is a Protocol subclass."""
        return bool(getattr(target, "_is_protocol", False))

    @staticmethod
    def _iter_protocol_inner_types(target: type) -> Sequence[tuple[str, type]]:
        """Return public inner classes defined directly on target."""
        return [
            (name, value)
            for name, value in vars(target).items()
            if isinstance(value, type) and not name.startswith("_")
        ]

    @staticmethod
    def _iter_protocol_mro_inner_types(target: type) -> Sequence[tuple[str, type]]:
        """Return public inner classes inherited through protocol namespace MRO."""
        inherited: MutableSequence[tuple[str, type]] = []
        seen: MutableSet[str] = set()
        for base in target.__mro__[1:]:
            if base is object:
                continue
            for name, value in FlextUtilitiesEnforcement._iter_protocol_inner_types(
                base,
            ):
                if name in seen:
                    continue
                seen.add(name)
                inherited.append((name, value))
        return inherited

    @staticmethod
    def _iter_protocol_effective_inner_types(
        target: type,
    ) -> Sequence[tuple[str, type]]:
        """Return direct inner classes or inherited protocol members for MRO holders."""
        direct_inner_types = FlextUtilitiesEnforcement._iter_protocol_inner_types(
            target,
        )
        if direct_inner_types:
            return direct_inner_types
        return FlextUtilitiesEnforcement._iter_protocol_mro_inner_types(target)

    @staticmethod
    def _is_protocol_namespace_holder(target: type) -> bool:
        """Return True when target is a container for nested protocol classes."""
        return bool(
            FlextUtilitiesEnforcement._iter_protocol_effective_inner_types(target),
        )

    @staticmethod
    def _check_protocol_inner_classes(target: type, *, path: str) -> t.StrSequence:
        """Validate nested protocol classes recursively."""
        errors: MutableSequence[str] = []
        for (
            name,
            value,
        ) in FlextUtilitiesEnforcement._iter_protocol_effective_inner_types(
            target,
        ):
            nested_path = f"{path}.{name}"
            if FlextUtilitiesEnforcement._is_protocol_class(value):
                errors.extend(
                    FlextUtilitiesEnforcement._check_protocol_inner_classes(
                        value,
                        path=nested_path,
                    ),
                )
                continue
            if FlextUtilitiesEnforcement._is_protocol_namespace_holder(value):
                errors.extend(
                    FlextUtilitiesEnforcement._check_protocol_inner_classes(
                        value,
                        path=nested_path,
                    ),
                )
                continue
            errors.append(
                f"{nested_path}: inner class must be a Protocol subclass or a namespace holder.",
            )
        return errors

    @staticmethod
    def check_protocols_inner_classes(target: type) -> t.StrSequence:
        """Allow namespace-holder classes while validating nested protocols."""
        return FlextUtilitiesEnforcement._check_protocol_inner_classes(
            target,
            path=target.__qualname__,
        )

    @staticmethod
    def check_protocols_runtime_checkable(target: type) -> t.StrSequence:
        """Protocol inner classes must be @runtime_checkable."""
        errors: MutableSequence[str] = []
        for name, value in FlextUtilitiesEnforcement._iter_protocol_inner_types(target):
            nested_path = f"{target.__qualname__}.{name}"
            if FlextUtilitiesEnforcement._is_protocol_class(value):
                if not getattr(value, "_is_runtime_protocol", False):
                    errors.append(
                        f"{nested_path}: Protocol must be @runtime_checkable.",
                    )
                errors.extend(
                    FlextUtilitiesEnforcement.check_protocols_runtime_checkable(value),
                )
                continue
            if FlextUtilitiesEnforcement._is_protocol_namespace_holder(value):
                errors.extend(
                    FlextUtilitiesEnforcement.check_protocols_runtime_checkable(value),
                )
        return errors

    @staticmethod
    def run_protocols(target: type) -> None:
        """Enforce governance rules on FlextProtocols subclasses.

        Called from FlextProtocols.__init_subclass__ on facade.
        """
        mode = c.ENFORCEMENT_MODE
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

        FlextUtilitiesEnforcement.run_namespace_checks(target, "protocols")

    # ------------------------------------------------------------------ #
    # Types layer enforcement                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_types_no_any_in_aliases(target: type) -> t.StrSequence:
        """PEP 695 type aliases must not reference Any — via beartype.door."""
        errors: MutableSequence[str] = []
        for name, value in vars(target).items():
            if name.startswith("_"):
                continue
            alias_value = getattr(value, "__value__", None)
            if alias_value is None:
                continue
            if FlextUtilitiesBeartypeEngine.alias_contains_any(alias_value):
                errors.append(
                    f"{target.__qualname__}.{name}: Any in type alias FORBIDDEN. "
                    f"Use t.* contracts.",
                )
        return errors

    @staticmethod
    def check_types_typeadapter_placement(target: type) -> t.StrSequence:
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
        mode = c.ENFORCEMENT_MODE
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

        FlextUtilitiesEnforcement.run_namespace_checks(target, "types")

    # ------------------------------------------------------------------ #
    # Utilities layer enforcement                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_utilities_method_types(target: type) -> t.StrSequence:
        """Public methods on utility classes must be static or classmethod."""
        errors: MutableSequence[str] = []
        own_dict = vars(target)
        exempt = c.ENFORCEMENT_UTILITIES_EXEMPT_METHODS
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
        mode = c.ENFORCEMENT_MODE
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

        FlextUtilitiesEnforcement.run_namespace_checks(target, "utilities")

    # ------------------------------------------------------------------ #
    # MRO Namespace enforcement                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_pascal_case(snake: str) -> str:
        """Convert snake_case to PascalCase: 'db_oracle' → 'DbOracle'."""
        normalized = snake.replace("-", "_")
        return "".join(part.capitalize() for part in normalized.split("_") if part)

    @staticmethod
    def _derive_prefix_from_path(target: type) -> str | None:
        """Derive project prefix from source file path (e.g., flext-core → FlextCore)."""
        try:
            src = inspect.getfile(target)
        except (TypeError, OSError):
            return None
        for parent in Path(src).parents:
            if (parent / "pyproject.toml").exists() and parent.name.startswith(
                "flext-",
            ):
                if parent.name == "flext-core":
                    return "Flext"
                slug = parent.name.removeprefix("flext-")
                return "Flext" + FlextUtilitiesEnforcement._to_pascal_case(slug)
        return None

    @staticmethod
    def _derive_project_prefix(target: type) -> str | None:
        """Derive expected class name prefix from module path."""
        module = getattr(target, "__module__", "") or ""
        package = module.split(".")[0] if module else ""
        if not package:
            return None
        for pkg, prefix in c.ENFORCEMENT_NAMESPACE_SPECIAL_PREFIXES:
            if package == pkg:
                return prefix
        package_surface_map: t.StrMapping = {
            "tests": "Tests",
            "examples": "Examples",
            "scripts": "Scripts",
        }
        if package in package_surface_map:
            surface_prefix = package_surface_map[package]
            project_prefix = FlextUtilitiesEnforcement._derive_prefix_from_path(target)
            if project_prefix is not None:
                return surface_prefix + project_prefix
            for base in target.__bases__:
                base_mod = getattr(base, "__module__", "") or ""
                base_pkg = base_mod.split(".")[0] if base_mod else ""
                if (
                    base_pkg.startswith("flext")
                    and base_pkg != "flext_tests"
                    and base.__name__.startswith("Flext")
                ):
                    parent = base.__name__
                    for suffix, _layer in c.ENFORCEMENT_NAMESPACE_LAYER_MAP:
                        if parent.endswith(suffix):
                            return surface_prefix + parent[: -len(suffix)]
                    return surface_prefix + parent
            return None
        if package.startswith("flext_"):
            suffix = package[len("flext_") :]
            return "Flext" + FlextUtilitiesEnforcement._to_pascal_case(suffix)
        return FlextUtilitiesEnforcement._to_pascal_case(package)

    @staticmethod
    def _derive_expected_namespace(target: type) -> str | None:
        """Derive expected inner namespace class name. No exemptions."""
        module = getattr(target, "__module__", "") or ""
        package = module.split(".")[0] if module else ""
        if not package:
            return None
        if package == "flext":
            return "Root"
        if package == "examples":
            return "Examples"
        if package == "tests":
            project_prefix = FlextUtilitiesEnforcement._derive_prefix_from_path(target)
            if project_prefix is not None:
                namespace = project_prefix.removeprefix("Flext")
                return namespace or "Core"
            return "Tests"
        if package.startswith("flext_"):
            suffix = package[len("flext_") :]
            return FlextUtilitiesEnforcement._to_pascal_case(suffix)
        return FlextUtilitiesEnforcement._to_pascal_case(package)

    @staticmethod
    def detect_layer(target: type) -> str | None:
        """Detect layer from class name suffix."""
        name = target.__name__
        for suffix, layer in c.ENFORCEMENT_NAMESPACE_LAYER_MAP:
            if name.endswith(suffix):
                return layer
        return None

    @staticmethod
    def _is_namespace_exempt(target: type) -> bool:
        """Check namespace exemption. Does NOT skip test modules."""
        if getattr(target, "_flext_enforcement_exempt", False):
            return True
        if target.__name__ in c.ENFORCEMENT_NAMESPACE_FACADE_ROOTS:
            return True
        return target.__name__ in c.ENFORCEMENT_INFRASTRUCTURE_BASES

    @staticmethod
    def _emit_namespace(qualname: str, layer: str, errors: t.StrSequence) -> None:
        """Emit namespace violation warning or TypeError."""
        mode = c.ENFORCEMENT_NAMESPACE_MODE
        if mode == "off":
            return
        msg = (
            f"\n{qualname} violates FLEXT {layer} namespace rules:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n\nFix: See AGENTS.md § Facades & Namespaces."
        )
        warnings.warn(msg, UserWarning, stacklevel=5)
        if mode == "strict":
            raise TypeError(msg)

    @staticmethod
    def check_class_prefix(target: type) -> t.StrSequence:
        """Validate class name starts with expected project prefix."""
        if target.__name__ in c.ENFORCEMENT_NAMESPACE_FACADE_ROOTS:
            return []
        if target.__name__ in c.ENFORCEMENT_INFRASTRUCTURE_BASES:
            return []
        prefix = FlextUtilitiesEnforcement._derive_project_prefix(target)
        if prefix is None:
            return []
        if not target.__name__.startswith(prefix):
            module = getattr(target, "__module__", "") or ""
            package = module.split(".")[0] if module else "unknown"
            return [
                (
                    f'Class "{target.__name__}" expected prefix "{prefix}" '
                    f'(derived from package "{package}").'
                ),
            ]
        return []

    @staticmethod
    def check_inner_namespace(target: type) -> t.StrSequence:
        """Validate inner namespace class names match project."""
        expected = FlextUtilitiesEnforcement._derive_expected_namespace(target)
        if expected is None:
            return []
        module = getattr(target, "__module__", "") or ""
        package = module.split(".")[0] if module else ""
        inherited_names: frozenset[str] = frozenset()
        if package in {"tests", "examples"}:
            inherited_names = frozenset(
                name
                for base in target.__bases__
                for name in vars(base)
                if isinstance(getattr(base, name, None), type)
            )
        errors: MutableSequence[str] = []
        for name, value in vars(target).items():
            if not isinstance(value, type):
                continue
            if name.startswith("_"):
                continue
            if isinstance(value, EnumType):
                continue
            if name in inherited_names:
                continue
            if name != expected:
                errors.append(
                    f'Inner class "{name}" must be named "{expected}" '
                    f"(matches project namespace).",
                )
        return errors

    @staticmethod
    def _scan_cross_layer_recursive(
        target: type,
        layer: str,
        path: str = "",
    ) -> t.StrSequence:
        """Recursively scan inner classes for StrEnum/Protocol in wrong layers."""
        errors: MutableSequence[str] = []
        check_strenum = layer not in c.ENFORCEMENT_NAMESPACE_STRENUM_ALLOWED_LAYERS
        check_protocol = layer not in c.ENFORCEMENT_NAMESPACE_PROTOCOL_ALLOWED_LAYERS

        for name, value in vars(target).items():
            if not isinstance(value, type):
                continue
            if name.startswith("_"):
                continue
            full = f"{path}.{name}" if path else name

            if isinstance(value, EnumType):
                if check_strenum:
                    errors.append(
                        f'StrEnum "{full}" must be in constants (c.*), not {layer}.',
                    )
                continue

            if check_protocol and FlextUtilitiesEnforcement._is_protocol_class(value):
                errors.append(
                    f'Protocol "{full}" must be in protocols (p.*), not {layer}.',
                )
                continue

            # Recurse into inner namespace classes
            errors.extend(
                FlextUtilitiesEnforcement._scan_cross_layer_recursive(
                    value,
                    layer,
                    full,
                ),
            )
        return errors

    @staticmethod
    def check_cross_layer_strenum(target: type, layer: str) -> t.StrSequence:
        """Detect StrEnum/IntEnum in wrong layer — recursive scan."""
        if layer in c.ENFORCEMENT_NAMESPACE_STRENUM_ALLOWED_LAYERS:
            return []
        return [
            e
            for e in FlextUtilitiesEnforcement._scan_cross_layer_recursive(
                target,
                layer,
            )
            if "StrEnum" in e
        ]

    @staticmethod
    def check_cross_layer_protocol(target: type, layer: str) -> t.StrSequence:
        """Detect Protocol in wrong layer — recursive scan."""
        if layer in c.ENFORCEMENT_NAMESPACE_PROTOCOL_ALLOWED_LAYERS:
            return []
        return [
            e
            for e in FlextUtilitiesEnforcement._scan_cross_layer_recursive(
                target,
                layer,
            )
            if "Protocol" in e
        ]

    @staticmethod
    def run_namespace_checks(target: type, layer: str) -> None:
        """Run all namespace checks for a given layer."""
        if c.ENFORCEMENT_NAMESPACE_MODE == "off":
            return
        if FlextUtilitiesEnforcement._is_namespace_exempt(target):
            return
        errors: MutableSequence[str] = []
        errors.extend(FlextUtilitiesEnforcement.check_class_prefix(target))
        errors.extend(FlextUtilitiesEnforcement.check_inner_namespace(target))
        errors.extend(
            FlextUtilitiesEnforcement.check_cross_layer_strenum(target, layer),
        )
        errors.extend(
            FlextUtilitiesEnforcement.check_cross_layer_protocol(target, layer),
        )
        if errors:
            FlextUtilitiesEnforcement._emit_namespace(
                target.__qualname__,
                layer,
                errors,
            )
