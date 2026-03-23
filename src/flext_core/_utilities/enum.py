"""Internal enum utilities - DO NOT IMPORT DIRECTLY.

This module provides enum utility functions following the generalized function pattern.
All functionality should be accessed via the u facade in flext_core.utilities.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from enum import StrEnum
from typing import ClassVar, Literal, TypeIs, overload

from pydantic import ValidationError

from flext_core import EnumT, m, r, t


class FlextUtilitiesEnum:
    """Utilities for working with StrEnum in a type-safe way."""

    _values_cache: ClassVar[MutableMapping[type[StrEnum], frozenset[str]]] = {}
    _names_cache: ClassVar[MutableMapping[type[StrEnum], frozenset[str]]] = {}
    _members_cache: ClassVar[MutableMapping[type[StrEnum], frozenset[StrEnum]]] = {}
    _V = m.Validators

    @staticmethod
    def _is_strenum_class(value: t.GuardInput) -> TypeIs[type[StrEnum]]:
        return isinstance(value, type) and issubclass(value, StrEnum)

    @staticmethod
    def _coerce[E: StrEnum](enum_cls: type[E], value: str | E) -> E:
        """Coerce value to enum - for Pydantic validators (internal helper)."""
        if isinstance(value, enum_cls):
            return value
        return enum_cls(value)

    @staticmethod
    def _is_member_by_name[E: StrEnum](name: str, enum_cls: type[E]) -> TypeIs[E]:
        """Check membership by name (internal helper)."""
        return name in enum_cls.__members__

    @staticmethod
    def _is_member_by_value[E: StrEnum](
        value: t.Scalar | E,
        enum_cls: type[E],
    ) -> TypeIs[E]:
        """Check membership by value (internal helper)."""
        if isinstance(value, enum_cls):
            return True
        if isinstance(value, str):
            value_map = enum_cls._value2member_map_
            return value in value_map
        return False

    @staticmethod
    def _parse(enum_cls: type[EnumT], value: str | EnumT) -> r[EnumT]:
        """Parse string to enum (internal helper)."""
        if isinstance(value, enum_cls):
            return r[EnumT].ok(value)
        try:
            return r[EnumT].ok(enum_cls(value))
        except ValueError:
            members_dict = enum_cls.__members__
            enum_members = list(members_dict.values())
            valid = ", ".join(m.value for m in enum_members)
            enum_name = enum_cls.__name__
            return r[EnumT].fail(f"Invalid {enum_name}: '{value}'. Valid: {valid}")

    @staticmethod
    def _validate_str(value: t.Scalar | StrEnum) -> r[str]:
        """Validate strict string input for parsing paths."""
        try:
            return r[str].ok(
                FlextUtilitiesEnum._V.strict_string_adapter().validate_python(value),
            )
        except ValidationError:
            return r[str].fail("Value is not a valid string input")

    @staticmethod
    def auto_value(name: str) -> str:
        """Generate lowercase value from enum member name.

        Use in StrEnum with `_generate_next_value_` override.

        """
        return name.lower()

    @staticmethod
    def bi_map[K, V](data: Mapping[K, V]) -> tuple[Mapping[K, V], Mapping[V, K]]:
        """Create bidirectional mapping from dict.

        Returns (forward, inverse) tuple of dicts.
        Use for replacing paired Mapping declarations.

        """
        forward = dict(data)
        inverse = {v: k for k, v in data.items()}
        return (forward, inverse)

    @staticmethod
    def coerce_by_name_validator[E: StrEnum](
        enum_cls: type[E],
    ) -> Callable[[t.Scalar | E], E]:
        """BeforeValidator that accepts name OR value of enum."""

        def _coerce(value: t.Scalar | E) -> E:
            if isinstance(value, enum_cls):
                return value
            if isinstance(value, str):
                members_dict = enum_cls.__members__
                if value in members_dict:
                    member = members_dict[value]
                    if isinstance(member, enum_cls):
                        return member
                try:
                    return enum_cls(value)
                except ValueError:
                    pass
            enum_name = enum_cls.__name__
            msg = f"Invalid {enum_name}: {value!r}"
            raise ValueError(msg)

        return _coerce

    @staticmethod
    def coerce_validator[E: StrEnum](enum_cls: type[E]) -> Callable[[t.Scalar | E], E]:
        """Create BeforeValidator for automatic coercion in Pydantic."""

        def _coerce(value: t.Scalar | E) -> E:
            if isinstance(value, enum_cls):
                return value
            if isinstance(value, str):
                try:
                    return enum_cls(value)
                except ValueError:
                    pass
            enum_name = enum_cls.__name__
            msg = f"Invalid {enum_name}: {value!r}"
            raise ValueError(msg)

        return _coerce

    @staticmethod
    def create_discriminated_union(
        _discriminator_field: str,
        *enum_classes: type[StrEnum],
    ) -> Mapping[str, type[StrEnum]]:
        """Create discriminated union mapping for Pydantic models.

        Args:
            discriminator_field: Field name used as discriminator
            *enum_classes: StrEnum classes to include in union

        Returns:
            Mapping of discriminator values to enum classes

        """
        union_map: MutableMapping[str, type[StrEnum]] = {}
        for enum_class in enum_classes:
            for member in enum_class.__members__.values():
                union_map[member.value] = enum_class
        return union_map

    @staticmethod
    def create_enum(name: str, values: Mapping[str, str]) -> type[StrEnum] | StrEnum:
        """Create StrEnum dynamically from values dict.

        Factory method for reducing StrEnum boilerplate during constants refactoring.
        Enables transitioning from class definitions to alias-based constants.

        Args:
            name: StrEnum class name (will be used as __name__)
            values: Dictionary mapping member names to string values

        Returns:
            Newly created StrEnum class with specified members

        """
        members_list = [(k, v) for k, v in values.items()]
        created = StrEnum(name, members_list)
        if isinstance(created, type) and issubclass(created, StrEnum):
            return created
        msg = f"StrEnum({name!r}) did not produce a StrEnum subclass"
        raise TypeError(msg)

    @overload
    @staticmethod
    def dispatch[E: StrEnum](
        value: t.Scalar | E,
        enum_cls: type[E],
        *,
        mode: Literal["is_member"] = "is_member",
        by_name: bool = False,
    ) -> TypeIs[E]: ...

    @overload
    @staticmethod
    def dispatch[E: StrEnum](
        value: t.Scalar | E,
        enum_cls: type[E],
        *,
        mode: Literal["is_name"],
        by_name: bool = False,
    ) -> TypeIs[E]: ...

    @overload
    @staticmethod
    def dispatch[E: StrEnum](
        value: t.Scalar | E,
        enum_cls: type[E],
        *,
        mode: Literal["parse"],
        by_name: bool = False,
    ) -> r[E]: ...

    @overload
    @staticmethod
    def dispatch[E: StrEnum](
        value: t.Scalar | E,
        enum_cls: type[E],
        *,
        mode: Literal["coerce"],
        by_name: bool = False,
    ) -> E: ...

    @staticmethod
    def dispatch[E: StrEnum](
        value: t.Scalar | E,
        enum_cls: type[E],
        *,
        mode: str = "is_member",
        by_name: bool = False,
    ) -> bool | r[E] | E:
        """Generalized enum utility dispatch function.

        Args:
            value: Value to check/parse/coerce
            enum_cls: The StrEnum class
            mode: Operation mode
                - "is_member": Check if value is member (returns bool, acts as TypeIs[E])
                - "is_name": Check if value is member by name (returns bool, acts as TypeIs[E])
                - "parse": Parse value to enum (returns r[E])
                - "coerce": Coerce value to enum (returns E, raises on failure)
            by_name: For is_member, check by name instead of value

        Returns:
            Depends on mode - bool (for is_member/is_name), r[E] (for parse), or E (for coerce)

        """
        if mode == "is_member":
            by_name_value = FlextUtilitiesEnum._validate_str(value)
            if by_name and by_name_value.is_success:
                is_member_result: bool = FlextUtilitiesEnum._is_member_by_name(
                    str(by_name_value.value),
                    enum_cls,
                )
                return is_member_result
            result_bool: bool = FlextUtilitiesEnum._is_member_by_value(value, enum_cls)
            return result_bool
        if mode == "is_name":
            return FlextUtilitiesEnum._is_member_by_name(str(value), enum_cls)
        if mode == "parse":
            if isinstance(value, enum_cls):
                return FlextUtilitiesEnum._parse(enum_cls, value)
            validated_value = FlextUtilitiesEnum._validate_str(value)
            if validated_value.is_success:
                return FlextUtilitiesEnum._parse(enum_cls, str(validated_value.value))
            return FlextUtilitiesEnum._parse(enum_cls, str(value))
        if mode == "coerce":
            if isinstance(value, enum_cls):
                return value
            validated_value = FlextUtilitiesEnum._validate_str(value)
            if validated_value.is_success:
                coerced: E = FlextUtilitiesEnum._coerce(
                    enum_cls,
                    str(validated_value.value),
                )
                return coerced
            value_str: str = str(value)
            coerced_str: E = FlextUtilitiesEnum._coerce(enum_cls, value_str)
            return coerced_str
        error_msg = f"Unknown mode: {mode}"
        raise ValueError(error_msg)

    @staticmethod
    def get_enum_values(enum_class: type[StrEnum]) -> Sequence[str]:
        """Get all values from StrEnum class.

        Returns immutable sequence for safe iteration.

        Args:
            enum_class: StrEnum class to extract values from

        Returns:
            Immutable sequence of enum string values

        """
        return tuple(member.value for member in enum_class.__members__.values())

    @staticmethod
    def is_member[E: StrEnum](enum_cls: type[E], value: t.Scalar | E) -> bool:
        """Generic membership check for any StrEnum."""
        if isinstance(value, enum_cls):
            return True
        if isinstance(value, str):
            value_map = enum_cls._value2member_map_
            return value in value_map
        return False

    @staticmethod
    def is_subset[E: StrEnum](
        enum_cls: type[E],
        valid_members: frozenset[E],
        value: t.Scalar | E,
    ) -> bool:
        """TypeIs for subset of a StrEnum."""
        if isinstance(value, enum_cls) and value in valid_members:
            return True
        if isinstance(value, str):
            try:
                member = enum_cls(value)
                return member in valid_members
            except ValueError:
                return False
        return False

    @staticmethod
    def members[E: StrEnum](enum_cls: type[E]) -> frozenset[E]:
        """Return frozenset of members (cached for performance)."""
        cached = FlextUtilitiesEnum._members_cache.get(enum_cls)
        if isinstance(cached, frozenset):
            return frozenset(enum_cls(member.value) for member in cached)
        members_dict: Mapping[str, E] = enum_cls.__members__
        result: frozenset[E] = frozenset(members_dict.values())
        FlextUtilitiesEnum._members_cache[enum_cls] = result
        return result

    @staticmethod
    def names[E: StrEnum](enum_cls: type[E]) -> frozenset[str]:
        """Return frozenset of member names (cached for performance)."""
        if enum_cls in FlextUtilitiesEnum._names_cache:
            return FlextUtilitiesEnum._names_cache[enum_cls]
        members_dict: Mapping[str, E] = enum_cls.__members__
        result = frozenset(members_dict.keys())
        FlextUtilitiesEnum._names_cache[enum_cls] = result
        return result

    @staticmethod
    def parse_enum(enum_cls: type[EnumT], value: str | EnumT) -> r[EnumT]:
        """Convert string to StrEnum with r."""
        if isinstance(value, enum_cls):
            return r[EnumT].ok(value)
        try:
            return r[EnumT].ok(enum_cls(value))
        except ValueError:
            members_dict = enum_cls.__members__
            enum_members = list(members_dict.values())
            valid = ", ".join(m.value for m in enum_members)
            enum_name = enum_cls.__name__
            return r[EnumT].fail(f"Invalid {enum_name}: '{value}'. Valid: {valid}")

    @staticmethod
    def parse_or_default[E: StrEnum](
        enum_cls: type[E],
        value: str | E | None,
        default: E,
    ) -> E:
        """Convert with fallback to default (never fails)."""
        if value is None:
            return default
        if isinstance(value, enum_cls):
            return value
        try:
            return enum_cls(value)
        except ValueError:
            return default

    @staticmethod
    def values[E: StrEnum](enum_cls: type[E]) -> frozenset[str]:
        """Return frozenset of values (cached for performance)."""
        if enum_cls in FlextUtilitiesEnum._values_cache:
            return FlextUtilitiesEnum._values_cache[enum_cls]
        members_dict: Mapping[str, E] = enum_cls.__members__
        result = frozenset(m.value for m in members_dict.values())
        FlextUtilitiesEnum._values_cache[enum_cls] = result
        return result


__all__ = ["FlextUtilitiesEnum"]
