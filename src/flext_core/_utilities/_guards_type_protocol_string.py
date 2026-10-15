"""Guards type protocol string module.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from flext_core import t
    from flext_core.typings import ProtocolGuardInput


class FlextUtilitiesGuardsTypeProtocolStringMixin:
    """String-keyed runtime predicates shared by the protocol guards."""

    @staticmethod
    def _has_len(value: ProtocolGuardInput) -> bool:
        """Return whether ``value`` exposes ``__len__``.

        Returns:
            The resulting ``bool``.

        """
        return hasattr(value, "__len__")

    @staticmethod
    def _is_none(value: ProtocolGuardInput) -> bool:
        """Return whether ``value`` is ``None``.

        Returns:
            The resulting ``bool``.

        """
        return value is None

    @staticmethod
    def _is_non_empty_mapping(value: ProtocolGuardInput) -> bool:
        """Return whether ``value`` is a non-empty mapping.

        Returns:
            The resulting ``bool``.

        """
        return isinstance(value, Mapping) and bool(
            cast("Mapping[str, t.JsonValue]", value),
        )

    @staticmethod
    def _is_non_empty_sequence(value: ProtocolGuardInput) -> bool:
        """Return whether ``value`` is a non-empty non-text sequence.

        Returns:
            The resulting ``bool``.

        """
        return (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and bool(cast("Sequence[t.JsonValue]", value))
        )

    @staticmethod
    def _string_predicates() -> Mapping[str, Callable[[ProtocolGuardInput], bool]]:
        """Return the canonical ``type_name`` → predicate registry.

        Returns:
            The resulting ``Mapping[str, Callable[[ProtocolGuardInput], bool]]``.

        """
        return {
            "str": lambda value: isinstance(value, str),
            "dict": lambda value: isinstance(value, dict),
            "list": lambda value: isinstance(value, list),
            "tuple": lambda value: isinstance(value, tuple),
            "sequence": lambda value: isinstance(value, (list, tuple, range)),
            "mapping": lambda value: isinstance(value, Mapping),
            "list_or_tuple": lambda value: isinstance(value, (list, tuple)),
            "sequence_not_str": lambda value: (
                isinstance(
                    value,
                    (list, tuple, range),
                )
                and not isinstance(value, str)
            ),
            "sequence_not_str_bytes": lambda value: (
                isinstance(
                    value,
                    (list, tuple, range),
                )
                and not isinstance(value, (str, bytes))
            ),
            "sized": FlextUtilitiesGuardsTypeProtocolStringMixin._has_len,
            "callable": callable,
            "bytes": lambda value: isinstance(value, bytes),
            "int": lambda value: isinstance(value, int),
            "float": lambda value: isinstance(value, float),
            "bool": lambda value: isinstance(value, bool),
            "none": FlextUtilitiesGuardsTypeProtocolStringMixin._is_none,
            "string_non_empty": lambda value: (
                isinstance(value, str)
                and bool(
                    value.strip(),
                )
            ),
            "dict_non_empty": (
                FlextUtilitiesGuardsTypeProtocolStringMixin._is_non_empty_mapping
            ),
            "list_non_empty": (
                FlextUtilitiesGuardsTypeProtocolStringMixin._is_non_empty_sequence
            ),
        }

    @staticmethod
    def _run_string_type_check(type_name: str, value: ProtocolGuardInput) -> bool:
        """Evaluate the registry predicate registered under ``type_name``.

        Returns:
            The resulting ``bool``.

        """
        predicates = FlextUtilitiesGuardsTypeProtocolStringMixin._string_predicates()
        predicate = predicates.get(type_name)
        if predicate is None:
            return False
        return predicate(value)


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocolStringMixin"]
