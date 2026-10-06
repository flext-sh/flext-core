"""Guards type protocol string module.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core._utilities._guards_type_protocol_types import ProtocolGuardInput


def _has_len(value: ProtocolGuardInput) -> bool:
    """Return whether ``value`` exposes ``__len__``.

    Returns:
        The resulting ``bool``.

    """
    return hasattr(value, "__len__")


def _is_none(value: ProtocolGuardInput) -> bool:
    """Return whether ``value`` is ``None``.

    Returns:
        The resulting ``bool``.

    """
    return value is None


class FlextUtilitiesGuardsTypeProtocolStringMixin:
    """String-keyed runtime predicates shared by the protocol guards."""

    @staticmethod
    def _string_predicates() -> Mapping[str, object]:
        """Return the canonical ``type_name`` → predicate registry.

        Returns:
            The resulting ``Mapping[str, object]``.

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
            "sized": _has_len,
            "callable": callable,
            "bytes": lambda value: isinstance(value, bytes),
            "int": lambda value: isinstance(value, int),
            "float": lambda value: isinstance(value, float),
            "bool": lambda value: isinstance(value, bool),
            "none": _is_none,
            "string_non_empty": lambda value: (
                isinstance(value, str)
                and bool(
                    value.strip(),
                )
            ),
            "dict_non_empty": lambda value: (
                isinstance(value, Mapping)
                and bool(
                    len(value),
                )
            ),
            "list_non_empty": lambda value: (
                isinstance(value, Sequence)
                and not isinstance(value, (str, bytes, bytearray))
                and bool(len(value))
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
        return bool(predicate(value))


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocolStringMixin"]
