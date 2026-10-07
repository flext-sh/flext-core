"""Guards type protocol string module.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flext_core._utilities._guards_type_protocol_types import ProtocolGuardInput

_STRING_TYPE_PREDICATES: dict[str, Callable[[ProtocolGuardInput], bool]] = {
    "str": lambda value: isinstance(value, str),
    "dict": lambda value: isinstance(value, dict),
    "list": lambda value: isinstance(value, list),
    "tuple": lambda value: isinstance(value, tuple),
    "sequence": lambda value: isinstance(value, (list, tuple, range)),
    "mapping": lambda value: isinstance(value, Mapping),
    "list_or_tuple": lambda value: isinstance(value, (list, tuple)),
    "sequence_not_str": lambda value: (
        isinstance(value, (list, tuple, range)) and not isinstance(value, str)
    ),
    "sequence_not_str_bytes": lambda value: (
        isinstance(value, (list, tuple, range)) and not isinstance(value, (str, bytes))
    ),
    "sized": lambda value: hasattr(value, "__len__"),
    "callable": callable,
    "bytes": lambda value: isinstance(value, bytes),
    "int": lambda value: isinstance(value, int),
    "float": lambda value: isinstance(value, float),
    "bool": lambda value: isinstance(value, bool),
    "none": lambda value: value is None,
    "string_non_empty": lambda value: isinstance(value, str) and bool(value.strip()),
    "dict_non_empty": lambda value: isinstance(value, Mapping) and len(value) > 0,
    "list_non_empty": lambda value: (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) > 0
    ),
}


class FlextUtilitiesGuardsTypeProtocolStringMixin:
    @staticmethod
    def _run_string_type_check(type_name: str, value: ProtocolGuardInput) -> bool:
        """Run the named string-keyed type predicate against the value.

        Returns:
            The resulting ``bool``.

        """
        predicate = _STRING_TYPE_PREDICATES.get(type_name)
        return predicate(value) if predicate is not None else False


__all__: list[str] = ["FlextUtilitiesGuardsTypeProtocolStringMixin"]
