"""Guard-related constants: type-predicate registries.

Extracts the static type-predicate dispatch table from FlextUtilitiesGuards
into the constants namespace per AGENTS.md §3.1 domain separation. The dict
maps type-name strings (``"str"``, ``"list"``, ``"dict_non_empty"``, etc.) to
predicate callables consumed by ``u.matches_type`` dispatch.

Protocol-based predicates stay inside ``_utilities/guards_type_protocol.py``
because they require lazy resolution of ``p.Settings``, ``p.Context`` etc. and
cannot be evaluated at constants-class-body time without triggering import
cycles.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableSequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_core import t


class FlextConstantsGuards:
    """Static type-predicate registry for u.matches_type dispatch."""

    STRING_TYPE_PREDICATES: Final[Mapping[str, Callable[[t.GuardInput], bool]]] = (
        MappingProxyType({
            "str": lambda v: isinstance(v, str),
            "dict": lambda v: isinstance(v, dict),
            "list": lambda v: isinstance(v, list),
            "tuple": lambda v: isinstance(v, tuple),
            "sequence": lambda v: isinstance(v, (list, tuple, range)),
            "mapping": lambda v: isinstance(v, Mapping),
            "list_or_tuple": lambda v: isinstance(v, (list, tuple)),
            "sequence_not_str": lambda v: (
                isinstance(v, (list, tuple, range)) and not isinstance(v, str)
            ),
            "sequence_not_str_bytes": lambda v: (
                isinstance(v, (list, tuple, range)) and not isinstance(v, (str, bytes))
            ),
            "sized": lambda v: hasattr(v, "__len__"),
            "callable": lambda v: callable(v),
            "bytes": lambda v: isinstance(v, bytes),
            "int": lambda v: isinstance(v, int),
            "float": lambda v: isinstance(v, float),
            "bool": lambda v: isinstance(v, bool),
            "none": lambda v: v is None,
            "string_non_empty": lambda v: isinstance(v, str) and bool(v.strip()),
        })
    )


__all__: MutableSequence[str] = ["FlextConstantsGuards"]
