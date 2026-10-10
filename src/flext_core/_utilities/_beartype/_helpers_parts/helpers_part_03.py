"""Type and module introspection helpers — annotation inspection + bytecode analysis.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import types as _types_mod
from collections.abc import Callable
from types import UnionType
from typing import TYPE_CHECKING, Union, get_args, get_origin

from flext_core._constants.enforcement import FlextConstantsEnforcement as c
from flext_core._utilities._beartype._helpers_parts.helpers_part_02 import (
    FlextUtilitiesBeartypeHelpers as FlextUtilitiesBeartypeHelpersPart02,
)

# Import directly from base modules to avoid a circular load through the public
# flext_core facade while this module is still being initialized.

if TYPE_CHECKING:
    from flext_core._protocols import FlextProtocolsBase as p
    from flext_core._typings.base import FlextTypingBase as t


class FlextUtilitiesBeartypeHelpers(FlextUtilitiesBeartypeHelpersPart02):
    @staticmethod
    def module_filename_for(module: _types_mod.ModuleType) -> str | None:
        filename = getattr(module, "__file__", None)
        return filename if isinstance(filename, str) else None

    @staticmethod
    def object_module_for(obj: p.AttributeProbe) -> _types_mod.ModuleType | None:
        module = inspect.getmodule(obj)
        return module if isinstance(module, _types_mod.ModuleType) else None

    @staticmethod
    def object_module_name_for(obj: p.AttributeProbe) -> str | None:
        module = inspect.getmodule(obj)
        name = getattr(module, "__name__", None)
        return name if isinstance(name, str) else None

    @staticmethod
    def count_union_members(hint: t.TypeHintSpecifier | None) -> int:
        h = FlextUtilitiesBeartypeHelpers
        h2 = h.unwrap_type_alias(hint)
        if h2 is None or get_origin(h2) not in {UnionType, Union}:
            return 0
        return sum(1 for a in get_args(h2) if a is not type(None))

    @staticmethod
    def matches_str_none_union(hint: t.TypeFormSpecifier | None) -> bool:
        h = FlextUtilitiesBeartypeHelpers
        h2 = h.unwrap_type_alias(hint)
        if h2 is None or get_origin(h2) not in {UnionType, Union}:
            return False
        return str in (a := get_args(h2)) and type(None) in a

    @staticmethod
    def alias_contains_any(
        alias_value: t.TypeFormSpecifier | None,
        *,
        owner: _types_mod.ModuleType | type | None = None,
    ) -> bool:
        h = FlextUtilitiesBeartypeHelpers
        return h.contains_any_recursive(alias_value, seen=set(), owner=owner)

    @staticmethod
    def mutable_kind(value: p.AttributeProbe) -> str | None:
        for kind in c.ENFORCEMENT_MUTABLE_RUNTIME_TYPES:
            if isinstance(value, kind):
                return kind.__name__
        return None

    @staticmethod
    def raw_collection_factory_kind(
        factory: type | Callable[..., p.AttributeProbe] | None,
    ) -> type | None:
        """Return the raw collection constructor ``factory`` is, if any.

        Returns:
            The bare or specialized collection constructor, else ``None``.

        """
        for kind in c.ENFORCEMENT_RAW_COLLECTION_FACTORIES:
            if factory is kind or get_origin(factory) is kind:
                return kind
        return None

    @staticmethod
    def has_relaxed_extra_base(target: type) -> bool:
        return any(
            b.__name__ in c.ENFORCEMENT_RELAXED_EXTRA_BASES for b in target.__mro__
        )


__all__: list[str] = ["FlextUtilitiesBeartypeHelpers"]
