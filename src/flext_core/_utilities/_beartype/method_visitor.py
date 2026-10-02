"""Method naming + static method enforcement via runtime introspection.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import types as _types_mod

from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t

_NO_VIOLATION: t.StrMapping | None = None
_BARE_VIOLATION: t.StrMapping = {}
_BINARY_ARITY: int = 2


class FlextUtilitiesBeartypeMethodVisitor:
    """METHOD_SHAPE — accessor-prefix and staticmethod-required governance."""

    @staticmethod
    def v_method_shape(
        params: me.MethodShapeParams,
        *args: type | str | _types_mod.FunctionType,
    ) -> t.StrMapping | None:
        """METHOD_SHAPE — accessor-prefix and staticmethod-required governance.

        Args shape varies: ``(target, name)`` for accessor checks (NAMESPACE
        category); ``(name, value)`` for utility-tier static-method checks
        (ATTR category).

        Returns:
            The resulting ``t.StrMapping | None``.

        """
        if len(args) != _BINARY_ARITY:
            return _NO_VIOLATION
        violation = _NO_VIOLATION
        match args:
            case (_, name) if isinstance(name, str) and isinstance(args[0], type):
                if not name.startswith("_"):
                    violation = next(
                        (
                            {"name": name, "suggestion": suggestion}
                            for prefix, suggestion in params.forbidden_prefixes.items()
                            if name.startswith(prefix)
                        ),
                        _NO_VIOLATION,
                    )
            case (name, value) if isinstance(name, str) and not isinstance(value, type):
                if all((
                    params.require_static_or_classmethod,
                    not isinstance(value, (staticmethod, classmethod)),
                    inspect.isfunction(value),
                )):
                    violation = _BARE_VIOLATION
            case _:
                pass
        return violation
