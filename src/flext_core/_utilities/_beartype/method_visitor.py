"""Method naming + static method enforcement via bytecode introspection."""

from __future__ import annotations

import inspect

from flext_core._models.enforcement import FlextModelsEnforcement as me
from flext_core._typings.base import FlextTypingBase as t

_NO_VIOLATION: t.StrMapping | None = None
_BARE_VIOLATION: t.StrMapping = {}
_BINARY_ARITY: int = 2


class FlextUtilitiesBeartypeMethodVisitor:
    """METHOD_SHAPE — accessor-prefix and staticmethod-required governance."""

    @staticmethod
    def _v_method_shape(
        params: me.MethodShapeParams,
        *args: object,
    ) -> t.StrMapping | None:
        """METHOD_SHAPE — accessor-prefix and staticmethod-required governance.

        Args shape varies: ``(target, name)`` for accessor checks (NAMESPACE
        category); ``(name, value)`` for utility-tier static-method checks
        (ATTR category).
        """
        if len(args) != _BINARY_ARITY:
            return _NO_VIOLATION
        a, b = args
        if isinstance(a, type) and isinstance(b, str):
            name = b
            if name.startswith("_"):
                return _NO_VIOLATION
            suggestions = (
                ("get_", "fetch_/resolve_/compute_"),
                ("set_", "configure/apply/update or model_copy(update=...)"),
                ("is_", "a noun/adjective (success, expired, connected, ...)"),
            )
            for prefix in params.forbidden_prefixes:
                suggestion = next(
                    (s for p, s in suggestions if p == prefix),
                    "use a domain verb",
                )
                if name.startswith(prefix):
                    return {"name": name, "suggestion": suggestion}
            return _NO_VIOLATION
        if isinstance(a, str) and not isinstance(b, type):
            if not params.require_static_or_classmethod:
                return _NO_VIOLATION
            if isinstance(b, (staticmethod, classmethod)):
                return _NO_VIOLATION
            if inspect.isfunction(b):
                return _BARE_VIOLATION
        return _NO_VIOLATION
