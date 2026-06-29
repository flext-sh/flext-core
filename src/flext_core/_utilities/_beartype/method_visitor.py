"""Method naming + static method enforcement via bytecode introspection."""

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
        """
        if len(args) != _BINARY_ARITY:
            return _NO_VIOLATION
        suggestions = (
            ("get_", "fetch_/resolve_/compute_"),
            ("set_", "configure/apply/update or model_copy(update=...)"),
            ("is_", "a noun/adjective (success, expired, connected, ...)"),
        )
        violation = _NO_VIOLATION
        match args:
            case (_, name) if isinstance(name, str) and isinstance(args[0], type):
                if not name.startswith("_"):
                    violation = next(
                        (
                            {"name": name, "suggestion": suggestion}
                            for prefix in params.forbidden_prefixes
                            for known_prefix, suggestion in suggestions
                            if prefix == known_prefix and name.startswith(prefix)
                        ),
                        next(
                            (
                                {"name": name, "suggestion": "use a domain verb"}
                                for prefix in params.forbidden_prefixes
                                if name.startswith(prefix)
                            ),
                            _NO_VIOLATION,
                        ),
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
