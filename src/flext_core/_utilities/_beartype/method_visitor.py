"""Method naming + static method enforcement via bytecode introspection."""

from __future__ import annotations

import inspect
import types as _types_mod

from ..._constants.enforcement import FlextConstantsEnforcement as c
from ..._models.enforcement import FlextModelsEnforcement as me
from ..._typings.base import FlextTypingBase as t

_NO_VIOLATION: t.StrMapping | None = None
_BARE_VIOLATION: t.StrMapping = {}
_BINARY_ARITY: int = 2


def _param_count(name: str, value: object) -> int | None:
    """Return effective parameter count for ``value`` or None if exempt."""
    if name.startswith("__") and name.endswith("__"):
        return None
    if name.startswith("model_"):
        return None
    func = None
    offset = 0
    if isinstance(value, (staticmethod, classmethod)):
        func = value.__func__
    elif inspect.isfunction(value):
        func = value
    elif inspect.ismethod(value):
        func = value.__func__
        offset = 1
    if func is None:
        return None
    code = getattr(func, "__code__", None)
    if not isinstance(code, _types_mod.CodeType):
        return None
    if offset == 0 and code.co_argcount > 0 and code.co_varnames[0] in {"self", "cls"}:
        offset = 1
    return code.co_argcount + code.co_kwonlyargcount - offset


class FlextUtilitiesBeartypeMethodVisitor:
    """METHOD_SHAPE — accessor-prefix and staticmethod-required governance."""

    @staticmethod
    def v_method_shape(
        params: me.MethodShapeParams, *args: type | str | _types_mod.FunctionType
    ) -> t.StrMapping | None:
        """METHOD_SHAPE — accessor-prefix, staticmethod-required, and param-cap governance.

        Args shape varies: ``(target, name)`` for accessor checks (NAMESPACE
        category); ``(name, value)`` for utility-tier static-method checks
        (ATTR category).
        """
        if len(args) != _BINARY_ARITY:
            return _NO_VIOLATION
        max_params = (
            c.SMELL_THRESHOLDS[params.smell_threshold] if params.smell_threshold else 0
        )
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
                if violation is _NO_VIOLATION and max_params > 0:
                    target = args[0]
                    value = vars(target).get(name)
                    count = _param_count(name, value)
                    if count is not None and count > max_params:
                        violation = {
                            "name": name,
                            "count": str(count),
                            "max": str(max_params),
                        }
            case (name, value) if isinstance(name, str) and not isinstance(value, type):
                if all((
                    params.require_static_or_classmethod,
                    not isinstance(value, (staticmethod, classmethod)),
                    inspect.isfunction(value),
                )):
                    violation = _BARE_VIOLATION
                if violation is _NO_VIOLATION and max_params > 0:
                    count = _param_count(name, value)
                    if count is not None and count > max_params:
                        violation = {
                            "name": name,
                            "count": str(count),
                            "max": str(max_params),
                        }
            case _:
                pass
        return violation
