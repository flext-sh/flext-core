"""Base loader for the typed Rope runtime boundary."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from types import ModuleType

from flext_infra.protocols import p


class FlextInfraUtilitiesRopeRuntimeBase:
    """Resolve Rope runtime attributes without importing Rope in leaf modules."""

    @classmethod
    def _module(cls, module_name: str) -> ModuleType:
        return import_module(module_name)

    @classmethod
    def runtime_type(cls, module_name: str, attribute: str) -> type[p.AttributeProbe]:
        candidate: type[p.AttributeProbe] | p.AttributeProbe = getattr(
            cls._module(module_name), attribute
        )
        if not isinstance(candidate, type):
            msg = f"rope attribute is not a type: {module_name}.{attribute}"
            raise TypeError(msg)
        return candidate

    @classmethod
    def _runtime_callable(
        cls, module_name: str, attribute: str
    ) -> Callable[..., p.AttributeProbe]:
        candidate: Callable[..., p.AttributeProbe] | p.AttributeProbe = getattr(
            cls._module(module_name), attribute
        )
        if not callable(candidate):
            msg = f"rope attribute is not callable: {module_name}.{attribute}"
            raise TypeError(msg)
        return candidate

    @classmethod
    def _exception_type(cls, module_name: str, attribute: str) -> type[BaseException]:
        candidate = cls.runtime_type(module_name, attribute)
        if not issubclass(candidate, BaseException):
            msg = f"rope attribute is not an exception type: {module_name}.{attribute}"
            raise TypeError(msg)
        return candidate


__all__: list[str] = ["FlextInfraUtilitiesRopeRuntimeBase"]
