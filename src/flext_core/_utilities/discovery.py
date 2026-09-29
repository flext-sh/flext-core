"""Factory and service-operation discovery.

Factory discovery serves container and decorator auto-registration without
circular dependencies. Service-operation discovery (ADR-019) reads the typed
operations of a service class lazily, when a CLI or tool asks for them, and
never at class creation.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import ast
import functools
import inspect
import operator
from collections import ChainMap
from types import FunctionType
from typing import TYPE_CHECKING

from pydantic import BaseModel

from flext_core import c, p, t

from .._models.container import FlextModelsContainer
from .._models.service import FlextModelsService

if TYPE_CHECKING:
    from types import ModuleType

    from flext_core.service import FlextService


class FlextUtilitiesDiscovery:
    """Auto-discovery of @factory() functions and typed service operations."""

    @staticmethod
    def _factory_config_for(
        module: ModuleType, name: str
    ) -> FlextModelsContainer.FactoryDecoratorConfig | None:
        func = vars(module).get(name)
        if func is None or not callable(func):
            return None
        config_raw = vars(func).get(c.FACTORY_ATTR)
        if not isinstance(config_raw, FlextModelsContainer.FactoryDecoratorConfig):
            return None
        return config_raw

    @staticmethod
    def scan_module(
        module: ModuleType,
    ) -> t.SequenceOf[tuple[str, FlextModelsContainer.FactoryDecoratorConfig]]:
        """Scan module for @factory()-decorated functions, sorted by name."""
        return sorted(
            [
                (name, config)
                for name in dir(module)
                if not name.startswith("_")
                and (
                    config := FlextUtilitiesDiscovery._factory_config_for(module, name)
                )
                is not None
            ],
            key=operator.itemgetter(0),
        )

    @staticmethod
    @functools.cache
    def service_operations(
        service_type: type[FlextService[p.Base]],
    ) -> tuple[FlextModelsService.ServiceOperation, ...]:
        """Return the typed operations of a service class, sorted by name.

        An operation is a plain public function declared in a class of the MRO
        below ``FlextService``; every name ``FlextService`` exposes, Pydantic
        validators and serializers, properties, classmethods and staticmethods
        are not operations. A malformed operation, a sibling-class name
        collision, or a service without operations raises ``TypeError`` naming
        the operation, annotation, module and fix. Results are cached per class.
        """
        from flext_core import s  # s sits above u: bind it at call time

        below = service_type.__mro__[: service_type.__mro__.index(s)]
        infos = service_type.__pydantic_decorators__
        decorated = {
            *infos.field_validators,
            *infos.model_validators,
            *infos.field_serializers,
            *infos.model_serializers,
            *infos.computed_fields,
        }
        names = (
            {
                name
                for owner in below
                for name in vars(owner)
                if not name.startswith("_")
            }
            - set(dir(s))
            - decorated
        )
        operations = tuple(
            FlextUtilitiesDiscovery._operation(service_type, below, name, member)
            for name in sorted(names)
            if isinstance(
                member := inspect.getattr_static(service_type, name), FunctionType
            )
        )
        if not operations:
            msg = c.ERR_SERVICE_NO_OPERATIONS.format(
                service=service_type.__qualname__, module=service_type.__module__
            )
            raise TypeError(msg)
        return operations

    @staticmethod
    def _operation(
        service_type: type, below: tuple[type, ...], name: str, func: FunctionType
    ) -> FlextModelsService.ServiceOperation:
        """Validate one operation's shape and build its typed description."""
        where = (service_type, name, func.__module__)
        error = FlextUtilitiesDiscovery._error
        signature = inspect.signature(func)
        owners = [owner for owner in below if name in vars(owner)]
        if any(not issubclass(owners[0], owner) for owner in owners[1:]):
            siblings = ", ".join(owner.__qualname__ for owner in owners)
            defect = c.ERR_SERVICE_OPERATION_COLLISION.format(owners=siblings)
            raise error(where, signature, defect)
        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            raise error(where, signature, c.ERR_SERVICE_OPERATION_ASYNC)
        if func.__type_params__:
            raise error(where, signature, c.ERR_SERVICE_OPERATION_GENERIC)
        params = tuple(signature.parameters.values())
        requests = params[1:]
        if (
            not params
            or len(requests) > 1
            or any(
                param.kind is not param.POSITIONAL_OR_KEYWORD
                or param.default is not param.empty
                for param in params
            )
        ):
            raise error(where, signature, c.ERR_SERVICE_OPERATION_SIGNATURE)
        doc = func.__doc__
        if doc is None or not doc.strip():
            raise error(where, signature, c.ERR_SERVICE_OPERATION_DOCSTRING)
        annotations = inspect.get_annotations(func)
        if "return" not in annotations:
            raise error(where, signature, c.ERR_SERVICE_OPERATION_RESULT)
        returned = annotations["return"]
        origin = FlextUtilitiesDiscovery._resolve(func, where, returned, subscript=True)
        if origin is not p.Result:
            raise error(where, returned, c.ERR_SERVICE_OPERATION_RESULT)
        request = None
        for param in requests:
            if param.name not in annotations:
                raise error(where, signature, c.ERR_SERVICE_OPERATION_REQUEST)
            annotation = annotations[param.name]
            request = FlextUtilitiesDiscovery._resolve(func, where, annotation)
            if not (isinstance(request, type) and issubclass(request, BaseModel)):
                raise error(where, annotation, c.ERR_SERVICE_OPERATION_REQUEST)
        return FlextModelsService.ServiceOperation(
            name=name, summary=doc.strip().splitlines()[0], request=request
        )

    @staticmethod
    def _resolve(
        func: FunctionType,
        where: tuple[type, str, str],
        annotation: t.TypeHintSpecifier,
        *,
        subscript: bool = False,
    ) -> t.TypeHintSpecifier:
        """Resolve an annotation to its object without ``eval``.

        The module declares ``from __future__ import annotations`` (fleet law),
        so every annotation is a string: it is parsed, never evaluated. Its
        dotted name resolves the first part in the function's module namespace
        (globals, then builtins, as Python resolves a module-level name) and the
        rest by attribute. ``subscript`` resolves the origin of ``X[...]``.
        """
        error = FlextUtilitiesDiscovery._error
        if not isinstance(annotation, str):
            raise error(where, annotation, c.ERR_SERVICE_OPERATION_EVALUATED)
        node = ast.parse(annotation, mode="eval").body
        if subscript:
            if not isinstance(node, ast.Subscript):
                raise error(where, annotation, c.ERR_SERVICE_OPERATION_RESULT)
            node = node.value
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.insert(0, node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            raise error(where, annotation, c.ERR_SERVICE_OPERATION_NAME)
        namespace = ChainMap(func.__globals__, func.__builtins__)
        if node.id not in namespace:
            raise error(where, annotation, c.ERR_SERVICE_OPERATION_UNBOUND)
        resolved = namespace[node.id]
        for part in parts:
            try:
                resolved = getattr(resolved, part)
            except AttributeError as exc:
                defect = c.ERR_SERVICE_OPERATION_UNBOUND
                raise error(where, annotation, defect) from exc
        return resolved

    @staticmethod
    def _error(
        where: tuple[type, str, str],
        annotation: t.TypeHintSpecifier | inspect.Signature,
        defect: str,
    ) -> TypeError:
        """Build the operation's ``TypeError`` naming annotation, module and fix."""
        service_type, operation, module = where
        return TypeError(
            c.ERR_SERVICE_OPERATION.format(
                service=service_type.__qualname__,
                operation=operation,
                module=module,
                annotation=annotation,
                defect=defect,
            )
        )


__all__: list[str] = ["FlextUtilitiesDiscovery"]
