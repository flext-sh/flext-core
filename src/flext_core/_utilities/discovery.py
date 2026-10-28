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
import warnings
from collections import ChainMap
from collections.abc import Sequence
from types import FunctionType
from typing import TYPE_CHECKING, TypeAliasType, get_protocol_members, is_protocol

from pydantic import BaseModel

from flext_core import c, m, p, t
from flext_core._utilities import FlextUtilitiesModelRuntime

if TYPE_CHECKING:
    from types import ModuleType


class FlextUtilitiesDiscovery:
    """Auto-discovery of @factory() functions and typed service operations."""

    @staticmethod
    def _factory_config_for(
        module: ModuleType,
        name: str,
    ) -> m.FactoryDecoratorConfig | None:
        func = vars(module).get(name)
        if func is None or not callable(func):
            return None
        config_raw = vars(func).get(c.FACTORY_ATTR)
        if not isinstance(config_raw, m.FactoryDecoratorConfig):
            return None
        return config_raw

    @staticmethod
    def scan_module(
        module: ModuleType,
    ) -> Sequence[tuple[str, m.FactoryDecoratorConfig]]:
        """Scan module for @factory()-decorated functions, sorted by name.

        Returns:
            The resulting ``Sequence[tuple[str,
                m.FactoryDecoratorConfig]]``.

        """
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
        service_type: type[p.Service[p.Base]],
        *,
        owner: type | None = None,
    ) -> tuple[p.ServiceOperation, ...]:
        """Return the typed operations of a service class, sorted by name.

        An operation is a plain public function declared in a class of the MRO
        below ``FlextService``; an explicit owner selects an inherited operation
        family from that same MRO. Every name ``FlextService`` exposes, Pydantic
        validators and serializers, properties, classmethods and staticmethods
        are not operations. A malformed operation, a sibling-class name
        collision, or a service without operations raises ``TypeError`` naming
        the operation, annotation, module and fix. Results are cached per class.

        Returns:
            The typed operations of a service class, sorted by name.

        Raises:
            TypeError: If the class is not a service or declares no operation.

        """
        service_facade = FlextUtilitiesDiscovery._service_root(service_type)
        below = service_type.__mro__[: service_type.__mro__.index(service_facade)]
        if owner is not None and owner not in below:
            message = (
                f"Operation owner {owner.__qualname__} is not inherited by "
                f"{service_type.__qualname__}"
            )
            raise TypeError(message)
        selected = (
            below
            if owner is None
            else tuple(base for base in owner.__mro__ if base in below)
        )
        infos = inspect.getattr_static(service_type, "__pydantic_decorators__")
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
                for operation_owner in selected
                for name in vars(operation_owner)
                if not name.startswith("_")
            }
            - set(dir(service_facade))
            - decorated
        )
        operations = tuple(
            FlextUtilitiesDiscovery._operation(service_type, below, name, member)
            for name in sorted(names)
            if isinstance(
                member := inspect.getattr_static(service_type, name),
                FunctionType,
            )
        )
        if not operations:
            msg = c.ERR_SERVICE_NO_OPERATIONS.format(
                service=service_type.__qualname__,
                module=service_type.__module__,
            )
            raise TypeError(msg)
        return operations

    @staticmethod
    def _service_root(service_type: type) -> type:
        """Return the MRO class that first declares the service's own members.

        The members are the ones ``p.Service`` declares beyond its bases
        (``execute``), so the root is derived from the protocol, never from a
        module name. Operations are declared only below it.

        Returns:
            The base-most MRO class declaring every own ``p.Service`` member.

        Raises:
            TypeError: If no MRO class declares them.

        """
        members = get_protocol_members(p.Service) & vars(p.Service).keys()
        roots = [
            klass
            for klass in service_type.__mro__
            if not is_protocol(klass) and members <= vars(klass).keys()
        ]
        if not roots:
            msg = c.ERR_SERVICE_NOT_A_SERVICE.format(
                service=service_type.__qualname__,
                module=service_type.__module__,
                members=", ".join(sorted(members)),
            )
            raise TypeError(msg)
        return roots[-1]

    @staticmethod
    def _operation(
        service_type: type,
        below: tuple[type, ...],
        name: str,
        func: FunctionType,
    ) -> p.ServiceOperation:
        """Validate one operation's shape and build its typed description.

        Returns:
            The resulting ``m.ServiceOperation``.

        """
        where = (service_type, name, func.__module__)
        signature = inspect.signature(func)
        FlextUtilitiesDiscovery._validate_operation_shape(func, below, where, signature)
        doc = FlextUtilitiesDiscovery._require_operation_docstring(
            func,
            where,
            signature,
        )
        request = FlextUtilitiesDiscovery._resolve_operation_request(
            func,
            where,
            signature,
        )
        return m.ServiceOperation(
            name=name,
            summary=doc.strip().splitlines()[0],
            request=request,
            handler=func,
        )

    @staticmethod
    def _validate_operation_shape(
        func: FunctionType,
        below: tuple[type, ...],
        where: tuple[type, str, str],
        signature: inspect.Signature,
    ) -> None:
        """Raise on shape defects: collisions, async, generics, signatures."""
        error = FlextUtilitiesDiscovery._error
        owners = [owner for owner in below if func.__name__ in vars(owner)]
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

    @staticmethod
    def _require_operation_docstring(
        func: FunctionType,
        where: tuple[type, str, str],
        signature: inspect.Signature,
    ) -> str:
        """Return the operation docstring, raising when missing or blank.

        Returns:
            The resulting ``str``.

        """
        error = FlextUtilitiesDiscovery._error
        doc = func.__doc__
        if doc is None or not doc.strip():
            raise error(
                where,
                signature,
                c.ERR_SERVICE_OPERATION_DOCSTRING,
            )
        return doc

    @staticmethod
    def _resolve_operation_request(
        func: FunctionType,
        where: tuple[type, str, str],
        signature: inspect.Signature,
    ) -> t.ModelClass[t.BaseModelType] | None:
        """Resolve a request Protocol and its single explicit model declaration.

        Returns:
            The resulting ``type | None``.

        """
        error = FlextUtilitiesDiscovery._error
        annotations = inspect.get_annotations(func)
        if "return" not in annotations:
            raise error(where, signature, c.ERR_SERVICE_OPERATION_RESULT)
        returned = annotations["return"]
        origin = FlextUtilitiesDiscovery._resolve(func, where, returned, subscript=True)
        if origin is not p.Result:
            raise error(where, returned, c.ERR_SERVICE_OPERATION_RESULT)
        metadata = FlextUtilitiesModelRuntime.handler_metadata(
            func,
            name=f"{where[2]}.{where[0].__qualname__}.{where[1]}",
        )
        if metadata is None and hasattr(where[0], "__mro__"):
            for base in where[0].__mro__[1:]:
                base_func = inspect.getattr_static(base, where[1], None)
                if isinstance(base_func, FunctionType):
                    base_meta = FlextUtilitiesModelRuntime.handler_metadata(
                        base_func,
                        name=f"{where[2]}.{base.__qualname__}.{where[1]}",
                    )
                    if base_meta is not None:
                        metadata = base_meta
                        break
        params = tuple(signature.parameters.values())[1:]
        if not params:
            if metadata is not None:
                raise error(where, signature, c.ERR_SERVICE_OPERATION_INPUTLESS_COMMAND)
            return None
        param = params[0]
        if param.name not in annotations:
            raise error(where, signature, c.ERR_SERVICE_OPERATION_REQUEST)
        annotation = annotations[param.name]
        contract = FlextUtilitiesDiscovery._resolve(func, where, annotation)
        if not is_protocol(contract):
            raise error(where, annotation, c.ERR_SERVICE_OPERATION_REQUEST)
        if metadata is None:
            raise error(where, annotation, c.ERR_SERVICE_OPERATION_METADATA)
        request = metadata.command
        if not issubclass(request, BaseModel):
            raise error(where, annotation, c.ERR_SERVICE_OPERATION_COMMAND_MODEL)
        missing = get_protocol_members(contract) - (
            request.model_fields.keys() | set(dir(request))
        )
        if missing:
            defect = c.ERR_SERVICE_OPERATION_PROTOCOL.format(
                missing=", ".join(sorted(missing))
            )
            raise error(where, annotation, defect)
        return request

    @staticmethod
    @warnings.deprecated(
        "compat: stdlib_ast_annotation_parse -> typing.get_type_hints; "
        "fix: rewire via make mod; this site needs adjustment to the correct "
        "FLEXT form: annotation analysis runs through the typing surface, "
        "never a hand-held stdlib ast parse (engine-map D1).",
    )
    def _annotation_node(
        annotation: t.TypeHintSpecifier,
        where: tuple[type, str, str],
        *,
        subscript: bool,
    ) -> ast.expr:
        """Parse a source annotation and select its requested outer node.

        Returns:
            The annotation expression to resolve.

        """
        error = FlextUtilitiesDiscovery._error
        if not isinstance(annotation, str):
            raise error(where, annotation, c.ERR_SERVICE_OPERATION_EVALUATED)
        node = ast.parse(annotation, mode="eval").body
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node = ast.parse(node.value, mode="eval").body
        if not subscript:
            return node
        if not isinstance(node, ast.Subscript):
            raise error(where, annotation, c.ERR_SERVICE_OPERATION_RESULT)
        return node.value

    @staticmethod
    @warnings.deprecated(
        "compat: stdlib_ast_annotation_walk -> typing.get_type_hints; "
        "fix: rewire via make mod; this site needs adjustment to the correct "
        "FLEXT form: annotations resolve through the typing surface, never "
        "manual ast node walking (engine-map D1).",
    )
    def _resolve(
        func: FunctionType,
        where: tuple[type, str, str],
        annotation: t.TypeHintSpecifier,
        *,
        subscript: bool = False,
    ) -> type:
        """Resolve an annotation to its object without ``eval``.

        The module declares ``from __future__ import annotations`` (fleet law),
        so every annotation is a string: it is parsed, never evaluated. Its
        dotted name resolves the first part in the function's module namespace
        (globals, then builtins, as Python resolves a module-level name) and the
        rest by attribute. ``subscript`` resolves the origin of ``X[...]``.

        Returns:
            The resolved class consumed by request and result validation.

        """
        error = FlextUtilitiesDiscovery._error
        node = FlextUtilitiesDiscovery._annotation_node(
            annotation, where, subscript=subscript
        )
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.insert(0, node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            raise error(where, annotation, c.ERR_SERVICE_OPERATION_NAME)
        # FunctionType owns this runtime descriptor, absent from its typing stub.
        namespace = ChainMap(
            func.__globals__,
            operator.attrgetter("__builtins__")(func),
        )
        if node.id not in namespace:
            raise error(where, annotation, c.ERR_SERVICE_OPERATION_UNBOUND)
        resolved = namespace[node.id]
        for part in parts:
            try:
                resolved = getattr(resolved, part)
            except AttributeError as exc:
                defect = c.ERR_SERVICE_OPERATION_UNBOUND
                raise error(where, annotation, defect) from exc
        if isinstance(resolved, TypeAliasType):
            resolved = resolved.__value__
        if not isinstance(resolved, type):
            defect = (
                c.ERR_SERVICE_OPERATION_RESULT
                if subscript
                else c.ERR_SERVICE_OPERATION_REQUEST
            )
            raise error(where, annotation, defect)
        return resolved

    @staticmethod
    def _error(
        where: tuple[type, str, str],
        annotation: t.TypeHintSpecifier | inspect.Signature,
        defect: str,
    ) -> TypeError:
        """Build the operation's ``TypeError`` naming annotation, module and fix.

        Returns:
            The resulting ``TypeError``.

        """
        service_type, operation, module = where
        return TypeError(
            c.ERR_SERVICE_OPERATION.format(
                service=service_type.__qualname__,
                operation=operation,
                module=module,
                annotation=annotation,
                defect=defect,
            ),
        )


__all__: list[str] = ["FlextUtilitiesDiscovery"]
