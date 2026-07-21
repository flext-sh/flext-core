"""CQRS handler foundation used by the dispatcher pipeline.

h defines the base class the dispatcher relies on for commands,
queries, and domain events. It favors structural typing over inheritance,
ensures validation and execution steps return ``r`` rather than
raising, and keeps handler metadata ready for registry/dispatcher discovery.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Unpack

from flext_core import c, e, p, x
from flext_core._utilities.handler import FlextUtilitiesHandler

if TYPE_CHECKING:
    from pydantic import ConfigDict


class FlextHandlers[MessageT_contra, ResultT](x):
    """Abstract CQRS handler with validation and railway-style execution.

    Provides the base implementation for Command Query Responsibility Segregation
    (CQRS) handlers, implementing structural typing via p.Handler[MessageT_contra]
    through duck typing (no inheritance required). This class serves as the foundation
    for implementing command, query, and event handlers with comprehensive validation,
    execution pipelines, metrics collection, and configuration management.
    """

    _expected_message_type: ClassVar[type | None] = None

    _expected_result_type: ClassVar[type | None] = None

    def __init__(self, *, settings: p.HandlerConfig | None = None) -> None:
        """Initialize handler with configuration and context.

        Sets up the handler with optional configuration parameters.
        The settings parameter accepts a m instance.

        Args:
            settings: Optional handler configuration model

        """
        super().__init__(
            settings_type=None, settings_overrides=None, initial_context=None
        )
        if settings is not None:
            self._config_model = settings
        else:
            from flext_core import m

            self._config_model = m.Handler(
                handler_id=f"handler_{id(self)}", handler_name=self.__class__.__name__
            )
        handler_type = self._config_model.handler_mode
        valid_handler_types = {
            c.HandlerType.COMMAND,
            c.HandlerType.QUERY,
            c.HandlerType.EVENT,
            c.HandlerType.OPERATION,
            c.HandlerType.SAGA,
        }
        if handler_type not in valid_handler_types:
            error_msg = c.ERR_HANDLER_INVALID_MODE.format(mode=handler_type)
            raise e.ValidationError(error_msg)
        handler_mode_literal = self._handler_type_to_literal(handler_type)
        self._runtime_state: p.HandlerRuntimeState = (
            FlextUtilitiesHandler.create_runtime_state(
                handler_name=self._config_model.handler_name,
                handler_mode=handler_mode_literal,
            )
        )

    def __init_subclass__(cls, **kwargs: Unpack[ConfigDict]) -> None:
        """Validate non-abstract subclasses implement a handle() method.

        Chains with FlextMixins.__init_subclass__ via super() to preserve
        MRO-based container auto-initialization. Skips validation for
        abstract subclasses (intermediate bases).

        Raises:
            TypeError: If a concrete subclass does not override handle().

        """
        _ = kwargs
        super().__init_subclass__()
        if cls.__module__.startswith("flext_core._handlers_parts"):
            return
        if "[" in cls.__qualname__:
            return
        abstract_methods_default: frozenset[str] = frozenset()
        abstract_methods = getattr(cls, "__abstractmethods__", abstract_methods_default)
        if abstract_methods:
            return
        for klass in cls.mro():
            if klass is FlextHandlers:
                msg = c.ERR_HANDLER_MISSING_HANDLE_IMPLEMENTATION.format(
                    qualname=cls.__qualname__
                )
                raise TypeError(msg)
            if c.MethodName.HANDLE in klass.__dict__:
                break

    @property
    def handler_name(self) -> str:
        """Handler name from configuration.

        Returns:
            str: The handler name

        """
        return self._runtime_state.execution_context.handler_name

    @property
    def mode(self) -> c.HandlerType:
        """Handler mode from configuration.

        Returns:
            c.HandlerType: The handler mode (command, query, event, saga)

        """
        return self._runtime_state.execution_context.handler_mode

    @staticmethod
    def _handler_type_to_literal(handler_type: c.HandlerType | str) -> c.HandlerType:
        """Coerce string or StrEnum to canonical HandlerType."""
        if isinstance(handler_type, c.HandlerType):
            return handler_type
        for member in c.HandlerType:
            if member.value == handler_type:
                return member
        raise TypeError(
            c.ERR_HANDLER_UNSUPPORTED_TYPE.format(handler_type=handler_type)
        )


__all__: list[str] = ["FlextHandlers"]
