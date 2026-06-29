"""Handler registration and discovery utilities.

Provides handler registration with binding, tracking, and batch operations
for the FLEXT dispatcher system.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import inspect
import sys
from collections.abc import (
    MutableMapping,
)
from typing import Annotated, ClassVar, Self, override

from pydantic import PrivateAttr

from flext_core import h, m, p, s, t, u


class FlextRegistry(s[bool]):
    """Application-layer registry for CQRS handlers.

    Extends s for automatic infrastructure (settings, context,
    container, logging) while providing handler registration and management
    capabilities. The registry pairs message types with handlers, enforces
    idempotent registration, and exposes batch operations that return ``r``
    summaries.

    It delegates to ``FlextDispatcher`` (which implements ``p.Dispatcher``)
    for actual handler registration and execution.
    """

    _state: m.RegistryState = PrivateAttr(default_factory=lambda: m.RegistryState())

    _class_plugin_storage: ClassVar[MutableMapping[str, t.RegistrablePlugin]] = {}

    _class_registered_keys: ClassVar[set[str]] = set()

    dispatcher: Annotated[
        p.Dispatcher | None,
        m.Field(
            exclude=True,
            description="The dispatcher instance for executing handlers.",
        ),
    ] = None

    @override
    def model_post_init(self, __context: t.ScalarMapping | None, /) -> None:
        """Post-initialization hook for registry.

        Initializes dispatcher state without triggering recursive runtime
        build (registry IS part of the runtime triple — building it here
        would recurse via ``build_service_runtime → build_registry``).
        """
        super().model_post_init(__context)
        resolved_dispatcher = (
            self.dispatcher if isinstance(self.dispatcher, p.Dispatcher) else None
        )
        self._state = m.RegistryState(dispatcher=resolved_dispatcher)

    def __init_subclass__(
        cls,
        **kwargs: t.Scalar | m.ConfigMap | t.ScalarList,
    ) -> None:
        """Auto-create per-subclass class-level storage.

        Each subclass gets its OWN storage (not shared with parent or siblings).
        This enables auto-discovery patterns where plugins registered via
        register_plugin(..., scope="class") are visible across all instances of that
        subclass.
        """
        super().__init_subclass__()
        cls._class_plugin_storage = {}  # MutableMapping[str, t.RegistrablePlugin]
        cls._class_registered_keys = set()  # set[str]

    @classmethod
    def create(
        cls,
        dispatcher: p.Dispatcher | None = None,
        *,
        runtime: m.ServiceRuntime | None = None,
        auto_discover_handlers: bool = False,
    ) -> Self:
        """Factory method to create a new FlextRegistry instance.

        This is the preferred way to instantiate FlextRegistry. It provides
        a clean factory pattern that each class owns, respecting Clean
        Architecture principles where higher layers create their own instances.

        Auto-discovery of handlers discovers all functions marked with
        @h.handler() decorator in the calling module and auto-registers them
        with built-in deduplication. This enables zero-settings handler
        registration for services with idempotent tracking.

        Args:
            dispatcher: Optional CommandBus instance (defaults to DSL dispatcher)
            runtime: Optional runtime snapshot whose container/context are reused
            auto_discover_handlers: If True, scan calling module for @handler()
                decorated functions and auto-register them with deduplication.
                Default: False.

        Returns:
            FlextRegistry instance with auto-discovered handlers if enabled.

        """
        if runtime is None:
            instance = cls(dispatcher=dispatcher or u.build_dispatcher())
        else:
            resolved = (
                dispatcher
                if isinstance(dispatcher, p.Dispatcher)
                else runtime.dispatcher
            )
            instance = cls(
                initial_context=runtime.context, dispatcher=resolved
            ).configure_runtime(runtime, dispatcher=resolved)
        if auto_discover_handlers:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_globals = frame.f_back.f_globals
                module_name = caller_globals.get("__name__", "__main__")
                caller_module = sys.modules.get(module_name)
                if caller_module:
                    handlers = h.Discovery.scan_module(caller_module)
                    for _handler_name, handler_func, _handler_config in handlers:
                        _ = handler_func
        return instance

    def configure_runtime(
        self,
        runtime: m.ServiceRuntime,
        *,
        dispatcher: p.Dispatcher | None = None,
    ) -> Self:
        """Bind this registry to a pre-built runtime snapshot."""
        resolved_dispatcher = (
            dispatcher if isinstance(dispatcher, p.Dispatcher) else runtime.dispatcher
        )
        self.dispatcher = resolved_dispatcher
        self._state = self._state.model_copy(update={"dispatcher": resolved_dispatcher})
        self._runtime = runtime.model_copy(
            update={
                "dispatcher": resolved_dispatcher,
                "registry": self,
            },
        )
        return self

    def _create_initial_runtime(self) -> m.ServiceRuntime:
        """Build the registry runtime without recursively materializing another registry."""
        return u.build_service_runtime(self, registry=self)


__all__: list[str] = ["FlextRegistry"]
