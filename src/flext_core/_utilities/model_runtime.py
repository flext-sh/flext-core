"""Runtime registry and settings-election utilities.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from inspect import getattr_static
from types import MethodType

import flext_core
from flext_core._utilities import FlextUtilitiesModel, FlextUtilitiesModelOptions


class FlextUtilitiesModelRuntime(
    FlextUtilitiesModelOptions,
    flext_core.FlextModels,
):
    """Runtime DSL: registry comparison, plugin keys and settings election.

    Dispatcher, registry and service-runtime construction belong to their
    owner classes (``FlextDispatcher``, ``FlextRegistry.create``,
    ``FlextContainer.build_service_runtime``), never to this layer.
    """

    @staticmethod
    def registry_same_value(
        previous: flext_core.p.RegisterableService | flext_core.p.DispatchableHandler,
        candidate: flext_core.p.RegisterableService | flext_core.p.DispatchableHandler,
    ) -> bool:
        """Compare collaborators by identity, immutable scalars by type and value.

        Returns:
            Whether both registrations identify the same collaborator, bound
            method, or equally valued immutable scalar of the same type.

        """
        if previous is candidate:
            return True
        if isinstance(previous, MethodType) and isinstance(candidate, MethodType):
            return (
                previous.__self__ is candidate.__self__
                and previous.__func__ is candidate.__func__
            )
        return (
            type(previous) is type(candidate)
            and any(
                type(previous) is scalar_type
                for scalar_type in flext_core.c.SCALAR_TYPES
            )
            and previous == candidate
        )

    @staticmethod
    def registry_plugin_key(
        category: str, name: str | None = None
    ) -> flext_core.p.Result[str]:
        """Validate a plugin key, or a category prefix when no name is supplied.

        Returns:
            The validated key or category prefix, or a failure when a component
            is empty or contains the configured separator.

        """
        separator = flext_core.c.REGISTRY_KEY_SEPARATOR
        if (
            not category
            or separator in category
            or (name is not None and (not name or separator in name))
        ):
            return flext_core.r[str].fail(
                flext_core.c.ERR_REGISTRY_PLUGIN_KEY_INVALID.format(separator=separator)
            )
        return flext_core.r[str].ok(
            f"{category}{separator}{name}"
            if name is not None
            else f"{category}{separator}"
        )

    @staticmethod
    def registry_registration_name(binding: flext_core.p.HandlerRegistration) -> str:
        """Read the normalized route published by the dispatcher.

        Returns:
            The declared string route, otherwise the handler's qualified type
            name used to identify the registration.

        """
        route = binding.message_type
        return route if isinstance(route, str) else type(binding.handler).__qualname__

    @staticmethod
    def handler_metadata(
        candidate: flext_core.p.RegisterableService
        | flext_core.p.DispatchableHandler
        | flext_core.p.ResolvedHandlerCallable,
        *,
        name: str,
    ) -> flext_core.p.DecoratorConfig | None:
        """Read validated decorator data without wrapping or invoking a callable.

        Returns:
            The validated decorator metadata, or None when none is attached.

        Raises:
            TypeError: If attached metadata does not satisfy its model contract.
            ValueError: If metadata requests an unsupported timeout or middleware.

        """
        target = candidate.__func__ if isinstance(candidate, MethodType) else candidate
        metadata = getattr_static(target, flext_core.c.HANDLER_ATTR, None)
        if metadata is None:
            return None
        if not isinstance(metadata, flext_core.m.DecoratorConfig):
            raise TypeError(flext_core.c.ERR_HANDLER_METADATA_INVALID.format(name=name))
        if metadata.timeout is not None or metadata.middleware:
            raise ValueError(flext_core.c.ERR_HANDLER_POLICY_UNSUPPORTED)
        return metadata

    @classmethod
    def registry_registration_details(
        cls, binding: flext_core.p.HandlerRegistration
    ) -> flext_core.p.RegistrationDetails:
        """Project a normalized binding without losing the handler's declared mode.

        Returns:
            Registration details containing the normalized identifier and the
            declared or inferred handler mode.

        """
        handler = binding.handler
        return flext_core.m.RegistrationDetails(
            registration_id=cls.registry_registration_name(binding),
            handler_mode=(
                handler.mode
                if isinstance(handler, flext_core.p.Handler)
                else flext_core.c.HandlerType.EVENT
                if binding.is_event
                else flext_core.c.HandlerType.COMMAND
            ),
        )

    @classmethod
    def resolve_runtime_settings(
        cls,
        runtime_options: flext_core.p.RuntimeBootstrapOptions,
    ) -> flext_core.p.Settings:
        """Return the injected provider or load the declared settings class.

        Returns:
            The elected settings provider.

        """
        settings = runtime_options.settings
        overrides = runtime_options.settings_overrides
        if settings is not None:
            return settings.clone(**overrides) if overrides else settings
        settings_type = (
            runtime_options.settings_type or FlextUtilitiesModel.settings_base()
        )
        return settings_type.fetch_global(overrides=overrides)


__all__: list[str] = ["FlextUtilitiesModelRuntime"]
