"""Runtime registry and settings-election utilities.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from inspect import getattr_static
from types import MethodType
from typing import Annotated, ClassVar
from warnings import deprecated

from flext_core import c, m, p, r
from flext_core._utilities import FlextUtilitiesModel, FlextUtilitiesModelOptions


class FlextUtilitiesModelRuntime(FlextUtilitiesModelOptions):
    """Runtime DSL: registry comparison, plugin keys and settings election.

    Dispatcher, registry and service-runtime construction belong to their
    owner classes (``FlextDispatcher``, ``FlextRegistry.create``,
    ``FlextContainer.build_service_runtime``), never to this layer.
    """

    # Utilities never carry the models family. These models were reachable
    # as u.<Name> only while this part inherited m; each marked pass-through
    # keeps such a read working, and the compat census reports and rewires
    # it to m.<Name>. They stay until the census, external consumers
    # included, reaches zero. A Pydantic model composing u sees them as
    # class variables, never fields.
    Project: Annotated[
        ClassVar,
        deprecated(
            "compat: u.Project -> m.Project; fix: read through m",
        ),
    ] = m.Project
    PyprojectDocument: Annotated[
        ClassVar,
        deprecated(
            "compat: u.PyprojectDocument -> m.PyprojectDocument; fix: read through m",
        ),
    ] = m.PyprojectDocument
    Value: Annotated[
        ClassVar,
        deprecated(
            "compat: u.Value -> m.Value; fix: read through m",
        ),
    ] = m.Value

    @staticmethod
    def registry_same_value(
        previous: p.RegisterableService | p.DispatchableHandler,
        candidate: p.RegisterableService | p.DispatchableHandler,
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
            and any(type(previous) is scalar_type for scalar_type in c.SCALAR_TYPES)
            and previous == candidate
        )

    @staticmethod
    def registry_plugin_key(category: str, name: str | None = None) -> p.Result[str]:
        """Validate a plugin key, or a category prefix when no name is supplied.

        Returns:
            The validated key or category prefix, or a failure when a component
            is empty or contains the configured separator.

        """
        separator = c.REGISTRY_KEY_SEPARATOR
        if (
            not category
            or separator in category
            or (name is not None and (not name or separator in name))
        ):
            return r[str].fail(
                c.ERR_REGISTRY_PLUGIN_KEY_INVALID.format(separator=separator)
            )
        return r[str].ok(
            f"{category}{separator}{name}"
            if name is not None
            else f"{category}{separator}"
        )

    @staticmethod
    def registry_registration_name(binding: p.HandlerRegistration) -> str:
        """Read the normalized route published by the dispatcher.

        Returns:
            The declared string route, otherwise the handler's qualified type
            name used to identify the registration.

        """
        route = binding.message_type
        return route if isinstance(route, str) else type(binding.handler).__qualname__

    @staticmethod
    def handler_metadata(
        candidate: p.RegisterableService
        | p.DispatchableHandler
        | p.ResolvedHandlerCallable,
        *,
        name: str,
    ) -> p.DecoratorConfig | None:
        """Read validated decorator data without wrapping or invoking a callable.

        Returns:
            The validated decorator metadata, or None when none is attached.

        Raises:
            TypeError: If attached metadata does not satisfy its model contract.
            ValueError: If metadata requests an unsupported timeout or middleware.

        """
        target = candidate.__func__ if isinstance(candidate, MethodType) else candidate
        metadata = getattr_static(target, c.HANDLER_ATTR, None)
        if metadata is None:
            return None
        if not isinstance(metadata, m.DecoratorConfig):
            raise TypeError(c.ERR_HANDLER_METADATA_INVALID.format(name=name))
        if metadata.timeout is not None or metadata.middleware:
            raise ValueError(c.ERR_HANDLER_POLICY_UNSUPPORTED)
        return metadata

    @classmethod
    def registry_registration_details(
        cls, binding: p.HandlerRegistration
    ) -> p.RegistrationDetails:
        """Project a normalized binding without losing the handler's declared mode.

        Returns:
            Registration details containing the normalized identifier and the
            declared or inferred handler mode.

        """
        handler = binding.handler
        return m.RegistrationDetails(
            registration_id=cls.registry_registration_name(binding),
            handler_mode=(
                handler.mode
                if isinstance(handler, p.Handler)
                else c.HandlerType.EVENT
                if binding.is_event
                else c.HandlerType.COMMAND
            ),
        )

    @classmethod
    def resolve_runtime_settings(
        cls,
        runtime_options: p.RuntimeBootstrapOptions,
    ) -> p.Settings:
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
