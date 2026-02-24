"""Context utility helpers for creating and managing context variables.

These helpers provide factory functions for creating StructlogProxyContextVar
instances with proper typing, ensuring consistent context variable creation
across the FLEXT ecosystem.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from flext_core._models.context import FlextModelsContext
from flext_core.protocols import p
from flext_core.typings import t


class FlextUtilitiesContext:
    """Context utility helpers for creating and managing context variables."""

    @staticmethod
    def create_str_proxy(
        key: str,
        default: str | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[str]:
        """Create StructlogProxyContextVar[str] instance.

        Helper factory for creating string-typed context variables with structlog
        as the single source of truth.

        Args:
            key: Context variable key name
            default: Optional default value

        Returns:
            StructlogProxyContextVar[str] instance

        Example:
            >>> var = uContext.create_str_proxy("correlation_id")
            >>> var.set("abc-123")
            >>> var.get()  # Returns "abc-123"

        """
        # Explicit instantiation with full type
        proxy: FlextModelsContext.StructlogProxyContextVar[str] = (
            FlextModelsContext.StructlogProxyContextVar[str](key, default=default)
        )
        return proxy

    @staticmethod
    def create_datetime_proxy(
        key: str,
        default: datetime | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[datetime]:
        """Create StructlogProxyContextVar[datetime] instance.

        Helper factory for creating datetime-typed context variables with structlog
        as the single source of truth.

        Args:
            key: Context variable key name
            default: Optional default value

        Returns:
            StructlogProxyContextVar[datetime] instance

        Example:
            >>> from datetime import datetime
            >>> var = uContext.create_datetime_proxy("start_time")
            >>> var.set(datetime.now())
            >>> var.get()  # Returns datetime instance

        """
        # Explicit instantiation with full type
        proxy: FlextModelsContext.StructlogProxyContextVar[datetime] = (
            FlextModelsContext.StructlogProxyContextVar[datetime](
                key,
                default=default,
            )
        )
        return proxy

    @staticmethod
    def create_dict_proxy(
        key: str,
        default: dict[str, t.ConfigMapValue] | None = None,
    ) -> FlextModelsContext.StructlogProxyContextVar[dict[str, t.ConfigMapValue]]:
        """Create StructlogProxyContextVar[dict] instance.

        Helper factory for creating dict-typed context variables with structlog
        as the single source of truth.

        Args:
            key: Context variable key name
            default: Optional default value

        Returns:
            StructlogProxyContextVar[dict[str, t.ConfigMapValue]] instance

        Example:
            >>> var = uContext.create_dict_proxy("metadata")
            >>> var.set({"key": "value"})
            >>> var.get()  # Returns dict

        """
        # Explicit instantiation with full type
        proxy: FlextModelsContext.StructlogProxyContextVar[
            dict[str, t.ConfigMapValue]
        ] = FlextModelsContext.StructlogProxyContextVar[dict[str, t.ConfigMapValue]](
            key,
            default=default,
        )
        return proxy

    @staticmethod
    def clone_runtime[T](
        runtime: T,
        *,
        context: p.Context | None = None,
        config_overrides: dict[str, t.ConfigMapValue] | None = None,
    ) -> T:
        """Clone runtime with optional overrides.

        Creates a new runtime instance with the same dispatcher and registry,
        but with optional context and config overrides.

        Args:
            runtime: Runtime instance to clone (must implement Runtime protocol).
            context: Optional new context. If not provided, uses runtime's context.
            config_overrides: Optional config field overrides.

        Returns:
            T: Cloned runtime instance.

        """
        cloned: T = runtime.__class__.__new__(runtime.__class__)
        if hasattr(runtime, "_dispatcher"):
            dispatcher_attr = "_dispatcher"
            setattr(cloned, dispatcher_attr, getattr(runtime, dispatcher_attr))
        if hasattr(runtime, "_registry"):
            registry_attr = "_registry"
            setattr(cloned, registry_attr, getattr(runtime, registry_attr))
        if hasattr(runtime, "_context"):
            context_attr = "_context"
            cloned_context = context or getattr(runtime, context_attr)
            setattr(cloned, context_attr, cloned_context)
        if hasattr(runtime, "_config"):
            config_attr = "_config"
            runtime_config = getattr(runtime, config_attr)
            if config_overrides:
                cloned_config = runtime_config.model_copy(update=config_overrides)
                setattr(cloned, config_attr, cloned_config)
            else:
                setattr(cloned, config_attr, runtime_config)
        return cloned

    @staticmethod
    def clone_container(
        container: p.DI,
        *,
        scope_id: str | None = None,
        overrides: Mapping[str, t.ConfigMapValue] | None = None,
    ) -> p.DI:
        """Clone container with scoping.

        Creates a scoped container instance with optional service overrides.

        Args:
            container: Container instance to clone (must implement DI protocol).
            scope_id: Optional scope identifier.
            overrides: Optional service overrides.

        Returns:
            p.DI: Scoped container instance.

        """
        return container.scoped(
            subproject=scope_id,
            services=overrides,
        )


uContext = FlextUtilitiesContext

__all__ = [
    "FlextUtilitiesContext",
    "uContext",
]
