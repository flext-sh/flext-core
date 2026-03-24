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

from pydantic import BaseModel

from flext_core import m, p, t


class FlextUtilitiesContext:
    """Context utility helpers for creating and managing context variables."""

    @staticmethod
    def clone_container(
        container: p.Container,
        *,
        scope_id: str | None = None,
        overrides: Mapping[str, t.RegisterableService] | None = None,
    ) -> p.Container:
        """Clone container with scoping.

        Creates a scoped container instance with optional service overrides.

        Args:
            container: Container instance to clone (must implement DI protocol).
            scope_id: Optional scope identifier.
            overrides: Optional service overrides.

        Returns:
            p.Container: Scoped container instance.

        """
        return container.scoped(subproject=scope_id, services=overrides)

    @staticmethod
    def clone_runtime[T](
        runtime: T,
        *,
        context: p.Context | None = None,
        config_overrides: t.ConfigMap | None = None,
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
        runtime_vars: t.ContainerMapping = (
            vars(runtime) if hasattr(runtime, "__dict__") else {}
        )
        if "_dispatcher" in runtime_vars:
            cloned._dispatcher = runtime_vars["_dispatcher"]
        if "_registry" in runtime_vars:
            cloned._registry = runtime_vars["_registry"]
        if "_context" in runtime_vars:
            cloned_context = context or runtime_vars["_context"]
            cloned._context = cloned_context
        if "_config" in runtime_vars:
            runtime_config = runtime_vars["_config"]
            if isinstance(config_overrides, t.ConfigMap):
                override_values = dict(config_overrides.items())
                if isinstance(runtime_config, BaseModel):
                    cloned_config = runtime_config.model_copy(update=override_values)
                    cloned._config = cloned_config
                else:
                    cloned._config = runtime_config
            else:
                cloned._config = runtime_config
        return cloned

    @staticmethod
    def create_datetime_proxy(
        key: str,
        default: datetime | None = None,
    ) -> m.StructlogProxyContextVar[datetime]:
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
            >>> var = u.Context.create_datetime_proxy("start_time")
            >>> var.set(datetime.now())
            >>> var.get()  # Returns datetime instance

        """
        proxy: m.StructlogProxyContextVar[datetime] = m.StructlogProxyContextVar[
            datetime
        ](key, default=default)
        return proxy

    @staticmethod
    def create_dict_proxy(
        key: str,
        default: t.ConfigMap | None = None,
    ) -> m.StructlogProxyContextVar[t.ConfigMap]:
        """Create StructlogProxyContextVar[dict] instance.

        Helper factory for creating dict-typed context variables with structlog
        as the single source of truth.

        Args:
            key: Context variable key name
            default: Optional default value

        Returns:
            StructlogProxyContextVar[t.ConfigMap] instance

        Example:
            >>> var = u.Context.create_dict_proxy("metadata")
            >>> var.set({"key": "value"})
            >>> var.get()  # Returns dict

        """
        proxy: m.StructlogProxyContextVar[t.ConfigMap] = m.StructlogProxyContextVar[
            t.ConfigMap
        ](key, default=default)
        return proxy

    @staticmethod
    def create_str_proxy(
        key: str,
        default: str | None = None,
    ) -> m.StructlogProxyContextVar[str]:
        """Create StructlogProxyContextVar[str] instance.

        Helper factory for creating string-typed context variables with structlog
        as the single source of truth.

        Args:
            key: Context variable key name
            default: Optional default value

        Returns:
            StructlogProxyContextVar[str] instance

        Example:
            >>> var = u.create_str_proxy("correlation_id")
            >>> var.set("abc-123")
            >>> var.get()  # Returns "abc-123"

        """
        proxy: m.StructlogProxyContextVar[str] = m.StructlogProxyContextVar[str](
            key,
            default=default,
        )
        return proxy


__all__ = ["FlextUtilitiesContext"]
