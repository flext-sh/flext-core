"""FlextProtocolsConfig - configuration protocols.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from flext_core._protocols.base import FlextProtocolsBase as _Base
from flext_core._protocols.result import FlextProtocolsResult as _Result

if TYPE_CHECKING:
    from flext_core import t


class FlextProtocolsConfig:
    """Protocols for configurable components and settings."""

    @runtime_checkable
    class Configurable(_Base.Base, Protocol):
        """Protocol for component configuration."""

        def configure(self, config: Mapping[str, t.Container] | None = None) -> Self:
            """Configure component with settings."""
            ...

    @runtime_checkable
    class Settings(
        _Result.HasModelDump,
        _Base.Base,
        Protocol,
    ):
        """Configuration protocol based on Pydantic BaseSettings pattern.

        Reflects real implementations like FlextSettings which uses Pydantic BaseSettings.
        Configuration items use direct field access (Pydantic standard) rather than
        explicit get/set methods. Supports cloning via model_copy() and optional
        override methods.
        """

        app_name: str
        "Application name bound to the configuration."
        version: str
        "Semantic version of the running application."
        enable_caching: bool
        "Enable caching for query operations."
        timeout_seconds: float
        "Default timeout in seconds for operations."
        dispatcher_auto_context: bool
        "Enable automatic context management in dispatcher."
        dispatcher_enable_logging: bool
        "Enable logging in dispatcher operations."

        def model_copy(
            self,
            *,
            update: Mapping[str, t.Container] | None = None,
            deep: bool = False,
        ) -> Self:
            """Create a copy of the model, optionally updating fields or deep copying.

            Args:
                update: Dictionary of values to update in the copied model.
                deep: If True, perform a deep copy of the model fields.

            Returns:
                A new instance of the model.

            """
            ...


__all__ = ["FlextProtocolsConfig"]
