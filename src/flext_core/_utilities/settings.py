"""Settings helpers for parameter access and manipulation.

Only the helpers used by ``src/`` (any FLEXT project) are retained.  Dead
validators (URL scheme / trace-debug), log-level bootstraps, and the
``_try_get_*`` attribute-resolution ladders were removed after the
fire-test audit - callers that need narrowing must go through Pydantic
models directly.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from flext_core._constants.environment import FlextConstantsEnvironment
from flext_core._constants.errors import FlextConstantsErrors
from flext_core._exceptions.factories import FlextExceptionsFactories as e
from flext_core.result import r

if TYPE_CHECKING:
    from flext_core import FlextProtocols as p, FlextTypes as t


class FlextUtilitiesSettings:
    """Settings utilities for environment resolution and DI registration."""

    @staticmethod
    def resolve_process_environment() -> dict[str, str]:
        """Resolve the inherited process environment as a plain string mapping."""
        return dict(os.environ)

    @staticmethod
    def resolve_env_file() -> str:
        """Resolve .env file path from FLEXT_ENV_FILE env var."""
        custom_env_file = os.environ.get(FlextConstantsEnvironment.ENV_FILE_ENV_VAR)
        if custom_env_file:
            custom_path = Path(custom_env_file)
            if custom_path.exists():
                return str(custom_path.resolve())
            return custom_env_file
        default_path = Path.cwd() / FlextConstantsEnvironment.ENV_FILE_DEFAULT
        if default_path.exists():
            return str(default_path.resolve())
        return FlextConstantsEnvironment.ENV_FILE_DEFAULT

    @staticmethod
    def register_factory(
        container: p.Container,
        name: str,
        factory: t.FactoryCallable,
    ) -> p.Result[bool]:
        """Register factory in DI container, verifying resolution succeeds."""
        _ = container.factory(name, factory)
        resolved = container.resolve(name)
        if resolved.failure:
            failure: p.Result[bool] = e.fail_operation(
                "resolve registered config factory",
                resolved.error or FlextConstantsErrors.ERR_CONFIG_FACTORY_REGISTRATION_FAILED,
            )
            return failure
        return r[bool].ok(True)


__all__: t.MutableSequenceOf[str] = ["FlextUtilitiesSettings"]
