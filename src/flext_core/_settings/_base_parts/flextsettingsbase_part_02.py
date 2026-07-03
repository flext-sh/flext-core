"""FlextSettingsBase — singleton + Pydantic-2 facade for every settings class.

Houses the canonical helper API shared by `FlextSettings` (root) and
project-specific settings classes (`FlextLdifSettings`, `FlextApiSettings`, …):

- per-class singleton (``_instance`` ClassVar with thread-safe lock),
- ``fetch_global()`` — return shared singleton (rule 1; runtime mutations
  via ``update_global`` propagate),
- ``clone(**overrides)`` — `model_copy(deep=True)` + re-validation for
  custom-injection isolation (rule 2),
- ``update_global(**overrides)`` — replace ``cls._instance`` via
  `model_copy(update=…)` + re-validation; pure Pydantic-2 mutation
  (no ``setattr``, no ``__setattr__``),
- ``validate_overrides(**overrides)`` — typo guard before clone/update,
- ``clone_for_injection(settings)`` — single helper for service/container
  constructors that accept an explicit ``settings=…`` argument (rule 2),
- ``reset_for_testing()`` — drop singleton slot,
- ``resolve_env_file(namespace=None)`` — centralised .env discovery (rule 5).

Extends ``pydantic_settings.BaseSettings`` so env-loading plus Pydantic-only
operations (``model_copy``, ``__pydantic_validator__``, ``model_fields``)
are typed at this layer. Project subclasses inherit ``FlextSettingsBase``
directly, giving them the helper API without inheriting root concrete fields
(rule 3).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Self

from flext_core._constants.environment import FlextConstantsEnvironment

from .flextsettingsbase_part_01 import (
    FlextSettingsBase as FlextSettingsBasePart01,
)


class FlextSettingsBase(FlextSettingsBasePart01):
    @classmethod
    def reset_for_testing(cls) -> None:
        """Reset the per-class singleton for test isolation."""
        cls.reset_instance()

    @classmethod
    def clone_for_injection(cls, settings: Self | None) -> Self | None:
        """Canonical helper for service/container constructors (rule 2).

        Returns:
            ``None`` when caller passed ``settings=None`` (caller resolves to
            ``cls.fetch_global()`` lazily so runtime mutations propagate);
            otherwise an isolated deep clone of the supplied instance.

        """
        if settings is None:
            return None
        if not isinstance(settings, cls):
            msg = f"clone_for_injection expected {cls.__name__}, got {type(settings).__name__}"
            raise TypeError(msg)
        return settings.clone()

    @classmethod
    def resolve_env_file(cls, namespace: str | None = None) -> str:
        """Centralised .env discovery (rule 5).

        Honours ``FLEXT_ENV_FILE`` env var; otherwise prefers a
        namespace-specific ``.env.flext-{namespace}`` (when ``namespace`` is
        given and the file exists) and falls back to the workspace ``.env``.
        """
        custom_env_file = os.environ.get(FlextConstantsEnvironment.ENV_FILE_ENV_VAR)
        if custom_env_file:
            custom_path = Path(custom_env_file)
            if custom_path.exists():
                return str(custom_path.resolve())
            return custom_env_file
        if namespace:
            scoped = Path.cwd() / f".env.flext-{namespace}"
            if scoped.exists():
                return str(scoped.resolve())
        default_path = Path.cwd() / FlextConstantsEnvironment.ENV_FILE_DEFAULT
        if default_path.exists():
            return str(default_path.resolve())
        return FlextConstantsEnvironment.ENV_FILE_DEFAULT


__all__: list[str] = ["FlextSettingsBase"]
