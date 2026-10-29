"""Behavioral contract tests for the pydantic-settings facade on ``m``.

Exercises the wide pydantic-settings exports consumed through ``m`` exactly as
fleet consumers do: observable override behavior only, never pydantic
internals or identity introspection.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_core import FlextModels
from tests import m, t, tm, u


class TestsFlextCorePydanticSettingsFacade:
    """Behavioral contract for the wide pydantic-settings facade exports."""

    @staticmethod
    def test_public_settings_preset_round_trips_json() -> None:
        """The public preset validates settings rather than a config mapping."""
        instance = FlextModels.BaseSettings()
        restored = FlextModels.BaseSettings.model_validate_json(
            instance.model_dump_json()
        )
        tm.that(restored.model_dump(), eq=instance.model_dump())

    @staticmethod
    def test_native_settings_alias_accepts_the_public_preset() -> None:
        """The native annotation accepts the original settings instance."""
        instance = FlextModels.BaseSettings()
        adapter = u.type_adapter(t.TypeForm(FlextModels.PydanticBaseSettings))
        tm.ok(adapter.validate_python(instance) is instance)

    class _TopicSettings(m.BaseSettings):
        """Settings model whose customise hook is annotated through the facade."""

        topic: str = "default"

        @override
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[m.PydanticBaseSettings],
            init_settings: m.PydanticBaseSettingsSource,
            env_settings: m.PydanticBaseSettingsSource,
            dotenv_settings: m.PydanticBaseSettingsSource,
            file_secret_settings: m.PydanticBaseSettingsSource,
        ) -> tuple[m.PydanticBaseSettingsSource, ...]:
            """Init-only resolution proves the override is consulted.

            Returns:
                The resulting ``tuple[m.PydanticBaseSettingsSource, ...]``.

            """
            _ = (settings_cls, env_settings, dotenv_settings, file_secret_settings)
            return (init_settings,)

    @staticmethod
    def test_base_settings_alias_provides_real_env_resolution() -> None:
        """The wide alias behaves as genuine pydantic-settings.

        Only the real settings base resolves constructor fields from the
        environment by default; a narrowed substitute (e.g. a plain model
        subclass) would leave the default in place.
        """

        class _EnvSettings(m.BaseSettings):
            """Settings whose topic prefix is owned by the environment."""

            topic_prefix: str = "unset"

        with u.Tests.env_vars_context(env_vars={"TOPIC_PREFIX": "from-env"}):
            tm.that(_EnvSettings().topic_prefix, eq="from-env")
        tm.that(_EnvSettings(topic_prefix="explicit").topic_prefix, eq="explicit")

    def test_settings_override_annotated_through_alias_is_consulted(self) -> None:
        with u.Tests.env_vars_context(env_vars={"TOPIC": "from-env"}):
            tm.that(self._TopicSettings().topic, eq="default")
            tm.that(self._TopicSettings(topic="explicit").topic, eq="explicit")
