"""Behavioral contract tests for the pydantic-settings facade on ``m``.

Exercises the wide pydantic-settings exports consumed through ``m`` exactly as
fleet consumers do: observable override behavior only, never pydantic
internals or identity introspection.
"""

from __future__ import annotations

from typing import override

from tests import u as test_u
from tests.models import m


class TestsFlextCorePydanticSettingsFacade:
    """Behavioral contract for the wide pydantic-settings facade exports."""

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
            """Init-only resolution proves the override is consulted."""
            _ = (settings_cls, env_settings, dotenv_settings, file_secret_settings)
            return (init_settings,)

    def test_base_settings_alias_provides_real_env_resolution(self) -> None:
        """The wide alias behaves as genuine pydantic-settings.

        Only the real settings base resolves constructor fields from the
        environment by default; a narrowed substitute (e.g. a plain model
        subclass) would leave the default in place.
        """

        class _EnvSettings(m.BaseSettings):
            """Settings whose topic prefix is owned by the environment."""

            topic_prefix: str = "unset"

        with test_u.Tests.env_vars_context(env_vars={"TOPIC_PREFIX": "from-env"}):
            assert _EnvSettings().topic_prefix == "from-env"
        assert _EnvSettings(topic_prefix="explicit").topic_prefix == "explicit"

    def test_settings_override_annotated_through_alias_is_consulted(self) -> None:
        with test_u.Tests.env_vars_context(env_vars={"TOPIC": "from-env"}):
            assert self._TopicSettings().topic == "default"
            assert self._TopicSettings(topic="explicit").topic == "explicit"
