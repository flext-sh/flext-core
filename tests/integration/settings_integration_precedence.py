"""Settings precedence integration case kept below module LOC cap.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, ClassVar

from typing_extensions import TypeForm

from tests import FlextSettings, c, m, t, tm, u

if TYPE_CHECKING:
    from pathlib import Path

_FLEXT_ENV_KEYS = (
    "FLEXT_LOG_LEVEL",
    "FLEXT_DEBUG",
    "FLEXT_TRACE",
    "FLEXT_TIMEZONE",
    "FLEXT_ENV_FILE",
)


class TestsFlextSettingsConfigTestCase(m.BaseModel):
    """Factory for configuration test cases."""

    __test__ = False

    model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

    test_name: Annotated[str, m.Field(description="Configuration test case name")]
    config_data: Annotated[
        t.JsonMapping,
        m.Field(description="Input configuration payload"),
    ]
    expected_values: Annotated[
        t.JsonMapping,
        m.Field(description="Expected effective values"),
    ] = m.Field(default_factory=u.empty(TypeForm(t.JsonMapping)))
    file_format: Annotated[str, m.Field(description="Configuration file format")] = (
        "json"
    )
    env_vars: Annotated[
        Mapping[str, str],
        m.Field(description="Environment variable overrides"),
    ] = m.Field(default_factory=u.empty(TypeForm(Mapping[str, str])))
    description: Annotated[
        str,
        m.Field(description="Human-readable test description"),
    ] = ""

    def create_temp_file(self, temp_dir: Path) -> Path:
        """Create temporary settings file.

        Returns:
            The resulting ``Path``.

        """
        file_path = temp_dir / f"test_config.{self.file_format}"
        if self.file_format == "json":
            u.Cli.json_write(file_path, self.config_data)
        elif self.file_format == "yaml":
            u.Cli.yaml_dump(file_path, self.config_data)
        elif self.file_format == "toml":
            content = "".join((f"{k} = {v!r}" for k, v in self.config_data.items()))
            _ = file_path.write_text(content)
        return file_path


class TestsFlextSettingsThreadSafetyTest(m.BaseModel):
    """Factory for thread safety test configurations."""

    __test__ = False

    model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

    thread_count: Annotated[
        int,
        m.Field(description="Number of concurrent threads"),
    ] = 5
    operations_per_thread: Annotated[
        int,
        m.Field(description="Operations per thread"),
    ] = 10
    description: Annotated[
        str,
        m.Field(description="Thread safety scenario description"),
    ] = ""


class TestsFlextSettingsConfigTestFactories:
    """Centralized factories for configuration tests."""

    @staticmethod
    def basic_config_cases() -> Sequence[TestsFlextSettingsConfigTestCase]:
        """Generate basic configuration test cases.

        Returns:
            The resulting ``Sequence[TestsFlextSettingsConfigTestCase]``.

        """
        return [
            TestsFlextSettingsConfigTestCase(
                test_name="basic_json",
                config_data={"app_name": "test_app", "debug": True, "port": 8080},
                expected_values={"app_name": "test_app", "debug": True, "port": 8080},
                file_format="json",
                description="Basic JSON configuration",
            ),
            TestsFlextSettingsConfigTestCase(
                test_name="basic_yaml",
                config_data={"database_url": "sqlite:///test.db", "timeout": 30},
                expected_values={"database_url": "sqlite:///test.db", "timeout": 30},
                file_format="yaml",
                description="Basic YAML configuration",
            ),
            TestsFlextSettingsConfigTestCase(
                test_name="env_override",
                config_data={"max_connections": 10},
                expected_values={"max_connections": 20},
                env_vars={"FLEXT_MAX_CONNECTIONS": "20"},
                file_format="yaml",
                description="Environment variable override",
            ),
        ]

    @staticmethod
    def thread_safety_cases() -> Sequence[TestsFlextSettingsThreadSafetyTest]:
        """Generate thread safety test cases.

        Returns:
            The resulting ``Sequence[TestsFlextSettingsThreadSafetyTest]``.

        """
        return [
            TestsFlextSettingsThreadSafetyTest(
                thread_count=3,
                operations_per_thread=5,
                description="Light concurrent access",
            ),
            TestsFlextSettingsThreadSafetyTest(
                thread_count=10,
                operations_per_thread=20,
                description="Heavy concurrent access",
            ),
        ]


class TestsFlextFlextSettingsFactories:
    """Expose the previous nested factory names through inheritance."""

    _ConfigTestCase: ClassVar[type[TestsFlextSettingsConfigTestCase]] = (
        TestsFlextSettingsConfigTestCase
    )
    _ThreadSafetyTest: ClassVar[type[TestsFlextSettingsThreadSafetyTest]] = (
        TestsFlextSettingsThreadSafetyTest
    )
    _ConfigTestFactories: ClassVar[type[TestsFlextSettingsConfigTestFactories]] = (
        TestsFlextSettingsConfigTestFactories
    )


class TestsFlextFlextSettingsPrecedenceCase(TestsFlextFlextSettingsFactories):
    """Assert Pydantic settings precedence across the current field surface."""

    @staticmethod
    def test_pydantic_settings_precedence_order(temp_dir: Path) -> None:
        """Defaults < .env file < env var < explicit init for universal fields."""
        with u.Tests.env_vars_context({}, vars_to_clear=_FLEXT_ENV_KEYS):
            FlextSettings.reset_for_testing()
            defaults = FlextSettings()
            tm.that(defaults.log_level, eq="INFO")
            tm.fail(defaults.debug)
            tm.fail(defaults.trace)
            tm.that(defaults.timezone, eq="UTC")

            FlextSettings.reset_for_testing()
            env_file = temp_dir / ".env"
            env_content = "FLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n"
            _ = env_file.write_text(env_content, encoding="utf-8")
            tm.that(env_file.read_text(encoding="utf-8"), eq=env_content)
            with u.Tests.env_vars_context({"FLEXT_ENV_FILE": str(env_file)}):
                from_dotenv = FlextSettings()
                tm.that(from_dotenv.log_level, eq="WARNING")
                tm.ok(from_dotenv.debug)

            FlextSettings.reset_for_testing()
            with u.Tests.env_vars_context({
                "FLEXT_LOG_LEVEL": "DEBUG",
                "FLEXT_DEBUG": c.ParserBooleanToken.TRUE,
                "FLEXT_TRACE": c.ParserBooleanToken.TRUE,
            }):
                from_env = FlextSettings()
                tm.that(from_env.log_level, eq="DEBUG")
                tm.ok(from_env.debug)
                tm.ok(from_env.trace)

            FlextSettings.reset_for_testing()
            explicit = FlextSettings(
                log_level="ERROR",
                debug=True,
                trace=False,
                timezone="America/Sao_Paulo",
            )
            tm.that(explicit.log_level, eq="ERROR")
            tm.ok(explicit.debug)
            tm.fail(explicit.trace)
            tm.that(explicit.timezone, eq="America/Sao_Paulo")
        FlextSettings.reset_for_testing()
