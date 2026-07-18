"""Settings integration factories kept outside the collected test module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from tests import m
from tests import u

if TYPE_CHECKING:
    from pathlib import Path

    from tests import t


class TestsFlextSettingsConfigTestCase(m.BaseModel):
    """Factory for configuration test cases."""

    model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

    test_name: Annotated[str, m.Field(description="Configuration test case name")]
    config_data: Annotated[
        t.JsonMapping, m.Field(description="Input configuration payload")
    ]
    expected_values: Annotated[
        t.JsonMapping, m.Field(description="Expected effective values")
    ] = m.Field(default_factory=dict)
    file_format: Annotated[str, m.Field(description="Configuration file format")] = (
        "json"
    )
    env_vars: Annotated[
        t.StrMapping, m.Field(description="Environment variable overrides")
    ] = m.Field(default_factory=dict)
    description: Annotated[
        str, m.Field(description="Human-readable test description")
    ] = ""

    def create_temp_file(self, temp_dir: Path) -> Path:
        """Create temporary settings file."""
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

    model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

    thread_count: Annotated[
        int, m.Field(description="Number of concurrent threads")
    ] = 5
    operations_per_thread: Annotated[
        int, m.Field(description="Operations per thread")
    ] = 10
    description: Annotated[
        str, m.Field(description="Thread safety scenario description")
    ] = ""


class TestsFlextSettingsConfigTestFactories:
    """Centralized factories for configuration tests."""

    @staticmethod
    def basic_config_cases() -> t.SequenceOf[TestsFlextSettingsConfigTestCase]:
        """Generate basic configuration test cases."""
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
    def thread_safety_cases() -> t.SequenceOf[TestsFlextSettingsThreadSafetyTest]:
        """Generate thread safety test cases."""
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
