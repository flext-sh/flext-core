"""CLI Pydantic domain models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, ClassVar

from flext_cli import t
from flext_cli._models._defaults import EMPTY_JSON_MAPPING
from flext_core import m, u


class FlextCliModelsBase:
    """Implementation part for FlextCliModelsBase."""

    class CommandOutput(m.Value):
        """Standardized external command execution payload. Use m.Cli.CommandOutput."""

        stdout: Annotated[str, m.Field("", description="Captured standard output")] = ""
        stderr: Annotated[str, m.Field("", description="Captured standard error")] = ""
        exit_code: Annotated[int, m.Field(description="Command exit code")] = 0
        duration: Annotated[
            t.NonNegativeFloat, m.Field(0.0, description="Duration in seconds")
        ] = 0.0

    class CommandBytesOutput(m.Value):
        """Byte-exact external command payload. Use m.Cli.CommandBytesOutput."""

        stdout: Annotated[
            bytes, m.Field(b"", description="Captured standard output as raw bytes")
        ] = b""
        stderr: Annotated[
            bytes, m.Field(b"", description="Captured standard error as raw bytes")
        ] = b""
        exit_code: Annotated[int, m.Field(description="Command exit code")] = 0
        duration: Annotated[
            t.NonNegativeFloat, m.Field(0.0, description="Duration in seconds")
        ] = 0.0

    class ProcessDeadline(m.Value):
        """Absolute monotonic process deadline. Use m.Cli.ProcessDeadline."""

        expires_at_monotonic: Annotated[
            t.PositiveFloat,
            m.Field(description="Absolute time.monotonic expiry in seconds"),
        ]
        termination_grace_seconds: Annotated[
            t.PositiveFloat,
            m.Field(description="Reserved graceful termination and drain budget"),
        ]
        timeout_exit_code: Annotated[
            t.PositiveInt,
            m.Field(description="Canonical exit code returned for deadline expiry"),
        ]

    class RuntimeComponents(m.BaseModel):
        """Availability state for canonical CLI runtime components."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)
        settings: Annotated[str, m.Field(description="Settings component state")]
        formatters: Annotated[str, m.Field(description="Formatters component state")]
        prompts: Annotated[str, m.Field(description="Prompts component state")]
        rules: Annotated[str, m.Field(description="Rules component state")]

    class RuntimeStatus(m.BaseModel):
        """Canonical public CLI runtime status payload."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)
        status: Annotated[str, m.Field(description="Overall service state")]
        service: Annotated[str, m.Field(description="Service identifier")]
        timestamp: Annotated[str, m.Field(description="Status generation timestamp")]
        version: Annotated[str, m.Field(description="CLI version")]
        components: Annotated[
            FlextCliModelsBase.RuntimeComponents,
            m.Field(description="Component availability states"),
        ]

    class DisplayData(m.BaseModel):
        """Key-value data for table/display — Pydantic v2 contract. Use m.Cli.DisplayData."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            extra="forbid", validate_assignment=True
        )
        data: Annotated[
            t.JsonMapping,
            m.Field(
                default_factory=lambda: EMPTY_JSON_MAPPING,
                description="Field-value pairs for display",
            ),
        ]

        @u.model_serializer
        def _serialize(self) -> t.JsonMapping:
            """Serialize the wrapper as its display payload."""
            return dict(self.data)

    class LoadedConfig(m.BaseModel):
        """Loaded configuration content wrapper — Pydantic v2 contract. Use m.Cli.LoadedConfig."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            extra="forbid", validate_assignment=True
        )
        content: Annotated[
            t.JsonMapping,
            m.Field(
                description="Loaded configuration content (dict or other JSON value)"
            ),
        ]

    class CliNormalizedJson(m.RootModel[t.JsonValue]):
        """Normalize raw JSON value with flat JSON serialization semantics.

        ``RootModel`` provides positional construction (``CliNormalizedJson(value)``)
        and root-level serialization natively — no custom ``__init__`` or
        ``model_serializer`` required.
        """

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=True)
        root: Annotated[
            t.JsonValue, m.Field(description="Normalized JSON-compatible value")
        ]

    class NormalizedJsonList(m.BaseModel):
        """Resolve normalized JSON to a dict with defaults. Use m.Cli.NormalizedJsonList."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            extra="forbid", validate_assignment=True
        )
        value: Annotated[
            t.JsonValue,
            m.Field(default_factory=dict, description="The normalized JSON value"),
        ]
        default: Annotated[
            t.JsonMapping,
            m.Field(
                default_factory=lambda: EMPTY_JSON_MAPPING,
                description="Default mapping if value is not a dict",
            ),
        ]

        @property
        def resolved(self) -> t.JsonMapping:
            """Resolve value to dict or return default."""
            if isinstance(self.value, Mapping):
                return self.value
            return self.default

    class SuccessSummaryDetails(m.RootModel[t.MappingKV[str, str]]):
        """Key-value success summary details. Use m.Cli.SuccessSummaryDetails."""

        root: t.MappingKV[str, str]


__all__: list[str] = ["FlextCliModelsBase"]
