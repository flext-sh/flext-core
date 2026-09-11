"""CLI Pydantic domain models."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, ClassVar

from flext_cli import c, t
from flext_core import m, u

_EMPTY_JSON_MAPPING: t.JsonMapping = MappingProxyType({})


class FlextCliModelsBase:
    """Implementation part for FlextCliModelsBase."""

    class LogLevelResolved(m.BaseModel):
        """Single contract for log level string."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid")
        raw: Annotated[
            str | None, m.Field(None, description="Raw log level input string")
        ]
        default: Annotated[
            str,
            m.Field(
                c.LogLevel.INFO, description="Default log level when raw is absent"
            ),
        ]

        @u.computed_field
        @property
        def resolved(self) -> str:
            """Resolved log level value."""
            return self.resolve()

        def resolve(self) -> str:
            """Type-safe accessor (bypasses pyrefly computed_field limitation)."""
            s = (self.raw or self.default).strip().upper()
            return s or self.default

    class TypedExtract(m.BaseModel):
        """Single contract for typed value extraction (str | bool | dict)."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid")
        type_kind: Annotated[t.Cli.TypeKind, m.Field(description="Requested type")]
        value: Annotated[
            t.JsonValue | None, m.Field(None, description="Value to extract and coerce")
        ]
        default: Annotated[
            t.JsonValue | None,
            m.Field(None, description="Fallback value when extraction fails"),
        ]

        @u.computed_field
        @property
        def resolved(self) -> t.Cli.TypedExtractValue:
            """Value coerced to type_kind, or default."""
            return self.resolve()

        def resolve(self) -> t.Cli.TypedExtractValue:
            """Type-safe accessor (bypasses pyrefly computed_field limitation)."""
            default_value = self._default_for_kind()
            if self.value is None:
                return default_value
            resolved_value: t.Cli.TypedExtractValue = default_value
            match self.type_kind:
                case c.Cli.TypeKind.STR:
                    resolved_str = str(self.value).strip() if self.value else ""
                    resolved_value = resolved_str or (
                        self.default if isinstance(self.default, str) else ""
                    )
                case c.Cli.TypeKind.BOOL:
                    resolved_value = bool(self.value)
                case c.Cli.TypeKind.DICT:
                    source_mapping = (
                        self.value
                        if isinstance(self.value, Mapping)
                        else self.default
                        if isinstance(self.default, Mapping)
                        else None
                    )
                    resolved_value = (
                        {
                            k: t.Cli.JSON_VALUE_ADAPTER.validate_python(vv)
                            for k, vv in source_mapping.items()
                        }
                        if source_mapping is not None
                        else _EMPTY_JSON_MAPPING
                    )
            return resolved_value

        def _default_for_kind(self) -> t.Cli.TypedExtractValue:
            """Return default typed value for the requested kind."""
            if self.type_kind == c.Cli.TypeKind.STR:
                return self.default if isinstance(self.default, str) else ""
            if self.type_kind == c.Cli.TypeKind.BOOL:
                return self.default if isinstance(self.default, bool) else False
            if isinstance(self.default, Mapping):
                return {
                    k: t.Cli.JSON_VALUE_ADAPTER.validate_python(vv)
                    for k, vv in self.default.items()
                }
            # NOTE (multi-agent): Empty mapping is immutable and typed once.
            return _EMPTY_JSON_MAPPING

    class JsonWriteOptions(m.BaseModel):
        """Options for JSON file write operations."""

        indent: int = u.Field(
            2, description="JSON indentation level", validate_default=True
        )
        sort_keys: bool = u.Field(
            False, description="Sort JSON keys", validate_default=True
        )
        ensure_ascii: bool = u.Field(
            False, description="Escape non-ASCII chars", validate_default=True
        )

    class TableRenderRequest(m.Value):
        """Validated table-rendering request for the Rich boundary."""

        columns: Annotated[
            t.StrSequence, m.Field(description="Ordered table column labels")
        ]
        rows: Annotated[
            t.SequenceOf[t.StrSequence], m.Field(description="Ordered table row values")
        ]
        title: Annotated[str, m.Field(description="Optional table title")] = ""


__all__: list[str] = ["FlextCliModelsBase"]
