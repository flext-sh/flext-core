"""CLI Pydantic domain models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, ClassVar, Self

from flext_cli import c, t
from flext_cli._models._defaults import EMPTY_STR_MAPPING
from flext_core import m, u


class FlextCliModelsBase:
    """Implementation part for FlextCliModelsBase."""

    class AtomicFileState(m.BaseModel):
        """Exact presence, bytes, and permission mode for one physical file."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            extra="forbid", frozen=True, arbitrary_types_allowed=True
        )
        path: Annotated[Path, m.Field(description="Absolute file path")]
        content: Annotated[
            bytes | None,
            m.Field(strict=True, description="Exact bytes, or None when absent"),
        ] = None
        mode: Annotated[
            int | None,
            m.Field(
                ge=0,
                le=0o7777,
                strict=True,
                description="Exact permission bits, or None when absent",
            ),
        ] = None
        device: Annotated[
            int | None,
            m.Field(
                ge=0, strict=True, description="Physical device, or None when absent"
            ),
        ] = None
        inode: Annotated[
            int | None,
            m.Field(
                ge=0, strict=True, description="Physical inode, or None when absent"
            ),
        ] = None

        @u.field_validator("path")
        @classmethod
        def _validate_absolute_path(cls, value: Path) -> Path:
            """Reject ambiguous relative identities instead of normalizing them."""
            normalized = Path(os.path.normpath(value))
            if (
                not value.is_absolute()
                or not value.name
                or ".." in value.parts
                or normalized != value
            ):
                msg = "atomic file state path must be absolute and normalized"
                raise ValueError(msg)
            return value

        @u.model_validator(mode="after")
        def _validate_presence_tuple(self) -> Self:
            """Require bytes and mode together for every existing state."""
            presence = tuple(
                value is not None
                for value in (self.content, self.mode, self.device, self.inode)
            )
            if any(presence) != all(presence):
                msg = (
                    "atomic file state must contain bytes, mode, device, and inode, "
                    "or none of them"
                )
                raise ValueError(msg)
            return self

    class PromptRuntimeState(m.FlexibleInternalModel):
        """Centralized runtime state for CLI prompt behavior."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            extra="forbid", validate_assignment=True
        )

        interactive: Annotated[
            bool, m.Field(True, description="Whether prompt interaction is enabled")
        ] = True
        quiet: Annotated[
            bool, m.Field(False, description="Whether prompt output is suppressed")
        ] = False
        default_timeout: Annotated[
            int, m.Field(description="Default prompt timeout in seconds")
        ] = c.Cli.PROMPT_DEFAULT_TIMEOUT

    class AuthCredentialsPayload(m.BaseModel):
        """Validated auth payload for token or username/password flows."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
            extra="forbid", validate_assignment=True
        )
        token: Annotated[
            str | None,
            m.Field(None, description="Direct authentication token", strict=True),
        ] = None
        username: Annotated[
            str, m.Field("", description="Authentication username", strict=True)
        ] = ""
        password: Annotated[
            str, m.Field("", description="Authentication password", strict=True)
        ] = ""

    class ProcessEnvironmentSpec(m.BaseModel):
        """Validated process environment contract for runtime command execution."""

        model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid", frozen=True)
        base_env: Annotated[
            t.StrMapping,
            m.Field(
                default_factory=lambda: EMPTY_STR_MAPPING,
                description="Base environment inherited from the current process",
            ),
        ]
        overrides: Annotated[
            t.StrMapping,
            m.Field(
                default_factory=lambda: EMPTY_STR_MAPPING,
                description="Explicit environment overrides for the child process",
            ),
        ]
        remove_keys: Annotated[
            t.StrSequence,
            m.Field(
                default_factory=tuple,
                description="Environment keys removed before applying overrides",
            ),
        ]

        @u.computed_field
        @property
        def resolved(self) -> dict[str, str]:
            """Resolved environment mapping after removals and overrides."""
            return self.resolve()

        def resolve(self) -> dict[str, str]:
            """Type-safe accessor for the computed environment mapping."""
            remove_keys = frozenset(self.remove_keys)
            resolved = {
                key: value
                for key, value in self.base_env.items()
                if key not in remove_keys
            }
            resolved.update(dict(self.overrides))
            return resolved


__all__: list[str] = ["FlextCliModelsBase"]
