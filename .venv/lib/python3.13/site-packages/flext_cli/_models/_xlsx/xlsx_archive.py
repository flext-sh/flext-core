"""Typed safe-OOXML inspection contracts."""

from __future__ import annotations

from typing import Annotated

# mro-j47u (kimi): models consume the local t facade; m -> t is forward at runtime.
from flext_cli import t
from flext_core import m


class FlextCliModelsXlsxArchive:
    """Immutable archive policies and inspection evidence."""

    # NOTE (multi-agent, mro-j2yt.1): callers supply policy as typed data;
    # the inspector owns safe XML parsing without document-specific rules.
    class XlsxArchivePolicy(m.FrozenModel):
        max_members: Annotated[
            int, m.Field(ge=1, description="Maximum archive member count.")
        ] = 2048
        max_member_uncompressed_bytes: Annotated[
            int, m.Field(ge=1, description="Maximum uncompressed member bytes.")
        ] = 67_108_864
        max_total_uncompressed_bytes: Annotated[
            int, m.Field(ge=1, description="Maximum total uncompressed bytes.")
        ] = 268_435_456
        forbidden_members: frozenset[str] = m.Field(
            default_factory=frozenset,
            description="Exact archive members that are forbidden.",
        )
        forbidden_prefixes: tuple[str, ...] = m.Field(
            default=(), strict=False, description="Forbidden member prefixes."
        )
        forbidden_worksheet_tags: frozenset[str] = m.Field(
            default_factory=frozenset, description="Forbidden local worksheet XML tags."
        )
        required_worksheet_count: (
            Annotated[int, m.Field(ge=1, description="Required worksheet count.")]
            | None
        ) = m.Field(default=None, description="Optional exact sheet count.")
        reject_defined_names: bool = m.Field(
            default=False, description="Reject workbook defined names."
        )
        reject_style_protection: bool = m.Field(
            default=False, description="Reject non-default style protection."
        )
        allowed_locked_tokens: frozenset[str | None] = m.Field(
            default_factory=frozenset, description="Accepted OOXML locked tokens."
        )
        allowed_hidden_tokens: frozenset[str | None] = m.Field(
            default_factory=frozenset, description="Accepted OOXML hidden tokens."
        )

    class XlsxArchiveViolation(m.FrozenModel):
        kind: t.Cli.XlsxArchiveViolationKind = m.Field(description="Violation kind.")
        location: Annotated[
            str, m.Field(min_length=1, description="Archive violation location.")
        ]
        detail: Annotated[str, m.Field(min_length=1, description="Violation evidence.")]

    class XlsxArchiveInventory(m.FrozenModel):
        members: tuple[str, ...] = m.Field(
            default=(), strict=False, description="Ordered archive members."
        )
        blocked_members: frozenset[str] = m.Field(
            default_factory=frozenset,
            description="Members skipped because a safety limit was exceeded.",
        )
        total_uncompressed_bytes: Annotated[
            int, m.Field(ge=0, description="Declared total uncompressed bytes.")
        ]
        violations: tuple[FlextCliModelsXlsxArchive.XlsxArchiveViolation, ...] = (
            m.Field(default=(), strict=False, description="Inventory violations.")
        )

    class XlsxArchiveInspection(m.FrozenModel):
        member_count: Annotated[
            int, m.Field(ge=0, description="Discovered archive member count.")
        ]
        worksheet_count: Annotated[
            int, m.Field(ge=0, description="Discovered worksheet count.")
        ]
        total_uncompressed_bytes: Annotated[
            int, m.Field(ge=0, description="Declared total uncompressed bytes.")
        ]
        violations: tuple[FlextCliModelsXlsxArchive.XlsxArchiveViolation, ...] = (
            m.Field(default=(), strict=False, description="Policy violations.")
        )
        clean: bool = m.Field(description="Whether no violation was found.")

    class XlsxArchiveInspectionRequest(m.FrozenModel):
        source: Annotated[
            bytes, m.Field(min_length=1, description="Workbook archive bytes.")
        ]
        policy: FlextCliModelsXlsxArchive.XlsxArchivePolicy = m.Field(
            description="Inspection policy."
        )


__all__: tuple[str, ...] = ("FlextCliModelsXlsxArchive",)
