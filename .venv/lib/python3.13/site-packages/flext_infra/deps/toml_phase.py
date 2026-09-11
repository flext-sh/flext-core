"""TOML phase service for applying m.Infra.Deps.Toml.PhaseConfig to docs.

Uses one cached per-document path resolver plus
u.Cli.toml_* sync/navigate primitives,
s for bootstrap.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated, override

from flext_infra import m, p, r, s, t, u
from flext_infra.deps._toml_phase_ops import FlextInfraTomlPhaseOps


class FlextInfraTomlPhaseService(FlextInfraTomlPhaseOps, s[t.StrSequence]):
    """Apply ``m.Infra.Deps.Toml.PhaseConfig`` phases to a TOML document."""

    doc: Annotated[
        t.Cli.TomlDocument, m.Field(exclude=True, description="Target TOML document")
    ]
    phases: Annotated[
        t.SequenceOf[m.Infra.Deps.Toml.PhaseConfig],
        m.Field(
            default_factory=tuple,
            description="Ordered TOML transformation phases to apply.",
        ),
    ]
    _table_cache: dict[t.StrSequence, t.Cli.TomlTable] = u.PrivateAttr(
        default_factory=dict
    )

    @classmethod
    def apply_phases(
        cls, doc: t.Cli.TomlDocument, *phases: m.Infra.Deps.Toml.PhaseConfig
    ) -> t.StrSequence:
        """Apply a declarative phase set to one TOML document."""
        result: t.StrSequence = cls.model_construct(doc=doc, phases=phases).apply()
        return result

    @classmethod
    def apply_payload_phases(
        cls, payload: t.MutableJsonMapping, *phases: m.Infra.Deps.Toml.PhaseConfig
    ) -> t.StrSequence:
        """Apply one declarative phase set to one plain TOML payload."""
        return [
            change
            for phase in phases
            for change in cls._apply_payload_phase(payload, phase, parent_path=())
        ]

    @override
    def execute(self) -> p.Result[t.StrSequence]:
        """Apply all phases and return one flat change list."""
        return r[t.StrSequence].create_from_callable(
            lambda: tuple(
                change
                for phase in self.phases
                for change in self._apply_phase(phase, parent_path=())
            ),
            error_code="toml_phase_execute",
        )

    def apply(self) -> t.StrSequence:
        """Apply phases and return one flat change list."""
        return [
            change
            for phase in self.phases
            for change in self._apply_phase(phase, parent_path=())
        ]

    @classmethod
    def _apply_payload_phase(
        cls,
        payload: t.MutableJsonMapping,
        phase: m.Infra.Deps.Toml.PhaseConfig,
        *,
        parent_path: t.StrSequence,
    ) -> t.StrSequence:
        """Apply one phase recursively against one normalized payload."""
        if phase.custom_handler is not None:
            msg = f"payload phases do not support custom handlers: {phase.name}"
            raise ValueError(msg)
        out: t.MutableSequenceOf[str] = []
        phase_path = (*parent_path, *phase.root_path, *phase.table_path)
        table = u.Cli.toml_mapping_ensure_path(payload, phase_path)
        prefix = u.Cli.toml_dot_path(*phase_path)
        for operation in phase.operations:
            cls._apply_payload_operation(table, operation, out, prefix)
        for nested in phase.nested_tables:
            out.extend(
                cls._apply_payload_phase(payload, nested, parent_path=phase_path)
            )
        return out

    def _resolve_phase_table(self, phase_path: t.StrSequence) -> t.Cli.TomlTable:
        """Resolve and cache one TOML table path for the current document."""
        cached_table = self._table_cache.get(phase_path)
        if cached_table is not None:
            return cached_table
        table: t.Cli.TomlTable = u.Cli.toml_ensure_path(self.doc, phase_path)
        self._table_cache[phase_path] = table
        return table

    def _apply_phase(
        self, phase: m.Infra.Deps.Toml.PhaseConfig, *, parent_path: t.StrSequence
    ) -> t.StrSequence:
        """Apply one phase recursively using cached path resolution."""
        out: t.MutableSequenceOf[str] = []

        if (
            not phase.operations
            and not phase.nested_tables
            and phase.custom_handler is not None
            and not phase.table_path
        ):
            out.extend(phase.custom_handler(self.doc))
            return out

        phase_path = (*parent_path, *phase.root_path, *phase.table_path)
        tbl = self._resolve_phase_table(phase_path)
        pfx = u.Cli.toml_dot_path(*phase_path)

        for operation in phase.operations:
            self._apply_operation(tbl, operation, out, pfx)

        for nested in phase.nested_tables:
            out.extend(self._apply_phase(nested, parent_path=phase_path))

        if phase.custom_handler is not None:
            out.extend(phase.custom_handler(self.doc))

        return out


__all__: list[str] = ["FlextInfraTomlPhaseService"]
