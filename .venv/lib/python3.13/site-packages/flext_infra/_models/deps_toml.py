"""TOML phase models with Builder DSL for deps configuration sync.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import chain
from typing import Annotated, Literal, Self, override

from flext_core import m
from flext_infra import c, t


class FlextInfraModelsDepsToml:
    """TOML operation models exposed through ``m.Infra.Deps.Toml``."""

    class Deps:
        """Dependency-management model domains."""

        class Toml:
            """Declarative TOML sync model domain."""

            class SetOp(m.ContractModel):
                """Set one TOML key to one JSON-compatible value."""

                kind: Literal[c.Infra.TomlOperationKind.SET] = m.Field(
                    c.Infra.TomlOperationKind.SET,
                    description="Operation kind",
                    validate_default=True,
                )
                key: str = m.Field(description="TOML key name")
                value: t.JsonValue = m.Field(description="JSON-compatible value")

            class ListOp(m.ContractModel):
                """Set or merge one TOML string list."""

                kind: Literal[c.Infra.TomlOperationKind.LIST] = m.Field(
                    c.Infra.TomlOperationKind.LIST,
                    description="Operation kind",
                    validate_default=True,
                )
                key: str = m.Field(description="TOML key name")
                values: t.StrSequence = m.Field(description="Expected values")
                strategy: Annotated[
                    c.Infra.TomlMergeMode,
                    m.Field(description="Merge strategy", validate_default=True),
                ] = c.Infra.TomlMergeMode.REPLACE
                sort: Annotated[
                    bool,
                    m.Field(
                        description="Sort values before sync", validate_default=True
                    ),
                ] = True

            class RemoveOp(m.ContractModel):
                """Remove one TOML key, optionally from a nested relative table."""

                kind: Literal[c.Infra.TomlOperationKind.REMOVE] = m.Field(
                    c.Infra.TomlOperationKind.REMOVE,
                    description="Operation kind",
                    validate_default=True,
                )
                key: str = m.Field(description="Key to remove")
                table_path: Annotated[
                    t.StrSequence,
                    m.Field(
                        description="Relative sub-table path", validate_default=True
                    ),
                ] = ()

            type Operation = Annotated[
                SetOp | ListOp | RemoveOp, m.Field(discriminator="kind")
            ]

            class PhaseConfig(m.ContractModel):
                """Declarative TOML phase with inline Builder DSL."""

                name: str = m.Field(description="Phase name")
                root_path: Annotated[
                    t.StrSequence, m.Field(description="Root path before table_path")
                ] = (c.Infra.TOOL,)
                table_path: Annotated[
                    t.StrSequence, m.Field(description="Primary table path")
                ] = ()
                operations: Annotated[
                    t.SequenceOf[FlextInfraModelsDepsToml.Deps.Toml.Operation],
                    m.Field(description="Declarative TOML operations"),
                ] = ()
                nested_tables: Annotated[
                    t.SequenceOf[FlextInfraModelsDepsToml.Deps.Toml.PhaseConfig],
                    m.Field(description="Nested TOML phase configs"),
                ] = ()
                custom_handler: Annotated[
                    Callable[..., t.StrSequence] | None,
                    m.Field(exclude=True, description="Custom handler"),
                ] = None

                class Builder(
                    m.Builder.Identity["FlextInfraModelsDepsToml.Deps.Toml.PhaseConfig"]
                ):
                    """Fluent builder for ``m.Infra.Deps.Toml.PhaseConfig``."""

                    def __init__(self, name: str) -> None:
                        super().__init__(
                            state=FlextInfraModelsDepsToml.Deps.Toml.PhaseConfig(
                                name=name
                            )
                        )

                    @override
                    def build(self) -> FlextInfraModelsDepsToml.Deps.Toml.PhaseConfig:
                        """Return the typed immutable phase owned by this builder."""
                        phase: FlextInfraModelsDepsToml.Deps.Toml.PhaseConfig = (
                            self.state
                        )
                        return phase

                    @classmethod
                    def _nested_operations(
                        cls,
                        *,
                        values: t.SequenceOf[tuple[str, t.JsonValue]] = (),
                        lists: t.SequenceOf[t.StrSequencePair] = (),
                        deprecated_keys: t.StrSequence = (),
                    ) -> tuple[FlextInfraModelsDepsToml.Deps.Toml.Operation, ...]:
                        """Nested operations."""
                        return tuple(
                            chain(
                                (
                                    FlextInfraModelsDepsToml.Deps.Toml.SetOp(
                                        key=key, value=value
                                    )
                                    for key, value in values
                                ),
                                (
                                    FlextInfraModelsDepsToml.Deps.Toml.ListOp(
                                        key=key, values=tuple(entries)
                                    )
                                    for key, entries in lists
                                ),
                                (
                                    FlextInfraModelsDepsToml.Deps.Toml.RemoveOp(key=key)
                                    for key in deprecated_keys
                                ),
                            )
                        )

                    def operation(
                        self,
                        operation_type: type[m.ContractModel],
                        /,
                        **data: t.JsonValue
                        | t.JsonPayload
                        | t.SequenceOf[t.JsonPayload],
                    ) -> Self:
                        """Operation."""
                        operation_item = operation_type.model_validate(data)
                        replaced: Self = self._replace(
                            self.state.model_copy(
                                update={
                                    "operations": (
                                        *self.state.operations,
                                        operation_item,
                                    )
                                }
                            )
                        )
                        return replaced

                    def root(self, *path: str) -> Self:
                        """Root."""
                        result: Self = self._path("root_path", *path)
                        return result

                    def table(self, *path: str) -> Self:
                        """Table."""
                        result: Self = self._path("table_path", *path)
                        return result

                    def value(self, key: str, value: t.JsonValue) -> Self:
                        """Value."""
                        return self.operation(
                            FlextInfraModelsDepsToml.Deps.Toml.SetOp,
                            key=key,
                            value=value,
                        )

                    def list(
                        self,
                        key: str,
                        values: t.StrSequence,
                        *,
                        strategy: c.Infra.TomlMergeMode = c.Infra.TomlMergeMode.REPLACE,
                        sort: bool = True,
                    ) -> Self:
                        """List."""
                        return self.operation(
                            FlextInfraModelsDepsToml.Deps.Toml.ListOp,
                            key=key,
                            values=tuple(values),
                            strategy=strategy,
                            sort=sort,
                        )

                    def deprecated(self, key: str, *sub_path: str) -> Self:
                        """Mark a key as deprecated by scheduling its removal."""
                        return self.operation(
                            FlextInfraModelsDepsToml.Deps.Toml.RemoveOp,
                            key=key,
                            table_path=tuple(sub_path),
                        )

                    def nested(
                        self,
                        *path: str,
                        values: t.SequenceOf[tuple[str, t.JsonValue]] = (),
                        lists: t.SequenceOf[t.StrSequencePair] = (),
                        deprecated_keys: t.StrSequence = (),
                    ) -> Self:
                        """Nested."""
                        nested_table = FlextInfraModelsDepsToml.Deps.Toml.PhaseConfig(
                            name=self.state.name,
                            root_path=(),
                            table_path=tuple(path),
                            operations=tuple(
                                self._nested_operations(
                                    values=values,
                                    lists=lists,
                                    deprecated_keys=deprecated_keys,
                                )
                            ),
                        )
                        replaced: Self = self._replace(
                            self.state.model_copy(
                                update={
                                    "nested_tables": (
                                        *self.state.nested_tables,
                                        nested_table,
                                    )
                                }
                            )
                        )
                        return replaced

                    def handler(self, fn: Callable[..., t.StrSequence]) -> Self:
                        """Set a custom handler function."""
                        replaced: Self = self._replace(
                            self.state.model_copy(update={"custom_handler": fn})
                        )
                        return replaced


__all__: list[str] = ["FlextInfraModelsDepsToml"]
