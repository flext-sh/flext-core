"""Regressions for complete, detached settings lifecycle validation.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar
from uuid import UUID

import pytest

from flext_core import FlextSettings
from tests import m, t, tm, u

if TYPE_CHECKING:
    from pathlib import Path


class TestsFlextCoreSettingsStateValidation:
    """Validate stored values independently of instance trust and serialization."""

    @staticmethod
    @pytest.mark.parametrize("invalid_port", [0, 65536])
    def test_model_instance_overrides_revalidate_real_port_constraints(
        invalid_port: int,
    ) -> None:
        """The LDAP port constraint also applies to unvalidated model_copy input."""

        class LdapSettings(m.BaseModel):
            port: int = m.Field(default=389, ge=1, le=65535)

        class Owner(FlextSettings):
            Ldap: LdapSettings = m.Field(default_factory=LdapSettings)

        published = Owner.fetch_global()
        nested = published.Ldap
        original = published.model_dump()
        bad = nested.model_copy(update={"port": invalid_port})
        with pytest.raises(m.ValidationError, match="port"):
            Owner.update_global(Ldap=bad)
        with pytest.raises(m.ValidationError, match="port"):
            published.clone(Ldap=bad)
        tm.ok(published is Owner.fetch_global())
        tm.ok(published.Ldap is nested)
        tm.that(published.model_dump(), eq=original)

    @staticmethod
    def test_model_instances_inside_collections_are_revalidated() -> None:
        """An otherwise trusted branch cannot hide invalid models in a list."""

        class Connection(m.BaseModel):
            port: int = m.Field(default=389, ge=1, le=65535)

        class Connections(m.BaseModel):
            entries: list[Connection] = m.Field(default_factory=list[Connection])

        class Owner(FlextSettings):
            connections: Connections = m.Field(default_factory=Connections)

        published = Owner.fetch_global()
        nested = published.connections
        bad = nested.model_copy(
            update={"entries": [Connection().model_copy(update={"port": 0})]}
        )
        with pytest.raises(m.ValidationError, match="port"):
            Owner.update_global(connections=bad)
        tm.ok(published is Owner.fetch_global())
        tm.ok(published.connections is nested)
        tm.that(nested.entries, eq=[])

    @staticmethod
    def test_required_overrides_join_sources_without_publishing_a_root(
        tmp_path: Path,
    ) -> None:
        """A missing or invalid source value may be supplied by an isolated override."""

        class Owner(FlextSettings):
            required_value: int = m.Field(ge=1)
            external_value: int

        selected_required = 31
        selected_external = 7
        env_file = tmp_path / "selected.env"
        env_file.write_text(
            f"FLEXT_EXTERNAL_VALUE={selected_external}\n", encoding="utf-8"
        )
        with u.Tests.env_vars_context(
            env_vars={
                "FLEXT_ENV_FILE": str(env_file),
                "FLEXT_REQUIRED_VALUE": "invalid-but-overridden",
            },
            vars_to_clear=("FLEXT_EXTERNAL_VALUE",),
        ):
            snapshot = Owner.fetch_global(
                overrides={"required_value": selected_required}
            )
        tm.that(snapshot.required_value, eq=selected_required)
        tm.that(snapshot.external_value, eq=selected_external)
        next_required = selected_required + 1
        with u.Tests.env_vars_context(
            env_vars={
                "FLEXT_REQUIRED_VALUE": str(next_required),
                "FLEXT_EXTERNAL_VALUE": str(selected_external),
            },
        ):
            published = Owner.fetch_global()
        tm.ok(published is not snapshot)
        tm.that(published.required_value, eq=next_required)

    @staticmethod
    def test_first_update_validates_required_fields_before_publication() -> None:
        """An invalid first write stays unpublished; a valid first write is usable."""

        class Owner(FlextSettings):
            required_value: int = m.Field(ge=1)

        with u.Tests.env_vars_context(
            env_vars={}, vars_to_clear=("FLEXT_REQUIRED_VALUE",)
        ):
            with pytest.raises(m.ValidationError, match="required_value"):
                Owner.update_global(required_value=0)
            with pytest.raises(m.ValidationError, match="required_value"):
                Owner.fetch_global()
            selected = 11
            published = Owner.update_global(required_value=selected)
            tm.ok(published is Owner.fetch_global())
            tm.that(published.required_value, eq=selected)

    @staticmethod
    def test_unrelated_updates_preserve_all_stored_fields() -> None:
        """Validation reads stored fields, including required and masked values."""

        class Owner(FlextSettings):
            required_hidden: str = m.Field(exclude=True)
            retained: str = m.Field(default="fallback", exclude=True)
            conditional: str = m.Field(
                default="visible", exclude_if=lambda value: value == "hidden"
            )
            serialized: int = 3

            @m.field_serializer("serialized")
            def serialize_value(self, value: int) -> str:
                return f"rendered:{value}"

        selected_required = "required-input"
        selected_retained = "retained-input"
        selected_conditional = "hidden"
        with u.Tests.env_vars_context(
            env_vars={"FLEXT_REQUIRED_HIDDEN": selected_required},
        ):
            published = Owner.fetch_global()
            selected_serialized = published.serialized
            Owner.update_global(
                retained=selected_retained, conditional=selected_conditional
            )
            selected_debug = not published.debug
            snapshot = published.clone(debug=selected_debug)
            updated = Owner.update_global(debug=selected_debug)
        tm.ok(updated is published is Owner.fetch_global())
        for instance in (published, snapshot):
            tm.that(instance.required_hidden, eq=selected_required)
            tm.that(instance.retained, eq=selected_retained)
            tm.that(instance.conditional, eq=selected_conditional)
            tm.that(instance.serialized, eq=selected_serialized)
            for excluded in ("required_hidden", "retained", "conditional"):
                tm.that(instance.model_dump(), lacks=excluded)
        tm.ok(snapshot is not published)

    @staticmethod
    def test_nested_excluded_fields_survive_partial_overrides() -> None:
        """Partial branch updates retain all stored fields, not only dump output."""

        class Branch(m.BaseModel):
            required_hidden: str = m.Field(exclude=True)
            retained: str = m.Field(default="fallback", exclude=True)
            port: int = m.Field(default=389, ge=1, le=65535)

        class Owner(FlextSettings):
            branch: Branch = m.Field(
                default_factory=lambda: Branch(required_hidden="nested-input")
            )

        published = Owner.fetch_global()
        selected_hidden = published.branch.required_hidden
        selected_retained = "selected"
        Owner.update_global(branch={"retained": selected_retained})
        selected_port = published.branch.port + 1
        updated = Owner.update_global(branch={"port": selected_port})
        tm.ok(updated is published)
        tm.that(published.branch.required_hidden, eq=selected_hidden)
        tm.that(published.branch.retained, eq=selected_retained)
        tm.that(published.branch.port, eq=selected_port)

    @staticmethod
    def test_fresh_alias_paths_accept_field_name_overrides() -> None:
        """Shared and indexed AliasPaths resolve before field-name overrides."""

        class Owner(FlextSettings):
            model_config: ClassVar[m.SettingsConfigDict] = m.SettingsConfigDict(
                extra="forbid", populate_by_name=False
            )
            endpoint: str = m.Field(validation_alias=m.AliasPath("connection", "host"))
            port: int = m.Field(
                ge=1, le=65535, validation_alias=m.AliasPath("connection", "ports", 0)
            )

        selected_endpoint = "override.example"
        with u.Tests.env_vars_context(
            env_vars={"CONNECTION": '{"host":"source.example","ports":[1389]}'},
        ):
            snapshot = Owner.fetch_global(overrides={"endpoint": selected_endpoint})
            published = Owner.fetch_global()
        tm.ok(snapshot is not published)
        tm.that(snapshot.endpoint, eq=selected_endpoint)
        tm.that(snapshot.port, eq=published.port)
        tm.that(snapshot.endpoint, ne=published.endpoint)
        selected_port = published.port + 1
        updated = Owner.update_global(port=selected_port)
        tm.ok(updated is published)
        tm.that(published.port, eq=selected_port)

    @staticmethod
    def test_partial_nested_alias_paths_preserve_stored_siblings() -> None:
        """Partial AliasPath input and canonical field names share one branch owner."""

        class Branch(m.BaseModel):
            model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
                extra="forbid", populate_by_name=False
            )
            endpoint: str = m.Field(
                default="initial", validation_alias=m.AliasPath("connection", "host")
            )
            port: int = m.Field(
                default=389,
                ge=1,
                le=65535,
                validation_alias=m.AliasPath("connection", "ports", 0),
            )

        class Owner(FlextSettings):
            branch: Branch = m.Field(default_factory=Branch)

        published = Owner.fetch_global()
        selected_port = published.branch.port + 1
        selected_endpoint = "selected"
        updated = Owner.update_global(
            branch={"connection": {"host": selected_endpoint, "ports": [selected_port]}}
        )
        tm.ok(updated is published)
        tm.that(published.branch.endpoint, eq=selected_endpoint)
        tm.that(published.branch.port, eq=selected_port)
        snapshot = published.clone(branch={"endpoint": "snapshot"})
        tm.that(snapshot.branch.port, eq=selected_port)
        tm.that(published.branch.endpoint, eq=selected_endpoint)
        tm.ok(snapshot.branch is not published.branch)

    @staticmethod
    @pytest.mark.parametrize("tuple_key", [False, True])
    def test_invalid_mapping_keys_preserve_published_state(tuple_key: bool) -> None:
        """Frozen model keys, including tuple members, cannot bypass constraints."""

        class PortKey(m.BaseModel, frozen=True):
            port: int = m.Field(default=389, ge=1, le=65535)

        class Branch(m.BaseModel):
            entries: t.MutableMappingKV[PortKey | tuple[PortKey, str], str]

        class Owner(FlextSettings):
            branch: Branch

        valid_key = PortKey()
        initial_entries: t.MutableMappingKV[PortKey | tuple[PortKey, str], str] = {
            valid_key: "direct",
            (valid_key, "tuple"): "tuple",
        }
        published = Owner.update_global(branch=Branch(entries=initial_entries))
        original_branch = published.branch
        original_mapping = original_branch.entries
        original_contents = tuple(original_mapping.items())
        invalid_port = 0
        bad_key = valid_key.model_copy(update={"port": invalid_port})
        bad_mapping_key = (bad_key, "tuple") if tuple_key else bad_key
        bad_branch = original_branch.model_copy(
            update={"entries": {bad_mapping_key: "invalid"}}
        )
        for operation in (Owner.update_global, published.clone):
            with pytest.raises(m.ValidationError, match="port"):
                operation(branch=bad_branch)
            tm.ok(published is Owner.fetch_global())
            tm.ok(published.branch is original_branch)
            tm.ok(original_branch.entries is original_mapping)
            tm.that(tuple(original_mapping.items()), eq=original_contents)
            tm.that(original_mapping[valid_key], eq=initial_entries[valid_key])
            tm.that(
                original_mapping[valid_key, "tuple"],
                eq=initial_entries[valid_key, "tuple"],
            )
            tm.that(bad_key.port, eq=invalid_port)
            tm.that(bad_branch.entries[bad_mapping_key], eq="invalid")

    @staticmethod
    def test_mapping_keys_normalize_without_corrupting_native_keys() -> None:
        """Validated key copies retain native opaque keys and usable tuple hashes."""

        class PortKey(m.BaseModel, frozen=True):
            port: int = m.Field(default=389, ge=1, le=65535)

        class Branch(m.BaseModel):
            entries: t.MutableMappingKV[PortKey | tuple[PortKey, UUID] | UUID, str]

        class Owner(FlextSettings):
            branch: Branch

        selected_port = 1389
        raw_key = PortKey().model_copy(update={"port": str(selected_port)})
        opaque_key = UUID(int=1)
        original_entries: t.MutableMappingKV[
            PortKey | tuple[PortKey, UUID] | UUID, str
        ] = {
            raw_key: "model",
            (raw_key, opaque_key): "tuple",
            opaque_key: "opaque",
        }
        branch = Branch(entries=original_entries)
        original_contents = tuple(branch.entries.items())
        published = Owner.update_global(branch=branch)
        snapshot = published.clone(branch=branch)
        normalized_key = PortKey(port=selected_port)
        for instance in (published, snapshot):
            tm.that(
                instance.branch.entries[normalized_key], eq=original_entries[raw_key]
            )
            tm.that(
                instance.branch.entries[normalized_key, opaque_key],
                eq=original_entries[raw_key, opaque_key],
            )
            tm.that(
                instance.branch.entries[opaque_key], eq=original_entries[opaque_key]
            )
        tm.ok(snapshot is not published)
        tm.ok(snapshot.branch is not published.branch)
        tm.that(tuple(branch.entries.items()), eq=original_contents)
        tm.that(raw_key.port, eq=str(selected_port))
        tm.that(branch.entries[raw_key], eq=original_entries[raw_key])
