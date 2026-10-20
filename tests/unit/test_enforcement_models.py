"""Model and field enforcement tests.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from collections.abc import MutableSequence
from pathlib import Path
from types import MappingProxyType
from typing import Annotated

import pytest

from flext_core.utilities import FlextUtilitiesEnforcement
from tests import c, m, t, tm, u
from tests.unit._enforcement_support import invalid_annotations, messages

type _BARE_DICT_TYPE = t.MappingKV[str, str]


class TestsFlextEnforcementModels:
    """Tests for ``FlextEnforcementModels``."""

    @staticmethod
    def test_fix_action_defaults_are_independent_validated_json_mappings() -> None:
        """Cold-imported fix contracts retain fresh defaults and JSON validation."""
        first = m.EnforcementFixAction(kind="manual", target="native-default")
        second = m.EnforcementFixAction(kind="manual", target="native-default")

        tm.that(first.params, eq={})
        tm.that(second.params, eq={})
        tm.that(first.params is not second.params, truth=True)
        restored = m.EnforcementFixAction.model_validate_json(first.model_dump_json())
        tm.that(restored, eq=first)
        with pytest.raises(m.ValidationError):
            m.EnforcementFixAction.model_validate({
                "kind": "manual",
                "target": "native-default",
                "params": {"unsupported": Path("not-json")},
            })

    @staticmethod
    def test_context_defaults_are_independent_readonly_contract_values() -> None:
        """Bootstrap defaults validate without sharing context collection state."""
        first_export = m.ContextExport()
        second_export = m.ContextExport()
        first_state = m.ContextRuntimeState()
        second_state = m.ContextRuntimeState()

        tm.that(first_export.data == second_export.data, truth=True)
        tm.that(first_export.data is not second_export.data, truth=True)
        tm.that(first_export.statistics == second_export.statistics, truth=True)
        tm.that(first_export.statistics is not second_export.statistics, truth=True)
        tm.that(
            first_state.statistics.operations
            == second_state.statistics.operations
            == {},
            truth=True,
        )
        tm.that(
            first_state.statistics.operations,
            where=lambda operations: (
                operations is not second_state.statistics.operations
            ),
        )
        tm.that(first_state.hooks == second_state.hooks, truth=True)
        tm.that(first_state.hooks is not second_state.hooks, truth=True)
        tm.that(first_state.scope_vars == second_state.scope_vars, truth=True)
        tm.that(first_state.scope_vars is not second_state.scope_vars, truth=True)
        for model in (m.ContextExport, m.ContextStatistics, m.ContextRuntimeState):
            tm.that(
                messages(u.check(model), fragment="no_raw_collections_field_default"),
                truth=False,
            )

    @staticmethod
    def test_any_field_detected(tmp_path: Path) -> None:
        """Test any field detected."""
        with invalid_annotations(tmp_path) as source:
            target = source.BadModel
            if not isinstance(target, type):
                pytest.fail("Invalid source fixture did not declare a model class")
            tm.ok(bool(messages(u.check(target), fragment="Any")))

    @staticmethod
    def test_typed_field_passes() -> None:
        """Test typed field passes."""

        class _M(m.ArbitraryTypesModel):
            name: Annotated[str, m.Field(description="d")] = "x"

        tm.that(messages(u.check(_M), fragment="Any"), truth=False)

    @staticmethod
    def test_bare_dict_detected() -> None:
        """Test bare dict detected."""

        class _M(m.ArbitraryTypesModel):
            data: _BARE_DICT_TYPE = m.Field(default_factory=dict, description="d")

        tm.that(
            messages(u.check(_M), fragment="no_raw_collections_field_default"),
            truth=True,
        )

    @staticmethod
    def test_bare_list_detected() -> None:
        """Test bare list detected."""

        class _M(m.ArbitraryTypesModel):
            items: list[str] = m.Field(default_factory=list, description="d")

        tm.that(
            messages(u.check(_M), fragment="no_raw_collections_field_default"),
            truth=True,
        )

    @staticmethod
    def test_mapping_passes() -> None:
        """Test mapping passes."""

        class _M(m.ArbitraryTypesModel):
            data: Annotated[
                t.StrMapping,
                m.Field(
                    description="d",
                ),
            ] = m.Field(default_factory=lambda: MappingProxyType[str, str]({}))

        tm.that(
            messages(u.check(_M), fragment="no_raw_collections_field_default"),
            truth=False,
        )

    @staticmethod
    def test_mutable_sequence_list_factory_passes() -> None:
        """Test mutable sequence list factory passes."""

        class _M(m.ArbitraryTypesModel):
            items: Annotated[MutableSequence[str], m.Field(description="d")] = m.Field(
                default_factory=list
            )

        tm.that(messages(u.check(_M), fragment="read-only field contract"), truth=False)

    @staticmethod
    def test_mutable_sequence_alias_nested_owner_defaults_pass() -> None:
        """An unresolved element does not change the declared mutable origin."""

        class _M(m.ArbitraryTypesModel):
            class Value(m.ContractModel):
                name: Annotated[str, m.Field(description="Value name")] = "x"

            items: t.MutableSequenceOf[_M.Value] = m.Field(
                default_factory=list,
                validate_default=True,
                description="Mutable sequence of nested values.",
            )

        tm.that(messages(u.check(_M), fragment="read-only field contract"), truth=False)
        first = _M()
        second = _M()
        tm.that(first.items == second.items, truth=True)
        tm.that(first.items is not second.items, truth=True)
        value = _M.Value()
        first.items.append(value)
        tm.that(first.items, eq=[value])
        tm.that(second.items, eq=[])
        provided: t.MutableSequenceOf[_M.Value] = [value]
        tm.that(_M(items=provided).items, eq=provided)
        tm.that(messages(u.check(_M), fragment="read-only field contract"), truth=False)
        with pytest.raises(m.ValidationError):
            _M.model_validate({"items": 42})

    @staticmethod
    def test_annotated_mutable_sequence_alias_nested_owner_passes() -> None:
        """Annotated metadata leaves the bound outer alias and defaults intact."""

        class _M(m.ArbitraryTypesModel):
            class Value(m.ContractModel):
                name: Annotated[str, m.Field(description="Value name")] = "x"

            items: Annotated[
                t.MutableSequenceOf[_M.Value],
                m.Field(description="Annotated mutable sequence of nested values."),
            ] = m.Field(default_factory=list, validate_default=True)

        tm.that(messages(u.check(_M), fragment="read-only field contract"), truth=False)
        first = _M()
        second = _M()
        tm.that(first.items == second.items, truth=True)
        tm.that(first.items is not second.items, truth=True)

    @staticmethod
    def test_mutable_mapping_forward_ref_dict_factory_passes() -> None:
        """Test mutable mapping forward ref dict factory passes."""

        class _M(m.ArbitraryTypesModel):
            class Value(m.ContractModel):
                name: Annotated[str, m.Field(description="Value name")] = "x"

            items: typing.MutableMapping[str, _M.Value] = m.Field(
                default_factory=dict,
                description="Mutable mapping contract.",
            )

        tm.that(messages(u.check(_M), fragment="read-only field contract"), truth=False)

    @staticmethod
    def test_mutable_mapping_kv_forward_ref_dict_factory_passes() -> None:
        """Test mutable mapping KV alias forward ref dict factory passes.

        A ``t.MutableMappingKV`` annotation behind a forward reference (e.g. a
        self-referential model class) must still be recognised as a mutable
        mapping contract, not flagged as a raw ``dict`` default.
        """

        class _M(m.ArbitraryTypesModel):
            class Value(m.ContractModel):
                name: Annotated[str, m.Field(description="Value name")] = "x"

            items: t.MutableMappingKV[str, _M.Value] = m.Field(
                default_factory=dict,
                validate_default=True,
                description="Mutable mapping KV contract.",
            )

        tm.that(messages(u.check(_M), fragment="read-only field contract"), truth=False)
        first = _M()
        second = _M()
        tm.that(first.items == second.items, truth=True)
        tm.that(first.items is not second.items, truth=True)
        value = _M.Value()
        first.items["value"] = value
        tm.that(first.items, eq={"value": value})
        tm.that(second.items, eq={})
        tm.that(messages(u.check(_M), fragment="read-only field contract"), truth=False)

    @staticmethod
    def test_mutable_collection_unions_keep_pre_and_post_completion_policy() -> None:
        """Native completion never turns a non-exact mutable union into an error."""

        class _M(m.ArbitraryTypesModel):
            class Value(m.ContractModel):
                name: Annotated[str, m.Field(description="Value name")] = "x"

            sequence_union: t.MutableSequenceOf[_M.Value] | None = m.Field(
                default_factory=list,
                validate_default=True,
                description="Optional mutable sequence with a PEP 604 union.",
            )
            sequence_optional: t.MutableSequenceOf[_M.Value] | None = m.Field(
                default_factory=list,
                validate_default=True,
                description="Optional mutable sequence with a typing contract.",
            )
            sequence_typing_union: t.MutableSequenceOf[_M.Value] | None = m.Field(
                default_factory=list,
                validate_default=True,
                description="Mutable sequence with a native typing union.",
            )
            mapping_union: t.MutableMappingKV[str, _M.Value] | None = m.Field(
                default_factory=dict,
                validate_default=True,
                description="Optional mutable mapping with a PEP 604 union.",
            )
            mapping_optional: t.MutableMappingKV[str, _M.Value] | None = m.Field(
                default_factory=dict,
                validate_default=True,
                description="Optional mutable mapping with a typing contract.",
            )
            mapping_typing_union: t.MutableMappingKV[str, _M.Value] | None = m.Field(
                default_factory=dict,
                validate_default=True,
                description="Mutable mapping with a native typing union.",
            )

        before = messages(u.check(_M), fragment="read-only field contract")
        tm.that(len(before), eq=6)
        instance = _M()
        tm.that(instance.sequence_union == instance.sequence_optional, truth=True)
        tm.that(instance.sequence_typing_union, eq=[])
        tm.that(instance.mapping_union == instance.mapping_optional, truth=True)
        tm.that(instance.mapping_typing_union, eq={})
        tm.that(messages(u.check(_M), fragment="read-only field contract"), eq=before)

    @staticmethod
    def test_mutable_json_mapping_alias_dict_factory_passes() -> None:
        """Test mutable json mapping alias dict factory passes."""

        class _M(m.ArbitraryTypesModel):
            items: Annotated[
                t.MutableJsonMapping,
                m.Field(
                    description="Mutable JSON mapping contract.",
                ),
            ] = m.Field(default_factory=dict)

        tm.that(messages(u.check(_M), fragment="read-only field contract"), truth=False)

    @staticmethod
    def test_sequence_list_factory_detected() -> None:
        """Test sequence list factory detected."""

        class _M(m.ArbitraryTypesModel):
            items: Annotated[t.StrSequence, m.Field(description="d")] = m.Field(
                default_factory=list
            )

        tm.that(messages(u.check(_M), fragment="read-only field contract"), truth=True)

    @staticmethod
    def test_sequence_alias_nested_owner_list_factory_detected() -> None:
        """A genuinely read-only origin remains rejected before element resolution."""

        class _M(m.ArbitraryTypesModel):
            class Value(m.ContractModel):
                name: Annotated[str, m.Field(description="Value name")] = "x"

            items: t.SequenceOf[_M.Value] = m.Field(
                default_factory=list,
                validate_default=True,
                description="Read-only sequence of nested values.",
            )

        tm.that(messages(u.check(_M), fragment="read-only field contract"), truth=True)

    @staticmethod
    def test_missing_description_detected() -> None:
        """Test missing description detected."""

        class _M(m.ArbitraryTypesModel):
            name: str = "test"

        tm.that(messages(u.check(_M), fragment="missing description"), truth=True)

    @staticmethod
    def test_description_present_passes() -> None:
        """Test description present passes."""

        class _M(m.ArbitraryTypesModel):
            name: Annotated[str, m.Field(description="A name")] = "test"

        tm.that(messages(u.check(_M), fragment="missing description"), truth=False)

    @staticmethod
    def test_v1_config_class_detected() -> None:
        """Test v1 config class detected."""
        with pytest.warns(DeprecationWarning, match="class-based"):

            class _M(m.ArbitraryTypesModel):
                class Config:
                    extra = "forbid"

                name: Annotated[str, m.Field(description="d")] = "x"

        tm.that(messages(u.check(_M), fragment="Pydantic v1"), truth=True)

    @staticmethod
    def test_flexible_internal_allows_ignore() -> None:
        """Test flexible internal allows ignore."""

        class _M(m.FlexibleInternalModel):
            name: Annotated[str, m.Field(description="d")] = "x"

        tm.that(messages(u.check(_M), fragment="extra"), truth=False)

    @staticmethod
    def test_mode_default_is_warn() -> None:
        """Test mode default is warn."""
        tm.that(c.ENFORCEMENT_MODE is c.EnforcementMode.WARN, truth=True)

    @staticmethod
    def test_enforcement_rules_loaded() -> None:
        """Test enforcement rules loaded."""
        tm.that(len(c.ENFORCEMENT_TAG_CATEGORY) > 0, truth=True)
        tm.that(
            all(
                cat in c.EnforcementCategory
                for cat in c.ENFORCEMENT_TAG_CATEGORY.values()
            ),
            truth=True,
        )
        # Every runtime rule row carries its own problem/fix text.
        tm.that(
            set(c.ENFORCEMENT_TAG_CATEGORY) <= set(c.ENFORCEMENT_RULES_TEXT), truth=True
        )

    @staticmethod
    def test_canonical_flext_core_class_satisfies_prefix_contract() -> None:
        """A correctly named ``flext_core`` class raises no class-prefix violation."""
        tm.that(
            messages(u.check(FlextUtilitiesEnforcement), fragment="class_prefix"),
            truth=False,
        )

    @pytest.mark.parametrize(
        ("module", "expect_prefix_violation"),
        [("flext_core.synthetic_module", True), ("fence", False)],
    )
    @staticmethod
    def test_class_prefix_enforced_only_for_knowable_projects(
        module: str,
        *,
        expect_prefix_violation: bool,
    ) -> None:
        """``flext_core`` demands the ``Flext`` prefix; doc-fence modules stay silent.

        A misnamed class inside the canonical ``flext_core`` src layer must be
        reported as missing the derived ``Flext`` project prefix, while a class
        whose module is a synthetic markdown code-fence (``fence``) is unknowable
        and must never have a prefix invented for it.
        """
        misnamed: type = type("CreateUserService", (), {})
        misnamed.__module__ = module
        misnamed.__qualname__ = "CreateUserService"

        hits = messages(u.check(misnamed), fragment="class_prefix")

        tm.that(bool(hits) is expect_prefix_violation, truth=True)
        if expect_prefix_violation:
            tm.that(any('prefix "Flext"' in message for message in hits), truth=True)
