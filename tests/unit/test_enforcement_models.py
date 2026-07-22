"""Model and field enforcement tests."""

from __future__ import annotations

import typing
from collections.abc import MutableSequence
from types import MappingProxyType
from typing import Annotated

import pytest
from pydantic.warnings import PydanticDeprecatedSince20

from flext_core._utilities.enforcement import FlextUtilitiesEnforcement
from tests.constants import c
from tests.models import m
from tests.typings import t
from tests.unit._enforcement_support import messages
from tests.utilities import u


class TestsFlextEnforcementModels:
    def test_any_field_detected(self) -> None:
        class _M(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        assert messages(u.check(_M), fragment="Any")

    def test_typed_field_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            name: Annotated[str, m.Field(description="d")] = "x"

        assert not messages(u.check(_M), fragment="Any")

    def test_bare_dict_detected(self) -> None:
        class _M(m.ArbitraryTypesModel):
            data: dict[str, str] = m.Field(default_factory=dict, description="d")

        assert messages(u.check(_M), fragment="bare ")

    def test_bare_list_detected(self) -> None:
        class _M(m.ArbitraryTypesModel):
            items: list[str] = m.Field(default_factory=list, description="d")

        assert messages(u.check(_M), fragment="bare ")

    def test_mapping_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            data: Annotated[
                t.StrMapping,
                m.Field(default_factory=lambda: MappingProxyType({}), description="d"),
            ]

        assert not messages(u.check(_M), fragment="bare ")

    def test_mutable_sequence_list_factory_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            items: Annotated[
                MutableSequence[str], m.Field(default_factory=list, description="d")
            ]

        assert not messages(u.check(_M), fragment="read-only field contract")

    def test_mutable_mapping_forward_ref_dict_factory_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            class Value(m.ContractModel):
                name: Annotated[str, m.Field(description="Value name")] = "x"

            items: typing.MutableMapping[str, _M.Value] = m.Field(
                default_factory=dict, description="Mutable mapping contract."
            )

        assert not messages(u.check(_M), fragment="read-only field contract")

    def test_mutable_json_mapping_alias_dict_factory_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            items: Annotated[
                t.MutableJsonMapping,
                m.Field(
                    default_factory=lambda: MappingProxyType({}),
                    description="Mutable JSON mapping contract.",
                ),
            ]

        assert not messages(u.check(_M), fragment="read-only field contract")

    def test_sequence_list_factory_detected(self) -> None:
        class _M(m.ArbitraryTypesModel):
            items: Annotated[
                t.StrSequence, m.Field(default_factory=list, description="d")
            ]

        assert messages(u.check(_M), fragment="read-only field contract")

    def test_missing_description_detected(self) -> None:
        class _M(m.ArbitraryTypesModel):
            name: str = "test"

        assert messages(u.check(_M), fragment="missing description")

    def test_description_present_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            name: Annotated[str, m.Field(description="A name")] = "test"

        assert not messages(u.check(_M), fragment="missing description")

    def test_v1_config_class_detected(self) -> None:
        with pytest.warns(PydanticDeprecatedSince20):

            class _M(m.ArbitraryTypesModel):
                class Config:
                    extra = "forbid"

                name: Annotated[str, m.Field(description="d")] = "x"

        assert messages(u.check(_M), fragment="Pydantic v1")

    def test_flexible_internal_allows_ignore(self) -> None:
        class _M(m.FlexibleInternalModel):
            name: Annotated[str, m.Field(description="d")] = "x"

        assert not messages(u.check(_M), fragment="extra")

    def test_mode_default_is_warn(self) -> None:
        assert c.ENFORCEMENT_MODE is c.EnforcementMode.WARN

    def test_enforcement_rules_loaded(self) -> None:
        assert len(c.ENFORCEMENT_RULES_TEXT) > 0
        assert len(c.ENFORCEMENT_TAG_CATEGORY) > 0
        assert all(
            cat in c.EnforcementCategory for cat in c.ENFORCEMENT_TAG_CATEGORY.values()
        )

    def test_canonical_flext_core_class_satisfies_prefix_contract(self) -> None:
        """A correctly named ``flext_core`` class raises no class-prefix violation."""
        assert not messages(u.check(FlextUtilitiesEnforcement), fragment="class_prefix")

    @pytest.mark.parametrize(
        ("module", "expect_prefix_violation"),
        [("flext_core.synthetic_module", True), ("fence", False)],
    )
    def test_class_prefix_enforced_only_for_knowable_projects(
        self, module: str, expect_prefix_violation: bool
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

        assert bool(hits) is expect_prefix_violation
        if expect_prefix_violation:
            assert any('prefix "Flext"' in message for message in hits)
