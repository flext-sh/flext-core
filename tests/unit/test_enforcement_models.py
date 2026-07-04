"""Model and field enforcement tests."""

from __future__ import annotations

import typing
from types import MappingProxyType
from typing import Annotated

import pytest
from pydantic.warnings import PydanticDeprecatedSince20

from flext_core import FlextUtilitiesEnforcement
from tests.constants import c
from tests.models import m
from tests.unit._enforcement_support import messages
from tests.utilities import u

if typing.TYPE_CHECKING:
    from collections.abc import MutableSequence

    from tests.typings import t


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
                MutableSequence[str],
                m.Field(default_factory=list, description="d"),
            ]

        assert not messages(u.check(_M), fragment="read-only field contract")

    def test_mutable_mapping_forward_ref_dict_factory_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            class Value(m.ContractModel):
                name: Annotated[str, m.Field(description="Value name")] = "x"

            items: typing.MutableMapping[str, _M.Value] = m.Field(
                default_factory=dict,
                description="Mutable mapping contract.",
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
                t.StrSequence,
                m.Field(default_factory=list, description="d"),
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

    def test_flext_core_uses_flext_override(self) -> None:
        """``flext_core`` is the canonical src layer and overrides to ``Flext``."""
        project = FlextUtilitiesEnforcement._project(FlextUtilitiesEnforcement)
        assert project is not None
        prefix, namespace = project
        assert prefix == "Flext"
        assert namespace == "Core"

    def test_synthetic_markdown_module_is_unknowable(self) -> None:
        """Docs code fences run in synthetic modules and must not invent prefixes."""
        fake = type("CreateUserService", (), {})
        fake.__module__ = "fence"

        assert FlextUtilitiesEnforcement._project(fake) is None
