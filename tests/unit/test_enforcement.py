"""Tests for runtime enforcement — query ``u.check(target)`` reports.

Every legacy per-rule helper (``u.check_no_any``, ``u.is_exempt``, etc.)
is gone. Tests assert over ``m.Report.violations`` filtered
by ``layer`` / ``severity`` / message fragment.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from collections.abc import (
    MutableSequence,
)
from types import MappingProxyType
from typing import Annotated, Final, Protocol, runtime_checkable

import pytest
from pydantic.warnings import PydanticDeprecatedSince20

from flext_core import (
    FlextModelsEnforcement,
    FlextModelsNamespace,
    FlextUtilitiesBeartypeEngine,
    FlextUtilitiesEnforcement,
)
from tests import TestsFlextCoreModelsMixins, c, m, t, u


def _messages(
    report: FlextModelsEnforcement.Report,
    *,
    fragment: str,
) -> list[str]:
    return [v.message for v in report.violations if fragment in v.message]


def _make_class(name: str, body: dict[str, object]) -> type:
    cls = type(name, (), body)
    cls.__qualname__ = name  # strip test-method qualname prefix
    cls.__module__ = "flext_core.synthetic"
    return cls


class TestsFlextCoreEnforcement:
    """Model field-level rules — no_any, no_bare_collection, no_mutable_default, etc."""

    def test_any_field_detected(self) -> None:
        class _M(m.ArbitraryTypesModel):
            data: Annotated[typing.Any, m.Field(description="d")] = None

        assert _messages(u.check(_M), fragment="Any")

    def test_typed_field_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            name: Annotated[str, m.Field(description="d")] = "x"

        assert not _messages(u.check(_M), fragment="Any")

    def test_bare_dict_detected(self) -> None:
        class _M(m.ArbitraryTypesModel):
            data: dict[str, str] = m.Field(default_factory=dict, description="d")

        assert _messages(u.check(_M), fragment="bare ")

    def test_bare_list_detected(self) -> None:
        class _M(m.ArbitraryTypesModel):
            items: list[str] = m.Field(default_factory=list, description="d")

        assert _messages(u.check(_M), fragment="bare ")

    def test_mapping_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            data: Annotated[
                t.StrMapping,
                m.Field(default_factory=lambda: MappingProxyType({}), description="d"),
            ]

        assert not _messages(u.check(_M), fragment="bare ")

    def test_mutable_sequence_list_factory_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            items: Annotated[
                MutableSequence[str],
                m.Field(default_factory=list, description="d"),
            ]

        assert not _messages(u.check(_M), fragment="read-only field contract")

    def test_mutable_mapping_forward_ref_dict_factory_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            class Value(m.ContractModel):
                name: Annotated[str, m.Field(description="Value name")] = "x"

            items: typing.MutableMapping[str, _M.Value] = m.Field(
                default_factory=dict,
                description="Mutable mapping contract.",
            )

        assert not _messages(u.check(_M), fragment="read-only field contract")

    def test_mutable_json_mapping_alias_dict_factory_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            items: Annotated[
                t.MutableJsonMapping,
                m.Field(
                    default_factory=lambda: MappingProxyType({}),
                    description="Mutable JSON mapping contract.",
                ),
            ]

        assert not _messages(u.check(_M), fragment="read-only field contract")

    def test_sequence_list_factory_detected(self) -> None:
        class _M(m.ArbitraryTypesModel):
            items: Annotated[
                t.StrSequence,
                m.Field(default_factory=list, description="d"),
            ]

        assert _messages(u.check(_M), fragment="read-only field contract")

    def test_missing_description_detected(self) -> None:
        class _M(m.ArbitraryTypesModel):
            name: str = "test"

        assert _messages(u.check(_M), fragment="missing description")

    def test_description_present_passes(self) -> None:
        class _M(m.ArbitraryTypesModel):
            name: Annotated[str, m.Field(description="A name")] = "test"

        assert not _messages(u.check(_M), fragment="missing description")

    def test_v1_config_class_detected(self) -> None:
        with pytest.warns(PydanticDeprecatedSince20):

            class _M(m.ArbitraryTypesModel):
                class Config:
                    extra = "forbid"

                name: Annotated[str, m.Field(description="d")] = "x"

        assert _messages(u.check(_M), fragment="Pydantic v1")

    def test_flexible_internal_allows_ignore(self) -> None:
        class _M(m.FlexibleInternalModel):
            name: Annotated[str, m.Field(description="d")] = "x"

        assert not _messages(u.check(_M), fragment="extra")

    def test_mode_default_is_warn(self) -> None:
        assert c.ENFORCEMENT_MODE is c.EnforcementMode.WARN

    def test_enforcement_rules_loaded(self) -> None:
        assert len(c.ENFORCEMENT_RULES) > 0
        assert all(
            row[0] in c.EnforcementCategory for row in c.ENFORCEMENT_RULES.values()
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

    def test_mutable_list_detected(self) -> None:
        class _CConstants:
            ITEMS: list[str] = ["a", "b"]

        assert _messages(
            u.check(_CConstants, layer="constants"),
            fragment="mutable constant",
        )

    def test_mutable_dict_detected(self) -> None:
        class _CConstants:
            DATA: dict[str, int] = {"x": 1}

        assert _messages(
            u.check(_CConstants, layer="constants"),
            fragment="mutable constant",
        )

    def test_frozenset_passes(self) -> None:
        class _CConstants:
            ITEMS: Final[frozenset[str]] = frozenset({"a"})

        assert not _messages(
            u.check(_CConstants, layer="constants"),
            fragment="mutable constant",
        )

    def test_tuple_passes(self) -> None:
        class _CConstants:
            ITEMS: Final[tuple[str, ...]] = ("a", "b")

        assert not _messages(
            u.check(_CConstants, layer="constants"),
            fragment="mutable constant",
        )

    def test_lowercase_constant_detected(self) -> None:
        class _CConstants:
            my_value: int = 42

        assert _messages(
            u.check(_CConstants, layer="constants"),
            fragment="UPPER_CASE",
        )

    def test_inner_namespace_mutable_detected(self) -> None:
        class _CConstants:
            class Inner:
                BAD: list[str] = ["x"]

        assert _messages(
            u.check(_CConstants, layer="constants"),
            fragment="mutable constant",
        )

    def test_non_protocol_inner_detected(self) -> None:
        class _PProtocols:
            class NotAProtocol:
                pass

        assert _messages(
            u.check(_PProtocols, layer="protocols"),
            fragment="Protocol",
        )

    def test_abc_passes(self) -> None:
        class _PProtocols:
            class SomeContract(ABC):
                @abstractmethod
                def do(self) -> None: ...

        msgs = _messages(u.check(_PProtocols, layer="protocols"), fragment="Protocol")
        assert not any("must be Protocol" in m for m in msgs)

    def test_non_runtime_protocol_detected(self) -> None:
        class _PProtocols:
            class InnerProto(Protocol):
                def do(self) -> None: ...

        assert _messages(
            u.check(_PProtocols, layer="protocols"),
            fragment="runtime_checkable",
        )

    def test_runtime_protocol_passes(self) -> None:
        class _PProtocols:
            @runtime_checkable
            class InnerProto(Protocol):
                def do(self) -> None: ...

        assert not _messages(
            u.check(_PProtocols, layer="protocols"),
            fragment="runtime_checkable",
        )

    def test_alias_with_any_detected(self) -> None:
        class _TTypes:
            type BadAlias = typing.Any

        assert _messages(u.check(_TTypes, layer="types"), fragment="Any in type alias")

    def test_clean_alias_passes(self) -> None:
        class _TTypes:
            type GoodAlias = str

        assert not _messages(
            u.check(_TTypes, layer="types"), fragment="Any in type alias"
        )

    def test_instance_method_detected(self) -> None:
        class _UUtilities:
            def run(self) -> None: ...

        assert _messages(
            u.check(_UUtilities, layer="utilities"),
            fragment="staticmethod",
        )

    def test_static_method_passes(self) -> None:
        class _UUtilities:
            @staticmethod
            def run() -> None: ...

        assert not _messages(
            u.check(_UUtilities, layer="utilities"),
            fragment="staticmethod",
        )

    def test_class_method_passes(self) -> None:
        class _UUtilities:
            @classmethod
            def run(cls) -> None: ...

        assert not _messages(
            u.check(_UUtilities, layer="utilities"),
            fragment="staticmethod",
        )

    @pytest.mark.parametrize(
        "base_cls",
        [
            m.ArbitraryTypesModel,
            m.StrictBoundaryModel,
            m.FlexibleInternalModel,
            m.ImmutableValueModel,
            m.TaggedModel,
            m.ContractModel,
        ],
    )
    def test_base_model_has_enforcement_hook(self, base_cls: type) -> None:
        assert hasattr(base_cls, "__pydantic_init_subclass__")

    def test_c_facade_inherits_namespace(self) -> None:
        assert issubclass(c, FlextModelsNamespace)

    def test_empty_report_is_falsy(self) -> None:
        report = m.Report()
        assert not report
        assert report.empty
        assert len(report) == 0

    def test_nonempty_report_is_truthy(self) -> None:
        v = m.Violation(
            qualname="X",
            layer="Model",
            severity="HARD rules",
            message="boom",
        )
        report = m.Report(violations=[v])
        assert report
        assert not report.empty
        assert len(report) == 1
        assert report[0] == "boom"
        assert "boom" in report

    def test_merge_reports(self) -> None:
        v = m.Violation(
            qualname="X",
            layer="Model",
            severity="HARD rules",
            message="a",
        )
        w = m.Violation(
            qualname="X",
            layer="Model",
            severity="HARD rules",
            message="b",
        )
        merged = m.Report(violations=[v, w])
        assert len(merged) == 2

    def test_function_local_class_skipped(self) -> None:
        """Classes defined inside functions carry ``<locals>`` in qualname."""

        class _OuterScope:
            def make(self) -> type:
                class Inner:  # nested inside a method → `<locals>` qualname
                    pass

                return Inner

        cls = _OuterScope().make()
        assert "<locals>" in cls.__qualname__
        # check returns empty report for function-local classes
        report = u.check(cls)
        assert all(v.layer != "namespace" for v in report.violations)

    def test_private_underscore_class_skipped(self) -> None:
        """Underscore-prefixed classes are implementation details, not facades."""

        class _PrivateHelper:
            pass

        report = u.check(_PrivateHelper)
        assert all(v.layer != "namespace" for v in report.violations)

    def test_generic_bracket_specialization_skipped(self) -> None:
        """Synthetic ``Foo[int]``-style names are Pydantic/Generic artifacts."""
        # Build a synthetic target with bracketed name — mimicking what Pydantic
        # generates for parameterized generic specializations.
        fake = type("Foo[int]", (), {})
        report = u.check(fake)
        assert all(v.layer != "namespace" for v in report.violations)

    def test_inner_class_qualname_exempts_prefix_check(self) -> None:
        """Classes with ``.`` in qualname (nested) skip class_prefix."""
        # Simulate a top-level class' inner class via a synthetic target whose
        # qualname signals nesting without being function-local.
        fake = type("InnerNs", (), {})
        fake.__qualname__ = "Outer.InnerNs"  # signals nested position
        fake.__module__ = "nonexistent_project"
        report = u.check(fake)
        assert not any(
            "class name missing project prefix" in v.message for v in report.violations
        )

    def test_facade_root_exempt(self) -> None:
        """Classes in ENFORCEMENT_NAMESPACE_FACADE_ROOTS skip prefix rule."""
        fake = type("FlextModels", (), {})  # literal root name
        fake.__module__ = "anything"
        report = u.check(fake)
        assert not any(
            "class name missing project prefix" in v.message for v in report.violations
        )

    def test_flext_core_override_returns_flext(self) -> None:
        """flext_core is the single src package that maps to ``Flext``."""
        project = FlextUtilitiesEnforcement._project(FlextUtilitiesEnforcement)
        assert project is not None
        prefix, _namespace = project
        assert prefix == "Flext"

    def test_tests_module_gets_tests_prefix_composition(self) -> None:
        """Classes in ``tests.*`` carry ``Tests`` + project prefix (e.g. TestsFlext)."""
        report = u.check(TestsFlextCoreModelsMixins)
        namespace_msgs = [
            v.message
            for v in report.violations
            if v.layer == "namespace" and "class name" in v.message
        ]
        # The class name IS "TestsFlextCoreModelsMixins" which starts with
        # "TestsFlext" — the composed prefix — so no class_prefix violation.
        assert not namespace_msgs

    def test_bare_collection_renders_kind(self) -> None:
        class _M(m.ArbitraryTypesModel):
            items: list[str] = m.Field(default_factory=list, description="d")

        msgs = [v.message for v in u.check(_M).violations]
        # The predicate supplies {"kind": "list", "replacement": "..."} which
        # the engine interpolates into the rule's problem/fix templates.
        assert any("bare list" in m for m in msgs)

    def test_missing_description_keeps_field_location(self) -> None:
        class _M(m.ArbitraryTypesModel):
            undoc: str = "x"

        msgs = [v.message for v in u.check(_M).violations]
        assert any(
            'Field "undoc"' in msg and "missing description" in msg for msg in msgs
        )

    def test_get_prefix_flagged(self) -> None:
        cls = _make_class("FlextCoreAccessedGet", {"get_user": lambda self: None})
        msgs = [
            v.message
            for v in u.check(cls).violations
            if 'accessor method "get_user"' in v.message
        ]
        assert msgs

    def test_set_prefix_flagged(self) -> None:
        cls = _make_class(
            "FlextCoreAccessedSet",
            {"set_config": lambda self, v: None},
        )
        msgs = [
            v.message
            for v in u.check(cls).violations
            if 'accessor method "set_config"' in v.message
        ]
        assert msgs

    def test_is_prefix_flagged(self) -> None:
        cls = _make_class("FlextCoreAccessedIs", {"is_ready": lambda self: True})
        msgs = [
            v.message
            for v in u.check(cls).violations
            if 'accessor method "is_ready"' in v.message
        ]
        assert msgs

    def test_fetch_prefix_allowed(self) -> None:
        cls = _make_class(
            "FlextCoreAccessedFetch",
            {"fetch_remote": lambda self: None},
        )
        msgs = [
            v.message
            for v in u.check(cls).violations
            if "accessor method" in v.message and "fetch_remote" in v.message
        ]
        assert not msgs

    def test_top_level_missing_inheritance_flagged(self) -> None:
        cls = _make_class("FlextWorkerSettings", {})
        msgs = [
            v.message
            for v in u.check(cls).violations
            if "must inherit FlextSettings" in v.message
        ]
        assert msgs

    def test_nested_settings_namespace_exempt(self) -> None:
        """Nested classes inside namespace containers are metadata, not settings."""
        fake = type("AutoSettings", (), {})
        fake.__qualname__ = "FlextModelsSettings.AutoSettings"
        fake.__module__ = "flext_core._models.settings"
        msgs = [
            v.message
            for v in u.check(fake).violations
            if "must inherit FlextSettings" in v.message
        ]
        assert not msgs

    def test_non_settings_name_exempt(self) -> None:
        cls = _make_class("FlextCoreService", {})
        msgs = [
            v.message
            for v in u.check(cls).violations
            if "must inherit FlextSettings" in v.message
        ]
        assert not msgs

    def test_direct_inner_class_detected(self) -> None:
        class _DirectHolder:
            class _SomeInner:
                pass

            class PublicInner:
                pass

        assert FlextUtilitiesBeartypeEngine.has_nested_namespace(_DirectHolder)

    def test_inherited_inner_class_detected(self) -> None:
        """A class without direct inner classes but inheriting them still qualifies."""

        class _Parent:
            class Nested:
                pass

        class _Empty(_Parent):
            pass

        assert FlextUtilitiesBeartypeEngine.has_nested_namespace(_Empty)

    def test_plain_class_without_nested_returns_false(self) -> None:
        class _Bare:
            x: int = 1

        assert not FlextUtilitiesBeartypeEngine.has_nested_namespace(_Bare)
