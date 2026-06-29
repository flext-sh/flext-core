"""Accessor and nested namespace enforcement tests."""

from __future__ import annotations

from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine
from tests import m, u
from tests.unit._enforcement_support import make_class


class TestsFlextEnforcementAccessors:
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
        cls = make_class("FlextCoreAccessedGet", {"get_user": lambda self: None})
        msgs = [
            v.message
            for v in u.check(cls).violations
            if 'accessor method "get_user"' in v.message
        ]
        assert msgs

    def test_set_prefix_flagged(self) -> None:
        cls = make_class(
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
        cls = make_class("FlextCoreAccessedIs", {"is_ready": lambda self: True})
        msgs = [
            v.message
            for v in u.check(cls).violations
            if 'accessor method "is_ready"' in v.message
        ]
        assert msgs

    def test_fetch_prefix_allowed(self) -> None:
        cls = make_class(
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
        cls = make_class("FlextWorkerSettings", {})
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
        cls = make_class("FlextCoreService", {})
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
