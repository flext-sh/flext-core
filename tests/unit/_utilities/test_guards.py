"""Behavioral contract tests for flext-core type guards.

Exercises the public guard surface exposed through the test ``u`` facade
(``FlextUtilitiesGuardsTypeProtocol`` / ``FlextUtilitiesGuardsTypeCore``).
Every assertion targets observable return-value behavior of a public guard,
never an internal helper or private attribute.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserList
from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import override

import pytest

from flext_core import p, t
from tests import c, m, r, tm, u


class TestsFlextCoreGuards:
    """Public-contract behavior of the flext-core type guards."""

    class _SampleModel(m.BaseModel):
        """Minimal real Pydantic model used to probe model-exclusion behavior."""

        value: int = 1

    @staticmethod
    def test_mapping_view_preserves_integer_keys_and_opaque_values() -> None:
        """The inspection view neither requires JSON nor reconstructs a mapping."""

        class Handle:
            """Opaque SDK value with no domain or serialization contract."""

        handle = Handle()
        mapping = {1: handle}
        subject: p.AttributeProbe = mapping
        if not u.mapping_view(subject):
            pytest.fail("native mapping rejected")
        tm.that(subject is mapping, truth=True)
        tm.that(1 in subject, truth=True)
        tm.that(subject[1] is handle, truth=True)
        tm.that(u.mapping_view(handle), truth=False)
        tm.that(u.mapping_view([handle]), truth=False)

    @staticmethod
    def test_mapping_view_does_not_iterate_or_normalize_lookup_errors() -> None:
        """Guarding adds no iteration, membership, length or item-access effects."""
        events: list[str] = []
        failure = LookupError("original item failure")

        class NativeMapping(Mapping[p.AttributeProbe, p.AttributeProbe]):
            @override
            def __contains__(self, key: p.AttributeProbe) -> bool:
                events.append("contains")
                return key == 1

            @override
            def __getitem__(self, key: p.AttributeProbe) -> p.AttributeProbe:
                events.append("getitem")
                raise failure

            @override
            def __iter__(self) -> Iterator[p.AttributeProbe]:
                events.append("iter")
                msg = "unexpected mapping iteration"
                raise AssertionError(msg)

            @override
            def __len__(self) -> int:
                events.append("len")
                return 1

        mapping = NativeMapping()
        subject: p.AttributeProbe = mapping
        if not u.mapping_view(subject):
            pytest.fail("native mapping rejected")
        tm.that(subject is mapping, truth=True)
        tm.that(events, eq=[])
        tm.that(1 in subject, truth=True)
        with pytest.raises(LookupError) as raised:
            subject[1]
        tm.that(raised.value is failure, truth=True)
        tm.that(events, eq=["contains", "getitem"])

    @staticmethod
    def test_result_view_inspection_does_not_evaluate_payload_properties() -> None:
        """A read-only result needs no railway operations or eager property access."""
        events: list[str] = []
        failure = RuntimeError("original payload failure")

        class ReadOnlyResult:
            @property
            def error(self) -> str | None:
                return None

            @property
            def error_code(self) -> str | None:
                return None

            @property
            def error_data(self) -> t.JsonMapping | None:
                return None

            @property
            def exception(self) -> BaseException | None:
                return None

            @property
            def failure(self) -> bool:
                return False

            @property
            def success(self) -> bool:
                events.append("success")
                return True

            @property
            def value(self) -> p.AttributeProbe:
                events.append("value")
                raise failure

        result = ReadOnlyResult()
        subject: p.AttributeProbe = result
        if not u.result_view(subject):
            pytest.fail("native read-only result rejected")
        tm.that(subject is result, truth=True)
        tm.that(events, eq=[])
        tm.that(subject.success, truth=True)
        with pytest.raises(RuntimeError) as raised:
            _ = subject.value
        tm.that(raised.value is failure, truth=True)
        tm.that(events, eq=["success", "value"])

    @staticmethod
    def test_subclass_view_refines_runtime_classes_without_calling_factories() -> None:
        """Native class inspection preserves identity and never calls a factory."""
        events: list[str] = []

        class Base:
            """Native runtime base for the SDK class view."""

        class Child(Base):
            """Valid native subclass."""

        class Unrelated:
            """A runtime class outside the requested hierarchy."""

        def factory() -> Base:
            events.append("factory")
            return Base()

        cases: tuple[tuple[p.AttributeProbe, bool], ...] = (
            (Base, True),
            (Child, True),
            (Unrelated, False),
            (Base(), False),
            (factory, False),
        )
        for candidate, expected in cases:
            if u.subclass_view(candidate, Base):
                tm.that(expected, truth=True)
                instance = candidate()
                tm.that(type(instance) is candidate, truth=True)
                tm.that(candidate is Base or candidate is Child, truth=True)
            else:
                tm.that(expected, truth=False)
        tm.that(events, eq=[])

    @staticmethod
    def test_subclass_view_preserves_native_class_check_exceptions() -> None:
        """An exceptional native subclass check escapes unchanged."""
        failure = TypeError("original subclass failure")

        class NativeClassCheck(type):
            @override
            def __subclasscheck__(cls, subclass: type[p.AttributeProbe]) -> bool:
                raise failure

        class Base(metaclass=NativeClassCheck):
            """Native base with an observable subclass-check failure."""

        with pytest.raises(TypeError) as raised:
            u.subclass_view(str, Base)
        tm.that(raised.value is failure, truth=True)

    @staticmethod
    def test_result_view_accepts_full_results_without_changing_result_like() -> None:
        """Full railway results retain native payload identity and both predicates."""

        class Handle:
            """Opaque SDK value with no serialization contract."""

        handle = Handle()
        result = r[p.AttributeProbe].ok(handle)
        subject: p.AttributeProbe = result
        if not u.result_view(subject):
            pytest.fail("native full result rejected")
        tm.that(subject is result, truth=True)
        tm.that(subject.value is handle, truth=True)
        tm.that(u.result_like(r[str].ok("value")), truth=True)
        tm.that(u.result_like(value=False), truth=False)
        tm.that(u.result_view(handle), truth=False)
        tm.that(u.result_view(value=True), truth=False)

    # ------------------------------------------------------------------
    # matches_type — string specs
    # ------------------------------------------------------------------
    @staticmethod
    @pytest.mark.parametrize(
        ("value", "spec", "expected"),
        [
            ("x", "str", True),
            (1, "str", False),
            (1, "int", True),
            (1.5, "float", True),
            (True, "bool", True),
            ({"k": "v"}, "mapping", True),
            ([1, 2], c.Action.LIST, True),
            ((1, 2), "tuple", True),
            (None, "none", True),
            ("x", "none", False),
        ],
    )
    def test_matches_type_string_spec_reflects_runtime_type(
        value: p.ProtocolGuardInput | None,
        spec: str,
        *,
        expected: bool,
    ) -> None:
        tm.that(u.matches_type(value, spec) is expected, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "spec", "expected"),
        [
            (" a ", "string_non_empty", True),
            ("", "string_non_empty", False),
            ("   ", "string_non_empty", False),
            ({"k": "v"}, "dict_non_empty", True),
            ({}, "dict_non_empty", False),
            (["v"], "list_non_empty", True),
            ([], "list_non_empty", False),
        ],
    )
    def test_matches_type_non_empty_specs_require_content(
        value: p.ProtocolGuardInput,
        spec: str,
        *,
        expected: bool,
    ) -> None:
        tm.that(u.matches_type(value, spec) is expected, truth=True)

    @staticmethod
    def test_matches_type_is_case_insensitive_for_string_specs() -> None:
        tm.that(u.matches_type("x", "STR") is True, truth=True)
        tm.that(u.matches_type(1, "Int") is True, truth=True)

    @staticmethod
    def test_matches_type_unknown_string_spec_returns_false() -> None:
        tm.that(u.matches_type("x", "no_such_spec") is False, truth=True)

    @staticmethod
    def test_matches_type_inspects_native_mapping_and_sequence_views() -> None:
        """Inspection preserves abstract collections without JSON admission."""
        mapping: t.JsonMapping = MappingProxyType({"native": 1})
        mutable_mapping: t.MutableMappingKV[str, t.JsonValue] = {"native": 1}
        sequence: t.StrSequence = UserList(["native"])
        mutable_sequence: t.MutableSequenceOf[str] = ["native"]
        tm.ok(u.matches_type(mapping, Mapping))
        tm.ok(u.matches_type(mutable_mapping, dict))
        tm.fail(u.matches_type(mapping, dict))
        tm.ok(u.matches_type(sequence, Sequence))
        tm.ok(u.matches_type(sequence, "list_non_empty"))
        tm.ok(u.matches_type(mutable_sequence, (list, tuple)))
        tm.fail(u.matches_type(sequence, (list, tuple)))
        tm.fail(u.matches_type(sequence, Mapping))
        tm.fail(u.matches_type(mapping, "no_such_spec"))

    @staticmethod
    @pytest.mark.parametrize(
        "spec",
        ["string_non_empty", "dict_non_empty", "list_non_empty"],
    )
    def test_matches_type_excludes_pydantic_models_from_non_empty_specs(
        spec: str,
    ) -> None:
        # A populated model would otherwise satisfy dict-like checks; the guard
        # contract deliberately excludes Pydantic models from these specs.
        tm.that(
            u.matches_type(TestsFlextCoreGuards._SampleModel(), spec) is False,
            truth=True,
        )

    # ------------------------------------------------------------------
    # matches_type — type and tuple specs
    # ------------------------------------------------------------------
    @staticmethod
    @pytest.mark.parametrize(
        ("value", "spec", "expected"),
        [
            (5, int, True),
            ("x", int, False),
            (5, (int, str), True),
            ("x", (int, str), True),
            (5.0, (int, str), False),
        ],
    )
    def test_matches_type_type_and_tuple_specs(
        value: p.ProtocolGuardInput,
        spec: type | t.VariadicTuple[type],
        *,
        expected: bool,
    ) -> None:
        tm.that(u.matches_type(value, spec) is expected, truth=True)

    @staticmethod
    def test_matches_type_invalid_scalar_spec_returns_false() -> None:
        # An out-of-contract spec (a bare scalar) must not match anything.
        tm.that(u.matches_type("x", 123) is False, truth=True)

    # ------------------------------------------------------------------
    # container — recursive JSON-compatibility contract
    # ------------------------------------------------------------------
    @staticmethod
    @pytest.mark.parametrize(
        "value",
        [
            "x",
            1,
            1.5,
            True,
            {"k": "v"},
            [1, "x", True],
            {"a": {"b": [1, 2]}},
            [[1], [2, 3]],
            [],
            {},
        ],
    )
    def test_container_accepts_scalars_and_nested_json(
        value: p.ProtocolGuardInput,
    ) -> None:
        tm.that(u.container_value(value) is True, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        "value",
        [None, object(), [1, object()], {"a": object()}, {"a": [object()]}],
    )
    def test_container_rejects_none_and_non_json_members(
        value: p.ProtocolGuardInput | None,
    ) -> None:
        tm.that(u.container_value(value) is False, truth=True)

    # ------------------------------------------------------------------
    # scalar / primitive
    # ------------------------------------------------------------------


class TestsFlextCoreScalarAndCollectionGuards:
    """Scalar classification and collection membership contracts."""

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("x", True),
            (1, True),
            (1.5, True),
            (True, True),
            ([1], False),
            ({}, False),
            (None, False),
        ],
    )
    def test_scalar_identifies_scalar_values(
        value: t.JsonValue,
        *,
        expected: bool,
    ) -> None:
        tm.that(u.scalar(value) is expected, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("x", True),
            (1, True),
            (1.5, True),
            (True, True),
            ([1], False),
            ({"k": 1}, False),
        ],
    )
    def test_primitive_identifies_primitive_values(
        value: t.JsonValue,
        *,
        expected: bool,
    ) -> None:
        tm.that(u.primitive(value) is expected, truth=True)

    # ------------------------------------------------------------------
    # collection guards
    # ------------------------------------------------------------------
    @staticmethod
    def test_mapping_and_list_value_discriminate_collections() -> None:
        tm.that(u.mapping({"k": "v"}) is True, truth=True)
        tm.that(u.mapping([1, 2]) is False, truth=True)
        tm.that(u.list_value([1, 2]) is True, truth=True)
        tm.that(u.list_value((1, 2)) is False, truth=True)
        counter = u.Tests.MemoryCounter()
        service = u.Tests.CountingService(counter=counter)
        registry: p.GuardInput = {"service": service}
        tm.that(u.mapping(registry), truth=True)
        tm.that(u.dict_like(registry), truth=True)
        tm.that(u.list_value([service]), truth=True)
        tm.that(u.list_like([service]), truth=True)
        tm.that(u.container_value(registry), truth=False)
        tm.that(service.counter is counter, truth=True)
        tm.that(service.execute().unwrap(), eq=1)
        tm.that(service.execute().unwrap(), eq=2)
        tm.that(c.MetricType.COUNTER in service.model_dump(), truth=False)
        first = m.Tests.HandlerObservations()
        second = m.Tests.HandlerObservations()
        tm.that(type(first.handled) is list, truth=True)
        tm.that(first.handled, eq=[])
        tm.that(first.handled is second.handled, truth=False)
        first.handled.append("first")
        tm.that(second.handled, eq=[])
        with pytest.raises(m.ValidationError):
            m.Tests.PublicPayload.model_validate({})
        payload = m.Tests.PublicPayload.model_validate({
            "rawName": " alice ",
            "visits": 2,
        })
        tm.that(
            payload.model_dump(),
            eq={
                "raw_name": "Alice",
                "visits": "2 visits",
                "label": "Alice:2",
            },
        )

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([1], True),
            ((1,), True),
            ("text", False),
            (b"bytes", False),
            ({"k": 1}, False),
        ],
    )
    def test_list_like_excludes_strings_and_bytes(
        value: t.JsonValue | bytes | tuple[int, ...],
        *,
        expected: bool,
    ) -> None:
        tm.that(u.list_like(value) is expected, truth=True)

    @staticmethod
    def test_dict_like_accepts_only_mappings() -> None:
        tm.that(u.dict_like({"k": "v"}) is True, truth=True)
        tm.that(u.dict_like([1, 2]) is False, truth=True)
        tm.that(u.dict_like("text") is False, truth=True)

    # ------------------------------------------------------------------
    # emptiness / non-empty guards
    # ------------------------------------------------------------------
    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, True),
            ("", True),
            ("x", False),
            ([], True),
            ([1], False),
            ({}, True),
            ({"k": 1}, False),
            (0, False),
        ],
    )
    def test_empty_value_reports_absence_or_empty_containers(
        value: t.JsonValue,
        *,
        expected: bool,
    ) -> None:
        tm.that(u.empty_value(value) is expected, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "expected"),
        [("x", True), (" a ", True), ("", False), ("   ", False), (1, False)],
    )
    def test_string_non_empty_requires_non_blank_string(
        value: p.GuardInput,
        *,
        expected: bool,
    ) -> None:
        tm.that(u.string_non_empty(value) is expected, truth=True)

    @staticmethod
    def test_dict_non_empty_requires_populated_mapping() -> None:
        tm.that(u.dict_non_empty({"k": 1}) is True, truth=True)
        tm.that(u.dict_non_empty({}) is False, truth=True)
        tm.that(u.dict_non_empty(None) is False, truth=True)
        events: list[str] = []
        failure = RuntimeError("original mapping truth failure")

        class NativeMapping(Mapping[str, int]):
            def __bool__(self) -> bool:
                events.append("bool")
                raise failure

            @override
            def __getitem__(self, key: str) -> int:
                raise failure

            @override
            def __iter__(self) -> Iterator[str]:
                raise failure

            @override
            def __len__(self) -> int:
                raise failure

        native = NativeMapping()
        with pytest.raises(RuntimeError) as raised:
            u.dict_non_empty(native)
        tm.that(raised.value is failure, truth=True)
        tm.that(events, eq=["bool"])

    # ------------------------------------------------------------------
    # instance_of / in_ / type_name
    # ------------------------------------------------------------------
    @staticmethod
    def test_instance_of_matches_concrete_type() -> None:
        tm.that(u.instance_of(5, int) is True, truth=True)
        tm.that(u.instance_of("x", int) is False, truth=True)

    @staticmethod
    @pytest.mark.parametrize(
        ("value", "container", "expected"),
        [
            (1, [1, 2, 3], True),
            (4, [1, 2, 3], False),
            ("k", {"k": 1}, True),
            (2, (1, 2), True),
            (1, {1, 2}, True),
            (1, "not-a-real-container", False),
        ],
    )
    def test_in_membership_only_for_true_containers(
        value: p.GuardInput,
        container: p.GuardInput,
        *,
        expected: bool,
    ) -> None:
        tm.that(u.in_(value, container) is expected, truth=True)

    @staticmethod
    def test_in_returns_false_for_unhashable_value() -> None:
        # A list is unhashable; membership in a set raises TypeError internally,
        # which the guard must swallow into a plain False.
        tm.that(u.in_([1], {1, 2}) is False, truth=True)

    @staticmethod
    def test_type_name_returns_runtime_qualname() -> None:
        tm.that(u.type_name("x") == "str", truth=True)
        tm.that(u.type_name(1) == "int", truth=True)
        tm.that(u.type_name([1]) == c.Action.LIST, truth=True)
        tm.that(u.type_name(None) == "NoneType", truth=True)
