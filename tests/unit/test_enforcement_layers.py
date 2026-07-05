"""Behavioral tests for the layered runtime-enforcement checker.

Exercises the public contract of ``u.check(target, layer=...)``: given a class,
it returns a typed :class:`m.Report` whose ``violations`` carry a public
``layer`` and a stable ``[rule_tag]`` in their ``message``. Tests assert only on
that observable output -- never on how the rules are implemented.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Final, Protocol, runtime_checkable

import pytest

from flext_core import m
from tests.utilities import u


class TestsFlextCoreEnforcementLayers:
    """Public behaviour of the layered enforcement checker."""

    @staticmethod
    def _rule_tags(report: m.Report) -> set[tuple[str, str]]:
        """Extract the observable ``(layer, rule_tag)`` pairs from a report.

        The rule tag is the trailing ``[...]`` marker every violation message
        carries; it is the stable public identity of the rule, so tests key on
        it instead of on prose wording.
        """
        pairs: set[tuple[str, str]] = set()
        for violation in report.violations:
            message = violation.message
            if message.endswith("]") and "[" in message:
                tag = message.rsplit("[", 1)[-1].rstrip("]")
                pairs.add((violation.layer, tag))
        return pairs

    @staticmethod
    def _constants_mutable_list() -> type:
        class _CConstants:
            ITEMS: list[str] = ["a", "b"]

        return _CConstants

    @staticmethod
    def _constants_mutable_dict() -> type:
        class _CConstants:
            DATA: dict[str, int] = {"x": 1}

        return _CConstants

    @staticmethod
    def _constants_lowercase() -> type:
        class _CConstants:
            my_value: int = 42

        return _CConstants

    @staticmethod
    def _constants_inner_mutable() -> type:
        class _CConstants:
            class Inner:
                BAD: list[str] = ["x"]

        return _CConstants

    @staticmethod
    def _protocols_non_protocol_inner() -> type:
        class _PProtocols:
            class NotAProtocol:
                pass

        return _PProtocols

    @staticmethod
    def _protocols_non_runtime() -> type:
        class _PProtocols:
            class InnerProto(Protocol):
                def do(self) -> None: ...

        return _PProtocols

    @staticmethod
    def _types_any_alias() -> type:
        class _TTypes:
            type BadAlias = typing.Any

        return _TTypes

    @staticmethod
    def _utilities_instance_method() -> type:
        class _UUtilities:
            def run(self) -> None: ...

        return _UUtilities

    @staticmethod
    def _constants_frozenset() -> type:
        class _CConstants:
            ITEMS: Final[frozenset[str]] = frozenset({"a"})

        return _CConstants

    @staticmethod
    def _constants_tuple() -> type:
        class _CConstants:
            ITEMS: Final[tuple[str, ...]] = ("a", "b")

        return _CConstants

    @staticmethod
    def _protocols_abc() -> type:
        class _PProtocols:
            class SomeContract(ABC):
                @abstractmethod
                def do(self) -> None: ...

        return _PProtocols

    @staticmethod
    def _protocols_runtime() -> type:
        class _PProtocols:
            @runtime_checkable
            class InnerProto(Protocol):
                def do(self) -> None: ...

        return _PProtocols

    @staticmethod
    def _types_clean_alias() -> type:
        class _TTypes:
            type GoodAlias = str

        return _TTypes

    @staticmethod
    def _utilities_static_method() -> type:
        class _UUtilities:
            @staticmethod
            def run() -> None: ...

        return _UUtilities

    @staticmethod
    def _utilities_class_method() -> type:
        class _UUtilities:
            @classmethod
            def run(cls) -> None: ...

        return _UUtilities

    def test_check_returns_typed_report(self) -> None:
        # Arrange / Act
        report = u.check(self._constants_mutable_list(), layer="constants")

        # Assert -- public contract: a typed Report with a list of violations.
        assert isinstance(report, m.Report)
        assert isinstance(report.violations, list)
        assert all(isinstance(v.message, str) for v in report.violations)
        assert all(isinstance(v.layer, str) for v in report.violations)

    @pytest.mark.parametrize(
        ("build", "layer", "expected_layer", "rule_tag"),
        [
            ("_constants_mutable_list", "constants", "Constants", "const_mutable"),
            ("_constants_mutable_dict", "constants", "Constants", "const_mutable"),
            ("_constants_lowercase", "constants", "Constants", "const_lowercase"),
            ("_constants_inner_mutable", "constants", "Constants", "const_mutable"),
            (
                "_protocols_non_protocol_inner",
                "protocols",
                "Protocols",
                "proto_inner_kind",
            ),
            ("_protocols_non_runtime", "protocols", "Protocols", "proto_not_runtime"),
            ("_types_any_alias", "types", "Types", "alias_any"),
            (
                "_utilities_instance_method",
                "utilities",
                "Utilities",
                "utility_not_static",
            ),
        ],
    )
    def test_non_compliant_class_is_flagged(
        self,
        build: str,
        layer: str,
        expected_layer: str,
        rule_tag: str,
    ) -> None:
        # Arrange
        target: type = getattr(self, build)()

        # Act
        report = u.check(target, layer=layer)

        # Assert -- the specific layer/rule fires on the offending class.
        assert (expected_layer, rule_tag) in self._rule_tags(report)

    @pytest.mark.parametrize(
        ("build", "layer", "rule_tag"),
        [
            ("_constants_frozenset", "constants", "const_mutable"),
            ("_constants_tuple", "constants", "const_mutable"),
            ("_protocols_abc", "protocols", "proto_inner_kind"),
            ("_protocols_abc", "protocols", "proto_not_runtime"),
            ("_protocols_runtime", "protocols", "proto_not_runtime"),
            ("_types_clean_alias", "types", "alias_any"),
            ("_utilities_static_method", "utilities", "utility_not_static"),
            ("_utilities_class_method", "utilities", "utility_not_static"),
        ],
    )
    def test_compliant_class_is_not_flagged(
        self,
        build: str,
        layer: str,
        rule_tag: str,
    ) -> None:
        # Arrange
        target: type = getattr(self, build)()

        # Act
        report = u.check(target, layer=layer)

        # Assert -- the rule stays silent for compliant input.
        assert rule_tag not in {tag for _, tag in self._rule_tags(report)}

    def test_layer_is_auto_detected_from_class_name(self) -> None:
        # Arrange -- a ``_CConstants``-named class carrying a constants violation.
        target = self._constants_mutable_list()

        # Act -- no explicit layer: the checker infers it from the name.
        report = u.check(target)

        # Assert -- constants rules run without being told to.
        assert ("Constants", "const_mutable") in self._rule_tags(report)

    def test_explicit_layer_overrides_auto_detection(self) -> None:
        # Arrange -- same constants violation, forced onto a different layer.
        target = self._constants_mutable_list()

        # Act
        forced_utilities = self._rule_tags(u.check(target, layer="utilities"))
        forced_constants = self._rule_tags(u.check(target, layer="constants"))

        # Assert -- the explicit layer gates which rules run.
        constants_hit = ("Constants", "const_mutable")
        assert constants_hit not in forced_utilities
        assert constants_hit in forced_constants

    def test_check_is_idempotent(self) -> None:
        # Arrange
        target = self._constants_inner_mutable()

        # Act
        first = u.check(target, layer="constants")
        second = u.check(target, layer="constants")

        # Assert -- pure query: repeated checks yield the same violation set.
        assert self._rule_tags(first) == self._rule_tags(second)

    def test_compliant_class_reports_no_layer_violation(self) -> None:
        # Arrange -- a class clean for the constants layer.
        target = self._constants_frozenset()

        # Act
        report = u.check(target, layer="constants")

        # Assert -- no Constants-layer rule fires.
        assert not any(layer == "Constants" for layer, _ in self._rule_tags(report))
