"""Integration tests — importing a module fires the enforcement hook.

Programmatic ``u.check(cls)`` reports are exercised by the ``test_enforcement*``
modules. These tests assert the production integration path one layer up: the
Pydantic ``__pydantic_init_subclass__`` + ``FlextModelsNamespace.__init_subclass__``
hooks emit ``FlextMroViolation`` warnings automatically when a module with
rule-violating top-level classes is imported, and stay silent for clean code.

The observable contract asserted here is the *public warning output*: the
``FlextMroViolation`` category (a public ``flext_core`` export) and the human
readable violation text a developer sees. No private module, collaborator, or
internal data structure is inspected.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pytest

from flext_core import e
from tests.typings import t

_CLEAN_MODULE = "tests.fixtures.clean_module"
_BAD_MODULE = "tests_flext_enforcement_integration_fixtures_bad"
_BAD_MODULE_SOURCE = """\
from __future__ import annotations

import typing
from collections.abc import MutableSequence
from typing import Annotated, ClassVar

from flext_core import m as core_m
from flext_core.models import FlextModelsNamespace
from tests.models import m
from tests.utilities import u


class TestsFlextBadAnyField(core_m.ArbitraryTypesModel):
    data: Annotated[typing.Any, u.Field(description="Intentionally Any.")] = None


class TestsFlextBadBareCollection(core_m.ArbitraryTypesModel):
    items: list[str] = u.Field(default_factory=list, description="Bare list.")


class TestsFlextBadMutableDefault(core_m.ArbitraryTypesModel):
    items: Annotated[
        MutableSequence[str],
        u.Field(description="Mutable default list."),
    ] = ["x"]


class TestsFlextBadMissingDesc(core_m.ArbitraryTypesModel):
    undocumented: str = ""


class TestsFlextBadInlineUnion(core_m.ArbitraryTypesModel):
    value: Annotated[
        str | int | float | bool | bytes,
        u.Field(description="Five-arm inline union."),
    ] = ""


class TestsFlextBadFrozen(core_m.ImmutableValueModel):
    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

    payload: Annotated[str, u.Field(description="Data payload.")] = ""


class TestsFlextBadAccessors(FlextModelsNamespace):
    def get_value(self) -> int:
        return 0

    def set_value(self, value: int) -> None:
        return None

    def is_ready(self) -> bool:
        return True


class TestsFlextBadWorkerSettings(FlextModelsNamespace):
    pass


class TestsFlextBadConstants(FlextModelsNamespace):
    items: ClassVar[list[str]] = ["a", "b"]


class TestsFlextBadClassVarConstant(FlextModelsNamespace):
    GROUPS: ClassVar[frozenset[str]] = frozenset({"a", "b"})
"""


def _capture_import_warnings(
    dotted: str, *, search_path: Path | None = None
) -> t.StrSequence:
    """Freshly import ``dotted`` and return every emitted violation message.

    The whole import runs inside the ``catch_warnings`` block so no warning
    leaks to the surrounding pytest session, and the module is evicted +
    ``sys.path`` restored afterwards so the shared interpreter is left clean.
    """
    inserted = search_path is not None
    if inserted:
        sys.path.insert(0, str(search_path))
    try:
        sys.modules.pop(dotted, None)
        importlib.invalidate_caches()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", e.MroViolation)
            importlib.import_module(dotted)
        return tuple(
            str(entry.message)
            for entry in caught
            if issubclass(entry.category, e.MroViolation)
            and "violates FLEXT" in str(entry.message)
        )
    finally:
        sys.modules.pop(dotted, None)
        if inserted:
            sys.path.remove(str(search_path))


class TestsFlextEnforcementIntegration:
    """Import-time enforcement hook behaviour on clean and violating modules."""

    def test_clean_module_import_emits_no_violation_warning(self) -> None:
        # Arrange / Act: importing a fully rule-compliant module.
        messages = _capture_import_warnings(_CLEAN_MODULE)

        # Assert: the hook produces no FLEXT violation warnings.
        assert messages == (), "Clean module import must be silent; got: " + " | ".join(
            messages
        )

    @pytest.fixture(scope="class")
    @classmethod
    def violation_messages(
        cls, tmp_path_factory: pytest.TempPathFactory
    ) -> t.StrSequence:
        module_root = tmp_path_factory.mktemp("enforcement_bad_fixture")
        (module_root / f"{_BAD_MODULE}.py").write_text(
            _BAD_MODULE_SOURCE, encoding="utf-8"
        )
        messages = _capture_import_warnings(_BAD_MODULE, search_path=module_root)
        assert messages, "Importing the violating module emitted no warnings"
        return messages

    def test_every_emitted_warning_is_the_public_category(
        self, violation_messages: t.StrSequence
    ) -> None:
        # The public FlextMroViolation export is a genuine Warning subclass and
        # is the exact category a caller can filter on.
        assert issubclass(e.MroViolation, Warning)
        assert all("violates FLEXT" in message for message in violation_messages)

    @pytest.mark.parametrize(
        ("fragment", "rule"),
        [
            ("Any is FORBIDDEN", "no_any"),
            ("bare list", "no_bare_collection"),
            ("mutable default list", "no_mutable_default"),
            ("missing description", "missing_description"),
            ("complex inline union with 5 arms", "no_inline_union"),
            ("must be frozen=True", "value_not_frozen"),
            ('accessor method "get_value"', "no_accessor_get"),
            ('accessor method "set_value"', "no_accessor_set"),
            ('accessor method "is_ready"', "no_accessor_is"),
            ("must inherit FlextSettings", "settings_inheritance"),
            ("mutable constant value", "const_mutable"),
            ("UPPER_CASE", "const_lowercase"),
            ("Constant 'GROUPS' declared", "classvar_constant_outside_constants"),
        ],
    )
    def test_rule_violation_is_reported_in_warning_text(
        self, violation_messages: t.StrSequence, fragment: str, rule: str
    ) -> None:
        # Assert: the observable warning output names the specific violation.
        assert any(fragment in message for message in violation_messages), (
            f"rule={rule!r}: no warning contained {fragment!r}"
        )

    @pytest.mark.parametrize(
        "class_name",
        [
            "TestsFlextBadAnyField",
            "TestsFlextBadBareCollection",
            "TestsFlextBadMutableDefault",
            "TestsFlextBadMissingDesc",
            "TestsFlextBadInlineUnion",
            "TestsFlextBadFrozen",
            "TestsFlextBadAccessors",
            "TestsFlextBadWorkerSettings",
            "TestsFlextBadConstants",
            "TestsFlextBadClassVarConstant",
        ],
    )
    def test_each_violating_class_is_named_in_a_warning(
        self, violation_messages: t.StrSequence, class_name: str
    ) -> None:
        # Assert: every rule-breaking top-level class triggers the hook.
        assert any(class_name in message for message in violation_messages), (
            f"No violation warning named {class_name!r}"
        )
