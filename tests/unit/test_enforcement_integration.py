"""Imports stay silent; explicit public queries report source-shape violations.

ADR-028 assigns automatic static enforcement to flext-infra. Importing classes
does not run the retired static hooks. The supported ``u.check(cls)`` query
returns reports without emitting warnings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pytest

from tests import t, tm, u

_CLEAN_MODULE = "tests.fixtures.clean_module"
_BAD_MODULE = "tests_flext_enforcement_integration_fixtures_bad"
_BAD_MODULE_SOURCE = """\
from __future__ import annotations

import typing
from collections.abc import MutableSequence
from typing import Annotated, ClassVar

from flext_core import m
from flext_core.models import FlextModelsNamespace
from tests import c, r
from tests import c, u


class TestsFlextBadAnyField(m.ArbitraryTypesModel):
    data: Annotated[typing.Any, m.Field(description="Intentionally Any.")] = None


class TestsFlextBadBareCollection(m.ArbitraryTypesModel):
    items: list[str] = m.Field(default_factory=list, description="Bare list.")


class TestsFlextBadMutableDefault(m.ArbitraryTypesModel):
    items: Annotated[
        MutableSequence[str],
        m.Field(description="Mutable default list."),
    ] = ["x"]


class TestsFlextBadMissingDesc(m.ArbitraryTypesModel):
    undocumented: str = ""


class TestsFlextBadInlineUnion(m.ArbitraryTypesModel):
    value: Annotated[
        str | int | float | bool | bytes,
        m.Field(description="Five-arm inline union."),
    ] = ""


class TestsFlextBadFrozen(m.ImmutableValueModel):
    model_config: ClassVar[m.ConfigDict] = m.ConfigDict(frozen=False)

    payload: Annotated[str, m.Field(description="Data payload.")] = ""


class TestsFlextBadAccessors(FlextModelsNamespace):
    def get_value(self) -> int:
        return 0

    def set_value(self, value: int) -> None:
        return None

    def is_ready(self) -> bool:
        return True


class TestsFlextBadWorkerSettings(m.BaseSettings, FlextModelsNamespace):
    pass


class TestsFlextBadConstants(FlextModelsNamespace):
    items: ClassVar[list[str]] = ["a", "b"]


class TestsFlextBadClassVarConstant(FlextModelsNamespace):
    GROUPS: ClassVar[frozenset[str]] = frozenset({"a", "b"})
"""


def _import_messages(
    dotted: str,
    *,
    search_path: Path | None = None,
    query: bool = False,
) -> t.StrSequence:
    """Import without warnings and optionally query classes before unloading them.

    Returns:
        Explicit violation messages for classes declared by the imported module.

    """
    inserted = search_path is not None
    if inserted:
        sys.path.insert(0, str(search_path))
    try:
        sys.modules.pop(dotted, None)
        importlib.invalidate_caches()
        with warnings.catch_warnings(record=True) as caught:
            module = importlib.import_module(dotted)
            messages = tuple(
                f"{item.qualname}: {item.message}"
                for target in list(vars(module).values())
                if query and isinstance(target, type) and target.__module__ == dotted
                for item in u.check(target).violations
            )
        tm.that(tuple(str(entry.message) for entry in caught), eq=())
        return messages
    finally:
        sys.modules.pop(dotted, None)
        if inserted:
            sys.path.remove(str(search_path))


class TestsFlextEnforcementIntegration:
    """Import silence and explicit detection for clean and violating modules."""

    @staticmethod
    def test_clean_module_import_emits_no_violation_warning() -> None:
        """Test clean module import emits no violation warning."""
        messages = _import_messages(_CLEAN_MODULE)

        tm.that(
            messages,
            eq=(),
            msg="Clean module import must be silent; got: " + " | ".join(messages),
        )

    @pytest.fixture(scope="class")
    @classmethod
    def violation_messages(
        cls,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> t.StrSequence:
        """Provide ``violation_messages``.

        Returns:
            The resulting ``t.StrSequence``.

        """
        module_root = tmp_path_factory.mktemp("enforcement_bad_fixture")
        (module_root / f"{_BAD_MODULE}.py").write_text(
            _BAD_MODULE_SOURCE,
            encoding="utf-8",
        )
        messages = _import_messages(_BAD_MODULE, search_path=module_root, query=True)
        tm.that(
            messages,
            truth=True,
            msg="Explicit detection of the violating module returned no findings",
        )
        return messages

    @staticmethod
    def test_violating_module_import_does_not_run_static_hooks(
        tmp_path: Path,
    ) -> None:
        """Known source violations do not trigger static enforcement on import."""
        (tmp_path / f"{_BAD_MODULE}.py").write_text(
            _BAD_MODULE_SOURCE,
            encoding="utf-8",
        )
        tm.that(
            _import_messages(_BAD_MODULE, search_path=tmp_path),
            eq=(),
        )

    @staticmethod
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
    def test_rule_violation_is_reported_by_explicit_query(
        violation_messages: t.StrSequence,
        fragment: str,
        rule: str,
    ) -> None:
        """The explicit public query reports each supported violation."""
        tm.that(
            any(fragment in message for message in violation_messages),
            truth=True,
            msg=f"rule={rule!r}: no finding contained {fragment!r}",
        )

    @staticmethod
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
    def test_each_violating_class_is_named_in_a_report(
        violation_messages: t.StrSequence,
        class_name: str,
    ) -> None:
        """Explicit reports identify the rule-breaking class."""
        tm.that(
            any(class_name in message for message in violation_messages),
            truth=True,
            msg=f"No violation finding named {class_name!r}",
        )

    @staticmethod
    def test_classvar_constant_detector_reports_every_constant_in_one_pass(
        tmp_path: Path,
    ) -> None:
        # Arrange: one class holding TWO constants outside _constants. The
        # detector must surface both in a single pass (no whack-a-mole).
        """Test classvar constant detector reports every constant in one pass."""
        module_root = tmp_path
        module_name = "_flext_census_multi_constant_fixture"
        (module_root / f"{module_name}.py").write_text(
            "from typing import ClassVar\n"
            "from flext_core.models import FlextModelsNamespace\n"
            "\n"
            "\n"
            "class CensusMultiConstant(FlextModelsNamespace):\n"
            "    GROUPS: ClassVar[frozenset[str]] = frozenset({'a', 'b'})\n"
            "    LIMITS: ClassVar[frozenset[str]] = frozenset({'x', 'y'})\n",
            encoding="utf-8",
        )

        messages = _import_messages(module_name, search_path=module_root, query=True)

        # Assert: BOTH constants are reported without any intermediate cure.
        tm.that(
            any("GROUPS" in message for message in messages),
            truth=True,
            msg="detector missed GROUPS: " + " | ".join(messages),
        )
        tm.that(
            any("LIMITS" in message for message in messages),
            truth=True,
            msg="detector missed LIMITS (whack-a-mole regression): "
            + " | ".join(messages),
        )
