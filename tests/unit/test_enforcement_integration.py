"""Integration tests — real-module import triggers the enforcement hook.

Programmatic ``u.check(cls)`` calls (covered by ``test_enforcement.py``)
prove the engine logic. These tests go one layer up: they import a clean real
module and an isolated temporary bad module, then assert the production path —
Pydantic's
``__pydantic_init_subclass__`` + ``FlextModelsNamespace.__init_subclass__``
hooks — actually fires the right warnings, and stays silent on clean code.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
import textwrap
import warnings
from collections.abc import (
    Iterator,
)
from pathlib import Path

import pytest

from tests.typings import t
from tests.utilities import u

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BAD_MODULE_DOTTED = "tests_flext_enforcement_integration_fixtures_bad"
_BAD_MODULE_SOURCE = """
from __future__ import annotations

import typing
from collections.abc import MutableSequence
from typing import Annotated, ClassVar

from flext_core import FlextModelsNamespace, m as core_m
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
""".strip()


def _import_fresh_silent(dotted: str) -> object:
    """Import ``dotted`` with a clean cache and fail on any warning."""
    sys.modules.pop(dotted, None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return importlib.import_module(dotted)


def _import_fresh_warning_messages(
    dotted: str,
    module_root: Path,
) -> t.StrSequence:
    script = textwrap.dedent(
        """
        from __future__ import annotations

        import importlib
        import json
        import sys
        import warnings

        from flext_core._constants.enforcement import FlextMroViolation

        module_root = sys.argv[1]
        dotted = sys.argv[2]
        sys.path.insert(0, module_root)
        importlib.invalidate_caches()
        sys.modules.pop(dotted, None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", FlextMroViolation)
            importlib.import_module(dotted)
        messages = [
            str(entry.message)
            for entry in caught
            if issubclass(entry.category, FlextMroViolation)
            and "violates FLEXT" in str(entry.message)
        ]
        print(json.dumps(messages, ensure_ascii=True))
        """
    ).strip()
    result = u.Cli.run_raw(
        [
            sys.executable,
            "-c",
            script,
            str(module_root),
            dotted,
        ],
        cwd=_PROJECT_ROOT,
    )
    assert result.success, result.error
    completed = result.value
    output = completed.stdout + completed.stderr
    assert completed.exit_code == 0, output
    payload: t.JsonValue = t.json_value_adapter().validate_json(completed.stdout)
    assert isinstance(payload, list), completed.stdout
    messages = tuple(item for item in payload if isinstance(item, str))
    assert len(messages) == len(payload), completed.stdout
    assert messages, f"{dotted} emitted no FLEXT violation warnings"
    return messages


def _write_bad_fixture_module(module_root: Path) -> str:
    module_path = module_root / f"{_BAD_MODULE_DOTTED}.py"
    module_path.write_text(_BAD_MODULE_SOURCE + "\n", encoding="utf-8")
    return _BAD_MODULE_DOTTED


def _violation_lines(messages: t.StrSequence) -> Iterator[str]:
    for text in messages:
        if "violates FLEXT" in text:
            yield from (
                line.strip()
                for line in text.splitlines()
                if line.strip().startswith("-")
            )


class TestsFlextEnforcementIntegration:
    def test_clean_fixture_is_silent(self) -> None:
        _import_fresh_silent(
            "tests.fixtures.clean_module",
        )

    @pytest.fixture(scope="class")
    def bad_fixture_module(
        self,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> tuple[Path, str]:
        module_root = tmp_path_factory.mktemp("enforcement_bad_fixture")
        return module_root, _write_bad_fixture_module(module_root)

    @pytest.fixture(scope="class")
    def violation_messages(
        self,
        bad_fixture_module: tuple[Path, str],
    ) -> t.StrSequence:
        module_root, dotted = bad_fixture_module
        return _import_fresh_warning_messages(
            dotted,
            module_root,
        )

    @pytest.fixture(scope="class")
    def violations(self, violation_messages: t.StrSequence) -> t.StrSequence:
        messages = violation_messages
        return list(_violation_lines(messages))

    @pytest.mark.parametrize(
        ("fragment", "tag"),
        [
            ("Any is FORBIDDEN", "no_any"),
            ("bare list", "no_bare_collection"),
            ("mutable default list", "no_mutable_default"),
            ("missing description", "missing_description"),
            ("complex inline union with 5 arms", "no_inline_union"),
            ("must be frozen=True", "value_not_frozen"),
            ('accessor method "get_value"', "no_accessor_methods.get"),
            ('accessor method "set_value"', "no_accessor_methods.set"),
            ('accessor method "is_ready"', "no_accessor_methods.is"),
            ("must inherit FlextSettings", "settings_inheritance"),
            ("mutable constant value", "const_mutable"),
            ("UPPER_CASE", "const_lowercase"),
            (
                "Constant 'GROUPS' declared",
                "classvar_constant_outside_constants",
            ),
        ],
    )
    def test_rule_fires_for_fragment(
        self,
        violations: t.StrSequence,
        fragment: str,
        tag: str,
    ) -> None:
        hit = [line for line in violations if fragment in line]
        assert hit, (
            f"tag={tag!r}: no violation line contained {fragment!r}. "
            f"Captured {len(violations)} lines; first 5: "
            + "; ".join(list(violations)[:5])
        )

    def test_rule_firings_cover_every_bad_class(
        self,
        violation_messages: t.StrSequence,
    ) -> None:
        """Every top-level bad class must produce at least one warning."""
        messages = violation_messages
        expected_classes = {
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
        }
        seen: set[str] = set()
        for text in messages:
            if "violates FLEXT" not in text:
                continue
            for name in expected_classes:
                if name in text:
                    seen.add(name)
        missing = expected_classes - seen
        assert not missing, f"No violations produced for: {missing}"
