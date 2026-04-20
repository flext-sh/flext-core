"""Integration tests — real-module import triggers the enforcement hook.

Programmatic ``u.check(cls)`` calls (covered by ``test_enforcement.py``)
prove the engine logic. These tests go one layer up: they import two real
modules (``_enforcement_integration_fixtures/clean_module.py`` and
``bad_module.py``) and assert the production path — Pydantic's
``__pydantic_init_subclass__`` + ``FlextModelsNamespace.__init_subclass__``
hooks — actually fires the right warnings, and stays silent on clean code.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import (
    Iterator,
    Sequence,
)

import pytest


def _import_fresh(dotted: str) -> tuple[object, Sequence[warnings.WarningMessage]]:
    """Import ``dotted`` with a clean cache and capture enforcement warnings."""
    sys.modules.pop(dotted, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module(dotted)
    return module, caught


def _violation_lines(messages: Sequence[warnings.WarningMessage]) -> Iterator[str]:
    for entry in messages:
        text = str(entry.message)
        if "violates FLEXT" in text:
            yield from (
                line.strip()
                for line in text.splitlines()
                if line.strip().startswith("-")
            )


class TestCleanModuleEmitsNothing:
    """Importing well-formed code MUST NOT trigger any enforcement warning."""

    def test_clean_fixture_is_silent(self) -> None:
        _module, messages = _import_fresh(
            "tests.unit._enforcement_integration_fixtures.clean_module",
        )
        offenders = list(_violation_lines(messages))
        assert not offenders, (
            "Clean fixture should produce zero enforcement warnings; got: "
            + "; ".join(offenders[:5])
        )


class TestBadModuleFiresExpectedRules:
    """Each bad class in the fixture triggers its dedicated rule."""

    @pytest.fixture(scope="class")
    def violations(self) -> Sequence[str]:
        _module, messages = _import_fresh(
            "tests.unit._enforcement_integration_fixtures.bad_module",
        )
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
        ],
    )
    def test_rule_fires_for_fragment(
        self,
        violations: Sequence[str],
        fragment: str,
        tag: str,
    ) -> None:
        hit = [line for line in violations if fragment in line]
        assert hit, (
            f"tag={tag!r}: no violation line contained {fragment!r}. "
            f"Captured {len(violations)} lines; first 5: "
            + "; ".join(list(violations)[:5])
        )

    def test_rule_firings_cover_every_bad_class(self) -> None:
        """Every top-level bad class must produce at least one warning."""
        _module, messages = _import_fresh(
            "tests.unit._enforcement_integration_fixtures.bad_module",
        )
        expected_classes = {
            "TestsFlextCoreBadAnyField",
            "TestsFlextCoreBadBareCollection",
            "TestsFlextCoreBadMutableDefault",
            "TestsFlextCoreBadMissingDesc",
            "TestsFlextCoreBadInlineUnion",
            "TestsFlextCoreBadFrozen",
            "TestsFlextCoreBadAccessors",
            "TestsFlextCoreBadWorkerSettings",
            "TestsFlextCoreBadConstants",
        }
        seen: set[str] = set()
        for entry in messages:
            text = str(entry.message)
            if "violates FLEXT" not in text:
                continue
            for name in expected_classes:
                if name in text:
                    seen.add(name)
        missing = expected_classes - seen
        assert not missing, f"No violations produced for: {missing}"
