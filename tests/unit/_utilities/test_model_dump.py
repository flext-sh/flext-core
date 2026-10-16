"""Behavioral contract tests for the public model-serialization utility.

Exercises ``u.dump`` — the observable ``model_dump`` projection and the
fail-loud contract when inline dump options are invalid. No private
attributes, collaborators, or internal data structures are inspected.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

from flext_core import m
from tests import u


class TestsFlextCoreModelDump:
    """Public-surface behavior of the ``u.dump`` utility."""

    @staticmethod
    def test_dump_projects_through_model_dump_json() -> None:
        model = m.Pagination()

        assert u.dump(model) == model.model_dump(mode="json")

    @staticmethod
    def test_dump_fails_loud_on_invalid_inline_options() -> None:
        model = m.Pagination()

        with pytest.raises(RuntimeError) as raised:
            u.dump(model, include=123)

        assert raised.value.__cause__ is not None
