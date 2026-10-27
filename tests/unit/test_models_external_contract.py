"""Behavioral tests for the external-contract model preset.

``m.ExternalContractModel`` parses the consumed subset of an external tool's
payload strictly and immutably, ignoring the keys the producer adds beyond it.
Asserted through the public ``m`` facade only.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

import pytest

from tests import c, m, tm


class TestsFlextCoreModelsExternalContract:
    """Tests for ``m.ExternalContractModel``."""

    class _External(m.ExternalContractModel):
        number: Annotated[int, m.Field(description="Consumed external field")]

    def test_ignores_foreign_keys(self) -> None:
        """Keys beyond the consumed subset are ignored, never stored."""
        model = self._External.model_validate({"number": 7, "foreign": "x"})
        tm.that(model.model_dump(), eq={"number": 7})

    def test_validates_strictly(self) -> None:
        """A consumed field rejects lax coercion of the external payload."""
        with pytest.raises(c.ValidationError) as failure:
            self._External.model_validate({"number": "7"})
        tm.that(failure.value.errors()[0]["type"], eq="int_type")

    def test_is_frozen(self) -> None:
        """A parsed external payload is immutable."""
        model = self._External.model_validate({"number": 7})
        tm.rejects_assignment(model, "number", 8, expected=c.ValidationError)

    def test_is_complete_at_import(self) -> None:
        """A subclass is fully built when declared, never lazily rebuilt."""
        tm.that(self._External.__pydantic_complete__, eq=True)
